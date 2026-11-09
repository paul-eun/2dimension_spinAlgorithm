import os

# worker 하나가 CPU 한 코어 정도만 사용하도록 제한
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
SIM_DIR = ROOT_DIR / "simulation"

sys.path.insert(0, str(SIM_DIR))

from xy_model_numba import XYModel2DFast


L = 256
J = 1.0

DEFAULT_TEMPERATURES = [
    0.30,
    0.60,
    0.80,
    0.875,
    0.90,
    0.925,
    1.00,
    1.20,
    1.50,
    1.80,
]


def autocorrelation(x, max_lag):
    """
    normalized autocorrelation rho(k)
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    var = np.mean(x * x)

    if var == 0:
        return np.ones(max_lag + 1)

    acf = np.empty(max_lag + 1, dtype=np.float64)
    acf[0] = 1.0

    for lag in range(1, max_lag + 1):
        acf[lag] = np.mean(
            x[:-lag] * x[lag:]
        ) / var

    return acf


def integrated_autocorrelation_time(acf):
    """
    간단한 positive-sequence estimator.

    rho(k)가 처음 0 이하가 되는 지점 전까지만 합산.
    tau_int = 1/2 + sum rho(k)
    """
    tau = 0.5

    for rho in acf[1:]:
        if rho <= 0:
            break
        tau += rho

    return tau


def suggested_spacing(tau):
    """
    보수적으로 5 tau_int 이상 간격 사용.
    """
    return max(1, int(np.ceil(5.0 * tau)))


def run_temperature(
    T,
    seed,
    burnin,
    n_measure,
    max_lag,
):
    model = XYModel2DFast(
        L=L,
        J=J,
        seed=seed,
    )

    # ---------------------------
    # burn-in
    # ---------------------------
    model.run(T, burnin)

    energy = np.empty(
        n_measure,
        dtype=np.float64,
    )

    magnetization = np.empty(
        n_measure,
        dtype=np.float64,
    )

    vortices = np.empty(
        n_measure,
        dtype=np.float64,
    )

    # ---------------------------
    # 매 sweep마다 측정
    # ---------------------------
    for i in range(n_measure):

        model.run(T, 1)

        energy[i] = (
            model.total_energy()
            / (L * L)
        )

        magnetization[i] = (
            model.magnetization()
        )

        nv, nav = model.count_vortices()

        # vortex + antivortex 총 defect 수
        vortices[i] = nv + nav

    # ---------------------------
    # autocorrelation
    # ---------------------------
    acf_E = autocorrelation(
        energy,
        max_lag,
    )

    acf_M = autocorrelation(
        magnetization,
        max_lag,
    )

    acf_V = autocorrelation(
        vortices,
        max_lag,
    )

    tau_E = integrated_autocorrelation_time(acf_E)
    tau_M = integrated_autocorrelation_time(acf_M)
    tau_V = integrated_autocorrelation_time(acf_V)

    spacing_E = suggested_spacing(tau_E)
    spacing_M = suggested_spacing(tau_M)
    spacing_V = suggested_spacing(tau_V)

    recommended = max(
        spacing_E,
        spacing_M,
        spacing_V,
    )

    return {
        "T": T,
        "seed": seed,

        "energy": energy,
        "magnetization": magnetization,
        "vortices": vortices,

        "acf_E": acf_E,
        "acf_M": acf_M,
        "acf_V": acf_V,

        "tau_E": tau_E,
        "tau_M": tau_M,
        "tau_V": tau_V,

        "spacing_E": spacing_E,
        "spacing_M": spacing_M,
        "spacing_V": spacing_V,

        "recommended": recommended,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--burnin",
        type=int,
        default=100,
        help="autocorrelation 측정 전 burn-in sweeps",
    )

    parser.add_argument(
        "--measure",
        type=int,
        default=300,
        help="매 sweep 측정할 총 sweep 수",
    )

    parser.add_argument(
        "--max-lag",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    temperatures = DEFAULT_TEMPERATURES

    out_dir = (
        THIS_DIR
        / "L256"
        / "autocorrelation"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("L=256 autocorrelation test")
    print("=" * 72)
    print(f"temperatures = {temperatures}")
    print(f"burn-in      = {args.burnin}")
    print(f"measure      = {args.measure}")
    print(f"max lag      = {args.max_lag}")
    print(f"workers      = {args.workers}")
    print("=" * 72, flush=True)

    futures = {}

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:

        for i, T in enumerate(temperatures):

            seed = 5000 + i

            future = executor.submit(
                run_temperature,
                T,
                seed,
                args.burnin,
                args.measure,
                args.max_lag,
            )

            futures[future] = T

        results = []

        for future in as_completed(futures):

            result = future.result()

            results.append(result)

            T = result["T"]

            print(
                f"T={T:5.3f} | "
                f"tau_E={result['tau_E']:6.2f} | "
                f"tau_M={result['tau_M']:6.2f} | "
                f"tau_V={result['tau_V']:6.2f} | "
                f"suggested sweeps_between="
                f"{result['recommended']}",
                flush=True,
            )

            np.savez_compressed(
                out_dir / f"autocorr_T{T:.3f}.npz",

                T=np.array(T),
                seed=np.array(result["seed"]),

                energy=result["energy"],
                magnetization=result["magnetization"],
                vortices=result["vortices"],

                acf_energy=result["acf_E"],
                acf_magnetization=result["acf_M"],
                acf_vortices=result["acf_V"],

                tau_energy=np.array(
                    result["tau_E"]
                ),

                tau_magnetization=np.array(
                    result["tau_M"]
                ),

                tau_vortices=np.array(
                    result["tau_V"]
                ),

                suggested_spacing=np.array(
                    result["recommended"]
                ),

                burnin=np.array(args.burnin),
                n_measure=np.array(args.measure),
                lattice_size=np.array(L),
            )

    # ---------------------------
    # summary
    # ---------------------------

    results.sort(
        key=lambda x: x["T"]
    )

    summary_path = (
        out_dir / "summary.txt"
    )

    with open(summary_path, "w") as f:

        header = (
            "T tau_E tau_M tau_V "
            "spacing_E spacing_M spacing_V "
            "recommended\n"
        )

        f.write(header)

        for r in results:

            line = (
                f"{r['T']:.3f} "
                f"{r['tau_E']:.6f} "
                f"{r['tau_M']:.6f} "
                f"{r['tau_V']:.6f} "
                f"{r['spacing_E']} "
                f"{r['spacing_M']} "
                f"{r['spacing_V']} "
                f"{r['recommended']}\n"
            )

            f.write(line)

    print()
    print("Summary:")
    print(summary_path)

    worst = max(
        r["recommended"]
        for r in results
    )

    print()
    print(
        "Maximum suggested sweeps_between "
        f"across tested temperatures = {worst}"
    )


if __name__ == "__main__":
    main()
