import os

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

TEMPERATURES = [
    0.800,
    0.875,
    0.900,
    0.925,
    1.000,
]

SEEDS = [
    7001,
    7002,
    7003,
]


def autocorrelation(x, max_lag):
    x = np.asarray(x, dtype=np.float64)

    x = x - np.mean(x)

    var = np.mean(x * x)

    # 상수 시계열이면 autocorrelation 정의가 의미 없음
    if var < 1e-15:
        return np.full(
            max_lag + 1,
            np.nan,
            dtype=np.float64,
        )

    acf = np.empty(
        max_lag + 1,
        dtype=np.float64,
    )

    acf[0] = 1.0

    for lag in range(1, max_lag + 1):
        acf[lag] = (
            np.mean(
                x[:-lag] * x[lag:]
            )
            / var
        )

    return acf


def integrated_autocorrelation_time(acf):
    if np.all(np.isnan(acf)):
        return np.nan

    tau = 0.5

    for rho in acf[1:]:

        if np.isnan(rho):
            break

        if rho <= 0:
            break

        tau += rho

    return tau


def suggested_spacing(tau):
    if np.isnan(tau):
        return np.nan

    return int(
        np.ceil(5.0 * tau)
    )


def run_one(
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

    # -------------------------
    # burn-in
    # -------------------------
    model.run(
        T,
        burnin,
    )

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

    # -------------------------
    # 매 sweep마다 측정
    # -------------------------
    for i in range(n_measure):

        model.run(
            T,
            1,
        )

        energy[i] = (
            model.total_energy()
            / (L * L)
        )

        magnetization[i] = (
            model.magnetization()
        )

        nv, nav = model.count_vortices()

        vortices[i] = (
            nv + nav
        )

    # -------------------------
    # ACF
    # -------------------------
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

    tau_E = integrated_autocorrelation_time(
        acf_E
    )

    tau_M = integrated_autocorrelation_time(
        acf_M
    )

    tau_V = integrated_autocorrelation_time(
        acf_V
    )

    spacing_E = suggested_spacing(
        tau_E
    )

    spacing_M = suggested_spacing(
        tau_M
    )

    spacing_V = suggested_spacing(
        tau_V
    )

    valid_spacings = [
        x
        for x in [
            spacing_E,
            spacing_M,
            spacing_V,
        ]
        if not (
            isinstance(x, float)
            and np.isnan(x)
        )
    ]

    recommended = (
        max(valid_spacings)
        if valid_spacings
        else np.nan
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


def fmt(x):
    if np.isnan(x):
        return "   NaN"
    return f"{x:6.2f}"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--burnin",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--measure",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--max-lag",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    out_dir = (
        THIS_DIR
        / "L256"
        / "autocorrelation_bkt"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    jobs = [
        (T, seed)
        for T in TEMPERATURES
        for seed in SEEDS
    ]

    print("=" * 78)
    print("L=256 BKT-region autocorrelation validation")
    print("=" * 78)
    print(f"temperatures = {TEMPERATURES}")
    print(f"seeds        = {SEEDS}")
    print(f"burn-in      = {args.burnin}")
    print(f"measure      = {args.measure}")
    print(f"max lag      = {args.max_lag}")
    print(f"workers      = {args.workers}")
    print(f"total jobs   = {len(jobs)}")
    print("=" * 78, flush=True)

    results = []

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:

        future_map = {}

        for T, seed in jobs:

            future = executor.submit(
                run_one,
                T,
                seed,
                args.burnin,
                args.measure,
                args.max_lag,
            )

            future_map[future] = (
                T,
                seed,
            )

        for future in as_completed(
            future_map
        ):

            result = future.result()

            results.append(result)

            print(
                f"T={result['T']:.3f} "
                f"seed={result['seed']} | "
                f"tau_E={fmt(result['tau_E'])} | "
                f"tau_M={fmt(result['tau_M'])} | "
                f"tau_V={fmt(result['tau_V'])} | "
                f"recommended="
                f"{result['recommended']}",
                flush=True,
            )

            np.savez_compressed(
                out_dir
                / (
                    f"autocorr_"
                    f"T{result['T']:.3f}_"
                    f"seed{result['seed']}.npz"
                ),

                T=np.array(result["T"]),
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

                burnin=np.array(
                    args.burnin
                ),

                n_measure=np.array(
                    args.measure
                ),

                lattice_size=np.array(
                    L
                ),
            )

    # --------------------------------
    # Temperature별 summary
    # --------------------------------

    print()
    print("=" * 78)
    print("Temperature summary")
    print("=" * 78)

    summary_lines = []

    header = (
        "T "
        "tauE_mean tauE_std tauE_max "
        "tauM_mean tauV_mean "
        "recommended_from_tauEmax\n"
    )

    summary_lines.append(
        header
    )

    for T in TEMPERATURES:

        subset = [
            r
            for r in results
            if np.isclose(
                r["T"],
                T,
            )
        ]

        tau_E = np.array(
            [
                r["tau_E"]
                for r in subset
            ],
            dtype=float,
        )

        tau_M = np.array(
            [
                r["tau_M"]
                for r in subset
            ],
            dtype=float,
        )

        tau_V = np.array(
            [
                r["tau_V"]
                for r in subset
            ],
            dtype=float,
        )

        tauE_mean = np.nanmean(
            tau_E
        )

        tauE_std = np.nanstd(
            tau_E,
            ddof=1,
        )

        tauE_max = np.nanmax(
            tau_E
        )

        tauM_mean = np.nanmean(
            tau_M
        )

        tauV_mean = np.nanmean(
            tau_V
        )

        recommended = int(
            np.ceil(
                5.0 * tauE_max
            )
        )

        print(
            f"T={T:.3f} | "
            f"tau_E="
            f"{tauE_mean:.2f} ± "
            f"{tauE_std:.2f} "
            f"(max {tauE_max:.2f}) | "
            f"5*tau_Emax -> "
            f"{recommended}",
            flush=True,
        )

        summary_lines.append(
            f"{T:.3f} "
            f"{tauE_mean:.6f} "
            f"{tauE_std:.6f} "
            f"{tauE_max:.6f} "
            f"{tauM_mean:.6f} "
            f"{tauV_mean:.6f} "
            f"{recommended}\n"
        )

    summary_path = (
        out_dir
        / "summary_by_temperature.txt"
    )

    summary_path.write_text(
        "".join(summary_lines)
    )

    all_tau_E = np.array(
        [
            r["tau_E"]
            for r in results
        ],
        dtype=float,
    )

    worst_tau_E = np.nanmax(
        all_tau_E
    )

    global_spacing = int(
        np.ceil(
            5.0 * worst_tau_E
        )
    )

    print()
    print("=" * 78)
    print(
        f"Worst tau_E across all runs = "
        f"{worst_tau_E:.3f}"
    )
    print(
        f"5 * worst tau_E = "
        f"{global_spacing}"
    )
    print(
        f"Summary saved to: "
        f"{summary_path}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
