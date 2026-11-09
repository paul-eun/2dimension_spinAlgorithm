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
    8001,
    8002,
    8003,
]


def running_mean(x, window):
    x = np.asarray(x, dtype=np.float64)

    if window <= 1:
        return x.copy()

    kernel = np.ones(window) / window

    return np.convolve(
        x,
        kernel,
        mode="valid",
    )


def run_one(T, seed, n_sweeps):

    model = XYModel2DFast(
        L=L,
        J=J,
        seed=seed,
    )

    energy = np.empty(
        n_sweeps,
        dtype=np.float64,
    )

    magnetization = np.empty(
        n_sweeps,
        dtype=np.float64,
    )

    vortices = np.empty(
        n_sweeps,
        dtype=np.float64,
    )

    for i in range(n_sweeps):

        model.run(T, 1)

        energy[i] = (
            model.total_energy()
            / (L * L)
        )

        magnetization[i] = (
            model.magnetization()
        )

        nv, nav = model.count_vortices()

        vortices[i] = nv + nav

    return {
        "T": T,
        "seed": seed,
        "energy": energy,
        "magnetization": magnetization,
        "vortices": vortices,
    }


def estimate_equilibration(x, tail_start=200, window=20, tolerance_sigma=2.0):
    """
    마지막 구간을 equilibrium reference로 사용하고,
    running mean이 reference 평균 ± tolerance_sigma * SEM
    안으로 들어온 뒤 유지되는 최초 시점을 찾는다.

    절대적인 통계적 증명은 아니고 burn-in 후보를 정하기 위한
    실용적 diagnostic.
    """

    x = np.asarray(x, dtype=np.float64)

    if tail_start >= len(x):
        raise ValueError("tail_start must be smaller than series length")

    ref = x[tail_start:]

    ref_mean = np.mean(ref)
    ref_std = np.std(ref, ddof=1)

    ref_sem = (
        ref_std
        / np.sqrt(len(ref))
    )

    # 너무 작은 SEM 때문에 기준이 지나치게 엄격해지는 것 방지
    tolerance = max(
        tolerance_sigma * ref_sem,
        0.10 * ref_std,
    )

    rm = running_mean(
        x,
        window,
    )

    lower = ref_mean - tolerance
    upper = ref_mean + tolerance

    inside = (
        (rm >= lower)
        & (rm <= upper)
    )

    # 한번 들어온 뒤 나머지 구간 대부분이 안정적인 최초 위치
    for i in range(len(inside)):

        remaining = inside[i:]

        if len(remaining) < 20:
            break

        fraction_inside = np.mean(
            remaining
        )

        if fraction_inside >= 0.90:
            return i + window - 1

    return np.nan


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--sweeps",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    out_dir = (
        THIS_DIR
        / "L256"
        / "burnin_test"
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

    print("=" * 76)
    print("L=256 burn-in validation")
    print("=" * 76)
    print(f"temperatures = {TEMPERATURES}")
    print(f"seeds        = {SEEDS}")
    print(f"sweeps/run   = {args.sweeps}")
    print(f"workers      = {args.workers}")
    print(f"total jobs   = {len(jobs)}")
    print("=" * 76, flush=True)

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
                args.sweeps,
            )

            future_map[future] = (
                T,
                seed,
            )

        for future in as_completed(
            future_map
        ):

            r = future.result()

            eq_E = estimate_equilibration(
                r["energy"]
            )

            eq_M = estimate_equilibration(
                r["magnetization"]
            )

            r["eq_E"] = eq_E
            r["eq_M"] = eq_M

            results.append(r)

            print(
                f"T={r['T']:.3f} "
                f"seed={r['seed']} | "
                f"eq_E={eq_E} | "
                f"eq_M={eq_M}",
                flush=True,
            )

            np.savez_compressed(
                out_dir
                / (
                    f"burnin_"
                    f"T{r['T']:.3f}_"
                    f"seed{r['seed']}.npz"
                ),

                T=np.array(r["T"]),
                seed=np.array(r["seed"]),

                energy=r["energy"],
                magnetization=r["magnetization"],
                vortices=r["vortices"],

                eq_energy=np.array(eq_E),
                eq_magnetization=np.array(eq_M),

                lattice_size=np.array(L),
            )

    print()
    print("=" * 76)
    print("Temperature summary")
    print("=" * 76)

    summary_lines = [
        "T eqE_max eqM_max recommended_burnin\n"
    ]

    global_recommended = 0

    for T in TEMPERATURES:

        subset = [
            r
            for r in results
            if np.isclose(
                r["T"],
                T,
            )
        ]

        eq_E = np.array(
            [
                r["eq_E"]
                for r in subset
            ],
            dtype=float,
        )

        eq_M = np.array(
            [
                r["eq_M"]
                for r in subset
            ],
            dtype=float,
        )

        valid_E = eq_E[
            np.isfinite(eq_E)
        ]

        valid_M = eq_M[
            np.isfinite(eq_M)
        ]

        max_E = (
            np.max(valid_E)
            if len(valid_E)
            else np.nan
        )

        max_M = (
            np.max(valid_M)
            if len(valid_M)
            else np.nan
        )

        valid = [
            x
            for x in [
                max_E,
                max_M,
            ]
            if np.isfinite(x)
        ]

        if valid:
            # 측정된 equilibration point보다 여유 있게
            recommended = int(
                np.ceil(
                    1.5 * max(valid)
                    / 10
                ) * 10
            )
        else:
            recommended = args.sweeps

        global_recommended = max(
            global_recommended,
            recommended,
        )

        print(
            f"T={T:.3f} | "
            f"max eq_E={max_E} | "
            f"max eq_M={max_M} | "
            f"recommended burn-in={recommended}",
            flush=True,
        )

        summary_lines.append(
            f"{T:.3f} "
            f"{max_E} "
            f"{max_M} "
            f"{recommended}\n"
        )

    summary_path = (
        out_dir
        / "summary.txt"
    )

    summary_path.write_text(
        "".join(summary_lines)
    )

    print()
    print("=" * 76)
    print(
        f"Global recommended burn-in = "
        f"{global_recommended}"
    )
    print(
        f"Summary saved to: "
        f"{summary_path}"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
