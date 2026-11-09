import os

# 각 worker가 내부적으로 추가 thread를 만들지 않도록 제한
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


# --------------------------------------------------
# Path 설정
# --------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
SIM_DIR = ROOT_DIR / "simulation"

sys.path.insert(0, str(SIM_DIR))

from xy_model_numba import generate_dataset, build_temperature_grid


L = 256
J = 1.0

N_BURNIN = 100
N_SAMPLES_PER_T = 60
SWEEPS_BETWEEN = 30

OUTPUT_DIR = THIS_DIR / "L256"
TMP_ROOT = OUTPUT_DIR / "_temperature_chunks"


# --------------------------------------------------
# Utility
# --------------------------------------------------

def format_time(seconds):
    seconds = max(0, int(seconds))

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:02d}"


def temperature_seed(base_seed, temp_index):
    """
    dataset seed와 temperature index를 이용해서
    각 온도마다 독립적인 deterministic seed 생성.
    """
    return int(base_seed * 100000 + temp_index)


def chunk_path(dataset_index, temp_index):
    folder = TMP_ROOT / f"dataset_{dataset_index}"

    return folder / f"T_{temp_index:03d}.npz"


# --------------------------------------------------
# 한 온도 생성
# --------------------------------------------------

def generate_one_temperature(
    dataset_index,
    temp_index,
    temperature,
    base_seed,
):
    """
    하나의 dataset / 하나의 temperature만 생성한다.

    결과는 즉시 임시 npz로 저장한다.
    """

    out_path = chunk_path(
        dataset_index,
        temp_index,
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 이미 완료된 온도라면 재계산하지 않음
    if out_path.exists():
        return {
            "dataset": dataset_index,
            "temp_index": temp_index,
            "temperature": temperature,
            "seconds": 0.0,
            "skipped": True,
        }

    seed = temperature_seed(
        base_seed,
        temp_index,
    )

    start = time.time()

    samples = generate_dataset(
        L=L,
        J=J,
        temperatures=[float(temperature)],
        n_burnin=N_BURNIN,
        n_samples_per_T=N_SAMPLES_PER_T,
        sweeps_between=SWEEPS_BETWEEN,
        seed=seed,
        verbose=False,
    )

    spin_configs = np.stack(
        [s["spin_config"] for s in samples],
        axis=0,
    )

    temperatures = np.asarray(
        [s["T"] for s in samples],
        dtype=np.float64,
    )

    energies = np.asarray(
        [s["energy"] for s in samples],
        dtype=np.float64,
    )

    magnetizations = np.asarray(
        [s["magnetization"] for s in samples],
        dtype=np.float64,
    )

    n_vortex = np.asarray(
        [s["n_vortex"] for s in samples],
        dtype=np.int64,
    )

    n_antivortex = np.asarray(
        [s["n_antivortex"] for s in samples],
        dtype=np.int64,
    )

    # 완전히 저장된 뒤 atomic rename
    part_path = out_path.with_suffix(".part.npz")

    np.savez_compressed(
        part_path,
        spin_configs=spin_configs,
        temperatures=temperatures,
        energies=energies,
        magnetizations=magnetizations,
        n_vortex=n_vortex,
        n_antivortex=n_antivortex,
        seed=np.array(seed, dtype=np.int64),
        dataset_index=np.array(dataset_index, dtype=np.int64),
        temperature_index=np.array(temp_index, dtype=np.int64),
        lattice_size=np.array(L, dtype=np.int64),
    )

    os.replace(
        part_path,
        out_path,
    )

    elapsed = time.time() - start

    return {
        "dataset": dataset_index,
        "temp_index": temp_index,
        "temperature": temperature,
        "seconds": elapsed,
        "skipped": False,
    }


# --------------------------------------------------
# dataset 하나로 병합
# --------------------------------------------------

def merge_dataset(
    dataset_index,
    temperatures,
    base_seed,
):
    print()
    print(
        f"[MERGE] dataset #{dataset_index} 병합 시작",
        flush=True,
    )

    spin_list = []
    temp_list = []
    energy_list = []
    mag_list = []
    vortex_list = []
    antivortex_list = []
    temperature_seeds = []

    for i, T in enumerate(temperatures):

        path = chunk_path(
            dataset_index,
            i,
        )

        if not path.exists():
            raise RuntimeError(
                f"missing chunk: {path}"
            )

        with np.load(path) as d:

            spin_list.append(
                d["spin_configs"]
            )

            temp_list.append(
                d["temperatures"]
            )

            energy_list.append(
                d["energies"]
            )

            mag_list.append(
                d["magnetizations"]
            )

            vortex_list.append(
                d["n_vortex"]
            )

            antivortex_list.append(
                d["n_antivortex"]
            )

            temperature_seeds.append(
                int(d["seed"])
            )

    spin_configs = np.concatenate(
        spin_list,
        axis=0,
    )

    all_temperatures = np.concatenate(
        temp_list,
        axis=0,
    )

    energies = np.concatenate(
        energy_list,
        axis=0,
    )

    magnetizations = np.concatenate(
        mag_list,
        axis=0,
    )

    n_vortex = np.concatenate(
        vortex_list,
        axis=0,
    )

    n_antivortex = np.concatenate(
        antivortex_list,
        axis=0,
    )

    final_path = (
        OUTPUT_DIR
        / f"xy_dataset_{dataset_index}.npz"
    )

    part_path = (
        OUTPUT_DIR
        / f"xy_dataset_{dataset_index}.part.npz"
    )

    np.savez_compressed(
        part_path,
        spin_configs=spin_configs,
        temperatures=all_temperatures,
        energies=energies,
        magnetizations=magnetizations,
        n_vortex=n_vortex,
        n_antivortex=n_antivortex,
        seed=np.array(
            base_seed,
            dtype=np.int64,
        ),
        temperature_seeds=np.asarray(
            temperature_seeds,
            dtype=np.int64,
        ),
        lattice_size=np.array(
            L,
            dtype=np.int64,
        ),
    )

    os.replace(
        part_path,
        final_path,
    )

    print(
        f"[MERGE] dataset #{dataset_index} 완료",
        flush=True,
    )

    print(
        f"        samples = {len(all_temperatures)}",
        flush=True,
    )

    print(
        f"        shape   = {spin_configs.shape}",
        flush=True,
    )

    print(
        f"        file    = {final_path}",
        flush=True,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="병렬 worker 수 (기본 24)",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="생성할 dataset 번호",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TMP_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_seeds = {
        1: 1001,
        2: 1002,
        3: 1003,
        4: 1004,
        5: 1005,
    }

    temperatures = list(
        build_temperature_grid()
    )

    print("=" * 70)
    print("L=256 parallel dataset generation")
    print("=" * 70)

    print(
        f"workers             = {args.workers}",
        flush=True,
    )

    print(
        f"datasets            = {args.datasets}",
        flush=True,
    )

    print(
        f"temperatures        = {len(temperatures)}",
        flush=True,
    )

    print(
        f"samples / T         = {N_SAMPLES_PER_T}",
        flush=True,
    )

    print(
        f"burn-in             = {N_BURNIN}",
        flush=True,
    )

    print(
        f"sweeps between      = {SWEEPS_BETWEEN}",
        flush=True,
    )

    total_tasks = (
        len(args.datasets)
        * len(temperatures)
    )

    tasks = []

    for dataset_index in args.datasets:

        if dataset_index not in base_seeds:
            raise ValueError(
                f"Unknown dataset index: "
                f"{dataset_index}"
            )

        for temp_index, T in enumerate(
            temperatures
        ):
            tasks.append(
                (
                    dataset_index,
                    temp_index,
                    float(T),
                    base_seeds[dataset_index],
                )
            )

    print(
        f"total temperature jobs = {total_tasks}",
        flush=True,
    )

    print("=" * 70, flush=True)

    start_all = time.time()

    completed = 0

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = [
            executor.submit(
                generate_one_temperature,
                *task,
            )
            for task in tasks
        ]

        for future in as_completed(futures):

            result = future.result()

            completed += 1

            elapsed = (
                time.time()
                - start_all
            )

            avg_per_job = (
                elapsed
                / completed
            )

            # 병렬 worker 수를 고려한
            # 단순 wall-time ETA
            remaining_jobs = (
                total_tasks
                - completed
            )

            estimated_remaining = (
                avg_per_job
                * remaining_jobs
            )

            progress = (
                100.0
                * completed
                / total_tasks
            )

            status = (
                "SKIP"
                if result["skipped"]
                else "DONE"
            )

            print(
                f"[{completed:3d}/{total_tasks}] "
                f"{progress:5.1f}% | "
                f"{status} | "
                f"D{result['dataset']} | "
                f"T={result['temperature']:.3f} | "
                f"job={format_time(result['seconds'])} | "
                f"elapsed={format_time(elapsed)} | "
                f"rough ETA={format_time(estimated_remaining)}",
                flush=True,
            )

    print()
    print("=" * 70)
    print("모든 temperature job 완료")
    print("=" * 70)

    for dataset_index in args.datasets:

        merge_dataset(
            dataset_index,
            temperatures,
            base_seeds[dataset_index],
        )

    total_elapsed = (
        time.time()
        - start_all
    )

    print()
    print(
        f"전체 완료: {format_time(total_elapsed)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
