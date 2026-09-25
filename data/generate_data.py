"""
XY 모델 시뮬레이션으로 CNN 학습용 데이터셋을 생성해서
data/xy_dataset_<번호>.npz 파일로 저장.

실행할 때마다 기존 파일을 덮어쓰지 않고 다음 번호로 저장함
(xy_dataset_1.npz, xy_dataset_2.npz, ...).
각 파일은 seed = 번호 로 생성되므로 파일끼리는 서로 독립적인 데이터이고,
같은 번호는 언제 다시 만들어도 똑같이 재현됨.

train.py는 이 파일들 중 하나를 validation 전용으로, 나머지를 학습용으로 씀.

실행:
    python data/generate_data.py
"""

import os
import re
import sys
import time

import numpy as np

# simulation/ 폴더를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulation"))
from xy_model_numba import generate_dataset, build_temperature_grid  # noqa: E402


DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def dataset_path(index):
    return os.path.join(DATA_DIR, f"xy_dataset_{index}.npz")


def existing_dataset_indices():
    """data/ 안에 있는 xy_dataset_<번호>.npz 들의 번호 (오름차순)."""
    indices = []
    for name in os.listdir(DATA_DIR):
        m = re.fullmatch(r"xy_dataset_(\d+)\.npz", name)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def next_dataset_index():
    """지금까지 만든 가장 큰 번호 + 1 (중간 파일을 지워도 번호가 생성 순서를 유지)."""
    indices = existing_dataset_indices()
    return indices[-1] + 1 if indices else 1


def main(L=64, n_burnin=50, n_samples_per_T=60, sweeps_between=5):

    index = next_dataset_index()
    seed = index
    print(f"데이터셋 #{index} 생성 (seed={seed})")

    temperatures = build_temperature_grid()
    print(f"온도 그리드 ({len(temperatures)}개): {temperatures}")

    t0 = time.perf_counter()
    samples = generate_dataset(
        L=L, J=1.0,
        temperatures=temperatures,
        n_burnin=n_burnin,
        n_samples_per_T=n_samples_per_T,
        sweeps_between=sweeps_between,
        seed=seed,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n총 {len(samples)}개 샘플 생성, {elapsed:.1f}초 소요")

    # list[dict] -> 배열로 정리해서 npz 하나에 압축 저장
    spin_configs = np.stack([s["spin_config"] for s in samples]).astype(np.float32)
    temps = np.array([s["T"] for s in samples], dtype=np.float32)
    energies = np.array([s["energy"] for s in samples], dtype=np.float32)
    mags = np.array([s["magnetization"] for s in samples], dtype=np.float32)
    n_vortex = np.array([s["n_vortex"] for s in samples], dtype=np.int32)
    n_antivortex = np.array([s["n_antivortex"] for s in samples], dtype=np.int32)

    out_path = dataset_path(index)

    np.savez_compressed(
        out_path,
        spin_configs=spin_configs,
        temperatures=temps,
        energies=energies,
        magnetizations=mags,
        n_vortex=n_vortex,
        n_antivortex=n_antivortex,
    )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n저장 완료: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
