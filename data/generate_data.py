"""
XY 모델 시뮬레이션으로 CNN 학습용 데이터셋을 생성해서
data/xy_dataset_<번호>.npz 파일로 저장.

실행할 때마다 기존 파일을 덮어쓰지 않고 다음 번호로 저장함
(xy_dataset_1.npz, xy_dataset_2.npz, ...).
seed는 실행할 때마다 무작위로 정해지므로 매번 새로운 데이터가 만들어지고,
사용한 seed는 npz 파일 안에 "seed"로 함께 저장됨.
같은 데이터를 재현하려면 그 값을 --seed로 넘기면 됨.

train.py는 이 파일들 중 하나를 validation 전용으로, 나머지를 학습용으로 씀.

test용 데이터는 --test 옵션으로 따로 생성해서 data/xy_testset.npz 로 저장.
test 온도(TEST_TEMPERATURES)는 학습 온도 그리드(0.025 간격) 사이의 온도라
학습/검증 데이터와 온도가 절대 겹치지 않음.

실행:
    python data/generate_data.py                 # 학습/검증용 xy_dataset_<다음 번호>.npz
    python data/generate_data.py --test          # test용 xy_testset.npz
    python data/generate_data.py --seed 12345    # 특정 seed로 재현 (--test와 같이 써도 됨)
"""

import argparse
import os
import re
import secrets
import sys
import time

import numpy as np

# simulation/ 폴더를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulation"))
from xy_model_numba import generate_dataset, build_temperature_grid  # noqa: E402


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(DATA_DIR, "xy_testset.npz")

# 학습 그리드(0.300, 0.325, ..., 2.000)의 정확히 한가운데 온도들, 0.15 간격
# (0.8125, 0.9625가 T_BKT ~ 0.893 양옆)
TEST_TEMPERATURES = [0.3625, 0.5125, 0.6625, 0.8125, 0.9625,
                     1.1125, 1.2625, 1.4125, 1.5625, 1.7125,
                     1.8625]


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


def random_seed():
    """OS 난수로 정한 seed (numba/NumPy RNG가 받는 32비트 범위)."""
    return secrets.randbits(32)


def save_samples(samples, out_path, seed):
    """generate_dataset() 결과(list[dict])를 배열로 정리해서 npz 하나에 압축 저장."""
    spin_configs = np.stack([s["spin_config"] for s in samples]).astype(np.float32)
    temps = np.array([s["T"] for s in samples], dtype=np.float32)
    energies = np.array([s["energy"] for s in samples], dtype=np.float32)
    mags = np.array([s["magnetization"] for s in samples], dtype=np.float32)
    n_vortex = np.array([s["n_vortex"] for s in samples], dtype=np.int32)
    n_antivortex = np.array([s["n_antivortex"] for s in samples], dtype=np.int32)

    np.savez_compressed(
        out_path,
        spin_configs=spin_configs,
        temperatures=temps,
        energies=energies,
        magnetizations=mags,
        n_vortex=n_vortex,
        n_antivortex=n_antivortex,
        seed=np.array(seed, dtype=np.int64),   # 재현용으로 사용한 seed 기록
    )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n저장 완료: {out_path} ({size_mb:.1f} MB)")


def main(L=64, n_burnin=50, n_samples_per_T=60, sweeps_between=8, seed=None):

    index = next_dataset_index()
    if seed is None:
        seed = random_seed()
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

    save_samples(samples, dataset_path(index), seed)


def main_test(L=64, n_burnin=200, n_samples_per_T=120, sweeps_between=8, seed=None):
    """
    test용 데이터셋 생성 -> data/xy_testset.npz (이미 있으면 덮어씀)

    test 온도끼리는 0.15씩 떨어져 있어서(학습 그리드는 0.025),
    온도를 바꾼 뒤 평형에 도달하도록 burn-in을 학습용보다 길게 줌.
    """
    if seed is None:
        seed = random_seed()
    temperatures = sorted(TEST_TEMPERATURES, reverse=True)   # 고온 -> 저온
    print(f"test 데이터셋 생성 (seed={seed})")
    print(f"test 온도 ({len(temperatures)}개): {temperatures}")

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

    save_samples(samples, TEST_PATH, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="학습 그리드 사이 온도에서 test용 데이터셋(xy_testset.npz) 생성")
    parser.add_argument("--seed", type=int, default=None,
                        help="재현용 seed (기본: 실행마다 무작위, 사용한 값은 npz에 저장됨)")
    args = parser.parse_args()

    if args.test:
        main_test(seed=args.seed)
    else:
        main(seed=args.seed)
