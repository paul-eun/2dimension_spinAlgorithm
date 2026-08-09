"""
XY 모델 시뮬레이션으로 CNN 학습용 데이터셋을 생성해서
data/xy_dataset.npz 파일로 저장.

한 번 실행해서 저장해두면, 이후 train.py는 이 파일을 불러오기만
하면 되므로 학습을 여러 번 반복해도 시뮬레이션을 다시 돌릴 필요가 없음.

실행:
    cd data
    python3 generate_data.py
"""

import os
import sys
import time

import numpy as np

# simulation/ 폴더를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulation"))
from xy_model_numba import generate_dataset, build_temperature_grid  # noqa: E402


def main(L=64, n_burnin=1000, n_samples_per_T=20, sweeps_between=40,
         seed=42, out_filename="xy_dataset.npz"):

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

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, out_filename)

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
