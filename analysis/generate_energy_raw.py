import os
import sys
import secrets
import numpy as np

REPO = "../2dimension_spinAlgorithm"

sys.path.append(os.path.join(REPO, "simulation"))

from xy_model_numba import generate_dataset


L = 64

TEMPERATURES = [1.60, 1.70]

N_FILES = 4
N_SAMPLES_PER_T = 300

N_BURNIN = 200
SWEEPS_BETWEEN = 8

OUT_DIR = os.path.join(REPO, "data", "energy_match")
os.makedirs(OUT_DIR, exist_ok=True)


def save_samples(samples, path, seeds):
    np.savez_compressed(
        path,
        spin_configs=np.array(
            [s["spin_config"] for s in samples],
            dtype=np.float32
        ),
        temperatures=np.array(
            [s["T"] for s in samples],
            dtype=np.float32
        ),
        energies=np.array(
            [s["energy"] for s in samples],
            dtype=np.float32
        ),
        magnetizations=np.array(
            [s["magnetization"] for s in samples],
            dtype=np.float32
        ),
        n_vortex=np.array(
            [s["n_vortex"] for s in samples],
            dtype=np.int32
        ),
        n_antivortex=np.array(
            [s["n_antivortex"] for s in samples],
            dtype=np.int32
        ),
        seeds=np.array(seeds, dtype=np.uint32),
    )


for file_idx in range(1, N_FILES + 1):

    all_samples = []
    used_seeds = []

    print("=" * 60)
    print(f"energy_raw_{file_idx}.npz 생성")

    # 두 온도를 서로 완전히 독립적인 simulation으로 생성
    for T in TEMPERATURES:

        seed = secrets.randbits(32)
        used_seeds.append(seed)

        print(f"T={T:.2f}, seed={seed}")

        samples = generate_dataset(
            L=L,
            J=1.0,
            temperatures=[T],
            n_burnin=N_BURNIN,
            n_samples_per_T=N_SAMPLES_PER_T,
            sweeps_between=SWEEPS_BETWEEN,
            seed=seed,
            verbose=True,
        )

        all_samples.extend(samples)

    path = os.path.join(
        OUT_DIR,
        f"energy_raw_{file_idx}.npz"
    )

    save_samples(all_samples, path, used_seeds)

    print(f"저장 완료: {path}")
    print(f"총 {len(all_samples)} samples\n")
