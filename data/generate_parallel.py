"""
조원의 haksulje/.../data/generate_256_parallel.py 와 같은 방식으로 스핀 데이터를 만들되,
격자 크기 L을 옵션으로 받음 (유한 크기 외삽용 L=32, 64, 128 데이터).

L=256과 똑같은 절차:
    - 온도마다 무작위 배치에서 독립적으로 시작 (annealing 아님), 온도마다 seed 따로
    - burn-in 100 sweep, 온도당 샘플 60개, 샘플 사이 30 sweep (Wolff)
    - 온도 61개 (1.80 ~ 0.30, 0.025 간격)
    - 온도 하나 끝날 때마다 임시 파일(_temperature_chunks)로 저장 -> 중간에 멈춰도 다시 실행하면 이어서 함
    - 다 끝나면 data/L<L>/xy_dataset_<번호>.npz 로 병합 (L=256 파일과 같은 항목)

seed: dataset seed = L*10 + 번호  (예: L=32 -> 321~325, L=128 -> 1281~1285)
      단 L=256은 조원 generate_256_parallel.py와 같은 1001~1005 -> 같은 데이터가 그대로 재현됨
      온도별 seed = dataset seed * 100000 + 온도 번호

실행:
    python3 data/generate_parallel.py --L 32
    python3 data/generate_parallel.py --L 256            # 조원 L256 데이터와 동일 (오래 걸림)
    python3 data/generate_parallel.py --L 128 --datasets 1 2 --workers 4
"""

import os

# 각 worker가 내부적으로 추가 thread를 만들지 않도록 제한
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "simulation"))

from xy_model_numba import build_temperature_grid, generate_dataset  # noqa: E402

J = 1.0
N_BURNIN = 100
N_SAMPLES_PER_T = 60
SWEEPS_BETWEEN = 30


def fmt(sec):
    sec = max(0, int(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def dataset_seed(L, k):
    if L == 256:
        return 1000 + k        # generate_256_parallel.py 의 base_seeds (1001~1005)
    return L * 10 + k


def temperature_seed(base_seed, temp_index):
    return int(base_seed * 100000 + temp_index)


def chunk_path(out_dir, k, i):
    return out_dir / "_temperature_chunks" / f"dataset_{k}" / f"T_{i:03d}.npz"


def generate_one_temperature(L, out_dir, k, i, T, base_seed):
    out = chunk_path(out_dir, k, i)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return k, i, T, 0.0, True

    seed = temperature_seed(base_seed, i)
    t0 = time.time()
    samples = generate_dataset(L=L, J=J, temperatures=[float(T)], n_burnin=N_BURNIN,
                               n_samples_per_T=N_SAMPLES_PER_T, sweeps_between=SWEEPS_BETWEEN,
                               seed=seed, verbose=False)
    part = out.with_suffix(".part.npz")
    np.savez_compressed(
        part,
        spin_configs=np.stack([s["spin_config"] for s in samples]),
        temperatures=np.array([s["T"] for s in samples], dtype=np.float64),
        energies=np.array([s["energy"] for s in samples], dtype=np.float64),
        magnetizations=np.array([s["magnetization"] for s in samples], dtype=np.float64),
        n_vortex=np.array([s["n_vortex"] for s in samples], dtype=np.int64),
        n_antivortex=np.array([s["n_antivortex"] for s in samples], dtype=np.int64),
        seed=np.array(seed, dtype=np.int64),
    )
    os.replace(part, out)
    return k, i, T, time.time() - t0, False


def merge_dataset(L, out_dir, k, temperatures, base_seed):
    keys = ["spin_configs", "temperatures", "energies", "magnetizations", "n_vortex", "n_antivortex"]
    parts = {key: [] for key in keys}
    t_seeds = []
    for i in range(len(temperatures)):
        with np.load(chunk_path(out_dir, k, i)) as d:
            for key in keys:
                parts[key].append(d[key])
            t_seeds.append(int(d["seed"]))
    final = out_dir / f"xy_dataset_{k}.npz"
    part = out_dir / f"xy_dataset_{k}.part.npz"
    np.savez_compressed(
        part,
        **{key: np.concatenate(v) for key, v in parts.items()},
        seed=np.array(base_seed, dtype=np.int64),
        temperature_seeds=np.array(t_seeds, dtype=np.int64),
        lattice_size=np.array(L, dtype=np.int64),
    )
    os.replace(part, final)
    print(f"[MERGE] {final}  ({final.stat().st_size / 1e6:.0f} MB)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--datasets", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    L = args.L
    out_dir = THIS_DIR / f"L{L}"
    out_dir.mkdir(parents=True, exist_ok=True)
    temperatures = list(build_temperature_grid())

    tasks = [(L, out_dir, k, i, float(T), dataset_seed(L, k))
             for k in args.datasets for i, T in enumerate(temperatures)]
    print(f"L={L}  datasets={args.datasets}  workers={args.workers}  온도 {len(temperatures)}개  "
          f"burn-in={N_BURNIN}  samples/T={N_SAMPLES_PER_T}  sweeps_between={SWEEPS_BETWEEN}  "
          f"작업 {len(tasks)}개", flush=True)

    t_all = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(generate_one_temperature, *t) for t in tasks]
        for n, fut in enumerate(as_completed(futures), start=1):
            k, i, T, sec, skipped = fut.result()
            el = time.time() - t_all
            print(f"[{n:3d}/{len(tasks)}] {'SKIP' if skipped else 'DONE'} D{k} T={T:.3f} "
                  f"job={fmt(sec)} elapsed={fmt(el)} ETA={fmt(el / n * (len(tasks) - n))}", flush=True)

    for k in args.datasets:
        merge_dataset(L, out_dir, k, temperatures, dataset_seed(L, k))
    print(f"전체 완료: {fmt(time.time() - t_all)}", flush=True)


if __name__ == "__main__":
    main()
