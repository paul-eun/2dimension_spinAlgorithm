"""
스핀 배치 데이터(xy_dataset_<번호>.npz)에서 vortex 지도를 계산해서
vortex_maps_<번호>.npz 로 저장.

각 plaquette(격자 한 칸)의 winding number:
    +1 = vortex, -1 = antivortex, 0 = 없음
plaquette 정의는 simulation/xy_model_numba.py의 _count_vortices_numba와 같음:
    (i,j) -> (i,j+1) -> (i+1,j+1) -> (i+1,j) 순으로 돈 각도 차(-π~π로 감음)의 합 / 2π
주기 경계조건이라 L x L 격자에서 plaquette도 L x L 개.

L=256 파일은 하나가 수 GB라서 통째로 메모리에 올리지 않고
spin_configs를 샘플 묶음 단위로 순서대로 읽어서 계산함.

저장 항목:
    vortex_maps  (N, L, L) int8
    temperatures (N,)
    source_file  원본 파일 이름
    seed, lattice_size  (원본에 있으면 그대로 복사)

실행:
    python3 data/make_vortex_maps.py <xy_dataset_1.npz> [<xy_dataset_2.npz> ...] --out-dir <폴더>
    예) python3 data/make_vortex_maps.py data/L32/xy_dataset_{1,2,3,4,5}.npz --out-dir vortexCNN/data/vortex_maps/L32
"""

import argparse
import os
import re
import time
import zipfile

import numpy as np


def wrap_angle(a):
    """각도 차이를 -π ~ π 로 감음."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def vortex_maps_from_theta(theta):
    """theta: (B, L, L) -> winding number 지도 (B, L, L) int8"""
    t00 = theta
    t01 = np.roll(theta, -1, axis=2)       # (i, j+1)
    t11 = np.roll(t01, -1, axis=1)         # (i+1, j+1)
    t10 = np.roll(theta, -1, axis=1)       # (i+1, j)
    w = (wrap_angle(t01 - t00) + wrap_angle(t11 - t01)
         + wrap_angle(t10 - t11) + wrap_angle(t00 - t10)) / (2 * np.pi)
    return np.rint(w).astype(np.int8)


def read_small(z, key):
    with z.open(key + ".npy") as f:
        return np.lib.format.read_array(f)


def make_vortex_maps(src_path, out_path, batch=32):
    with zipfile.ZipFile(src_path) as z:
        names = {n[:-4] for n in z.namelist()}
        temperatures = read_small(z, "temperatures")
        extra = {k: read_small(z, k) for k in ("seed", "lattice_size") if k in names}

        with z.open("spin_configs.npy") as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
            if fortran:
                raise ValueError("Fortran 순서 배열은 지원하지 않음")
            n, _, L, _ = shape                         # (N, 2, L, L)
            per_sample = 2 * L * L * dtype.itemsize

            maps = np.empty((n, L, L), dtype=np.int8)
            for start in range(0, n, batch):
                b = min(batch, n - start)
                x = np.frombuffer(f.read(b * per_sample), dtype=dtype).reshape(b, 2, L, L)
                maps[start:start + b] = vortex_maps_from_theta(np.arctan2(x[:, 1], x[:, 0]))

    np.savez_compressed(
        out_path,
        vortex_maps=maps,
        temperatures=temperatures,
        source_file=np.array(os.path.basename(src_path)),
        **extra,
    )
    return maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", help="xy_dataset_<번호>.npz 파일들")
    parser.add_argument("--out-dir", required=True, help="vortex_maps_<번호>.npz 를 저장할 폴더")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for src in args.sources:
        m = re.fullmatch(r"xy_dataset_(\d+)\.npz", os.path.basename(src))
        if not m:
            raise ValueError(f"파일 이름이 xy_dataset_<번호>.npz 형식이 아님: {src}")
        out_path = os.path.join(args.out_dir, f"vortex_maps_{m.group(1)}.npz")

        t0 = time.perf_counter()
        maps = make_vortex_maps(src, out_path)
        n_v = (maps == 1).sum(axis=(1, 2))
        n_a = (maps == -1).sum(axis=(1, 2))
        print(f"{os.path.basename(src)} -> {out_path}  "
              f"shape={maps.shape}  vortex 수 = antivortex 수: {np.array_equal(n_v, n_a)}  "
              f"({time.perf_counter() - t0:.0f}초, {os.path.getsize(out_path) / 1e6:.1f} MB)",
              flush=True)


if __name__ == "__main__":
    main()
