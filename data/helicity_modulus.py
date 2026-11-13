"""
스핀 배치 데이터(xy_dataset_<번호>.npz)에서 helicity modulus Υ(T)를 계산하고,
Υ(T) = 2T/π 가 되는 온도(유한 크기 L에서의 BKT 전이온도 T_c(L))를 구함.

정의 (J = 1, k_B = 1, 주기 경계조건):
    x 방향 결합 <ij>_x 에 대해
        E_x = Σ cos(θ_i - θ_j)          (x 방향 결합 에너지의 부호를 바꾼 것)
        I_x = Σ sin(θ_i - θ_j)          (x 방향 "스핀 전류")
    Υ_x = (1/L²) [ <E_x> - (1/T) <I_x²> ]
    y 방향도 똑같이 구해서 Υ = (Υ_x + Υ_y) / 2

    cos, sin 채널을 그대로 쓰면 각도 계산 없이:
        cos(θ_i - θ_j) = c_i c_j + s_i s_j
        sin(θ_i - θ_j) = s_i c_j - c_i s_j

L=256 파일은 하나가 수 GB라서 spin_configs를 샘플 묶음 단위로 순서대로 읽음.
샘플별 E_x, E_y, I_x, I_y 를 helicity_<번호>.npz 로 저장해 두므로,
통계/그래프는 원본을 다시 읽지 않고 --analyze 만으로 다시 그릴 수 있음.

실행:
    python3 data/helicity_modulus.py <xy_dataset_1.npz ...> --out-dir vortexCNN/data/helicity/L256
    python3 data/helicity_modulus.py --analyze --out-dir vortexCNN/data/helicity/L256
"""

import argparse
import glob
import os
import re
import time
import zipfile

import numpy as np

T_BKT = 0.893


def bond_sums(spins):
    """spins: (B, 2, L, L) [cos, sin] -> 샘플별 (E_x, E_y, I_x, I_y)"""
    c, s = spins[:, 0], spins[:, 1]
    out = []
    for axis in (2, 1):                         # x 방향(열), y 방향(행)
        cn, sn = np.roll(c, -1, axis=axis), np.roll(s, -1, axis=axis)   # 이웃 j (주기 경계)
        e = (c * cn + s * sn).sum(axis=(1, 2))  # Σ cos(θ_i - θ_j)
        i = (s * cn - c * sn).sum(axis=(1, 2))  # Σ sin(θ_i - θ_j)
        out += [e, i]
    ex, ix, ey, iy = out
    return ex, ey, ix, iy


def read_small(z, key):
    with z.open(key + ".npy") as f:
        return np.lib.format.read_array(f)


def process_file(src, out_path, batch=32):
    with zipfile.ZipFile(src) as z:
        temps = read_small(z, "temperatures").astype(np.float64)
        energies = read_small(z, "energies").astype(np.float64)
        with z.open("spin_configs.npy") as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
            n, _, L, _ = shape
            per = 2 * L * L * dtype.itemsize
            res = np.empty((4, n))
            for start in range(0, n, batch):
                b = min(batch, n - start)
                x = np.frombuffer(f.read(b * per), dtype=dtype).reshape(b, 2, L, L).astype(np.float64)
                res[:, start:start + b] = bond_sums(x)
    ex, ey, ix, iy = res
    # 검증: 전체 에너지 E = -J Σ_<ij> cos = -(E_x + E_y) 이어야 함
    max_err = np.max(np.abs(-(ex + ey) - energies) / (L * L))
    np.savez(out_path, temperatures=temps, E_x=ex, E_y=ey, I_x=ix, I_y=iy, L=L,
             source_file=os.path.basename(src))
    return L, max_err


def helicity_per_T(path):
    d = np.load(path)
    T, L = d["temperatures"], int(d["L"])
    ts = np.unique(np.round(T, 4))
    ups = []
    for t in ts:
        m = np.isclose(T, t, atol=1e-4)
        ux = (d["E_x"][m].mean() - (d["I_x"][m] ** 2).mean() / t) / (L * L)
        uy = (d["E_y"][m].mean() - (d["I_y"][m] ** 2).mean() / t) / (L * L)
        ups.append(0.5 * (ux + uy))
    return ts, np.array(ups), L


def crossing(ts, ups):
    """Υ(T) - 2T/π 가 +에서 -로 바뀌는 온도 (온도 오름차순, 선형 보간)."""
    g = ups - 2 * ts / np.pi
    for k in range(len(ts) - 1):
        if g[k] > 0 >= g[k + 1]:
            return ts[k] + (ts[k + 1] - ts[k]) * g[k] / (g[k] - g[k + 1])
    return float("nan")


def analyze(out_dir, cnn_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "helicity_*.npz")),
                   key=lambda p: int(re.search(r"helicity_(\d+)", p).group(1)))
    per_file = [helicity_per_T(p) for p in files]
    ts, L = per_file[0][0], per_file[0][2]
    U = np.array([u for _, u, _ in per_file])        # (파일 수, 온도 수)
    tcs = np.array([crossing(ts, u) for u in U])
    u_mean = U.mean(0)
    u_err = U.std(0, ddof=1) / np.sqrt(len(U)) if len(U) > 1 else np.zeros_like(u_mean)

    print(f"\nL={L}, 파일 {len(files)}개")
    print("   T      Υ(T) (파일 평균 ± 표준오차)     2T/π")
    for t, u, e in zip(ts, u_mean, u_err):
        if 0.75 <= t <= 1.1:
            print(f"  {t:.3f}   {u:.4f} ± {e:.4f}          {2 * t / np.pi:.4f}"
                  + ("   <-" if u > 2 * t / np.pi and np.isfinite(u) else ""))
    print("\n파일별 T_c(L) [Υ = 2T/π 교차]: " + ", ".join(f"{t:.4f}" for t in tcs))
    tc, tc_err = tcs.mean(), (tcs.std(ddof=1) / np.sqrt(len(tcs)) if len(tcs) > 1 else 0.0)
    print(f"T_c(L={L}) = {tc:.4f} ± {tc_err:.4f}    (무한 격자 T_BKT ≈ {T_BKT})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(ts, u_mean, yerr=u_err, fmt="o-", ms=3, color="k", label=f"Υ(T), L={L}")
    ax.plot(ts, 2 * ts / np.pi, "--", color="gray", label="2T/π")
    ax.axvline(tc, color="k", lw=1, label=f"Υ = 2T/π : T_c(L)={tc:.3f}")
    ax.axvline(T_BKT, color="k", ls=":", lw=1, label=f"T_BKT (L=∞) = {T_BKT}")
    ax.set_xlabel("T [J]")
    ax.set_ylabel("Υ [J]")

    # vortexCNN 결과가 있으면 같은 온도축에 P(고온상)를 겹쳐 그림
    curves = sorted(glob.glob(os.path.join(cnn_dir, "curves_*.npz")))
    if curves:
        ax2 = ax.twinx()
        for p in curves:
            c = np.load(p)
            ax2.plot(c["test_T"], c["test_cnn"], "-", lw=1.2, alpha=0.8,
                     label=f"CNN P(high), train T≤{float(c['t_low'])}/≥{float(c['t_high'])}: "
                           f"T_c={float(c['test_tc_cnn']):.3f}")
        ax2.set_ylabel("CNN P(high-T phase)")
        ax2.set_ylim(-0.02, 1.02)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    else:
        ax.legend(fontsize=8)
    ax.set_xlim(ts[0], ts[-1])
    fig.tight_layout()
    png = os.path.join(out_dir, "helicity_vs_cnn.png")
    fig.savefig(png, dpi=150)
    print(f"저장: {png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sources", nargs="*", help="xy_dataset_<번호>.npz 파일들 (스핀 배치)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--analyze", action="store_true", help="저장된 helicity_*.npz로 통계/그래프만")
    p.add_argument("--cnn-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "..", "vortexCNN", "results", "L256"),
                   help="CNN 곡선(curves_*.npz)을 겹쳐 그릴 폴더")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for src in args.sources:
        k = re.fullmatch(r"xy_dataset_(\d+)\.npz", os.path.basename(src)).group(1)
        out = os.path.join(args.out_dir, f"helicity_{k}.npz")
        t0 = time.perf_counter()
        L, err = process_file(src, out)
        print(f"{os.path.basename(src)} -> {out}  L={L}  "
              f"에너지 검증 최대오차(per site)={err:.1e}  ({time.perf_counter() - t0:.0f}초)", flush=True)

    analyze(args.out_dir, args.cnn_dir)


if __name__ == "__main__":
    main()
