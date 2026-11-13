"""
L마다 구한 전이온도 T_c(L)을 모아서 L -> ∞ 로 외삽.

BKT 유한 크기 공식:  T_c(L) = T_BKT + a / (ln L)^2
    x = 1/(ln L)^2 로 두면 직선이므로, x=0 (L=∞) 에서의 절편이 T_BKT.

모으는 값:
    - helicity modulus: vortexCNN/data/helicity/L<L>/helicity_*.npz 에서 파일별 Υ = 2T/π 교차 -> 평균 ± 표준오차
    - CNN / 밀도 MLP (양 끝 학습): vortexCNN/results/L<L>/curves_tlow<lo>_thigh<hi>_seed0.npz 의 test T_c

실행:
    python3 vortexCNN/extrapolate.py
    python3 vortexCNN/extrapolate.py --sizes 32 64 128 256 --exclude-fit 32
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "data"))
from helicity_modulus import crossing, helicity_per_T  # noqa: E402

T_BKT = 0.893
WINDOWS = [(0.7, 1.1), (0.6, 1.2), (0.5, 1.3)]


def helicity_tc(L):
    files = sorted(glob.glob(os.path.join(ROOT, "vortexCNN", "data", "helicity", f"L{L}", "helicity_*.npz")))
    if not files:
        return np.nan, np.nan
    tcs = np.array([crossing(*helicity_per_T(p)[:2]) for p in files])
    return tcs.mean(), tcs.std(ddof=1) / np.sqrt(len(tcs))


def curves_path(L, lo, hi):
    d = os.path.join(ROOT, "vortexCNN", "results", f"L{L}")
    return os.path.join(d, f"curves_tlow{lo}_thigh{hi}_seed0.npz")


def ml_tc(L, lo, hi):
    p = curves_path(L, lo, hi)
    if not os.path.exists(p):
        return np.nan, np.nan
    c = np.load(p)
    return float(c["test_tc_cnn"]), float(c["test_tc_mlp"])


def fit(sizes, tcs, errs=None):
    """T_c = T_BKT + a x,  x = 1/(ln L)^2.  가중 최소제곱 (errs 있으면)."""
    x = 1 / np.log(sizes) ** 2
    ok = np.isfinite(tcs)
    x, y = x[ok], tcs[ok]
    w = None if errs is None else 1 / np.maximum(errs[ok], 1e-4)
    if len(x) > 2:
        (a, t_inf), cov = np.polyfit(x, y, 1, w=w, cov="unscaled" if w is not None else True)
    else:
        (a, t_inf), cov = np.polyfit(x, y, 1, w=w), np.full((2, 2), np.nan)
    return t_inf, np.sqrt(cov[1, 1]), a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--exclude-fit", type=int, nargs="*", default=[], help="피팅에서 뺄 L")
    args = p.parse_args()
    sizes = np.array(args.sizes)
    fit_mask = ~np.isin(sizes, args.exclude_fit)

    table = {}
    hel = np.array([helicity_tc(L) for L in sizes])
    table["helicity Υ"] = (hel[:, 0], hel[:, 1])
    for lo, hi in WINDOWS:
        v = np.array([ml_tc(L, lo, hi) for L in sizes])
        table[f"CNN {lo}/{hi}"] = (v[:, 0], None)
        table[f"MLP {lo}/{hi}"] = (v[:, 1], None)

    print("L        : " + "  ".join(f"{L:>7d}" for L in sizes) + "   |  L=∞ 외삽 (기울기 a)")
    results = {}
    for name, (tc, err) in table.items():
        e = None if err is None else err[fit_mask]
        t_inf, t_err, a = fit(sizes[fit_mask], tc[fit_mask], e)
        results[name] = (tc, err, t_inf, t_err, a)
        cells = "  ".join(f"{t:7.4f}" if np.isfinite(t) else "      -" for t in tc)
        print(f"{name:9s}: {cells}   |  {t_inf:.4f} ± {t_err:.4f}  (a={a:+.3f})")
    print(f"참고: 문헌 T_BKT = {T_BKT}" + (f"   (피팅 제외 L: {args.exclude_fit})" if args.exclude_fit else ""))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = 1 / np.log(sizes) ** 2
    xline = np.linspace(0, xs.max() * 1.05, 50)
    colors = {"0.7/1.1": "C0", "0.6/1.2": "C1", "0.5/1.3": "C2"}
    for name, (tc, err, t_inf, t_err, a) in results.items():
        if name.startswith("helicity"):
            ax.errorbar(xs, tc, yerr=err, fmt="ko", ms=6, capsize=3, label=f"Υ = 2T/π  → {t_inf:.3f}")
            ax.plot(xline, t_inf + a * xline, "k-", lw=1)
        elif name.startswith("CNN"):
            w = name.split()[1]
            ax.plot(xs, tc, "o", color=colors[w], label=f"CNN window {w}  → {t_inf:.3f}")
            ax.plot(xline, t_inf + a * xline, "-", color=colors[w], lw=1)
    ax.axhline(T_BKT, color="gray", ls=":", label=f"T_BKT (literature) = {T_BKT}")
    for x, L in zip(xs, sizes):
        ax.annotate(f"L={L}", (x, ax.get_ylim()[0]), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, color="gray")
    ax.set_xlabel("1 / (ln L)²     (x = 0  ->  L = ∞)")
    ax.set_ylabel("T_c(L)")
    ax.set_xlim(0, xs.max() * 1.08)
    ax.set_title("Finite-size extrapolation: T_c(L) = T_BKT + a/(ln L)²")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(ROOT, "vortexCNN", "results", "extrapolation.png")
    fig.savefig(png, dpi=150)
    print(f"저장: {png}")


if __name__ == "__main__":
    main()
