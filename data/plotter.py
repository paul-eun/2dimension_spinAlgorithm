import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TBKT = 0.893


def load_all_npz():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "xy_dataset_*.npz")))
    testset = os.path.join(DATA_DIR, "xy_testset.npz")
    if os.path.exists(testset):
        paths.append(testset)

    if not paths:
        raise FileNotFoundError(
            f"No xy_dataset_*.npz or xy_testset.npz found in {DATA_DIR}.\n"
            f"Run 'python data/generate_data.py' first."
        )

    by_T = {}
    for path in paths:
        with np.load(path) as data:
            temps = data["temperatures"]
            mags = data["magnetizations"]
            n_vortex = data["n_vortex"]
            n_antivortex = data["n_antivortex"]

        for i in range(len(temps)):
            T = round(float(temps[i]), 4)
            entry = by_T.setdefault(T, {"mag": [], "vortex": []})
            entry["mag"].append(float(mags[i]))
            entry["vortex"].append(float(n_vortex[i] + n_antivortex[i]))

        print(f"Loaded: {os.path.basename(path)} ({len(temps)} samples)")

    return by_T


def summarize(by_T):
    Ts = sorted(by_T.keys())
    n_samples = np.array([len(by_T[T]["mag"]) for T in Ts])

    mag_mean = np.array([np.mean(by_T[T]["mag"]) for T in Ts])
    mag_std = np.array([np.std(by_T[T]["mag"]) for T in Ts])
    mag_sem = mag_std / np.sqrt(n_samples)

    vortex_mean = np.array([np.mean(by_T[T]["vortex"]) for T in Ts])
    vortex_std = np.array([np.std(by_T[T]["vortex"]) for T in Ts])
    vortex_sem = vortex_std / np.sqrt(n_samples)

    return (np.array(Ts), mag_mean, mag_std, mag_sem,
            vortex_mean, vortex_std, vortex_sem, n_samples)


def plot(Ts, mag_mean, mag_std, mag_sem, vortex_mean, vortex_std, vortex_sem,
         out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    panels = [
        (axes[0, 0], mag_mean, mag_std, "tab:blue",
         "|M| (mean ± STD)", "Magnetization |M| vs Temperature (STD)"),
        (axes[0, 1], vortex_mean, vortex_std, "tab:orange",
         "vortex+antivortex (mean ± STD)", "Vortex Density vs Temperature (STD)"),
        (axes[1, 0], mag_mean, mag_sem, "tab:blue",
         "|M| (mean ± SEM)", "Magnetization |M| vs Temperature (SEM)"),
        (axes[1, 1], vortex_mean, vortex_sem, "tab:orange",
         "vortex+antivortex (mean ± SEM)", "Vortex Density vs Temperature (SEM)"),
    ]
    for ax, mean, err, color, ylabel, title in panels:
        ax.errorbar(Ts, mean, yerr=err, fmt="o-", ms=3, lw=1,
                    capsize=2, color=color, ecolor=color, alpha=0.8)
        ax.axvline(TBKT, color="red", ls="--", lw=1, label=f"T_BKT ~ {TBKT}")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[1, 0].set_xlabel("T")
    axes[1, 1].set_xlabel("T")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


def main():
    by_T = load_all_npz()
    (Ts, mag_mean, mag_std, mag_sem,
     vortex_mean, vortex_std, vortex_sem, n_samples) = summarize(by_T)

    print(f"\nNumber of temperature points: {len(Ts)} (range {Ts.min():.4f} ~ {Ts.max():.4f})")
    print(f"{'T':>8s}  {'|M| mean':>10s}  {'|M| std':>10s}  {'|M| sem':>10s}  "
          f"{'vortex mean':>12s}  {'vortex std':>10s}  {'vortex sem':>10s}  {'n':>5s}")
    for i in range(len(Ts)):
        print(f"{Ts[i]:8.4f}  {mag_mean[i]:10.4f}  {mag_std[i]:10.4f}  "
              f"{mag_sem[i]:10.4f}  {vortex_mean[i]:12.2f}  {vortex_std[i]:10.2f}  "
              f"{vortex_sem[i]:10.2f}  {n_samples[i]:5d}")

    out_path = os.path.join(DATA_DIR, "physics_check.png")
    plot(Ts, mag_mean, mag_std, mag_sem, vortex_mean, vortex_std, vortex_sem,
         out_path)


if __name__ == "__main__":
    main()
