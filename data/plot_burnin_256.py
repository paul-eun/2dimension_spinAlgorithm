from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


BASE = Path("data/L256/burnin_test")
OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [
    0.800,
    0.875,
    0.900,
    0.925,
    1.000,
]

SEEDS = [
    8001,
    8002,
    8003,
]


for T in TEMPERATURES:

    # ------------------------------------------------
    # Energy
    # ------------------------------------------------

    plt.figure(figsize=(9, 6))

    for seed in SEEDS:

        path = BASE / f"burnin_T{T:.3f}_seed{seed}.npz"
        d = np.load(path)

        E = d["energy"]

        plt.plot(
            np.arange(1, len(E) + 1),
            E,
            alpha=0.45,
            label=f"seed {seed}",
        )

        # 20-sweep running average
        window = 20

        E_smooth = np.convolve(
            E,
            np.ones(window) / window,
            mode="valid",
        )

        plt.plot(
            np.arange(window, len(E) + 1),
            E_smooth,
            linewidth=2,
        )

    plt.axvline(
        50,
        linestyle="--",
        label="50 sweeps",
    )

    plt.axvline(
        100,
        linestyle="--",
        label="100 sweeps",
    )

    plt.axvline(
        200,
        linestyle="--",
        label="200 sweeps",
    )

    plt.xlabel("Sweep")
    plt.ylabel("Energy density E / L²")
    plt.title(f"L=256 Energy equilibration, T={T:.3f}")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT / f"energy_T{T:.3f}.png",
        dpi=200,
    )

    plt.close()

    # ------------------------------------------------
    # Magnetization
    # ------------------------------------------------

    plt.figure(figsize=(9, 6))

    for seed in SEEDS:

        path = BASE / f"burnin_T{T:.3f}_seed{seed}.npz"
        d = np.load(path)

        M = d["magnetization"]

        plt.plot(
            np.arange(1, len(M) + 1),
            M,
            alpha=0.45,
            label=f"seed {seed}",
        )

        window = 20

        M_smooth = np.convolve(
            M,
            np.ones(window) / window,
            mode="valid",
        )

        plt.plot(
            np.arange(window, len(M) + 1),
            M_smooth,
            linewidth=2,
        )

    plt.axvline(50, linestyle="--")
    plt.axvline(100, linestyle="--")
    plt.axvline(200, linestyle="--")

    plt.xlabel("Sweep")
    plt.ylabel("|M|")
    plt.title(f"L=256 Magnetization equilibration, T={T:.3f}")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT / f"magnetization_T{T:.3f}.png",
        dpi=200,
    )

    plt.close()


print(f"Saved figures to: {OUT}")
