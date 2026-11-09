import glob
import numpy as np
import matplotlib.pyplot as plt

L = 64
N = L * L

# generated data files
files = sorted(glob.glob("../2dimension_spinAlgorithm/data/xy_dataset_*.npz"))

print("사용할 파일:")
for f in files:
    print(" ", f)

all_T = []
all_e = []

for path in files:
    with np.load(path) as data:
        T = data["temperatures"]
        E = data["energies"]

        # energy per spin
        e = E / N

        all_T.append(T)
        all_e.append(e)

T = np.concatenate(all_T)
e = np.concatenate(all_e)

print(f"\n총 샘플 수: {len(T)}")
print(f"T 범위: {T.min():.3f} ~ {T.max():.3f}")
print(f"Energy density 범위: {e.min():.4f} ~ {e.max():.4f}")

plt.figure(figsize=(10, 7))

plt.scatter(
    T,
    e,
    c=T,
    s=8,
    alpha=0.4,
    cmap="viridis"
)

plt.xlabel("Temperature T")
plt.ylabel("Energy density E / L^2")
plt.title("Energy distribution of 2D XY model")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("energy_scatter.png", dpi=200)