import os
import numpy as np

L = 64

DATA_DIR = "../2dimension_spinAlgorithm/data/energy_match"

FILES = [
    os.path.join(DATA_DIR, f"energy_raw_{i}.npz")
    for i in range(1, 5)
]

T1 = 1.60
T2 = 1.70

E_LOW = -0.6966
E_HIGH = -0.6766
BIN_WIDTH = 0.001

RANDOM_SEED = 12345
rng = np.random.default_rng(RANDOM_SEED)


# --------------------------------------------------
# 데이터 로드
# --------------------------------------------------
keys = [
    "spin_configs",
    "temperatures",
    "energies",
    "magnetizations",
    "n_vortex",
    "n_antivortex",
]

combined = {key: [] for key in keys}

for path in FILES:
    data = np.load(path)

    for key in keys:
        combined[key].append(data[key])

for key in keys:
    combined[key] = np.concatenate(combined[key], axis=0)

T = combined["temperatures"]
e = combined["energies"] / (L * L)

print(f"전체 raw samples = {len(T)}")


# --------------------------------------------------
# Energy matching
# --------------------------------------------------
n_bins = int(round((E_HIGH - E_LOW) / BIN_WIDTH))
edges = np.linspace(E_LOW, E_HIGH, n_bins + 1)

matched_indices = []

print("\n=== Energy bin matching ===")

for i in range(n_bins):

    low = edges[i]
    high = edges[i + 1]

    idx1 = np.where(
        np.isclose(T, T1, atol=1e-6)
        & (e >= low)
        & (e < high)
    )[0]

    idx2 = np.where(
        np.isclose(T, T2, atol=1e-6)
        & (e >= low)
        & (e < high)
    )[0]

    n = min(len(idx1), len(idx2))

    if n == 0:
        continue

    sel1 = rng.choice(idx1, size=n, replace=False)
    sel2 = rng.choice(idx2, size=n, replace=False)

    matched_indices.extend(sel1)
    matched_indices.extend(sel2)

    print(
        f"[{low:.4f}, {high:.4f}) "
        f"-> {n} samples/T"
    )

matched_indices = np.array(matched_indices)

# 순서 섞기
rng.shuffle(matched_indices)


# --------------------------------------------------
# matched dataset 저장
# --------------------------------------------------
MATCHED_PATH = os.path.join(
    DATA_DIR,
    "energy_matched_test.npz"
)

np.savez_compressed(
    MATCHED_PATH,
    **{
        key: combined[key][matched_indices]
        for key in keys
    }
)

matched_T = T[matched_indices]
matched_e = e[matched_indices]

n1 = np.sum(np.isclose(matched_T, T1))
n2 = np.sum(np.isclose(matched_T, T2))

print("\n=== Energy-matched dataset ===")
print(f"T={T1:.2f}: {n1}")
print(f"T={T2:.2f}: {n2}")
print(f"total = {len(matched_indices)}")


# --------------------------------------------------
# Energy 통계 확인
# --------------------------------------------------
for temp in [T1, T2]:

    mask = np.isclose(matched_T, temp)

    values = matched_e[mask]

    print(
        f"T={temp:.2f}: "
        f"mean(e)={values.mean():.6f}, "
        f"std(e)={values.std():.6f}, "
        f"min={values.min():.6f}, "
        f"max={values.max():.6f}"
    )


# --------------------------------------------------
# 같은 크기의 normal control 생성
# --------------------------------------------------
N_PER_T = min(n1, n2)

normal_indices = []

for temp in [T1, T2]:

    idx = np.where(
        np.isclose(T, temp, atol=1e-6)
    )[0]

    selected = rng.choice(
        idx,
        size=N_PER_T,
        replace=False
    )

    normal_indices.extend(selected)

normal_indices = np.array(normal_indices)
rng.shuffle(normal_indices)

NORMAL_PATH = os.path.join(
    DATA_DIR,
    "normal_control_test.npz"
)

np.savez_compressed(
    NORMAL_PATH,
    **{
        key: combined[key][normal_indices]
        for key in keys
    }
)

print("\n=== Normal control dataset ===")
print(f"T={T1:.2f}: {N_PER_T}")
print(f"T={T2:.2f}: {N_PER_T}")
print(f"total = {2 * N_PER_T}")

print("\n저장 완료:")
print(MATCHED_PATH)
print(NORMAL_PATH)
