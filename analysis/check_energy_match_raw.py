import os
import numpy as np

L = 64

DATA_DIR = "../2dimension_spinAlgorithm/data/energy_match"

FILES = [
    os.path.join(DATA_DIR, f"energy_raw_{i}.npz")
    for i in range(1, 5)
]

# coarse energy window
E_LOW = -0.6966
E_HIGH = -0.6766

# fine matching bin width
BIN_WIDTH = 0.001

TARGET_TEMPS = [1.60, 1.70]


# --------------------------------------------------
# 데이터 합치기
# --------------------------------------------------
all_T = []
all_E = []

for path in FILES:
    data = np.load(path)

    all_T.append(data["temperatures"])
    all_E.append(data["energies"])

all_T = np.concatenate(all_T)
all_E = np.concatenate(all_E)

# energy density
all_e = all_E / (L * L)


print("=" * 70)
print("Raw energy-match dataset 확인")
print("=" * 70)

print(f"전체 sample 수: {len(all_T)}")

for T in TARGET_TEMPS:
    mask = np.isclose(all_T, T, atol=1e-6)
    print(f"T={T:.2f}: {mask.sum()} samples")


# --------------------------------------------------
# coarse window 안에 몇 개 있는지
# --------------------------------------------------
print()
print(
    f"Coarse energy window: "
    f"[{E_LOW:.4f}, {E_HIGH:.4f})"
)

for T in TARGET_TEMPS:

    mask = (
        np.isclose(all_T, T, atol=1e-6)
        & (all_e >= E_LOW)
        & (all_e < E_HIGH)
    )

    n = mask.sum()

    print(
        f"T={T:.2f}: "
        f"{n}/1200 "
        f"({100*n/1200:.2f}%)"
    )


# --------------------------------------------------
# fine energy bin 별 count
# --------------------------------------------------
n_bins = int(round((E_HIGH - E_LOW) / BIN_WIDTH))
edges = np.linspace(E_LOW, E_HIGH, n_bins + 1)

print()
print("=" * 70)
print(f"Fine bin 분석: Δe = {BIN_WIDTH}")
print("=" * 70)

total_matched_per_T = 0

for i in range(n_bins):

    low = edges[i]
    high = edges[i + 1]

    counts = []

    for T in TARGET_TEMPS:

        mask = (
            np.isclose(all_T, T, atol=1e-6)
            & (all_e >= low)
            & (all_e < high)
        )

        counts.append(int(mask.sum()))

    # 이 bin에서 두 온도에 공통으로 사용할 수 있는 수
    matched = min(counts)

    total_matched_per_T += matched

    print(
        f"[{low:.4f}, {high:.4f})  "
        f"T=1.60: {counts[0]:3d}   "
        f"T=1.70: {counts[1]:3d}   "
        f"match 가능: {matched:3d}/T"
    )


print()
print("=" * 70)
print("최종 예상")
print("=" * 70)

print(
    f"Energy histogram을 동일하게 맞출 경우 "
    f"온도당 사용 가능한 sample ≈ {total_matched_per_T}"
)

print(
    f"전체 matched dataset 크기 ≈ "
    f"{2 * total_matched_per_T}"
)
