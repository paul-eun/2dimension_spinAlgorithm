import os
import math
import numpy as np

# --------------------------------------------------
# 설정
# --------------------------------------------------
L = 64

# 지금은 설계용으로 기존 1~4번 파일만 사용
DATA_DIR = "../2dimension_spinAlgorithm/data"
FILES = [
    os.path.join(DATA_DIR, f"xy_dataset_{i}.npz")
    for i in range(1, 5)
]

# 조사할 energy density window 폭
WINDOW_WIDTHS = [0.005, 0.01, 0.02]

# 현재 파일은 온도당 60 sample
SAMPLES_PER_T_PER_FILE = 60
N_FILES = len(FILES)

# 나중에 energy-matched test에서
# 온도당 이 정도 확보하고 싶다고 가정
TARGET_SAMPLES_PER_T = 100

# 너무 희귀한 꼬리 부분을 후보로 잡지 않도록
# 현재 4개 파일에서 최소 이 개수 이상 존재하는 온도만 표시
MIN_COUNT = 5


# --------------------------------------------------
# 데이터 로드
# --------------------------------------------------
all_T = []
all_e = []

for path in FILES:
    data = np.load(path)

    T = data["temperatures"]
    E = data["energies"]

    # energy density
    e = E / (L * L)

    all_T.append(T)
    all_e.append(e)

all_T = np.concatenate(all_T)
all_e = np.concatenate(all_e)

temps = np.array(sorted(np.unique(np.round(all_T, 6))))

print(f"총 sample 수: {len(all_T)}")
print(f"온도 개수: {len(temps)}")
print(f"온도당 총 sample 수: {SAMPLES_PER_T_PER_FILE * N_FILES}")
print(f"energy density 범위: {all_e.min():.4f} ~ {all_e.max():.4f}")


# --------------------------------------------------
# energy window scan
# --------------------------------------------------
for width in WINDOW_WIDTHS:

    print("\n" + "=" * 80)
    print(f"Energy window width Δe = {width}")
    print("=" * 80)

    # window를 조금씩 이동시키면서 검색
    step = width / 4

    centers = np.arange(
        all_e.min() + width / 2,
        all_e.max() - width / 2,
        step
    )

    results = []

    for center in centers:

        e_low = center - width / 2
        e_high = center + width / 2

        counts = {}

        for T in temps:
            mask = (
                np.isclose(all_T, T, atol=1e-6)
                & (all_e >= e_low)
                & (all_e < e_high)
            )

            n = np.sum(mask)

            if n >= MIN_COUNT:
                counts[T] = int(n)

        # 최소 두 온도 이상이 같은 energy window에 있어야
        # energy matching 후보가 됨
        if len(counts) >= 2:

            min_count = min(counts.values())
            total_count = sum(counts.values())

            results.append(
                (
                    len(counts),
                    min_count,
                    total_count,
                    e_low,
                    e_high,
                    counts,
                )
            )

    # 많은 온도가 겹치는 window를 먼저,
    # 그 다음 최소 sample 수가 많은 순으로 정렬
    results.sort(
        key=lambda x: (x[0], x[1], x[2]),
        reverse=True
    )

    if len(results) == 0:
        print("조건을 만족하는 window가 없습니다.")
        continue

    # 상위 10개 후보 출력
    for rank, result in enumerate(results[:10], start=1):

        n_temps, min_count, total_count, e_low, e_high, counts = result

        print(
            f"\n[{rank}] "
            f"e = [{e_low:.4f}, {e_high:.4f})"
        )

        print(
            f"    겹치는 온도 수 = {n_temps}, "
            f"최소 count = {min_count}"
        )

        for T, count in sorted(counts.items()):

            # 현재 4개 파일, 온도당 240개 중 몇 %가 살아남았는가
            p = count / (SAMPLES_PER_T_PER_FILE * N_FILES)

            # 새 데이터에서도 같은 survival fraction이라고 가정했을 때
            # TARGET sample 확보에 필요한 파일 개수
            required_files = math.ceil(
                TARGET_SAMPLES_PER_T
                / (SAMPLES_PER_T_PER_FILE * p)
            )

            print(
                f"      T={T:.3f}: "
                f"{count:3d}/240 "
                f"({100*p:5.1f}%)"
                f"  -> 예상 필요 파일 ≈ {required_files}"
            )