import numpy as np
from pathlib import Path

# 데이터 불러오기
data = np.load(r"C:\Users\hdh05\Downloads\xy_dataset_1.npz")

temperatures = data["temperatures"]
spin_configs = data["spin_configs"]

# 결과 저장 폴더
output_dir = Path(r"C:\Users\hdh05\Downloads\vortex_maps")
output_dir.mkdir(exist_ok=True)

def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_vortex_map(snapshot):
    Sx = snapshot[0]
    Sy = snapshot[1]

    theta = np.arctan2(Sy, Sx)

    a = theta
    b = np.roll(theta, -1, axis=1)
    c = np.roll(b, -1, axis=0)
    d = np.roll(theta, -1, axis=0)

    winding = (
        wrap(b - a)
        + wrap(c - b)
        + wrap(d - c)
        + wrap(a - d)
    ) / (2 * np.pi)

    return np.rint(winding).astype(np.int8)

# 전체 볼텍스 맵 배열
all_vortex_maps = np.empty(
    (len(temperatures), 256, 256),
    dtype=np.int8
)

# 전체 3660개 스냅샷 분석
for i in range(len(temperatures)):
    all_vortex_maps[i] = calculate_vortex_map(spin_configs[i])

    if (i + 1) % 60 == 0:
        print(f"완료: {i + 1}/{len(temperatures)}")

# 계산된 볼텍스 맵을 하나의 파일에 저장
np.savez_compressed(
    output_dir / "all_vortex_maps.npz",
    vortex_maps=all_vortex_maps,
    temperatures=temperatures
)

print("모든 볼텍스 맵 생성 및 저장 완료!")
print("저장 위치:", output_dir)