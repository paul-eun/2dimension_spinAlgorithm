# XY BKT Shared Dataset

2D XY model 프로젝트에서 사용한 데이터셋 정리

데이터는 크게 두 종류로 나뉜다.

1. **Temperature regression data**  
   CNN / MLP 학습, 검증, 테스트에 사용한 기본 데이터
2. **Energy-control data**  
   모델의 온도 예측이 energy 정보에 얼마나 의존하는지 확인하기 위한 비교용 데이터

---

## 1. 폴더 구조

```text
XY_BKT_shared_data/
├── temperature_data/
│   ├── xy_dataset_1.npz
│   ├── xy_dataset_2.npz
│   ├── xy_dataset_3.npz
│   ├── xy_dataset_4.npz
│   └── xy_testset.npz
│
└── energy_control/
    ├── normal_control_test.npz
    └── energy_matched_test.npz
```

---

# 2. Temperature regression data

위치:

```text
temperature_data/
```

공통 시뮬레이션 조건:

- Model: 2D XY model
- Lattice size: `L = 64`
- Coupling constant: `J = 1.0`
- Algorithm: Wolff cluster algorithm
- Periodic boundary condition 사용
- Spin configuration shape: `(2, 64, 64)`
  - channel 0: `cos(theta)`
  - channel 1: `sin(theta)`

Wolff algorithm에서 **1 sweep**은 cluster update 1번이 아니라,

```text
누적 flipped spin 수 >= L^2
```

가 될 때까지 여러 번 cluster update를 반복한 것을 의미한다.

---

## 2.1 `xy_dataset_1.npz`

**용도:** Training data

- Temperature range: `T = 0.30 ~ 1.80`
- Temperature interval: `0.025`
- Number of temperatures: `61`
- Samples per temperature: `60`
- Burn-in: `50 sweeps`
- Sweeps between samples: `8`
- Total samples: `3660`
- File size: 약 `104 MB`

---

## 2.2 `xy_dataset_2.npz`

**용도:** Training data

설정은 `xy_dataset_1.npz`와 동일하다.

- Temperature range: `T = 0.30 ~ 1.80`
- Temperature interval: `0.025`
- Number of temperatures: `61`
- Samples per temperature: `60`
- Burn-in: `50 sweeps`
- Sweeps between samples: `8`
- Total samples: `3660`
- File size: 약 `104 MB`

---

## 2.3 `xy_dataset_3.npz`

**용도:** Training data

설정은 `xy_dataset_1.npz`와 동일하다.

- Temperature range: `T = 0.30 ~ 1.80`
- Temperature interval: `0.025`
- Number of temperatures: `61`
- Samples per temperature: `60`
- Burn-in: `50 sweeps`
- Sweeps between samples: `8`
- Total samples: `3660`
- File size: 약 `104 MB`

### Training data 전체

`xy_dataset_1~3.npz`를 모두 합치면:

```text
3660 × 3 = 10,980 samples
```

---

## 2.4 `xy_dataset_4.npz`

**용도:** Validation data

Training data와 동일한 온도 grid와 sampling 조건을 사용한 별도의 validation용 데이터이다.

- Temperature range: `T = 0.30 ~ 1.80`
- Temperature interval: `0.025`
- Number of temperatures: `61`
- Samples per temperature: `60`
- Burn-in: `50 sweeps`
- Sweeps between samples: `8`
- Total samples: `3660`
- File size: 약 `104 MB`

---

## 2.5 `xy_testset.npz`

**용도:** Unseen-temperature test data

Training / validation temperature와 겹치지 않는 중간 온도에서 생성한 test set.

사용한 온도:

```text
0.3625
0.5125
0.6625
0.8125
0.9625
1.1125
1.2625
1.4125
1.5625
1.7125
```

설정:

- Number of temperatures: `10`
- Samples per temperature: `120`
- Burn-in: `200 sweeps`
- Sweeps between samples: `8`
- Total samples: `1200`
- File size: 약 `34 MB`

---

# 3. Temperature regression용 NPZ 내부 구조

`xy_dataset_1~4.npz`, `xy_testset.npz`에는 다음 항목들이 저장되어 있다.

```python
data = np.load("xy_dataset_1.npz")

data["spin_configs"]
data["temperatures"]
data["energies"]
data["magnetizations"]
data["n_vortex"]
data["n_antivortex"]
```

각 항목의 의미:

### `spin_configs`

```text
shape = (N, 2, 64, 64)
```

CNN 입력으로 사용하는 spin configuration이다.

- `spin_configs[:, 0, :, :] = cos(theta)`
- `spin_configs[:, 1, :, :] = sin(theta)`

### `temperatures`

각 configuration이 생성된 온도이다.

### `energies`

각 configuration의 total energy이다.

Energy density가 필요한 경우:

```python
L = 64
energy_density = data["energies"] / (L * L)
```

### `magnetizations`

각 configuration의 magnetization magnitude.

### `n_vortex`

Plaquette winding number로 계산한 vortex 개수.

### `n_antivortex`

Plaquette winding number로 계산한 antivortex 개수.

---

# 4. Energy-control data

위치:

```text
energy_control/
```

이 데이터는 CNN이 온도를 예측할 때 energy-related information에 얼마나 의존하는지 확인하기 위해 만든 control dataset이다.

두 파일 모두 다음 두 온도를 포함한다.

```text
T = 1.60
T = 1.70
```

또한 두 파일 모두 최종 sample 수를 동일하게 맞췄다.

```text
T = 1.60 : 127 samples
T = 1.70 : 127 samples
Total    : 254 samples
```

---

## 4.1 `normal_control_test.npz`

**용도:** Energy matching을 하지 않은 비교용 control data

같은 raw simulation pool에서 `T=1.60`과 `T=1.70`의 sample을 각각 127개씩 무작위로 선택했다.

즉, 두 온도가 원래 가지고 있던 energy distribution 차이는 그대로 남아 있다.

이 파일의 목적은 아래의 질문을 확인하는 것이다.

```text
Energy를 따로 통제하지 않은 상태에서
모델이 T=1.60과 T=1.70을 얼마나 잘 구분하는가?
```

설정:

- `T = 1.60`: 127 samples
- `T = 1.70`: 127 samples
- Total: `254 samples`
- File size: 약 `7.3 MB`

---

## 4.2 `energy_matched_test.npz`

**용도:** 두 온도의 energy distribution을 거의 동일하게 맞춘 control data

Energy density를

```text
e = E / L^2
```

로 정의한 뒤, 두 온도의 energy distribution이 겹치는 영역에서 sample을 선택했다

사용한 energy density 범위:

```text
-0.6966 <= e < -0.6766
```

Fine bin width:

```text
Δe = 0.001
```

각 bin에서 `T=1.60`과 `T=1.70`에 존재하는 sample 수를 비교하고,

```text
min(N_T=1.60, N_T=1.70)
```

만큼 양쪽에서 동일한 수를 선택.

예를 들어 한 energy bin에

```text
T=1.60 : 15 samples
T=1.70 :  9 samples
```

가 있다면,

```text
T=1.60 : 9 samples 선택
T=1.70 : 9 samples 선택
```

하는 방식임.

최종 결과:

- `T = 1.60`: 127 samples
- `T = 1.70`: 127 samples
- Total: `254 samples`
- File size: 약 `7.3 MB`

최종 energy density 통계:

```text
T = 1.60
mean(e) = -0.688322
std(e)  =  0.005344

T = 1.70
mean(e) = -0.688329
std(e)  =  0.005320
```

두 온도의 energy distribution이 거의 동일하도록 만든 데이터이다.

이 파일의 목적은 아래의 질문을 확인하기 위한 용도이다.

```text
두 온도의 energy 정보가 거의 같아도
모델이 T=1.60과 T=1.70을 구분할 수 있는가?
```

---

# 5. `normal_control_test.npz`와 `energy_matched_test.npz`의 차이

두 파일은 sample 수가 동일하다.

차이는 **sample을 선택하는 방법**이다.

### Normal control

```text
Raw data
   ↓
Random sampling
   ↓
normal_control_test.npz
```

Energy distribution 차이를 그대로 유지한다.

### Energy matched

```text
Raw data
   ↓
Energy density 계산
   ↓
Energy binning
   ↓
각 bin에서 두 온도의 sample 수를 동일하게 선택
   ↓
energy_matched_test.npz
```

Energy distribution 차이를 거의 제거하고 비교하면,

```text
모델의 온도 구분 능력이 energy 정보에 얼마나 의존하는가?
```

를 확인할 수 있을 것. (실제로 확인되었다)

---

# 6. 간단한 데이터 로딩 예시

## Temperature data

```python
import numpy as np

data = np.load("temperature_data/xy_dataset_1.npz")

spin = data["spin_configs"]
temperature = data["temperatures"]
energy = data["energies"]

print(spin.shape)
print(temperature.shape)
print(energy.shape)
```

## Energy-control data

```python
import numpy as np

normal = np.load("energy_control/normal_control_test.npz")
matched = np.load("energy_control/energy_matched_test.npz")

print(normal["spin_configs"].shape)
print(matched["spin_configs"].shape)
```

---

# 7. 파일별 용도 요약

| File | Purpose | Samples |
|---|---|---:|
| `xy_dataset_1.npz` | Training | 3660 |
| `xy_dataset_2.npz` | Training | 3660 |
| `xy_dataset_3.npz` | Training | 3660 |
| `xy_dataset_4.npz` | Validation | 3660 |
| `xy_testset.npz` | Unseen-temperature test | 1200 |
| `normal_control_test.npz` | Normal energy-control set | 254 |
| `energy_matched_test.npz` | Energy-matched control set | 254 |

---

# 8. 전체 파일 크기

대략적인 파일 크기:

```text
xy_dataset_1.npz             104 MB
xy_dataset_2.npz             104 MB
xy_dataset_3.npz             104 MB
xy_dataset_4.npz             104 MB
xy_testset.npz                34 MB
normal_control_test.npz      7.3 MB
energy_matched_test.npz      7.3 MB
```

전체 약:

```text
464 MB
```

---

# 9. 사용 시 참고

- 기본 CNN / MLP 학습은 `xy_dataset_1~4.npz`를 사용한다.
- 일반적인 test는 `xy_testset.npz`를 사용한다.
- `normal_control_test.npz`와 `energy_matched_test.npz`는 **학습용이 아니라 energy dependence 확인을 위한 control test data**이다.
- Vortex map 학습 등 새로운 task를 만들 경우에는 `spin_configs`를 입력으로 사용하고, 필요한 label은 각 configuration에서 별도로 계산하면 된다.
