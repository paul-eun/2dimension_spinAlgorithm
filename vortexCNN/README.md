# vortexCNN

2D XY 모델의 **vortex 지도**(plaquette마다 +1 vortex / −1 antivortex / 0)를 입력으로 받는 CNN으로
BKT 전이온도를 추정하고, 같은 데이터에서 계산한 **helicity modulus Υ**와 비교하는 실험.
격자 크기 L = 32, 64, 128, 256 에서 각각 T_c(L)을 구한 뒤 `T_c(L) = T_BKT + a/(ln L)²` 로 L → ∞ 외삽.

## 결과 요약

| | L=32 | L=64 | L=128 | L=256 | L=∞ 외삽 |
|---|---|---|---|---|---|
| **helicity Υ = 2T/π** | 0.920 | 0.916 | 0.914 | 0.902 | **0.894 ± 0.004** (문헌 0.893) |
| CNN, 학습 띠 0.7/1.1 | 0.970 | 0.968 | 0.963 | 0.954 | 0.948 |
| CNN, 학습 띠 0.6/1.2 | 0.991 | 0.998 | 0.993 | 0.990 | 0.992 |
| CNN, 학습 띠 0.5/1.3 | 1.043 | 1.034 | 1.043 | 1.055 | 1.055 |

- Υ는 L이 커질수록 내려가서 문헌값 T_BKT ≈ 0.893 으로 외삽됨.
- 양 끝 학습 CNN의 T_c는 L보다 **학습 띠**에 따라 정해지고, 외삽해도 0.893으로 모이지 않음.
  (vortex 밀도 등 국소적인 양이 T_BKT에서 매끄럽게 변하므로, 0/1 경계를 어디에 둘지를 띠가 정함)
- L=256 confusion 방식(T′ 18개)은 정확도가 모든 T′에서 0.97 이상이라 W 모양이 거의 나타나지 않음.
- 그래프: `results/extrapolation.png`, L별 곡선: `results/L<L>/phase_curve_*.png`, Υ(T): `data/helicity/L<L>/helicity_vs_cnn.png`
- 주의: CNN 값은 seed 1개. L=256의 띠 0.7/1.1, 0.5/1.3 은 3 epoch, 나머지는 10 epoch.

## 폴더 구조

```
vortexCNN/
├── model.py         XYPhaseCNN (conv 16/32/64, 주기 경계 padding, GAP), VortexDensityMLP (비교용)
├── dataset.py       vortex 지도 불러오기 (data/vortex_maps/L<L>/vortex_maps_<번호>.npz)
├── train.py         양 끝 학습 (Carrasquilla–Melko): T ≤ t_low -> 0, T ≥ t_high -> 1, P = 0.5 인 온도 = T_c
├── confusion.py     confusion 방식 (van Nieuwenburg et al.): 경계 후보 T′마다 학습, 정확도의 W 모양
├── extrapolate.py   L별 T_c(L) 표 + L -> ∞ 외삽 그래프
├── data/
│   ├── helicity/L<L>/       샘플별 E_x, E_y, I_x, I_y (Υ 계산용)       [git 포함]
│   └── vortex_maps/L<L>/    vortex 지도 (약 100 MB)                    [git 제외 -> 다시 만들기]
├── results/L<L>/    학습된 모델(.pt), 온도별 P 곡선(curves_*.npz), 그래프   [git 포함]
└── logs/            학습/분석 로그
```

데이터 분할 (모든 L 공통): train = 파일 1, 2, 3 / validation = 4 / test = 5.
데이터 설정 (모든 L 공통): 온도 61개 (0.30 ~ 1.80, 0.025 간격), 온도당 60개, Wolff,
온도마다 무작위 배치에서 독립 시작, burn-in 100, 샘플 간격 30 sweep.

## 바로 해볼 수 있는 것 (git에 포함된 파일만으로)

```bash
python3 vortexCNN/extrapolate.py                     # L별 표 + results/extrapolation.png
python3 vortexCNN/extrapolate.py --exclude-fit 32    # L=32 빼고 외삽
python3 data/helicity_modulus.py --analyze --out-dir vortexCNN/data/helicity/L256   # Υ(T) 표/그래프
```

## 데이터 다시 만들기 (학습을 다시 하려면 필요)

스핀 데이터와 vortex 지도는 용량 때문에 git에 없음. 저장소 루트에서:

```bash
# 1) 스핀 배치 생성 -> data/L<L>/xy_dataset_{1..5}.npz
#    L=32 약 1분, L=64 약 3분, L=128 약 40분 (6 worker 기준), L=256 은 매우 오래 걸림
python3 data/generate_parallel.py --L 32
python3 data/generate_parallel.py --L 64
python3 data/generate_parallel.py --L 128
python3 data/generate_parallel.py --L 256     # seed 1001~1005: generate_256_parallel.py 와 같은 데이터
#    (--workers 로 병렬 수 조절, 중간에 멈춰도 다시 실행하면 이어서 함)

# 2) vortex 지도 -> vortexCNN/data/vortex_maps/L<L>/
for L in 32 64 128 256; do
  python3 data/make_vortex_maps.py data/L$L/xy_dataset_{1,2,3,4,5}.npz --out-dir vortexCNN/data/vortex_maps/L$L
done

# 3) (선택) helicity modulus -> vortexCNN/data/helicity/L<L>/   (이미 git에 포함됨)
python3 data/helicity_modulus.py data/L32/xy_dataset_{1,2,3,4,5}.npz --out-dir vortexCNN/data/helicity/L32
```

## 학습

```bash
python3 vortexCNN/train.py --L 64 --t-low 0.7 --t-high 1.1 --epochs 10     # -> results/L64/
python3 vortexCNN/confusion.py --t-min 0.6 --t-max 1.3 --epochs 3          # L=256, -> results/L256/
```

- 필요 패키지: numpy, torch, matplotlib, numba (데이터 생성).
- Apple Silicon에서는 MPS를 자동으로 씀. RAM 8 GB에서 L=256 confusion은 `--batch 16` 권장.
