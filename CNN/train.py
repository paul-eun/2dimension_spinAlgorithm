"""
전체 파이프라인: XY 모델 시뮬레이션 데이터 생성 -> CNN 학습 -> 평가

simulation/xy_model_numba.py 를 import해서 데이터를 만들고,
그 결과를 cnn/dataset.py 로 감싸서 cnn/model.py 로 학습합니다.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# simulation/ 폴더를 import 경로에 추가 (quick_test 모드에서만 실제로 필요)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulation"))
from xy_model_numba import generate_dataset, build_temperature_grid  # noqa: E402

from dataset import XYSpinDataset, split_by_temperature, load_dataset_npz  # noqa: E402
from model import XYTemperatureCNN  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "xy_dataset.npz")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
    mse = total_loss / len(loader.dataset)
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    mae = np.mean(np.abs(pred - true))
    return mse, mae, pred, true


def main(L=64, n_burnin=800, n_samples_per_T=15, sweeps_between=40,
         n_epochs=30, batch_size=32, lr=1e-3, seed=42, quick_test=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---------- 1) 데이터 준비 ----------
    if quick_test:
        # 빠른 동작 확인용: 파일 없이 즉석에서 축소 규모로 생성
        temperatures = build_temperature_grid()[::4]
        n_burnin, n_samples_per_T, sweeps_between = 100, 4, 10
        print(f"\n[quick_test 모드] 온도 그리드 ({len(temperatures)}개): {temperatures}")
        t0 = time.perf_counter()
        samples = generate_dataset(
            L=L, J=1.0, temperatures=temperatures,
            n_burnin=n_burnin, n_samples_per_T=n_samples_per_T,
            sweeps_between=sweeps_between, seed=seed, verbose=True,
        )
        print(f"\n데이터 생성 완료: {len(samples)}개 샘플, "
              f"{time.perf_counter()-t0:.1f}초 소요")
    else:
        # 실전 모드: 미리 저장해둔 데이터 파일을 불러옴 (재시뮬레이션 안 함)
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"데이터 파일이 없습니다: {DATA_PATH}\n"
                f"먼저 'data/generate_data.py'를 실행해서 데이터를 생성하세요."
            )
        samples = load_dataset_npz(DATA_PATH)
        print(f"\n데이터 파일 로드 완료: {DATA_PATH} ({len(samples)}개 샘플)")

    # ---------- 2) train/val/test 분할 (온도 단위 분할) ----------
    all_temps = sorted(set(round(s["T"], 3) for s in samples))
    n_test_temps = max(1, len(all_temps) // 6)
    rng = np.random.default_rng(seed)
    test_temps = rng.choice(all_temps, size=n_test_temps, replace=False).tolist()

    train_samples, val_samples, test_samples = split_by_temperature(
        samples, test_temperatures=test_temps, val_fraction=0.15, seed=seed
    )
    print(f"\ntrain={len(train_samples)}  val={len(val_samples)}  "
          f"test={len(test_samples)} (test 온도: {sorted(test_temps)})")

    train_loader = DataLoader(XYSpinDataset(train_samples), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(XYSpinDataset(val_samples), batch_size=batch_size)
    test_loader = DataLoader(XYSpinDataset(test_samples), batch_size=batch_size)

    # ---------- 3) 모델 / 옵티마이저 ----------
    model = XYTemperatureCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ---------- 4) 학습 루프 ----------
    print("\n=== 학습 시작 ===")
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
        if epoch % max(1, n_epochs // 10) == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{n_epochs}  "
                  f"train_MSE={train_loss:.4f}  val_MSE={val_mse:.4f}  val_MAE={val_mae:.4f}")

    # ---------- 5) 최종 테스트 평가 ----------
    test_mse, test_mae, test_pred, test_true = evaluate(model, test_loader, criterion, device)
    print(f"\n=== 최종 테스트 결과 ===")
    print(f"test MSE={test_mse:.4f}  test MAE={test_mae:.4f}")

    # 온도별 오차 확인 (T_BKT 근방에서 오차가 커지는지 확인하는 용도)
    print("\n온도별 예측 예시 (실제 T -> 평균 예측 T):")
    for t in sorted(set(test_true.round(3))):
        mask = np.isclose(test_true, t, atol=1e-3)
        print(f"  T={t:.3f}  ->  예측 평균={test_pred[mask].mean():.3f}  "
              f"(샘플 {mask.sum()}개)")

    return model, (test_pred, test_true)


if __name__ == "__main__":
    # 기본은 빠른 동작 확인 모드 (실제 15,000개 규모는 quick_test=False로 별도 실행)
    main(quick_test=True, n_epochs=15)
