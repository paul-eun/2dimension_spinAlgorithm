"""
전체 파이프라인: 저장된 XY 모델 데이터 로드 -> CNN 학습 -> 평가

data/generate_data.py 가 저장한 data/xy_dataset_<번호>.npz 들을 불러와서
cnn/dataset.py 로 감싸고 cnn/model.py 로 학습합니다.

데이터 분할:
    - validation: 파일 하나를 통째로 사용 (기본: 가장 마지막 번호).
                  학습 파일과 독립적으로 시뮬레이션된 스핀 배치임.
    - test      : 무작위로 고른 온도들의 샘플 (모든 파일에서).
                  이 온도들은 train/val 어디에도 들어가지 않음.
    - train     : validation 파일을 뺀 나머지 파일들의 (test 온도 외) 샘플

실행:
    python CNN/train.py            # 가장 마지막 번호 파일을 validation으로
    python CNN/train.py --val 1    # 1번 파일을 validation으로
    python CNN/train.py --quick    # 파일 없이 소규모 데이터를 즉석 생성해서 동작만 빠르게 확인
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# simulation/, data/ 폴더를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from xy_model_numba import generate_dataset, build_temperature_grid  # noqa: E402
from generate_data import dataset_path, existing_dataset_indices  # noqa: E402

from dataset import XYSpinDataset, split_by_temperature, load_dataset_npz  # noqa: E402
from model import XYTemperatureCNN  # noqa: E402


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


def load_train_val_files(val_index=None):
    """저장된 데이터 파일들을 (학습용 샘플, validation 샘플)로 나눠서 불러옴."""
    indices = existing_dataset_indices()
    if len(indices) < 2:
        raise FileNotFoundError(
            f"데이터 파일이 {len(indices)}개뿐입니다 (학습용 + validation용 최소 2개 필요).\n"
            f"'python data/generate_data.py'를 {2 - len(indices)}번 더 실행하세요."
        )
    if val_index is None:
        val_index = indices[-1]
    if val_index not in indices:
        raise ValueError(f"validation으로 지정한 {val_index}번 파일이 없습니다 (있는 번호: {indices})")

    train_samples = []
    for k in indices:
        if k != val_index:
            train_samples += load_dataset_npz(dataset_path(k))
    val_samples = load_dataset_npz(dataset_path(val_index))

    train_files = [k for k in indices if k != val_index]
    print(f"\n학습 파일: {train_files} ({len(train_samples)}개 샘플)")
    print(f"validation 파일: [{val_index}] ({len(val_samples)}개 샘플)")
    return train_samples, val_samples


def main(L=64, n_epochs=30, batch_size=32, lr=1e-3, seed=42,
         quick_test=False, val_index=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---------- 1) 데이터 준비 ----------
    if quick_test:
        # 빠른 동작 확인용: 파일 없이 즉석에서 축소 규모로 생성
        # (학습용과 validation용을 서로 다른 seed로 따로 시뮬레이션)
        temperatures = build_temperature_grid()[::4]
        print(f"\n[quick_test 모드] 온도 그리드 ({len(temperatures)}개): {temperatures}")
        t0 = time.perf_counter()
        train_pool, val_pool = [
            generate_dataset(
                L=L, J=1.0, temperatures=temperatures,
                n_burnin=100, n_samples_per_T=4, sweeps_between=10,
                seed=s, verbose=False,
            )
            for s in (1, 2)
        ]
        print(f"데이터 생성 완료: 학습용 {len(train_pool)}개 + validation용 {len(val_pool)}개, "
              f"{time.perf_counter()-t0:.1f}초 소요")
    else:
        # 실전 모드: 미리 저장해둔 데이터 파일들을 불러옴 (재시뮬레이션 안 함)
        train_pool, val_pool = load_train_val_files(val_index)

    # ---------- 2) test 온도 분리 (온도 단위) ----------
    all_temps = sorted(set(round(s["T"], 3) for s in train_pool))
    n_test_temps = max(1, len(all_temps) // 6)
    rng = np.random.default_rng(seed)
    test_temps = rng.choice(all_temps, size=n_test_temps, replace=False).tolist()

    # test 온도는 train/val 양쪽에서 모두 빼고, 두 쪽의 해당 샘플을 모두 test로 사용
    train_samples, test_from_train = split_by_temperature(train_pool, test_temps)
    val_samples, test_from_val = split_by_temperature(val_pool, test_temps)
    test_samples = test_from_train + test_from_val
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="파일 없이 소규모 데이터를 즉석 생성해서 동작만 빠르게 확인")
    parser.add_argument("--val", type=int, default=None,
                        help="validation으로 쓸 데이터 파일 번호 (기본: 가장 마지막 번호)")
    args = parser.parse_args()

    if args.quick:
        main(quick_test=True, n_epochs=15)
    else:
        main(val_index=args.val)
