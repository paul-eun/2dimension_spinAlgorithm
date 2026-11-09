"""
CNN 비교용: 에너지 밀도 e = E / L^2 하나만 입력으로 쓰는 MLP로 온도 예측

data/generate_data.py 가 저장한 data/xy_dataset_<번호>.npz 들을 불러와서
mlp/dataset.py 로 감싸고 mlp/model.py 로 학습합니다.

CNN(CNN/train.py)과 비교가 공정하도록 다음을 모두 똑같이 맞춤:
    - train     : validation 파일을 뺀 나머지 xy_dataset_<번호>.npz (61개 온도 전부)
    - validation: xy_dataset_<번호>.npz 중 하나를 통째로 사용 (기본: 가장 마지막 번호)
    - test      : data/xy_testset.npz (학습 온도 그리드 사이의 온도)
    - 손실 함수 MSE, 옵티마이저 Adam(lr=1e-3), batch 32
입력만 "스핀 배치" -> "에너지 밀도"로 다름.

실행:
    python MLP/train.py            # 가장 마지막 번호 파일을 validation으로
    python MLP/train.py --val 1    # 1번 파일을 validation으로
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# data/ 폴더를 import 경로에 추가 (데이터 파일 번호 규칙을 generate_data.py와 공유)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from generate_data import dataset_path, existing_dataset_indices, TEST_PATH  # noqa: E402

from dataset import XYEnergyDataset, load_energy_npz  # noqa: E402
from model import XYEnergyMLP  # noqa: E402


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
    """저장된 데이터 파일들을 (학습용 e, T), (validation e, T)로 나눠서 불러옴."""
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

    train_files = [k for k in indices if k != val_index]
    loaded = [load_energy_npz(dataset_path(k)) for k in train_files]
    train_e = np.concatenate([e for e, _ in loaded])
    train_T = np.concatenate([T for _, T in loaded])
    val_e, val_T = load_energy_npz(dataset_path(val_index))

    print(f"\n학습 파일: {train_files} ({len(train_T)}개 샘플)")
    print(f"validation 파일: [{val_index}] ({len(val_T)}개 샘플)")
    return (train_e, train_T), (val_e, val_T)


def load_test_file():
    """test 전용 파일(학습 그리드 사이 온도)을 불러옴."""
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"test 데이터 파일이 없습니다: {TEST_PATH}\n"
            f"'python data/generate_data.py --test'를 먼저 실행하세요."
        )
    test_e, test_T = load_energy_npz(TEST_PATH)
    print(f"test 파일: {os.path.basename(TEST_PATH)} ({len(test_T)}개 샘플)")
    return test_e, test_T


def main(n_epochs=100, batch_size=32, lr=1e-3, seed=42, val_index=None):

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"seed: {seed}")

    # ---------- 1) 데이터 준비 ----------
    (tr_e, tr_T), (va_e, va_T) = load_train_val_files(val_index)
    te_e, te_T = load_test_file()

    test_temps = sorted(set(round(float(t), 4) for t in te_T))
    print(f"\ntrain={len(tr_T)}  val={len(va_T)}  "
          f"test={len(te_T)} (test 온도: {test_temps})")

    train_loader = DataLoader(XYEnergyDataset(tr_e, tr_T), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(XYEnergyDataset(va_e, va_T), batch_size=batch_size)
    test_loader = DataLoader(XYEnergyDataset(te_e, te_T), batch_size=batch_size)

    # ---------- 2) 모델 / 옵티마이저 ----------
    model = XYEnergyMLP()
    model.set_normalization(tr_e.mean(), tr_e.std())   # 학습 데이터 기준으로만 표준화
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ---------- 3) 학습 루프 (val이 가장 좋았던 epoch의 모델도 따로 보관) ----------
    print("\n=== 학습 시작 ===")
    best_val_mae, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
        if val_mae < best_val_mae:
            best_val_mae, best_epoch = val_mae, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % max(1, n_epochs // 10) == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{n_epochs}  "
                  f"train_MSE={train_loss:.4f}  val_MSE={val_mse:.4f}  val_MAE={val_mae:.4f}")

    # ---------- 4) 최종 테스트 평가 ----------
    _, last_mae, _, _ = evaluate(model, test_loader, criterion, device)
    model.load_state_dict(best_state)
    os.makedirs("MLP/checkpoints", exist_ok=True)

    checkpoint_path = f"MLP/checkpoints/mlp_best_seed{seed}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": int(best_epoch),
            "best_val_mae": float(best_val_mae),
            "seed": int(seed),
        },
        checkpoint_path
    )

    print(f"best model 저장: {checkpoint_path}")
    test_mse, test_mae, test_pred, test_true = evaluate(model, test_loader, criterion, device)
    print(f"\n=== 최종 테스트 결과 ===")
    print(
        f"best validation: epoch={best_epoch}, "
        f"val_MAE={best_val_mae:.4f}"
    )
    print(f"마지막 epoch 모델: test MAE={last_mae:.4f}")
    print(
        f"val 최고 epoch({best_epoch}) 모델: "
        f"test MSE={test_mse:.4f}  "
        f"test MAE={test_mae:.4f}"
    )

    print("\n온도별 예측 (val 최고 epoch 모델, 실제 T -> 평균 예측 T ± 표준편차):")
    for t in sorted(set(test_true.round(4))):
        mask = np.isclose(test_true, t, atol=1e-4)
        print(f"  T={t:.4f}  ->  {test_pred[mask].mean():.3f} ± {test_pred[mask].std():.3f}  "
              f"(샘플 {mask.sum()}개)")

    return model, (test_pred, test_true)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--val",
        type=int,
        default=None,
        help="validation으로 쓸 데이터 파일 번호 (기본: 가장 마지막 번호)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed"
    )

    args = parser.parse_args()

    main(
        n_epochs=args.epochs,
        seed=args.seed,
        val_index=args.val
    )
