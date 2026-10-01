"""
전체 파이프라인: 저장된 XY 모델 데이터 로드 -> CNN 학습 -> 평가

data/generate_data.py 가 저장한 data/xy_dataset_<번호>.npz 들을 불러와서
cnn/dataset.py 로 감싸고 cnn/model.py 로 학습합니다.

데이터 분할:
    - train     : validation 파일을 뺀 나머지 xy_dataset_<번호>.npz (69개 온도 전부)
    - validation: xy_dataset_<번호>.npz 중 하나를 통째로 사용 (기본: 가장 마지막 번호).
                  학습 파일과 독립적으로 시뮬레이션된 스핀 배치임.
    - test      : data/xy_testset.npz ('python data/generate_data.py --test'로 생성).
                  학습 온도 그리드(0.025 간격) 사이의 온도라 train/val과 온도가 겹치지 않음.

실행:
    python CNN/train.py            # 가장 마지막 번호 파일을 validation으로
    python CNN/train.py --val 1    # 1번 파일을 validation으로
    python CNN/train.py --quick    # 파일 없이 소규모 데이터를 즉석 생성해서 동작만 빠르게 확인
"""

import argparse
import copy
import json
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
from generate_data import dataset_path, existing_dataset_indices, TEST_PATH  # noqa: E402

from dataset import XYSpinDataset, load_dataset_npz  # noqa: E402
from model import XYTemperatureCNN  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)


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


def load_test_file():
    """test 전용 파일(학습 그리드 사이 온도)을 불러옴."""
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"test 데이터 파일이 없습니다: {TEST_PATH}\n"
            f"'python data/generate_data.py --test'를 먼저 실행하세요."
        )
    test_samples = load_dataset_npz(TEST_PATH)
    print(f"test 파일: {os.path.basename(TEST_PATH)} ({len(test_samples)}개 샘플)")
    return test_samples


def main(L=64, n_epochs=30, batch_size=32, lr=1e-3,
         quick_test=False, val_index=None, seed=42, patience=7):

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---------- 1) 데이터 준비 ----------
    if quick_test:
        # 빠른 동작 확인용: 파일 없이 즉석에서 축소 규모로 생성
        # (학습용/validation용/test용을 서로 다른 seed로 따로 시뮬레이션,
        #  test는 학습 온도 사이의 온도에서)
        temperatures = build_temperature_grid()[::4]
        test_temperatures = [1.55, 1.05, 0.55]
        print(f"\n[quick_test 모드] 온도 그리드 ({len(temperatures)}개): {temperatures}")
        t0 = time.perf_counter()
        train_samples, val_samples, test_samples = [
            generate_dataset(
                L=L, J=1.0, temperatures=temps,
                n_burnin=100, n_samples_per_T=4, sweeps_between=10,
                seed=s, verbose=False,
            )
            for temps, s in ((temperatures, 1), (temperatures, 2), (test_temperatures, 0))
        ]
        print(f"데이터 생성 완료: 학습용 {len(train_samples)}개 + validation용 {len(val_samples)}개 "
              f"+ test용 {len(test_samples)}개, {time.perf_counter()-t0:.1f}초 소요")
    else:
        # 실전 모드: 미리 저장해둔 데이터 파일들을 불러옴 (재시뮬레이션 안 함)
        train_samples, val_samples = load_train_val_files(val_index)
        test_samples = load_test_file()

    test_temps = sorted(set(round(s["T"], 4) for s in test_samples))
    print(f"\ntrain={len(train_samples)}  val={len(val_samples)}  "
          f"test={len(test_samples)} (test 온도: {test_temps})")

    train_loader = DataLoader(XYSpinDataset(train_samples), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(XYSpinDataset(val_samples), batch_size=batch_size)
    test_loader = DataLoader(XYSpinDataset(test_samples), batch_size=batch_size)

    # ---------- 2) 모델 / 옵티마이저 ----------
    model = XYTemperatureCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ---------- 3) 학습 루프 ----------
    print("\n=== 학습 시작 ===")
    history = {"train_mse": [], "val_mse": [], "val_mae": []}
    best_val_mae, best_epoch, best_state = float("inf"), 0, None
    epochs_without_improvement = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae, best_epoch = val_mae, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"  epoch {epoch:3d}/{n_epochs}  "
              f"train_MSE={train_loss:.4f}  val_MSE={val_mse:.4f}  val_MAE={val_mae:.4f}"
              f"{'  * BEST' if is_best else ''}")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping: validation 성능이 {patience} epoch 동안 개선되지 않음")
            break

    model.load_state_dict(best_state)
    print(f"\nbest epoch = {best_epoch} (val_MAE={best_val_mae:.4f}) 가중치로 복원 후 평가")

    # ---------- 4) 최종 테스트 평가 ----------
    test_mse, test_mae, test_pred, test_true = evaluate(model, test_loader, criterion, device)
    print(f"\n=== 최종 테스트 결과 ===")
    print(f"test MSE={test_mse:.4f}  test MAE={test_mae:.4f}")

    # 온도별 오차 확인 (T_BKT 근방에서 오차가 커지는지 확인하는 용도)
    temperature_mae = {}
    print("\n온도별 예측 예시 (실제 T -> 평균 예측 T):")
    for t in sorted(set(test_true.round(4))):
        mask = np.isclose(test_true, t, atol=1e-4)
        mae_t = float(np.mean(np.abs(test_pred[mask] - test_true[mask])))
        temperature_mae[f"{t:.4f}"] = mae_t
        print(f"  T={t:.4f}  ->  예측 평균={test_pred[mask].mean():.3f}  "
              f"MAE={mae_t:.4f}  (샘플 {mask.sum()}개)")

    os.makedirs(RESULT_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(RESULT_DIR, "best_model.pt"))
    np.savez(
        os.path.join(RESULT_DIR, "history.npz"),
        train_mse=np.array(history["train_mse"]),
        val_mse=np.array(history["val_mse"]),
        val_mae=np.array(history["val_mae"]),
    )
    result = {
        "seed": seed,
        "patience": patience,
        "n_epochs_run": len(history["train_mse"]),
        "best_epoch": best_epoch,
        "best_val_mae": float(best_val_mae),
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "temperature_mae": temperature_mae,
    }
    with open(os.path.join(RESULT_DIR, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장 완료: {RESULT_DIR}")

    return model, (test_pred, test_true)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="파일 없이 소규모 데이터를 즉석 생성해서 동작만 빠르게 확인")
    parser.add_argument("--val", type=int, default=None,
                        help="validation으로 쓸 데이터 파일 번호 (기본: 가장 마지막 번호)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--patience", type=int, default=7,
                        help="early stopping patience (epochs)")
    args = parser.parse_args()

    if args.quick:
        main(quick_test=True, n_epochs=15, seed=args.seed, patience=args.patience)
    else:
        main(val_index=args.val, seed=args.seed, patience=args.patience)
