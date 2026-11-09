import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

REPO = "../2dimension_spinAlgorithm"

sys.path.append(os.path.join(REPO, "CNN"))

from model import XYTemperatureCNN


# --------------------------------------------------
# 경로
# --------------------------------------------------
CHECKPOINT = os.path.join(
    REPO,
    "CNN",
    "checkpoints",
    "maxpool_best.pt"
)

DATA_DIR = os.path.join(
    REPO,
    "data",
    "energy_match"
)

NORMAL_PATH = os.path.join(
    DATA_DIR,
    "normal_control_test.npz"
)

MATCHED_PATH = os.path.join(
    DATA_DIR,
    "energy_matched_test.npz"
)

L = 64
BATCH_SIZE = 32


# --------------------------------------------------
# 모델 로드
# --------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"device: {device}")

model = XYTemperatureCNN().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"checkpoint: epoch={checkpoint['best_epoch']}, "
    f"val_MAE={checkpoint['best_val_mae']:.4f}"
)


# --------------------------------------------------
# 하나의 dataset 평가
# --------------------------------------------------
@torch.no_grad()
def evaluate_file(path, name):

    data = np.load(path)

    x = torch.tensor(
        data["spin_configs"],
        dtype=torch.float32
    )

    y = torch.tensor(
        data["temperatures"],
        dtype=torch.float32
    )

    energies = data["energies"] / (L * L)

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    preds = []
    truths = []

    for xb, yb in loader:

        xb = xb.to(device)

        pred = model(xb)

        preds.append(
            pred.cpu().numpy()
        )

        truths.append(
            yb.numpy()
        )

    preds = np.concatenate(preds)
    truths = np.concatenate(truths)

    mae = np.mean(
        np.abs(preds - truths)
    )

    mse = np.mean(
        (preds - truths) ** 2
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"samples = {len(truths)}")
    print(f"MSE = {mse:.6f}")
    print(f"MAE = {mae:.6f}")

    result = {}

    for T in sorted(np.unique(truths)):

        mask = np.isclose(
            truths,
            T,
            atol=1e-6
        )

        p = preds[mask]
        e = energies[mask]

        result[float(T)] = {
            "mean_pred": float(p.mean()),
            "std_pred": float(p.std()),
            "mean_energy": float(e.mean()),
            "std_energy": float(e.std()),
        }

        print(
            f"\nT={T:.2f}"
        )

        print(
            f"  prediction: "
            f"mean={p.mean():.5f}, "
            f"std={p.std():.5f}"
        )

        print(
            f"  energy density: "
            f"mean={e.mean():.6f}, "
            f"std={e.std():.6f}"
        )

    temps = sorted(result.keys())

    T_low = temps[0]
    T_high = temps[1]

    delta_pred = (
        result[T_high]["mean_pred"]
        - result[T_low]["mean_pred"]
    )

    delta_true = T_high - T_low

    separation_ratio = (
        delta_pred / delta_true
    )

    print()
    print(
        f"실제 온도 차이 ΔT_true = "
        f"{delta_true:.5f}"
    )

    print(
        f"평균 예측 차이 ΔT_pred = "
        f"{delta_pred:.5f}"
    )

    print(
        f"separation ratio "
        f"ΔT_pred / ΔT_true = "
        f"{separation_ratio:.3f}"
    )

    return {
        "mae": mae,
        "mse": mse,
        "delta_pred": delta_pred,
        "separation_ratio": separation_ratio,
    }


# --------------------------------------------------
# 비교
# --------------------------------------------------
normal = evaluate_file(
    NORMAL_PATH,
    "NORMAL CONTROL"
)

matched = evaluate_file(
    MATCHED_PATH,
    "ENERGY-MATCHED"
)


print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    f"Normal MAE          = "
    f"{normal['mae']:.6f}"
)

print(
    f"Energy-matched MAE  = "
    f"{matched['mae']:.6f}"
)

print()

print(
    f"Normal ΔT_pred       = "
    f"{normal['delta_pred']:.6f}"
)

print(
    f"Matched ΔT_pred      = "
    f"{matched['delta_pred']:.6f}"
)

print()

print(
    f"Normal separation ratio  = "
    f"{normal['separation_ratio']:.3f}"
)

print(
    f"Matched separation ratio = "
    f"{matched['separation_ratio']:.3f}"
)
