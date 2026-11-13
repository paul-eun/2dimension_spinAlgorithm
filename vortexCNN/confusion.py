"""
confusion 방식(van Nieuwenburg, Liu, Huber 2017)으로 전이온도 추정.

양 끝 학습과 달리 전이 근처 온도도 학습에 모두 씀. 전이온도를 미리 알려주지 않고:
    1) 경계 후보 T' 를 여러 개 정함
    2) 후보마다 온도 범위 안의 모든 샘플에 라벨: T < T' -> 0,  T > T' -> 1
       (CNN 구조는 train.py와 같은 XYPhaseCNN, 매 후보마다 처음부터 새로 학습)
    3) 후보마다 test 파일 정확도를 잼
    4) 정확도를 T'에 대해 그림
         - T'가 범위 양 끝이면 거의 모든 샘플이 한 라벨이라 정확도가 저절로 높음
         - T'가 진짜 전이점이면 라벨이 실제 상의 차이와 맞아서 학습이 잘 됨 -> 가운데 봉우리
         - 그 사이 엉뚱한 T'에서는 같은 상을 억지로 둘로 나눠야 해서 정확도가 떨어짐
       => "W" 모양, 가운데 봉우리 위치 = 추정 전이온도
비교용으로 vortex 밀도 MLP도 같은 방식으로 학습.

데이터: train = vortex_maps_1,2,3 / val = 4 (가장 좋은 epoch 선택) / test = 5 (정확도 측정)
결과는 후보 하나 끝날 때마다 저장하므로, 중간에 멈춰도 다시 실행하면 이어서 함.

실행:
    python3 vortexCNN/confusion.py
    python3 vortexCNN/confusion.py --t-min 0.6 --t-max 1.3 --epochs 3
"""

import argparse
import copy
import gc
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import VortexMapDataset, load_vortex_maps, vortex_density, vortex_map_path
from model import XYPhaseCNN
from train import pick_device, run_epoch, train_density_mlp

T_C_HELICITY = 0.902   # data/helicity_modulus.py 로 구한 L=256의 Υ = 2T/π 교차점

DEFAULT_CANDIDATES = [0.6625, 0.7125, 0.7625, 0.8125, 0.8375, 0.8625, 0.8875, 0.9125,
                      0.9375, 0.9625, 0.9875, 1.0125, 1.0375, 1.0625, 1.1125, 1.1625,
                      1.2125, 1.2625]


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = n = 0
    for x, y in loader:
        logit = model(x.to(device)).cpu()
        correct += ((logit > 0).float() == y).sum().item()
        n += len(y)
    return correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t-min", type=float, default=0.6, help="사용할 온도 범위 아래 끝")
    p.add_argument("--t-max", type=float, default=1.3, help="사용할 온도 범위 위 끝")
    p.add_argument("--candidates", type=float, nargs="+", default=DEFAULT_CANDIDATES)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=16,
                   help="L=256에서 batch 32는 GPU 메모리를 3GB 넘게 써서 RAM 8GB Mac에서는 16 권장")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "L256"))
    args = p.parse_args()

    device = pick_device()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"confusion_T{args.t_min}-{args.t_max}_seed{args.seed}"
    out_npz = os.path.join(args.out_dir, tag + ".npz")

    eps = 1e-4
    in_range = lambda T: (T >= args.t_min - eps) & (T <= args.t_max + eps)

    # ---------- 데이터 (한 번만 읽음) ----------
    t0 = time.perf_counter()
    loaded = [load_vortex_maps(vortex_map_path(k), keep=in_range) for k in (1, 2, 3)]
    tr_maps = np.concatenate([m for m, _ in loaded])
    tr_T = np.concatenate([t for _, t in loaded])
    del loaded
    va_maps, va_T = load_vortex_maps(vortex_map_path(4), keep=in_range)
    te_maps, te_T = load_vortex_maps(vortex_map_path(5), keep=in_range)
    tr_rho, va_rho, te_rho = vortex_density(tr_maps), vortex_density(va_maps), vortex_density(te_maps)
    print(f"device: {device}  온도 범위 {args.t_min}~{args.t_max} ({len(np.unique(np.round(tr_T, 4)))}개 온도)  "
          f"train={len(tr_T)} val={len(va_T)} test={len(te_T)}  ({time.perf_counter() - t0:.0f}초)")

    # ---------- 이전 실행 결과 이어받기 ----------
    done = {}
    if os.path.exists(out_npz):
        d = np.load(out_npz)
        done = {round(float(t), 4): (float(a), float(b))
                for t, a, b in zip(d["candidates"], d["acc_cnn"], d["acc_mlp"])}
        print(f"이전 결과 {len(done)}개 이어받음: {sorted(done)}")

    for t_split in args.candidates:
        key = round(t_split, 4)
        if key in done:
            continue
        label = lambda T: (T > t_split).astype(np.float32)
        tr_y, va_y, te_y = label(tr_T), label(va_T), label(te_T)
        t1 = time.perf_counter()

        # --- CNN: 매 후보마다 같은 seed로 처음부터 ---
        torch.manual_seed(args.seed)
        model = XYPhaseCNN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        crit = nn.BCEWithLogitsLoss()
        tr_loader = DataLoader(VortexMapDataset(tr_maps, tr_y), batch_size=args.batch, shuffle=True)
        va_loader = DataLoader(VortexMapDataset(va_maps, va_y), batch_size=64)
        te_loader = DataLoader(VortexMapDataset(te_maps, te_y), batch_size=64)
        best, best_state = float("inf"), None
        for _ in range(args.epochs):
            run_epoch(model, tr_loader, crit, device, opt)
            va_loss, _ = run_epoch(model, va_loader, crit, device)
            if va_loss < best:
                best, best_state = va_loss, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        acc_cnn = accuracy(model, te_loader, device)

        # --- 비교: vortex 밀도 MLP ---
        mlp = train_density_mlp(tr_rho, tr_y, va_rho, va_y, pos_weight=1.0, epochs=100, seed=args.seed)
        with torch.no_grad():
            acc_mlp = ((mlp(torch.from_numpy(te_rho)) > 0).numpy() == te_y.astype(bool)).mean()

        # 다음 후보로 넘어가기 전에 GPU(MPS) 메모리를 비움.
        # 안 비우면 후보마다 새 모델의 메모리가 쌓여서 RAM 8GB Mac에서 swap으로 거의 멈춤.
        del model, opt, tr_loader, va_loader, te_loader, best_state
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

        done[key] = (acc_cnn, float(acc_mlp))
        cands = sorted(done)
        np.savez(out_npz, candidates=np.array(cands),
                 acc_cnn=np.array([done[c][0] for c in cands]),
                 acc_mlp=np.array([done[c][1] for c in cands]),
                 t_min=args.t_min, t_max=args.t_max, epochs=args.epochs)
        print(f"  T'={t_split:.4f}  test 정확도: CNN {acc_cnn:.4f}   밀도 MLP {acc_mlp:.4f}   "
              f"(저온 라벨 비율 {1 - te_y.mean():.2f}, {time.perf_counter() - t1:.0f}초)", flush=True)

    # ---------- 정리 / 그래프 ----------
    cands = np.array(sorted(done))
    acc_cnn = np.array([done[c][0] for c in cands])
    acc_mlp = np.array([done[c][1] for c in cands])
    inner = (cands > args.t_min + 0.1) & (cands < args.t_max - 0.1)   # 양 끝(자명하게 높은 곳) 제외
    print("\n=== 결과 ===")
    for name, a in (("CNN", acc_cnn), ("밀도 MLP", acc_mlp)):
        k = np.argmax(np.where(inner, a, -1))
        print(f"  {name}: 안쪽 구간 정확도 최고 T' = {cands[k]:.4f} (정확도 {a[k]:.4f})")
    print(f"  참고: helicity modulus T_c(L=256) = {T_C_HELICITY}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(cands, acc_cnn, "o-", label="CNN (vortex map)")
    ax.plot(cands, acc_mlp, "s--", alpha=0.7, label="MLP (vortex density)")
    ax.axvline(T_C_HELICITY, color="k", lw=1, label=f"helicity modulus T_c(256) = {T_C_HELICITY}")
    ax.axvline(0.893, color="k", ls=":", lw=1, label="T_BKT (L=∞) = 0.893")
    ax.set_xlabel("trial boundary T'")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Confusion scheme, L=256, T in [{args.t_min}, {args.t_max}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.out_dir, tag + ".png")
    fig.savefig(png, dpi=150)
    print(f"저장: {png}")


if __name__ == "__main__":
    main()
