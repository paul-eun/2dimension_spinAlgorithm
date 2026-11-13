"""
vortex 지도로 저온상/고온상을 분류하는 CNN을 학습하고, 그 출력으로 전이온도를 추정.

방법 (Carrasquilla & Melko 2017 방식):
    1) 학습: 상이 확실한 양 끝 온도만 사용
         T <= t_low  -> 라벨 0 (저온상: vortex가 쌍으로 묶임)
         T >= t_high -> 라벨 1 (고온상: vortex가 풀림)
       그 사이 온도(전이 근처)는 학습에 전혀 쓰지 않음.
    2) 평가: test 파일의 61개 온도 전부를 넣고 온도별로 P(고온상) 평균을 냄.
    3) P(고온상) = 0.5 가 되는 온도 = 모델이 추정한 전이온도 T_c(L)

데이터 분할 (조원 규칙):
    train      : vortex_maps_1, 2, 3  (양 끝 온도만)
    validation : vortex_maps_4        (양 끝 온도로 손실을 재서 가장 좋은 epoch 선택,
                                       그리고 전 온도 곡선도 그려서 test와 일치하는지 확인)
    test       : vortex_maps_5        (전 온도)

비교 기준: vortex 밀도 하나만 쓰는 MLP를 같은 조건으로 학습.
    CNN ≈ MLP  -> CNN은 vortex 개수만 센 것
    CNN != MLP -> CNN이 vortex의 배치(공간 구조)를 쓰고 있음

실행:
    python3 vortexCNN/train.py
    python3 vortexCNN/train.py --t-low 0.5 --t-high 1.3 --epochs 15
"""

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import VortexMapDataset, load_vortex_maps, vortex_density, vortex_dir_for, vortex_map_path
from model import VortexDensityMLP, XYPhaseCNN

T_BKT = 0.893   # 무한 격자 문헌값 (J=1)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def crossing(temps, prob, level=0.5):
    """온도 오름차순 곡선 prob(T)가 처음으로 level을 넘는 온도 (선형 보간)."""
    for k in range(len(temps) - 1):
        p0, p1 = prob[k] - level, prob[k + 1] - level
        if p0 < 0 <= p1:
            return temps[k] + (temps[k + 1] - temps[k]) * (-p0) / (p1 - p0)
    return float("nan")


def per_temperature(temps, prob):
    ts = np.unique(np.round(temps, 4))
    mean = np.array([prob[np.isclose(temps, t, atol=1e-4)].mean() for t in ts])
    std = np.array([prob[np.isclose(temps, t, atol=1e-4)].std() for t in ts])
    return ts, mean, std


# ---------------------------------------------------------------- CNN
def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logit = model(x)
            loss = criterion(logit, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            correct += ((logit > 0).float() == y).sum().item()
            n += len(y)
    return total / n, correct / n


@torch.no_grad()
def predict_prob(model, maps, device, batch_size=64):
    model.eval()
    loader = DataLoader(VortexMapDataset(maps, np.zeros(len(maps))), batch_size=batch_size)
    return np.concatenate([torch.sigmoid(model(x.to(device))).cpu().numpy() for x, _ in loader])


# ---------------------------------------------------------------- baseline MLP
def train_density_mlp(tr_rho, tr_y, va_rho, va_y, pos_weight, epochs=200, seed=0):
    torch.manual_seed(seed)
    model = VortexDensityMLP()
    model.set_normalization(tr_rho.mean(), tr_rho.std())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    tr = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(tr_rho), torch.from_numpy(tr_y)),
                    batch_size=32, shuffle=True)
    va_x, va_t = torch.from_numpy(va_rho), torch.from_numpy(va_y)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for x, y in tr:
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = crit(model(va_x), va_t).item()
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--val", type=int, default=4)
    p.add_argument("--test", type=int, default=5)
    p.add_argument("--t-low", type=float, default=0.6, help="이 온도 이하 = 저온상 라벨 0")
    p.add_argument("--t-high", type=float, default=1.2, help="이 온도 이상 = 고온상 라벨 1")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--L", type=int, default=256, help="격자 크기: vortexCNN/data/vortex_maps/L<L> 을 씀")
    p.add_argument("--out-dir", default=None,
                   help="기본: vortexCNN/results/L<L>")
    args = p.parse_args()
    results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if args.out_dir is None:
        args.out_dir = os.path.join(results, f"L{args.L}")
    vdir = vortex_dir_for(args.L)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"tlow{args.t_low}_thigh{args.t_high}_seed{args.seed}"
    print(f"device: {device}   라벨: T<={args.t_low} -> 0 (저온상),  T>={args.t_high} -> 1 (고온상)")

    eps = 1e-4
    extreme = lambda T: (T <= args.t_low + eps) | (T >= args.t_high - eps)
    label = lambda T: (T >= args.t_high - eps).astype(np.float32)

    # ---------- 1) 데이터 ----------
    t0 = time.perf_counter()
    loaded = [load_vortex_maps(vortex_map_path(k, vdir), keep=extreme) for k in args.train]
    tr_maps = np.concatenate([m for m, _ in loaded])
    tr_T = np.concatenate([t for _, t in loaded])
    del loaded
    va_maps_all, va_T_all = load_vortex_maps(vortex_map_path(args.val, vdir))
    te_maps, te_T = load_vortex_maps(vortex_map_path(args.test, vdir))
    va_mask = extreme(va_T_all)
    tr_y, va_y = label(tr_T), label(va_T_all[va_mask])
    L = tr_maps.shape[-1]

    n1 = tr_y.sum()
    n0 = len(tr_y) - n1
    pos_weight = n0 / n1   # 저온/고온 온도 개수가 달라도 두 클래스가 같은 비중이 되도록
    print(f"L={L}  train(양끝 온도만)={len(tr_y)} [저온 {int(n0)} / 고온 {int(n1)}]  "
          f"val(양끝)={va_mask.sum()}  test(전 온도)={len(te_T)}   "
          f"({time.perf_counter() - t0:.0f}초)")

    # ---------- 2) CNN 학습 ----------
    model = XYPhaseCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    tr_loader = DataLoader(VortexMapDataset(tr_maps, tr_y), batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(VortexMapDataset(va_maps_all[va_mask], va_y), batch_size=64)

    print("\n=== CNN 학습 ===")
    best, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, args.epochs + 1):
        t1 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, tr_loader, crit, device, opt)
        va_loss, va_acc = run_epoch(model, va_loader, crit, device)
        if va_loss < best:
            best, best_epoch, best_state = va_loss, epoch, copy.deepcopy(model.state_dict())
        print(f"  epoch {epoch:2d}  train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
              f"val loss={va_loss:.4f} acc={va_acc:.4f}  ({time.perf_counter() - t1:.0f}초)", flush=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"cnn_{tag}.pt"))
    print(f"val 손실 최소 epoch: {best_epoch}")

    # ---------- 3) baseline: vortex 밀도 MLP ----------
    mlp = train_density_mlp(vortex_density(tr_maps), tr_y,
                            vortex_density(va_maps_all[va_mask]), va_y, pos_weight, seed=args.seed)

    # ---------- 4) 전 온도에서 P(고온상) 곡선 -> 전이온도 ----------
    results = {}
    for name, maps, T in (("val", va_maps_all, va_T_all), ("test", te_maps, te_T)):
        p_cnn = predict_prob(model, maps, device)
        with torch.no_grad():
            p_mlp = torch.sigmoid(mlp(torch.from_numpy(vortex_density(maps)))).numpy()
        ts, cnn_m, cnn_s = per_temperature(T, p_cnn)
        _, mlp_m, mlp_s = per_temperature(T, p_mlp)
        results[name] = dict(T=ts, cnn=cnn_m, cnn_std=cnn_s, mlp=mlp_m, mlp_std=mlp_s,
                             tc_cnn=crossing(ts, cnn_m), tc_mlp=crossing(ts, mlp_m))

    te = results["test"]
    print("\n=== test 파일: 온도별 P(고온상) 평균 ===")
    print("   T     CNN(vortex 지도)   MLP(vortex 밀도)   비고")
    for k, t in enumerate(te["T"]):
        note = "학습(저온)" if t <= args.t_low + eps else "학습(고온)" if t >= args.t_high - eps else ""
        print(f"  {t:.3f}   {te['cnn'][k]:.3f} ± {te['cnn_std'][k]:.3f}    "
              f"{te['mlp'][k]:.3f} ± {te['mlp_std'][k]:.3f}    {note}")

    print("\n=== 추정 전이온도 (P = 0.5 되는 온도) ===")
    for name in ("val", "test"):
        r = results[name]
        print(f"  {name:4s}: CNN {r['tc_cnn']:.4f}   밀도 MLP {r['tc_mlp']:.4f}")
    print(f"  참고: 무한 격자 T_BKT ≈ {T_BKT}")

    # ---------- 5) 저장 ----------
    np.savez(os.path.join(args.out_dir, f"curves_{tag}.npz"),
             **{f"{n}_{k}": v for n, r in results.items() for k, v in r.items()},
             t_low=args.t_low, t_high=args.t_high, L=L)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(te["T"][0], args.t_low, color="tab:blue", alpha=0.08, label="train (label 0)")
    ax.axvspan(args.t_high, te["T"][-1], color="tab:red", alpha=0.08, label="train (label 1)")
    ax.errorbar(te["T"], te["cnn"], yerr=te["cnn_std"], fmt="o-", ms=3, capsize=2,
                label=f"CNN (vortex map)  T_c={te['tc_cnn']:.3f}")
    ax.errorbar(te["T"], te["mlp"], yerr=te["mlp_std"], fmt="s--", ms=3, capsize=2, alpha=0.7,
                label=f"MLP (vortex density)  T_c={te['tc_mlp']:.3f}")
    ax.axhline(0.5, color="gray", lw=0.8)
    ax.axvline(T_BKT, color="k", ls=":", label=f"T_BKT (L=inf) = {T_BKT}")
    ax.set_xlabel("T [J]")
    ax.set_ylabel("P(high-T phase)")
    ax.set_title(f"L={L}, test = vortex_maps_{args.test}")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    png = os.path.join(args.out_dir, f"phase_curve_{tag}.png")
    fig.savefig(png, dpi=150)
    print(f"\n저장: {png}")


if __name__ == "__main__":
    main()
