"""
vortexCNN/data/vortex_maps/L<L>/vortex_maps_<번호>.npz 를 불러와서 PyTorch Dataset으로 감싸는 부분.

vortex 지도는 int8(-1, 0, +1)로 그대로 메모리에 두고,
배치를 꺼낼 때만 float로 바꿈 (L=256 파일 3개를 float32로 올리면 수 GB라서).
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def vortex_dir_for(L):
    """격자 크기 L의 vortex 지도 폴더: vortexCNN/data/vortex_maps/L<L>"""
    return os.path.join(DATA_DIR, "vortex_maps", f"L{L}")


VORTEX_DIR = vortex_dir_for(256)   # 기본값 (confusion.py 등)


def vortex_map_path(index, vortex_dir=VORTEX_DIR):
    return os.path.join(vortex_dir, f"vortex_maps_{index}.npz")


def load_vortex_maps(path, keep=None):
    """
    (maps int8 (N, L, L), temperatures float32 (N,)) 반환.
    keep: temperatures -> bool mask 함수. 주면 해당 샘플만 남김 (메모리 절약).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 가 없음. vortex 지도는 용량 때문에 git에 올리지 않았으므로 다시 만들어야 함 "
            f"(vortexCNN/README.md 의 '데이터 다시 만들기' 참고)")
    with np.load(path) as data:
        temps = data["temperatures"].astype(np.float32)
        maps = data["vortex_maps"]
    if keep is not None:
        mask = keep(temps)
        maps, temps = maps[mask], temps[mask]
    return maps, temps


def vortex_density(maps):
    """vortex(+1) 개수 / L^2.  (antivortex 수는 주기 경계에서 항상 같음)"""
    L = maps.shape[-1]
    return (maps == 1).sum(axis=(1, 2)).astype(np.float32) / (L * L)


class VortexMapDataset(Dataset):
    def __init__(self, maps, labels):
        self.maps = maps                                   # int8 (N, L, L)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.maps[idx].astype(np.float32)).unsqueeze(0)   # (1, L, L)
        return x, torch.tensor(self.labels[idx])
