"""
data/generate_data.py가 저장한 xy_dataset_<번호>.npz에서
에너지 밀도 e = E / L^2 와 온도 T만 꺼내서 PyTorch Dataset으로 감싸는 래퍼.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class XYEnergyDataset(Dataset):
    def __init__(self, energy_density, temperatures):
        self.energy_density = np.asarray(energy_density, dtype=np.float32).reshape(-1, 1)  # (N, 1)
        self.temperatures = np.asarray(temperatures, dtype=np.float32)                    # (N,)

    def __len__(self):
        return len(self.temperatures)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.energy_density[idx])  # (1,)
        y = torch.tensor(self.temperatures[idx])         # scalar
        return x, y


def load_energy_npz(path):
    """
    npz 파일 하나에서 (에너지 밀도, 온도) 배열을 불러옴.
    저장된 energies는 격자 전체 에너지라서 L^2로 나눔
    (L은 spin_configs 배열의 마지막 축 크기).
    """
    with np.load(path) as data:
        L = data["spin_configs"].shape[-1]
        energy_density = data["energies"].astype(np.float64) / (L * L)
        temperatures = data["temperatures"]
    return energy_density, temperatures
