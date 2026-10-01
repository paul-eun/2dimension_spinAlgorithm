"""
simulation/xy_model_numba.py의 generate_dataset()이 반환하는
list[dict] 형태를 PyTorch Dataset으로 감싸는 래퍼.

simulation 쪽은 CNN에 대해 전혀 모르고, 여기서만 PyTorch와 연결합니다.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class XYSpinDataset(Dataset):
    def __init__(self, samples):
        """
        Parameters
        ----------
        samples : list[dict]
            generate_dataset()이 반환한 리스트.
            각 원소는 {"T", "spin_config", ...} 형태의 dict.
        """
        self.spin_configs = [s["spin_config"] for s in samples]
        self.temperatures = np.array(
            [s["T"] for s in samples], dtype=np.float32
        )

    def __len__(self):
        return len(self.temperatures)

    def __getitem__(self, idx):
        x = torch.from_numpy(
            np.ascontiguousarray(self.spin_configs[idx], dtype=np.float32)
        )   # (2, L, L)
        y = torch.tensor(self.temperatures[idx])         # scalar
        return x, y


def load_dataset_npz(path):
    """
    generate_data.py가 저장한 .npz 파일을 불러와서
    generate_dataset()이 반환하는 것과 같은 list[dict] 형태로 복원.
    """
    with np.load(path) as data:
        temps = data["temperatures"]
        spin_configs = data["spin_configs"]
        energies = data["energies"]
        mags = data["magnetizations"]
        n_vortex = data["n_vortex"]
        n_antivortex = data["n_antivortex"]

    samples = []
    for i in range(len(temps)):
        samples.append({
            "T": float(temps[i]),
            "spin_config": spin_configs[i],
            "energy": float(energies[i]),
            "magnetization": float(mags[i]),
            "n_vortex": int(n_vortex[i]),
            "n_antivortex": int(n_antivortex[i]),
        })
    return samples
