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
        # (N, 2, L, L) 형태로 미리 쌓아둠 -- __getitem__마다 새로 변환하지 않도록
        self.spin_configs = np.stack(
            [s["spin_config"] for s in samples]
        ).astype(np.float32)
        self.temperatures = np.array(
            [s["T"] for s in samples], dtype=np.float32
        )

    def __len__(self):
        return len(self.temperatures)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.spin_configs[idx])   # (2, L, L)
        y = torch.tensor(self.temperatures[idx])         # scalar
        return x, y


def load_dataset_npz(path):
    """
    generate_data.py가 저장한 .npz 파일을 불러와서
    generate_dataset()이 반환하는 것과 같은 list[dict] 형태로 복원.
    """
    data = np.load(path)
    n = len(data["temperatures"])
    samples = []
    for i in range(n):
        samples.append({
            "T": float(data["temperatures"][i]),
            "spin_config": data["spin_configs"][i],
            "energy": float(data["energies"][i]),
            "magnetization": float(data["magnetizations"][i]),
            "n_vortex": int(data["n_vortex"][i]),
            "n_antivortex": int(data["n_antivortex"][i]),
        })
    return samples


def split_by_temperature(samples, test_temperatures, val_fraction=0.15, seed=0):
    """
    온도 '포인트' 단위로 train/val/test를 나눔.

    같은 온도에서 뽑힌 여러 샘플들은 서로 비슷하기 때문에(특히
    decorrelation이 완벽하지 않으면), 같은 온도의 샘플이 train과
    test에 동시에 섞여 들어가면 일반화 성능을 과대평가하게 됨.
    그래서 "샘플 단위"가 아니라 "온도 단위"로 test를 미리 통째로 떼어냄.

    Parameters
    ----------
    test_temperatures : list[float]
        테스트용으로 완전히 분리할 온도 값들 (학습에 전혀 사용 안 함)
    val_fraction : float
        나머지(train_val) 중 검증용으로 뗄 비율 (샘플 단위로 무작위 분할)
    """
    rng = np.random.default_rng(seed)
    test_set = set(round(t, 6) for t in test_temperatures)

    train_val = [s for s in samples if round(s["T"], 6) not in test_set]
    test = [s for s in samples if round(s["T"], 6) in test_set]

    idx = rng.permutation(len(train_val))
    n_val = int(len(train_val) * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train = [train_val[i] for i in train_idx]
    val = [train_val[i] for i in val_idx]

    return train, val, test
