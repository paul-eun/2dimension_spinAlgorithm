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
    # npz는 data["키"]로 접근할 때마다 배열 전체를 새로 압축 해제하므로,
    # 반복문 밖에서 각 배열을 딱 한 번씩만 꺼내둠
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


def split_by_temperature(samples, test_temperatures):
    """
    test 온도에 해당하는 샘플만 통째로 떼어냄 -> (나머지, test)

    같은 온도에서 뽑힌 여러 샘플들은 서로 비슷하기 때문에(특히
    decorrelation이 완벽하지 않으면), 같은 온도의 샘플이 train과
    test에 동시에 섞여 들어가면 일반화 성능을 과대평가하게 됨.
    그래서 "샘플 단위"가 아니라 "온도 단위"로 test를 미리 통째로 떼어냄.
    (validation은 train.py에서 아예 별도로 생성된 파일을 사용)

    Parameters
    ----------
    test_temperatures : list[float]
        테스트용으로 완전히 분리할 온도 값들 (학습에 전혀 사용 안 함)
    """
    test_set = set(round(t, 6) for t in test_temperatures)
    rest = [s for s in samples if round(s["T"], 6) not in test_set]
    test = [s for s in samples if round(s["T"], 6) in test_set]
    return rest, test
