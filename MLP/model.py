"""
에너지 밀도 -> 온도 회귀 MLP (CNN 비교용)

입력: (batch, 1)  -- 에너지 밀도 e = E / L^2
출력: (batch,)    -- 예측 온도 (스칼라)

CNN이 스핀 배치에서 사실상 에너지만 읽어서 온도를 결정한다면,
이 MLP와 같은 성능/같은 예측을 보여야 함.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class XYEnergyMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()

        # e는 대략 -2 ~ -0.5 범위라 그대로 넣으면 학습이 느림.
        # 학습 데이터의 평균/표준편차로 표준화 (set_normalization으로 지정,
        # buffer라서 state_dict에 같이 저장됨)
        self.register_buffer("e_mean", torch.zeros(1))
        self.register_buffer("e_std", torch.ones(1))

        # 은닉층 2개 (각 hidden개 뉴런)
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def set_normalization(self, mean, std):
        self.e_mean.fill_(float(mean))
        self.e_std.fill_(float(std))

    def forward(self, x):
        x = (x - self.e_mean) / self.e_std
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)               # (batch, 1) -- 회귀이므로 활성화 함수 없음

        return x.squeeze(-1)          # (batch,)


if __name__ == "__main__":
    # 구조가 올바르게 동작하는지 더미 입력으로 빠르게 확인
    model = XYEnergyMLP()
    dummy = torch.randn(4, 1)  # batch=4
    out = model(dummy)
    print(f"입력 shape: {dummy.shape}")
    print(f"출력 shape: {out.shape}")  # (4,) 이어야 정상
    n_params = sum(p.numel() for p in model.parameters())
    print(f"전체 파라미터 개수: {n_params:,}")
