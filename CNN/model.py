"""
XY 모델 스핀 배치 -> 온도 회귀 CNN

입력: (batch, 2, L, L)  -- 채널 0: cos(theta), 채널 1: sin(theta)
출력: (batch,)          -- 예측 온도 (스칼라)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularConv2d(nn.Module):
    """
    Periodic boundary condition을 반영한 컨볼루션.

    일반 nn.Conv2d의 zero-padding은 격자 가장자리를 "벽"으로 취급하지만,
    우리 시뮬레이션은 PBC(도넛 모양)이므로 가장자리도 반대편과 실제로 이웃임.
    F.pad(mode='circular')로 감싸서 채운 뒤 padding=0으로 컨볼루션.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="circular")
        return self.conv(x)


class XYTemperatureCNN(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        # 채널 수를 점점 늘려가며 국소 정렬 -> 소용돌이 패턴 같은
        # 계층적 특징을 잡아내도록 설계
        self.conv1 = CircularConv2d(in_channels, 16, 3)
        self.conv2 = CircularConv2d(16, 32, 3)
        self.conv3 = CircularConv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2)

        # 위치와 무관하게 "격자 전체의 통계적 성질"만 보도록 GAP 사용
        # (Flatten 대신 -> 입력 크기(L)가 달라져도 구조를 그대로 재사용 가능)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = self.gap(x).flatten(1)   # (batch, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)               # (batch, 1) -- 회귀이므로 활성화 함수 없음

        return x.squeeze(-1)          # (batch,)


if __name__ == "__main__":
    # 구조가 올바르게 동작하는지 더미 입력으로 빠르게 확인
    model = XYTemperatureCNN()
    dummy = torch.randn(4, 2, 64, 64)  # batch=4, L=64
    out = model(dummy)
    print(f"입력 shape: {dummy.shape}")
    print(f"출력 shape: {out.shape}")  # (4,) 이어야 정상
    n_params = sum(p.numel() for p in model.parameters())
    print(f"전체 파라미터 개수: {n_params:,}")
