"""
XY 모델 스핀 배치 -> 온도 회귀 CNN

입력: (batch, 2, L, L)  -- 채널 0: cos(theta), 채널 1: sin(theta)
출력: (batch,)          -- 예측 온도 (스칼라)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class XYTemperatureCNN(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        # 채널 수를 점점 늘려가며 국소 정렬 -> 소용돌이 패턴 같은
        # 계층적 특징을 잡아내도록 설계
        #
        # padding_mode="circular": 시뮬레이션이 PBC(도넛 모양)이므로
        # 가장자리를 0으로 채우지 않고 반대편 스핀으로 감싸서 채움
        #
        # bias=False: 바로 뒤 BatchNorm이 평균을 빼고 자체 shift(beta)를
        # 더하므로 conv의 bias는 상쇄되어 의미가 없음
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1,
                               padding_mode="circular", bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1,
                               padding_mode="circular", bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1,
                               padding_mode="circular", bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        # MaxPool 대신 AvgPool: 온도는 격자 전체의 "평균 밀도"
        # (에너지 밀도, vortex 밀도)로 결정됨.
        # - AvgPool은 밀도를 그대로 보존 (AvgPool 뒤 GAP = 그냥 GAP)
        # - MaxPool은 2x2 안에 vortex가 2개여도 1개처럼 보여서,
        #   vortex가 많은 고온 영역에서 밀도 정보가 포화됨
        self.pool = nn.AvgPool2d(2)

        # 위치와 무관하게 "격자 전체의 통계적 성질"만 보도록 GAP 사용
        # (Flatten 대신 -> 입력 크기(L)가 달라져도 구조를 그대로 재사용 가능)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
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
