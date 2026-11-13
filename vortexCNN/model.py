"""
vortex 지도 -> 상(phase) 분류 CNN

입력: (batch, 1, L, L)  -- 각 칸의 winding number (+1 vortex, -1 antivortex, 0 없음)
출력: (batch,)          -- logit. sigmoid(logit) = "고온상(vortex가 풀린 상)일 확률"

특징 추출부(conv-BN-ReLU x3, AvgPool, GAP)는 CNN/model.py의 XYTemperatureCNN과
완전히 같음. 바뀐 것은 두 가지뿐:
    1) 입력 채널 2(cos, sin) -> 1(vortex 지도)
    2) 마지막 출력의 의미: 온도(회귀) -> 고온상일 확률의 logit(분류)
       (sigmoid는 손실 함수 BCEWithLogitsLoss 안에서 처리되므로 forward에는 없음)
그래서 "스핀 입력 CNN" 결과와 "vortex 입력 CNN" 결과를 같은 구조로 비교할 수 있음.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class XYPhaseCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        # circular padding: vortex 지도도 주기 경계조건으로 계산했으므로 격자가 도넛 모양
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1,
                               padding_mode="circular", bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1,
                               padding_mode="circular", bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1,
                               padding_mode="circular", bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.AvgPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)   # 입력 크기 L과 무관 -> L=64, 128, 256 모두 사용 가능

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
        x = self.fc2(x)               # (batch, 1) -- logit

        return x.squeeze(-1)          # (batch,)


class VortexDensityMLP(nn.Module):
    """
    비교 기준(baseline): vortex 밀도 하나만 입력으로 받는 MLP.
    CNN이 이것보다 나은 게 없다면 CNN은 "vortex 개수만 센 것".
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(1.0))
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def set_normalization(self, mean, std):
        self.mean.fill_(float(mean))
        self.std.fill_(float(std))

    def forward(self, rho):           # rho: (batch,)
        x = ((rho - self.mean) / self.std).unsqueeze(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


if __name__ == "__main__":
    model = XYPhaseCNN()
    dummy = torch.randn(4, 1, 256, 256)
    print(f"입력 {tuple(dummy.shape)} -> 출력 {tuple(model(dummy).shape)}")
    print(f"CNN 파라미터: {sum(p.numel() for p in model.parameters()):,}")
