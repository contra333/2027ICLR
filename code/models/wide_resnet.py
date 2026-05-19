from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, dropout_rate: float):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out if not isinstance(self.shortcut, nn.Identity) else x)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth: int, widen_factor: int, dropout_rate: float, num_classes: int):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WRN depth must satisfy (depth - 4) % 6 == 0, got {depth}")
        blocks_per_group = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.in_planes = channels[0]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = self._make_group(channels[1], blocks_per_group, stride=1, dropout_rate=dropout_rate)
        self.block2 = self._make_group(channels[2], blocks_per_group, stride=2, dropout_rate=dropout_rate)
        self.block3 = self._make_group(channels[3], blocks_per_group, stride=2, dropout_rate=dropout_rate)
        self.bn = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.feature_dim = channels[3]
        self._init_weights()

    def _make_group(self, out_planes: int, blocks: int, stride: int, dropout_rate: float) -> nn.Sequential:
        layers = [WideBasicBlock(self.in_planes, out_planes, stride=stride, dropout_rate=dropout_rate)]
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(WideBasicBlock(self.in_planes, out_planes, stride=1, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        features = torch.flatten(out, 1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits


def standard_wrn_28_10_dropout03(num_classes: int) -> WideResNet:
    return WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=num_classes)


def standard_wrn_28_10_nodrop(num_classes: int) -> WideResNet:
    return WideResNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=num_classes)
