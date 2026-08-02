"""TrackNetV3 的轻量 PyTorch 网络定义。

兼容公开 TrackNetV3_TableTennis 项目的 TrackNet state_dict：输入为
``seq_len * 3`` 个 RGB 通道；使用 ``bg_mode=concat`` 时，前面额外拼接
一个 RGB 中值背景，输出为每个输入帧一张球热力图。
"""
import torch
from torch import nn


class Conv2DBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Double2DConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class Triple2DConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return self.conv_3(x)


class TrackNetV3(nn.Module):
    """公开 TrackNetV3_TableTennis 的 TrackNet 主网络。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _up(x, target):
        return nn.functional.interpolate(x, size=target.shape[-2:], mode="nearest")

    def forward(self, x):
        x1 = self.down_block_1(x)
        x2 = self.down_block_2(nn.functional.max_pool2d(x1, 2))
        x3 = self.down_block_3(nn.functional.max_pool2d(x2, 2))
        x = self.bottleneck(nn.functional.max_pool2d(x3, 2))
        x = self.up_block_1(torch.cat([self._up(x, x3), x3], dim=1))
        x = self.up_block_2(torch.cat([self._up(x, x2), x2], dim=1))
        x = self.up_block_3(torch.cat([self._up(x, x1), x1], dim=1))
        return self.sigmoid(self.predictor(x))
