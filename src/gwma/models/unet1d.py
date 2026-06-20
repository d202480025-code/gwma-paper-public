from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet1D(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64, depth: int = 4) -> None:
        super().__init__()
        channels = [base_channels * 2**index for index in range(depth + 1)]
        self.stem = ConvBlock(in_channels, channels[0])
        self.down = nn.ModuleList(
            [
                nn.Sequential(nn.MaxPool1d(2), ConvBlock(channels[index], channels[index + 1]))
                for index in range(depth)
            ]
        )
        self.bottleneck = ConvBlock(channels[-1], channels[-1])
        self.up = nn.ModuleList(
            [
                ConvBlock(channels[index + 1] + channels[index], channels[index])
                for index in range(depth - 1, -1, -1)
            ]
        )
        self.output = nn.Conv1d(channels[0], in_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, None]:
        del mask_ratio
        original_length = x.shape[-1]
        skips = [self.stem(x)]
        hidden = skips[0]
        for down_block in self.down:
            hidden = down_block(hidden)
            skips.append(hidden)
        hidden = self.bottleneck(hidden)
        for up_block, skip in zip(self.up, reversed(skips[:-1])):
            hidden = F.interpolate(hidden, size=skip.shape[-1], mode="linear", align_corners=False)
            hidden = up_block(torch.cat((hidden, skip), dim=1))
        output = self.output(hidden)
        return output[..., :original_length], None
