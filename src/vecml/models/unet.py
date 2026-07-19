"""Compact U-Net for the cleanup/segmentation front-end.

Deliberately small (a few million params): the constraint on the Mac is
iteration speed, not VRAM. GroupNorm + SiLU throughout because BatchNorm
misbehaves at the tiny batch sizes used for overfit sanity runs.

The final output head is a swappable module passed to the constructor, so the
same backbone can later drive a label-map head, a multi-task head, etc. The v0
head (RGBHead) is plain 3-channel RGB regression squashed to [0, 1].
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def _norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with a group count that always divides the channel count."""
    groups = 8
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by GroupNorm + SiLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RGBHead(nn.Module):
    """v0 output head: 1x1 conv to 3 channels, sigmoid to [0, 1]."""

    out_channels = 3

    def __init__(self, in_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(x))


class DualHead(nn.Module):
    """RGB regression + K-way label-map logits from the shared features.

    The RGB branch keeps the cleanup task (and doubles as the colour oracle
    for predicted regions); the label branch is the paint-by-numbers head.
    Logits are returned raw: CrossEntropyLoss wants them unsoftmaxed, and the
    softmax probabilities at region boundaries carry sub-pixel edge position
    that the engine will eventually want, so nothing here may argmax early.
    """

    out_channels = 3

    def __init__(self, in_ch: int, n_classes: int = 16):
        super().__init__()
        self.n_classes = n_classes
        self.rgb = nn.Conv2d(in_ch, 3, 1)
        self.label = nn.Conv2d(in_ch, n_classes, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"rgb": torch.sigmoid(self.rgb(x)), "logits": self.label(x)}


class UNet(nn.Module):
    """Encoder 32->64->128, bottleneck 256, symmetric decoder with skips.

    At 256x256 input the spatial ladder is 256 -> 128 -> 64 -> 32 (bottleneck)
    and back up to 256. The output head is swappable; pass a factory that takes
    the final feature-channel count and returns an nn.Module.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base: int = 32,
        head: Callable[[int], nn.Module] | None = None,
    ):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        # Encoder.
        self.enc1 = ConvBlock(in_ch, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck.
        self.bottleneck = ConvBlock(c3, c4)

        # Decoder: transposed-conv upsample, concat skip, ConvBlock.
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)

        self.head = head(c1) if head is not None else RGBHead(c1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)  # c1 @ H
        e2 = self.enc2(self.pool(e1))  # c2 @ H/2
        e3 = self.enc3(self.pool(e2))  # c3 @ H/4
        b = self.bottleneck(self.pool(e3))  # c4 @ H/8

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def count_params(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
