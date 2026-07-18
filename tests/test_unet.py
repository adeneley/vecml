"""Forward-shape and head-swap tests for the compact U-Net."""

import torch
from torch import nn

from vecml.models.unet import UNet, count_params


def test_forward_shape():
    model = UNet()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    assert y.shape == (2, 3, 256, 256)
    y = y.detach()
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0  # sigmoid head


def test_param_count_is_a_few_million():
    n = count_params(UNet())
    assert 1_000_000 < n < 20_000_000, n


def test_swappable_head():
    class OneChannelHead(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.proj = nn.Conv2d(in_ch, 1, 1)

        def forward(self, x):
            return self.proj(x)

    model = UNet(head=OneChannelHead)
    y = model(torch.randn(1, 3, 128, 128))
    assert y.shape == (1, 1, 128, 128)
