import torch
import torch.nn as nn
from .common import PSA, Conv, C2f


class DecomNet(nn.Module):
    """
    Improved Retinex Decomposition Network using standard blocks from common.py.
    Uses InstanceNorm (via Gencov) and C2f blocks for better feature extraction.

    Input: Low-light image (B, 3, H, W)
    Output: 
        - Reflectance (B, 3, H, W)
        - Illumination (B, 1, H, W)
    """

    def __init__(self, channel=64):
        super(DecomNet, self).__init__()

        # Stem: 3 -> channel
        self.stem = Conv(3, channel)

        # Body: Deep feature extraction with Residual connections (C2f)
        self.body = nn.Sequential(
            C2f(channel, channel)
        )

        # Reflectance Branch
        self.r_head = nn.Sequential(
            Conv(channel, channel),
            Conv(channel, 3, k=3)
        )

        # Illumination Branch
        self.i_head = nn.Sequential(
            Conv(channel, channel),
            Conv(channel, 1, k=3)
        )

    def forward(self, x):
        feat = self.stem(x)
        feat = self.body(feat)

        # Output constrained to [0, 1]
        r = torch.sigmoid(self.r_head(feat))
        i = torch.sigmoid(self.i_head(feat))
        return r, i
