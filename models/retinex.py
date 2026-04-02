import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import C2f, Concat, Conv, ConvTranspose


class GlobalContextBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=max(1, min(8, channels)), num_channels=channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=max(1, min(num_heads, channels // 32 or 1)), batch_first=True)
        self.norm2 = nn.GroupNorm(num_groups=max(1, min(8, channels)), num_channels=channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        attn_input = self.norm1(x).flatten(2).transpose(1, 2)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output.transpose(1, 2).reshape(batch_size, channels, height, width)

        ffn_input = self.norm2(x)
        x = x + self.ffn(ffn_input)
        return x


class DecomNet(nn.Module):
    """
    Three-scale Retinex decomposition network with a global bottleneck block.

    It remains lightweight, but compared with the previous 2-downsample design:
      - sees a larger spatial context
      - models illumination at 1/8 scale
      - uses a global attention bottleneck to reduce local-only bias
    """

    def __init__(self, base_channel=32):
        super().__init__()

        self.cv0 = Conv(3, base_channel, k=3, s=1)
        self.c2f0 = C2f(base_channel, base_channel, n=1, shortcut=True)

        self.down1 = Conv(base_channel, base_channel * 2, k=3, s=2)
        self.c2f1 = C2f(base_channel * 2, base_channel * 2, n=1, shortcut=True)

        self.down2 = Conv(base_channel * 2, base_channel * 4, k=3, s=2)
        self.c2f2 = C2f(base_channel * 4, base_channel * 4, n=1, shortcut=True)

        self.down3 = Conv(base_channel * 4, base_channel * 8, k=3, s=2)
        self.c2f3 = C2f(base_channel * 8, base_channel * 8, n=1, shortcut=True)
        self.global_bottleneck = GlobalContextBlock(base_channel * 8)

        self.up2 = ConvTranspose(base_channel * 8, base_channel * 4, k=2, s=2)
        self.concat2 = Concat()
        self.c2f_up2 = C2f(base_channel * 8, base_channel * 4, n=1, shortcut=True)

        self.up1 = ConvTranspose(base_channel * 4, base_channel * 2, k=2, s=2)
        self.concat1 = Concat()
        self.c2f_up1 = C2f(base_channel * 4, base_channel * 2, n=1, shortcut=True)

        self.up0 = ConvTranspose(base_channel * 2, base_channel, k=2, s=2)
        self.concat0 = Concat()
        self.c2f_up0 = C2f(base_channel * 2, base_channel, n=1, shortcut=True)

        self.r_head = nn.Sequential(
            Conv(base_channel, base_channel, k=3),
            nn.Conv2d(base_channel, 3, kernel_size=3, padding=1),
        )
        self.i_head = nn.Sequential(
            Conv(base_channel, base_channel, k=3),
            nn.Conv2d(base_channel, 1, kernel_size=3, padding=1),
        )

    @staticmethod
    def _resize_if_needed(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape[2:] != target.shape[2:]:
            source = F.interpolate(source, size=target.shape[2:], mode="bilinear", align_corners=False)
        return source

    def forward(self, x):
        x0 = self.cv0(x)
        feat0 = self.c2f0(x0)

        x1 = self.down1(feat0)
        feat1 = self.c2f1(x1)

        x2 = self.down2(feat1)
        feat2 = self.c2f2(x2)

        x3 = self.down3(feat2)
        feat3 = self.c2f3(x3)
        feat3 = self.global_bottleneck(feat3)

        up2 = self._resize_if_needed(self.up2(feat3), feat2)
        dec2 = self.c2f_up2(self.concat2([up2, feat2]))

        up1 = self._resize_if_needed(self.up1(dec2), feat1)
        dec1 = self.c2f_up1(self.concat1([up1, feat1]))

        up0 = self._resize_if_needed(self.up0(dec1), feat0)
        dec0 = self.c2f_up0(self.concat0([up0, feat0]))

        reflectance = torch.sigmoid(self.r_head(dec0))
        illumination = torch.sigmoid(self.i_head(dec0))
        return reflectance, illumination
