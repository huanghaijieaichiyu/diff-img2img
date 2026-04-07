import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import C2f, Concat, Conv, ConvGNAct, ConvTranspose, DWConvGNAct, GatedFusion, GlobalContextBlock, MBConvBlock, Transformer2DBlock, NAFBlock, LayerNorm2d, PooledGlobalContextBlock


def _resize_if_needed(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[2:] != target.shape[2:]:
        source = F.interpolate(source, size=target.shape[2:], mode="bilinear", align_corners=False)
    return source


class SmallDecomNet(nn.Module):
    """
    Efficiency-oriented Retinex decomposition.
    Uses MBConv-style blocks and shallower depth for 8GB-class training.
    """

    def __init__(self, base_channel=24):
        super().__init__()
        c1, c2, c3 = base_channel, base_channel * 2, base_channel * 4

        self.stem = DWConvGNAct(3, c1, k=3, s=1)
        self.enc0 = MBConvBlock(c1, c1)
        self.down1 = MBConvBlock(c1, c2, stride=2)
        self.enc1 = MBConvBlock(c2, c2)
        self.down2 = MBConvBlock(c2, c3, stride=2)
        self.bottleneck = nn.Sequential(
            MBConvBlock(c3, c3),
            SqueezeExciteLite(c3),
        )

        self.up1 = ConvTranspose(c3, c2, k=2, s=2)
        self.fuse1 = GatedFusion(c2)
        self.dec1 = MBConvBlock(c2, c2)

        self.up0 = ConvTranspose(c2, c1, k=2, s=2)
        self.fuse0 = GatedFusion(c1)
        self.dec0 = MBConvBlock(c1, c1)

        self.r_head = nn.Sequential(ConvGNAct(c1, c1, k=3), nn.Conv2d(c1, 3, kernel_size=3, padding=1))
        self.i_head = nn.Sequential(ConvGNAct(c1, c1, k=3), nn.Conv2d(c1, 1, kernel_size=3, padding=1))

    def forward(self, x):
        feat0 = self.enc0(self.stem(x))
        feat1 = self.enc1(self.down1(feat0))
        feat2 = self.bottleneck(self.down2(feat1))

        dec1 = self.dec1(self.fuse1(_resize_if_needed(self.up1(feat2), feat1), feat1))
        dec0 = self.dec0(self.fuse0(_resize_if_needed(self.up0(dec1), feat0), feat0))
        return torch.sigmoid(self.r_head(dec0)), torch.sigmoid(self.i_head(dec0))


class SqueezeExciteLite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(8, channels // 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        scale = self.pool(x)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class MiddleDecomNet(nn.Module):
    """
    Balanced Retinex decomposition.
    Three-scale U-Net + one global context block.
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

        self.r_head = nn.Sequential(Conv(base_channel, base_channel, k=3), nn.Conv2d(base_channel, 3, kernel_size=3, padding=1))
        self.i_head = nn.Sequential(Conv(base_channel, base_channel, k=3), nn.Conv2d(base_channel, 1, kernel_size=3, padding=1))

    def forward(self, x):
        feat0 = self.c2f0(self.cv0(x))
        feat1 = self.c2f1(self.down1(feat0))
        feat2 = self.c2f2(self.down2(feat1))
        feat3 = self.global_bottleneck(self.c2f3(self.down3(feat2)))

        dec2 = self.c2f_up2(self.concat2([_resize_if_needed(self.up2(feat3), feat2), feat2]))
        dec1 = self.c2f_up1(self.concat1([_resize_if_needed(self.up1(dec2), feat1), feat1]))
        dec0 = self.c2f_up0(self.concat0([_resize_if_needed(self.up0(dec1), feat0), feat0]))
        return torch.sigmoid(self.r_head(dec0)), torch.sigmoid(self.i_head(dec0))


class MaxDecomNet(nn.Module):
    """
    Quality-oriented Retinex decomposition.
    Uses deeper scale depth, global context, transformer refinement and gated skip fusion.
    """

    def __init__(self, base_channel=48):
        super().__init__()
        c1, c2, c3, c4 = base_channel, base_channel * 2, base_channel * 4, base_channel * 8

        self.stem = ConvGNAct(3, c1, k=3, s=1)
        self.enc0 = nn.Sequential(C2f(c1, c1, n=2, shortcut=True), Transformer2DBlock(c1, num_heads=2))

        self.down1 = ConvGNAct(c1, c2, k=3, s=2)
        self.enc1 = nn.Sequential(C2f(c2, c2, n=2, shortcut=True), Transformer2DBlock(c2, num_heads=4))

        self.down2 = ConvGNAct(c2, c3, k=3, s=2)
        self.enc2 = nn.Sequential(C2f(c3, c3, n=2, shortcut=True), Transformer2DBlock(c3, num_heads=8))

        self.down3 = ConvGNAct(c3, c4, k=3, s=2)
        self.bottleneck = nn.Sequential(
            C2f(c4, c4, n=2, shortcut=True),
            GlobalContextBlock(c4, num_heads=8),
            Transformer2DBlock(c4, num_heads=8),
        )

        self.up2 = ConvTranspose(c4, c3, k=2, s=2)
        self.fuse2 = GatedFusion(c3)
        self.dec2 = nn.Sequential(C2f(c3, c3, n=2, shortcut=True), Transformer2DBlock(c3, num_heads=8))

        self.up1 = ConvTranspose(c3, c2, k=2, s=2)
        self.fuse1 = GatedFusion(c2)
        self.dec1 = nn.Sequential(C2f(c2, c2, n=2, shortcut=True), Transformer2DBlock(c2, num_heads=4))

        self.up0 = ConvTranspose(c2, c1, k=2, s=2)
        self.fuse0 = GatedFusion(c1)
        self.dec0 = nn.Sequential(C2f(c1, c1, n=2, shortcut=True), Transformer2DBlock(c1, num_heads=2))

        self.r_head = nn.Sequential(ConvGNAct(c1, c1, k=3), nn.Conv2d(c1, 3, kernel_size=3, padding=1))
        self.i_head = nn.Sequential(ConvGNAct(c1, c1, k=3), nn.Conv2d(c1, 1, kernel_size=3, padding=1))

    def forward(self, x):
        feat0 = self.enc0(self.stem(x))
        feat1 = self.enc1(self.down1(feat0))
        feat2 = self.enc2(self.down2(feat1))
        feat3 = self.bottleneck(self.down3(feat2))

        dec2 = self.dec2(self.fuse2(_resize_if_needed(self.up2(feat3), feat2), feat2))
        dec1 = self.dec1(self.fuse1(_resize_if_needed(self.up1(dec2), feat1), feat1))
        dec0 = self.dec0(self.fuse0(_resize_if_needed(self.up0(dec1), feat0), feat0))
        return torch.sigmoid(self.r_head(dec0)), torch.sigmoid(self.i_head(dec0))


class NAFDecomNet(nn.Module):
    """
    P1 Improvement: NAFNet-based Retinex decomposition.
    Uses simple gating mechanism for efficient and effective decomposition.
    Reference: "Simple Baselines for Image Restoration" (ECCV 2022)
    """

    def __init__(self, base_channel=32, num_blocks=4):
        super().__init__()
        self.stem = nn.Conv2d(3, base_channel, kernel_size=3, padding=1)

        # Encoder with NAFBlocks
        self.enc0 = nn.Sequential(*[NAFBlock(base_channel) for _ in range(num_blocks)])
        self.down1 = nn.Conv2d(base_channel, base_channel * 2, kernel_size=3, stride=2, padding=1)
        self.enc1 = nn.Sequential(*[NAFBlock(base_channel * 2) for _ in range(num_blocks)])
        self.down2 = nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Sequential(*[NAFBlock(base_channel * 4) for _ in range(num_blocks)])

        # Bottleneck
        self.bottleneck = nn.Sequential(*[NAFBlock(base_channel * 4) for _ in range(num_blocks)])

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channel * 4, base_channel * 2, kernel_size=2, stride=2)
        self.fuse1 = nn.Conv2d(base_channel * 4, base_channel * 2, kernel_size=1)  # Reduce channels after concat
        self.dec1 = nn.Sequential(*[NAFBlock(base_channel * 2) for _ in range(num_blocks)])
        self.up0 = nn.ConvTranspose2d(base_channel * 2, base_channel, kernel_size=2, stride=2)
        self.fuse0 = nn.Conv2d(base_channel * 2, base_channel, kernel_size=1)  # Reduce channels after concat
        self.dec0 = nn.Sequential(*[NAFBlock(base_channel) for _ in range(num_blocks)])

        # Output heads
        self.r_head = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1),
            LayerNorm2d(base_channel),
            nn.SiLU(),
            nn.Conv2d(base_channel, 3, kernel_size=3, padding=1)
        )
        self.i_head = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1),
            LayerNorm2d(base_channel),
            nn.SiLU(),
            nn.Conv2d(base_channel, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder
        feat0 = self.enc0(self.stem(x))
        feat1 = self.enc1(self.down1(feat0))
        feat2 = self.bottleneck(self.enc2(self.down2(feat1)))

        # Decoder with skip connections
        up1 = _resize_if_needed(self.up1(feat2), feat1)
        dec1 = self.dec1(self.fuse1(torch.cat([up1, feat1], dim=1)))
        up0 = _resize_if_needed(self.up0(dec1), feat0)
        dec0 = self.dec0(self.fuse0(torch.cat([up0, feat0], dim=1)))

        return torch.sigmoid(self.r_head(dec0)), torch.sigmoid(self.i_head(dec0))


class NAFLiteDecomNet(nn.Module):
    """
    Lightweight NAF-style decomposition with pooled global context.
    Uses fewer NAF blocks and a pooled attention bottleneck to preserve the
    long-range illumination cues without the full attention cost.
    """

    def __init__(self, base_channel=24, num_blocks=2, pooled_size=8):
        super().__init__()
        c1, c2, c3 = base_channel, base_channel * 2, base_channel * 4

        self.stem = nn.Conv2d(3, c1, kernel_size=3, padding=1)

        self.enc0 = nn.Sequential(*[NAFBlock(c1) for _ in range(num_blocks)])
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.enc1 = nn.Sequential(*[NAFBlock(c2) for _ in range(num_blocks)])
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Sequential(*[NAFBlock(c3) for _ in range(num_blocks)])

        self.bottleneck = nn.Sequential(
            NAFBlock(c3),
            PooledGlobalContextBlock(c3, num_heads=4, pooled_size=pooled_size),
            NAFBlock(c3),
        )

        self.up1 = ConvTranspose(c3, c2, k=2, s=2)
        self.fuse1 = GatedFusion(c2)
        self.dec1 = nn.Sequential(*[NAFBlock(c2) for _ in range(num_blocks)])

        self.up0 = ConvTranspose(c2, c1, k=2, s=2)
        self.fuse0 = GatedFusion(c1)
        self.dec0 = nn.Sequential(*[NAFBlock(c1) for _ in range(num_blocks)])

        self.r_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            LayerNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, 3, kernel_size=3, padding=1),
        )
        self.i_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            LayerNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        feat0 = self.enc0(self.stem(x))
        feat1 = self.enc1(self.down1(feat0))
        feat2 = self.bottleneck(self.enc2(self.down2(feat1)))

        dec1 = self.dec1(self.fuse1(_resize_if_needed(self.up1(feat2), feat1), feat1))
        dec0 = self.dec0(self.fuse0(_resize_if_needed(self.up0(dec1), feat0), feat0))
        return torch.sigmoid(self.r_head(dec0)), torch.sigmoid(self.i_head(dec0))


def build_decom_net(variant: str, base_channel: int):
    variant = (variant or "middle").lower()
    if variant in {"small", "mobile", "lite"}:
        return SmallDecomNet(base_channel=base_channel)
    if variant in {"max", "quality", "large"}:
        return MaxDecomNet(base_channel=base_channel)
    if variant in {"naf_lite", "naflite", "efficient"}:
        return NAFLiteDecomNet(base_channel=base_channel, num_blocks=2)
    if variant in {"naf", "nafnet"}:
        return NAFDecomNet(base_channel=base_channel, num_blocks=4)
    return MiddleDecomNet(base_channel=base_channel)


DecomNet = MiddleDecomNet
