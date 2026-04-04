"""
Common neural network building blocks.
Used by the Retinex, conditioning, and scale-specific architecture variants.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvTranspose(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv_transpose(x)))


class ConvGNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=autopad(k), groups=g, bias=False)
        self.norm = nn.GroupNorm(num_groups=max(1, min(8, c2)), num_channels=c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DWConvGNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.depthwise = ConvGNAct(c1, c1, k=k, s=s, g=c1, act=act)
        self.pointwise = ConvGNAct(c1, c2, k=1, s=1, act=act)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        scale = self.pool(x)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class MBConvBlock(nn.Module):
    def __init__(self, c1: int, c2: int, stride: int = 1, expansion: int = 2):
        super().__init__()
        hidden = max(c1, int(c1 * expansion))
        self.use_residual = stride == 1 and c1 == c2

        layers = []
        if hidden != c1:
            layers.append(ConvGNAct(c1, hidden, k=1, s=1))
        layers.extend([
            ConvGNAct(hidden, hidden, k=3, s=stride, g=hidden),
            SqueezeExcite(hidden),
            nn.Conv2d(hidden, c2, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, c2)), num_channels=c2),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return F.silu(out)


class GatedFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.proj = ConvGNAct(channels * 2, channels, k=1, s=1)

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        if left.shape[2:] != right.shape[2:]:
            right = F.interpolate(right, size=left.shape[2:], mode="bilinear", align_corners=False)
        cat = torch.cat([left, right], dim=1)
        gate = torch.sigmoid(self.gate(cat))
        mixed = gate * left + (1.0 - gate) * right
        return self.proj(torch.cat([mixed, left + right], dim=1))


class GlobalContextBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=max(1, min(8, channels)), num_channels=channels)
        heads = max(1, min(num_heads, max(1, channels // 32)))
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
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
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer2DBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.context = GlobalContextBlock(channels, num_heads=num_heads)
        self.local = ConvGNAct(channels, channels, k=3, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local(self.context(x))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
