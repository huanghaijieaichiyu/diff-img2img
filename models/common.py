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


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


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
        self.norm = nn.GroupNorm(num_groups=_group_count(c2), num_channels=c2)
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
            nn.GroupNorm(num_groups=_group_count(c2), num_channels=c2),
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
        self.norm1 = nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels)
        heads = max(1, min(num_heads, max(1, channels // 32)))
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm2 = nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels)
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


class PooledGlobalContextBlock(nn.Module):
    """
    Efficiency-oriented global context block.
    Pools features to a small token grid before attention, then broadcasts the
    refined context back to the original resolution.
    """

    def __init__(self, channels: int, num_heads: int = 4, pooled_size: int = 8):
        super().__init__()
        self.pooled_size = max(2, int(pooled_size))
        self.norm1 = nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels)
        heads = max(1, min(num_heads, max(1, channels // 32)))
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm2 = nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        pooled_h = min(self.pooled_size, height)
        pooled_w = min(self.pooled_size, width)

        pooled = F.adaptive_avg_pool2d(self.norm1(x), output_size=(pooled_h, pooled_w))
        attn_input = pooled.flatten(2).transpose(1, 2)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, channels, pooled_h, pooled_w)
        if (pooled_h, pooled_w) != (height, width):
            attn_output = F.interpolate(attn_output, size=(height, width), mode="bilinear", align_corners=False)

        x = x + attn_output
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


# ============================================================================
#  P1 Improvement: NAFNet Architecture Components
#  Reference: "Simple Baselines for Image Restoration" (Chen et al., 2022)
# ============================================================================

class SimpleGate(nn.Module):
    """
    Simple gating mechanism from NAFNet.
    Splits channels in half and multiplies them element-wise.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNormFunction(torch.autograd.Function):
    """Optimized LayerNorm for 2D feature maps"""
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps (channels)"""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class NAFBlock(nn.Module):
    """
    P1 Improvement: NAFNet Block for efficient image restoration.
    Uses simple gating instead of complex attention mechanisms.

    Reference: "Simple Baselines for Image Restoration" (ECCV 2022)
    """
    def __init__(self, c, dw_expansion=2, ffn_expansion=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * dw_expansion

        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = c * ffn_expansion
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class CrossAttentionBlock(nn.Module):
    """
    P1 Improvement: Cross-Attention for better condition injection.
    Allows the model to attend to conditioning information more effectively.
    """
    def __init__(self, dim, context_dim=None, num_heads=8, dropout=0.0):
        super().__init__()
        context_dim = context_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = LayerNorm2d(dim)
        self.norm_context = LayerNorm2d(context_dim)

        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(context_dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(context_dim, dim, kernel_size=1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        """
        Args:
            x: (B, C, H, W) - query features
            context: (B, C_ctx, H, W) - context features (if None, self-attention)
        """
        if context is None:
            context = x

        B, C, H, W = x.shape

        # Normalize
        x_norm = self.norm1(x)
        context_norm = self.norm_context(context)

        # Project to Q, K, V
        q = self.to_q(x_norm)
        k = self.to_k(context_norm)
        v = self.to_v(context_norm)

        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # B, heads, HW, head_dim
        k = k.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        out = attn @ v  # B, heads, HW, head_dim
        out = out.transpose(2, 3).reshape(B, C, H, W)

        # Project out
        out = self.to_out(out)

        return x + out
