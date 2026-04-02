import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_hvi_lite(x_01: torch.Tensor) -> torch.Tensor:
    intensity = (
        0.299 * x_01[:, 0:1] +
        0.587 * x_01[:, 1:2] +
        0.114 * x_01[:, 2:3]
    )
    chroma_r = x_01[:, 0:1] - intensity
    chroma_b = x_01[:, 2:3] - intensity
    return torch.cat([intensity, chroma_r, chroma_b], dim=1)


class LearnableHVITransform(nn.Module):
    """
    Stable learnable variant of HVI-lite.
    It starts from the hand-crafted HVI-lite transform and learns a small residual.
    """

    def __init__(self):
        super().__init__()
        self.delta_proj = nn.Conv2d(3, 3, kernel_size=1, bias=True)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    def forward(self, x_01: torch.Tensor) -> torch.Tensor:
        fixed_hvi = rgb_to_hvi_lite(x_01)
        delta = 0.1 * torch.tanh(self.delta_proj(x_01))
        learned_hvi = fixed_hvi + delta
        intensity = learned_hvi[:, 0:1].clamp(0.0, 1.0)
        chroma = learned_hvi[:, 1:]
        return torch.cat([intensity, chroma], dim=1)


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(num_groups=max(1, min(8, out_channels)), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class PyramidConditionAdapter(nn.Module):
    """
    Learnable HVI + dual-branch condition encoder + multi-scale FiLM state generator.

    Branch A: illumination-oriented features
    Branch B: reflectance / color / detail-oriented features
    """

    def __init__(
        self,
        block_channels,
        cond_out_channels: int = 7,
        base_channels: int = 32,
        use_retinex: bool = True,
        conditioning_space: str = "hvi_lite",
    ):
        super().__init__()
        self.block_channels = list(block_channels)
        self.cond_out_channels = cond_out_channels
        self.base_channels = base_channels
        self.use_retinex = use_retinex
        self.conditioning_space = conditioning_space
        self.hvi_transform = LearnableHVITransform()

        illumination_in_channels = 2 if use_retinex else 1
        detail_in_channels = 5

        self.illum_stem = ConvGNAct(illumination_in_channels, base_channels)
        self.illum_down1 = ConvGNAct(base_channels, base_channels * 2, stride=2)
        self.illum_down2 = ConvGNAct(base_channels * 2, base_channels * 4, stride=2)
        self.illum_down3 = ConvGNAct(base_channels * 4, base_channels * 8, stride=2)

        self.detail_stem = ConvGNAct(detail_in_channels, base_channels)
        self.detail_down1 = ConvGNAct(base_channels, base_channels * 2, stride=2)
        self.detail_down2 = ConvGNAct(base_channels * 2, base_channels * 4, stride=2)
        self.detail_down3 = ConvGNAct(base_channels * 4, base_channels * 8, stride=2)

        self.fuse0 = ConvGNAct(base_channels * 2, base_channels)
        self.fuse1 = ConvGNAct(base_channels * 4, base_channels * 2)
        self.fuse2 = ConvGNAct(base_channels * 8, base_channels * 4)
        self.fuse3 = ConvGNAct(base_channels * 16, base_channels * 8)

        self.noisy_proj0 = ConvGNAct(3, base_channels)
        self.noisy_proj1 = ConvGNAct(base_channels, base_channels * 2, stride=2)
        self.noisy_proj2 = ConvGNAct(base_channels * 2, base_channels * 4, stride=2)
        self.noisy_proj3 = ConvGNAct(base_channels * 4, base_channels * 8, stride=2)

        self.local_film0 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.local_film1 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.local_film2 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)
        self.local_film3 = nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1)

        fused_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.down_stage_map = [min(index, len(fused_channels) - 1) for index in range(len(self.block_channels))]
        self.up_stage_map = [max(0, len(fused_channels) - 1 - index) for index in range(len(self.block_channels))]
        up_block_channels = list(reversed(self.block_channels))

        self.down_film_layers = nn.ModuleList([
            nn.Conv2d(fused_channels[self.down_stage_map[index]], self.block_channels[index] * 2, kernel_size=1)
            for index in range(len(self.block_channels))
        ])
        self.mid_film = nn.Conv2d(fused_channels[-1], self.block_channels[-1] * 2, kernel_size=1)
        self.up_film_layers = nn.ModuleList([
            nn.Conv2d(fused_channels[self.up_stage_map[index]], up_block_channels[index] * 2, kernel_size=1)
            for index in range(len(up_block_channels))
        ])

        self.cond_merge = ConvGNAct(sum(fused_channels), base_channels)
        self.noisy_merge = ConvGNAct(sum(fused_channels), base_channels)
        self.condition_head = nn.Conv2d(base_channels, cond_out_channels, kernel_size=1)
        self.noisy_delta_head = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def build_condition_space(self, low_light_01: torch.Tensor, conditioning_space: str | None = None) -> torch.Tensor:
        mode = conditioning_space or self.conditioning_space
        if mode == "hvi_lite":
            return self.hvi_transform(low_light_01)
        return low_light_01

    @staticmethod
    def _film_modulation(noisy_feat: torch.Tensor, cond_feat: torch.Tensor, film_layer: nn.Module) -> torch.Tensor:
        gamma, beta = film_layer(cond_feat).chunk(2, dim=1)
        return noisy_feat * (1.0 + torch.tanh(gamma)) + beta

    def _build_dual_branch_pyramid(self, illumination_input: torch.Tensor, detail_input: torch.Tensor):
        illum0 = self.illum_stem(illumination_input)
        illum1 = self.illum_down1(illum0)
        illum2 = self.illum_down2(illum1)
        illum3 = self.illum_down3(illum2)

        detail0 = self.detail_stem(detail_input)
        detail1 = self.detail_down1(detail0)
        detail2 = self.detail_down2(detail1)
        detail3 = self.detail_down3(detail2)

        fused0 = self.fuse0(torch.cat([illum0, detail0], dim=1))
        fused1 = self.fuse1(torch.cat([illum1, detail1], dim=1))
        fused2 = self.fuse2(torch.cat([illum2, detail2], dim=1))
        fused3 = self.fuse3(torch.cat([illum3, detail3], dim=1))
        return [fused0, fused1, fused2, fused3]

    def _build_noisy_pyramid(self, noisy_images: torch.Tensor):
        noisy0 = self.noisy_proj0(noisy_images)
        noisy1 = self.noisy_proj1(noisy0)
        noisy2 = self.noisy_proj2(noisy1)
        noisy3 = self.noisy_proj3(noisy2)
        return [noisy0, noisy1, noisy2, noisy3]

    def forward(self, noisy_images: torch.Tensor, illumination_input: torch.Tensor, detail_input: torch.Tensor):
        fused_feats = self._build_dual_branch_pyramid(illumination_input, detail_input)
        noisy_feats = self._build_noisy_pyramid(noisy_images)

        mod0 = self._film_modulation(noisy_feats[0], fused_feats[0], self.local_film0)
        mod1 = self._film_modulation(noisy_feats[1], fused_feats[1], self.local_film1)
        mod2 = self._film_modulation(noisy_feats[2], fused_feats[2], self.local_film2)
        mod3 = self._film_modulation(noisy_feats[3], fused_feats[3], self.local_film3)

        target_size = noisy_images.shape[2:]
        merged_cond = torch.cat([
            fused_feats[0],
            F.interpolate(fused_feats[1], size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(fused_feats[2], size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(fused_feats[3], size=target_size, mode="bilinear", align_corners=False),
        ], dim=1)
        merged_noisy = torch.cat([
            mod0,
            F.interpolate(mod1, size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(mod2, size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(mod3, size=target_size, mode="bilinear", align_corners=False),
        ], dim=1)

        condition_map = self.condition_head(self.cond_merge(merged_cond))
        noisy_delta = 0.1 * torch.tanh(self.noisy_delta_head(self.noisy_merge(merged_noisy)))

        down_state = [
            projector(fused_feats[self.down_stage_map[index]])
            for index, projector in enumerate(self.down_film_layers)
        ]
        mid_state = self.mid_film(fused_feats[-1])
        up_state = [
            projector(fused_feats[self.up_stage_map[index]])
            for index, projector in enumerate(self.up_film_layers)
        ]

        return {
            "noisy_delta": noisy_delta,
            "condition_map": condition_map,
            "condition_pyramid": fused_feats,
            "film_state": {
                "down": down_state,
                "mid": mid_state,
                "up": up_state,
            },
        }
