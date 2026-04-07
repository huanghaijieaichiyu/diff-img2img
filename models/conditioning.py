import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvGNAct, DWConvGNAct, GatedFusion, GlobalContextBlock, MBConvBlock, Transformer2DBlock, CrossAttentionBlock, NAFBlock, PooledGlobalContextBlock


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


class ConditionAdapterBase(nn.Module):
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

    def build_condition_space(self, low_light_01: torch.Tensor, conditioning_space: str | None = None) -> torch.Tensor:
        mode = conditioning_space or self.conditioning_space
        if mode == "hvi_lite":
            return self.hvi_transform(low_light_01)
        return low_light_01

    def _make_stage_maps(self, fused_channels):
        down_stage_map = [min(index, len(fused_channels) - 1) for index in range(len(self.block_channels))]
        up_stage_map = [max(0, len(fused_channels) - 1 - index) for index in range(len(self.block_channels))]
        return down_stage_map, up_stage_map

    @staticmethod
    def _film_modulation(noisy_feat: torch.Tensor, cond_feat: torch.Tensor, film_layer: nn.Module) -> torch.Tensor:
        gamma, beta = film_layer(cond_feat).chunk(2, dim=1)
        return noisy_feat * (1.0 + torch.tanh(gamma)) + beta

    def _pack_outputs(self, noisy_images, fused_feats, noisy_mod_feats, cond_merge, noisy_merge, down_film_layers, mid_film, up_film_layers, down_stage_map, up_stage_map):
        target_size = noisy_images.shape[2:]
        merged_cond = torch.cat(
            [F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False) if feat.shape[2:] != target_size else feat for feat in fused_feats],
            dim=1,
        )
        merged_noisy = torch.cat(
            [F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False) if feat.shape[2:] != target_size else feat for feat in noisy_mod_feats],
            dim=1,
        )

        condition_map = cond_merge(merged_cond)
        noisy_delta = noisy_merge(merged_noisy)

        down_state = [layer(fused_feats[down_stage_map[index]]) for index, layer in enumerate(down_film_layers)]
        up_state = [layer(fused_feats[up_stage_map[index]]) for index, layer in enumerate(up_film_layers)]

        return {
            "noisy_delta": noisy_delta,
            "condition_map": condition_map,
            "condition_pyramid": fused_feats,
            "film_state": {
                "down": down_state,
                "mid": mid_film(fused_feats[-1]),
                "up": up_state,
            },
        }


class SmallConditionAdapter(ConditionAdapterBase):
    """
    Parameter-efficient condition adapter.
    Uses MBConv / depthwise operations and fewer strong attention modules.
    """

    def __init__(self, block_channels, cond_out_channels=7, base_channels=24, use_retinex=True, conditioning_space="hvi_lite"):
        super().__init__(block_channels, cond_out_channels, base_channels, use_retinex, conditioning_space)
        illum_in = 2 if use_retinex else 1
        detail_in = 5

        self.illum_stem = DWConvGNAct(illum_in, base_channels, k=3, s=1)
        self.illum_down1 = MBConvBlock(base_channels, base_channels * 2, stride=2)
        self.illum_down2 = MBConvBlock(base_channels * 2, base_channels * 4, stride=2)

        self.detail_stem = DWConvGNAct(detail_in, base_channels, k=3, s=1)
        self.detail_down1 = MBConvBlock(base_channels, base_channels * 2, stride=2)
        self.detail_down2 = MBConvBlock(base_channels * 2, base_channels * 4, stride=2)

        self.fuse0 = GatedFusion(base_channels)
        self.fuse1 = GatedFusion(base_channels * 2)
        self.fuse2 = GatedFusion(base_channels * 4)

        self.noisy_proj0 = DWConvGNAct(3, base_channels, k=3, s=1)
        self.noisy_proj1 = MBConvBlock(base_channels, base_channels * 2, stride=2)
        self.noisy_proj2 = MBConvBlock(base_channels * 2, base_channels * 4, stride=2)

        self.local_film0 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=1)
        self.local_film1 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=1)
        self.local_film2 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=1)

        fused_channels = [base_channels, base_channels * 2, base_channels * 4]
        self.down_stage_map, self.up_stage_map = self._make_stage_maps(fused_channels)
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

        merged_channels = sum(fused_channels)
        self.cond_merge = nn.Sequential(
            ConvGNAct(merged_channels, base_channels, k=1, s=1),
            nn.Conv2d(base_channels, cond_out_channels, kernel_size=1),
        )
        self.noisy_merge = nn.Sequential(
            ConvGNAct(merged_channels, base_channels, k=1, s=1),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noisy_images: torch.Tensor, illumination_input: torch.Tensor, detail_input: torch.Tensor):
        illum0 = self.illum_stem(illumination_input)
        illum1 = self.illum_down1(illum0)
        illum2 = self.illum_down2(illum1)

        detail0 = self.detail_stem(detail_input)
        detail1 = self.detail_down1(detail0)
        detail2 = self.detail_down2(detail1)

        fused0 = self.fuse0(illum0, detail0)
        fused1 = self.fuse1(illum1, detail1)
        fused2 = self.fuse2(illum2, detail2)
        fused_feats = [fused0, fused1, fused2]

        noisy0 = self.noisy_proj0(noisy_images)
        noisy1 = self.noisy_proj1(noisy0)
        noisy2 = self.noisy_proj2(noisy1)

        mod0 = self._film_modulation(noisy0, fused0, self.local_film0)
        mod1 = self._film_modulation(noisy1, fused1, self.local_film1)
        mod2 = self._film_modulation(noisy2, fused2, self.local_film2)

        packed = self._pack_outputs(
            noisy_images,
            fused_feats,
            [mod0, mod1, mod2],
            self.cond_merge,
            self.noisy_merge,
            self.down_film_layers,
            self.mid_film,
            self.up_film_layers,
            self.down_stage_map,
            self.up_stage_map,
        )
        packed["noisy_delta"] = 0.08 * packed["noisy_delta"]
        return packed


class SmallConditionAdapterV2(ConditionAdapterBase):
    """
    HVI dual-path lightweight adapter with NAF local modeling and pooled global
    context at the deepest stage.
    """

    def __init__(self, block_channels, cond_out_channels=7, base_channels=24, use_retinex=True, conditioning_space="hvi_lite"):
        super().__init__(block_channels, cond_out_channels, base_channels, use_retinex, conditioning_space)
        illum_in = 2 if use_retinex else 1
        detail_in = 5

        self.illum_stem = DWConvGNAct(illum_in, base_channels, k=3, s=1)
        self.illum_refine0 = NAFBlock(base_channels)
        self.illum_down1 = MBConvBlock(base_channels, base_channels * 2, stride=2)
        self.illum_refine1 = NAFBlock(base_channels * 2)
        self.illum_down2 = MBConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.illum_refine2 = nn.Sequential(
            NAFBlock(base_channels * 4),
            PooledGlobalContextBlock(base_channels * 4, num_heads=4, pooled_size=8),
        )

        self.detail_stem = DWConvGNAct(detail_in, base_channels, k=3, s=1)
        self.detail_refine0 = NAFBlock(base_channels)
        self.detail_down1 = MBConvBlock(base_channels, base_channels * 2, stride=2)
        self.detail_refine1 = NAFBlock(base_channels * 2)
        self.detail_down2 = MBConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.detail_refine2 = nn.Sequential(
            NAFBlock(base_channels * 4),
            PooledGlobalContextBlock(base_channels * 4, num_heads=4, pooled_size=8),
        )

        self.fuse0 = GatedFusion(base_channels)
        self.fuse1 = GatedFusion(base_channels * 2)
        self.fuse2 = GatedFusion(base_channels * 4)

        self.noisy_proj0 = DWConvGNAct(3, base_channels, k=3, s=1)
        self.noisy_proj1 = MBConvBlock(base_channels, base_channels * 2, stride=2)
        self.noisy_proj2 = MBConvBlock(base_channels * 2, base_channels * 4, stride=2)

        self.local_film0 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=1)
        self.local_film1 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=1)
        self.local_film2 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=1)

        fused_channels = [base_channels, base_channels * 2, base_channels * 4]
        self.down_stage_map, self.up_stage_map = self._make_stage_maps(fused_channels)
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

        merged_channels = sum(fused_channels)
        self.cond_merge = nn.Sequential(
            ConvGNAct(merged_channels, base_channels, k=1, s=1),
            nn.Conv2d(base_channels, cond_out_channels, kernel_size=1),
        )
        self.noisy_merge = nn.Sequential(
            ConvGNAct(merged_channels, base_channels, k=1, s=1),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noisy_images: torch.Tensor, illumination_input: torch.Tensor, detail_input: torch.Tensor):
        illum0 = self.illum_refine0(self.illum_stem(illumination_input))
        illum1 = self.illum_refine1(self.illum_down1(illum0))
        illum2 = self.illum_refine2(self.illum_down2(illum1))

        detail0 = self.detail_refine0(self.detail_stem(detail_input))
        detail1 = self.detail_refine1(self.detail_down1(detail0))
        detail2 = self.detail_refine2(self.detail_down2(detail1))

        fused0 = self.fuse0(illum0, detail0)
        fused1 = self.fuse1(illum1, detail1)
        fused2 = self.fuse2(illum2, detail2)
        fused_feats = [fused0, fused1, fused2]

        noisy0 = self.noisy_proj0(noisy_images)
        noisy1 = self.noisy_proj1(noisy0)
        noisy2 = self.noisy_proj2(noisy1)

        mod0 = self._film_modulation(noisy0, fused0, self.local_film0)
        mod1 = self._film_modulation(noisy1, fused1, self.local_film1)
        mod2 = self._film_modulation(noisy2, fused2, self.local_film2)

        packed = self._pack_outputs(
            noisy_images,
            fused_feats,
            [mod0, mod1, mod2],
            self.cond_merge,
            self.noisy_merge,
            self.down_film_layers,
            self.mid_film,
            self.up_film_layers,
            self.down_stage_map,
            self.up_stage_map,
        )
        packed["noisy_delta"] = 0.07 * packed["noisy_delta"]
        return packed


class MiddleConditionAdapter(ConditionAdapterBase):
    """
    Balanced dual-branch adapter.
    Similar spirit to CIDNet-style decoupling, but still lightweight.
    """

    def __init__(self, block_channels, cond_out_channels=7, base_channels=32, use_retinex=True, conditioning_space="hvi_lite"):
        super().__init__(block_channels, cond_out_channels, base_channels, use_retinex, conditioning_space)
        illum_in = 2 if use_retinex else 1
        detail_in = 5

        self.illum_stem = ConvGNAct(illum_in, base_channels)
        self.illum_down1 = ConvGNAct(base_channels, base_channels * 2, s=2)
        self.illum_down2 = ConvGNAct(base_channels * 2, base_channels * 4, s=2)
        self.illum_down3 = ConvGNAct(base_channels * 4, base_channels * 8, s=2)

        self.detail_stem = ConvGNAct(detail_in, base_channels)
        self.detail_down1 = ConvGNAct(base_channels, base_channels * 2, s=2)
        self.detail_down2 = ConvGNAct(base_channels * 2, base_channels * 4, s=2)
        self.detail_down3 = ConvGNAct(base_channels * 4, base_channels * 8, s=2)

        self.fuse0 = ConvGNAct(base_channels * 2, base_channels)
        self.fuse1 = ConvGNAct(base_channels * 4, base_channels * 2)
        self.fuse2 = ConvGNAct(base_channels * 8, base_channels * 4)
        self.fuse3 = ConvGNAct(base_channels * 16, base_channels * 8)

        self.noisy_proj0 = ConvGNAct(3, base_channels)
        self.noisy_proj1 = ConvGNAct(base_channels, base_channels * 2, s=2)
        self.noisy_proj2 = ConvGNAct(base_channels * 2, base_channels * 4, s=2)
        self.noisy_proj3 = ConvGNAct(base_channels * 4, base_channels * 8, s=2)

        self.local_film0 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.local_film1 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.local_film2 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)
        self.local_film3 = nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1)

        fused_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.down_stage_map, self.up_stage_map = self._make_stage_maps(fused_channels)
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

        merged_channels = sum(fused_channels)
        self.cond_merge = nn.Sequential(
            ConvGNAct(merged_channels, base_channels, k=1, s=1),
            nn.Conv2d(base_channels, cond_out_channels, kernel_size=1),
        )
        self.noisy_merge = nn.Sequential(
            ConvGNAct(merged_channels, base_channels, k=1, s=1),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noisy_images: torch.Tensor, illumination_input: torch.Tensor, detail_input: torch.Tensor):
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
        fused_feats = [fused0, fused1, fused2, fused3]

        noisy0 = self.noisy_proj0(noisy_images)
        noisy1 = self.noisy_proj1(noisy0)
        noisy2 = self.noisy_proj2(noisy1)
        noisy3 = self.noisy_proj3(noisy2)

        mod0 = self._film_modulation(noisy0, fused0, self.local_film0)
        mod1 = self._film_modulation(noisy1, fused1, self.local_film1)
        mod2 = self._film_modulation(noisy2, fused2, self.local_film2)
        mod3 = self._film_modulation(noisy3, fused3, self.local_film3)

        packed = self._pack_outputs(
            noisy_images,
            fused_feats,
            [mod0, mod1, mod2, mod3],
            self.cond_merge,
            self.noisy_merge,
            self.down_film_layers,
            self.mid_film,
            self.up_film_layers,
            self.down_stage_map,
            self.up_stage_map,
        )
        packed["noisy_delta"] = 0.1 * packed["noisy_delta"]
        return packed


class MaxConditionAdapter(ConditionAdapterBase):
    """
    Quality-oriented adapter.
    Adds transformer/global refinement on top of the balanced decoupled branches.
    """

    def __init__(self, block_channels, cond_out_channels=7, base_channels=48, use_retinex=True, conditioning_space="hvi_lite"):
        super().__init__(block_channels, cond_out_channels, base_channels, use_retinex, conditioning_space)
        illum_in = 2 if use_retinex else 1
        detail_in = 5

        self.illum_stem = ConvGNAct(illum_in, base_channels)
        self.illum_down1 = ConvGNAct(base_channels, base_channels * 2, s=2)
        self.illum_down2 = ConvGNAct(base_channels * 2, base_channels * 4, s=2)
        self.illum_down3 = ConvGNAct(base_channels * 4, base_channels * 8, s=2)

        self.detail_stem = ConvGNAct(detail_in, base_channels)
        self.detail_down1 = ConvGNAct(base_channels, base_channels * 2, s=2)
        self.detail_down2 = ConvGNAct(base_channels * 2, base_channels * 4, s=2)
        self.detail_down3 = ConvGNAct(base_channels * 4, base_channels * 8, s=2)

        self.fuse0 = GatedFusion(base_channels)
        self.fuse1 = GatedFusion(base_channels * 2)
        self.fuse2 = GatedFusion(base_channels * 4)
        self.fuse3 = GatedFusion(base_channels * 8)

        self.refine2 = Transformer2DBlock(base_channels * 4, num_heads=8)
        self.refine3 = nn.Sequential(
            GlobalContextBlock(base_channels * 8, num_heads=8),
            Transformer2DBlock(base_channels * 8, num_heads=8),
        )

        self.noisy_proj0 = ConvGNAct(3, base_channels)
        self.noisy_proj1 = ConvGNAct(base_channels, base_channels * 2, s=2)
        self.noisy_proj2 = ConvGNAct(base_channels * 2, base_channels * 4, s=2)
        self.noisy_proj3 = ConvGNAct(base_channels * 4, base_channels * 8, s=2)

        self.local_film0 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.local_film1 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.local_film2 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)
        self.local_film3 = nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1)

        fused_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.down_stage_map, self.up_stage_map = self._make_stage_maps(fused_channels)
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

        merged_channels = sum(fused_channels)
        self.merge_refiner = Transformer2DBlock(base_channels, num_heads=4)
        self.cond_merge = nn.Sequential(
            ConvGNAct(merged_channels, base_channels, k=1, s=1),
            self.merge_refiner,
            nn.Conv2d(base_channels, cond_out_channels, kernel_size=1),
        )
        self.noisy_merge = nn.Sequential(
            ConvGNAct(merged_channels, base_channels, k=1, s=1),
            Transformer2DBlock(base_channels, num_heads=4),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noisy_images: torch.Tensor, illumination_input: torch.Tensor, detail_input: torch.Tensor):
        illum0 = self.illum_stem(illumination_input)
        illum1 = self.illum_down1(illum0)
        illum2 = self.illum_down2(illum1)
        illum3 = self.illum_down3(illum2)

        detail0 = self.detail_stem(detail_input)
        detail1 = self.detail_down1(detail0)
        detail2 = self.detail_down2(detail1)
        detail3 = self.detail_down3(detail2)

        fused0 = self.fuse0(illum0, detail0)
        fused1 = self.fuse1(illum1, detail1)
        fused2 = self.refine2(self.fuse2(illum2, detail2))
        fused3 = self.refine3(self.fuse3(illum3, detail3))
        fused_feats = [fused0, fused1, fused2, fused3]

        noisy0 = self.noisy_proj0(noisy_images)
        noisy1 = self.noisy_proj1(noisy0)
        noisy2 = self.noisy_proj2(noisy1)
        noisy3 = self.noisy_proj3(noisy2)

        mod0 = self._film_modulation(noisy0, fused0, self.local_film0)
        mod1 = self._film_modulation(noisy1, fused1, self.local_film1)
        mod2 = self._film_modulation(noisy2, fused2, self.local_film2)
        mod3 = self._film_modulation(noisy3, fused3, self.local_film3)

        packed = self._pack_outputs(
            noisy_images,
            fused_feats,
            [mod0, mod1, mod2, mod3],
            self.cond_merge,
            self.noisy_merge,
            self.down_film_layers,
            self.mid_film,
            self.up_film_layers,
            self.down_stage_map,
            self.up_stage_map,
        )
        packed["noisy_delta"] = 0.1 * packed["noisy_delta"]
        return packed


class MaxConditionAdapterV2(ConditionAdapterBase):
    """
    Deep-only cross-attention condition adapter.
    Keeps global cross-attention confined to the deepest stage to avoid the
    64x64 memory cost of full attention on 8GB-class training.
    """

    def __init__(self, block_channels, cond_out_channels=7, base_channels=48, use_retinex=True, conditioning_space="hvi_lite"):
        super().__init__(block_channels, cond_out_channels, base_channels, use_retinex, conditioning_space)
        illum_in = 2 if use_retinex else 1
        detail_in = 5

        self.illum_stem = ConvGNAct(illum_in, base_channels)
        self.illum_down1 = ConvGNAct(base_channels, base_channels * 2, s=2)
        self.illum_down2 = ConvGNAct(base_channels * 2, base_channels * 4, s=2)
        self.illum_down3 = ConvGNAct(base_channels * 4, base_channels * 8, s=2)

        self.detail_stem = ConvGNAct(detail_in, base_channels)
        self.detail_down1 = ConvGNAct(base_channels, base_channels * 2, s=2)
        self.detail_down2 = ConvGNAct(base_channels * 2, base_channels * 4, s=2)
        self.detail_down3 = ConvGNAct(base_channels * 4, base_channels * 8, s=2)

        self.fuse0 = GatedFusion(base_channels)
        self.fuse1 = GatedFusion(base_channels * 2)
        self.fuse2 = GatedFusion(base_channels * 4)
        self.fuse3 = GatedFusion(base_channels * 8)

        # Only the deepest stage uses cross-attention to keep the adapter
        # practical for 8GB-class training while still strengthening context use.
        self.cross_attn3 = CrossAttentionBlock(base_channels * 8, base_channels * 8, num_heads=8)

        self.noisy_proj0 = ConvGNAct(3, base_channels)
        self.noisy_proj1 = ConvGNAct(base_channels, base_channels * 2, s=2)
        self.noisy_proj2 = ConvGNAct(base_channels * 2, base_channels * 4, s=2)
        self.noisy_proj3 = ConvGNAct(base_channels * 4, base_channels * 8, s=2)

        self.local_film0 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.local_film1 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.local_film2 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)
        self.local_film3 = nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1)

        fused_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.down_stage_map, self.up_stage_map = self._make_stage_maps(fused_channels)
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

        merged_channels = sum(fused_channels)
        self.cond_merge = nn.Sequential(
            ConvGNAct(merged_channels, base_channels, k=1, s=1),
            nn.Conv2d(base_channels, cond_out_channels, kernel_size=1),
        )
        self.noisy_merge = nn.Sequential(
            ConvGNAct(merged_channels, base_channels, k=1, s=1),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noisy_images: torch.Tensor, illumination_input: torch.Tensor, detail_input: torch.Tensor):
        illum0 = self.illum_stem(illumination_input)
        illum1 = self.illum_down1(illum0)
        illum2 = self.illum_down2(illum1)
        illum3 = self.illum_down3(illum2)

        detail0 = self.detail_stem(detail_input)
        detail1 = self.detail_down1(detail0)
        detail2 = self.detail_down2(detail1)
        detail3 = self.detail_down3(detail2)

        fused0 = self.fuse0(illum0, detail0)
        fused1 = self.fuse1(illum1, detail1)
        fused2 = self.fuse2(illum2, detail2)
        fused3 = self.fuse3(illum3, detail3)

        noisy0 = self.noisy_proj0(noisy_images)
        noisy1 = self.noisy_proj1(noisy0)
        noisy2 = self.noisy_proj2(noisy1)
        noisy3 = self.noisy_proj3(noisy2)

        noisy3_attn = self.cross_attn3(noisy3, fused3)

        mod0 = self._film_modulation(noisy0, fused0, self.local_film0)
        mod1 = self._film_modulation(noisy1, fused1, self.local_film1)
        mod2 = self._film_modulation(noisy2, fused2, self.local_film2)
        mod3 = self._film_modulation(noisy3_attn, fused3, self.local_film3)

        fused_feats = [fused0, fused1, fused2, fused3]
        packed = self._pack_outputs(
            noisy_images,
            fused_feats,
            [mod0, mod1, mod2, mod3],
            self.cond_merge,
            self.noisy_merge,
            self.down_film_layers,
            self.mid_film,
            self.up_film_layers,
            self.down_stage_map,
            self.up_stage_map,
        )
        packed["noisy_delta"] = 0.1 * packed["noisy_delta"]
        return packed


def build_condition_adapter(variant: str, block_channels, cond_out_channels=7, base_channels=32, use_retinex=True, conditioning_space="hvi_lite"):
    variant = (variant or "middle").lower()
    if variant in {"small", "mobile", "lite"}:
        return SmallConditionAdapter(
            block_channels=block_channels,
            cond_out_channels=cond_out_channels,
            base_channels=base_channels,
            use_retinex=use_retinex,
            conditioning_space=conditioning_space,
        )
    if variant in {"small_v2", "efficient", "lite_v2"}:
        return SmallConditionAdapterV2(
            block_channels=block_channels,
            cond_out_channels=cond_out_channels,
            base_channels=base_channels,
            use_retinex=use_retinex,
            conditioning_space=conditioning_space,
        )
    if variant in {"max", "quality", "large"}:
        return MaxConditionAdapter(
            block_channels=block_channels,
            cond_out_channels=cond_out_channels,
            base_channels=base_channels,
            use_retinex=use_retinex,
            conditioning_space=conditioning_space,
        )
    if variant in {"max_v2", "cross_attn"}:
        return MaxConditionAdapterV2(
            block_channels=block_channels,
            cond_out_channels=cond_out_channels,
            base_channels=base_channels,
            use_retinex=use_retinex,
            conditioning_space=conditioning_space,
        )
    return MiddleConditionAdapter(
        block_channels=block_channels,
        cond_out_channels=cond_out_channels,
        base_channels=base_channels,
        use_retinex=use_retinex,
        conditioning_space=conditioning_space,
    )


PyramidConditionAdapter = MiddleConditionAdapter
