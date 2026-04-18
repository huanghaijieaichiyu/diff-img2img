from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F

try:
    from torchvision.io import decode_jpeg, encode_jpeg
except Exception:  # pragma: no cover - optional runtime capability
    decode_jpeg = None
    encode_jpeg = None


class TorchLowLightDegrader:
    """
    Torch-native low-light degradation pipeline.

    Input / output are RGB tensors in [0, 1], shape (C, H, W).
    The parameter schema is compatible with scripts.darker.Darker ranges.
    """

    DEFAULT_RANGES = {
        "gamma": (1.5, 4.0),
        "linear_attenuation": (0.25, 0.7),
        "saturation_factor": (0.4, 0.85),
        "color_shift_factor": (0.0, 0.12),
        "headlight_boost": (0.0, 0.9),
        "noise_k": (0.005, 0.04),
        "noise_sigma_read": (2.0, 15.0),
        "vignette_prob": 0.5,
        "vignette_strength": (0.2, 0.6),
        "motion_blur_prob": 0.2,
        "motion_blur_kernel": (3, 9),
        "jpeg_artifact_prob": 0.3,
        "jpeg_quality": (40, 85),
    }

    def __init__(self, randomize: bool = True, param_ranges: dict[str, Any] | None = None):
        self.randomize = randomize
        self.ranges = dict(self.DEFAULT_RANGES)
        if param_ranges:
            self.ranges.update(param_ranges)

    def _sample_params(self) -> dict[str, Any]:
        r = self.ranges
        kernel_min, kernel_max = r["motion_blur_kernel"]
        kernel_choices = [k for k in range(int(kernel_min), int(kernel_max) + 1) if k % 2 == 1]
        return {
            "gamma": random.uniform(*r["gamma"]),
            "linear_attenuation": random.uniform(*r["linear_attenuation"]),
            "saturation_factor": random.uniform(*r["saturation_factor"]),
            "color_shift_factor": random.uniform(*r["color_shift_factor"]),
            "headlight_boost": random.uniform(*r["headlight_boost"]),
            "noise_k": random.uniform(*r["noise_k"]),
            "noise_sigma_read": random.uniform(*r["noise_sigma_read"]),
            "use_vignette": random.random() < r["vignette_prob"],
            "vignette_strength": random.uniform(*r["vignette_strength"]),
            "use_motion_blur": random.random() < r["motion_blur_prob"],
            "motion_blur_kernel": random.choice(kernel_choices),
            "motion_blur_angle": random.uniform(0.0, 180.0),
            "use_jpeg": random.random() < r["jpeg_artifact_prob"],
            "jpeg_quality": random.randint(*r["jpeg_quality"]),
        }

    @staticmethod
    def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
        # image in [0,1], shape (3,H,W)
        r, g, b = image[0], image[1], image[2]
        maxc = torch.max(image, dim=0).values
        minc = torch.min(image, dim=0).values
        delta = maxc - minc

        v = maxc
        s = torch.where(maxc > 0.0, delta / (maxc + 1e-8), torch.zeros_like(maxc))

        h = torch.zeros_like(maxc)
        valid = delta > 1e-8
        rc = (maxc - r) / (delta + 1e-8)
        gc = (maxc - g) / (delta + 1e-8)
        bc = (maxc - b) / (delta + 1e-8)

        h = torch.where(valid & (r == maxc), bc - gc, h)
        h = torch.where(valid & (g == maxc), 2.0 + rc - bc, h)
        h = torch.where(valid & (b == maxc), 4.0 + gc - rc, h)
        h = (h / 6.0) % 1.0
        h = torch.where(valid, h, torch.zeros_like(h))

        return torch.stack([h, s, v], dim=0)

    @staticmethod
    def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
        h, s, v = hsv[0], hsv[1], hsv[2]
        h6 = (h % 1.0) * 6.0
        i = torch.floor(h6).to(torch.int64) % 6
        f = h6 - torch.floor(h6)

        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        r = torch.zeros_like(v)
        g = torch.zeros_like(v)
        b = torch.zeros_like(v)

        r = torch.where(i == 0, v, r)
        g = torch.where(i == 0, t, g)
        b = torch.where(i == 0, p, b)

        r = torch.where(i == 1, q, r)
        g = torch.where(i == 1, v, g)
        b = torch.where(i == 1, p, b)

        r = torch.where(i == 2, p, r)
        g = torch.where(i == 2, v, g)
        b = torch.where(i == 2, t, b)

        r = torch.where(i == 3, p, r)
        g = torch.where(i == 3, q, g)
        b = torch.where(i == 3, v, b)

        r = torch.where(i == 4, t, r)
        g = torch.where(i == 4, p, g)
        b = torch.where(i == 4, v, b)

        r = torch.where(i == 5, v, r)
        g = torch.where(i == 5, p, g)
        b = torch.where(i == 5, q, b)

        return torch.stack([r, g, b], dim=0)

    @staticmethod
    def _gaussian_kernel_1d(ksize: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ksize = max(3, int(ksize) | 1)
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1.0) + 0.8
        coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) * 0.5
        kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma + 1e-8))
        return kernel / kernel.sum().clamp_min(1e-8)

    @staticmethod
    def _headlight_mask(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        area_ratio = min(1.0, (height * width) / float(512 * 512))
        if area_ratio < 0.35:
            max_lights = 1
        elif area_ratio < 0.75:
            max_lights = 2
        else:
            max_lights = 4
        num_lights = random.randint(0, max_lights)
        if num_lights == 0:
            return torch.zeros((height, width), device=device, dtype=dtype)

        yy = torch.arange(height, device=device, dtype=dtype).view(height, 1)
        xx = torch.arange(width, device=device, dtype=dtype).view(1, width)
        mask = torch.zeros((height, width), device=device, dtype=dtype)

        for _ in range(num_lights):
            cx = random.uniform(0.1, 0.9) * width
            cy = random.uniform(0.4, 0.95) * height
            rx = random.uniform(0.08, 0.35) * width
            ry = random.uniform(0.06, 0.25) * height
            sharpness = random.uniform(0.8, 3.5)
            intensity = random.uniform(0.2, 1.0)

            dist = ((xx - cx) / max(rx, 1e-4)) ** 2 + ((yy - cy) / max(ry, 1e-4)) ** 2
            light = torch.exp(-dist * sharpness) * intensity
            mask = torch.maximum(mask, light)

        kx = max(3, int(width * 0.04) | 1)
        ky = max(3, int(height * 0.04) | 1)
        kernel_x = TorchLowLightDegrader._gaussian_kernel_1d(kx, device, dtype).view(1, 1, 1, kx)
        kernel_y = TorchLowLightDegrader._gaussian_kernel_1d(ky, device, dtype).view(1, 1, ky, 1)
        blurred = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel_x, padding=(0, kx // 2))
        blurred = F.conv2d(blurred, kernel_y, padding=(ky // 2, 0))
        return blurred.squeeze(0).squeeze(0).clamp(0.0, 1.0)

    @staticmethod
    def _apply_motion_blur(image: torch.Tensor, kernel_size: int, angle_deg: float) -> torch.Tensor:
        kernel_size = max(3, int(kernel_size) | 1)
        if kernel_size <= 1:
            return image

        base = torch.zeros((kernel_size, kernel_size), device=image.device, dtype=image.dtype)
        base[kernel_size // 2, :] = 1.0
        theta = float(angle_deg) * 3.141592653589793 / 180.0
        rot = torch.tensor(
            [[torch.cos(torch.tensor(theta, device=image.device, dtype=image.dtype)),
              -torch.sin(torch.tensor(theta, device=image.device, dtype=image.dtype)),
              0.0],
             [torch.sin(torch.tensor(theta, device=image.device, dtype=image.dtype)),
              torch.cos(torch.tensor(theta, device=image.device, dtype=image.dtype)),
              0.0]],
            device=image.device,
            dtype=image.dtype,
        ).unsqueeze(0)
        grid = F.affine_grid(rot, size=(1, 1, kernel_size, kernel_size), align_corners=False)
        rotated = F.grid_sample(
            base.unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        kernel = (rotated / rotated.sum().clamp_min(1e-8)).view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(image.shape[0], 1, 1, 1)
        blurred = F.conv2d(image.unsqueeze(0), kernel, padding=kernel_size // 2, groups=image.shape[0])
        return blurred.squeeze(0).clamp(0.0, 1.0)

    @staticmethod
    def _apply_vignette(image: torch.Tensor, strength: float) -> torch.Tensor:
        _, h, w = image.shape
        yy = torch.arange(h, device=image.device, dtype=image.dtype).view(h, 1)
        xx = torch.arange(w, device=image.device, dtype=image.dtype).view(1, w)
        cy = h / 2.0
        cx = w / 2.0
        max_dist = max((cx * cx + cy * cy) ** 0.5, 1e-6)
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_dist
        vignette = (1.0 - float(strength) * (dist ** 2)).clamp(0.0, 1.0)
        return (image * vignette.unsqueeze(0)).clamp(0.0, 1.0)

    @staticmethod
    def _apply_jpeg_artifact(image: torch.Tensor, quality: int) -> torch.Tensor:
        if encode_jpeg is None or decode_jpeg is None:
            return image
        try:
            original_device = image.device
            image_u8 = image.mul(255.0).clamp(0, 255).to(torch.uint8).cpu()
            encoded = encode_jpeg(image_u8, quality=int(quality))
            decoded = decode_jpeg(encoded).to(dtype=image.dtype)
            return decoded.div(255.0).to(device=original_device).clamp(0.0, 1.0)
        except Exception:
            return image

    @staticmethod
    def _enforce_exposure(source: torch.Tensor, degraded: torch.Tensor, min_ratio: float = 0.10, max_ratio: float = 0.55) -> torch.Tensor:
        source_gray = 0.299 * source[0] + 0.587 * source[1] + 0.114 * source[2]
        degraded_gray = 0.299 * degraded[0] + 0.587 * degraded[1] + 0.114 * degraded[2]
        source_mean = source_gray.mean().clamp_min(1e-6)
        degraded_mean = degraded_gray.mean()
        ratio = degraded_mean / source_mean

        adjusted = degraded
        if ratio < min_ratio:
            alpha = float(torch.clamp((min_ratio - ratio) / max(min_ratio, 1e-6), min=0.0, max=1.0).item())
            adjusted = adjusted * (1.0 - alpha) + source * alpha * 0.35
        elif ratio > max_ratio:
            scale = float((max_ratio / ratio.clamp_min(1e-6)).item())
            adjusted = adjusted * scale
        return adjusted.clamp(0.0, 1.0)

    def degrade(self, image_rgb: torch.Tensor, params: dict[str, Any] | None = None) -> torch.Tensor:
        """
        Args:
            image_rgb: Tensor in [0, 1], shape (3, H, W)
            params: Optional sampled params dict. If omitted and randomize=True, sample per call.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[0] != 3:
            raise ValueError(f"Expected RGB tensor with shape (3, H, W), got {tuple(image_rgb.shape)}")

        image = image_rgb.float().clamp(0.0, 1.0)
        params = params or (self._sample_params() if self.randomize else {
            "gamma": 2.5,
            "linear_attenuation": 0.5,
            "saturation_factor": 0.6,
            "color_shift_factor": 0.1,
            "headlight_boost": 0.5,
            "noise_k": 0.02,
            "noise_sigma_read": 5.0,
            "use_vignette": False,
            "vignette_strength": 0.0,
            "use_motion_blur": False,
            "motion_blur_kernel": 5,
            "motion_blur_angle": 0.0,
            "use_jpeg": False,
            "jpeg_quality": 70,
        })

        _, h, w = image.shape
        mask = self._headlight_mask(h, w, image.device, image.dtype)

        gamma = float(params["gamma"])
        attenuation = float(params["linear_attenuation"])
        hsv = self._rgb_to_hsv(image)
        v = hsv[2]
        v_dark = (v ** gamma) * attenuation
        boost = float(params["headlight_boost"])
        v_final = v_dark * (1.0 - mask * boost) + v * (mask * boost)
        hsv[2] = v_final.clamp(0.0, 1.0)
        hsv[1] = (hsv[1] * float(params["saturation_factor"])).clamp(0.0, 1.0)

        adjusted = self._hsv_to_rgb(hsv).clamp(0.0, 1.0)

        shift_factor = float(params["color_shift_factor"])
        local_darkness = (1.0 - v_final).clamp(0.0, 1.0)
        shift_map = shift_factor * (1.0 - mask) * local_darkness
        adjusted[2] = (adjusted[2] + shift_map).clamp(0.0, 1.0)       # blue+
        adjusted[0] = (adjusted[0] - shift_map * 0.5).clamp(0.0, 1.0)  # red-

        if bool(params.get("use_vignette", False)):
            adjusted = self._apply_vignette(adjusted, float(params.get("vignette_strength", 0.4)))

        if bool(params.get("use_motion_blur", False)):
            adjusted = self._apply_motion_blur(
                adjusted,
                int(params.get("motion_blur_kernel", 5)),
                float(params.get("motion_blur_angle", 0.0)),
            )

        adjusted_255 = adjusted * 255.0
        noise_k = max(0.0, float(params["noise_k"]))
        if noise_k > 0:
            shot = torch.poisson((adjusted_255 * noise_k).clamp_min(0.0)) / noise_k - adjusted_255
        else:
            shot = torch.zeros_like(adjusted_255)

        sigma_read = float(params["noise_sigma_read"])
        channel_sigma = sigma_read * (0.8 + 0.4 * torch.rand((3, 1, 1), device=adjusted.device, dtype=adjusted.dtype))
        read = torch.randn_like(adjusted_255) * channel_sigma
        degraded_255 = (adjusted_255 + shot + read).clamp(0.0, 255.0)

        degraded = degraded_255 / 255.0

        if bool(params.get("use_jpeg", False)):
            degraded = self._apply_jpeg_artifact(degraded, int(params.get("jpeg_quality", 70)))

        return self._enforce_exposure(image, degraded)
