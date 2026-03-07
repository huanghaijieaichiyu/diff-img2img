"""
Utility functions for the Diff-Img2Img project.
Contains: SSIM, SNR helpers, Charbonnier loss, and path utils.
"""
import os
import random
from math import exp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_random_seed(seed: int = 42, deterministic: bool = False, benchmark: bool = False) -> int:
    """
    Sets the random seed for reproducibility.

    Args:
        seed: The random seed. Defaults to 42.
        deterministic: Enforce deterministic CUDA behavior (slower).
        benchmark: Enable CUDA benchmarking (faster but less reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

    return seed


# ============================================================================
#  SSIM
# ============================================================================

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(
        channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.to(img1.device)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.to(img1.device)
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


# ============================================================================
#  Path Utils
# ============================================================================

def Save_path(path: str, model: str = 'train') -> str:
    """
    Generates a unique file path, incrementing a counter if the path exists.
    """
    file_path = os.path.join(path, model)
    i = 1
    while os.path.exists(file_path):
        file_path = os.path.join(path, f'{model}({i})')
        i += 1
    return file_path


# ============================================================================
#  SNR Helper Functions (for Min-SNR weighting)
# ============================================================================

def compute_snr(noise_scheduler, timesteps):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    if alphas_cumprod.device != timesteps.device:
        alphas_cumprod = alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
    return snr


def compute_min_snr_loss_weights(noise_scheduler, timesteps, snr_gamma=5.0):
    snr = compute_snr(noise_scheduler, timesteps)
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")

    if prediction_type == "v_prediction":
        min_snr = torch.stack([snr, torch.ones_like(snr) * snr_gamma], dim=1).min(dim=1)[0]
        weights = min_snr / (snr + 1)
    elif prediction_type == "epsilon":
        gamma_over_snr = snr_gamma / (snr + 1e-8)
        weights = torch.stack([torch.ones_like(gamma_over_snr), gamma_over_snr], dim=1).min(dim=1)[0]
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")
    return weights.detach()


def charbonnier_loss_elementwise(pred, target, eps=1e-3):
    """Element-wise Charbonnier Loss: sqrt((x-y)^2 + eps^2)"""
    return torch.sqrt((pred - target)**2 + eps**2)
