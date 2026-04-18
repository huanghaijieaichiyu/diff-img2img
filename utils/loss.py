import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_if_needed(image: torch.Tensor, target_size: int | None) -> torch.Tensor:
    if target_size is None:
        return image
    if image.shape[-1] <= target_size and image.shape[-2] <= target_size:
        return image
    return F.interpolate(image, size=(target_size, target_size), mode="bilinear", align_corners=False)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 approximation) for robust regression.
    Formula: sqrt(x^2 + eps^2)
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y, reduction="mean"):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if reduction == "none":
            return loss.mean(dim=(1, 2, 3))
        return torch.mean(loss)


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) Loss.
    """
    def __init__(self, window_size=11, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([torch.exp(-(torch.tensor(x) - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2, reduction="mean"):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.to(img1.device)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        per_sample = 1 - ssim_map.mean(dim=(1, 2, 3))
        if reduction == "none":
            return per_sample
        return per_sample.mean()


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) Loss.
    Uses a pre-trained VGG network to measure perceptual distance.
    
    This provides much better visual quality than pixel-only losses,
    as it captures high-level texture and structural features.
    """
    def __init__(self, net='vgg', enabled=True, resize_to=None):
        super(LPIPSLoss, self).__init__()
        self.resize_to = resize_to
        if not enabled:
            self.available = False
            self.loss_fn = None
            return
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net=net, verbose=False)
            # Freeze LPIPS network
            for param in self.loss_fn.parameters():
                param.requires_grad = False
            self.available = True
        except ImportError:
            print("Warning: lpips package not found. LPIPS loss disabled.")
            print("Install with: pip install lpips")
            self.available = False
    
    def forward(self, pred, target, reduction="mean"):
        """
        Args:
            pred, target: Images in [-1, 1] range, shape (B, C, H, W)
        Returns:
            Scalar LPIPS loss
        """
        if not self.available:
            zero = torch.tensor(0.0, device=pred.device)
            if reduction == "none":
                return zero.expand(pred.shape[0])
            return zero
        pred = _resize_if_needed(pred, self.resize_to)
        target = _resize_if_needed(target, self.resize_to)
        per_sample = self.loss_fn(pred, target).view(pred.shape[0], -1).mean(dim=1)
        if reduction == "none":
            return per_sample
        return per_sample.mean()


class HaarWaveletLoss(nn.Module):
    """
    Cheap detail-preserving loss using fixed Haar wavelet filters.
    Only the high-frequency bands are supervised.
    """

    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [-1.0, -1.0]],
                [[1.0, -1.0], [1.0, -1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ],
            dtype=torch.float32,
        ) / 2.0
        self.register_buffer("kernel", kernel.unsqueeze(1))

    def forward(self, pred, target, reduction="mean"):
        channels = pred.shape[1]
        weight = self.kernel.repeat(channels, 1, 1, 1)
        pred_coeffs = F.conv2d(pred, weight, stride=2, padding=0, groups=channels)
        target_coeffs = F.conv2d(target, weight, stride=2, padding=0, groups=channels)
        pred_coeffs = pred_coeffs.view(pred.shape[0], channels, 4, pred_coeffs.shape[-2], pred_coeffs.shape[-1])
        target_coeffs = target_coeffs.view(target.shape[0], channels, 4, target_coeffs.shape[-2], target_coeffs.shape[-1])
        per_sample = torch.abs(pred_coeffs[:, :, 1:] - target_coeffs[:, :, 1:]).mean(dim=(1, 2, 3, 4))
        if reduction == "none":
            return per_sample
        return per_sample.mean()


class FrequencyDomainLoss(nn.Module):
    """
    Frequency-domain reconstruction loss.
    Uses a log-magnitude spectrum comparison to emphasize high-frequency detail.
    """

    def __init__(self, resize_to: int | None = None):
        super().__init__()
        self.resize_to = resize_to

    def forward(self, pred, target, reduction="mean"):
        pred = _resize_if_needed(pred.float(), self.resize_to)
        target = _resize_if_needed(target.float(), self.resize_to)

        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        pred_mag = torch.log1p(torch.abs(pred_fft))
        target_mag = torch.log1p(torch.abs(target_fft))
        per_sample = torch.abs(pred_mag - target_mag).mean(dim=(1, 2, 3))
        if reduction == "none":
            return per_sample
        return per_sample.mean()


class EdgeLoss(nn.Module):
    """
    Edge-aware gradient loss using fixed Sobel filters.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        self.register_buffer("sobel_kernel", torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1))

    def forward(self, pred, target, reduction="mean"):
        pred = pred.float()
        target = target.float()
        channels = pred.shape[1]
        weight = self.sobel_kernel.repeat(channels, 1, 1, 1)
        pred_grad = F.conv2d(pred, weight, padding=1, groups=channels)
        target_grad = F.conv2d(target, weight, padding=1, groups=channels)
        pred_grad = pred_grad.view(pred.shape[0], channels, 2, pred_grad.shape[-2], pred_grad.shape[-1])
        target_grad = target_grad.view(target.shape[0], channels, 2, target_grad.shape[-2], target_grad.shape[-1])
        pred_mag = torch.sqrt(pred_grad[:, :, 0] ** 2 + pred_grad[:, :, 1] ** 2 + 1e-6)
        target_mag = torch.sqrt(target_grad[:, :, 0] ** 2 + target_grad[:, :, 1] ** 2 + 1e-6)
        per_sample = torch.abs(pred_mag - target_mag).mean(dim=(1, 2, 3))
        if reduction == "none":
            return per_sample
        return per_sample.mean()


class EMAWeightedLossBalancer(nn.Module):
    """
    Dynamic loss balancer inspired by modern multi-loss training practice.
    Keeps a running EMA of each loss magnitude and rescales terms to keep them comparable.
    """

    def __init__(self, num_losses: int, decay: float = 0.98, min_scale: float = 0.25, max_scale: float = 4.0):
        super().__init__()
        self.decay = decay
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.register_buffer("loss_ema", torch.ones(num_losses))
        self.register_buffer("step_count", torch.zeros((), dtype=torch.long))

    def forward(self, losses):
        detached_losses = torch.stack([loss.detach().float() for loss in losses])
        if self.step_count.item() == 0:
            self.loss_ema.copy_(detached_losses.clamp_min(1e-6))
        else:
            self.loss_ema.mul_(self.decay).add_(detached_losses * (1.0 - self.decay))

        reference = self.loss_ema[0].clamp_min(1e-6)
        weights = (reference / self.loss_ema.clamp_min(1e-6)).clamp(self.min_scale, self.max_scale)
        total = sum(weight * loss for weight, loss in zip(weights, losses))
        self.step_count.add_(1)
        return total, weights


class UncertaintyWeightedLoss(nn.Module):
    """
    P0 Improvement: Uncertainty-based automatic loss weighting.
    Learns optimal weights for multiple loss terms during training.
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018)
    """
    def __init__(self, num_losses=3):
        super().__init__()
        # Log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        """
        Args:
            losses: List of loss tensors [l1, l2, l3, ...]
        Returns:
            Weighted sum of losses
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted)
        return sum(weighted_losses), {f'weight_{i}': torch.exp(-self.log_vars[i]).item() for i in range(len(losses))}


class CompositeLoss(nn.Module):
    """
    Composite Loss for Low-Light Image Enhancement.
    Combines:
    1. Pixel Loss (Charbonnier) - Robust reconstruction.
    2. SSIM Loss - Structural similarity.
    3. LPIPS Loss - Perceptual quality.

    P0 Improvement: Now supports uncertainty-based automatic weighting.
    """
    def __init__(self,
                 w_char=1.0,
                 w_ssim=0.1,
                 w_lpips=0.1,
                 w_wavelet=0.0,
                 w_frequency=0.0,
                 w_edge=0.0,
                 device='cuda',
                 use_uncertainty_weighting=True,
                 loss_balance_mode='fixed',
                 loss_balance_decay=0.98,
                 loss_balance_warmup_steps=0,
                 use_lpips=True,
                 lpips_resize=None):
        super(CompositeLoss, self).__init__()
        self.w_char = w_char
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        self.w_wavelet = w_wavelet
        self.w_frequency = w_frequency
        self.w_edge = w_edge
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_lpips = use_lpips
        self.loss_balance_mode = loss_balance_mode
        self.loss_balance_warmup_steps = max(0, int(loss_balance_warmup_steps))
        self.loss_balancer = None

        self.char_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.lpips_loss = LPIPSLoss(enabled=use_lpips, resize_to=lpips_resize)
        self.wavelet_loss = HaarWaveletLoss() if w_wavelet > 0 else None
        self.frequency_loss = FrequencyDomainLoss(resize_to=lpips_resize) if w_frequency > 0 else None
        self.edge_loss = EdgeLoss() if w_edge > 0 else None

        self.active_term_names = ["char", "ssim"]
        if use_lpips:
            self.active_term_names.append("lpips")
        if self.wavelet_loss is not None:
            self.active_term_names.append("wavelet")
        if self.frequency_loss is not None:
            self.active_term_names.append("frequency")
        if self.edge_loss is not None:
            self.active_term_names.append("edge")

        if use_uncertainty_weighting:
            self.uncertainty_weighter = UncertaintyWeightedLoss(num_losses=len(self.active_term_names))
        elif self.loss_balance_mode == "ema":
            self.loss_balancer = EMAWeightedLossBalancer(
                num_losses=len(self.active_term_names),
                decay=loss_balance_decay,
            )
        else:
            self.loss_balancer = None

    def forward(self, pred, target, sample_weight: torch.Tensor | None = None):
        """
        Args:
            pred: Predicted image (B, C, H, W) - Expected range roughly [-1, 1]
            target: Ground truth image (B, C, H, W) - Expected range [-1, 1]
        """
        if pred.device != target.device:
            target = target.to(pred.device)

        l_char = self.char_loss(pred, target, reduction="none")

        # Normalize to [0, 1] for SSIM calculation
        pred_01 = (pred.clamp(-1, 1) + 1) / 2
        target_01 = (target.clamp(-1, 1) + 1) / 2
        l_ssim = self.ssim_loss(pred_01, target_01, reduction="none")

        l_lpips = self.lpips_loss(pred.clamp(-1, 1), target.clamp(-1, 1), reduction="none") if self.use_lpips else torch.zeros(pred.shape[0], device=pred.device)
        l_wavelet = self.wavelet_loss(pred_01, target_01, reduction="none") if self.wavelet_loss is not None else torch.zeros(pred.shape[0], device=pred.device)
        l_frequency = self.frequency_loss(pred_01, target_01, reduction="none") if self.frequency_loss is not None else torch.zeros(pred.shape[0], device=pred.device)
        l_edge = self.edge_loss(pred_01, target_01, reduction="none") if self.edge_loss is not None else torch.zeros(pred.shape[0], device=pred.device)

        if sample_weight is not None:
            sample_weight = sample_weight.to(pred.device).float()
            weight_sum = sample_weight.sum().clamp_min(1e-8)

            def reduce_samples(values):
                return (values * sample_weight).sum() / weight_sum
        else:
            def reduce_samples(values):
                return values.mean()

        if self.use_uncertainty_weighting:
            active_losses = [reduce_samples(l_char), reduce_samples(l_ssim)]
            if self.use_lpips:
                active_losses.append(reduce_samples(l_lpips))
            if self.wavelet_loss is not None:
                active_losses.append(reduce_samples(l_wavelet))
            if self.frequency_loss is not None:
                active_losses.append(reduce_samples(l_frequency))
            if self.edge_loss is not None:
                active_losses.append(reduce_samples(l_edge))
            total_loss, weights = self.uncertainty_weighter(active_losses)
            named_weights = {
                f"w_{name}": weights.get(f"weight_{index}")
                for index, name in enumerate(self.active_term_names)
            }
            logs = {
                'l_pix': reduce_samples(l_char).detach(),
                'l_ssim': reduce_samples(l_ssim).detach(),
                'l_lpips': reduce_samples(l_lpips).detach(),
                'l_wavelet': reduce_samples(l_wavelet).detach(),
                'l_frequency': reduce_samples(l_frequency).detach(),
                'l_edge': reduce_samples(l_edge).detach(),
                'l_total': total_loss.detach(),
                'w_char': named_weights.get('w_char', self.w_char),
                'w_ssim': named_weights.get('w_ssim', self.w_ssim),
                'w_lpips': named_weights.get('w_lpips', self.w_lpips),
                'w_wavelet': named_weights.get('w_wavelet', self.w_wavelet),
                'w_frequency': named_weights.get('w_frequency', self.w_frequency),
                'w_edge': named_weights.get('w_edge', self.w_edge),
            }
        else:
            base_terms = [
                self.w_char * l_char,
                self.w_ssim * l_ssim,
            ]
            if self.use_lpips:
                base_terms.append(self.w_lpips * l_lpips)
            if self.wavelet_loss is not None:
                base_terms.append(self.w_wavelet * l_wavelet)
            if self.frequency_loss is not None:
                base_terms.append(self.w_frequency * l_frequency)
            if self.edge_loss is not None:
                base_terms.append(self.w_edge * l_edge)

            base_loss_scalars = [reduce_samples(term) for term in base_terms]

            if self.loss_balance_mode == "ema" and self.loss_balancer is not None:
                balanced_loss, weights = self.loss_balancer(base_loss_scalars)
                weight_map = {
                    name: weights[index].detach().item()
                    for index, name in enumerate(self.active_term_names)
                }
                if self.loss_balancer.step_count.item() <= self.loss_balance_warmup_steps:
                    total_loss = sum(base_loss_scalars)
                else:
                    total_loss = balanced_loss
            else:
                total_loss = sum(base_loss_scalars)
                weight_map = {
                    'char': self.w_char,
                    'ssim': self.w_ssim,
                    'lpips': self.w_lpips if self.use_lpips else 0.0,
                    'wavelet': self.w_wavelet if self.wavelet_loss is not None else 0.0,
                    'frequency': self.w_frequency if self.frequency_loss is not None else 0.0,
                    'edge': self.w_edge if self.edge_loss is not None else 0.0,
                }

            logs = {
                'l_pix': reduce_samples(l_char).detach(),
                'l_ssim': reduce_samples(l_ssim).detach(),
                'l_lpips': reduce_samples(l_lpips).detach(),
                'l_wavelet': reduce_samples(l_wavelet).detach(),
                'l_frequency': reduce_samples(l_frequency).detach(),
                'l_edge': reduce_samples(l_edge).detach(),
                'l_total': total_loss.detach(),
                'w_char': weight_map.get('char', self.w_char),
                'w_ssim': weight_map.get('ssim', self.w_ssim),
                'w_lpips': weight_map.get('lpips', self.w_lpips),
                'w_wavelet': weight_map.get('wavelet', self.w_wavelet),
                'w_frequency': weight_map.get('frequency', self.w_frequency),
                'w_edge': weight_map.get('edge', self.w_edge),
            }

        return total_loss, logs
