import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 approximation) for robust regression.
    Formula: sqrt(x^2 + eps^2)
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
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

    def forward(self, img1, img2):
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
        return 1 - ssim_map.mean()


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) Loss.
    Uses a pre-trained VGG network to measure perceptual distance.
    
    This provides much better visual quality than pixel-only losses,
    as it captures high-level texture and structural features.
    """
    def __init__(self, net='vgg'):
        super(LPIPSLoss, self).__init__()
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
    
    def forward(self, pred, target):
        """
        Args:
            pred, target: Images in [-1, 1] range, shape (B, C, H, W)
        Returns:
            Scalar LPIPS loss
        """
        if not self.available:
            return torch.tensor(0.0, device=pred.device)
        return self.loss_fn(pred, target).mean()


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
                 device='cuda',
                 use_uncertainty_weighting=True):
        super(CompositeLoss, self).__init__()
        self.w_char = w_char
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        self.use_uncertainty_weighting = use_uncertainty_weighting

        self.char_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.lpips_loss = LPIPSLoss()

        # P0: Add uncertainty weighting
        if use_uncertainty_weighting:
            self.uncertainty_weighter = UncertaintyWeightedLoss(num_losses=3)

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image (B, C, H, W) - Expected range roughly [-1, 1]
            target: Ground truth image (B, C, H, W) - Expected range [-1, 1]
        """
        if pred.device != target.device:
            target = target.to(pred.device)

        l_char = self.char_loss(pred, target)

        # Normalize to [0, 1] for SSIM calculation
        pred_01 = (pred.clamp(-1, 1) + 1) / 2
        target_01 = (target.clamp(-1, 1) + 1) / 2
        l_ssim = self.ssim_loss(pred_01, target_01)

        # LPIPS expects [-1, 1] range directly
        l_lpips = self.lpips_loss(pred.clamp(-1, 1), target.clamp(-1, 1))

        # P0: Use uncertainty weighting if enabled
        if self.use_uncertainty_weighting:
            total_loss, weights = self.uncertainty_weighter([l_char, l_ssim, l_lpips])
            logs = {
                'l_pix': l_char.detach(),
                'l_ssim': l_ssim.detach(),
                'l_lpips': l_lpips.detach(),
                'l_total': total_loss.detach(),
                'w_char': weights.get('weight_0', self.w_char),
                'w_ssim': weights.get('weight_1', self.w_ssim),
                'w_lpips': weights.get('weight_2', self.w_lpips),
            }
        else:
            total_loss = (self.w_char * l_char +
                          self.w_ssim * l_ssim +
                          self.w_lpips * l_lpips)
            logs = {
                'l_pix': l_char.detach(),
                'l_ssim': l_ssim.detach(),
                'l_lpips': l_lpips.detach(),
                'l_total': total_loss.detach()
            }

        return total_loss, logs