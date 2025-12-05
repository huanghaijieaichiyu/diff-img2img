import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDomainLoss(nn.Module):
    """
    Frequency Domain Loss using Fast Fourier Transform (FFT).
    Computes L1 loss between the amplitude and phase of the prediction and target.
    Uses orthonormal normalization to ensure scale invariance.
    """
    def __init__(self, loss_type='l1', alpha=1.0, beta=1.0):
        super(FrequencyDomainLoss, self).__init__()
        self.loss_type = loss_type
        self.alpha = alpha  # Weight for amplitude loss
        self.beta = beta    # Weight for phase loss

    def forward(self, pred, target):
        # FFT with ortho normalization for energy preservation
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.fft2(target, dim=(-2, -1), norm='ortho')

        pred_amp = torch.abs(pred_fft)
        pred_phase = torch.angle(pred_fft)

        target_amp = torch.abs(target_fft)
        target_phase = torch.angle(target_fft)

        if self.loss_type == 'l1':
            loss_amp = F.l1_loss(pred_amp, target_amp)
            loss_phase = F.l1_loss(pred_phase, target_phase)
        else:
            loss_amp = F.mse_loss(pred_amp, target_amp)
            loss_phase = F.mse_loss(pred_phase, target_phase)

        return self.alpha * loss_amp + self.beta * loss_phase


class EdgeLoss(nn.Module):
    """
    Edge Loss using Sobel filters with pre-smoothing.
    Computes L1 loss between the edge maps of prediction and target.
    Includes Gaussian blurring to suppress noise and improve robustness.
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # 3x3 Gaussian Kernel for smoothing
        # approx sigma=0.85
        self.gaussian_kernel = torch.Tensor([
            [1/16, 2/16, 1/16],
            [2/16, 4/16, 2/16],
            [1/16, 2/16, 1/16]
        ])
        
        # Sobel Kernels
        self.sobel_kernel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobel_kernel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        device = pred.device
        
        # Prepare kernels
        gaussian = self.gaussian_kernel.expand(c, 1, 3, 3).to(device)
        sobel_x = self.sobel_kernel_x.expand(c, 1, 3, 3).to(device)
        sobel_y = self.sobel_kernel_y.expand(c, 1, 3, 3).to(device)
        
        # 1. Smooth images
        # Use replication padding to reduce border artifacts
        pad_p = (1, 1, 1, 1)
        pred_smooth = F.conv2d(F.pad(pred, pad_p, mode='replicate'), gaussian, groups=c)
        target_smooth = F.conv2d(F.pad(target, pad_p, mode='replicate'), gaussian, groups=c)
        
        # 2. Compute Edges
        # Apply Sobel X
        pred_dx = F.conv2d(F.pad(pred_smooth, pad_p, mode='replicate'), sobel_x, groups=c)
        target_dx = F.conv2d(F.pad(target_smooth, pad_p, mode='replicate'), sobel_x, groups=c)
        
        # Apply Sobel Y
        pred_dy = F.conv2d(F.pad(pred_smooth, pad_p, mode='replicate'), sobel_y, groups=c)
        target_dy = F.conv2d(F.pad(target_smooth, pad_p, mode='replicate'), sobel_y, groups=c)
        
        # 3. Compute Loss
        # Combine gradients (magnitude) or sum losses separately
        # Summing L1 losses of gradients is standard for Edge Loss
        loss_x = self.loss(pred_dx, target_dx)
        loss_y = self.loss(pred_dy, target_dy)
        
        return loss_x + loss_y


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.reduction = 'none'
    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


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
    Implementation independent of utils.misic to avoid circular imports.
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
                window = window.cuda(img1.get_device())
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


class CompositeLoss(nn.Module):
    """
    Composite Loss for Low-Light Image Enhancement (Autonomous Driving).
    Combines:
    1. Pixel Loss (Charbonnier) - Robust reconstruction.
    2. Edge Loss (Sobel) - Preserves lane lines/objects edges.
    3. Frequency Loss (FFT) - Global consistency.
    4. SSIM Loss - Structural similarity.
    """
    def __init__(self, 
                 w_char=1.0, 
                 w_edge=0.05, 
                 w_freq=0.01, 
                 w_ssim=0.1,
                 device='cuda'):
        super(CompositeLoss, self).__init__()
        self.w_char = w_char
        self.w_edge = w_edge
        self.w_freq = w_freq
        self.w_ssim = w_ssim
        
        self.char_loss = CharbonnierLoss()
        self.edge_loss = EdgeLoss()
        self.freq_loss = FrequencyDomainLoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image (B, C, H, W) - Expected range roughly [-1, 1]
            target: Ground truth image (B, C, H, W) - Expected range [-1, 1]
        """
        # Ensure inputs are on the same device
        if pred.device != target.device:
            target = target.to(pred.device)

        l_char = self.char_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        l_freq = self.freq_loss(pred, target)
        
        # Normalize to [0, 1] for SSIM calculation
        # Diffusion models typically operate in [-1, 1]
        pred_01 = (pred.clamp(-1, 1) + 1) / 2
        target_01 = (target.clamp(-1, 1) + 1) / 2
        l_ssim = self.ssim_loss(pred_01, target_01)
        
        total_loss = (self.w_char * l_char +
                      self.w_edge * l_edge +
                      self.w_freq * l_freq +
                      self.w_ssim * l_ssim)
        
        logs = {
            'l_pix': l_char.detach(),
            'l_edge': l_edge.detach(),
            'l_freq': l_freq.detach(),
            'l_ssim': l_ssim.detach(),
            'l_total': total_loss.detach()
        }
        
        return total_loss, logs


