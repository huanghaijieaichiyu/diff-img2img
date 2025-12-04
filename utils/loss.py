import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDomainLoss(nn.Module):
    """
    Frequency Domain Loss using Fast Fourier Transform (FFT).
    Computes L1 loss between the amplitude and phase of the prediction and target.
    """
    def __init__(self, loss_type='l1', alpha=1.0, beta=1.0):
        super(FrequencyDomainLoss, self).__init__()
        self.loss_type = loss_type
        self.alpha = alpha  # Weight for amplitude loss
        self.beta = beta    # Weight for phase loss

    def forward(self, pred, target):
        # FFT
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))

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
    Edge Loss using Sobel filters.
    Computes L1 loss between the edge maps of prediction and target.
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.125, .25, .125], [.25, .5, .25], [.125, .25, .125]])
        self.kernel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        
        # Create Sobel kernels for each channel
        # Vertical edges
        kernel_v = self.kernel.expand(c, 1, 3, 3).to(pred.device)
        # Horizontal edges
        kernel_h = self.kernel.t().expand(c, 1, 3, 3).to(pred.device)
        
        pred_v = F.conv2d(pred, kernel_v, groups=c, padding=1)
        pred_h = F.conv2d(pred, kernel_h, groups=c, padding=1)
        
        target_v = F.conv2d(target, kernel_v, groups=c, padding=1)
        target_h = F.conv2d(target, kernel_h, groups=c, padding=1)
        
        loss = self.loss(pred_v, target_v) + self.loss(pred_h, target_h)
        return loss


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

