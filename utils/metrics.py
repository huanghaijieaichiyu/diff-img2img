import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def try_compute_niqe(images_01: torch.Tensor) -> Optional[float]:
    """
    Optional NIQE wrapper.
    Returns None when no supported backend is available.
    """
    try:
        from skvideo.measure import niqe
    except ImportError:
        return None

    scores = []
    images_np = images_01.detach().cpu().numpy()
    for image in images_np:
        image_hwc = np.transpose(image, (1, 2, 0))
        image_gray = np.clip(np.dot(image_hwc[..., :3], [0.299, 0.587, 0.114]), 0.0, 1.0)
        image_gray = (image_gray * 255.0).astype(np.uint8)
        try:
            score = float(niqe(image_gray))
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return None
    return float(np.mean(scores))


class SemanticFeatureMetric:
    def __init__(self, device: torch.device, backbone: str = "resnet18"):
        self.device = device
        self.backbone_name = backbone
        self.model = None
        self.available = False

        if backbone == "none":
            return

        try:
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.DEFAULT
            backbone_model = resnet18(weights=weights)
            modules = list(backbone_model.children())[:-1]
            self.model = torch.nn.Sequential(*modules).to(device)
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.available = True
        except Exception:
            self.model = None
            self.available = False

    def compute(self, pred_01: torch.Tensor, target_01: torch.Tensor) -> Optional[float]:
        if not self.available or self.model is None:
            return None

        pred_resized = F.interpolate(pred_01, size=(224, 224), mode="bilinear", align_corners=False)
        target_resized = F.interpolate(target_01, size=(224, 224), mode="bilinear", align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]

        pred_norm = (pred_resized - mean) / std
        target_norm = (target_resized - mean) / std

        with torch.no_grad():
            pred_feat = self.model(pred_norm).flatten(1)
            target_feat = self.model(target_norm).flatten(1)

        return torch.norm(pred_feat - target_feat, dim=1).mean().item()


def benchmark_step_runtime(callable_fn, warmup: int = 1, repeats: int = 1) -> Dict[str, float]:
    for _ in range(max(0, warmup)):
        callable_fn()

    start = time.perf_counter()
    for _ in range(max(1, repeats)):
        callable_fn()
    duration = time.perf_counter() - start

    return {
        "avg_seconds": duration / max(1, repeats),
        "total_seconds": duration,
    }
