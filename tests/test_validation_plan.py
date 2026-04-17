from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from core.engine import DiffusionEngine, normalize_validation_metrics, resolve_validation_step_counts
from models.diffusion import CombinedModel


def test_normalize_validation_metrics_defaults_to_psnr_ssim():
    assert normalize_validation_metrics(None) == ("psnr", "ssim")


def test_normalize_validation_metrics_keeps_psnr_ssim_and_dedupes():
    metrics = normalize_validation_metrics(["lpips", "ssim", "psnr", "lpips"])
    assert metrics == ("psnr", "ssim", "lpips")


def test_resolve_validation_step_counts_fast_uses_train_steps_only():
    step_counts = resolve_validation_step_counts(
        8,
        [8, 20],
        [8],
        fast=True,
    )
    assert step_counts == [8]


def test_resolve_validation_step_counts_full_keeps_benchmark_steps():
    step_counts = resolve_validation_step_counts(
        8,
        [8, 20],
        [8],
        fast=False,
    )
    assert step_counts == [8, 20]


def test_resolve_training_epoch_limit_rounds_up():
    # 3 updates/epoch, 10 total steps -> 4 epochs required.
    assert DiffusionEngine._resolve_training_epoch_limit(10, 3) == 4


def test_resolve_training_epoch_limit_at_least_one_epoch():
    assert DiffusionEngine._resolve_training_epoch_limit(1, 0) == 1


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_grouped_lr_optimizer_supports_compiled_child_modules():
    engine = DiffusionEngine.__new__(DiffusionEngine)
    compiled_unet = nn.Linear(4, 4)
    compiled_unet.compile(mode="reduce-overhead")
    engine.training_model = CombinedModel(
        compiled_unet,
        decom_model=nn.Linear(4, 4),
        condition_adapter=nn.Linear(4, 4),
    )
    engine.criterion = nn.Linear(1, 1)
    engine.args = SimpleNamespace(lr=1e-4)

    optimizer = engine._build_optimizer_with_grouped_lr()

    assert len(optimizer.param_groups) > 0
