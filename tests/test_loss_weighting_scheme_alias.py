from __future__ import annotations

import torch

from utils.misc import compute_adaptive_loss_weights


class _DummySchedulerConfig:
    prediction_type = "epsilon"


class _DummyScheduler:
    config = _DummySchedulerConfig()

    def __init__(self):
        # Monotonic alpha schedule in (0, 1]
        self.alphas_cumprod = torch.linspace(0.999, 0.01, steps=1000)


def test_compute_adaptive_loss_weights_accepts_mini_snr_alias():
    scheduler = _DummyScheduler()
    timesteps = torch.tensor([0, 50, 300, 900], dtype=torch.long)

    weights_alias = compute_adaptive_loss_weights(
        scheduler,
        timesteps,
        weighting_scheme="mini_snr",
        snr_gamma=5.0,
    )
    weights_canonical = compute_adaptive_loss_weights(
        scheduler,
        timesteps,
        weighting_scheme="min_snr",
        snr_gamma=5.0,
    )

    assert torch.allclose(weights_alias, weights_canonical)
