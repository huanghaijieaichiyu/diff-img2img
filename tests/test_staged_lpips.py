from __future__ import annotations

from types import SimpleNamespace

from core.engine import DiffusionEngine


def test_configure_and_update_staged_lpips_runtime_state():
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.lpips_stage_enable = True
    engine.lpips_stage_start_ratio = 0.8
    engine.args = SimpleNamespace(max_train_steps=100)
    engine.accelerator = SimpleNamespace(is_main_process=False)
    engine.criterion = SimpleNamespace(use_lpips=True)
    engine._lpips_stage_last_enabled = None

    engine._configure_lpips_stage_schedule()
    assert engine._lpips_stage_start_step == 80

    engine._update_lpips_runtime_state(joint_step=10)
    assert engine.criterion.use_lpips is False

    engine._update_lpips_runtime_state(joint_step=90)
    assert engine.criterion.use_lpips is True


def test_configure_staged_lpips_clamps_start_step():
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.lpips_stage_enable = True
    engine.lpips_stage_start_ratio = 1.0
    engine.args = SimpleNamespace(max_train_steps=100)
    engine.accelerator = SimpleNamespace(is_main_process=False)

    engine._configure_lpips_stage_schedule()
    assert engine._lpips_stage_start_step == 99
