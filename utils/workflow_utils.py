from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any

from datasets.prepare_data import summarize_prepared_cache as summarize_prepared_cache_state
from utils.project_config import build_preview_namespace, build_runtime_summary, load_preset_summary, resolve_config_path


def recommended_prepare_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return max(1, min(8, cpu_count // 2))


def resolve_cache_dir(data_dir: str, prepared_cache_dir: str | None) -> str:
    if prepared_cache_dir:
        return os.path.abspath(prepared_cache_dir)
    return os.path.abspath(os.path.join(data_dir, ".prepared"))


def summarize_prepared_cache(
    data_dir: str,
    prepared_cache_dir: str | None,
    variant_count: int,
    synthesis_seed: int,
    darker_ranges: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return summarize_prepared_cache_state(
        data_dir,
        prepared_cache_dir,
        variant_count=variant_count,
        synthesis_seed=synthesis_seed,
        darker_ranges=darker_ranges,
    )


def quote_command(command: list[str]) -> str:
    return shlex.join(command)


def build_prepare_command(
    config_path: str,
    data_dir: str,
    prepared_cache_dir: str,
    variant_count: int,
    prepare_workers: int,
    synthesis_seed: int,
    prepare_force: bool,
    darker_ranges_text: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "main.py",
        "--mode",
        "prepare",
        "--config",
        str(resolve_config_path(config_path)),
        "--data_dir",
        data_dir,
        "--offline_variant_count",
        str(variant_count),
        "--prepare_workers",
        str(prepare_workers),
        "--synthesis_seed",
        str(synthesis_seed),
    ]
    if prepared_cache_dir:
        cmd.extend(["--prepared_cache_dir", prepared_cache_dir])
    if prepare_force:
        cmd.append("--prepare_force")
    if darker_ranges_text.strip():
        cmd.extend(["--darker_ranges", darker_ranges_text])
    return cmd


def build_train_command(
    model_scale: str,
    data_dir: str,
    output_dir: str,
    resolution: int,
    batch_size: int,
    epochs: int,
    lr: float,
    train_profile: str,
    variant_count: int,
    prepare_workers: int,
    synthesis_seed: int,
    use_retinex: bool,
    resume: str,
    prepared_cache_dir: str,
    prepare_force: bool,
    darker_ranges_text: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "main.py",
        "--mode",
        "train",
        "--config",
        str(resolve_config_path(model_scale)),
        "--data_dir",
        data_dir,
        "--output_dir",
        output_dir,
        "--resolution",
        str(resolution),
        "--batch_size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--train_profile",
        train_profile,
        "--offline_variant_count",
        str(variant_count),
        "--prepare_workers",
        str(prepare_workers),
        "--synthesis_seed",
        str(synthesis_seed),
    ]
    if use_retinex:
        cmd.append("--use_retinex")
    else:
        cmd.append("--no-use_retinex")
    if resume:
        cmd.extend(["--resume", resume])
    if prepared_cache_dir:
        cmd.extend(["--prepared_cache_dir", prepared_cache_dir])
    if prepare_force:
        cmd.append("--prepare_force")
    if darker_ranges_text.strip():
        cmd.extend(["--darker_ranges", darker_ranges_text])
    return cmd


def build_preview_summary(
    model_scale: str,
    *,
    mode: str,
    data_dir: str,
    output_dir: str | None = None,
    model_path: str | None = None,
    resolution: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    train_profile: str | None = None,
    variant_count: int | None = None,
    prepare_workers: int | None = None,
    synthesis_seed: int | None = None,
    use_retinex: bool | None = None,
    resume: str | None = None,
    prepared_cache_dir: str | None = None,
    prepare_force: bool | None = None,
    num_validation_images: int | None = None,
) -> dict[str, Any]:
    overrides = {
        "mode": mode,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "model_path": model_path,
        "resolution": resolution,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "train_profile": train_profile,
        "offline_variant_count": variant_count,
        "prepare_workers": prepare_workers,
        "synthesis_seed": synthesis_seed,
        "use_retinex": use_retinex,
        "resume": resume,
        "prepared_cache_dir": prepared_cache_dir,
        "prepare_force": prepare_force,
        "num_validation_images": num_validation_images,
    }
    filtered = {key: value for key, value in overrides.items() if value is not None}
    preview_args = build_preview_namespace(model_scale, filtered)
    return build_runtime_summary(preview_args)


def load_preset_card(model_scale: str) -> dict[str, Any]:
    return load_preset_summary(model_scale)
