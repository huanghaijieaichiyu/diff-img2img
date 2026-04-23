from __future__ import annotations

import shlex
import sys
from typing import Any

from utils.project_config import build_preview_namespace, build_runtime_summary, load_preset_summary, resolve_config_path


def quote_command(command: list[str]) -> str:
    return shlex.join(command)


def build_train_command(
    model_scale: str,
    data_dir: str,
    output_dir: str,
    resolution: int,
    batch_size: int,
    epochs: int,
    lr: float,
    use_retinex: bool,
    resume: str,
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
    ]
    if use_retinex:
        cmd.append("--use_retinex")
    else:
        cmd.append("--no-use_retinex")
    if resume:
        cmd.extend(["--resume", resume])
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
    use_retinex: bool | None = None,
    resume: str | None = None,
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
        "use_retinex": use_retinex,
        "resume": resume,
        "num_validation_images": num_validation_images,
    }
    filtered = {key: value for key, value in overrides.items() if value is not None}
    preview_args = build_preview_namespace(model_scale, filtered)
    return build_runtime_summary(preview_args)


def load_preset_card(model_scale: str) -> dict[str, Any]:
    return load_preset_summary(model_scale)
