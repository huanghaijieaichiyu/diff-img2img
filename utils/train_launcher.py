#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_config import resolve_config_path


LOCAL_ACCELERATE_BIN = PROJECT_ROOT / ".venv" / "bin" / "accelerate"
DEFAULT_DATA_DIR = "/mnt/f/datasets/kitti_LOL"
DEFAULT_OUTPUT_DIR = "runs/retinex"
DEFAULT_TRAIN_PROFILE = "auto"

_ACTIVE_PROCESS: subprocess.Popen | None = None


@dataclass(frozen=True)
class OptionSpec:
    env_name: str
    cli_flag: str
    kind: str = "scalar"
    default: str | None = None


@dataclass(frozen=True)
class LaunchPlan:
    run_mode: str
    accelerate_bin: str
    config_path: str
    data_dir: str
    output_dir: str
    primary_command: tuple[str, ...]
    post_success_command: tuple[str, ...] | None = None


COMMON_OPTION_SPECS = (
    OptionSpec("PREPARED_CACHE_DIR", "--prepared_cache_dir"),
    OptionSpec("NUM_WORKERS", "--num_workers"),
    OptionSpec("PREFETCH_FACTOR", "--prefetch_factor"),
    OptionSpec("PERSISTENT_WORKERS", "--persistent_workers", kind="bool_optional"),
    OptionSpec("PIN_MEMORY", "--pin_memory", kind="bool_optional"),
    OptionSpec("DECODE_CACHE_SIZE", "--decode_cache_size"),
    OptionSpec("OPENCV_THREADS_PER_WORKER", "--opencv_threads_per_worker"),
    OptionSpec("MIXED_PRECISION", "--mixed_precision"),
    OptionSpec("USE_RETINEX", "--use_retinex", kind="bool_optional"),
    OptionSpec("ATTENTION_BACKEND", "--attention_backend"),
    OptionSpec("USE_TORCH_COMPILE", "--use_torch_compile", kind="bool_optional"),
    OptionSpec("TORCH_COMPILE_MODE", "--torch_compile_mode"),
    OptionSpec("ALLOW_UNSAFE_COMPILE_WITH_FILM", "--allow_unsafe_compile_with_film", kind="bool_optional"),
)

TRAIN_OPTION_SPECS = (
    OptionSpec("PREPARE_WORKERS", "--prepare_workers"),
    OptionSpec("OFFLINE_VARIANT_COUNT", "--offline_variant_count"),
    OptionSpec("SYNTHESIS_SEED", "--synthesis_seed"),
    OptionSpec("VALIDATION_STEPS", "--validation_steps"),
    OptionSpec("NUM_VALIDATION_IMAGES", "--num_validation_images"),
    OptionSpec("BENCHMARK_INFERENCE_STEPS", "--benchmark_inference_steps", kind="list"),
    OptionSpec("BATCH_SIZE", "--batch_size"),
    OptionSpec("GRADIENT_ACCUMULATION_STEPS", "--gradient_accumulation_steps"),
    OptionSpec("MAX_TRAIN_STEPS", "--max_train_steps"),
    OptionSpec("SEMANTIC_BACKBONE", "--semantic_backbone"),
    OptionSpec("NR_METRIC", "--nr_metric"),
    OptionSpec("RESOLUTION", "--resolution"),
    OptionSpec("EPOCHS", "--epochs"),
    OptionSpec("LR", "--lr"),
    OptionSpec("RESUME", "--resume"),
    OptionSpec("DARKER_RANGES", "--darker_ranges"),
    OptionSpec("PREPARE_FORCE", "--prepare_force", kind="bool_flag"),
)

VALIDATE_OPTION_SPECS = (
    OptionSpec("FULL_EVAL_BATCH_SIZE", "--batch_size", default="2"),
    OptionSpec("FULL_EVAL_NUM_VALIDATION_IMAGES", "--num_validation_images", default="12"),
    OptionSpec("FULL_EVAL_BENCHMARK_INFERENCE_STEPS", "--benchmark_inference_steps", kind="list", default="8 20"),
    OptionSpec("FULL_EVAL_SEMANTIC_BACKBONE", "--semantic_backbone", default="resnet18"),
    OptionSpec("FULL_EVAL_NR_METRIC", "--nr_metric", default="niqe"),
)


def _normalize_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _split_list(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return shlex.split(value)


def _append_option(command: list[str], spec: OptionSpec, environ: Mapping[str, str]) -> None:
    raw_value = environ.get(spec.env_name)
    if raw_value in (None, ""):
        raw_value = spec.default

    if raw_value in (None, ""):
        return

    if spec.kind == "scalar":
        command.extend([spec.cli_flag, raw_value])
        return

    if spec.kind == "list":
        parts = _split_list(raw_value)
        if parts:
            command.extend([spec.cli_flag, *parts])
        return

    if spec.kind == "bool_optional":
        normalized = _normalize_bool(raw_value)
        if normalized is True:
            command.append(spec.cli_flag)
        elif normalized is False:
            command.append(f"--no-{spec.cli_flag[2:]}")
        return

    if spec.kind == "bool_flag":
        if _normalize_bool(raw_value) is True:
            command.append(spec.cli_flag)
        return

    raise ValueError(f"Unsupported option kind: {spec.kind}")


def resolve_accelerate_bin(environ: Mapping[str, str]) -> str:
    override = environ.get("ACCELERATE_BIN")
    if override:
        return override
    if LOCAL_ACCELERATE_BIN.exists():
        return str(LOCAL_ACCELERATE_BIN)
    return "accelerate"


def build_train_command(
    *,
    accelerate_bin: str,
    config_path: str,
    data_dir: str,
    output_dir: str,
    environ: Mapping[str, str],
) -> tuple[str, ...]:
    command = [
        accelerate_bin,
        "launch",
        "main.py",
        "--mode",
        "train",
        "--config",
        config_path,
        "--data_dir",
        data_dir,
        "--output_dir",
        output_dir,
        "--train_profile",
        environ.get("TRAIN_PROFILE", DEFAULT_TRAIN_PROFILE),
    ]
    for spec in COMMON_OPTION_SPECS:
        _append_option(command, spec, environ)
    for spec in TRAIN_OPTION_SPECS:
        _append_option(command, spec, environ)
    return tuple(command)


def build_validate_command(
    *,
    accelerate_bin: str,
    config_path: str,
    data_dir: str,
    output_dir: str,
    environ: Mapping[str, str],
) -> tuple[str, ...]:
    model_path = environ.get("MODEL_PATH") or os.path.join(output_dir, "best_model")
    validation_output_dir = environ.get("VALIDATION_OUTPUT_DIR") or os.path.join(output_dir, "full_eval")

    command = [
        accelerate_bin,
        "launch",
        "main.py",
        "--mode",
        "validate",
        "--config",
        config_path,
        "--data_dir",
        data_dir,
        "--model_path",
        model_path,
        "--output_dir",
        validation_output_dir,
    ]
    for spec in COMMON_OPTION_SPECS:
        _append_option(command, spec, environ)
    for spec in VALIDATE_OPTION_SPECS:
        _append_option(command, spec, environ)
    return tuple(command)


def plan_from_env(environ: Mapping[str, str]) -> LaunchPlan:
    run_mode = (environ.get("RUN_MODE") or "train").strip().lower()
    if run_mode not in {"train", "validate"}:
        raise SystemExit(f"Unsupported RUN_MODE={run_mode!r}. Expected 'train' or 'validate'.")

    model_size = (environ.get("MODEL_SIZE") or "middle").strip() or "middle"
    config_key = environ.get("CONFIG_PATH") or model_size
    config_path = resolve_config_path(config_key)
    data_dir = environ.get("DATA_DIR") or DEFAULT_DATA_DIR
    output_dir = environ.get("OUTPUT_DIR") or DEFAULT_OUTPUT_DIR
    accelerate_bin = resolve_accelerate_bin(environ)

    if run_mode == "validate":
        primary_command = build_validate_command(
            accelerate_bin=accelerate_bin,
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            environ=environ,
        )
        return LaunchPlan(
            run_mode=run_mode,
            accelerate_bin=accelerate_bin,
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            primary_command=primary_command,
        )

    primary_command = build_train_command(
        accelerate_bin=accelerate_bin,
        config_path=config_path,
        data_dir=data_dir,
        output_dir=output_dir,
        environ=environ,
    )
    post_success_command = None
    if _normalize_bool(environ.get("RUN_FULL_EVAL_AFTER_TRAIN")) is True:
        post_success_command = build_validate_command(
            accelerate_bin=accelerate_bin,
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            environ=environ,
        )

    return LaunchPlan(
        run_mode=run_mode,
        accelerate_bin=accelerate_bin,
        config_path=config_path,
        data_dir=data_dir,
        output_dir=output_dir,
        primary_command=primary_command,
        post_success_command=post_success_command,
    )


def _warn_if_cross_mounted_data(data_dir: str, prepared_cache_dir: str | None) -> None:
    active_cache_dir = prepared_cache_dir or os.path.join(data_dir, ".prepared")
    if data_dir.startswith("/mnt/") or active_cache_dir.startswith("/mnt/"):
        print("[start_train][warning] dataset or prepared cache appears to be on a cross-mounted path.", file=sys.stderr)
        print("[start_train][warning] For trustworthy throughput measurements, prefer a local Linux SSD.", file=sys.stderr)


def _print_command(command: tuple[str, ...]) -> None:
    print(f"[start_train] command={shlex.join(command)}", flush=True)


def _terminate_active_process(_signum, _frame) -> None:
    global _ACTIVE_PROCESS
    if _ACTIVE_PROCESS is not None and _ACTIVE_PROCESS.poll() is None:
        print("\n[start_train] interrupt received, stopping launcher...", file=sys.stderr, flush=True)
        try:
            os.killpg(_ACTIVE_PROCESS.pid, signal.SIGTERM)
        except (AttributeError, ProcessLookupError):
            _ACTIVE_PROCESS.terminate()
        try:
            _ACTIVE_PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    raise SystemExit(130)


def _run_command(command: tuple[str, ...]) -> int:
    global _ACTIVE_PROCESS
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, start_new_session=True)
    _ACTIVE_PROCESS = process
    try:
        return process.wait()
    finally:
        _ACTIVE_PROCESS = None


def main() -> int:
    signal.signal(signal.SIGINT, _terminate_active_process)
    signal.signal(signal.SIGTERM, _terminate_active_process)

    plan = plan_from_env(os.environ)
    _warn_if_cross_mounted_data(plan.data_dir, os.environ.get("PREPARED_CACHE_DIR"))

    print(f"[start_train] accelerate_bin={plan.accelerate_bin}", flush=True)
    print(f"[start_train] run_mode={plan.run_mode}", flush=True)
    print(f"[start_train] config={plan.config_path}", flush=True)
    print(f"[start_train] data_dir={plan.data_dir}", flush=True)
    print(f"[start_train] output_dir={plan.output_dir}", flush=True)

    _print_command(plan.primary_command)
    status = _run_command(plan.primary_command)
    if status != 0:
        return status

    if plan.post_success_command is not None:
        _print_command(plan.post_success_command)
        return _run_command(plan.post_success_command)

    return 0


if __name__ == "__main__":
    sys.exit(main())
