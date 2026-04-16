from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


MODEL_CONFIG_PRESETS = {
    "small": "configs/train/small.yaml",
    "small_throughput": "configs/train/small_throughput.yaml",
    "middle": "configs/train/middle.yaml",
    "max": "configs/train/max.yaml",
}

LEGACY_MODEL_CONFIG_ALIASES = {
    "small_sota": "small",
    "small_accum": "small",
    "middle_sota": "middle",
    "middle_accum": "middle",
    "max_accum": "max",
}

CONFIG_PRECEDENCE = [
    "CLI explicit args",
    "shell/UI explicit launch overrides",
    "configs/train/*.yaml defaults",
    "TRAIN_PROFILES backfill only when values are absent",
]


def flatten_config_tree(config_node: dict[str, Any]) -> dict[str, Any]:
    preserved_dict_keys = {"darker_ranges"}
    flat: dict[str, Any] = {}
    for key, value in (config_node or {}).items():
        if isinstance(value, dict) and key not in preserved_dict_keys:
            flat.update(flatten_config_tree(value))
        else:
            flat[key] = value
    return flat


def resolve_config_path(config_value: str | None) -> str:
    if not config_value:
        config_value = "small"
    if config_value in LEGACY_MODEL_CONFIG_ALIASES:
        replacement = LEGACY_MODEL_CONFIG_ALIASES[config_value]
        raise ValueError(
            f"Legacy config '{config_value}' has been removed. Please use '{replacement}' instead."
        )
    resolved = MODEL_CONFIG_PRESETS.get(config_value, config_value)
    return str(Path(resolved))


def load_config_defaults(config_path: str) -> dict[str, Any]:
    resolved_path = Path(resolve_config_path(config_path))
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    defaults = flatten_config_tree(config_data)
    defaults["config"] = str(resolved_path)
    defaults["config_name"] = config_data.get("meta", {}).get("name", resolved_path.stem)
    return defaults


def load_preset_summary(config_value: str) -> dict[str, Any]:
    config_path = Path(resolve_config_path(config_value))
    summary = {
        "name": config_path.stem,
        "config_path": str(config_path),
        "description": "",
        "target_vram_gb": "-",
        "resolution": "-",
        "batch_size": "-",
        "gradient_accumulation_steps": 1,
        "effective_batch": "-",
    }
    if not config_path.exists():
        return summary

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    meta = data.get("meta", {})
    runtime = data.get("runtime", {})
    optimization = data.get("optimization", {})
    batch_size = int(optimization.get("batch_size", 1))
    grad_acc = int(optimization.get("gradient_accumulation_steps", 1))
    summary.update(
        {
            "name": meta.get("name", config_path.stem),
            "description": meta.get("description", ""),
            "target_vram_gb": meta.get("target_vram_gb", "-"),
            "resolution": runtime.get("resolution", "-"),
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_acc,
            "effective_batch": batch_size * grad_acc,
        }
    )
    return summary


def effective_batch_size(batch_size: int | None, gradient_accumulation_steps: int | None) -> int | None:
    if batch_size is None or gradient_accumulation_steps is None:
        return None
    return max(1, int(batch_size)) * max(1, int(gradient_accumulation_steps))


def build_runtime_summary(args: argparse.Namespace | SimpleNamespace | dict[str, Any]) -> dict[str, Any]:
    if isinstance(args, dict):
        payload = args
    else:
        payload = vars(args)

    config_path = payload.get("config")
    config_name = payload.get("config_name")
    if config_path:
        resolved_path = Path(resolve_config_path(str(config_path)))
        config_path = str(resolved_path)
        if not config_name:
            config_name = resolved_path.stem

    benchmark_steps = payload.get("benchmark_inference_steps")
    if benchmark_steps is None:
        benchmark_steps = []
    elif not isinstance(benchmark_steps, list):
        benchmark_steps = list(benchmark_steps)

    summary = {
        "mode": payload.get("mode"),
        "config_name": config_name,
        "config_path": config_path,
        "config_precedence": CONFIG_PRECEDENCE,
        "data_dir": payload.get("data_dir"),
        "output_dir": payload.get("output_dir"),
        "model_path": payload.get("model_path"),
        "prepared_cache_dir": payload.get("prepared_cache_dir"),
        "train_profile": payload.get("train_profile"),
        "mixed_precision": payload.get("mixed_precision"),
        "resolution": payload.get("resolution"),
        "batch_size": payload.get("batch_size"),
        "gradient_accumulation_steps": payload.get("gradient_accumulation_steps"),
        "effective_batch_size": payload.get("effective_batch_size") or effective_batch_size(
            payload.get("batch_size"),
            payload.get("gradient_accumulation_steps"),
        ),
        "min_effective_batch_size": payload.get("min_effective_batch_size"),
        "num_workers": payload.get("num_workers"),
        "prefetch_factor": payload.get("prefetch_factor"),
        "persistent_workers": payload.get("persistent_workers"),
        "pin_memory": payload.get("pin_memory"),
        "decode_cache_size": payload.get("decode_cache_size"),
        "opencv_threads_per_worker": payload.get("opencv_threads_per_worker"),
        "prepare_on_train": payload.get("prepare_on_train"),
        "prepare_workers": payload.get("prepare_workers"),
        "prepare_force": payload.get("prepare_force"),
        "offline_variant_count": payload.get("offline_variant_count"),
        "synthesis_seed": payload.get("synthesis_seed"),
        "validation_steps": payload.get("validation_steps"),
        "num_validation_images": payload.get("num_validation_images"),
        "num_inference_steps": payload.get("num_inference_steps"),
        "benchmark_inference_steps": benchmark_steps,
        "train_fast_validation": payload.get("train_fast_validation"),
        "train_validation_metrics": payload.get("train_validation_metrics"),
        "train_validation_benchmark_steps": payload.get("train_validation_benchmark_steps"),
        "use_retinex": payload.get("use_retinex"),
        "conditioning_space": payload.get("conditioning_space"),
        "inject_mode": payload.get("inject_mode"),
        "decom_variant": payload.get("decom_variant"),
        "condition_variant": payload.get("condition_variant"),
        "prepared_train_resolution": payload.get("prepared_train_resolution"),
        "semantic_backbone": payload.get("semantic_backbone"),
        "nr_metric": payload.get("nr_metric"),
        "attention_backend": payload.get("attention_backend"),
        "use_torch_compile": payload.get("use_torch_compile"),
        "torch_compile_mode": payload.get("torch_compile_mode"),
        "enable_xformers_memory_efficient_attention": payload.get("enable_xformers_memory_efficient_attention"),
        "resolved_unet_backend": payload.get("unet_backend_resolved_backend"),
        "resume": payload.get("resume"),
    }
    return summary


def runtime_summary_lines(summary: dict[str, Any], prefix: str = "[config-summary]") -> list[str]:
    return [
        f"{prefix} mode={summary.get('mode')} config={summary.get('config_name')} config_path={summary.get('config_path')}",
        f"{prefix} data_dir={summary.get('data_dir')} prepared_cache_dir={summary.get('prepared_cache_dir')} output_dir={summary.get('output_dir')}",
        f"{prefix} batch_size={summary.get('batch_size')} grad_accum={summary.get('gradient_accumulation_steps')} "
        f"effective_batch={summary.get('effective_batch_size')} min_effective_batch={summary.get('min_effective_batch_size')}",
        f"{prefix} loader=num_workers:{summary.get('num_workers')} prefetch_factor:{summary.get('prefetch_factor')} "
        f"persistent_workers:{summary.get('persistent_workers')} pin_memory:{summary.get('pin_memory')} "
        f"decode_cache_size:{summary.get('decode_cache_size')} opencv_threads_per_worker:{summary.get('opencv_threads_per_worker')}",
        f"{prefix} validation=steps:{summary.get('validation_steps')} num_validation_images:{summary.get('num_validation_images')} "
        f"num_inference_steps:{summary.get('num_inference_steps')} benchmark_inference_steps:{summary.get('benchmark_inference_steps')}",
        f"{prefix} train_validation=fast:{summary.get('train_fast_validation')} "
        f"metrics:{summary.get('train_validation_metrics')} steps:{summary.get('train_validation_benchmark_steps')}",
        f"{prefix} model=use_retinex:{summary.get('use_retinex')} decom_variant:{summary.get('decom_variant')} "
        f"condition_variant:{summary.get('condition_variant')} conditioning_space:{summary.get('conditioning_space')} "
        f"inject_mode:{summary.get('inject_mode')} prepared_train_resolution:{summary.get('prepared_train_resolution')}",
        f"{prefix} backend=requested:{summary.get('attention_backend')} compile:{summary.get('use_torch_compile')} "
        f"compile_mode:{summary.get('torch_compile_mode')} xformers_requested:{summary.get('enable_xformers_memory_efficient_attention')} "
        f"resolved:{summary.get('resolved_unet_backend')}",
    ]


def print_runtime_summary(args: argparse.Namespace | SimpleNamespace | dict[str, Any], print_fn=print) -> dict[str, Any]:
    summary = build_runtime_summary(args)
    for line in runtime_summary_lines(summary):
        print_fn(line)
    return summary


def serialize_args(args: argparse.Namespace | SimpleNamespace | dict[str, Any]) -> dict[str, Any]:
    if isinstance(args, dict):
        payload = dict(args)
    else:
        payload = dict(vars(args))

    serializable: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            serializable[key] = value
        elif isinstance(value, list):
            serializable[key] = value
        elif isinstance(value, dict):
            serializable[key] = value
        else:
            serializable[key] = str(value)
    return serializable


def build_preview_namespace(config_path: str, overrides: dict[str, Any]) -> SimpleNamespace:
    payload = load_config_defaults(config_path)
    payload.update(overrides)
    payload["config"] = str(Path(resolve_config_path(config_path)))
    payload["config_name"] = payload.get("config_name", Path(payload["config"]).stem)
    payload["effective_batch_size"] = effective_batch_size(
        payload.get("batch_size"),
        payload.get("gradient_accumulation_steps"),
    )
    return SimpleNamespace(**payload)


def summary_as_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2)
