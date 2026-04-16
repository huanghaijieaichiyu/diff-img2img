from __future__ import annotations

from utils.project_config import (
    build_preview_namespace,
    build_runtime_summary,
    load_config_defaults,
    load_preset_summary,
    resolve_config_path,
)


def test_resolve_config_path_for_named_preset():
    assert resolve_config_path("middle").endswith("configs/train/middle.yaml")
    assert resolve_config_path("small_throughput").endswith("configs/train/small_throughput.yaml")


def test_load_config_defaults_exposes_flattened_values():
    defaults = load_config_defaults("small")
    assert defaults["config_name"] == "small"
    assert defaults["condition_variant"] == "small_v2"
    assert defaults["decom_variant"] == "naf_lite"


def test_build_preview_namespace_and_runtime_summary():
    preview_args = build_preview_namespace(
        "middle",
        {
            "mode": "train",
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/run",
            "batch_size": 3,
            "gradient_accumulation_steps": 5,
            "prepared_cache_dir": "/tmp/data/.prepared",
        },
    )
    summary = build_runtime_summary(preview_args)
    assert summary["config_name"] == "middle"
    assert summary["effective_batch_size"] == 15
    assert summary["prepared_cache_dir"] == "/tmp/data/.prepared"
    assert summary["train_fast_validation"] is None


def test_load_preset_summary_matches_yaml_shape():
    summary = load_preset_summary("middle")
    assert summary["name"] == "middle"
    assert summary["effective_batch"] == summary["batch_size"] * summary["gradient_accumulation_steps"]


def test_runtime_summary_includes_train_validation_fields():
    summary = build_runtime_summary(
        {
            "mode": "train",
            "config": "middle",
            "config_name": "middle",
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/run",
            "train_fast_validation": True,
            "train_validation_metrics": ["psnr", "ssim"],
            "train_validation_benchmark_steps": [8],
        }
    )
    assert summary["train_fast_validation"] is True
    assert summary["train_validation_metrics"] == ["psnr", "ssim"]
    assert summary["train_validation_benchmark_steps"] == [8]


def test_runtime_summary_includes_backend_fields():
    summary = build_runtime_summary(
        {
            "mode": "train",
            "config": "small",
            "config_name": "small",
            "attention_backend": "auto",
            "use_torch_compile": True,
            "torch_compile_mode": "reduce-overhead",
            "enable_xformers_memory_efficient_attention": True,
            "unet_backend_resolved_backend": "xformers",
        }
    )

    assert summary["attention_backend"] == "auto"
    assert summary["use_torch_compile"] is True
    assert summary["torch_compile_mode"] == "reduce-overhead"
    assert summary["enable_xformers_memory_efficient_attention"] is True
    assert summary["resolved_unet_backend"] == "xformers"


def test_runtime_summary_includes_inject_mode_and_prepared_train_resolution():
    summary = build_runtime_summary(
        {
            "mode": "train",
            "config": "small_throughput",
            "config_name": "small_throughput",
            "inject_mode": "concat_pyramid",
            "prepared_train_resolution": 256,
        }
    )

    assert summary["inject_mode"] == "concat_pyramid"
    assert summary["prepared_train_resolution"] == 256
