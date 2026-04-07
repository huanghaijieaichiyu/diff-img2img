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


def test_load_preset_summary_matches_yaml_shape():
    summary = load_preset_summary("middle")
    assert summary["name"] == "middle"
    assert summary["effective_batch"] == summary["batch_size"] * summary["gradient_accumulation_steps"]
