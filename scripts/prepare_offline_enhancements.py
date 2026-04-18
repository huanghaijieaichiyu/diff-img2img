#!/usr/bin/env python3
"""
Offline enhancement cache builder for Diff-Img2Img.

This script prepares the multi-variant low-light cache once before training.
It intentionally does not run as part of the normal training launch path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_config import load_config_defaults, resolve_config_path
from datasets.prepare_data import prepare_training_data  # noqa: E402


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build offline training enhancements cache")
    parser.add_argument("--config", type=str, default="small_throughput", help="Preset name or YAML config path")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--prepared-cache-dir", type=str, default=None, help="Output prepared cache directory")
    parser.add_argument("--variant-count", type=int, default=None, help="How many low-light variants to synthesize per image")
    parser.add_argument("--synthesis-seed", type=int, default=None, help="Base random seed for offline synthesis")
    parser.add_argument("--prepared-train-resolution", type=int, default=None, help="Optional resized train cache resolution")
    parser.add_argument("--darker-ranges", type=str, default=None, help="JSON/YAML dict overriding Darker parameter ranges")
    parser.add_argument("--degradation-backend", type=str, choices=["opencv", "torch"], default=None,
                        help="Backend used for offline low-light synthesis")
    parser.add_argument("--prepare-workers", type=int, default=None, help="Worker count for offline synthesis")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False, help="Force rebuilding the cache")
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    defaults = load_config_defaults(args.config)
    resolved_config = resolve_config_path(args.config)

    data_dir = args.data_dir or defaults.get("data_dir")
    if not data_dir:
        raise SystemExit("data_dir is required")

    prepared_cache_dir = args.prepared_cache_dir or defaults.get("prepared_cache_dir")
    variant_count = args.variant_count or int(defaults.get("offline_variant_count", 3))
    synthesis_seed = args.synthesis_seed if args.synthesis_seed is not None else int(defaults.get("synthesis_seed", 42))
    prepared_train_resolution = args.prepared_train_resolution
    if prepared_train_resolution is None:
        prepared_train_resolution = defaults.get("prepared_train_resolution")
    darker_ranges = args.darker_ranges if args.darker_ranges is not None else defaults.get("darker_ranges")
    degradation_backend = args.degradation_backend or defaults.get("degradation_backend") or "torch"
    prepare_workers = args.prepare_workers or int(defaults.get("prepare_workers") or defaults.get("num_workers") or 4)

    manifest_path = prepare_training_data(
        data_dir=data_dir,
        prepared_cache_dir=prepared_cache_dir,
        variant_count=variant_count,
        synthesis_seed=synthesis_seed,
        darker_ranges=darker_ranges,
        degradation_backend=degradation_backend,
        prepare_workers=prepare_workers,
        prepared_train_resolution=prepared_train_resolution,
        force=bool(args.force),
    )

    print(f"[offline-prepare] config={resolved_config} backend={degradation_backend} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
