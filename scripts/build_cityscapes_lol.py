from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cityscapes_hf import export_cityscapes_dataset
from utils.masked_semantic_dataset import discover_masked_samples, process_masked_samples
from utils.semantic_night_synthesis import (
    DEFAULT_SEMANTIC_PROFILE,
    RoadSceneNightSynthesizer,
    default_sky_asset_dir,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _subset_mapping(diffimg_layout: bool) -> dict[str, str]:
    if diffimg_layout:
        return {"train": "our485", "val": "eval15", "test": "test"}
    return {"train": "train", "val": "val", "test": "test"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a complete Cityscapes LoL dataset by exporting Cityscapes and generating low-light/night data."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/path/to/cityscapes",
        help="Root directory of the raw Cityscapes dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/path/to/cityscapes_lol",
        help="Destination root for the generated Cityscapes LoL dataset.",
    )
    parser.add_argument(
        "--export-root",
        type=str,
        default=None,
        help="Optional intermediate export root for Cityscapes images+masks layout.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fine",
        choices=["fine", "coarse"],
        help="Cityscapes annotation mode.",
    )
    parser.add_argument(
        "--target-type",
        type=str,
        default="semantic",
        choices=["semantic"],
        help="Target type to export.",
    )
    parser.add_argument(
        "--include-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also include the Cityscapes test split when exporting and processing.",
    )
    parser.add_argument(
        "--diffimg-layout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Output dataset in DiffImg layout (our485/eval15) instead of Cityscapes train/val/test.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Maximum number of train samples to export.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Maximum number of val samples to export.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Maximum number of test samples to export.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing exported files and generated outputs.",
    )
    parser.add_argument(
        "--sky-asset-dir",
        type=str,
        default=default_sky_asset_dir(),
        help="Night sky asset directory used by the synthesizer.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    export_root = Path(args.export_root) if args.export_root else output_root / "exported"
    output_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)

    _log(f"Exporting Cityscapes dataset from {dataset_root} to {export_root}")
    export_cityscapes_dataset(
        dataset_root=dataset_root,
        output_root=export_root,
        mode=args.mode,
        target_type=args.target_type,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit if args.include_test else None,
        overwrite=args.overwrite,
    )

    synthesizer = RoadSceneNightSynthesizer(
        sky_asset_dir=args.sky_asset_dir,
        randomize=True,
        profile=DEFAULT_SEMANTIC_PROFILE,
    )

    mapping = _subset_mapping(bool(args.diffimg_layout))
    splits = ["train", "val"] + (["test"] if args.include_test else [])
    for split in splits:
        subset = mapping.get(split, split)
        split_root = export_root / split
        if not split_root.exists():
            if split == "test":
                continue
            raise FileNotFoundError(f"Expected split directory not found: {split_root}")

        samples = discover_masked_samples(split_root)
        out_subset_root = output_root / subset
        count = process_masked_samples(
            samples,
            output_root=out_subset_root,
            synthesizer=synthesizer,
            overwrite=bool(args.overwrite),
            log_fn=lambda message, split=split: _log(f"{split}: {message}"),
        )
        _log(f"Processed {count} samples for split={split} -> subset={subset}")

    _log(f"Built Cityscapes LoL dataset at {output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
