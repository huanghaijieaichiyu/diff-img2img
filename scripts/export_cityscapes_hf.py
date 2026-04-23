from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cityscapes_hf import export_cityscapes_dataset, export_cityscapes_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Cityscapes images and semantic masks into a simple images+masks layout."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/path/to/cityscapes",
        help="Root directory of the Cityscapes dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/path/to/cityscapes_exported",
        help="Output root for exported images and masks.",
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
        help="Maximum number of test samples to export, if present.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing exported files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    counts = export_cityscapes_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        mode=args.mode,
        target_type=args.target_type,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        overwrite=bool(args.overwrite),
    )

    for split, count in counts.items():
        print(f"exported {count} samples for {split}")
    print(f"Cityscapes export completed to {Path(args.output_root).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
