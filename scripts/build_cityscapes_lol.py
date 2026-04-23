from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cityscapes_hf import export_cityscapes_dataset
from utils.masked_semantic_dataset import discover_masked_samples, process_masked_samples
from utils.semantic_night_synthesis import (
    CITYSCAPES_PRECOMPUTED_MODEL_ID,
    DEFAULT_SEMANTIC_PROFILE,
    RoadSceneNightSynthesizer,
    build_semantic_cache_from_precomputed,
    default_sky_asset_dir,
    semantic_contract_hash,
    semantic_label_space_for_model_id,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _write_usage_notes(output_root: Path, cache_dirs: dict[str, Path], *, diffimg_layout: bool) -> None:
    semantic_label_space = semantic_label_space_for_model_id(CITYSCAPES_PRECOMPUTED_MODEL_ID)
    payload = {
        "semantic_model_id": CITYSCAPES_PRECOMPUTED_MODEL_ID,
        "semantic_label_space": semantic_label_space,
        "semantic_contract_hash": semantic_contract_hash(semantic_label_space),
        "semantic_profile": DEFAULT_SEMANTIC_PROFILE,
        "semantic_device": "cpu",
        "semantic_cache_dirs": {key: str(path.resolve()) for key, path in cache_dirs.items()},
        "dataset_layout": "diffimg" if diffimg_layout else "cityscapes",
        "train_link_requires_diffimg_layout": True,
    }
    (output_root / "masked_semantic_night_usage.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )




def _image_entries_hashes(image_entries: list[tuple[str, Path]]) -> tuple[str, str]:
    index_digest = hashlib.sha256()
    stat_digest = hashlib.sha256()
    for image_id, image_path in image_entries:
        resolved = image_path.resolve()
        index_digest.update(image_id.encode("utf-8"))
        index_digest.update(b"\0")
        index_digest.update(str(resolved).encode("utf-8"))
        index_digest.update(b"\n")
        try:
            stat = resolved.stat()
            size = int(stat.st_size)
            mtime_ns = int(stat.st_mtime_ns)
        except OSError:
            size = -1
            mtime_ns = -1
        stat_digest.update(image_id.encode("utf-8"))
        stat_digest.update(b"\0")
        stat_digest.update(str(size).encode("utf-8"))
        stat_digest.update(b":")
        stat_digest.update(str(mtime_ns).encode("utf-8"))
        stat_digest.update(b"\n")
    return index_digest.hexdigest(), stat_digest.hexdigest()

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
    cache_dirs: dict[str, Path] = {}

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

        if split in {"train", "val"}:
            cache_dir = output_root / f"semantic_cache_{split}"
            image_entries = [
                (sample.sample_id, out_subset_root / "high" / f"{sample.sample_id}.png")
                for sample in samples
            ]
            masks_by_id = {}
            for sample in samples:
                mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise RuntimeError(f"Failed to read mask for semantic cache: {sample.mask_path}")
                if mask.ndim == 3:
                    mask = mask[..., 0]
                masks_by_id[sample.sample_id] = mask.astype("uint8")
            source_image_index_hash, source_image_stat_hash = _image_entries_hashes(image_entries)
            build_semantic_cache_from_precomputed(
                image_entries,
                masks_by_id=masks_by_id,
                cache_dir=cache_dir,
                model_id=CITYSCAPES_PRECOMPUTED_MODEL_ID,
                device="cpu",
                profile=DEFAULT_SEMANTIC_PROFILE,
                source_image_index_hash=source_image_index_hash,
                source_image_stat_hash=source_image_stat_hash,
                log_fn=lambda message, split=split: _log(f"{split} cache: {message}"),
            )
            cache_dirs[split] = cache_dir

    _write_usage_notes(output_root, cache_dirs, diffimg_layout=bool(args.diffimg_layout))
    _log(f"Built Cityscapes LoL dataset at {output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
