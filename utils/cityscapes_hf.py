from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import Cityscapes

VALID_CITYSCAPES_MODES = {"fine", "coarse"}
VALID_CITYSCAPES_SPLITS = {"train", "val", "test"}
HF_PARQUET_DIRNAME = "data"


@dataclass(frozen=True)
class CityscapesSample:
    split: str
    sample_id: str
    image_rgb: np.ndarray
    semantic_label_ids: np.ndarray
    source_path: str
    mask_source_path: str


def _normalize_split(split: str) -> str:
    normalized = str(split).lower()
    if normalized not in VALID_CITYSCAPES_SPLITS:
        raise ValueError(
            f"Unsupported Cityscapes split: {split!r}. Expected one of {sorted(VALID_CITYSCAPES_SPLITS)}"
        )
    return normalized


def _sample_id_from_path(path: str, root: str | Path) -> str:
    path_obj = Path(path)
    root_path = Path(root)
    try:
        relative = path_obj.relative_to(root_path)
    except Exception:
        relative = path_obj

    parts = list(relative.parts)
    if parts and parts[0] in {"leftImg8bit", "gtFine"}:
        parts = parts[1:]
    if len(parts) >= 2 and parts[0] in VALID_CITYSCAPES_SPLITS:
        parts = parts[1:]
    sample_path = Path(*parts).with_suffix("")
    return sample_path.as_posix().replace("/", "__")


def _decode_image_dict(payload: dict, mode: str = "RGB") -> np.ndarray:
    image_bytes = payload.get("bytes") if isinstance(payload, dict) else None
    if image_bytes is None:
        raise ValueError("Expected HuggingFace image payload with a 'bytes' field")
    image = Image.open(BytesIO(image_bytes))
    if mode:
        image = image.convert(mode)
    return np.asarray(image, dtype=np.uint8)


def _is_hf_parquet_layout(dataset_root: Path) -> bool:
    data_dir = dataset_root / HF_PARQUET_DIRNAME
    return data_dir.exists() and any(data_dir.glob("*.parquet"))


def _iter_hf_parquet_samples(dataset_root: Path, split: str, limit: int | None = None) -> Iterable[CityscapesSample]:
    data_dir = dataset_root / HF_PARQUET_DIRNAME
    parquet_paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet shards found for split={split!r} under {data_dir}")

    emitted = 0
    for shard_index, parquet_path in enumerate(parquet_paths):
        frame = pd.read_parquet(parquet_path)
        for row_index, row in frame.iterrows():
            if limit is not None and emitted >= int(limit):
                return
            image_rgb = _decode_image_dict(row["image"], mode="RGB")
            semantic_label_ids = _decode_image_dict(row["semantic_segmentation"], mode="L")
            sample_id = f"{split}__shard{shard_index:02d}__row{int(row_index):05d}"
            yield CityscapesSample(
                split=split,
                sample_id=sample_id,
                image_rgb=image_rgb,
                semantic_label_ids=semantic_label_ids,
                source_path=f"{parquet_path}#row={int(row_index)}:image",
                mask_source_path=f"{parquet_path}#row={int(row_index)}:semantic_segmentation",
            )
            emitted += 1


def iter_cityscapes_samples(
    dataset_root: str | Path,
    split: str = "train",
    mode: str = "fine",
    target_type: str = "semantic",
    limit: int | None = None,
) -> Iterable[CityscapesSample]:
    split = _normalize_split(split)
    mode = str(mode).lower()
    if mode not in VALID_CITYSCAPES_MODES:
        raise ValueError(
            f"Unsupported Cityscapes mode: {mode!r}. Expected one of {sorted(VALID_CITYSCAPES_MODES)}"
        )

    dataset_root = Path(dataset_root)
    if _is_hf_parquet_layout(dataset_root):
        yield from _iter_hf_parquet_samples(dataset_root, split=split, limit=limit)
        return

    dataset = Cityscapes(dataset_root, split=split, mode=mode, target_type=target_type)
    count = 0
    for index in range(len(dataset)):
        if limit is not None and count >= int(limit):
            break
        image, target = dataset[index]
        image_path = dataset.images[index]
        mask_path = dataset.targets[index][0] if isinstance(dataset.targets[index], list) else dataset.targets[index]
        image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        label_ids = np.asarray(target, dtype=np.uint8)
        sample_id = _sample_id_from_path(image_path, dataset_root)
        yield CityscapesSample(
            split=split,
            sample_id=sample_id,
            image_rgb=image_rgb,
            semantic_label_ids=label_ids,
            source_path=str(image_path),
            mask_source_path=str(mask_path),
        )
        count += 1


def export_cityscapes_split(
    dataset_root: str | Path,
    output_root: str | Path,
    split: str = "train",
    mode: str = "fine",
    target_type: str = "semantic",
    overwrite: bool = False,
    limit: int | None = None,
) -> list[str]:
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    image_root = output_root / split / "images"
    mask_root = output_root / split / "masks"
    image_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    for sample in iter_cityscapes_samples(
        dataset_root,
        split=split,
        mode=mode,
        target_type=target_type,
        limit=limit,
    ):
        image_path = image_root / f"{sample.sample_id}.png"
        mask_path = mask_root / f"{sample.sample_id}.png"

        if overwrite or not image_path.exists():
            Image.fromarray(sample.image_rgb).save(image_path)
        if overwrite or not mask_path.exists():
            Image.fromarray(sample.semantic_label_ids).save(mask_path)
        exported.append(sample.sample_id)

    return exported


def export_cityscapes_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    mode: str = "fine",
    target_type: str = "semantic",
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    output_root = Path(output_root)
    dataset_root = Path(dataset_root)
    counts: dict[str, int] = {}
    counts["train"] = len(
        export_cityscapes_split(
            dataset_root,
            output_root,
            split="train",
            mode=mode,
            target_type=target_type,
            overwrite=overwrite,
            limit=train_limit,
        )
    )
    counts["val"] = len(
        export_cityscapes_split(
            dataset_root,
            output_root,
            split="val",
            mode=mode,
            target_type=target_type,
            overwrite=overwrite,
            limit=val_limit,
        )
    )
    if _is_hf_parquet_layout(dataset_root) or (dataset_root / "leftImg8bit" / "test").exists():
        try:
            counts["test"] = len(
                export_cityscapes_split(
                    dataset_root,
                    output_root,
                    split="test",
                    mode=mode,
                    target_type=target_type,
                    overwrite=overwrite,
                    limit=test_limit,
                )
            )
        except FileNotFoundError:
            pass
    return counts
