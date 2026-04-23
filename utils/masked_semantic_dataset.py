from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import shutil

import cv2
import numpy as np

from utils.semantic_night_synthesis import RoadSceneNightSynthesizer

VALID_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


@dataclass(frozen=True)
class MaskedSample:
    sample_id: str
    image_path: Path
    mask_path: Path


def discover_masked_samples(
    split_root: str | Path,
    image_dirname: str = "images",
    mask_dirname: str = "masks",
) -> list[MaskedSample]:
    split_root = Path(split_root)
    image_root = split_root / image_dirname
    mask_root = split_root / mask_dirname
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_root}")

    samples: list[MaskedSample] = []
    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        sample_id = image_path.relative_to(image_root).with_suffix("").as_posix().replace("/", "__")
        mask_path = mask_root / image_path.relative_to(image_root).with_suffix(".png")
        if not mask_path.exists():
            fallback = mask_root / f"{sample_id}.png"
            if fallback.exists():
                mask_path = fallback
            else:
                raise FileNotFoundError(f"Mask not found for sample {sample_id}: {mask_path}")
        samples.append(MaskedSample(sample_id=sample_id, image_path=image_path, mask_path=mask_path))

    if not samples:
        raise RuntimeError(f"No image files found under {image_root}")
    return samples


def process_masked_samples(
    samples: list[MaskedSample],
    output_root: str | Path,
    synthesizer: RoadSceneNightSynthesizer,
    overwrite: bool = False,
    log_fn: Callable[[str], None] | None = None,
    copy_high: bool = True,
    copy_semantic: bool = True,
) -> int:
    output_root = Path(output_root)
    high_root = output_root / "high"
    low_root = output_root / "low"
    semantic_root = output_root / "semantic"
    high_root.mkdir(parents=True, exist_ok=True)
    low_root.mkdir(parents=True, exist_ok=True)
    semantic_root.mkdir(parents=True, exist_ok=True)

    count = 0
    total = len(samples)
    for sample in samples:
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {sample.image_path}")
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {sample.mask_path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        confidence = np.ones(mask.shape, dtype=np.float32)

        filename = f"{sample.sample_id}.png"
        high_path = high_root / filename
        low_path = low_root / filename
        semantic_path = semantic_root / filename

        if copy_high and (overwrite or not high_path.exists()):
            shutil.copy2(sample.image_path, high_path)

        if copy_semantic and (overwrite or not semantic_path.exists()):
            if not cv2.imwrite(str(semantic_path), mask.astype(np.uint8)):
                raise RuntimeError(f"Failed to write semantic mask: {semantic_path}")

        if overwrite or not low_path.exists():
            low_tensor = synthesizer.synthesize(image_rgb, mask.astype(np.uint8), confidence)
            low_rgb = (
                low_tensor.permute(1, 2, 0)
                .cpu()
                .numpy()
                .clip(0.0, 1.0) * 255.0
            ).astype(np.uint8)
            low_bgr = cv2.cvtColor(low_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(low_path), low_bgr):
                raise RuntimeError(f"Failed to write low image: {low_path}")

        count += 1
        if log_fn and (count == 1 or count == total or count % 200 == 0):
            log_fn(f"processed {count}/{total}")

    return count
