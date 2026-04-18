from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from datasets.data_set import LowLightDataset
from datasets.prepare_data import (
    load_manifest_entries,
    load_manifest_info,
    manifest_info_path,
    prepare_training_data,
    validate_prepared_cache,
)


def _write_rgb_image(path: Path, value: int) -> None:
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_prepare_training_data_writes_manifest_info_and_validates(tmp_path):
    data_dir = tmp_path / "dataset"
    _write_rgb_image(data_dir / "our485" / "high" / "0.png", 96)
    _write_rgb_image(data_dir / "our485" / "high" / "1.png", 144)

    manifest_path = prepare_training_data(
        data_dir,
        None,
        variant_count=1,
        synthesis_seed=7,
        darker_ranges=None,
        prepare_workers=1,
        force=False,
    )

    info_path = manifest_info_path(manifest_path)
    assert Path(manifest_path).exists()
    assert info_path.exists()

    manifest_entries = load_manifest_entries(manifest_path)
    info_payload = load_manifest_info(info_path)
    assert info_payload is not None
    assert info_payload["entries"] == manifest_entries
    assert manifest_entries[0]["high_path_root"] == "data_dir"
    assert manifest_entries[0]["low_path_root"] == "prepared_cache_dir"
    assert not Path(manifest_entries[0]["high_path"]).is_absolute()
    assert not Path(manifest_entries[0]["low_path"]).is_absolute()
    assert validate_prepared_cache(
        data_dir,
        None,
        variant_count=1,
        synthesis_seed=7,
        darker_ranges=None,
    ) == manifest_path


def test_manifest_info_invalidates_when_expected_fingerprint_changes(tmp_path):
    data_dir = tmp_path / "dataset"
    _write_rgb_image(data_dir / "our485" / "high" / "0.png", 96)

    manifest_path = prepare_training_data(
        data_dir,
        None,
        variant_count=1,
        synthesis_seed=11,
        darker_ranges=None,
        prepare_workers=1,
        force=False,
    )
    assert Path(manifest_path).exists()

    assert validate_prepared_cache(
        data_dir,
        None,
        variant_count=2,
        synthesis_seed=11,
        darker_ranges=None,
    ) is None


def test_validate_prepared_cache_is_backend_specific(tmp_path):
    data_dir = tmp_path / "dataset"
    _write_rgb_image(data_dir / "our485" / "high" / "0.png", 96)

    manifest_path = prepare_training_data(
        data_dir,
        None,
        variant_count=1,
        synthesis_seed=23,
        darker_ranges=None,
        degradation_backend="torch",
        prepare_workers=1,
        force=False,
    )

    assert validate_prepared_cache(
        data_dir,
        None,
        variant_count=1,
        synthesis_seed=23,
        darker_ranges=None,
        degradation_backend="torch",
    ) == manifest_path

    assert validate_prepared_cache(
        data_dir,
        None,
        variant_count=1,
        synthesis_seed=23,
        darker_ranges=None,
        degradation_backend="opencv",
    ) is None


def test_prepare_training_data_writes_train_resolution_cache_when_requested(tmp_path):
    data_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _write_rgb_image(data_dir / "our485" / "high" / "0.png", 96)

    manifest_path = prepare_training_data(
        data_dir,
        cache_dir,
        variant_count=1,
        synthesis_seed=13,
        darker_ranges=None,
        prepare_workers=1,
        prepared_train_resolution=8,
        force=False,
    )

    manifest_entries = load_manifest_entries(manifest_path)
    entry = manifest_entries[0]
    assert entry["train_resolution"] == 8
    assert entry["train_low_path_root"] == "prepared_cache_dir"
    assert entry["train_high_path_root"] == "prepared_cache_dir"
    assert validate_prepared_cache(
        data_dir,
        cache_dir,
        variant_count=1,
        synthesis_seed=13,
        darker_ranges=None,
        prepared_train_resolution=8,
    ) == manifest_path

    dataset = LowLightDataset(
        image_dir=str(data_dir),
        img_size=8,
        phase="train",
        manifest_path=str(manifest_path),
        prepared_cache_dir=str(cache_dir),
    )
    low_path, high_path = dataset.data[0]
    assert "train_cache/8" in low_path
    assert "train_cache/8" in high_path


def test_dataset_falls_back_to_jsonl_when_manifest_info_missing(tmp_path):
    manifest_path = tmp_path / "train_manifest.jsonl"
    entries = [
        {
            "image_id": "a",
            "variant_idx": 0,
            "high_path": "/tmp/high_a.png",
            "low_path": "/tmp/low_a.png",
        },
        {
            "image_id": "b",
            "variant_idx": 0,
            "high_path": "/tmp/high_b.png",
            "low_path": "/tmp/low_b.png",
        },
    ]
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    dataset = LowLightDataset(
        image_dir=str(tmp_path),
        img_size=8,
        phase="train",
        manifest_path=str(manifest_path),
    )
    assert dataset.data == [(entry["low_path"], entry["high_path"])
                            for entry in entries]


def test_dataset_resolves_relative_manifest_entries_against_active_roots(tmp_path):
    data_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    high_path = data_dir / "our485" / "high" / "0.png"
    low_path = cache_dir / "our485" / "low" / "0__v00.png"
    _write_rgb_image(high_path, 96)
    _write_rgb_image(low_path, 24)

    manifest_path = cache_dir / "train_manifest.jsonl"
    entries = [
        {
            "image_id": "0",
            "variant_idx": 0,
            "high_path": "our485/high/0.png",
            "high_path_root": "data_dir",
            "low_path": "our485/low/0__v00.png",
            "low_path_root": "prepared_cache_dir",
        }
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    dataset = LowLightDataset(
        image_dir=str(data_dir),
        img_size=8,
        phase="train",
        manifest_path=str(manifest_path),
        prepared_cache_dir=str(cache_dir),
    )
    assert dataset.data == [
        (str(low_path.resolve()), str(high_path.resolve()))]
    assert resolve_manifest_entry_path(
        entries[0],
        "low_path",
        data_dir=str(data_dir),
        prepared_cache_dir=str(cache_dir),
    ) == str(low_path.resolve())


def test_validate_prepared_cache_survives_cache_relocation_with_relative_manifest(tmp_path):
    data_dir = tmp_path / "dataset"
    initial_cache_dir = tmp_path / "cache_a"
    relocated_cache_dir = tmp_path / "cache_b"
    _write_rgb_image(data_dir / "our485" / "high" / "0.png", 96)

    manifest_path = prepare_training_data(
        data_dir,
        initial_cache_dir,
        variant_count=1,
        synthesis_seed=5,
        darker_ranges=None,
        prepare_workers=1,
        force=False,
    )
    assert Path(manifest_path).exists()

    initial_cache_dir.rename(relocated_cache_dir)

    relocated_manifest_path = validate_prepared_cache(
        data_dir,
        relocated_cache_dir,
        variant_count=1,
        synthesis_seed=5,
        darker_ranges=None,
    )
    assert relocated_manifest_path == str(
        (relocated_cache_dir / "train_manifest.jsonl").resolve())


def test_force_prepare_skips_validate_path(tmp_path, monkeypatch):
    data_dir = tmp_path / "dataset"
    _write_rgb_image(data_dir / "our485" / "high" / "0.png", 96)

    def _raise_if_called(*args, **kwargs):  # pragma: no cover：行为断言辅助函数
        raise AssertionError(
            "validate_prepared_cache should not be called when force=True")

    monkeypatch.setattr(
        "datasets.prepare_data.validate_prepared_cache", _raise_if_called)

    manifest_path, prepared = ensure_prepared_training_data(
        data_dir,
        None,
        variant_count=1,
        synthesis_seed=3,
        darker_ranges=None,
        prepare_workers=1,
        force=True,
        prepare_on_train=True,
    )

    assert prepared is True
    assert Path(manifest_path).exists()


def test_prepare_reuses_low_cache_when_only_train_resolution_changes(tmp_path):
    data_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _write_rgb_image(data_dir / "our485" / "high" / "0.png", 96)

    prepare_training_data(
        data_dir,
        cache_dir,
        variant_count=1,
        synthesis_seed=17,
        darker_ranges=None,
        prepare_workers=1,
        prepared_train_resolution=8,
        force=False,
    )

    low_variant = cache_dir / "our485" / "low" / "0__v00.png"
    assert low_variant.exists()
    mtime_before = low_variant.stat().st_mtime

    prepare_training_data(
        data_dir,
        cache_dir,
        variant_count=1,
        synthesis_seed=17,
        darker_ranges=None,
        prepare_workers=1,
        prepared_train_resolution=12,
        force=False,
    )

    assert low_variant.exists()
    assert low_variant.stat().st_mtime == mtime_before
    assert (cache_dir / "train_cache" / "12" /
            "our485" / "low" / "0__v00.png").exists()
    assert (cache_dir / "train_cache" / "12" /
            "our485" / "high" / "0.png").exists()


def test_dataset_missing_train_dirs_do_not_raise(tmp_path):
    data_dir = tmp_path / "dataset"
    (data_dir / "our485").mkdir(parents=True, exist_ok=True)

    dataset = LowLightDataset(image_dir=str(
        data_dir), img_size=8, phase="train")
    assert len(dataset) == 0


def test_validate_cache_invalidates_when_source_file_changes_in_place(tmp_path):
    data_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    source_path = data_dir / "our485" / "high" / "0.png"
    _write_rgb_image(source_path, 90)

    manifest_path = prepare_training_data(
        data_dir,
        cache_dir,
        variant_count=1,
        synthesis_seed=19,
        darker_ranges=None,
        prepare_workers=1,
        force=False,
    )
    assert Path(manifest_path).exists()
    assert validate_prepared_cache(
        data_dir,
        cache_dir,
        variant_count=1,
        synthesis_seed=19,
        darker_ranges=None,
    ) == manifest_path

    time.sleep(0.002)
    _write_rgb_image(source_path, 140)

    assert validate_prepared_cache(
        data_dir,
        cache_dir,
        variant_count=1,
        synthesis_seed=19,
        darker_ranges=None,
    ) is None
