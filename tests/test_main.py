from __future__ import annotations

from types import SimpleNamespace

from main import _apply_cross_mount_throughput_defaults


def test_cross_mount_throughput_defaults_apply_when_cli_did_not_override():
    args = SimpleNamespace(
        mode="train",
        data_dir="/mnt/f/datasets/kitti_LOL",
        prepared_cache_dir="/mnt/f/datasets/kitti_LOL/.prepared",
        prefetch_factor=4,
        decode_cache_size=256,
        validation_steps=1000,
        num_validation_images=8,
        _explicit_cli_dests=set(),
    )

    _apply_cross_mount_throughput_defaults(args)

    assert args.prefetch_factor == 2
    assert args.decode_cache_size == 32
    assert args.validation_steps == 2000
    assert args.num_validation_images == 6
    assert args.cross_mount_tuning_applied == {
        "prefetch_factor": 2,
        "decode_cache_size": 32,
        "validation_steps": 2000,
        "num_validation_images": 6,
    }


def test_cross_mount_throughput_defaults_respect_explicit_cli_overrides():
    args = SimpleNamespace(
        mode="train",
        data_dir="/mnt/f/datasets/kitti_LOL",
        prepared_cache_dir="/mnt/f/datasets/kitti_LOL/.prepared",
        prefetch_factor=4,
        decode_cache_size=256,
        validation_steps=1000,
        num_validation_images=8,
        _explicit_cli_dests={"prefetch_factor", "num_validation_images"},
    )

    _apply_cross_mount_throughput_defaults(args)

    assert args.prefetch_factor == 4
    assert args.decode_cache_size == 32
    assert args.validation_steps == 2000
    assert args.num_validation_images == 8
    assert args.cross_mount_tuning_applied == {
        "decode_cache_size": 32,
        "validation_steps": 2000,
    }
