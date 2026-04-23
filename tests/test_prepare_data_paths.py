from pathlib import Path

from datasets.prepare_data import resolve_eval_pair_dirs, resolve_training_high_dir


def test_resolve_training_high_dir_supports_diffimg_layout(tmp_path: Path) -> None:
    high_dir = tmp_path / "our485" / "high"
    high_dir.mkdir(parents=True)

    assert resolve_training_high_dir(tmp_path) == high_dir


def test_resolve_training_high_dir_supports_train_layout(tmp_path: Path) -> None:
    high_dir = tmp_path / "train" / "high"
    high_dir.mkdir(parents=True)

    assert resolve_training_high_dir(tmp_path) == high_dir


def test_resolve_eval_pair_dirs_supports_eval15_layout(tmp_path: Path) -> None:
    low_dir = tmp_path / "eval15" / "low"
    high_dir = tmp_path / "eval15" / "high"
    low_dir.mkdir(parents=True)
    high_dir.mkdir(parents=True)

    assert resolve_eval_pair_dirs(tmp_path) == (low_dir, high_dir)


def test_resolve_eval_pair_dirs_supports_val_layout(tmp_path: Path) -> None:
    low_dir = tmp_path / "val" / "low"
    high_dir = tmp_path / "val" / "high"
    low_dir.mkdir(parents=True)
    high_dir.mkdir(parents=True)

    assert resolve_eval_pair_dirs(tmp_path) == (low_dir, high_dir)
