from pathlib import Path

import pytest

from utils.project_config import resolve_repo_config_path


def test_resolve_repo_config_path_accepts_preset_name() -> None:
    resolved = resolve_repo_config_path("middle", repo_root=Path.cwd())
    assert resolved.name == "middle.yaml"


def test_resolve_repo_config_path_accepts_repo_relative_path() -> None:
    resolved = resolve_repo_config_path("configs/train/small.yaml", repo_root=Path.cwd())
    assert resolved.name == "small.yaml"


def test_resolve_repo_config_path_rejects_outside_repo_configs(tmp_path: Path) -> None:
    outside = tmp_path / "external.yaml"
    outside.write_text("meta: {}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        resolve_repo_config_path(str(outside), repo_root=Path.cwd())
