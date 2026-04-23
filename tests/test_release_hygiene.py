from pathlib import Path
import subprocess

from start_train import build_command, create_parser
from utils.video_writer import _build_parser as build_video_writer_parser


def test_start_train_uses_sanitized_dataset_placeholder() -> None:
    args = create_parser().parse_args([])
    assert args.data_dir == "/path/to/dataset"


def test_start_train_build_command_keeps_prepare_on_train_enabled() -> None:
    args = create_parser().parse_args(
        ["--config", "middle", "--data-dir", "/path/to/dataset", "--output-dir", "runs/exp"]
    )
    command = build_command(args)
    assert "--prepare-on-train" in command


def test_video_writer_cli_requires_explicit_paths() -> None:
    parser = build_video_writer_parser()
    args = parser.parse_args(["frames", "output.mp4", "--fps", "24"])
    assert args.image_path == "frames"
    assert args.video_path == "output.mp4"
    assert args.fps == 24


def test_prepare_data_module_is_not_gitignored() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "check-ignore", "-q", "datasets/prepare_data.py"],
        cwd=repo_root,
        check=False,
    )
    assert result.returncode == 1
