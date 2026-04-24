from __future__ import annotations

import re
from pathlib import Path


_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_step(checkpoint_path: str | Path) -> int | None:
    match = _CHECKPOINT_DIR_RE.match(Path(checkpoint_path).name)
    if match is None:
        return None
    return int(match.group(1))


def list_checkpoint_dirs(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []

    checkpoints = [
        item
        for item in root.iterdir()
        if item.is_dir() and checkpoint_step(item) is not None
    ]
    return sorted(checkpoints, key=lambda item: checkpoint_step(item) or -1)


def resolve_resume_checkpoint(resume: str | None, output_dir: str | Path) -> Path | None:
    if resume is None:
        return None

    resume_value = str(resume).strip()
    if not resume_value:
        return None

    if resume_value == "latest":
        checkpoints = list_checkpoint_dirs(output_dir)
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints matching checkpoint-<step> found under {Path(output_dir)}"
            )
        return checkpoints[-1]

    checkpoint_path = Path(resume_value)
    if not checkpoint_path.is_absolute():
        output_checkpoint = Path(output_dir) / checkpoint_path
        if checkpoint_path.parent == Path(".") and output_checkpoint.exists():
            checkpoint_path = output_checkpoint
        elif not checkpoint_path.exists() and output_checkpoint.exists():
            checkpoint_path = output_checkpoint

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Resume checkpoint not found or is not a directory: {resume_value}"
        )

    if checkpoint_step(checkpoint_path) is None:
        raise ValueError(
            f"Resume checkpoint directory must be named checkpoint-<step>: {checkpoint_path}"
        )

    return checkpoint_path
