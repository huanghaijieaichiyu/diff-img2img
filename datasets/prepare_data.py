import hashlib
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from scripts.darker import Darker, configure_cpu_image_backend
from utils.semantic_night_synthesis import (
    CITYSCAPES_PRECOMPUTED_MODEL_ID,
    DEFAULT_SEMANTIC_MODEL_ID,
    DEFAULT_SEMANTIC_PROFILE,
    SEMANTIC_SYNTHESIS_VERSION,
    RoadSceneNightSynthesizer,
    build_semantic_cache,
    default_semantic_cache_dir,
    load_semantic_cache,
    resolve_semantic_device,
    semantic_contract_hash,
    semantic_label_space_for_model_id,
    sky_asset_dir_hash,
    validate_semantic_cache,
)
from utils.torch_low_light_degrader import TorchLowLightDegrader


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
PREPARE_VERSION = 1
MANIFEST_INFO_VERSION = 1
SOURCE_MODE = "regenerate_all"
_WORKER_DARKER: Darker | None = None
_WORKER_TORCH_DEGRADER: TorchLowLightDegrader | None = None
_WORKER_NIGHT_SYNTHESIZER: RoadSceneNightSynthesizer | None = None
_WORKER_DEGRADATION_BACKEND = "opencv"
PATH_ROOT_DATA_DIR = "data_dir"
PATH_ROOT_PREPARED_CACHE_DIR = "prepared_cache_dir"
TRAIN_CACHE_DIRNAME = "train_cache"
TRAIN_HIGH_DIR_CANDIDATES = (
    ("our485", "high"),
    ("train", "high"),
)
EVAL_DIR_CANDIDATES = (
    ("eval15", "low", "high"),
    ("val", "low", "high"),
)


def normalize_degradation_backend(degradation_backend: str | None) -> str:
    backend = str(degradation_backend or "opencv").strip().lower()
    if backend not in {"opencv", "torch"}:
        raise ValueError(
            f"Unsupported degradation_backend={degradation_backend!r}. Expected 'opencv' or 'torch'."
        )
    return backend


def normalize_semantic_profile(semantic_profile: str | None) -> str:
    profile = str(semantic_profile or DEFAULT_SEMANTIC_PROFILE).strip()
    if profile != DEFAULT_SEMANTIC_PROFILE:
        raise ValueError(
            f"Unsupported semantic_profile={semantic_profile!r}. Expected '{DEFAULT_SEMANTIC_PROFILE}'."
        )
    return profile


@dataclass(frozen=True)
class PreparedPaths:
    cache_dir: Path
    low_dir: Path
    manifest_path: Path
    meta_path: Path
    info_path: Path


def resolve_prepared_paths(data_dir: str | os.PathLike[str], prepared_cache_dir: str | os.PathLike[str] | None = None) -> PreparedPaths:
    cache_dir = Path(prepared_cache_dir) if prepared_cache_dir else Path(
        data_dir) / ".prepared"
    manifest_path = cache_dir / "train_manifest.jsonl"
    return PreparedPaths(
        cache_dir=cache_dir,
        low_dir=cache_dir / "our485" / "low",
        manifest_path=manifest_path,
        meta_path=cache_dir / "prepare_meta.json",
        info_path=manifest_info_path(manifest_path),
    )


def manifest_info_path(manifest_path: str | os.PathLike[str]) -> Path:
    manifest_path = Path(manifest_path)
    return manifest_path.with_suffix(".info.pkl")


def resolve_semantic_cache_dir(
    prepared_cache_dir: str | os.PathLike[str],
    semantic_cache_dir: str | os.PathLike[str] | None,
) -> str:
    if semantic_cache_dir:
        return _abspath(semantic_cache_dir)
    return default_semantic_cache_dir(prepared_cache_dir)


def train_cache_root(paths: PreparedPaths, train_resolution: int) -> Path:
    return paths.cache_dir / TRAIN_CACHE_DIRNAME / str(int(train_resolution)) / "our485"


def train_cache_low_dir(paths: PreparedPaths, train_resolution: int) -> Path:
    return train_cache_root(paths, train_resolution) / "low"


def train_cache_high_dir(paths: PreparedPaths, train_resolution: int) -> Path:
    return train_cache_root(paths, train_resolution) / "high"


def normalize_darker_ranges(darker_ranges: Any) -> dict[str, Any] | None:
    if darker_ranges in (None, "", {}):
        return None
    if isinstance(darker_ranges, dict):
        return darker_ranges
    if isinstance(darker_ranges, str):
        parsed = yaml.safe_load(darker_ranges)
        if not isinstance(parsed, dict):
            raise ValueError("darker_ranges must decode to a dict.")
        return parsed
    raise TypeError(f"Unsupported darker_ranges type: {type(darker_ranges)!r}")


def resolve_darker_ranges(darker_ranges: dict[str, Any] | None) -> dict[str, Any]:
    resolved = dict(Darker.DEFAULT_RANGES)
    if darker_ranges:
        resolved.update(darker_ranges)
    return resolved


def darker_ranges_hash(darker_ranges: dict[str, Any] | None) -> str:
    resolved = resolve_darker_ranges(darker_ranges)
    serialized = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _source_image_index_hash(image_entries: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for image_id, image_path in image_entries:
        digest.update(image_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_abspath(image_path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_image_stat_hash(image_entries: list[tuple[str, Path]]) -> str:
    """Hash file size and mtime to detect in-place source image updates."""
    digest = hashlib.sha256()
    for image_id, image_path in image_entries:
        try:
            stat = image_path.stat()
            size = int(stat.st_size)
            mtime_ns = int(stat.st_mtime_ns)
        except OSError:
            size = -1
            mtime_ns = -1
        digest.update(image_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("utf-8"))
        digest.update(b":")
        digest.update(str(mtime_ns).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_repo_git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        return commit or None
    except Exception:
        return None


def _build_dataset_fingerprint(data_dir: str | os.PathLike[str], image_entries: list[tuple[str, Path]]) -> dict[str, Any]:
    data_dir_abs = _abspath(data_dir)
    high_dir_abs = _abspath(resolve_training_high_dir(data_dir))
    return {
        "dataset_root": data_dir_abs,
        "source_high_dir": high_dir_abs,
        "source_image_count": len(image_entries),
        "source_image_index_hash": _source_image_index_hash(image_entries),
        "source_image_stat_hash": _source_image_stat_hash(image_entries),
        "git_commit": _resolve_repo_git_commit(),
    }


def _iter_high_images(high_dir: Path) -> list[tuple[str, Path]]:
    if not high_dir.exists():
        raise FileNotFoundError(
            f"Training high-light directory not found: {high_dir}")

    image_entries: list[tuple[str, Path]] = []
    for path in sorted(high_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        relative_stem = path.relative_to(high_dir).with_suffix(
            "").as_posix().replace("/", "__")
        image_entries.append((relative_stem, path))

    if not image_entries:
        raise RuntimeError(f"No training images found under {high_dir}")
    return image_entries


def resolve_training_high_dir(data_dir: str | os.PathLike[str]) -> Path:
    data_root = Path(data_dir)
    for parts in TRAIN_HIGH_DIR_CANDIDATES:
        candidate = data_root.joinpath(*parts)
        if candidate.exists():
            return candidate
    expected = ", ".join(str(data_root.joinpath(*parts)) for parts in TRAIN_HIGH_DIR_CANDIDATES)
    raise FileNotFoundError(
        "Training high-light directory not found. "
        f"Expected one of: {expected}"
    )


def resolve_eval_pair_dirs(data_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
    data_root = Path(data_dir)
    for subset, low_name, high_name in EVAL_DIR_CANDIDATES:
        subset_root = data_root / subset
        low_dir = subset_root / low_name
        high_dir = subset_root / high_name
        if low_dir.exists() and high_dir.exists():
            return low_dir, high_dir
    expected = ", ".join(
        str(data_root / subset / "{low,high}") for subset, _, _ in EVAL_DIR_CANDIDATES
    )
    raise FileNotFoundError(
        "Validation low/high directories not found. "
        f"Expected one of: {expected}"
    )


def _stable_seed(base_seed: int, image_id: str, variant_idx: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{image_id}:{variant_idx}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32 - 1)


def _prepare_task_payload(
    image_id: str,
    high_path: Path,
    low_path: Path,
    low_filename: str,
    variant_idx: int,
    base_seed: int,
    darker_ranges: dict[str, Any] | None,
    semantic_cache_path: str | None = None,
    semantic_label_space: str | None = None,
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "high_path": _abspath(high_path),
        "low_path": _abspath(low_path),
        "low_filename": low_filename,
        "variant_idx": variant_idx,
        "seed": _stable_seed(base_seed, image_id, variant_idx),
        "darker_ranges": darker_ranges,
        "semantic_cache_path": semantic_cache_path,
        "semantic_label_space": semantic_label_space,
    }


def _prepare_pool_initializer(
    darker_ranges: dict[str, Any] | None,
    degradation_backend: str,
    semantic_synthesis: bool = False,
    sky_asset_dir: str | None = None,
    semantic_profile: str = DEFAULT_SEMANTIC_PROFILE,
) -> None:
    global _WORKER_DARKER, _WORKER_TORCH_DEGRADER, _WORKER_NIGHT_SYNTHESIZER, _WORKER_DEGRADATION_BACKEND
    configure_cpu_image_backend(num_threads=1)
    _WORKER_DEGRADATION_BACKEND = normalize_degradation_backend(
        degradation_backend)
    if _WORKER_DEGRADATION_BACKEND == "torch":
        _WORKER_DARKER = None
        if semantic_synthesis:
            _WORKER_TORCH_DEGRADER = None
            _WORKER_NIGHT_SYNTHESIZER = RoadSceneNightSynthesizer(
                sky_asset_dir=sky_asset_dir,
                param_ranges=darker_ranges,
                randomize=True,
                profile=semantic_profile,
            )
        else:
            _WORKER_NIGHT_SYNTHESIZER = None
            _WORKER_TORCH_DEGRADER = TorchLowLightDegrader(
                randomize=True,
                param_ranges=darker_ranges,
            )
    else:
        _WORKER_NIGHT_SYNTHESIZER = None
        _WORKER_TORCH_DEGRADER = None
        _WORKER_DARKER = Darker(randomize=True, param_ranges=darker_ranges)


def _prepare_single_image(task: dict[str, Any]) -> tuple[str, int, str, bool, str]:
    global _WORKER_DARKER, _WORKER_TORCH_DEGRADER, _WORKER_NIGHT_SYNTHESIZER, _WORKER_DEGRADATION_BACKEND
    try:
        seed = int(task["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        high_image = cv2.imread(task["high_path"], cv2.IMREAD_COLOR)
        if high_image is None:
            return task["image_id"], int(task["variant_idx"]), task["low_path"], False, f"Failed to read image: {task['high_path']}"

        if _WORKER_DEGRADATION_BACKEND == "torch":
            high_rgb = cv2.cvtColor(high_image, cv2.COLOR_BGR2RGB)
            high_tensor = torch.from_numpy(
                np.ascontiguousarray(high_rgb.transpose(2, 0, 1))
            ).float() / 255.0
            semantic_cache_path = task.get("semantic_cache_path")
            if semantic_cache_path:
                synthesizer = _WORKER_NIGHT_SYNTHESIZER
                if synthesizer is None:
                    configure_cpu_image_backend(num_threads=1)
                    synthesizer = RoadSceneNightSynthesizer(
                        sky_asset_dir="",
                        param_ranges=task["darker_ranges"],
                        randomize=True,
                        profile=DEFAULT_SEMANTIC_PROFILE,
                    )
                    _WORKER_NIGHT_SYNTHESIZER = synthesizer
                label_map, confidence_map = load_semantic_cache(semantic_cache_path)
                with torch.no_grad():
                    degraded_tensor = synthesizer.synthesize(
                        high_tensor,
                        label_map,
                        confidence_map,
                        label_space=task.get("semantic_label_space"),
                    )
            else:
                torch_degrader = _WORKER_TORCH_DEGRADER
                if torch_degrader is None:
                    configure_cpu_image_backend(num_threads=1)
                    torch_degrader = TorchLowLightDegrader(
                        randomize=True,
                        param_ranges=task["darker_ranges"],
                    )
                    _WORKER_TORCH_DEGRADER = torch_degrader
                with torch.no_grad():
                    degraded_tensor = torch_degrader.degrade(high_tensor)
            degraded_rgb = degraded_tensor.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            degraded = cv2.cvtColor(degraded_rgb, cv2.COLOR_RGB2BGR)
        else:
            darker = _WORKER_DARKER
            if darker is None:
                configure_cpu_image_backend(num_threads=1)
                darker = Darker(randomize=True,
                                param_ranges=task["darker_ranges"])
                _WORKER_DARKER = darker
            degraded = darker.degrade_single(high_image)

        low_path = Path(task["low_path"])
        if not cv2.imwrite(str(low_path), degraded):
            return task["image_id"], int(task["variant_idx"]), task["low_path"], False, f"Failed to write image: {low_path}"
        return task["image_id"], int(task["variant_idx"]), task["low_path"], True, ""
    except Exception as exc:  # pragma: no cover - defensive failure path
        return task["image_id"], int(task["variant_idx"]), task["low_path"], False, str(exc)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest_entries(manifest_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def load_manifest_info(info_path: str | os.PathLike[str]) -> dict[str, Any] | None:
    info_path = Path(info_path)
    if not info_path.exists():
        return None
    with info_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload if isinstance(payload, dict) else None


def _matching_metadata(
    meta: dict[str, Any] | None,
    *,
    variant_count: int,
    synthesis_seed: int,
    darker_hash: str,
    degradation_backend: str,
    prepared_train_resolution: int | None = None,
    semantic_synthesis: bool = False,
    semantic_model_id: str | None = None,
    semantic_profile: str | None = None,
    sky_asset_hash: str | None = None,
    semantic_label_space: str | None = None,
    semantic_contract_hash_value: str | None = None,
) -> bool:
    if not meta:
        return False
    return (
        meta.get("prepare_version") == PREPARE_VERSION
        and meta.get("source_mode") == SOURCE_MODE
        and int(meta.get("variant_count", -1)) == int(variant_count)
        and int(meta.get("synthesis_seed", -1)) == int(synthesis_seed)
        and meta.get("darker_ranges_hash") == darker_hash
        and str(meta.get("degradation_backend", "opencv")) == str(degradation_backend)
        and int(meta.get("prepared_train_resolution") or 0) == int(prepared_train_resolution or 0)
        and bool(meta.get("semantic_synthesis", False)) == bool(semantic_synthesis)
        and (str(meta.get("semantic_model_id") or "") == str(semantic_model_id or "") if semantic_synthesis else True)
        and (str(meta.get("semantic_profile") or "") == str(semantic_profile or "") if semantic_synthesis else True)
        and (str(meta.get("sky_asset_dir_hash") or "") == str(sky_asset_hash or "") if semantic_synthesis else True)
        and (str(meta.get("semantic_label_space") or "") == str(semantic_label_space or "") if semantic_synthesis else True)
        and (str(meta.get("semantic_contract_hash") or "") == str(semantic_contract_hash_value or "") if semantic_synthesis else True)
        and int(meta.get("semantic_synthesis_version") or 0) == int(SEMANTIC_SYNTHESIS_VERSION if semantic_synthesis else 0)
    )


def _matching_core_metadata(
    meta: dict[str, Any] | None,
    *,
    variant_count: int,
    synthesis_seed: int,
    darker_hash: str,
    degradation_backend: str,
    semantic_synthesis: bool = False,
    semantic_model_id: str | None = None,
    semantic_profile: str | None = None,
    sky_asset_hash: str | None = None,
    semantic_label_space: str | None = None,
    semantic_contract_hash_value: str | None = None,
) -> bool:
    """Match cache identity excluding train-resolution derivative cache settings."""
    if not meta:
        return False
    return (
        meta.get("prepare_version") == PREPARE_VERSION
        and meta.get("source_mode") == SOURCE_MODE
        and int(meta.get("variant_count", -1)) == int(variant_count)
        and int(meta.get("synthesis_seed", -1)) == int(synthesis_seed)
        and meta.get("darker_ranges_hash") == darker_hash
        and str(meta.get("degradation_backend", "opencv")) == str(degradation_backend)
        and bool(meta.get("semantic_synthesis", False)) == bool(semantic_synthesis)
        and (str(meta.get("semantic_model_id") or "") == str(semantic_model_id or "") if semantic_synthesis else True)
        and (str(meta.get("semantic_profile") or "") == str(semantic_profile or "") if semantic_synthesis else True)
        and (str(meta.get("sky_asset_dir_hash") or "") == str(sky_asset_hash or "") if semantic_synthesis else True)
        and (str(meta.get("semantic_label_space") or "") == str(semantic_label_space or "") if semantic_synthesis else True)
        and (str(meta.get("semantic_contract_hash") or "") == str(semantic_contract_hash_value or "") if semantic_synthesis else True)
        and int(meta.get("semantic_synthesis_version") or 0) == int(SEMANTIC_SYNTHESIS_VERSION if semantic_synthesis else 0)
    )


def _dataset_fingerprint_mismatch_reasons(
    meta: dict[str, Any] | None,
    *,
    data_dir: str | os.PathLike[str],
    image_entries: list[tuple[str, Path]],
) -> list[str]:
    if not meta:
        return ["metadata_missing"]

    fingerprint = _build_dataset_fingerprint(data_dir, image_entries)
    reasons: list[str] = []
    for key in ("dataset_root", "source_high_dir", "source_image_count", "source_image_index_hash", "source_image_stat_hash"):
        if meta.get(key) != fingerprint.get(key):
            reasons.append(f"{key}_mismatch")
    return reasons


def _abspath(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.fspath(path))


def _relpath(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> str:
    return Path(os.path.relpath(os.fspath(path), os.fspath(root))).as_posix()


def _expected_low_filename(image_id: str, variant_idx: int) -> str:
    return f"{image_id}__v{variant_idx:02d}.png"


def _expected_low_path(low_dir: Path, image_id: str, variant_idx: int) -> str:
    return _abspath(low_dir / _expected_low_filename(image_id, variant_idx))


def resolve_manifest_entry_path(
    entry: dict[str, Any],
    field_name: str,
    *,
    data_dir: str | os.PathLike[str],
    prepared_cache_dir: str | os.PathLike[str] | None,
) -> str | None:
    raw_path = entry.get(field_name)
    if not raw_path:
        return None

    path = Path(str(raw_path))
    if path.is_absolute():
        return _abspath(path)

    root_name = entry.get(f"{field_name}_root")
    if root_name is None:
        root_name = PATH_ROOT_PREPARED_CACHE_DIR if field_name == "low_path" else PATH_ROOT_DATA_DIR

    if root_name == PATH_ROOT_DATA_DIR:
        base_dir = Path(data_dir)
    elif root_name == PATH_ROOT_PREPARED_CACHE_DIR:
        base_dir = resolve_prepared_paths(
            data_dir, prepared_cache_dir).cache_dir
    else:
        return _abspath(path)

    return _abspath(base_dir / path)


def _build_manifest_entry(
    *,
    data_dir: str | os.PathLike[str],
    paths: PreparedPaths,
    image_id: str,
    variant_idx: int,
    high_path: str | os.PathLike[str],
    prepared_train_resolution: int | None = None,
) -> dict[str, Any]:
    entry = {
        "image_id": image_id,
        "variant_idx": int(variant_idx),
        "high_path": _relpath(high_path, data_dir),
        "high_path_root": PATH_ROOT_DATA_DIR,
        "low_path": _relpath(paths.low_dir / _expected_low_filename(image_id, int(variant_idx)), paths.cache_dir),
        "low_path_root": PATH_ROOT_PREPARED_CACHE_DIR,
    }
    if prepared_train_resolution:
        train_resolution = int(prepared_train_resolution)
        entry.update(
            {
                "train_resolution": train_resolution,
                "train_high_path": _relpath(
                    train_cache_high_dir(
                        paths, train_resolution) / f"{image_id}.png",
                    paths.cache_dir,
                ),
                "train_high_path_root": PATH_ROOT_PREPARED_CACHE_DIR,
                "train_low_path": _relpath(
                    train_cache_low_dir(
                        paths, train_resolution) / _expected_low_filename(image_id, int(variant_idx)),
                    paths.cache_dir,
                ),
                "train_low_path_root": PATH_ROOT_PREPARED_CACHE_DIR,
            }
        )
    return entry


def _write_metadata(paths: PreparedPaths, payload: dict[str, Any]) -> None:
    with open(paths.meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _write_manifest_info(paths: PreparedPaths, payload: dict[str, Any]) -> None:
    with paths.info_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _build_manifest_info_payload(
    paths: PreparedPaths,
    *,
    variant_count: int,
    synthesis_seed: int,
    darker_hash: str,
    degradation_backend: str,
    source_image_index_hash: str,
    source_image_stat_hash: str,
    manifest_entries: list[dict[str, Any]],
    prepared_train_resolution: int | None,
    semantic_synthesis: bool = False,
    semantic_model_id: str | None = None,
    semantic_profile: str | None = None,
    sky_asset_hash: str | None = None,
    semantic_label_space: str | None = None,
    semantic_contract_hash_value: str | None = None,
) -> dict[str, Any]:
    return {
        "version": MANIFEST_INFO_VERSION,
        "manifest_path": _abspath(paths.manifest_path),
        "prepared_cache_dir": _abspath(paths.cache_dir),
        "variant_count": int(variant_count),
        "synthesis_seed": int(synthesis_seed),
        "darker_ranges_hash": darker_hash,
        "degradation_backend": degradation_backend,
        "source_image_index_hash": source_image_index_hash,
        "source_image_stat_hash": source_image_stat_hash,
        "prepared_train_resolution": int(prepared_train_resolution) if prepared_train_resolution else None,
        "semantic_synthesis": bool(semantic_synthesis),
        "semantic_model_id": semantic_model_id if semantic_synthesis else None,
        "semantic_profile": semantic_profile if semantic_synthesis else None,
        "sky_asset_dir_hash": sky_asset_hash if semantic_synthesis else None,
        "semantic_label_space": semantic_label_space if semantic_synthesis else None,
        "semantic_contract_hash": semantic_contract_hash_value if semantic_synthesis else None,
        "semantic_synthesis_version": int(SEMANTIC_SYNTHESIS_VERSION if semantic_synthesis else 0),
        "entries": manifest_entries,
    }


def _list_prepared_low_filenames(low_dir: Path) -> set[str]:
    if not low_dir.exists():
        return set()

    filenames: set[str] = set()
    with os.scandir(low_dir) as iterator:
        for entry in iterator:
            if not entry.is_file():
                continue
            if entry.name.lower().endswith(VALID_EXTENSIONS):
                filenames.add(entry.name)
    return filenames


def _list_prepared_png_stems(directory: Path) -> set[str]:
    if not directory.exists():
        return set()

    stems: set[str] = set()
    with os.scandir(directory) as iterator:
        for entry in iterator:
            if not entry.is_file():
                continue
            if entry.name.lower().endswith(".png"):
                stems.add(Path(entry.name).stem)
    return stems


def _progress_interval(total: int) -> int:
    if total <= 0:
        return 1
    return max(1, min(100, total // 20 or 1))


def _task_chunksize(total: int, worker_count: int) -> int:
    if total <= 0:
        return 1
    return max(1, math.ceil(total / max(1, worker_count * 8)))


def _log_prepare(message: str) -> None:
    print(f"[prepare] {message}", flush=True)


def _should_resume_from_metadata(
    meta: dict[str, Any] | None,
    *,
    variant_count: int,
    synthesis_seed: int,
    darker_hash: str,
    degradation_backend: str,
    prepared_train_resolution: int | None = None,
) -> bool:
    if not _matching_metadata(
        meta,
        variant_count=variant_count,
        synthesis_seed=synthesis_seed,
        darker_hash=darker_hash,
        degradation_backend=degradation_backend,
        prepared_train_resolution=prepared_train_resolution,
    ):
        return False
    return str(meta.get("status", "")).lower() in {"ready", "preparing", "interrupted"}


def summarize_prepared_cache(
    data_dir: str | os.PathLike[str],
    prepared_cache_dir: str | os.PathLike[str] | None,
    *,
    variant_count: int,
    synthesis_seed: int,
    darker_ranges: dict[str, Any] | None = None,
    degradation_backend: str = "opencv",
    prepared_train_resolution: int | None = None,
    semantic_synthesis: bool = False,
    semantic_model_id: str | None = None,
    semantic_profile: str | None = None,
    semantic_cache_dir: str | os.PathLike[str] | None = None,
    sky_asset_dir: str | os.PathLike[str] | None = None,
    semantic_device: str = "auto",
) -> dict[str, Any]:
    data_dir = str(Path(data_dir))
    darker_ranges = normalize_darker_ranges(darker_ranges)
    degradation_backend = normalize_degradation_backend(degradation_backend)
    paths = resolve_prepared_paths(data_dir, prepared_cache_dir)
    semantic_profile = normalize_semantic_profile(semantic_profile)
    semantic_model_id = str(semantic_model_id or DEFAULT_SEMANTIC_MODEL_ID)
    semantic_device = resolve_semantic_device(semantic_device)
    semantic_cache_dir = resolve_semantic_cache_dir(paths.cache_dir, semantic_cache_dir)
    sky_asset_hash = sky_asset_dir_hash(sky_asset_dir) if semantic_synthesis else None
    semantic_label_space = semantic_label_space_for_model_id(semantic_model_id) if semantic_synthesis else None
    semantic_contract_hash_value = semantic_contract_hash(semantic_label_space) if semantic_synthesis else None
    meta = _load_json(paths.meta_path)
    darker_hash = darker_ranges_hash(darker_ranges)

    source_error: str | None = None
    try:
        image_entries = _iter_high_images(resolve_training_high_dir(data_dir))
    except Exception as exc:
        image_entries = []
        source_error = str(exc)

    image_count = len(image_entries)
    expected_entries = image_count * int(variant_count)
    manifest_entries = 0
    if paths.manifest_path.exists():
        with paths.manifest_path.open("r", encoding="utf-8") as handle:
            manifest_entries = sum(1 for line in handle if line.strip())

    status = "missing"
    reasons: list[str] = []
    if source_error:
        reasons.append("source_images_unavailable")

    if meta:
        status = str(meta.get("status", "invalid")).lower()
        if status not in {"ready", "preparing", "interrupted"}:
            status = "invalid"
            reasons.append("metadata_status_invalid")
    if paths.manifest_path.exists() and not meta:
        status = "stale"
        reasons.append("manifest_without_metadata")

    if meta and not _matching_metadata(
        meta,
        variant_count=variant_count,
        synthesis_seed=synthesis_seed,
        darker_hash=darker_hash,
        degradation_backend=degradation_backend,
        prepared_train_resolution=prepared_train_resolution,
        semantic_synthesis=semantic_synthesis,
        semantic_model_id=semantic_model_id,
        semantic_profile=semantic_profile,
        sky_asset_hash=sky_asset_hash,
        semantic_label_space=semantic_label_space,
        semantic_contract_hash_value=semantic_contract_hash_value,
    ):
        status = "stale"
        if int(meta.get("variant_count", -1)) != int(variant_count):
            reasons.append("variant_count_mismatch")
        if int(meta.get("synthesis_seed", -1)) != int(synthesis_seed):
            reasons.append("synthesis_seed_mismatch")
        if meta.get("darker_ranges_hash") != darker_hash:
            reasons.append("darker_ranges_hash_mismatch")
        if str(meta.get("degradation_backend", "opencv")) != str(degradation_backend):
            reasons.append("degradation_backend_mismatch")
        if bool(meta.get("semantic_synthesis", False)) != bool(semantic_synthesis):
            reasons.append("semantic_synthesis_mismatch")
        if semantic_synthesis and str(meta.get("semantic_model_id") or "") != semantic_model_id:
            reasons.append("semantic_model_id_mismatch")
        if semantic_synthesis and str(meta.get("semantic_profile") or "") != semantic_profile:
            reasons.append("semantic_profile_mismatch")
        if semantic_synthesis and str(meta.get("sky_asset_dir_hash") or "") != str(sky_asset_hash or ""):
            reasons.append("sky_asset_dir_hash_mismatch")
        if semantic_synthesis and str(meta.get("semantic_label_space") or "") != str(semantic_label_space or ""):
            reasons.append("semantic_label_space_mismatch")
        if semantic_synthesis and str(meta.get("semantic_contract_hash") or "") != str(semantic_contract_hash_value or ""):
            reasons.append("semantic_contract_hash_mismatch")
    meta_train_resolution = int(
        meta.get("prepared_train_resolution") or 0) if meta else 0
    expected_train_resolution = int(prepared_train_resolution or 0)
    if meta and meta_train_resolution != expected_train_resolution:
        status = "stale"
        reasons.append("prepared_train_resolution_mismatch")

    if meta and image_entries:
        fingerprint_reasons = _dataset_fingerprint_mismatch_reasons(
            meta, data_dir=data_dir, image_entries=image_entries)
        if fingerprint_reasons:
            status = "stale"
            reasons.extend(fingerprint_reasons)

    if meta and status == "ready":
        if not paths.manifest_path.exists():
            status = "stale"
            reasons.append("manifest_missing")
        elif expected_entries > 0 and manifest_entries != expected_entries:
            status = "stale"
            reasons.append("manifest_entry_count_mismatch")
        elif semantic_synthesis and image_entries and not validate_semantic_cache(
            image_entries,
            cache_dir=semantic_cache_dir,
            model_id=semantic_model_id,
            device=semantic_device,
            profile=semantic_profile,
            label_space=semantic_label_space,
            source_image_index_hash=_source_image_index_hash(image_entries),
            source_image_stat_hash=_source_image_stat_hash(image_entries),
        ):
            status = "stale"
            reasons.append("semantic_cache_missing_or_stale")

    if meta and status == "invalid":
        status = "stale"

    return {
        "cache_dir": _abspath(paths.cache_dir),
        "manifest_path": _abspath(paths.manifest_path),
        "meta_path": _abspath(paths.meta_path),
        "info_path": _abspath(paths.info_path),
        "info_exists": paths.info_path.exists(),
        "semantic_cache_dir": semantic_cache_dir if semantic_synthesis else None,
        "high_images": image_count,
        "expected_entries": expected_entries,
        "manifest_entries": manifest_entries,
        "meta": meta,
        "status": status,
        "status_reasons": reasons,
        "source_error": source_error,
    }


def validate_prepared_cache(
    data_dir: str | os.PathLike[str],
    prepared_cache_dir: str | os.PathLike[str] | None,
    *,
    variant_count: int,
    synthesis_seed: int,
    darker_ranges: dict[str, Any] | None,
    degradation_backend: str = "opencv",
    prepared_train_resolution: int | None = None,
    semantic_synthesis: bool = False,
    semantic_model_id: str | None = None,
    semantic_profile: str | None = None,
    semantic_cache_dir: str | os.PathLike[str] | None = None,
    sky_asset_dir: str | os.PathLike[str] | None = None,
    semantic_device: str = "auto",
) -> str | None:
    darker_hash = darker_ranges_hash(normalize_darker_ranges(darker_ranges))
    degradation_backend = normalize_degradation_backend(degradation_backend)
    paths = resolve_prepared_paths(data_dir, prepared_cache_dir)
    semantic_profile = normalize_semantic_profile(semantic_profile)
    semantic_model_id = str(semantic_model_id or DEFAULT_SEMANTIC_MODEL_ID)
    semantic_device = resolve_semantic_device(semantic_device)
    semantic_cache_dir = resolve_semantic_cache_dir(paths.cache_dir, semantic_cache_dir)
    sky_asset_hash = sky_asset_dir_hash(sky_asset_dir) if semantic_synthesis else None
    semantic_label_space = semantic_label_space_for_model_id(semantic_model_id) if semantic_synthesis else None
    semantic_contract_hash_value = semantic_contract_hash(semantic_label_space) if semantic_synthesis else None
    if not paths.manifest_path.exists() or not paths.meta_path.exists() or not paths.info_path.exists():
        return None

    meta = _load_json(paths.meta_path)
    if not _matching_metadata(
        meta,
        variant_count=variant_count,
        synthesis_seed=synthesis_seed,
        darker_hash=darker_hash,
        degradation_backend=degradation_backend,
        prepared_train_resolution=prepared_train_resolution,
        semantic_synthesis=semantic_synthesis,
        semantic_model_id=semantic_model_id,
        semantic_profile=semantic_profile,
        sky_asset_hash=sky_asset_hash,
        semantic_label_space=semantic_label_space,
        semantic_contract_hash_value=semantic_contract_hash_value,
    ):
        return None
    if str(meta.get("status", "")).lower() != "ready":
        return None

    expected_high = _iter_high_images(resolve_training_high_dir(data_dir))
    if _dataset_fingerprint_mismatch_reasons(meta, data_dir=data_dir, image_entries=expected_high):
        return None
    if semantic_synthesis and not validate_semantic_cache(
        expected_high,
        cache_dir=semantic_cache_dir,
        model_id=semantic_model_id,
        device=semantic_device,
        profile=semantic_profile,
        label_space=semantic_label_space,
        source_image_index_hash=_source_image_index_hash(expected_high),
        source_image_stat_hash=_source_image_stat_hash(expected_high),
    ):
        return None

    expected_high_map = {image_id: _abspath(
        high_path) for image_id, high_path in expected_high}
    expected_ids = {image_id for image_id, _ in expected_high}
    expected_total = len(expected_ids) * int(variant_count)
    entries = load_manifest_entries(paths.manifest_path)
    if len(entries) != expected_total:
        return None

    available_low_filenames = _list_prepared_low_filenames(paths.low_dir)
    variant_tracker: dict[str, set[int]] = {
        image_id: set() for image_id in expected_ids}
    for entry in entries:
        image_id = entry.get("image_id")
        variant_idx = entry.get("variant_idx")
        low_path = resolve_manifest_entry_path(
            entry,
            "low_path",
            data_dir=data_dir,
            prepared_cache_dir=prepared_cache_dir,
        )
        high_path = resolve_manifest_entry_path(
            entry,
            "high_path",
            data_dir=data_dir,
            prepared_cache_dir=prepared_cache_dir,
        )
        if image_id not in variant_tracker:
            return None
        if not isinstance(variant_idx, int) or not (0 <= variant_idx < int(variant_count)):
            return None
        if not low_path or not high_path:
            return None
        if os.path.basename(low_path) not in available_low_filenames:
            return None
        if expected_high_map[image_id] != _abspath(high_path):
            return None
        expected_low_path = _expected_low_path(
            paths.low_dir, image_id, variant_idx)
        if _abspath(low_path) != expected_low_path:
            return None
        variant_tracker[image_id].add(variant_idx)

    if any(len(variants) != int(variant_count) for variants in variant_tracker.values()):
        return None

    info_payload = load_manifest_info(paths.info_path)
    if info_payload is None:
        return None
    if int(info_payload.get("version", -1)) != MANIFEST_INFO_VERSION:
        return None
    if int(info_payload.get("variant_count", -1)) != int(variant_count):
        return None
    if int(info_payload.get("synthesis_seed", -1)) != int(synthesis_seed):
        return None
    if info_payload.get("darker_ranges_hash") != darker_hash:
        return None
    if str(info_payload.get("degradation_backend", "opencv")) != str(degradation_backend):
        return None
    if int(info_payload.get("prepared_train_resolution") or 0) != int(prepared_train_resolution or 0):
        return None
    if bool(info_payload.get("semantic_synthesis", False)) != bool(semantic_synthesis):
        return None
    if semantic_synthesis:
        if str(info_payload.get("semantic_model_id") or "") != semantic_model_id:
            return None
        if str(info_payload.get("semantic_profile") or "") != semantic_profile:
            return None
        if str(info_payload.get("sky_asset_dir_hash") or "") != str(sky_asset_hash or ""):
            return None
        if str(info_payload.get("semantic_label_space") or "") != str(semantic_label_space or ""):
            return None
        if str(info_payload.get("semantic_contract_hash") or "") != str(semantic_contract_hash_value or ""):
            return None
        if int(info_payload.get("semantic_synthesis_version") or 0) != int(SEMANTIC_SYNTHESIS_VERSION):
            return None
    expected_fingerprint = _build_dataset_fingerprint(data_dir, expected_high)
    if info_payload.get("source_image_index_hash") != expected_fingerprint["source_image_index_hash"]:
        return None
    if info_payload.get("source_image_stat_hash") != expected_fingerprint["source_image_stat_hash"]:
        return None
    info_entries = info_payload.get("entries")
    if not isinstance(info_entries, list) or len(info_entries) != len(entries):
        return None
    for manifest_entry, info_entry in zip(entries, info_entries):
        if manifest_entry != info_entry:
            return None

    if prepared_train_resolution:
        train_resolution = int(prepared_train_resolution)
        expected_low_dir = train_cache_low_dir(paths, train_resolution)
        expected_high_dir = train_cache_high_dir(paths, train_resolution)
        if not expected_low_dir.exists() or not expected_high_dir.exists():
            return None
        expected_train_high_filenames = {
            f"{image_id}.png" for image_id in expected_ids}
        expected_train_low_filenames = {
            _expected_low_filename(image_id, variant_idx)
            for image_id in expected_ids
            for variant_idx in range(int(variant_count))
        }
        if expected_train_low_filenames - _list_prepared_low_filenames(expected_low_dir):
            return None
        if expected_train_high_filenames - _list_prepared_low_filenames(expected_high_dir):
            return None
        for entry in entries:
            low_train_path = resolve_manifest_entry_path(
                entry,
                "train_low_path",
                data_dir=data_dir,
                prepared_cache_dir=prepared_cache_dir,
            )
            high_train_path = resolve_manifest_entry_path(
                entry,
                "train_high_path",
                data_dir=data_dir,
                prepared_cache_dir=prepared_cache_dir,
            )
            if low_train_path is None or high_train_path is None:
                return None
            if not Path(low_train_path).exists() or not Path(high_train_path).exists():
                return None
    return str(paths.manifest_path)


def prepare_training_data(
    data_dir: str | os.PathLike[str],
    prepared_cache_dir: str | os.PathLike[str] | None,
    *,
    variant_count: int,
    synthesis_seed: int,
    darker_ranges: dict[str, Any] | None,
    degradation_backend: str = "opencv",
    prepare_workers: int,
    prepared_train_resolution: int | None = None,
    semantic_synthesis: bool = False,
    semantic_model_id: str | None = None,
    semantic_profile: str | None = None,
    semantic_cache_dir: str | os.PathLike[str] | None = None,
    sky_asset_dir: str | os.PathLike[str] | None = None,
    semantic_device: str = "auto",
    force: bool = False,
) -> str:
    data_dir = str(Path(data_dir))
    darker_ranges = normalize_darker_ranges(darker_ranges)
    degradation_backend = normalize_degradation_backend(degradation_backend)
    paths = resolve_prepared_paths(data_dir, prepared_cache_dir)
    semantic_profile = normalize_semantic_profile(semantic_profile)
    semantic_model_id = str(semantic_model_id or DEFAULT_SEMANTIC_MODEL_ID)
    semantic_device = resolve_semantic_device(semantic_device)
    semantic_cache_dir = resolve_semantic_cache_dir(paths.cache_dir, semantic_cache_dir)
    sky_asset_dir = _abspath(str(sky_asset_dir)) if sky_asset_dir else None
    sky_asset_hash = sky_asset_dir_hash(sky_asset_dir) if semantic_synthesis else None
    semantic_label_space = semantic_label_space_for_model_id(semantic_model_id) if semantic_synthesis else None
    semantic_contract_hash_value = semantic_contract_hash(semantic_label_space) if semantic_synthesis else None
    high_dir = resolve_training_high_dir(data_dir)
    if _abspath(paths.cache_dir) == _abspath(data_dir):
        raise ValueError("prepared_cache_dir must not be the same as data_dir")
    if semantic_synthesis and degradation_backend != "torch":
        raise ValueError("semantic_synthesis requires degradation_backend='torch'")
    image_entries = _iter_high_images(high_dir)
    darker_hash = darker_ranges_hash(darker_ranges)
    meta = _load_json(paths.meta_path)
    dataset_fingerprint = _build_dataset_fingerprint(data_dir, image_entries)
    resume_from_existing = _matching_core_metadata(
        meta,
        variant_count=variant_count,
        synthesis_seed=synthesis_seed,
        darker_hash=darker_hash,
        degradation_backend=degradation_backend,
        semantic_synthesis=semantic_synthesis,
        semantic_model_id=semantic_model_id,
        semantic_profile=semantic_profile,
        sky_asset_hash=sky_asset_hash,
        semantic_label_space=semantic_label_space,
        semantic_contract_hash_value=semantic_contract_hash_value,
    )
    if resume_from_existing:
        resume_from_existing = not _dataset_fingerprint_mismatch_reasons(
            meta, data_dir=data_dir, image_entries=image_entries)
    current_train_resolution = int(
        meta.get("prepared_train_resolution") or 0) if meta else 0
    target_train_resolution = int(prepared_train_resolution or 0)
    train_resolution_changed = bool(meta) and resume_from_existing and (
        current_train_resolution != target_train_resolution)
    metadata_mismatch = meta is not None and not resume_from_existing
    if force or metadata_mismatch:
        reason = "force rebuild requested" if force else "prepared cache metadata changed"
        _log_prepare(
            f"rebuilding cache at {_abspath(paths.cache_dir)} because {reason}")
        shutil.rmtree(paths.cache_dir, ignore_errors=True)
    elif train_resolution_changed:
        _log_prepare(
            "prepared_train_resolution changed; reusing low-light variants and refreshing train cache only"
        )

    semantic_cache_paths: dict[str, str] = {}
    if semantic_synthesis:
        _log_prepare(
            f"building semantic cache at {semantic_cache_dir} model={semantic_model_id} device={semantic_device}"
        )
        semantic_cache_paths = build_semantic_cache(
            image_entries,
            cache_dir=semantic_cache_dir,
            model_id=semantic_model_id,
            device=semantic_device,
            profile=semantic_profile,
            source_image_index_hash=dataset_fingerprint["source_image_index_hash"],
            source_image_stat_hash=dataset_fingerprint["source_image_stat_hash"],
            log_fn=_log_prepare,
        )

    paths.low_dir.mkdir(parents=True, exist_ok=True)
    existing_low_filenames = _list_prepared_low_filenames(paths.low_dir)
    if resume_from_existing and existing_low_filenames:
        _log_prepare(
            f"resuming cache preparation from status={meta.get('status')} with "
            f"{len(existing_low_filenames)} prepared variants already present"
        )
    elif meta is None and existing_low_filenames:
        _log_prepare(
            "found prepared low-light variants without current metadata; scanning once and reusing what matches expected filenames"
        )
    elif not existing_low_filenames:
        _log_prepare(
            f"starting prepared cache build at {_abspath(paths.cache_dir)}")

    expected_tasks: list[dict[str, Any]] = []
    pending_tasks: list[dict[str, Any]] = []
    expected_filenames: set[str] = set()
    for image_id, high_path in image_entries:
        for variant_idx in range(int(variant_count)):
            low_filename = _expected_low_filename(image_id, variant_idx)
            low_path = paths.low_dir / low_filename
            task = _prepare_task_payload(
                image_id=image_id,
                high_path=high_path,
                low_path=low_path,
                low_filename=low_filename,
                variant_idx=variant_idx,
                base_seed=int(synthesis_seed),
                darker_ranges=darker_ranges,
                semantic_cache_path=semantic_cache_paths.get(image_id),
                semantic_label_space=semantic_label_space,
            )
            expected_tasks.append(task)
            expected_filenames.add(low_filename)
            if low_filename not in existing_low_filenames:
                pending_tasks.append(task)

    meta_payload = {
        "prepare_version": PREPARE_VERSION,
        "source_mode": SOURCE_MODE,
        "variant_count": int(variant_count),
        "synthesis_seed": int(synthesis_seed),
        "prepared_train_resolution": int(prepared_train_resolution) if prepared_train_resolution else None,
        "darker_ranges_hash": darker_hash,
        "degradation_backend": degradation_backend,
        "darker_ranges": resolve_darker_ranges(darker_ranges),
        "semantic_synthesis": bool(semantic_synthesis),
        "semantic_model_id": semantic_model_id if semantic_synthesis else None,
        "semantic_profile": semantic_profile if semantic_synthesis else None,
        "semantic_cache_dir": semantic_cache_dir if semantic_synthesis else None,
        "sky_asset_dir": sky_asset_dir if semantic_synthesis else None,
        "sky_asset_dir_hash": sky_asset_hash if semantic_synthesis else None,
        "semantic_label_space": semantic_label_space if semantic_synthesis else None,
        "semantic_contract_hash": semantic_contract_hash_value if semantic_synthesis else None,
        "semantic_synthesis_version": int(SEMANTIC_SYNTHESIS_VERSION if semantic_synthesis else 0),
        "semantic_cache_entries": len(semantic_cache_paths) if semantic_synthesis else 0,
        "image_count": len(image_entries),
        "manifest_entries": len(expected_tasks),
        "completed_entries": len(expected_filenames & existing_low_filenames),
        "prepared_cache_dir": _abspath(paths.cache_dir),
        "manifest_path": _abspath(paths.manifest_path),
        "low_dir": _abspath(paths.low_dir),
        "train_cache_low_completed": 0,
        "train_cache_low_total": 0,
        "train_cache_high_completed": 0,
        "train_cache_high_total": 0,
        "status": "preparing",
        "last_update_ts": time.time(),
    }
    meta_payload.update(dataset_fingerprint)
    _write_metadata(paths, meta_payload)

    _log_prepare(
        f"dataset={_abspath(data_dir)} images={len(image_entries)} variants_per_image={int(variant_count)} "
        f"expected_entries={len(expected_tasks)} existing={len(expected_filenames & existing_low_filenames)} "
            f"missing={len(pending_tasks)} workers={max(1, int(prepare_workers))} backend={degradation_backend}"
    )

    failures: list[tuple[str, str]] = []
    completed_entries = len(expected_filenames & existing_low_filenames)
    total_entries = len(expected_tasks)
    resumed_entries = completed_entries
    generated_entries = 0
    started_at = time.perf_counter()
    executor: ProcessPoolExecutor | None = None
    interrupted = False
    try:
        if pending_tasks:
            worker_count = max(1, int(prepare_workers))
            chunksize = _task_chunksize(len(pending_tasks), worker_count)
            if worker_count == 1:
                _prepare_pool_initializer(
                    darker_ranges,
                    degradation_backend,
                    semantic_synthesis,
                    sky_asset_dir,
                    semantic_profile,
                )
                results = map(_prepare_single_image, pending_tasks)
            else:
                executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_prepare_pool_initializer,
                    initargs=(
                        darker_ranges,
                        degradation_backend,
                        semantic_synthesis,
                        sky_asset_dir,
                        semantic_profile,
                    ),
                )
                results = executor.map(_prepare_single_image,
                                       pending_tasks, chunksize=chunksize)
            log_every = _progress_interval(len(pending_tasks))
            last_logged = time.monotonic()
            _log_prepare(
                f"cpu backend ready: chunksize={chunksize} opencv_threads_per_worker=1")
            for index, result in enumerate(results, start=1):
                image_id, variant_idx, low_path, ok, error = result
                if ok:
                    completed_entries += 1
                    generated_entries += 1
                else:
                    failures.append((f"{image_id}__v{variant_idx:02d}", error))

                now = time.monotonic()
                if index == 1 or index == len(pending_tasks) or index % log_every == 0 or (now - last_logged) >= 2.0:
                    _log_prepare(
                        f"progress generated={index}/{len(pending_tasks)} completed={completed_entries}/{total_entries} "
                        f"missing={total_entries - completed_entries}"
                    )
                    last_logged = now
    except KeyboardInterrupt:
        interrupted = True
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        meta_payload["status"] = "interrupted"
        meta_payload["completed_entries"] = completed_entries
        meta_payload["last_update_ts"] = time.time()
        _write_metadata(paths, meta_payload)
        elapsed = max(time.perf_counter() - started_at, 1e-8)
        _log_prepare(
            f"interrupted by user; preserved {completed_entries}/{total_entries} prepared entries for resume "
            f"(generated_this_run={generated_entries} elapsed={elapsed:.2f}s throughput={generated_entries / elapsed:.2f} img/s)"
        )
        raise
    finally:
        if executor is not None and not interrupted:
            executor.shutdown(wait=True, cancel_futures=False)

    if failures:
        preview = ", ".join(
            f"{image_id}: {error}" for image_id, error in failures[:3])
        meta_payload["status"] = "failed"
        meta_payload["completed_entries"] = completed_entries
        meta_payload["last_update_ts"] = time.time()
        _write_metadata(paths, meta_payload)
        raise RuntimeError(
            f"Failed to prepare {len(failures)} training images. Examples: {preview}")

    available_low_filenames = _list_prepared_low_filenames(paths.low_dir)
    manifest_entries: list[dict[str, Any]] = []
    missing_filenames: list[str] = []
    for task in expected_tasks:
        low_filename = task["low_filename"]
        if low_filename not in available_low_filenames:
            missing_filenames.append(low_filename)
            continue
        manifest_entries.append(
            _build_manifest_entry(
                data_dir=data_dir,
                paths=paths,
                image_id=task["image_id"],
                variant_idx=int(task["variant_idx"]),
                high_path=task["high_path"],
                prepared_train_resolution=prepared_train_resolution,
            )
        )

    if missing_filenames:
        meta_payload["status"] = "failed"
        meta_payload["completed_entries"] = len(manifest_entries)
        meta_payload["last_update_ts"] = time.time()
        _write_metadata(paths, meta_payload)
        preview = ", ".join(missing_filenames[:3])
        raise RuntimeError(
            f"Prepared low-light variants are still missing after prepare ({len(missing_filenames)} missing). Examples: {preview}"
        )

    manifest_entries.sort(key=lambda item: (
        item["image_id"], item["variant_idx"]))

    with open(paths.manifest_path, "w", encoding="utf-8") as handle:
        for entry in manifest_entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    if prepared_train_resolution:
        train_resolution = int(prepared_train_resolution)
        # If we are truly resuming the same cache identity, preserve partial resized cache and continue.
        reset_train_cache = train_resolution_changed or not resume_from_existing
        if reset_train_cache:
            shutil.rmtree(train_cache_root(
                paths, train_resolution), ignore_errors=True)

        resized_low_dir = train_cache_low_dir(paths, train_resolution)
        resized_high_dir = train_cache_high_dir(paths, train_resolution)
        resized_low_dir.mkdir(parents=True, exist_ok=True)
        resized_high_dir.mkdir(parents=True, exist_ok=True)

        expected_low_filenames = {
            os.path.basename(str(entry.get("train_low_path") or ""))
            for entry in manifest_entries
            if entry.get("train_low_path")
        }
        expected_high_ids = {
            str(entry.get("image_id"))
            for entry in manifest_entries
            if entry.get("image_id") is not None
        }
        existing_resized_low = _list_prepared_low_filenames(resized_low_dir)
        existing_resized_high_ids = _list_prepared_png_stems(resized_high_dir)
        resized_low_completed = len(expected_low_filenames & existing_resized_low)
        resized_high_completed = len(expected_high_ids & existing_resized_high_ids)
        resized_low_total = len(expected_low_filenames)
        resized_high_total = len(expected_high_ids)

        meta_payload["train_cache_low_completed"] = resized_low_completed
        meta_payload["train_cache_low_total"] = resized_low_total
        meta_payload["train_cache_high_completed"] = resized_high_completed
        meta_payload["train_cache_high_total"] = resized_high_total
        meta_payload["last_update_ts"] = time.time()
        _write_metadata(paths, meta_payload)

        _log_prepare(
            f"building resized train cache at {train_cache_root(paths, train_resolution)} "
            f"(low {resized_low_completed}/{resized_low_total}, high {resized_high_completed}/{resized_high_total})"
        )

        written_high_ids: set[str] = set(existing_resized_high_ids)
        resized_log_every = _progress_interval(max(1, len(manifest_entries)))
        try:
            for index, entry in enumerate(manifest_entries, start=1):
                image_id = str(entry["image_id"])
                low_source = resolve_manifest_entry_path(
                    entry,
                    "low_path",
                    data_dir=data_dir,
                    prepared_cache_dir=prepared_cache_dir,
                )
                high_source = resolve_manifest_entry_path(
                    entry,
                    "high_path",
                    data_dir=data_dir,
                    prepared_cache_dir=prepared_cache_dir,
                )
                if low_source is None or high_source is None:
                    raise RuntimeError(
                        f"Unable to resolve resized-cache sources for image_id={image_id}")

                low_target = resized_low_dir / os.path.basename(low_source)
                if low_target.name not in existing_resized_low:
                    low_bgr = cv2.imread(low_source, cv2.IMREAD_COLOR)
                    if low_bgr is None:
                        raise RuntimeError(
                            f"Failed to read prepared low-light image for resized cache: {low_source}")
                    low_interp = cv2.INTER_AREA if min(
                        low_bgr.shape[:2]) >= train_resolution else cv2.INTER_LINEAR
                    low_resized = cv2.resize(
                        low_bgr, (train_resolution, train_resolution), interpolation=low_interp)
                    if not cv2.imwrite(str(low_target), low_resized):
                        raise RuntimeError(
                            f"Failed to write resized low-light cache image: {low_target}")
                    existing_resized_low.add(low_target.name)
                    resized_low_completed += 1

                if image_id not in written_high_ids:
                    high_bgr = cv2.imread(high_source, cv2.IMREAD_COLOR)
                    if high_bgr is None:
                        raise RuntimeError(
                            f"Failed to read clean image for resized cache: {high_source}")
                    high_interp = cv2.INTER_AREA if min(
                        high_bgr.shape[:2]) >= train_resolution else cv2.INTER_LINEAR
                    high_resized = cv2.resize(
                        high_bgr, (train_resolution, train_resolution), interpolation=high_interp)
                    high_target = resized_high_dir / f"{image_id}.png"
                    if not cv2.imwrite(str(high_target), high_resized):
                        raise RuntimeError(
                            f"Failed to write resized clean cache image: {high_target}")
                    written_high_ids.add(image_id)
                    resized_high_completed += 1

                if index == 1 or index == len(manifest_entries) or index % resized_log_every == 0:
                    meta_payload["train_cache_low_completed"] = resized_low_completed
                    meta_payload["train_cache_low_total"] = resized_low_total
                    meta_payload["train_cache_high_completed"] = resized_high_completed
                    meta_payload["train_cache_high_total"] = resized_high_total
                    meta_payload["last_update_ts"] = time.time()
                    _write_metadata(paths, meta_payload)
                    _log_prepare(
                        f"resized cache progress low={resized_low_completed}/{resized_low_total} "
                        f"high={resized_high_completed}/{resized_high_total}"
                    )
        except KeyboardInterrupt:
            meta_payload["status"] = "interrupted"
            meta_payload["train_cache_low_completed"] = resized_low_completed
            meta_payload["train_cache_low_total"] = resized_low_total
            meta_payload["train_cache_high_completed"] = resized_high_completed
            meta_payload["train_cache_high_total"] = resized_high_total
            meta_payload["last_update_ts"] = time.time()
            _write_metadata(paths, meta_payload)
            _log_prepare(
                f"resized cache interrupted; progress low={resized_low_completed}/{resized_low_total} "
                f"high={resized_high_completed}/{resized_high_total}"
            )
            raise
        except Exception as exc:
            meta_payload["status"] = "failed"
            meta_payload["train_cache_low_completed"] = resized_low_completed
            meta_payload["train_cache_low_total"] = resized_low_total
            meta_payload["train_cache_high_completed"] = resized_high_completed
            meta_payload["train_cache_high_total"] = resized_high_total
            meta_payload["last_update_ts"] = time.time()
            _write_metadata(paths, meta_payload)
            raise RuntimeError(f"Failed while building resized train cache: {exc}") from exc

    meta_payload["manifest_entries"] = len(manifest_entries)
    meta_payload["completed_entries"] = len(manifest_entries)
    if prepared_train_resolution:
        meta_payload["train_cache_low_completed"] = len(manifest_entries)
        meta_payload["train_cache_low_total"] = len(manifest_entries)
        unique_high_count = len({str(entry.get("image_id")) for entry in manifest_entries})
        meta_payload["train_cache_high_completed"] = unique_high_count
        meta_payload["train_cache_high_total"] = unique_high_count
    meta_payload["status"] = "ready"
    meta_payload["last_update_ts"] = time.time()
    _write_metadata(paths, meta_payload)
    elapsed = max(time.perf_counter() - started_at, 1e-8)
    meta_payload["last_run_elapsed_sec"] = elapsed
    meta_payload["last_run_generated_entries"] = generated_entries
    meta_payload["last_run_reused_entries"] = resumed_entries
    _write_metadata(paths, meta_payload)
    _write_manifest_info(
        paths,
        _build_manifest_info_payload(
            paths,
            variant_count=variant_count,
            synthesis_seed=synthesis_seed,
            darker_hash=darker_hash,
            degradation_backend=degradation_backend,
            source_image_index_hash=dataset_fingerprint["source_image_index_hash"],
            source_image_stat_hash=dataset_fingerprint["source_image_stat_hash"],
            manifest_entries=manifest_entries,
            prepared_train_resolution=prepared_train_resolution,
            semantic_synthesis=semantic_synthesis,
            semantic_model_id=semantic_model_id,
            semantic_profile=semantic_profile,
            sky_asset_hash=sky_asset_hash,
            semantic_label_space=semantic_label_space,
            semantic_contract_hash_value=semantic_contract_hash_value,
        ),
    )
    _log_prepare(
        f"cache ready: manifest={_abspath(paths.manifest_path)} entries={len(manifest_entries)} "
        f"generated_this_run={generated_entries} reused_existing={resumed_entries} "
        f"elapsed={elapsed:.2f}s throughput={generated_entries / elapsed if generated_entries else 0.0:.2f} img/s"
    )

    return str(paths.manifest_path)


def ensure_prepared_training_data(
    data_dir: str | os.PathLike[str],
    prepared_cache_dir: str | os.PathLike[str] | None,
    *,
    variant_count: int,
    synthesis_seed: int,
    darker_ranges: dict[str, Any] | None,
    degradation_backend: str = "opencv",
    prepare_workers: int,
    prepared_train_resolution: int | None = None,
    semantic_synthesis: bool = False,
    semantic_model_id: str | None = None,
    semantic_profile: str | None = None,
    semantic_cache_dir: str | os.PathLike[str] | None = None,
    sky_asset_dir: str | os.PathLike[str] | None = None,
    semantic_device: str = "auto",
    force: bool = False,
    prepare_on_train: bool = True,
) -> tuple[str, bool]:
    if force:
        _log_prepare("cache miss: force rebuild requested")
        manifest_path = prepare_training_data(
            data_dir,
            prepared_cache_dir,
            variant_count=variant_count,
            synthesis_seed=synthesis_seed,
            darker_ranges=darker_ranges,
            degradation_backend=degradation_backend,
            prepare_workers=prepare_workers,
            prepared_train_resolution=prepared_train_resolution,
            semantic_synthesis=semantic_synthesis,
            semantic_model_id=semantic_model_id,
            semantic_profile=semantic_profile,
            semantic_cache_dir=semantic_cache_dir,
            sky_asset_dir=sky_asset_dir,
            semantic_device=semantic_device,
            force=True,
        )
        return manifest_path, True

    manifest_path = validate_prepared_cache(
        data_dir,
        prepared_cache_dir,
        variant_count=variant_count,
        synthesis_seed=synthesis_seed,
        darker_ranges=darker_ranges,
        degradation_backend=degradation_backend,
        prepared_train_resolution=prepared_train_resolution,
        semantic_synthesis=semantic_synthesis,
        semantic_model_id=semantic_model_id,
        semantic_profile=semantic_profile,
        semantic_cache_dir=semantic_cache_dir,
        sky_asset_dir=sky_asset_dir,
        semantic_device=semantic_device,
    )
    if manifest_path is not None:
        _log_prepare(
            f"cache hit: using existing manifest at {_abspath(manifest_path)}")
        return manifest_path, False
    if not prepare_on_train:
        raise RuntimeError(
            "Prepared multi-variant training data is missing or stale. "
            "Run `--mode prepare` or enable `--prepare-on-train`."
        )

    _log_prepare("cache miss: cache missing, stale, or incomplete")

    manifest_path = prepare_training_data(
        data_dir,
        prepared_cache_dir,
        variant_count=variant_count,
        synthesis_seed=synthesis_seed,
        darker_ranges=darker_ranges,
        degradation_backend=degradation_backend,
        prepare_workers=prepare_workers,
        prepared_train_resolution=prepared_train_resolution,
        semantic_synthesis=semantic_synthesis,
        semantic_model_id=semantic_model_id,
        semantic_profile=semantic_profile,
        semantic_cache_dir=semantic_cache_dir,
        sky_asset_dir=sky_asset_dir,
        semantic_device=semantic_device,
        force=force,
    )
    return manifest_path, True
