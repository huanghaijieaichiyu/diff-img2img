from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import hashlib
import json
import os
import random

import cv2
import numpy as np
import torch

from utils.torch_low_light_degrader import TorchLowLightDegrader

VALID_ASSET_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DEFAULT_SEMANTIC_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
DEFAULT_SEMANTIC_PROFILE = "road_scene_v1"
SEMANTIC_SYNTHESIS_VERSION = 2
CITYSCAPES_PRECOMPUTED_MODEL_ID = "precomputed/cityscapes-labelids-v1"
CITYSCAPES_LABEL_SPACE_LABEL_IDS = "labelIds"
CITYSCAPES_LABEL_SPACE_TRAIN_IDS = "trainIds"

CITYSCAPES_CLASS_GROUPS_BY_LABEL_SPACE: dict[str, dict[str, set[int]]] = {
    CITYSCAPES_LABEL_SPACE_LABEL_IDS: {
        "road": {6, 7, 8, 9, 10},
        "building": {11, 12, 13, 14, 15, 16, 17, 18},
        "light": {19},
        "sign": {20},
        "vegetation": {21, 22},
        "sky": {23},
        "person": {24, 25},
        "vehicle": {26, 27, 28, 29, 30, 31, 32, 33},
    },
    CITYSCAPES_LABEL_SPACE_TRAIN_IDS: {
        "road": {0, 1},
        "building": {2, 3, 4, 5},
        "light": {6},
        "sign": {7},
        "vegetation": {8, 9},
        "sky": {10},
        "person": {11, 12},
        "vehicle": {13, 14, 15, 16, 17, 18},
    },
}

CITYSCAPES_ALLOWED_LABEL_VALUES: dict[str, set[int]] = {
    CITYSCAPES_LABEL_SPACE_LABEL_IDS: set(range(34)) | {255},
    CITYSCAPES_LABEL_SPACE_TRAIN_IDS: set(range(19)) | {255},
}

# Backward-compatible alias for the canonical precomputed Cityscapes path.
CITYSCAPES_CLASS_GROUPS = CITYSCAPES_CLASS_GROUPS_BY_LABEL_SPACE[CITYSCAPES_LABEL_SPACE_LABEL_IDS]


@dataclass(frozen=True)
class SemanticCacheConfig:
    cache_dir: str
    model_id: str
    device: str
    profile: str
    label_space: str
    contract_hash: str
    source_image_index_hash: str | None
    source_image_stat_hash: str | None
    image_ids: tuple[str, ...]


class RoadSceneSemanticSegmenter:
    def __init__(self, model_id: str = DEFAULT_SEMANTIC_MODEL_ID, device: str = "auto"):
        self.model_id = str(model_id or DEFAULT_SEMANTIC_MODEL_ID)
        self.device = resolve_semantic_device(device)

    def predict(self, image_rgb: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        image = _to_numpy_rgb(image_rgb)
        height, width = image.shape[:2]
        image_f = image.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(image_f, cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1]
        val = hsv[..., 2]
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]

        red = image_f[..., 0]
        green = image_f[..., 1]
        blue = image_f[..., 2]
        label_map = np.full((height, width), 2, dtype=np.uint8)
        confidence = np.full((height, width), 0.55, dtype=np.float32)

        sky = (yy < 0.45) & (blue > green * 0.95) & (blue > red * 1.05) & (val > 0.35)
        vegetation = (green > red * 1.08) & (green > blue * 1.02) & (sat > 0.18)
        sign = (sat > 0.45) & (val > 0.35) & ((red > 0.45) | ((red > 0.35) & (green > 0.35)))
        light = (val > 0.72) & (sat > 0.2) & (yy > 0.2)
        road = (yy > 0.45) & (sat < 0.22) & (val > 0.08) & (val < 0.72)
        vehicle = (yy > 0.42) & (sat > 0.2) & ~vegetation & ~sign & ~light
        person = (yy > 0.35) & (yy < 0.95) & (sat > 0.15) & (val > 0.15) & (val < 0.85)

        label_map[sky] = 10
        confidence[sky] = 0.9
        label_map[vegetation] = 8
        confidence[vegetation] = 0.8
        label_map[road] = 0
        confidence[road] = 0.75
        label_map[sign] = 7
        confidence[sign] = 0.8
        label_map[light] = 6
        confidence[light] = 0.82
        label_map[vehicle] = 13
        confidence[vehicle] = np.maximum(confidence[vehicle], 0.68)
        label_map[person & ~vehicle & ~road & ~sky] = 11
        confidence[person & ~vehicle & ~road & ~sky] = np.maximum(confidence[person & ~vehicle & ~road & ~sky], 0.6)

        return label_map.astype(np.uint8), confidence.astype(np.float32)


class RoadSceneNightSynthesizer:
    def __init__(
        self,
        sky_asset_dir: str | os.PathLike[str] | None = None,
        param_ranges: dict | None = None,
        randomize: bool = True,
        profile: str = DEFAULT_SEMANTIC_PROFILE,
    ):
        profile = str(profile or DEFAULT_SEMANTIC_PROFILE).strip()
        if profile != DEFAULT_SEMANTIC_PROFILE:
            raise ValueError(
                f"Unsupported semantic synthesis profile: {profile!r}. Expected {DEFAULT_SEMANTIC_PROFILE!r}."
            )
        self.profile = profile
        self.randomize = bool(randomize)
        self.param_ranges = dict(param_ranges or {})
        self.degrader = TorchLowLightDegrader(randomize=randomize, param_ranges=param_ranges)
        self.sky_asset_dir = _abspath(sky_asset_dir) if sky_asset_dir else default_sky_asset_dir()
        self.sky_assets = list_sky_assets(self.sky_asset_dir)

    def _class_mask(
        self,
        label_map: np.ndarray,
        confidence_map: np.ndarray,
        group_name: str,
        *,
        label_space: str | None = None,
    ) -> np.ndarray:
        group_ids = cityscapes_class_groups(label_space).get(group_name)
        if not group_ids:
            return np.zeros(label_map.shape, dtype=np.float32)
        mask = np.isin(label_map.astype(np.int32), list(group_ids)).astype(np.float32)
        confidence = confidence_map.astype(np.float32)
        if confidence.shape != mask.shape:
            confidence = np.ones(mask.shape, dtype=np.float32)
        return np.clip(mask * confidence, 0.0, 1.0)

    def _build_masks(
        self,
        label_map: np.ndarray,
        confidence_map: np.ndarray,
        *,
        label_space: str | None = None,
    ) -> dict[str, np.ndarray]:
        label_space = normalize_cityscapes_label_space(label_space, model_id=CITYSCAPES_PRECOMPUTED_MODEL_ID)
        groups = cityscapes_class_groups(label_space)
        return {
            name: self._class_mask(label_map, confidence_map, name, label_space=label_space)
            for name in groups
        }

    def _sky_mask_is_usable(self, sky_mask: np.ndarray) -> bool:
        coverage = float(sky_mask.mean())
        if coverage < 0.02:
            return False
        if coverage > 0.7:
            return False
        return _largest_component_fraction(sky_mask > 0.5) > 0.1

    def _load_random_sky_asset(self, height: int, width: int) -> np.ndarray | None:
        if not self.sky_assets:
            return None
        asset_path = random.choice(self.sky_assets)
        asset = cv2.imread(str(asset_path), cv2.IMREAD_COLOR)
        if asset is None:
            return None
        asset = cv2.cvtColor(asset, cv2.COLOR_BGR2RGB)
        asset = cv2.resize(asset, (width, height), interpolation=cv2.INTER_LINEAR)
        return asset.astype(np.float32) / 255.0

    def _render_gradient_sky(self, image_rgb: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        top = np.array([0.018, 0.028, 0.110], dtype=np.float32)
        bottom = np.array([0.070, 0.100, 0.180], dtype=np.float32)
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        gradient = top * (1.0 - yy) + bottom * yy
        sky = np.repeat(gradient, width, axis=1)
        if self.randomize:
            noise = np.random.normal(0.0, 0.006, size=sky.shape).astype(np.float32)
            sky = np.clip(sky + noise, 0.0, 1.0)
            star_noise = np.random.rand(height, width).astype(np.float32)
            stars = np.zeros_like(sky)
            star_field = np.clip((star_noise < 0.0009).astype(np.float32), 0.0, 1.0)
            star_field = cv2.GaussianBlur(star_field, (5, 5), sigmaX=1.3)
            stars[..., 2] += star_field * 0.40
            stars[..., 1] += star_field * 0.25
            stars[..., 0] += star_field * 0.12
            sky = np.clip(sky + stars, 0.0, 1.0)
        return sky

    def _match_sky_asset(self, source_rgb: np.ndarray, sky_rgb: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
        mask = sky_mask > 0.05
        if not np.any(mask):
            return sky_rgb
        source_mean = source_rgb[mask].mean(axis=0)
        sky_mean = sky_rgb[mask].mean(axis=0)
        scale = np.divide(np.maximum(source_mean, 0.03), np.maximum(sky_mean, 1e-4))
        scale = np.clip(scale, 0.20, 0.55)
        matched = sky_rgb * scale.reshape(1, 1, 3)
        matched[..., 2] *= 0.92
        matched[..., 1] *= 0.82
        matched[..., 0] *= 0.72
        return np.clip(matched, 0.0, 1.0)

    def _render_sky(self, image_rgb: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
        sky_rgb = self._load_random_sky_asset(*image_rgb.shape[:2])
        if sky_rgb is None:
            return self._render_gradient_sky(image_rgb, sky_mask)
        return self._match_sky_asset(image_rgb, sky_rgb, sky_mask)

    def _apply_class_aware_darkening(self, image_rgb: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
        hsv = cv2.cvtColor(image_rgb.astype(np.float32), cv2.COLOR_RGB2HSV)
        height, width = image_rgb.shape[:2]
        gamma_map = np.full((height, width), 1.65, dtype=np.float32)
        atten_map = np.full((height, width), 0.22, dtype=np.float32)
        sat_map = np.full((height, width), 0.55, dtype=np.float32)

        def blend(mask_name: str, gamma: float, attenuation: float, saturation: float) -> None:
            soft = _blur_map(masks.get(mask_name), sigma=max(height, width) * 0.01)
            gamma_map[:] = gamma_map * (1.0 - soft) + gamma * soft
            atten_map[:] = atten_map * (1.0 - soft) + attenuation * soft
            sat_map[:] = sat_map * (1.0 - soft) + saturation * soft

        blend("sky", 1.00, 0.58, 0.60)
        blend("road", 1.35, 0.28, 0.65)
        blend("building", 1.45, 0.24, 0.40)
        blend("vegetation", 1.55, 0.20, 0.32)
        blend("vehicle", 1.25, 0.34, 0.72)
        blend("person", 1.38, 0.28, 0.58)
        blend("sign", 1.12, 0.34, 0.84)
        blend("light", 1.00, 0.78, 1.00)

        hsv[..., 1] = np.clip(hsv[..., 1] * sat_map, 0.0, 1.0)
        hsv[..., 2] = np.clip(np.power(np.clip(hsv[..., 2], 0.0, 1.0), gamma_map) * atten_map, 0.0, 1.0)
        return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)

    def _random_light_pool(self, height: int, width: int, focus_mask: np.ndarray) -> np.ndarray:
        if focus_mask is None:
            return np.zeros((height, width), dtype=np.float32)
        focus = np.clip(focus_mask.astype(np.float32), 0.0, 1.0)
        if float(focus.max()) <= 0.0:
            return np.zeros((height, width), dtype=np.float32)
        sigma = max(3.0, 0.022 * max(height, width))
        spread = _blur_map(focus, sigma=sigma)
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        distance_falloff = np.clip(0.20 + 0.80 * (yy ** 1.8), 0.0, 1.0)
        spread = spread * distance_falloff
        if self.randomize:
            spread = spread * random.uniform(0.70, 1.00)
        return np.clip(spread, 0.0, 1.0)

    def _compose_illumination(self, source_rgb: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
        height, width = source_rgb.shape[:2]
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        road_focus = np.clip(
            masks.get("road", 0.0)
            + 0.8 * masks.get("light", 0.0)
            + 0.45 * masks.get("vehicle", 0.0)
            + 0.15 * masks.get("sign", 0.0),
            0.0,
            1.0,
        )
        warm_pool = self._random_light_pool(height, width, road_focus)
        ambient = _blur_map(masks.get("sky"), sigma=max(height, width) * 0.05)
        influence = np.clip(
            1.0 * masks.get("road", 0.0)
            + 0.45 * masks.get("building", 0.0)
            + 0.35 * masks.get("vehicle", 0.0),
            0.0,
            1.0,
        )
        near_field = np.clip(0.10 + 0.90 * (yy ** 1.6), 0.0, 1.0)
        road_mask = np.clip(masks.get("road", 0.0), 0.0, 1.0)
        reflective_road = _blur_map(road_mask * near_field, sigma=max(height, width) * 0.012)
        reflective_road *= np.clip(warm_pool * 1.2 + 0.25 * masks.get("vehicle", 0.0), 0.0, 1.0)

        illumination = np.zeros_like(source_rgb, dtype=np.float32)
        illumination[..., 0] += warm_pool * influence * near_field * 0.08
        illumination[..., 1] += warm_pool * influence * near_field * 0.06
        illumination[..., 2] += warm_pool * influence * near_field * 0.03
        illumination[..., 0] += reflective_road * 0.12
        illumination[..., 1] += reflective_road * 0.10
        illumination[..., 2] += reflective_road * 0.08
        illumination[..., 2] += ambient * 0.015 * (1.0 - 0.55 * yy)
        return np.clip(illumination, 0.0, 0.16)

    def _apply_emissive_boost(
        self,
        image_rgb: np.ndarray,
        source_rgb: np.ndarray,
        masks: dict[str, np.ndarray],
    ) -> np.ndarray:
        emissive = np.clip(
            1.0 * masks.get("light", 0.0)
            + 0.5 * masks.get("sign", 0.0)
            + 0.25 * masks.get("vehicle", 0.0),
            0.0,
            1.0,
        )
        emissive = _blur_map(emissive, sigma=max(image_rgb.shape[:2]) * 0.008)
        highlight = np.maximum(image_rgb, source_rgb)
        boosted = image_rgb * (1.0 - emissive[..., None] * 0.22) + highlight * emissive[..., None] * 0.22
        return np.clip(boosted, 0.0, 1.0)

    def synthesize(
        self,
        image_rgb: np.ndarray | torch.Tensor,
        label_map: np.ndarray,
        confidence_map: np.ndarray,
        apply_base_noise: bool = True,
        label_space: str | None = None,
    ) -> torch.Tensor:
        source_rgb = _to_numpy_rgb(image_rgb).astype(np.float32) / 255.0
        label_map = np.asarray(label_map).astype(np.uint8)
        confidence_map = np.asarray(confidence_map).astype(np.float32)
        if confidence_map.shape != label_map.shape:
            confidence_map = np.ones(label_map.shape, dtype=np.float32)
        label_space = normalize_cityscapes_label_space(label_space, model_id=CITYSCAPES_PRECOMPUTED_MODEL_ID)
        validate_cityscapes_label_map(label_map, label_space=label_space)

        masks = self._build_masks(label_map, confidence_map, label_space=label_space)
        working = source_rgb.copy()
        sky_mask = masks.get("sky", np.zeros(label_map.shape, dtype=np.float32))
        if self._sky_mask_is_usable(sky_mask):
            sky_rgb = self._render_sky(working, sky_mask)
            sky_alpha = _blur_map(sky_mask, sigma=max(working.shape[:2]) * 0.01)[..., None]
            working = np.clip(working * (1.0 - sky_alpha) + sky_rgb * sky_alpha, 0.0, 1.0)

        darkened = self._apply_class_aware_darkening(working, masks)
        illuminated = np.clip(darkened + self._compose_illumination(source_rgb, masks), 0.0, 1.0)
        boosted = self._apply_emissive_boost(illuminated, source_rgb, masks)

        final_tensor = _to_tensor_rgb(boosted)
        source_tensor = _to_tensor_rgb(source_rgb)
        if apply_base_noise:
            final_tensor = TorchLowLightDegrader._enforce_exposure(source_tensor, final_tensor, min_ratio=0.08, max_ratio=0.60)
        return final_tensor.clamp(0.0, 1.0)


def _abspath(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.fspath(path))


def resolve_semantic_device(device: str | None) -> str:
    resolved = str(device or "auto").strip().lower()
    if resolved == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return resolved




def semantic_cache_device_for_model_id(model_id: str | None, device: str | None) -> str:
    if str(model_id or DEFAULT_SEMANTIC_MODEL_ID).strip() == CITYSCAPES_PRECOMPUTED_MODEL_ID:
        return "cpu"
    return resolve_semantic_device(device)

def default_sky_asset_dir(repo_root: str | os.PathLike[str] | None = None) -> str:
    base = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    return _abspath(base / "assets" / "night_sky")


def default_semantic_cache_dir(prepared_cache_dir: str | os.PathLike[str]) -> str:
    return _abspath(Path(prepared_cache_dir) / "semantic_cache")


def semantic_label_space_for_model_id(model_id: str | None) -> str:
    if str(model_id or DEFAULT_SEMANTIC_MODEL_ID).strip() == CITYSCAPES_PRECOMPUTED_MODEL_ID:
        return CITYSCAPES_LABEL_SPACE_LABEL_IDS
    return CITYSCAPES_LABEL_SPACE_TRAIN_IDS


def normalize_cityscapes_label_space(label_space: str | None, *, model_id: str | None = None) -> str:
    if label_space is None or str(label_space).strip() == "":
        return semantic_label_space_for_model_id(model_id)
    normalized = str(label_space).strip()
    if normalized not in CITYSCAPES_CLASS_GROUPS_BY_LABEL_SPACE:
        raise ValueError(
            f"Unsupported Cityscapes label space: {label_space!r}. "
            f"Expected one of {sorted(CITYSCAPES_CLASS_GROUPS_BY_LABEL_SPACE)}."
        )
    return normalized


def cityscapes_class_groups(label_space: str | None, *, model_id: str | None = None) -> dict[str, set[int]]:
    normalized = normalize_cityscapes_label_space(label_space, model_id=model_id)
    return CITYSCAPES_CLASS_GROUPS_BY_LABEL_SPACE[normalized]


def semantic_contract_hash(label_space: str | None, *, model_id: str | None = None) -> str:
    normalized = normalize_cityscapes_label_space(label_space, model_id=model_id)
    payload = {
        "label_space": normalized,
        "groups": {
            name: sorted(group_ids)
            for name, group_ids in sorted(cityscapes_class_groups(normalized).items())
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def validate_cityscapes_label_map(
    label_map: np.ndarray,
    *,
    label_space: str | None,
    model_id: str | None = None,
) -> None:
    normalized = normalize_cityscapes_label_space(label_space, model_id=model_id)
    allowed = CITYSCAPES_ALLOWED_LABEL_VALUES[normalized]
    values = {int(value) for value in np.unique(np.asarray(label_map).astype(np.int32))}
    invalid = sorted(value for value in values if value not in allowed)
    if invalid:
        raise ValueError(
            f"Unsupported labels for Cityscapes {normalized}: {invalid[:10]} "
            f"(showing up to 10 values)."
        )


def list_sky_assets(sky_asset_dir: str | os.PathLike[str]) -> list[Path]:
    directory = Path(sky_asset_dir)
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_ASSET_EXTENSIONS
    )


def sky_asset_dir_hash(sky_asset_dir: str | os.PathLike[str] | None) -> str:
    digest = hashlib.sha256()
    if not sky_asset_dir:
        digest.update(b"<none>")
        return digest.hexdigest()
    directory = Path(sky_asset_dir)
    digest.update(_abspath(directory).encode("utf-8"))
    if not directory.exists():
        digest.update(b"<missing>")
        return digest.hexdigest()
    for asset in list_sky_assets(directory):
        try:
            stat = asset.stat()
            digest.update(str(asset.relative_to(directory)).encode("utf-8"))
            digest.update(str(int(stat.st_size)).encode("utf-8"))
            digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
        except OSError:
            digest.update(str(asset).encode("utf-8"))
            digest.update(b"-1")
    return digest.hexdigest()


def semantic_cache_meta_path(cache_dir: str | os.PathLike[str]) -> Path:
    return Path(cache_dir) / "_meta.json"


def semantic_cache_entry_path(cache_dir: str | os.PathLike[str], image_id: str) -> Path:
    return Path(cache_dir) / f"{image_id}.npz"


def load_semantic_cache(cache_path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path) as payload:
        label_map = payload["label_map"].astype(np.uint8)
        confidence_map = payload["confidence_map"].astype(np.float32)
    return label_map, confidence_map


def write_semantic_cache_entry(
    cache_dir: str | os.PathLike[str],
    image_id: str,
    label_map: np.ndarray,
    confidence_map: np.ndarray,
) -> str:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = semantic_cache_entry_path(cache_dir, image_id)
    np.savez_compressed(
        cache_path,
        label_map=np.asarray(label_map).astype(np.uint8),
        confidence_map=np.asarray(confidence_map).astype(np.float32),
    )
    return _abspath(cache_path)


def _write_json(path: str | os.PathLike[str], payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _read_json(path: str | os.PathLike[str]) -> dict | None:
    json_path = Path(path)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def load_semantic_cache_meta(cache_dir: str | os.PathLike[str]) -> dict | None:
    return _read_json(semantic_cache_meta_path(cache_dir))


def resolve_semantic_cache_label_space(
    cache_dir: str | os.PathLike[str],
    *,
    model_id: str | None = None,
) -> str:
    meta = load_semantic_cache_meta(cache_dir)
    return normalize_cityscapes_label_space(
        meta.get("label_space") if isinstance(meta, dict) else None,
        model_id=model_id,
    )


def _build_semantic_cache_meta(config: SemanticCacheConfig) -> dict:
    return {
        "version": SEMANTIC_SYNTHESIS_VERSION,
        "cache_dir": config.cache_dir,
        "model_id": config.model_id,
        "device": config.device,
        "profile": config.profile,
        "label_space": config.label_space,
        "semantic_contract_hash": config.contract_hash,
        "source_image_index_hash": config.source_image_index_hash,
        "source_image_stat_hash": config.source_image_stat_hash,
        "image_ids": list(config.image_ids),
        "entry_count": len(config.image_ids),
        "status": "ready",
    }


def _semantic_cache_meta_matches(meta: dict | None, config: SemanticCacheConfig) -> bool:
    if not isinstance(meta, dict):
        return False
    return (
        int(meta.get("version", -1)) == SEMANTIC_SYNTHESIS_VERSION
        and str(meta.get("cache_dir", "")) == config.cache_dir
        and str(meta.get("model_id", "")) == config.model_id
        and str(meta.get("profile", "")) == config.profile
        and str(meta.get("label_space", "")) == config.label_space
        and str(meta.get("semantic_contract_hash", "")) == config.contract_hash
        and str(meta.get("source_image_index_hash") or "") == str(config.source_image_index_hash or "")
        and str(meta.get("source_image_stat_hash") or "") == str(config.source_image_stat_hash or "")
        and list(meta.get("image_ids") or []) == list(config.image_ids)
    )


def validate_semantic_cache(
    image_entries: list[tuple[str, Path]],
    *,
    cache_dir: str | os.PathLike[str],
    model_id: str = DEFAULT_SEMANTIC_MODEL_ID,
    device: str = "auto",
    profile: str = DEFAULT_SEMANTIC_PROFILE,
    label_space: str | None = None,
    source_image_index_hash: str | None = None,
    source_image_stat_hash: str | None = None,
) -> bool:
    label_space = normalize_cityscapes_label_space(label_space, model_id=model_id)
    config = SemanticCacheConfig(
        cache_dir=_abspath(cache_dir),
        model_id=str(model_id or DEFAULT_SEMANTIC_MODEL_ID),
        device=semantic_cache_device_for_model_id(model_id, device),
        profile=str(profile or DEFAULT_SEMANTIC_PROFILE),
        label_space=label_space,
        contract_hash=semantic_contract_hash(label_space),
        source_image_index_hash=source_image_index_hash,
        source_image_stat_hash=source_image_stat_hash,
        image_ids=tuple(image_id for image_id, _ in image_entries),
    )
    meta = load_semantic_cache_meta(config.cache_dir)
    if not _semantic_cache_meta_matches(meta, config):
        return False
    return all(semantic_cache_entry_path(config.cache_dir, image_id).exists() for image_id, _ in image_entries)


def build_semantic_cache(
    image_entries: list[tuple[str, Path]],
    *,
    cache_dir: str | os.PathLike[str],
    model_id: str = DEFAULT_SEMANTIC_MODEL_ID,
    device: str = "auto",
    profile: str = DEFAULT_SEMANTIC_PROFILE,
    source_image_index_hash: str | None = None,
    source_image_stat_hash: str | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, str]:
    cache_dir = _abspath(cache_dir)
    label_space = normalize_cityscapes_label_space(None, model_id=model_id)
    config = SemanticCacheConfig(
        cache_dir=cache_dir,
        model_id=str(model_id or DEFAULT_SEMANTIC_MODEL_ID),
        device=semantic_cache_device_for_model_id(model_id, device),
        profile=str(profile or DEFAULT_SEMANTIC_PROFILE),
        label_space=label_space,
        contract_hash=semantic_contract_hash(label_space),
        source_image_index_hash=source_image_index_hash,
        source_image_stat_hash=source_image_stat_hash,
        image_ids=tuple(image_id for image_id, _ in image_entries),
    )
    if validate_semantic_cache(
        image_entries,
        cache_dir=cache_dir,
        model_id=config.model_id,
        device=config.device,
        profile=config.profile,
        label_space=config.label_space,
        source_image_index_hash=source_image_index_hash,
        source_image_stat_hash=source_image_stat_hash,
    ):
        return {
            image_id: _abspath(semantic_cache_entry_path(cache_dir, image_id))
            for image_id, _ in image_entries
        }
    if config.model_id == CITYSCAPES_PRECOMPUTED_MODEL_ID:
        raise FileNotFoundError(
            "semantic_model_id='precomputed/cityscapes-labelids-v1' requires a ready "
            f"semantic_cache_dir with matching metadata; strict precomputed mode forbids fallback rebuilds: {cache_dir}"
        )

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    segmenter = RoadSceneSemanticSegmenter(model_id=config.model_id, device=config.device)
    cache_paths: dict[str, str] = {}
    total = len(image_entries)
    for index, (image_id, image_path) in enumerate(image_entries, start=1):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read source image for semantic cache: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        label_map, confidence_map = segmenter.predict(image_rgb)
        validate_cityscapes_label_map(label_map, label_space=config.label_space)
        cache_paths[image_id] = write_semantic_cache_entry(cache_dir, image_id, label_map, confidence_map)
        if log_fn and (index == 1 or index == total or index % 200 == 0):
            log_fn(f"semantic cache {index}/{total}")

    _write_json(semantic_cache_meta_path(cache_dir), _build_semantic_cache_meta(config))
    return cache_paths


def build_semantic_cache_from_precomputed(
    image_entries: list[tuple[str, Path]],
    *,
    masks_by_id: dict[str, np.ndarray],
    cache_dir: str | os.PathLike[str],
    model_id: str = CITYSCAPES_PRECOMPUTED_MODEL_ID,
    device: str = "cpu",
    profile: str = DEFAULT_SEMANTIC_PROFILE,
    source_image_index_hash: str | None = None,
    source_image_stat_hash: str | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, str]:
    cache_dir = _abspath(cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    label_space = normalize_cityscapes_label_space(None, model_id=model_id)
    cache_paths: dict[str, str] = {}
    total = len(image_entries)
    for index, (image_id, _) in enumerate(image_entries, start=1):
        if image_id not in masks_by_id:
            raise KeyError(f"Missing precomputed semantic mask for image_id={image_id}")
        label_map = np.asarray(masks_by_id[image_id])
        if label_map.ndim == 3:
            label_map = label_map[..., 0]
        validate_cityscapes_label_map(label_map, label_space=label_space)
        confidence_map = np.ones(label_map.shape, dtype=np.float32)
        cache_paths[image_id] = write_semantic_cache_entry(cache_dir, image_id, label_map, confidence_map)
        if log_fn and (index == 1 or index == total or index % 200 == 0):
            log_fn(f"semantic cache {index}/{total}")

    config = SemanticCacheConfig(
        cache_dir=cache_dir,
        model_id=str(model_id or CITYSCAPES_PRECOMPUTED_MODEL_ID),
        device=semantic_cache_device_for_model_id(model_id, device),
        profile=str(profile or DEFAULT_SEMANTIC_PROFILE),
        label_space=label_space,
        contract_hash=semantic_contract_hash(label_space),
        source_image_index_hash=source_image_index_hash,
        source_image_stat_hash=source_image_stat_hash,
        image_ids=tuple(image_id for image_id, _ in image_entries),
    )
    _write_json(semantic_cache_meta_path(cache_dir), _build_semantic_cache_meta(config))
    return cache_paths


def _to_numpy_rgb(image_rgb: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image_rgb, torch.Tensor):
        tensor = image_rgb.detach().cpu().float()
        if tensor.ndim != 3:
            raise ValueError(f"Expected RGB tensor with shape (C, H, W), got {tuple(tensor.shape)}")
        if tensor.shape[0] == 3:
            array = tensor.permute(1, 2, 0).numpy()
        elif tensor.shape[-1] == 3:
            array = tensor.numpy()
        else:
            raise ValueError(f"Expected RGB tensor, got {tuple(tensor.shape)}")
        if array.max() <= 1.0:
            array = array * 255.0
        return np.clip(array, 0.0, 255.0).astype(np.uint8)
    array = np.asarray(image_rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {tuple(array.shape)}")
    if array.dtype != np.uint8:
        scaled = array.astype(np.float32)
        if scaled.max() <= 1.0:
            scaled = scaled * 255.0
        array = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return array


def _to_tensor_rgb(image_rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(image_rgb, torch.Tensor):
        tensor = image_rgb.detach().clone().float()
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            return tensor.clamp(0.0, 1.0) if tensor.max().item() <= 1.0 else tensor.div(255.0).clamp(0.0, 1.0)
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
            return tensor.clamp(0.0, 1.0) if tensor.max().item() <= 1.0 else tensor.div(255.0).clamp(0.0, 1.0)
        raise ValueError(f"Expected RGB tensor, got {tuple(tensor.shape)}")
    array = np.asarray(image_rgb).astype(np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {tuple(array.shape)}")
    if array.max() > 1.0:
        array = array / 255.0
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float().clamp(0.0, 1.0)


def _blur_map(image: np.ndarray | None, sigma: float) -> np.ndarray:
    if image is None:
        raise ValueError("image must not be None")
    image = np.asarray(image).astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D map, got {tuple(image.shape)}")
    ksize = max(3, int(round(float(sigma) * 4)) | 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=max(float(sigma), 1e-3), sigmaY=max(float(sigma), 1e-3))


def _largest_component_fraction(mask: np.ndarray) -> float:
    binary = np.asarray(mask).astype(np.uint8)
    if binary.ndim != 2 or int(binary.max()) == 0:
        return 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0.0
    component_sizes = stats[1:, cv2.CC_STAT_AREA]
    total = int(component_sizes.sum())
    if total <= 0:
        return 0.0
    return float(component_sizes.max() / total)


def _validate_cityscapes_contracts() -> None:
    for label_space, groups in CITYSCAPES_CLASS_GROUPS_BY_LABEL_SPACE.items():
        allowed = CITYSCAPES_ALLOWED_LABEL_VALUES[label_space]
        seen: dict[int, str] = {}
        for group_name, group_ids in groups.items():
            invalid = sorted(value for value in group_ids if value not in allowed)
            if invalid:
                raise ValueError(
                    f"Invalid Cityscapes {label_space} ids for group {group_name}: {invalid}"
                )
            for value in sorted(group_ids):
                previous = seen.get(value)
                if previous is not None:
                    raise ValueError(
                        f"Overlapping Cityscapes {label_space} semantic groups: "
                        f"{previous} and {group_name} both include {value}"
                    )
                seen[value] = group_name


_validate_cityscapes_contracts()
