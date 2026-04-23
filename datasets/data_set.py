import os
import random
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets.prepare_data import (
    load_manifest_entries,
    load_manifest_info,
    manifest_info_path,
    resolve_manifest_entry_path,
    resolve_eval_pair_dirs,
    resolve_training_high_dir,
)
from utils.torch_low_light_degrader import TorchLowLightDegrader


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _rgb_to_tensor(image_rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(image_rgb, torch.Tensor):
        return image_rgb.float()
    image_tensor = torch.from_numpy(np.ascontiguousarray(
        image_rgb.transpose(2, 0, 1))).float()
    return image_tensor / 255.0


def _normalize_to_model_space(image_rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
    image_tensor = _rgb_to_tensor(image_rgb)
    if image_tensor.max().item() > 1.5:
        image_tensor = image_tensor / 255.0
    return image_tensor.mul(2.0).sub(1.0)


def _resize_rgb(image_rgb: np.ndarray | torch.Tensor, size: int) -> torch.Tensor:
    image_tensor = _rgb_to_tensor(image_rgb)
    if image_tensor.ndim != 3:
        raise ValueError(
            f"Expected image tensor with shape (C, H, W), got {tuple(image_tensor.shape)}")
    if image_tensor.shape[-2:] == (size, size):
        return image_tensor
    resized = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


class LowLightDataset(Dataset):
    def __init__(
        self,
        image_dir,
        img_size=256,
        phase="train",
        online_synthesis: bool = False,
        darker_ranges: Optional[dict] = None,
        manifest_path: Optional[str] = None,
        paired_samples: Optional[list[dict]] = None,
        decode_cache_size: int = 0,
        prepared_cache_dir: Optional[str] = None,
    ):
        self.image_dir = image_dir
        self.img_size = img_size
        self.phase = phase
        self.online_synthesis = online_synthesis and (phase == "train")
        self.darker_ranges = darker_ranges
        self.torch_degrader = None
        self.data = []
        self.decode_cache_size = max(0, int(decode_cache_size or 0))
        self.decode_cache = OrderedDict()
        self.prepared_cache_dir = prepared_cache_dir

        if phase == "predict":
            if os.path.exists(image_dir):
                for root, _, files in os.walk(image_dir):
                    for filename in files:
                        if filename.lower().endswith(VALID_EXTENSIONS):
                            self.data.append(os.path.join(root, filename))
            self.data.sort()
            if not self.data:
                print(f"警告: 在 {image_dir} 中未找到图片。")
            return

        if phase == "train" and (manifest_path or paired_samples):
            manifest_entries = paired_samples if paired_samples is not None else self._load_prepared_entries(
                manifest_path)
            self.data = []
            missing_entries = 0
            for entry in manifest_entries:
                if not entry.get("low_path") or not entry.get("high_path"):
                    continue
                use_train_cache = int(
                    entry.get("train_resolution") or 0) == int(self.img_size)
                low_field = "train_low_path" if use_train_cache and entry.get(
                    "train_low_path") else "low_path"
                high_field = "train_high_path" if use_train_cache and entry.get(
                    "train_high_path") else "high_path"
                low_path = resolve_manifest_entry_path(
                    entry,
                    low_field,
                    data_dir=self.image_dir,
                    prepared_cache_dir=self.prepared_cache_dir,
                )
                high_path = resolve_manifest_entry_path(
                    entry,
                    high_field,
                    data_dir=self.image_dir,
                    prepared_cache_dir=self.prepared_cache_dir,
                )
                raw_low_path = str(entry.get(low_field) or "")
                raw_high_path = str(entry.get(high_field) or "")
                allow_missing_for_external_paths = os.path.isabs(
                    raw_low_path) and os.path.isabs(raw_high_path)
                if not low_path or not high_path:
                    missing_entries += 1
                    continue
                if not allow_missing_for_external_paths and (not os.path.exists(low_path) or not os.path.exists(high_path)):
                    missing_entries += 1
                    continue
                self.data.append((low_path, high_path))
            if missing_entries > 0:
                print(
                    f"警告: prepared manifest 中有 {missing_entries} 条记录无法解析或文件不存在，已跳过。")
            if not self.data:
                raise RuntimeError(
                    f"Prepared training manifest is empty: {manifest_path}")
            return

        if phase == "train":
            try:
                high_dir = str(resolve_training_high_dir(image_dir))
            except FileNotFoundError as exc:
                print(f"警告: {exc}，数据集为空。")
                return
            low_dir = os.path.join(os.path.dirname(high_dir), "low")
            if not self.online_synthesis and not os.path.exists(low_dir):
                print(f"警告: 目录 {low_dir} 不存在，训练数据集为空。")
                return
        elif phase == "test":
            try:
                low_dir_path, high_dir_path = resolve_eval_pair_dirs(image_dir)
            except FileNotFoundError as exc:
                print(f"警告: {exc}，数据集为空。")
                return
            low_dir = str(low_dir_path)
            high_dir = str(high_dir_path)
        else:
            raise ValueError("phase must be 'train', 'test' or 'predict'")

        image_names = [
            filename for filename in os.listdir(high_dir)
            if filename.lower().endswith(VALID_EXTENSIONS)
        ]
        image_names.sort()

        for image_name in image_names:
            high_path = os.path.join(high_dir, image_name)
            if self.online_synthesis:
                self.data.append((None, high_path))
            else:
                low_path = os.path.join(low_dir, image_name)
                if os.path.exists(low_path):
                    self.data.append((low_path, high_path))

    def __len__(self):
        return len(self.data)

    def _get_torch_degrader(self):
        if self.torch_degrader is None:
            self.torch_degrader = TorchLowLightDegrader(
                randomize=True,
                param_ranges=self.darker_ranges,
            )
        return self.torch_degrader

    @staticmethod
    def _load_prepared_entries(manifest_path: str | None) -> list[dict]:
        if not manifest_path:
            return []
        info_payload = load_manifest_info(manifest_info_path(manifest_path))
        if isinstance(info_payload, dict):
            info_entries = info_payload.get("entries")
            if info_payload.get("manifest_path") == os.path.abspath(manifest_path) and isinstance(info_entries, list):
                return info_entries
        return load_manifest_entries(manifest_path)

    def _read_rgb(self, path: str) -> np.ndarray:
        if self.decode_cache_size > 0 and path in self.decode_cache:
            self.decode_cache.move_to_end(path)
            return self.decode_cache[path]

        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.decode_cache_size > 0:
            self.decode_cache[path] = image_rgb
            if len(self.decode_cache) > self.decode_cache_size:
                self.decode_cache.popitem(last=False)
        return image_rgb

    def _random_crop_coords(self, height: int, width: int):
        crop_h = min(self.img_size, height)
        crop_w = min(self.img_size, width)
        top = 0 if height == crop_h else random.randint(0, height - crop_h)
        left = 0 if width == crop_w else random.randint(0, width - crop_w)
        return top, left, crop_h, crop_w

    def _train_online_pair(self, high_rgb: np.ndarray):
        height, width = high_rgb.shape[:2]
        top, left, crop_h, crop_w = self._random_crop_coords(height, width)
        high_patch = high_rgb[top:top + crop_h, left:left + crop_w]
        high_patch = _resize_rgb(high_patch, self.img_size)

        torch_degrader = self._get_torch_degrader()
        low_patch = torch_degrader.degrade(high_patch)

        if random.random() > 0.5:
            low_patch = torch.flip(low_patch, dims=(2,))
            high_patch = torch.flip(high_patch, dims=(2,))

        return low_patch, high_patch

    def _train_precomputed_pair(self, low_rgb: np.ndarray, high_rgb: np.ndarray):
        low_tensor = _rgb_to_tensor(low_rgb)
        high_tensor = _rgb_to_tensor(high_rgb)
        height, width = low_tensor.shape[-2:]
        top, left, crop_h, crop_w = self._random_crop_coords(height, width)
        low_patch = low_tensor[:, top:top + crop_h, left:left + crop_w]
        high_patch = high_tensor[:, top:top + crop_h, left:left + crop_w]

        if low_patch.shape[-2:] != (self.img_size, self.img_size):
            low_patch = _resize_rgb(low_patch, self.img_size)
            high_patch = _resize_rgb(high_patch, self.img_size)

        if random.random() > 0.5:
            low_patch = torch.flip(low_patch, dims=(2,))
            high_patch = torch.flip(high_patch, dims=(2,))

        return low_patch, high_patch

    def __getitem__(self, idx):
        if self.phase == "predict":
            image_rgb = self._read_rgb(self.data[idx])
            image_rgb = _resize_rgb(image_rgb, self.img_size)
            return _normalize_to_model_space(image_rgb)

        low_path, high_path = self.data[idx]
        high_rgb = self._read_rgb(high_path)

        if self.phase == "train":
            if self.online_synthesis:
                low_rgb, high_rgb = self._train_online_pair(high_rgb)
            else:
                low_rgb = self._read_rgb(low_path)
                low_rgb, high_rgb = self._train_precomputed_pair(
                    low_rgb, high_rgb)
        else:
            low_rgb = self._read_rgb(low_path)
            low_rgb = _resize_rgb(low_rgb, self.img_size)
            high_rgb = _resize_rgb(high_rgb, self.img_size)

        return _normalize_to_model_space(low_rgb), _normalize_to_model_space(high_rgb)
