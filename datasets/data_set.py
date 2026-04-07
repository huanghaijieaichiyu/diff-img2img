import os
import random
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.prepare_data import load_manifest_entries
from scripts.darker import Darker


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _normalize_to_model_space(image_rgb: np.ndarray) -> torch.Tensor:
    image_tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float() / 255.0
    return image_tensor.mul(2.0).sub(1.0)


def _resize_rgb(image_rgb: np.ndarray, size: int) -> np.ndarray:
    interpolation = cv2.INTER_AREA if min(image_rgb.shape[:2]) >= size else cv2.INTER_LINEAR
    return cv2.resize(image_rgb, (size, size), interpolation=interpolation)


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
    ):
        self.image_dir = image_dir
        self.img_size = img_size
        self.phase = phase
        self.online_synthesis = online_synthesis and (phase == "train")
        self.darker_ranges = darker_ranges
        self.darker = None
        self.data = []

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
            manifest_entries = paired_samples if paired_samples is not None else load_manifest_entries(manifest_path)
            self.data = [
                (entry["low_path"], entry["high_path"])
                for entry in manifest_entries
                if entry.get("low_path") and entry.get("high_path")
            ]
            if not self.data:
                raise RuntimeError(f"Prepared training manifest is empty: {manifest_path}")
            return

        if phase == "train":
            subset = "our485"
        elif phase == "test":
            subset = "eval15"
        else:
            raise ValueError("phase must be 'train', 'test' or 'predict'")

        subset_dir = os.path.join(image_dir, subset)
        if not os.path.exists(subset_dir):
            print(f"警告: 目录 {subset_dir} 不存在，数据集为空。")
            return

        high_dir = os.path.join(subset_dir, "high")
        low_dir = os.path.join(subset_dir, "low")
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

    def _get_darker(self):
        if self.darker is None:
            self.darker = Darker(randomize=True, param_ranges=self.darker_ranges)
        return self.darker

    @staticmethod
    def _read_rgb(path: str) -> np.ndarray:
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

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
        high_patch = _resize_rgb(high_patch, self.img_size) if high_patch.shape[:2] != (self.img_size, self.img_size) else high_patch

        darker = self._get_darker()
        low_patch_bgr = darker.degrade_single(cv2.cvtColor(high_patch, cv2.COLOR_RGB2BGR))
        low_patch = cv2.cvtColor(low_patch_bgr, cv2.COLOR_BGR2RGB)

        if random.random() > 0.5:
            low_patch = np.flip(low_patch, axis=1).copy()
            high_patch = np.flip(high_patch, axis=1).copy()

        return low_patch, high_patch

    def _train_precomputed_pair(self, low_rgb: np.ndarray, high_rgb: np.ndarray):
        height, width = low_rgb.shape[:2]
        top, left, crop_h, crop_w = self._random_crop_coords(height, width)
        low_patch = low_rgb[top:top + crop_h, left:left + crop_w]
        high_patch = high_rgb[top:top + crop_h, left:left + crop_w]

        if low_patch.shape[:2] != (self.img_size, self.img_size):
            low_patch = _resize_rgb(low_patch, self.img_size)
            high_patch = _resize_rgb(high_patch, self.img_size)

        if random.random() > 0.5:
            low_patch = np.flip(low_patch, axis=1).copy()
            high_patch = np.flip(high_patch, axis=1).copy()

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
                low_rgb, high_rgb = self._train_precomputed_pair(low_rgb, high_rgb)
        else:
            low_rgb = self._read_rgb(low_path)
            low_rgb = _resize_rgb(low_rgb, self.img_size)
            high_rgb = _resize_rgb(high_rgb, self.img_size)

        return _normalize_to_model_space(low_rgb), _normalize_to_model_space(high_rgb)
