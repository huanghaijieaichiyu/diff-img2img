import os
import random
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
TRAIN_HIGH_DIR_CANDIDATES = (
    ("our485", "high"),
    ("train", "high"),
)
EVAL_DIR_CANDIDATES = (
    ("eval15", "low", "high"),
    ("val", "low", "high"),
)
TRAIN_ROTATION_DEGREES = (-5.0, 5.0)


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


def resolve_training_high_dir(data_dir: str | os.PathLike[str]) -> str:
    data_root = os.fspath(data_dir)
    for parts in TRAIN_HIGH_DIR_CANDIDATES:
        candidate = os.path.join(data_root, *parts)
        if os.path.exists(candidate):
            return candidate
    expected = ", ".join(os.path.join(data_root, *parts)
                         for parts in TRAIN_HIGH_DIR_CANDIDATES)
    raise FileNotFoundError(
        "Training high-light directory not found. "
        f"Expected one of: {expected}"
    )


def resolve_eval_pair_dirs(data_dir: str | os.PathLike[str]) -> tuple[str, str]:
    data_root = os.fspath(data_dir)
    for subset, low_name, high_name in EVAL_DIR_CANDIDATES:
        subset_root = os.path.join(data_root, subset)
        low_dir = os.path.join(subset_root, low_name)
        high_dir = os.path.join(subset_root, high_name)
        if os.path.exists(low_dir) and os.path.exists(high_dir):
            return low_dir, high_dir
    expected = ", ".join(os.path.join(data_root, subset, "{low,high}")
                         for subset, _, _ in EVAL_DIR_CANDIDATES)
    raise FileNotFoundError(
        "Validation low/high directories not found. "
        f"Expected one of: {expected}"
    )


class LowLightDataset(Dataset):
    def __init__(
        self,
        image_dir,
        img_size=256,
        phase="train",
        decode_cache_size: int = 0,
    ):
        self.image_dir = image_dir
        self.img_size = img_size
        self.phase = phase
        self.data = []
        self.decode_cache_size = max(0, int(decode_cache_size or 0))
        self.decode_cache = OrderedDict()

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

        if phase == "train":
            try:
                high_dir = str(resolve_training_high_dir(image_dir))
            except FileNotFoundError as exc:
                print(f"警告: {exc}，数据集为空。")
                return
            low_dir = os.path.join(os.path.dirname(high_dir), "low")
            if not os.path.exists(low_dir):
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
            low_path = os.path.join(low_dir, image_name)
            if os.path.exists(low_path):
                self.data.append((low_path, high_path))

    def __len__(self):
        return len(self.data)

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

    def _train_source_pair(self, low_rgb: np.ndarray, high_rgb: np.ndarray):
        low_tensor = _rgb_to_tensor(low_rgb)
        high_tensor = _rgb_to_tensor(high_rgb)

        if low_tensor.shape[-2:] != high_tensor.shape[-2:]:
            high_tensor = F.interpolate(
                high_tensor.unsqueeze(0),
                size=low_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if min(low_tensor.shape[-2:]) < self.img_size:
            low_tensor = _resize_rgb(low_tensor, self.img_size)
            high_tensor = _resize_rgb(high_tensor, self.img_size)

        top, left, crop_h, crop_w = transforms.RandomCrop.get_params(
            low_tensor, output_size=(self.img_size, self.img_size)
        )
        low_patch = TF.crop(low_tensor, top, left, crop_h, crop_w)
        high_patch = TF.crop(high_tensor, top, left, crop_h, crop_w)

        angle = transforms.RandomRotation.get_params(TRAIN_ROTATION_DEGREES)
        low_patch = TF.rotate(
            low_patch, angle, interpolation=TF.InterpolationMode.BILINEAR)
        high_patch = TF.rotate(
            high_patch, angle, interpolation=TF.InterpolationMode.BILINEAR)

        if random.random() > 0.5:
            low_patch = TF.hflip(low_patch)
            high_patch = TF.hflip(high_patch)

        return low_patch, high_patch

    def __getitem__(self, idx):
        if self.phase == "predict":
            image_rgb = self._read_rgb(self.data[idx])
            image_rgb = _resize_rgb(image_rgb, self.img_size)
            return _normalize_to_model_space(image_rgb)

        low_path, high_path = self.data[idx]
        high_rgb = self._read_rgb(high_path)

        if self.phase == "train":
            low_rgb = self._read_rgb(low_path)
            low_rgb, high_rgb = self._train_source_pair(low_rgb, high_rgb)
        else:
            low_rgb = self._read_rgb(low_path)
            low_rgb = _resize_rgb(low_rgb, self.img_size)
            high_rgb = _resize_rgb(high_rgb, self.img_size)

        return _normalize_to_model_space(low_rgb), _normalize_to_model_space(high_rgb)
