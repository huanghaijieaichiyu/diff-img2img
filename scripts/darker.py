import cv2
import os
import numpy as np
from tqdm import tqdm
import random
from typing import Optional, Union
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 10) -> np.ndarray:
    if image is None:
        return None
    img_float = image.astype(np.float32)
    row, col, ch = img_float.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(img_float + gauss, 0, 255)
    return noisy.astype(np.uint8)


def create_headlight_mask(height: int, width: int, center_y_factor=0.9, center_x_factor=0.5,
                          beam_width_factor=0.6, falloff_sharpness=2.0, max_intensity=1.0) -> np.ndarray:
    Y, X = np.ogrid[:height, :width]
    dist_y = np.maximum(0, (center_y_factor * height - Y) /
                        (center_y_factor * height))
    center_x = center_x_factor * width
    dist_x = np.abs(X - center_x) / (beam_width_factor * width / 2)

    falloff_y = np.exp(-falloff_sharpness * dist_y)
    falloff_x = np.exp(-falloff_sharpness * 0.5 *
                       np.maximum(0, dist_x - (1-dist_y)*0.3)**2)

    mask = np.clip(max_intensity * falloff_y * falloff_x, 0, 1)
    return cv2.GaussianBlur(mask, (int(width*0.05) | 1, int(height*0.05) | 1), 0).astype(np.float32)


class Darker:
    def __init__(self, data_dir: Optional[Union[str, Path]] = None,
                 gamma: float = 2.5,  # SOTA style gamma correction
                 linear_attenuation: float = 0.5,  # Linear scaling factor
                 phase: str = "train"):
        self.gamma = gamma
        self.linear_attenuation = linear_attenuation
        self.phase = phase
        self.data_dir = Path(data_dir) if data_dir else None
        self.mask_cache = {}

        if self.data_dir:
            base_dir = "our485" if phase == "train" else "eval15"
            self.high_dir = self.data_dir / base_dir / "high"
            self.low_dir = self.data_dir / base_dir / "low"
            os.makedirs(self.low_dir, exist_ok=True)

    def get_mask(self, h, w, **params):
        key = (h, w)
        if key not in self.mask_cache:
            self.mask_cache[key] = create_headlight_mask(h, w, **params)
        return self.mask_cache[key]

    def adjust_image(self, img: np.ndarray, mask: np.ndarray,
                     saturation_factor: float = 0.6,
                     color_shift_factor: float = 0.1,
                     noise_sigma: float = 5.0,
                     headlight_boost: float = 0.8) -> np.ndarray:

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[:, :, 2] / 255.0

        # 1. Gamma Correction + Linear Attenuation (More realistic physics approximation)
        # Low light = (High light)^Gamma * Attenuation
        v_dark = (v ** self.gamma) * self.linear_attenuation

        # 2. Headlight Simulation
        # Brighten areas based on mask
        mask_val = mask if mask.ndim == 2 else mask[:, :, 0]
        v_final = v_dark * (1 - mask_val * headlight_boost) + \
            v * (mask_val * headlight_boost)

        hsv[:, :, 2] = np.clip(v_final * 255.0, 0, 255)

        # 3. Saturation adjustment
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)

        adjusted_bgr = cv2.cvtColor(hsv.astype(
            np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # 4. Color Shift (Purkinje effect - blue shift in dark)
        # Only shift in dark areas (low mask value)
        shift_map = color_shift_factor * \
            (1.0 - mask_val) * np.mean(v_final) * 255
        adjusted_bgr[:, :, 0] += shift_map  # Blue +
        adjusted_bgr[:, :, 2] -= shift_map * 0.5  # Red -

        adjusted_bgr = np.clip(adjusted_bgr, 0, 255).astype(np.uint8)

        # 5. Noise
        if noise_sigma > 0:
            return add_gaussian_noise(adjusted_bgr, sigma=noise_sigma)
        return adjusted_bgr

    def _process_single_image(self, filename: str, mask_params: dict, effect_params: dict):
        """Helper function to process a single image in a worker process."""
        try:
            input_path = self.high_dir / filename
            img = cv2.imread(str(input_path))
            if img is None:
                return

            mask = self.get_mask(img.shape[0], img.shape[1], **mask_params)
            dark = self.adjust_image(img, mask, **effect_params)

            output_path = self.low_dir / filename
            cv2.imwrite(str(output_path), dark)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    def process_images(self, mask_params={}, num_workers: int = None, **effect_params):
        if not self.data_dir:
            raise RuntimeError("No data_dir provided")

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        files = [f for f in os.listdir(
            self.high_dir) if f.lower().endswith(valid_exts)]

        if not files:
            print(f"No images found in {self.high_dir}")
            return

        if num_workers is None:
            # Use all available cores
            num_workers = max(1, (os.cpu_count() or 1))

        print(f"Processing {len(files)} images with {num_workers} workers...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self._process_single_image,
                                f, mask_params, effect_params)
                for f in files
            ]

            for _ in tqdm(as_completed(futures), total=len(files), desc="Progress"):
                pass


if __name__ == '__main__':
    # Configuration for realistic low-light synthesis
    config = {
        "gamma": 6.5,                # Higher = darker midtones
        "linear_attenuation": 0.15,  # Global brightness scale
        "headlight_boost": 0.9,      # Headlight brightness intensity
        "noise_sigma": 8.0,          # Sensor noise
        "saturation_factor": 0.5,    # Color washout
        "color_shift_factor": 0.15,  # Blue tint strength
        "mask_params": {
            "center_y_factor": 0.9,
            "beam_width_factor": 0.7,
            "falloff_sharpness": 2.5
        }
    }

    # Separate parameters
    init_keys = ['gamma', 'linear_attenuation']
    init_params = {k: v for k, v in config.items() if k in init_keys}

    mask_params = config.get('mask_params', {})

    # Effect params are those not in init_keys and not mask_params
    effect_params = {k: v for k, v in config.items(
    ) if k not in init_keys and k != 'mask_params'}

    # Note: Update data_dir to your actual dataset path
    data_dir = "/mnt/f/datasets/nuscenes_lol"  # Change this to your dataset path

    print(f"Initializing Darker with: {init_params}")
    try:
        darker = Darker(data_dir=data_dir, phase="train", **init_params)
        darker.process_images(mask_params=mask_params, **effect_params)

        # Also process test set if needed
        # darker_test = Darker(data_dir=data_dir, phase="test", **init_params)
        # darker_test.process_images(mask_params=mask_params, **effect_params)

        print("Darker script completed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your data_dir is correct.")
