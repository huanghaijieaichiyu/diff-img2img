import cv2
import os
import numpy as np
from tqdm import tqdm
import random
import time
from typing import Optional, Union, Dict, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


# ============================================================================
#  Noise Models
# ============================================================================


def configure_cpu_image_backend(num_threads: int = 1) -> None:
    """
    Keep OpenCV/BLAS thread fan-out under control inside worker processes.

    The degradation pipeline already parallelizes across processes, so allowing
    OpenCV to spawn many internal threads per worker usually hurts throughput on
    CPU-bound prepare jobs.
    """
    thread_value = str(max(1, int(num_threads)))
    for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_name] = thread_value
    try:
        cv2.setNumThreads(int(thread_value))
    except Exception:
        pass
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def add_poisson_gaussian_noise(image: np.ndarray,
                               k: float = 0.02,
                               sigma_read: float = 5.0) -> np.ndarray:
    """
    Signal-dependent Poisson-Gaussian mixed noise model.
    More realistic than pure Gaussian — models sensor physics:
      - Poisson (shot noise): proportional to signal intensity
      - Gaussian (read noise): additive, signal-independent

    Args:
        image: Input image [0, 255] uint8
        k: Poisson noise scaling factor (higher = more shot noise)
        sigma_read: Standard deviation of read noise (Gaussian)
    """
    if image is None:
        return None
    img_float = image.astype(np.float32)

    # Shot noise (signal-dependent Poisson)
    # Scale down, apply poisson, scale back
    safe_img = np.maximum(img_float, 0)
    if k > 0:
        poisson_noise = np.random.poisson(safe_img * k).astype(np.float32) / k - img_float
    else:
        poisson_noise = np.zeros_like(img_float)

    # Read noise (additive Gaussian, independent per channel).
    # Vectorize the sampling to avoid a small Python loop per image.
    channel_sigma = sigma_read * np.random.uniform(0.8, 1.2, size=(1, 1, img_float.shape[2])).astype(np.float32)
    read_noise = np.random.normal(0.0, 1.0, size=img_float.shape).astype(np.float32) * channel_sigma

    noisy = img_float + poisson_noise + read_noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ============================================================================
#  Headlight / Light Source Simulation
# ============================================================================

def create_random_headlight_mask(height: int, width: int,
                                 num_lights: Optional[int] = None) -> np.ndarray:
    """
    Generate a randomized multi-source headlight mask.
    Simulates varying positions, shapes, and intensities of light sources
    as seen in real driving scenarios (oncoming cars, streetlights, reflections).

    Args:
        height, width: Image dimensions
        num_lights: Number of light sources (None = random 0-4)
    """
    mask = np.zeros((height, width), dtype=np.float32)

    if num_lights is None:
        area_ratio = min(1.0, (height * width) / float(512 * 512))
        if area_ratio < 0.35:
            max_lights = 1
        elif area_ratio < 0.75:
            max_lights = 2
        else:
            max_lights = 4
        num_lights = random.randint(0, max_lights)

    if num_lights == 0:
        return mask

    Y, X = np.ogrid[:height, :width]

    for _ in range(num_lights):
        # Randomize light source position (more likely in lower half)
        cx = random.uniform(0.1, 0.9) * width
        cy = random.uniform(0.4, 0.95) * height

        # Randomize elliptical shape
        radius_x = random.uniform(0.08, 0.35) * width
        radius_y = random.uniform(0.06, 0.25) * height

        # Compute normalized distance
        dist = ((X - cx) / max(radius_x, 1)) ** 2 + ((Y - cy) / max(radius_y, 1)) ** 2

        # Randomize falloff sharpness and intensity
        sharpness = random.uniform(0.8, 3.5)
        intensity = random.uniform(0.2, 1.0)

        light = np.exp(-dist * sharpness) * intensity
        mask = np.maximum(mask, light)

    # Smooth the result
    ksize_w = int(width * 0.04) | 1
    ksize_h = int(height * 0.04) | 1
    mask = cv2.GaussianBlur(mask, (ksize_w, ksize_h), 0)

    return np.clip(mask, 0, 1).astype(np.float32)


def enforce_reasonable_exposure(source: np.ndarray,
                                degraded: np.ndarray,
                                min_ratio: float = 0.10,
                                max_ratio: float = 0.55) -> np.ndarray:
    """
    Keep synthesized low-light samples inside a practical luminance range.
    This prevents extremely dark or barely-darkened outliers.
    """
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    degraded_gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)

    source_mean = max(source_gray.mean(), 1.0)
    degraded_mean = degraded_gray.mean()
    ratio = degraded_mean / source_mean

    adjusted = degraded.astype(np.float32)

    if ratio < min_ratio:
        alpha = min(1.0, (min_ratio - ratio) / max(min_ratio, 1e-6))
        adjusted = adjusted * (1.0 - alpha) + source.astype(np.float32) * alpha * 0.35
    elif ratio > max_ratio:
        scale = max_ratio / max(ratio, 1e-6)
        adjusted = adjusted * scale

    return np.clip(adjusted, 0, 255).astype(np.uint8)


# ============================================================================
#  Additional Degradation Effects
# ============================================================================

def apply_vignetting(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Simulate lens vignetting (darkened corners/edges).

    Args:
        image: Input image [0, 255] uint8
        strength: Vignetting strength (0 = none, 1 = strong)
    """
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]

    # Normalized distance from center
    cy, cx = h / 2, w / 2
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max_dist

    # Smooth falloff
    vignette = 1.0 - strength * (dist ** 2)
    vignette = np.clip(vignette, 0, 1).astype(np.float32)

    result = image.astype(np.float32) * vignette[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_jpeg_artifact(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Simulate JPEG compression artifacts.

    Args:
        image: Input image [0, 255] uint8
        quality: JPEG quality (lower = more artifacts)
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded


def apply_motion_blur(image: np.ndarray,
                      kernel_size: int = 7,
                      angle: float = 0) -> np.ndarray:
    """
    Simulate motion blur caused by long exposure in low light.

    Args:
        image: Input image [0, 255] uint8
        kernel_size: Blur kernel size (odd number)
        angle: Motion direction in degrees
    """
    kernel_size = max(3, kernel_size | 1)  # ensure odd
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel_size

    # Rotate kernel by angle
    center = (kernel_size // 2, kernel_size // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel /= kernel.sum() + 1e-8

    return cv2.filter2D(image, -1, kernel)


# ============================================================================
#  Main Darker Class
# ============================================================================

_BATCH_DARKER = None


def _batch_worker_initializer(data_dir: str,
                              gamma: float,
                              linear_attenuation: float,
                              phase: str,
                              randomize: bool,
                              param_ranges: Optional[Dict[str, Any]]) -> None:
    global _BATCH_DARKER
    configure_cpu_image_backend(num_threads=1)
    _BATCH_DARKER = Darker(
        data_dir=data_dir,
        gamma=gamma,
        linear_attenuation=linear_attenuation,
        phase=phase,
        randomize=randomize,
        param_ranges=param_ranges,
    )


def _batch_process_single_image(filename: str) -> tuple[str, bool, str]:
    global _BATCH_DARKER
    try:
        if _BATCH_DARKER is None:
            return filename, False, "Batch worker is not initialized."
        input_path = _BATCH_DARKER.high_dir / filename
        img = cv2.imread(str(input_path))
        if img is None:
            return filename, False, f"Failed to read image: {input_path}"

        dark = _BATCH_DARKER.degrade_single(img)
        output_path = _BATCH_DARKER.low_dir / filename
        if not cv2.imwrite(str(output_path), dark):
            return filename, False, f"Failed to write image: {output_path}"
        return filename, True, ""
    except Exception as exc:  # pragma: no cover - defensive path
        return filename, False, str(exc)


class Darker:
    """
    Enhanced low-light image synthesis engine with realistic degradation.

    Supports two modes:
      1. Offline batch processing (process_images) — for pre-generating datasets
      2. Single-image processing (degrade_single) — for online synthesis

    Key improvements over the original:
      - Randomized degradation parameters per image
      - Poisson-Gaussian mixed noise model
      - Multi-source random headlight simulation
      - Vignetting, motion blur, JPEG artifact simulation
      - Configurable parameter ranges for diverse training data
    """

    # Default parameter ranges for randomized degradation
    DEFAULT_RANGES = {
        "gamma": (1.5, 4.0),
        "linear_attenuation": (0.25, 0.7),
        "saturation_factor": (0.4, 0.85),
        "color_shift_factor": (0.0, 0.12),
        "headlight_boost": (0.0, 0.9),
        # Poisson-Gaussian noise
        "noise_k": (0.005, 0.04),
        "noise_sigma_read": (2.0, 15.0),
        # Extra effects probabilities
        "vignette_prob": 0.5,
        "vignette_strength": (0.2, 0.6),
        "motion_blur_prob": 0.2,
        "motion_blur_kernel": (3, 9),
        "jpeg_artifact_prob": 0.3,
        "jpeg_quality": (40, 85),
    }

    def __init__(self, data_dir: Optional[Union[str, Path]] = None,
                 gamma: float = 2.5,
                 linear_attenuation: float = 0.5,
                 phase: str = "train",
                 randomize: bool = True,
                 param_ranges: Optional[Dict[str, Any]] = None):
        """
        Args:
            data_dir: Root directory of the dataset
            gamma: Default gamma value (used when randomize=False)
            linear_attenuation: Default attenuation (used when randomize=False)
            phase: "train" or "test"
            randomize: If True, randomize all degradation parameters per image
            param_ranges: Custom parameter ranges (overrides DEFAULT_RANGES)
        """
        self.gamma = gamma
        self.linear_attenuation = linear_attenuation
        self.phase = phase
        self.randomize = randomize
        self.data_dir = Path(data_dir) if data_dir else None

        # Merge custom ranges with defaults
        self.ranges = dict(self.DEFAULT_RANGES)
        if param_ranges:
            self.ranges.update(param_ranges)

        if self.data_dir:
            base_dir = "our485" if phase == "train" else "eval15"
            self.high_dir = self.data_dir / base_dir / "high"
            self.low_dir = self.data_dir / base_dir / "low"
            os.makedirs(self.low_dir, exist_ok=True)

    def _sample_params(self) -> dict:
        """Sample random degradation parameters from configured ranges."""
        r = self.ranges
        return {
            "gamma": random.uniform(*r["gamma"]),
            "linear_attenuation": random.uniform(*r["linear_attenuation"]),
            "saturation_factor": random.uniform(*r["saturation_factor"]),
            "color_shift_factor": random.uniform(*r["color_shift_factor"]),
            "headlight_boost": random.uniform(*r["headlight_boost"]),
            "noise_k": random.uniform(*r["noise_k"]),
            "noise_sigma_read": random.uniform(*r["noise_sigma_read"]),
            "use_vignette": random.random() < r["vignette_prob"],
            "vignette_strength": random.uniform(*r["vignette_strength"]),
            "use_motion_blur": random.random() < r["motion_blur_prob"],
            "motion_blur_kernel": random.choice(range(r["motion_blur_kernel"][0],
                                                       r["motion_blur_kernel"][1] + 1, 2)),
            "motion_blur_angle": random.uniform(0, 180),
            "use_jpeg": random.random() < r["jpeg_artifact_prob"],
            "jpeg_quality": random.randint(*r["jpeg_quality"]),
        }

    def degrade_single(self, img: np.ndarray, params: Optional[dict] = None) -> np.ndarray:
        """
        Apply full degradation pipeline to a single image.
        This is the primary entry point for online (in-Dataset) synthesis.

        Args:
            img: Input BGR image [0, 255] uint8
            params: Degradation parameters dict. If None, samples randomly.

        Returns:
            Degraded BGR image [0, 255] uint8
        """
        if img is None:
            return None

        if params is None:
            params = self._sample_params() if self.randomize else {
                "gamma": self.gamma,
                "linear_attenuation": self.linear_attenuation,
                "saturation_factor": 0.6,
                "color_shift_factor": 0.1,
                "headlight_boost": 0.5,
                "noise_k": 0.02,
                "noise_sigma_read": 5.0,
                "use_vignette": False,
                "vignette_strength": 0.0,
                "use_motion_blur": False,
                "motion_blur_kernel": 5,
                "motion_blur_angle": 0,
                "use_jpeg": False,
                "jpeg_quality": 70,
            }

        h, w = img.shape[:2]
        gamma = params["gamma"]
        attenuation = params["linear_attenuation"]

        # --- 1. Headlight mask ---
        mask = create_random_headlight_mask(h, w)

        # --- 2. Gamma + Attenuation in V channel ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[:, :, 2] / 255.0

        v_dark = (v ** gamma) * attenuation

        # --- 3. Headlight glow ---
        boost = params["headlight_boost"]
        v_final = v_dark * (1 - mask * boost) + v * (mask * boost)

        hsv[:, :, 2] = np.clip(v_final * 255.0, 0, 255)

        # --- 4. Saturation reduction ---
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params["saturation_factor"], 0, 255)

        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # --- 5. Color shift (Purkinje effect: stronger in locally darker regions) ---
        shift_factor = params["color_shift_factor"]
        local_darkness = np.clip(1.0 - v_final, 0.0, 1.0)
        shift_map = shift_factor * (1.0 - mask) * local_darkness * 255
        adjusted[:, :, 0] += shift_map     # Blue +
        adjusted[:, :, 2] -= shift_map * 0.5  # Red -
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        # --- 6. Vignetting ---
        if params.get("use_vignette", False):
            adjusted = apply_vignetting(adjusted, strength=params["vignette_strength"])

        # --- 7. Motion blur ---
        if params.get("use_motion_blur", False):
            adjusted = apply_motion_blur(adjusted,
                                         kernel_size=params["motion_blur_kernel"],
                                         angle=params["motion_blur_angle"])

        # --- 8. Realistic noise (Poisson-Gaussian) ---
        adjusted = add_poisson_gaussian_noise(adjusted,
                                              k=params["noise_k"],
                                              sigma_read=params["noise_sigma_read"])

        # --- 9. Optional JPEG compression artifact ---
        if params.get("use_jpeg", False):
            adjusted = apply_jpeg_artifact(adjusted, quality=params["jpeg_quality"])

        return enforce_reasonable_exposure(img, adjusted)

    def _process_single_image(self, filename: str):
        """Process a single image with randomized or fixed degradation."""
        try:
            input_path = self.high_dir / filename
            img = cv2.imread(str(input_path))
            if img is None:
                return

            dark = self.degrade_single(img)
            output_path = self.low_dir / filename
            cv2.imwrite(str(output_path), dark)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    def process_images(self, num_workers: int = None, **kwargs):
        """Batch process all images in the dataset directory."""
        if not self.data_dir:
            raise RuntimeError("No data_dir provided")

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        files = [f for f in os.listdir(
            self.high_dir) if f.lower().endswith(valid_exts)]

        if not files:
            print(f"No images found in {self.high_dir}")
            return

        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1))

        print(f"Processing {len(files)} images with {num_workers} workers...")
        print(f"Randomization: {'ON' if self.randomize else 'OFF'}")
        print(f"Default gamma={self.gamma}, attenuation={self.linear_attenuation}")
        chunksize = max(1, len(files) // max(1, num_workers * 8))
        print(f"OpenCV threads per worker: 1 | chunksize: {chunksize}")
        started_at = time.perf_counter()
        processed = 0
        failed = 0

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_batch_worker_initializer,
            initargs=(
                str(self.data_dir),
                float(self.gamma),
                float(self.linear_attenuation),
                str(self.phase),
                bool(self.randomize),
                self.ranges,
            ),
        ) as executor:
            for filename, ok, error in tqdm(
                executor.map(_batch_process_single_image, files, chunksize=chunksize),
                total=len(files),
                desc="Progress",
            ):
                processed += 1
                if not ok:
                    failed += 1
                    print(f"Error processing {filename}: {error}")

        elapsed = max(time.perf_counter() - started_at, 1e-8)
        succeeded = processed - failed
        print(
            f"Done! Output saved to {self.low_dir} | "
            f"succeeded={succeeded} failed={failed} elapsed={elapsed:.2f}s "
            f"throughput={processed / elapsed:.2f} img/s"
        )


if __name__ == '__main__':
    # =====================================================================
    #  Configuration for realistic low-light synthesis
    #  Key change: randomize=True enables per-image random degradation
    # =====================================================================

    data_dir = "/mnt/f/datasets/nuscenes_lol"  # Change to your dataset path

    # Custom parameter ranges (optional, these are already the defaults)
    custom_ranges = {
        "gamma": (1.5, 4.0),              # Much lower than old 6.5
        "linear_attenuation": (0.25, 0.7), # Higher than old 0.15
        "saturation_factor": (0.4, 0.85),
        "color_shift_factor": (0.0, 0.12),
        "headlight_boost": (0.0, 0.9),
        "noise_k": (0.005, 0.04),
        "noise_sigma_read": (2.0, 15.0),
        "vignette_prob": 0.5,
        "vignette_strength": (0.2, 0.6),
        "motion_blur_prob": 0.2,
        "motion_blur_kernel": (3, 9),
        "jpeg_artifact_prob": 0.3,
        "jpeg_quality": (40, 85),
    }

    print("Initializing enhanced Darker with randomized degradation...")
    try:
        darker = Darker(
            data_dir=data_dir,
            phase="train",
            randomize=True,
            param_ranges=custom_ranges,
        )
        darker.process_images()

        print("Darker script completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your data_dir is correct.")
