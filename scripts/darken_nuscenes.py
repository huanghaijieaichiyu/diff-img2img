"""
Batch darken all camera images in a nuscenes dataset folder.

Usage:
    python scripts/darken_nuscenes.py

This script:
  1. Scans /mnt/f/datasets/nuscenes for all camera images (samples/CAM_*, sweeps/CAM_*)
  2. Applies randomized low-light degradation via the Darker engine
  3. Saves results to /mnt/f/datasets/nuscenes_lowlight with identical directory structure
  4. Copies all non-image files (annotations, lidar, radar, etc.) as symlinks
"""


import cv2
import os
import sys
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

try:
    from scripts.darker import Darker
except ImportError:
    raise ImportError("Please install the project first: pip install -e .")
# =============================================
#  Configuration
# =============================================
SRC_DIR = Path("/mnt/f/datasets/nuscenes")
DST_DIR = Path("/mnt/f/datasets/nuscenes_lowlight")

# Only process camera folders
CAMERA_PREFIXES = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                   "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

MAX_WORKERS = 8  # Parallel processing workers


def collect_image_tasks(src_root: Path, dst_root: Path):
    """Find all camera images and build (src, dst) pairs."""
    tasks = []
    for sub in ("samples", "sweeps"):
        sub_dir = src_root / sub
        if not sub_dir.exists():
            continue
        for cam_folder in sorted(sub_dir.iterdir()):
            if not cam_folder.is_dir():
                continue
            if not cam_folder.name.startswith(CAMERA_PREFIXES):
                continue
            for img_file in cam_folder.iterdir():
                if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    rel = img_file.relative_to(src_root)
                    tasks.append((str(img_file), str(dst_root / rel)))
    return tasks


def process_single_image(args):
    """Worker function: darken one image and save."""
    src_path, dst_path, darker_config = args
    try:
        img = cv2.imread(src_path)
        if img is None:
            return src_path, False, "Failed to read"

        darker = Darker(randomize=True, param_ranges=darker_config)
        degraded = darker.degrade_single(img)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, degraded)
        return src_path, True, ""
    except Exception as e:
        return src_path, False, str(e)


def copy_non_image_data(src_root: Path, dst_root: Path):
    """Copy/symlink non-camera directories and files to preserve dataset structure."""
    # Directories to copy entirely (as symlinks for speed)
    for item in src_root.iterdir():
        dst_item = dst_root / item.name
        if dst_item.exists():
            continue

        if item.name in ("samples", "sweeps"):
            # For samples/sweeps: create dir, copy non-camera subfolders as symlinks
            dst_item.mkdir(parents=True, exist_ok=True)
            for sub in item.iterdir():
                dst_sub = dst_item / sub.name
                if dst_sub.exists():
                    continue
                if sub.is_dir() and not sub.name.startswith(CAMERA_PREFIXES):
                    # Non-camera data (LIDAR, RADAR): symlink
                    os.symlink(str(sub.resolve()), str(dst_sub))
        elif item.is_dir():
            # Other directories (v1.0-mini, maps, etc.): symlink
            os.symlink(str(item.resolve()), str(dst_item))
        elif item.is_file():
            # Root files (pkl, LICENSE, etc.): symlink
            os.symlink(str(item.resolve()), str(dst_item))


def main():
    print(f"Source:      {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    print()

    assert SRC_DIR.exists(), f"Source not found: {SRC_DIR}"
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy non-image data (symlinks)
    print("📂 Linking non-image data...")
    copy_non_image_data(SRC_DIR, DST_DIR)

    # 2. Collect image tasks
    print("🔍 Scanning for camera images...")
    tasks = collect_image_tasks(SRC_DIR, DST_DIR)
    print(f"   Found {len(tasks)} camera images to process.\n")

    if not tasks:
        print("No images found!")
        return

    # 3. Darker configuration
    darker_config = {
        "gamma": (1.5, 4.0),
        "linear_attenuation": (0.25, 0.7),
        "saturation_factor": (0.4, 0.85),
        "color_shift_factor": (0.0, 0.12),
        "headlight_boost": (0.0, 0.9),
        "noise_k": (0.0, 0.00),
        "noise_sigma_read": (0, 0),
        "vignette_prob": 0.5,
        "vignette_strength": (0.2, 0.6),
        "motion_blur_prob": 0.15,
        "motion_blur_kernel": (3, 7),
        "jpeg_artifact_prob": 0.2,
        "jpeg_quality": (50, 90),
    }

    # 4. Process in parallel
    print(f"🌙 Darkening images with {MAX_WORKERS} workers...")
    worker_args = [(src, dst, darker_config) for src, dst in tasks]

    success, fail = 0, 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(
            process_single_image, arg): arg for arg in worker_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            src_path, ok, err = future.result()
            if ok:
                success += 1
            else:
                fail += 1
                if fail <= 5:
                    print(f"  ⚠️ Failed: {src_path} — {err}")

    print(f"\n✅ Done! Processed {success} images ({fail} failed)")
    print(f"   Output: {DST_DIR}")


if __name__ == "__main__":
    main()
