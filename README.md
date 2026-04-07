# Diff-Img2Img: Low-Light Image Enhancement with Retinex-Diffusion

[![Model Download](https://img.shields.io/badge/Model%20Download-Cloud-blue?style=flat-square&logo=icloud)](https://cloud.189.cn/web/share?code=AJ7fUzBbuUzm) (Access Code: q2u9)

A **Conditional Diffusion Model** framework for low-light image enhancement, integrating **Retinex Theory** for physically-grounded illumination decomposition. Built with PyTorch and 🤗 Diffusers.

## 🆕 Latest Updates (2026-04-06)

**SOTA Improvements Implemented:**
- ✅ P0 fixes: Smooth Retinex loss ramp + corrected EMA timing
- ✅ P1 enhancements: Charbonnier diffusion loss + cosine warmup + adaptive EMA
- ✅ Architecture: NAFNet decomposition + Cross-Attention conditioning
- ✅ Training engine: Min-SNR / P2 / EDM weighting support + optional uncertainty-based loss balancing

**Expected improvements:** +30% stability, +15% convergence speed, +0.5-1.0 dB PSNR

📚 See [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) for details.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Retinex-Diffusion Architecture** | DecomNet decomposes images into Reflectance/Illumination, conditioning the diffusion UNet |
| **SOTA Training** | Preset-driven training with cosine warmup, adaptive EMA, smooth Retinex ramps |
| **Physics-Based Synthesis** | Poisson-Gaussian noise, headlights, vignetting, motion blur, JPEG artifacts |
| **Prepared Multi-Variant Cache** | Training auto-builds a 3-variant offline low-light cache from `our485/high` when prepared data is missing |
| **Advanced Losses** | Charbonnier + SSIM + LPIPS with uncertainty weighting |
| **Web UI Studio** | Streamlit dashboard for training, evaluation, and visualization |

## 🖼️ Gallery

|          Input (Low Light)          |      Output (Enhanced)      |
| :------------------------------: | :------------------------------: |
| ![Low Light](examples/fake.png) | ![Enhanced](examples/real.png) |

## 🛠️ Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- 8GB VRAM recommended for full training (`middle`)
- 6GB VRAM is enough for the compact official preset (`small`)

## 🚀 Quick Start

```bash
# 1. Setup
git clone https://github.com/yourusername/diff-img2img.git
cd diff-img2img
conda create -n diff-img2img python=3.10 && conda activate diff-img2img
pip install -r requirements.txt
accelerate config default

# 2. Train with the recommended 8GB preset
MODEL_SIZE=middle \
DATA_DIR=/path/to/dataset \
OUTPUT_DIR=runs/middle_exp \
TRAIN_PROFILE=auto \
bash start_train.sh

# 3. Predict
python main.py --mode predict \
    --model_path runs/middle_exp \
    --data_dir test_images/ \
    --output_dir predictions
```

## 📋 Configuration Presets

| Config | VRAM | Features | Use Case |
|--------|------|----------|----------|
| `small` | ~6GB | Compact NAF + deep-only cross-attention preset | Best fit for 6-8GB GPUs |
| `middle` | ~8GB | NAF decomposition + stronger conditioner + effective batch 8 | **Recommended full-training preset** |
| `max` | 64GB+ | Transformer refinement, global context | Maximum quality |

## 💻 Training

```bash
# Recommended wrapper (uses accelerate launch internally)
MODEL_SIZE=middle \
DATA_DIR=/path/to/dataset \
OUTPUT_DIR=runs/exp \
TRAIN_PROFILE=auto \
bash start_train.sh

# Equivalent raw command
accelerate launch main.py --mode train \
    --config configs/train/middle.yaml \
    --data_dir /path/to/dataset \
    --output_dir runs/exp \
    --train_profile auto

# Build or rebuild the prepared multi-variant cache explicitly
python main.py --mode prepare \
    --config configs/train/middle.yaml \
    --data_dir /path/to/dataset \
    --offline_variant_count 3
```

**Key flags:**
- `--config`: Model preset or YAML path. Recommended full run: `configs/train/middle.yaml`
- `--train_profile`: `auto` for normal training
- `start_train.sh`: convenience entrypoint that reads `MODEL_SIZE`, `DATA_DIR`, `OUTPUT_DIR`, and `TRAIN_PROFILE`
- training now only consumes prepared multi-variant low-light data; single-variant `our485/low` is not treated as ready-to-train
- if `.prepared/train_manifest.jsonl` or `.prepared/prepare_meta.json` is missing/stale, training first regenerates a 3-variant offline cache from `our485/high`
- the prepare step now prints scan/build progress, supports interruption, and resumes from missing variants on the next run when metadata still matches
- `--mode prepare`: run the same offline cache builder without starting training
- `--offline_variant_count`: target number of prepared low-light variants per training image (default: `3`)
- `--prepare_force`: force rebuilding the prepared cache
- `--use_retinex`: enabled by the training presets already; keep it on unless you are doing ablations
- `--ema` / `--no-ema`: EMA model (default: on with adaptive decay)
- `--mixed_precision`: fp16/bf16/no (default: fp16)

## 🔬 Validation & Prediction

```bash
# Validate
python main.py --mode validate \
    --model_path runs/exp \
    --data_dir /path/to/dataset

# Predict images
python main.py --mode predict \
    --model_path runs/exp \
    --data_dir test_images/ \
    --output_dir results/

# Predict video
python main.py --mode predict \
    --model_path runs/exp \
    --video_path input.mp4 \
    --output_dir video_results/
```

## 🎨 Web UI

```bash
python main.py --mode ui
# Opens Streamlit dashboard at http://localhost:8501
```

## 📚 Documentation

- [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) - Quick overview of SOTA improvements
- [P0_P1_FIXES_REPORT.md](P0_P1_FIXES_REPORT.md) - Detailed P0/P1 fixes
- [SOTA_IMPROVEMENTS_REPORT.md](SOTA_IMPROVEMENTS_REPORT.md) - Complete SOTA report
- [TRAINING_LOGIC_REVIEW.md](TRAINING_LOGIC_REVIEW.md) - Training logic analysis
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [notebooks/train_test_notebook.ipynb](notebooks/train_test_notebook.ipynb) - End-to-end notebook walkthrough for dataset prep, training, inference, and evaluation

## 🏗️ Architecture

```
Input (Low-Light) 
    ↓
Retinex Decomposition (NAFNet/U-Net)
    ├─ Reflectance (R)
    └─ Illumination (I)
    ↓
Condition Adapter (FiLM + Cross-Attention)
    ↓
Diffusion UNet (v-prediction)
    ↓
Enhanced Output
```

## 📊 Training Details

**Loss Functions:**
- Diffusion: Charbonnier + EDM weighting
- X0 reconstruction: Charbonnier + SSIM + LPIPS (uncertainty weighted)
- Retinex: Reconstruction + consistency + exposure + TV

**Optimization:**
- Optimizer: AdamW (lr=1e-4, betas=(0.9, 0.999))
- Scheduler: Cosine annealing with warmup
- EMA: Adaptive decay (0.95 → 0.9999)
- Gradient clipping: 4.0

**Training Stages:**
1. Retinex warmup (optional): Train decomposition network
2. Joint training: Train diffusion + decomposition together

## 🎯 Performance

Expected improvements with SOTA configurations:
- Training stability: +30%
- Convergence speed: +15%
- Final PSNR: +0.5-1.0 dB vs baseline

## 📝 Citation

```bibtex
@article{diff-img2img,
  title={Diff-Img2Img: Low-Light Image Enhancement with Retinex-Diffusion},
  author={Your Name},
  year={2026}
}
```

## 📄 License

See [LICENCE](LICENCE) file.

## 🙏 Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/)
- [🤗 Diffusers](https://github.com/huggingface/diffusers)
- [Accelerate](https://github.com/huggingface/accelerate)
- [Streamlit](https://streamlit.io/)

SOTA improvements inspired by:
- EDM (Karras et al., 2022)
- NAFNet (Chen et al., 2022)
- Min-SNR (Hang et al., 2023)
- Stable Diffusion (Rombach et al., 2022)
