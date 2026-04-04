# Diff-Img2Img: Low-Light Image Enhancement with Retinex-Diffusion

[![Model Download](https://img.shields.io/badge/Model%20Download-Cloud-blue?style=flat-square&logo=icloud)](https://cloud.189.cn/web/share?code=AJ7fUzBbuUzm) (Access Code: q2u9)

A **Conditional Diffusion Model** framework for low-light image enhancement, integrating **Retinex Theory** for physically-grounded illumination decomposition. Built with PyTorch and 🤗 Diffusers.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Retinex-Diffusion Architecture** | DecomNet decomposes images into Reflectance/Illumination, conditioning the diffusion UNet for stable training |
| **Physics-Based Data Synthesis** | `Darker` engine with Poisson-Gaussian noise, multi-source headlights, vignetting, motion blur, JPEG artifacts |
| **Online Synthesis** | On-the-fly random degradation during training — each epoch sees different low-light variants |
| **Advanced Losses** | Min-SNR weighted diffusion loss + low-timestep Charbonnier/SSIM/LPIPS reconstruction branch |
| **Min-SNR Weighting** | Stabilized diffusion training via Min-SNR-γ loss weighting |
| **Web UI Studio** | Full-stack Streamlit dashboard: data prep → training → evaluation → visualization |

## 🖼️ Gallery

|          Input (Low Light)          |      Output (Enhanced)      |
| :------------------------------: | :------------------------------: |
| ![Low Light](examples/fake.png) | ![Enhanced](examples/real.png) |

## 🛠️ Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- ~6GB VRAM (batch_size=4, resolution=256)

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/yourusername/diff-img2img.git
cd diff-img2img
conda create -n diff-img2img python=3.10 && conda activate diff-img2img
pip install -r requirements.txt

# 2. Prepare dataset (synthesize low-light from normal images)
python3 scripts/darker.py

# 3. Train
accelerate launch main.py --mode train \
    --config configs/train/small.yaml \
    --data_dir "../datasets/kitti_LOL" \
    --output_dir "runs/retinex_exp" \
    --use_retinex \
    --train_profile auto

# 4. Predict
python3 main.py --mode predict \
    --model_path runs/retinex_exp \
    --data_dir ../datasets/test_images \
    --output_dir predictions \
    --use_retinex
```

## 💻 CLI Reference

### Training

```bash
accelerate launch main.py --mode train \
    --config configs/train/middle.yaml \
    --data_dir "../datasets/kitti_LOL" \
    --output_dir "runs/experiment" \
    --use_retinex \
    --train_profile auto
```

**Key training flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--train_profile` | auto | High-level training preset (`auto` or `debug_online`) |
| `--config` | `configs/train/small.yaml` | YAML model/runtime preset (`small`, `middle`, `max` or a custom path) |
| `--log_interval` | 10 | How often to refresh training summaries |
| `--use_retinex` | off | Enable Retinex decomposition conditioning |
| `--ema` / `--no-ema` | `--ema` | Enable or disable EMA |
| `--mixed_precision` | fp16 | Mixed precision policy |

**Built-in YAML presets**

| Preset | Target VRAM | Retinex Branch | Condition Adapter | Diffusion Backbone Focus | Recommended Use |
|------|---------|-------------|-------------------|--------------------------|-----------------|
| `small` | ~8GB | `SmallDecomNet`: MBConv-style lightweight decomposition | `SmallConditionAdapter`: depthwise/MBConv, lighter FiLM | Smaller UNet width, fewer attention-heavy stages | Fit-constrained training, faster iteration, baseline experiments |
| `small_accum` | ~8GB | `MiddleDecomNet` | `MiddleConditionAdapter` | Stronger-than-small network via micro-batch `1` + high grad accumulation | When 8GB needs more quality than `small` can provide |
| `middle` | ~8GB borderline / 16GB comfortable | `MiddleDecomNet`: 3-scale U-Net + one global context block | `MiddleConditionAdapter`: balanced dual-branch HVI/illumination encoder | Balanced UNet width with moderate attention | Better quality than `small`; 8GB usually requires micro-batch `1` |
| `middle_accum` | ~16GB | `MaxDecomNet` | `MaxConditionAdapter` | Wider UNet and stronger condition path using high gradient accumulation | Quality-oriented single-GPU training without jumping to `max` |
| `max` | 64GB+ | `MaxDecomNet`: deeper quality-oriented Retinex with transformer/global refinement | `MaxConditionAdapter`: stronger dual-branch encoder with transformer/global refinement | Wider/deeper UNet prioritizing quality | Quality-first runs on large-memory accelerators |
| `max_accum` | 64GB+ | `MaxDecomNet` | `MaxConditionAdapter` | Larger-than-max UNet with deeper layers and higher gradient accumulation | Upper-bound quality experiments on large-memory accelerators |

**What changes across scales**

| Component | `small` | `middle` | `max` |
|------|---------|----------|-------|
| Design goal | Parameter efficiency and trainability | Balanced quality / cost | Maximum restoration quality |
| Main building blocks | Depthwise conv + MBConv | Conv/GN + dual-branch FiLM | Transformer/global context + gated fusion |
| Condition space | Learnable HVI-lite | Learnable HVI-lite | Learnable HVI-lite |
| Retinex context range | Local to medium | Medium to global | Strong global modeling |
| Typical config file | `configs/train/small.yaml` / `small_accum.yaml` | `configs/train/middle.yaml` / `middle_accum.yaml` | `configs/train/max.yaml` / `max_accum.yaml` |

**YAML layout**

```yaml
meta:
runtime:
model:
optimization:
loss:
schedule:
evaluation:
```

The `model` section controls architecture family and width/depth, while `optimization / loss / schedule` provide scale-appropriate defaults. CLI flags can still override YAML values when needed.

> Note: gradient clipping improves optimization stability, but it does **not** materially reduce activation memory. If you need an 8GB-friendly run, memory fit mainly comes from model width/depth, resolution, and micro-batch size.

### Prediction

```bash
# Images
python3 main.py --mode predict \
    --model_path runs/experiment \
    --data_dir test_images/ \
    --output_dir results/ \
    --use_retinex

# Video
python3 main.py --mode predict \
    --model_path runs/experiment \
    --video_path input.mp4 \
    --output_dir video_results/ \
    --use_retinex
```

### Validation

```bash
python3 main.py --mode validate \
    --model_path runs/experiment \
    --data_dir "../datasets/kitti_LOL" \
    --use_retinex
# Metrics saved to: runs/experiment/metrics.txt (PSNR, SSIM, LPIPS)
```

### Web UI

```bash
python3 main.py --mode ui
# Or directly: streamlit run ui/app.py
```

## 📂 Project Structure

```
diff-img2img/
├── core/
│   └── engine.py           # DiffusionEngine: train, validate, predict
├── models/
│   ├── common.py           # Conv, C2f, ConvTranspose building blocks
│   ├── diffusion.py        # CombinedModel (UNet + DecomNet wrapper)
│   └── retinex.py          # DecomNet for Retinex decomposition
├── datasets/
│   └── data_set.py         # LowLightDataset (supports profile-driven data loading)
├── scripts/
│   ├── darker.py           # Physics-based degradation engine
│   └── visual_val.py       # Visualization & inference helpers
├── utils/
│   ├── loss.py             # Charbonnier + SSIM + LPIPS losses
│   └── misc.py             # SSIM, SNR helpers, seed utils
├── ui/
│   └── app.py              # Streamlit Web UI
├── notebooks/
│   └── train_test_notebook.ipynb  # Interactive tutorial
├── main.py                 # Unified CLI entry point
├── start_train.sh          # Training launch script
└── accelerate_config.yaml  # HF Accelerate config
```

## 🔬 Architecture

```
Input (Low-Light) ──┐
                    ├─→ DecomNet ──→ Reflectance (R) + Illumination (I)
                    │                      │
                    │           ┌───────────┤
                    │           ▼           ▼
                    │    ┌──────────────────────┐
 Noise (z_t) ──────┼──→│  Conditional UNet2D   │──→ Predicted noise/velocity
                    │    │  (concat: z_t, R, I) │
                    │    └──────────────────────┘
                    │
                    └─→ Composite Loss (Charbonnier + SSIM + LPIPS)
                        Retinex Loss (Recon + Reflectance + TV)
                        Diffusion Loss (Min-SNR weighted)
```

## 📊 Dataset Format

```
your_dataset/
├── our485/          # Training split (485 pairs)
│   ├── high/        # Normal-light ground truth
│   └── low/         # Low-light images (can be auto-generated)
└── eval15/          # Test split (15 pairs)
    ├── high/
    └── low/
```

> **Tip**: `--train_profile auto` prefers precomputed low/high pairs. Use `--train_profile debug_online` when you explicitly want online degradation experiments.

## 📄 License

[MIT License](LICENSE)

## 🤝 Contact

For issues, please submit a GitHub Issue or contact: huangxiaohai99@126.com
