# Diff-Img2Img: Low-Light Image Enhancement with Retinex-Diffusion

[![Model Download](https://img.shields.io/badge/Model%20Download-Cloud-blue?style=flat-square&logo=icloud)](https://cloud.189.cn/web/share?code=AJ7fUzBbuUzm) (Access Code: q2u9)

A **Conditional Diffusion Model** framework for low-light image enhancement, integrating **Retinex Theory** for physically-grounded illumination decomposition. Built with PyTorch and 🤗 Diffusers.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Retinex-Diffusion Architecture** | DecomNet decomposes images into Reflectance/Illumination, conditioning the diffusion UNet for stable training |
| **Physics-Based Data Synthesis** | `Darker` engine with Poisson-Gaussian noise, multi-source headlights, vignetting, motion blur, JPEG artifacts |
| **Online Synthesis** | On-the-fly random degradation during training — each epoch sees different low-light variants |
| **Advanced Losses** | Charbonnier (pixel) + SSIM (structure) + **LPIPS** (perceptual) composite loss |
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
python scripts/darker.py

# 3. Train
accelerate launch main.py --mode train \
    --data_dir "../datasets/kitti_LOL" \
    --output_dir "runs/retinex_exp" \
    --use_retinex \
    --online_synthesis \
    --epochs 100 \
    --batch_size 4 \
    --resolution 256

# 4. Predict
python main.py --mode predict \
    --model_path runs/retinex_exp \
    --data_dir ../datasets/test_images \
    --output_dir predictions \
    --use_retinex
```

## 💻 CLI Reference

### Training

```bash
accelerate launch main.py --mode train \
    --data_dir "../datasets/kitti_LOL" \
    --output_dir "runs/experiment" \
    --use_retinex \
    --online_synthesis \
    --freeze_decom_steps 2000 \
    --epochs 100 \
    --batch_size 4 \
    --resolution 256 \
    --mixed_precision fp16 \
    --prediction_type v_prediction \
    --gradient_accumulation_steps 4
```

**Key training flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--use_retinex` | off | Enable Retinex decomposition conditioning |
| `--online_synthesis` | off | On-the-fly random degradation (recommended) |
| `--freeze_decom_steps` | 0 | Freeze DecomNet for first N steps |
| `--prediction_type` | v_prediction | Diffusion target type |
| `--offset_noise` | off | Enable offset noise for better dark regions |
| `--checkpoints_total_limit` | None | Auto-delete old checkpoints (e.g., 5) |

### Prediction

```bash
# Images
python main.py --mode predict \
    --model_path runs/experiment \
    --data_dir test_images/ \
    --output_dir results/ \
    --use_retinex

# Video
python main.py --mode predict \
    --model_path runs/experiment \
    --video_path input.mp4 \
    --output_dir video_results/ \
    --use_retinex
```

### Validation

```bash
python main.py --mode validate \
    --model_path runs/experiment \
    --data_dir "../datasets/kitti_LOL" \
    --use_retinex
# Metrics saved to: runs/experiment/metrics.txt (PSNR, SSIM, LPIPS)
```

### Web UI

```bash
python main.py --mode ui
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
│   └── data_set.py         # LowLightDataset (supports online_synthesis)
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

> **Tip**: With `--online_synthesis`, you only need the `high/` directory — low-light images are synthesized on-the-fly.

## 📄 License

[MIT License](LICENSE)

## 🤝 Contact

For issues, please submit a GitHub Issue or contact: huangxiaohai99@126.com
