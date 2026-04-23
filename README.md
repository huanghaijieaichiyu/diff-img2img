# Diff-Img2Img

[![Model Download](https://img.shields.io/badge/Model%20Download-Cloud-blue?style=flat-square&logo=icloud)](https://cloud.189.cn/web/share?code=AJ7fUzBbuUzm) (Access Code: `q2u9`)

Diff-Img2Img is a low-light image enhancement project built around a Retinex-guided conditional diffusion pipeline. The current training flow is preset-driven and trains directly on paired source low/high images.

## What Is In This Repo

- Retinex + diffusion training and inference entrypoint in `main.py`
- Official training presets in `configs/train/{small,middle,max}.yaml`
- Minimal train-only launcher in `start_train.py`
- Streamlit UI in `ui/app.py`

## Presets

| Preset | Target VRAM | Default Resolution | Notes |
| --- | --- | --- | --- |
| `small` | 6 GB | 256 | Compact preset for 6-8 GB GPUs with lighter conditioning and effective batch 8 |
| `middle` | 8 GB | 256 | Recommended preset with NAF decomposition and deep-only cross-attention conditioning |
| `max` | 64 GB | 512 | High-capacity preset for maximum quality runs |

All three presets are versioned YAML configs under `configs/train/` and can also be passed directly to `--config`.

## Quick Start

```bash
# 1. Environment
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
accelerate config default

# 2. Train with the recommended preset
python3 start_train.py \
  --config middle \
  --data-dir /path/to/dataset \
  --output-dir runs/middle_exp

# 3. Launch the UI studio
python3 main.py --mode ui
```

## Dataset

Training expects paired data under a dataset root such as:

```text
<data_dir>/
  our485/
    low/
    high/
  eval15/
    low/
    high/
```

Key points:

- Training consumes the existing paired source images in `our485/low` and `our485/high`.
- `train` mode does not generate additional low-light images or build a `.prepared` cache.
- Training-time augmentation uses synchronized torchvision crop/rotation/flip parameters for each low/high pair.

## Training

The runtime interface remains in `main.py`. `start_train.py` is now a **train-only** launcher that keeps the public training flow intentionally small, similar to mainstream diffusion/accelerate scripts.

```bash
python3 start_train.py \
  --config middle \
  --data-dir /path/to/dataset \
  --output-dir runs/exp
```

For non-training modes or advanced internal controls, call `main.py` directly.

The default UNet attention path now uses PyTorch's built-in SDPA processors, and compile defaults to `max-autotune-no-cudagraphs` for better training stability.

## Validation And Prediction

```bash
# Quantitative validation
python3 main.py \
  --mode validate \
  --config middle \
  --model_path runs/exp/best_model \
  --data_dir /path/to/dataset \
  --output_dir runs/exp/full_eval

# Image prediction
python3 main.py --mode predict \
  --model_path runs/exp \
  --data_dir /path/to/test_images \
  --output_dir predictions

# Video prediction
python3 main.py --mode predict \
  --model_path runs/exp \
  --video_path input.mp4 \
  --output_dir video_results
```

## Web UI

The Streamlit studio provides:

- preset-aware training launch
- live logs and metric plots from `training_metrics.csv`
- validation entrypoints
- single-image visualization on `eval15`

Start it with either of the following:

```bash
python3 main.py --mode ui
python3 run_app.py
```

Default URL: `http://localhost:8501`

## Run Summaries

Use the built-in artifact utility to summarize experiments:

```bash
python3 scripts/summarize_run.py --run-dir runs/exp
```

## Repo Layout

```text
configs/train/      preset YAMLs
core/               engine and training logic
datasets/           paired low/high dataset loader
models/             Retinex, conditioning, diffusion modules
scripts/            utilities and visualization helpers
ui/                 Streamlit app
main.py             unified CLI entrypoint
start_train.py      minimal train-only launcher
```

## Example Gallery

| Input | Output |
| :---: | :---: |
| ![Low Light](examples/fake.png) | ![Enhanced](examples/real.png) |

## License

See [LICENCE](LICENCE).
