# Diff-Img2Img

[![Model Download](https://img.shields.io/badge/Model%20Download-Cloud-blue?style=flat-square&logo=icloud)](https://cloud.189.cn/web/share?code=AJ7fUzBbuUzm) (Access Code: `q2u9`)

Diff-Img2Img is a low-light image enhancement project built around a Retinex-guided conditional diffusion pipeline. The current training flow is preset-driven, uses a prepared offline low-light cache, and ships with a Streamlit studio for dataset preparation, training monitoring, evaluation, and qualitative visualization.

## What Is In This Repo

- Retinex + diffusion training and inference entrypoint in `main.py`
- Official training presets in `configs/train/{small,middle,max}.yaml`
- Offline prepared-cache builder that regenerates low-light variants from `our485/high`
- Streamlit UI in `ui/app.py`
- Convenience launcher in `start_train.sh` backed by `utils/train_launcher.py`

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
MODEL_SIZE=middle \
DATA_DIR=/path/to/dataset \
OUTPUT_DIR=runs/middle_exp \
TRAIN_PROFILE=auto \
bash start_train.sh

# 3. Launch the UI studio
python3 main.py --mode ui
```

## Dataset And Prepared Cache

Training expects paired data under a dataset root such as:

```text
<data_dir>/
  our485/
    high/
  eval15/
    low/
    high/
```

Before the first epoch, training validates or builds a prepared cache under:

```text
<data_dir>/.prepared/
  prepare_meta.json
  train_manifest.jsonl
  our485/low/...
```

Key points:

- Training consumes the prepared cache, not a single pre-existing `our485/low` folder.
- If the cache is missing or stale, `train` will rebuild it automatically.
- Interrupted prepare runs can resume as long as the metadata still matches the requested settings.

To prepare the cache explicitly:

```bash
python3 main.py --mode prepare \
  --config configs/train/middle.yaml \
  --data_dir /path/to/dataset \
  --offline_variant_count 3 \
  --prepare_workers 4 \
  --synthesis_seed 42
```

## Training

The recommended way to launch training is the wrapper script:

```bash
MODEL_SIZE=middle \
DATA_DIR=/path/to/dataset \
OUTPUT_DIR=runs/exp \
TRAIN_PROFILE=auto \
PREPARE_WORKERS=4 \
OFFLINE_VARIANT_COUNT=3 \
SYNTHESIS_SEED=42 \
bash start_train.sh
```

Environment variables understood by `start_train.sh`:

- `MODEL_SIZE`: `small`, `middle`, or `max`
- `DATA_DIR`: dataset root
- `OUTPUT_DIR`: run directory
- `TRAIN_PROFILE`: `auto` or `debug_online`
- `CONFIG_PATH`: optional explicit YAML path override
- `RUN_MODE`: `train` or `validate`
- `MODEL_PATH`: checkpoint directory used when `RUN_MODE=validate`
- `PREPARED_CACHE_DIR`: optional cache override, defaults to `<DATA_DIR>/.prepared`
- `PREPARE_WORKERS`: worker count for offline cache generation
- `OFFLINE_VARIANT_COUNT`: number of synthetic low-light variants per training image
- `SYNTHESIS_SEED`: base seed for offline synthesis
- `PREPARE_FORCE`: set to `1` to rebuild the cache from scratch
- `RUN_FULL_EVAL_AFTER_TRAIN`: set to `1` to launch the standardized offline validation right after a successful train run
- Explicit runtime overrides such as `NUM_WORKERS`, `PREFETCH_FACTOR`, `BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `VALIDATION_STEPS`, `BENCHMARK_INFERENCE_STEPS`, `SEMANTIC_BACKBONE`, and `NR_METRIC`

`start_train.sh` is now a thin shell wrapper. The detailed environment-variable parsing and command assembly live in `utils/train_launcher.py`, while the resolved defaults still come from `configs/train/*.yaml` and `main.py`.

Equivalent raw command:

```bash
python3 -m accelerate.commands.launch main.py --mode train \
  --config configs/train/middle.yaml \
  --data_dir /path/to/dataset \
  --output_dir runs/exp \
  --train_profile auto
```

## Validation And Prediction

```bash
# Quantitative validation
python3 main.py --mode validate \
  --model_path runs/exp \
  --data_dir /path/to/dataset \
  --output_dir runs/exp_eval

# Equivalent wrapper-based validation
RUN_MODE=validate \
MODEL_SIZE=middle \
DATA_DIR=/path/to/dataset \
OUTPUT_DIR=runs/exp \
MODEL_PATH=runs/exp/best_model \
bash start_train.sh

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

- prepared-cache inspection and manual rebuild
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

Use the built-in artifact utilities to summarize and compare experiments:

```bash
python3 scripts/summarize_run.py --run-dir runs/exp
python3 scripts/compare_runs.py --run-dir runs/exp_a --run-dir runs/exp_b --sort-by psnr
```

## Repo Layout

```text
configs/train/      preset YAMLs
core/               engine and training logic
datasets/           data preparation helpers
models/             Retinex, conditioning, diffusion modules
scripts/            utilities and visualization helpers
ui/                 Streamlit app
main.py             unified CLI entrypoint
start_train.sh      thin shell wrapper for training
utils/train_launcher.py
```

## Example Gallery

| Input | Output |
| :---: | :---: |
| ![Low Light](examples/fake.png) | ![Enhanced](examples/real.png) |

## License

See [LICENCE](LICENCE).
