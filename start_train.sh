#!/usr/bin/env bash

set -euo pipefail

TRAIN_PROFILE="${TRAIN_PROFILE:-auto}"
MODEL_SIZE="${MODEL_SIZE:-small_sota}"
DATA_DIR="${DATA_DIR:-/mnt/f/datasets/kitti_LOL}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/retinex}"
CONFIG_PATH="${CONFIG_PATH:-configs/train/${MODEL_SIZE}.yaml}"

accelerate launch main.py --mode train \
    --config "${CONFIG_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --use_retinex \
    --train_profile "${TRAIN_PROFILE}"
