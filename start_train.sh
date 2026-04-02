#!/usr/bin/env bash

set -euo pipefail

TRAIN_PROFILE="${TRAIN_PROFILE:-auto}"
DATA_DIR="${DATA_DIR:-/mnt/f/datasets/kitti_LOL}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/retinex}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RESOLUTION="${RESOLUTION:-256}"

accelerate launch main.py --mode train \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --resolution "${RESOLUTION}" \
    --use_retinex \
    --train_profile "${TRAIN_PROFILE}"
