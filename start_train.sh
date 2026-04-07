#!/usr/bin/env bash

set -euo pipefail

TRAIN_PROFILE="${TRAIN_PROFILE:-auto}"
MODEL_SIZE="${MODEL_SIZE:-small}"
DATA_DIR="${DATA_DIR:-/mnt/f/datasets/kitti_LOL}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/retinex}"
CONFIG_PATH="${CONFIG_PATH:-configs/train/${MODEL_SIZE}.yaml}"
PREPARED_CACHE_DIR="${PREPARED_CACHE_DIR:-}"
PREPARE_WORKERS="${PREPARE_WORKERS:-}"
OFFLINE_VARIANT_COUNT="${OFFLINE_VARIANT_COUNT:-}"
SYNTHESIS_SEED="${SYNTHESIS_SEED:-}"
PREPARE_FORCE="${PREPARE_FORCE:-0}"

cleanup() {
    local exit_code="${1:-130}"
    if [[ -n "${LAUNCH_PID:-}" ]]; then
        echo
        echo "[start_train] interrupt received, stopping launcher..." >&2
        kill -TERM -- "-${LAUNCH_PID}" 2>/dev/null || kill -TERM "${LAUNCH_PID}" 2>/dev/null || true
        wait "${LAUNCH_PID}" 2>/dev/null || true
    fi
    exit "${exit_code}"
}

trap 'cleanup 130' INT TERM

echo "[start_train] config=${CONFIG_PATH}"
echo "[start_train] data_dir=${DATA_DIR}"
echo "[start_train] output_dir=${OUTPUT_DIR}"
echo "[start_train] train_profile=${TRAIN_PROFILE}"
echo "[start_train] prepared_cache_dir=${PREPARED_CACHE_DIR:-${DATA_DIR}/.prepared}"
echo "[start_train] prepare_workers=${PREPARE_WORKERS:-auto}"
echo "[start_train] offline_variant_count=${OFFLINE_VARIANT_COUNT:-3}"
echo "[start_train] synthesis_seed=${SYNTHESIS_SEED:-42}"
echo "[start_train] prepare_force=${PREPARE_FORCE}"

cmd=(
    accelerate launch main.py --mode train
    --config "${CONFIG_PATH}"
    --data_dir "${DATA_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --use_retinex
    --train_profile "${TRAIN_PROFILE}"
)

if [[ -n "${PREPARED_CACHE_DIR}" ]]; then
    cmd+=(--prepared_cache_dir "${PREPARED_CACHE_DIR}")
fi
if [[ -n "${PREPARE_WORKERS}" ]]; then
    cmd+=(--prepare_workers "${PREPARE_WORKERS}")
fi
if [[ -n "${OFFLINE_VARIANT_COUNT}" ]]; then
    cmd+=(--offline_variant_count "${OFFLINE_VARIANT_COUNT}")
fi
if [[ -n "${SYNTHESIS_SEED}" ]]; then
    cmd+=(--synthesis_seed "${SYNTHESIS_SEED}")
fi
if [[ "${PREPARE_FORCE}" == "1" || "${PREPARE_FORCE}" == "true" || "${PREPARE_FORCE}" == "TRUE" ]]; then
    cmd+=(--prepare_force)
fi

setsid "${cmd[@]}" &
LAUNCH_PID=$!
wait "${LAUNCH_PID}"
status=$?
trap - INT TERM
exit "${status}"
