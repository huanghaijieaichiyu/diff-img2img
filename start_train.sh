#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-}"
if [[ -z "${ACCELERATE_BIN}" ]]; then
    if [[ -x "${SCRIPT_DIR}/.venv/bin/accelerate" ]]; then
        ACCELERATE_BIN="${SCRIPT_DIR}/.venv/bin/accelerate"
    else
        ACCELERATE_BIN="accelerate"
    fi
fi

RUN_MODE="${RUN_MODE:-train}"
MODEL_SIZE="${MODEL_SIZE:-small}"
DATA_DIR="${DATA_DIR:-/mnt/f/datasets/kitti_LOL}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/retinex}"
CONFIG_PATH="${CONFIG_PATH:-configs/train/${MODEL_SIZE}.yaml}"
MODEL_PATH="${MODEL_PATH:-}"
TRAIN_PROFILE="${TRAIN_PROFILE:-auto}"
USE_RETINEX="${USE_RETINEX:-}"
PREPARED_CACHE_DIR="${PREPARED_CACHE_DIR:-}"
PREPARE_WORKERS="${PREPARE_WORKERS:-}"
OFFLINE_VARIANT_COUNT="${OFFLINE_VARIANT_COUNT:-}"
SYNTHESIS_SEED="${SYNTHESIS_SEED:-}"
PREPARE_FORCE="${PREPARE_FORCE:-0}"
RUN_FULL_EVAL_AFTER_TRAIN="${RUN_FULL_EVAL_AFTER_TRAIN:-0}"
VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-}"
FULL_EVAL_BATCH_SIZE="${FULL_EVAL_BATCH_SIZE:-2}"
FULL_EVAL_NUM_VALIDATION_IMAGES="${FULL_EVAL_NUM_VALIDATION_IMAGES:-12}"
FULL_EVAL_BENCHMARK_INFERENCE_STEPS="${FULL_EVAL_BENCHMARK_INFERENCE_STEPS:-8 20}"
FULL_EVAL_SEMANTIC_BACKBONE="${FULL_EVAL_SEMANTIC_BACKBONE:-resnet18}"
FULL_EVAL_NR_METRIC="${FULL_EVAL_NR_METRIC:-niqe}"

NUM_WORKERS="${NUM_WORKERS:-}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-}"
PIN_MEMORY="${PIN_MEMORY:-}"
DECODE_CACHE_SIZE="${DECODE_CACHE_SIZE:-}"
OPENCV_THREADS_PER_WORKER="${OPENCV_THREADS_PER_WORKER:-}"
VALIDATION_STEPS="${VALIDATION_STEPS:-}"
NUM_VALIDATION_IMAGES="${NUM_VALIDATION_IMAGES:-}"
BENCHMARK_INFERENCE_STEPS="${BENCHMARK_INFERENCE_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
SEMANTIC_BACKBONE="${SEMANTIC_BACKBONE:-}"
NR_METRIC="${NR_METRIC:-}"
MIXED_PRECISION="${MIXED_PRECISION:-}"
RESOLUTION="${RESOLUTION:-}"
EPOCHS="${EPOCHS:-}"
LR="${LR:-}"
RESUME="${RESUME:-}"
DARKER_RANGES="${DARKER_RANGES:-}"

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

normalize_bool() {
    local value="${1:-}"
    value="${value,,}"
    case "${value}" in
        1|true|yes|on) echo "true" ;;
        0|false|no|off) echo "false" ;;
        *) echo "" ;;
    esac
}

append_arg_if_set() {
    local -n target_cmd="$1"
    local var_name="$2"
    local cli_name="$3"
    if [[ -n "${!var_name:-}" ]]; then
        target_cmd+=("${cli_name}" "${!var_name}")
    fi
}

append_bool_optional_arg() {
    local -n target_cmd="$1"
    local var_name="$2"
    local cli_name="$3"
    local normalized
    normalized="$(normalize_bool "${!var_name:-}")"
    if [[ "${normalized}" == "true" ]]; then
        target_cmd+=("${cli_name}")
    elif [[ "${normalized}" == "false" ]]; then
        target_cmd+=("--no-${cli_name#--}")
    fi
}

append_list_arg_if_set() {
    local -n target_cmd="$1"
    local var_name="$2"
    local cli_name="$3"
    if [[ -n "${!var_name:-}" ]]; then
        local parts=()
        read -r -a parts <<< "${!var_name}"
        if [[ "${#parts[@]}" -gt 0 ]]; then
            target_cmd+=("${cli_name}" "${parts[@]}")
        fi
    fi
}

print_command() {
    local -a parts=("$@")
    printf '[start_train] command='
    printf '%q ' "${parts[@]}"
    printf '\n'
}

warn_if_cross_mounted_data() {
    local active_cache_dir="${PREPARED_CACHE_DIR:-${DATA_DIR}/.prepared}"
    if [[ "${DATA_DIR}" == /mnt/* || "${active_cache_dir}" == /mnt/* ]]; then
        echo "[start_train][warning] dataset or prepared cache appears to be on a cross-mounted path." >&2
        echo "[start_train][warning] For trustworthy throughput measurements, prefer a local Linux SSD." >&2
    fi
}

build_train_cmd() {
    TRAIN_CMD=(
        "${ACCELERATE_BIN}" launch main.py --mode train
        --config "${CONFIG_PATH}"
        --data_dir "${DATA_DIR}"
        --output_dir "${OUTPUT_DIR}"
        --train_profile "${TRAIN_PROFILE}"
    )
    append_arg_if_set TRAIN_CMD PREPARED_CACHE_DIR --prepared_cache_dir
    append_arg_if_set TRAIN_CMD PREPARE_WORKERS --prepare_workers
    append_arg_if_set TRAIN_CMD OFFLINE_VARIANT_COUNT --offline_variant_count
    append_arg_if_set TRAIN_CMD SYNTHESIS_SEED --synthesis_seed
    append_bool_optional_arg TRAIN_CMD USE_RETINEX --use_retinex
    append_arg_if_set TRAIN_CMD NUM_WORKERS --num_workers
    append_arg_if_set TRAIN_CMD PREFETCH_FACTOR --prefetch_factor
    append_bool_optional_arg TRAIN_CMD PERSISTENT_WORKERS --persistent_workers
    append_bool_optional_arg TRAIN_CMD PIN_MEMORY --pin_memory
    append_arg_if_set TRAIN_CMD DECODE_CACHE_SIZE --decode_cache_size
    append_arg_if_set TRAIN_CMD OPENCV_THREADS_PER_WORKER --opencv_threads_per_worker
    append_arg_if_set TRAIN_CMD VALIDATION_STEPS --validation_steps
    append_arg_if_set TRAIN_CMD NUM_VALIDATION_IMAGES --num_validation_images
    append_list_arg_if_set TRAIN_CMD BENCHMARK_INFERENCE_STEPS --benchmark_inference_steps
    append_arg_if_set TRAIN_CMD BATCH_SIZE --batch_size
    append_arg_if_set TRAIN_CMD GRADIENT_ACCUMULATION_STEPS --gradient_accumulation_steps
    append_arg_if_set TRAIN_CMD MAX_TRAIN_STEPS --max_train_steps
    append_arg_if_set TRAIN_CMD SEMANTIC_BACKBONE --semantic_backbone
    append_arg_if_set TRAIN_CMD NR_METRIC --nr_metric
    append_arg_if_set TRAIN_CMD MIXED_PRECISION --mixed_precision
    append_arg_if_set TRAIN_CMD RESOLUTION --resolution
    append_arg_if_set TRAIN_CMD EPOCHS --epochs
    append_arg_if_set TRAIN_CMD LR --lr
    append_arg_if_set TRAIN_CMD RESUME --resume
    if [[ "${PREPARE_FORCE}" == "1" || "${PREPARE_FORCE,,}" == "true" ]]; then
        TRAIN_CMD+=(--prepare_force)
    fi
    if [[ -n "${DARKER_RANGES}" ]]; then
        TRAIN_CMD+=(--darker_ranges "${DARKER_RANGES}")
    fi
}

build_validate_cmd() {
    local validation_model_path="${MODEL_PATH:-${OUTPUT_DIR}/best_model}"
    local validation_output_dir="${VALIDATION_OUTPUT_DIR:-${OUTPUT_DIR}/full_eval}"

    VALIDATE_CMD=(
        "${ACCELERATE_BIN}" launch main.py --mode validate
        --config "${CONFIG_PATH}"
        --data_dir "${DATA_DIR}"
        --model_path "${validation_model_path}"
        --output_dir "${validation_output_dir}"
        --batch_size "${FULL_EVAL_BATCH_SIZE}"
        --num_validation_images "${FULL_EVAL_NUM_VALIDATION_IMAGES}"
        --semantic_backbone "${FULL_EVAL_SEMANTIC_BACKBONE}"
        --nr_metric "${FULL_EVAL_NR_METRIC}"
    )
    append_arg_if_set VALIDATE_CMD PREPARED_CACHE_DIR --prepared_cache_dir
    append_arg_if_set VALIDATE_CMD NUM_WORKERS --num_workers
    append_arg_if_set VALIDATE_CMD PREFETCH_FACTOR --prefetch_factor
    append_bool_optional_arg VALIDATE_CMD PERSISTENT_WORKERS --persistent_workers
    append_bool_optional_arg VALIDATE_CMD PIN_MEMORY --pin_memory
    append_arg_if_set VALIDATE_CMD DECODE_CACHE_SIZE --decode_cache_size
    append_arg_if_set VALIDATE_CMD OPENCV_THREADS_PER_WORKER --opencv_threads_per_worker
    append_arg_if_set VALIDATE_CMD MIXED_PRECISION --mixed_precision
    append_bool_optional_arg VALIDATE_CMD USE_RETINEX --use_retinex
    append_list_arg_if_set VALIDATE_CMD FULL_EVAL_BENCHMARK_INFERENCE_STEPS --benchmark_inference_steps
}

warn_if_cross_mounted_data

echo "[start_train] accelerate_bin=${ACCELERATE_BIN}"
echo "[start_train] run_mode=${RUN_MODE}"
echo "[start_train] config=${CONFIG_PATH}"
echo "[start_train] data_dir=${DATA_DIR}"
echo "[start_train] output_dir=${OUTPUT_DIR}"

if [[ "${RUN_MODE}" == "validate" ]]; then
    build_validate_cmd
    print_command "${VALIDATE_CMD[@]}"
    exec "${VALIDATE_CMD[@]}"
fi

build_train_cmd
print_command "${TRAIN_CMD[@]}"

setsid "${TRAIN_CMD[@]}" &
LAUNCH_PID=$!
wait "${LAUNCH_PID}"
status=$?
trap - INT TERM

if [[ "${status}" -eq 0 ]]; then
    eval_toggle="$(normalize_bool "${RUN_FULL_EVAL_AFTER_TRAIN}")"
    if [[ "${eval_toggle}" == "true" ]]; then
        build_validate_cmd
        print_command "${VALIDATE_CMD[@]}"
        "${VALIDATE_CMD[@]}"
    fi
fi

exit "${status}"
