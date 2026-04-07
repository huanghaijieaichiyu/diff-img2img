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
TRAIN_PROFILE="${TRAIN_PROFILE:-auto}"
PERF_PROFILE="${PERF_PROFILE:-auto_8gb}"
MODEL_SIZE="${MODEL_SIZE:-small}"
DATA_DIR="${DATA_DIR:-/mnt/f/datasets/kitti_LOL}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/retinex}"
CONFIG_PATH="${CONFIG_PATH:-configs/train/${MODEL_SIZE}.yaml}"
MODEL_PATH="${MODEL_PATH:-}"
PREPARED_CACHE_DIR="${PREPARED_CACHE_DIR:-}"
PREPARE_WORKERS="${PREPARE_WORKERS:-}"
OFFLINE_VARIANT_COUNT="${OFFLINE_VARIANT_COUNT:-}"
SYNTHESIS_SEED="${SYNTHESIS_SEED:-}"
PREPARE_FORCE="${PREPARE_FORCE:-0}"
RUN_FULL_EVAL_AFTER_TRAIN="${RUN_FULL_EVAL_AFTER_TRAIN:-0}"
VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-}"
FULL_EVAL_BATCH_SIZE="${FULL_EVAL_BATCH_SIZE:-}"
FULL_EVAL_NUM_VALIDATION_IMAGES="${FULL_EVAL_NUM_VALIDATION_IMAGES:-}"
FULL_EVAL_BENCHMARK_INFERENCE_STEPS="${FULL_EVAL_BENCHMARK_INFERENCE_STEPS:-}"
FULL_EVAL_SEMANTIC_BACKBONE="${FULL_EVAL_SEMANTIC_BACKBONE:-}"
FULL_EVAL_NR_METRIC="${FULL_EVAL_NR_METRIC:-}"

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

set_default_if_empty() {
    local var_name="$1"
    local default_value="$2"
    if [[ -z "${!var_name:-}" ]]; then
        printf -v "${var_name}" '%s' "${default_value}"
    fi
}

apply_phase_a_defaults() {
    set_default_if_empty NUM_WORKERS "6"
    set_default_if_empty PREFETCH_FACTOR "4"
    set_default_if_empty PERSISTENT_WORKERS "true"
    set_default_if_empty PIN_MEMORY "true"
    set_default_if_empty DECODE_CACHE_SIZE "48"
    set_default_if_empty OPENCV_THREADS_PER_WORKER "1"
    set_default_if_empty PREPARE_WORKERS "8"
    set_default_if_empty VALIDATION_STEPS "1000"
    set_default_if_empty NUM_VALIDATION_IMAGES "6"
    set_default_if_empty BENCHMARK_INFERENCE_STEPS "8"
    set_default_if_empty SEMANTIC_BACKBONE "none"
    set_default_if_empty NR_METRIC "none"
}

apply_phase_c_cpu_free_defaults() {
    apply_phase_a_defaults
    set_default_if_empty NUM_WORKERS "8"
}

apply_phase_c_cpu_bound_defaults() {
    apply_phase_a_defaults
    set_default_if_empty NUM_WORKERS "6"
    set_default_if_empty PREFETCH_FACTOR "2"
    set_default_if_empty DECODE_CACHE_SIZE "32"
}

apply_perf_profile_defaults() {
    case "${PERF_PROFILE}" in
        baseline)
            ;;
        auto_8gb)
            case "${MODEL_SIZE}" in
                middle) apply_phase_a_defaults ;;
                small)
                    apply_phase_a_defaults
                    set_default_if_empty BATCH_SIZE "4"
                    set_default_if_empty GRADIENT_ACCUMULATION_STEPS "4"
                    ;;
            esac
            ;;
        phase_a|middle_phase_a)
            apply_phase_a_defaults
            ;;
        small_phase_b|small_fast)
            apply_phase_a_defaults
            set_default_if_empty BATCH_SIZE "4"
            set_default_if_empty GRADIENT_ACCUMULATION_STEPS "4"
            ;;
        small_phase_b_safe|small_safe)
            apply_phase_a_defaults
            set_default_if_empty BATCH_SIZE "2"
            set_default_if_empty GRADIENT_ACCUMULATION_STEPS "8"
            ;;
        phase_c_cpu_free)
            apply_phase_c_cpu_free_defaults
            ;;
        phase_c_cpu_bound)
            apply_phase_c_cpu_bound_defaults
            ;;
        *)
            echo "[start_train] unknown PERF_PROFILE=${PERF_PROFILE}" >&2
            exit 2
            ;;
    esac
}

append_arg_if_set() {
    local var_name="$1"
    local cli_name="$2"
    if [[ -n "${!var_name:-}" ]]; then
        cmd+=("${cli_name}" "${!var_name}")
    fi
}

append_bool_optional_arg() {
    local var_name="$1"
    local cli_name="$2"
    local normalized
    normalized="$(normalize_bool "${!var_name:-}")"
    if [[ "${normalized}" == "true" ]]; then
        cmd+=("${cli_name}")
    elif [[ "${normalized}" == "false" ]]; then
        cmd+=("--no-${cli_name#--}")
    fi
}

append_list_arg_if_set() {
    local var_name="$1"
    local cli_name="$2"
    if [[ -n "${!var_name:-}" ]]; then
        local parts=()
        read -r -a parts <<< "${!var_name}"
        if [[ "${#parts[@]}" -gt 0 ]]; then
            cmd+=("${cli_name}" "${parts[@]}")
        fi
    fi
}

warn_if_cross_mounted_data() {
    local active_cache_dir="${PREPARED_CACHE_DIR:-${DATA_DIR}/.prepared}"
    if [[ "${DATA_DIR}" == /mnt/* || "${active_cache_dir}" == /mnt/* ]]; then
        echo "[start_train][warning] dataset or prepared cache appears to be on a cross-mounted path." >&2
        echo "[start_train][warning] For trustworthy throughput measurements, prefer a local Linux SSD." >&2
    fi
}

build_validate_cmd() {
    local validation_model_path="${MODEL_PATH:-${OUTPUT_DIR}/best_model}"
    local validation_output_dir="${VALIDATION_OUTPUT_DIR:-${OUTPUT_DIR}/full_eval}"
    local validation_benchmark_steps="${FULL_EVAL_BENCHMARK_INFERENCE_STEPS:-8 20}"
    local validation_num_images="${FULL_EVAL_NUM_VALIDATION_IMAGES:-12}"
    local validation_semantic="${FULL_EVAL_SEMANTIC_BACKBONE:-resnet18}"
    local validation_nr_metric="${FULL_EVAL_NR_METRIC:-niqe}"
    local validation_batch_size="${FULL_EVAL_BATCH_SIZE:-2}"
    local validation_mixed_precision="${MIXED_PRECISION:-fp16}"

    validate_cmd=(
        "${ACCELERATE_BIN}" launch main.py --mode validate
        --config "${CONFIG_PATH}"
        --data_dir "${DATA_DIR}"
        --model_path "${validation_model_path}"
        --output_dir "${validation_output_dir}"
        --batch_size "${validation_batch_size}"
        --mixed_precision "${validation_mixed_precision}"
        --num_validation_images "${validation_num_images}"
        --semantic_backbone "${validation_semantic}"
        --nr_metric "${validation_nr_metric}"
    )

    if [[ -n "${PREPARED_CACHE_DIR}" ]]; then
        validate_cmd+=(--prepared_cache_dir "${PREPARED_CACHE_DIR}")
    fi
    if [[ -n "${NUM_WORKERS}" ]]; then
        validate_cmd+=(--num_workers "${NUM_WORKERS}")
    fi
    if [[ -n "${PREFETCH_FACTOR}" ]]; then
        validate_cmd+=(--prefetch_factor "${PREFETCH_FACTOR}")
    fi
    append_bool_optional_arg_for_validate PERSISTENT_WORKERS --persistent_workers
    append_bool_optional_arg_for_validate PIN_MEMORY --pin_memory
    if [[ -n "${DECODE_CACHE_SIZE}" ]]; then
        validate_cmd+=(--decode_cache_size "${DECODE_CACHE_SIZE}")
    fi
    if [[ -n "${OPENCV_THREADS_PER_WORKER}" ]]; then
        validate_cmd+=(--opencv_threads_per_worker "${OPENCV_THREADS_PER_WORKER}")
    fi
    if [[ -n "${validation_benchmark_steps}" ]]; then
        local bench_parts=()
        read -r -a bench_parts <<< "${validation_benchmark_steps}"
        validate_cmd+=(--benchmark_inference_steps "${bench_parts[@]}")
    fi
}

append_bool_optional_arg_for_validate() {
    local var_name="$1"
    local cli_name="$2"
    local normalized
    normalized="$(normalize_bool "${!var_name:-}")"
    if [[ "${normalized}" == "true" ]]; then
        validate_cmd+=("${cli_name}")
    elif [[ "${normalized}" == "false" ]]; then
        validate_cmd+=("--no-${cli_name#--}")
    fi
}

apply_perf_profile_defaults
warn_if_cross_mounted_data

echo "[start_train] config=${CONFIG_PATH}"
echo "[start_train] accelerate_bin=${ACCELERATE_BIN}"
echo "[start_train] run_mode=${RUN_MODE}"
echo "[start_train] perf_profile=${PERF_PROFILE}"
echo "[start_train] data_dir=${DATA_DIR}"
echo "[start_train] output_dir=${OUTPUT_DIR}"
echo "[start_train] train_profile=${TRAIN_PROFILE}"
echo "[start_train] prepared_cache_dir=${PREPARED_CACHE_DIR:-${DATA_DIR}/.prepared}"
echo "[start_train] prepare_workers=${PREPARE_WORKERS:-auto}"
echo "[start_train] offline_variant_count=${OFFLINE_VARIANT_COUNT:-3}"
echo "[start_train] synthesis_seed=${SYNTHESIS_SEED:-42}"
echo "[start_train] prepare_force=${PREPARE_FORCE}"
echo "[start_train] num_workers=${NUM_WORKERS:-config/default}"
echo "[start_train] prefetch_factor=${PREFETCH_FACTOR:-config/default}"
echo "[start_train] persistent_workers=${PERSISTENT_WORKERS:-config/default}"
echo "[start_train] pin_memory=${PIN_MEMORY:-config/default}"
echo "[start_train] decode_cache_size=${DECODE_CACHE_SIZE:-config/default}"
echo "[start_train] opencv_threads_per_worker=${OPENCV_THREADS_PER_WORKER:-config/default}"
echo "[start_train] validation_steps=${VALIDATION_STEPS:-config/default}"
echo "[start_train] num_validation_images=${NUM_VALIDATION_IMAGES:-config/default}"
echo "[start_train] benchmark_inference_steps=${BENCHMARK_INFERENCE_STEPS:-config/default}"
echo "[start_train] batch_size=${BATCH_SIZE:-config/default}"
echo "[start_train] gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-config/default}"
echo "[start_train] max_train_steps=${MAX_TRAIN_STEPS:-config/default}"
echo "[start_train] semantic_backbone=${SEMANTIC_BACKBONE:-config/default}"
echo "[start_train] nr_metric=${NR_METRIC:-config/default}"
echo "[start_train] full_eval_batch_size=${FULL_EVAL_BATCH_SIZE:-2}"
echo "[start_train] full_eval_num_validation_images=${FULL_EVAL_NUM_VALIDATION_IMAGES:-12}"
echo "[start_train] full_eval_benchmark_inference_steps=${FULL_EVAL_BENCHMARK_INFERENCE_STEPS:-8 20}"
echo "[start_train] full_eval_semantic_backbone=${FULL_EVAL_SEMANTIC_BACKBONE:-resnet18}"
echo "[start_train] full_eval_nr_metric=${FULL_EVAL_NR_METRIC:-niqe}"
echo "[start_train] run_full_eval_after_train=${RUN_FULL_EVAL_AFTER_TRAIN}"

if [[ "${RUN_MODE}" == "validate" ]]; then
    build_validate_cmd
    echo "[start_train] validate_model_path=${MODEL_PATH:-${OUTPUT_DIR}/best_model}"
    echo "[start_train] validation_output_dir=${VALIDATION_OUTPUT_DIR:-${OUTPUT_DIR}/full_eval}"
    exec "${validate_cmd[@]}"
fi

cmd=(
    "${ACCELERATE_BIN}" launch main.py --mode train
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
append_arg_if_set NUM_WORKERS --num_workers
append_arg_if_set PREFETCH_FACTOR --prefetch_factor
append_bool_optional_arg PERSISTENT_WORKERS --persistent_workers
append_bool_optional_arg PIN_MEMORY --pin_memory
append_arg_if_set DECODE_CACHE_SIZE --decode_cache_size
append_arg_if_set OPENCV_THREADS_PER_WORKER --opencv_threads_per_worker
append_arg_if_set VALIDATION_STEPS --validation_steps
append_arg_if_set NUM_VALIDATION_IMAGES --num_validation_images
append_list_arg_if_set BENCHMARK_INFERENCE_STEPS --benchmark_inference_steps
append_arg_if_set BATCH_SIZE --batch_size
append_arg_if_set GRADIENT_ACCUMULATION_STEPS --gradient_accumulation_steps
append_arg_if_set MAX_TRAIN_STEPS --max_train_steps
append_arg_if_set SEMANTIC_BACKBONE --semantic_backbone
append_arg_if_set NR_METRIC --nr_metric
append_arg_if_set MIXED_PRECISION --mixed_precision

setsid "${cmd[@]}" &
LAUNCH_PID=$!
wait "${LAUNCH_PID}"
status=$?
trap - INT TERM

if [[ "${status}" -eq 0 ]]; then
    eval_toggle="$(normalize_bool "${RUN_FULL_EVAL_AFTER_TRAIN}")"
    if [[ "${eval_toggle}" == "true" ]]; then
        build_validate_cmd
        echo "[start_train] training finished, starting full offline validation..."
        "${validate_cmd[@]}"
    fi
fi

exit "${status}"
