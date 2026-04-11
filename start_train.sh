#!/usr/bin/env bash
#
# 训练启动脚本 - Bash 版本
#
# 使用示例:
#   ./start_train.sh                                    # 使用默认配置
#   ./start_train.sh middle /path/to/dataset            # 指定配置和数据集
#   RUN_MODE=validate ./start_train.sh middle           # 验证模式
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 参数
CONFIG="${1:-small}"
DATA_DIR="${2:-/mnt/f/datasets/kitti_LOL}"
OUTPUT_DIR="${3:-runs/retinex}"

# 环境变量
RUN_MODE="${RUN_MODE:-train}"
RESUME="${RESUME:-}"
MODEL_PATH="${MODEL_PATH:-}"

# 查找 Python
if [[ -x "${SCRIPT_DIR}/.venv/bin/python3" ]]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python3"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
    PYTHON="python3"
fi

# 构建参数
ARGS=(
    --config "$CONFIG"
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
    --mode "$RUN_MODE"
)

[[ -n "$RESUME" ]] && ARGS+=(--resume "$RESUME")
[[ -n "$MODEL_PATH" ]] && ARGS+=(--model-path "$MODEL_PATH")

# 执行
exec "${PYTHON}" "${SCRIPT_DIR}/start_train.py" "${ARGS[@]}"
