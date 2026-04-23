#!/usr/bin/env python3
"""
训练启动脚本 - DiffImg2Img 项目

`start_train.py` 是一个极简训练入口，只负责：
- 选择训练 preset/config
- 指定数据目录与输出目录
- 可选地从 checkpoint 恢复
- 通过 accelerate 启动 `main.py --mode train`

数据准备不再通过这个脚本暴露参数。训练开始前会自动检查 prepared cache，
若缺失或陈旧则自动重建。

使用示例:
    python start_train.py --config middle --data-dir /path/to/dataset
    python start_train.py --config middle --data-dir /path/to/dataset --output-dir runs/exp
    python start_train.py --config middle --data-dir /path/to/dataset --resume latest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utils.project_config import load_preset_summary

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _warn_if_cross_mounted_data(data_dir: str) -> None:
    prepared_cache_dir = str(Path(data_dir) / ".prepared")
    if data_dir.startswith("/mnt/") or prepared_cache_dir.startswith("/mnt/"):
        print(
            "[start_train][warning] dataset or prepared cache appears to be on a cross-mounted path.",
            file=sys.stderr,
        )
        print(
            "[start_train][warning] For trustworthy throughput measurements, prefer a local Linux SSD.",
            file=sys.stderr,
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DiffImg2Img 极简训练启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="small_throughput",
        help="训练预设或 YAML 配置路径 (默认: small_throughput)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/path/to/dataset",
        help="数据集目录路径 (默认: /path/to/dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/retinex",
        help="训练输出目录 (默认: runs/retinex)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="从检查点恢复训练 (指定路径或 'latest')",
    )
    return parser


def find_accelerate() -> str:
    venv_accelerate = PROJECT_ROOT / ".venv" / "bin" / "accelerate"
    return str(venv_accelerate) if venv_accelerate.exists() else "accelerate"


def build_command(args: argparse.Namespace) -> list[str]:
    main_script = PROJECT_ROOT / "main.py"
    cmd = [
        find_accelerate(),
        "launch",
        str(main_script),
        "--mode",
        "train",
        "--config",
        args.config,
        "--data_dir",
        args.data_dir,
        "--prepare-on-train",
        "--output_dir",
        args.output_dir,
    ]
    if args.resume:
        cmd.extend(["--resume", args.resume])
    return cmd


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    _warn_if_cross_mounted_data(args.data_dir)

    try:
        summary = load_preset_summary(args.config)
        print("=" * 70)
        print(f"配置: {summary['name']} - {summary['description']}")
        print(f"显存: {summary['target_vram_gb']} | 数据集: {args.data_dir}")
        print(f"输出: {args.output_dir}")
        print("=" * 70)
    except Exception:
        pass

    cmd = build_command(args)
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode
    except KeyboardInterrupt:
        print("\n训练被中断", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
