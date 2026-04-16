#!/usr/bin/env python3
"""
训练启动脚本 - DiffImg2Img 项目

简洁的训练启动脚本，只需指定配置文件和数据集路径。
所有详细配置从 YAML 配置文件加载。

使用示例:
    # 使用默认配置
    python start_train.py

    # 指定配置和数据集
    python start_train.py --config middle --data-dir /path/to/dataset

    # 从检查点恢复
    python start_train.py --config middle --resume latest

    # 验证模式
    python start_train.py --mode validate --model-path runs/retinex/best_model
"""

from utils.project_config import load_preset_summary, resolve_config_path
import argparse
import subprocess
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _warn_if_cross_mounted_data(data_dir: str) -> None:
    prepared_cache_dir = str(Path(data_dir) / ".prepared")
    if data_dir.startswith("/mnt/") or prepared_cache_dir.startswith("/mnt/"):
        print("[start_train][warning] dataset or prepared cache appears to be on a cross-mounted path.", file=sys.stderr)
        print("[start_train][warning] For trustworthy throughput measurements, prefer a local Linux SSD.", file=sys.stderr)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="DiffImg2Img 训练启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="small_throughput",
        help="配置文件: small / middle / max 或 YAML 文件路径 (默认: small)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/f/datasets/kitti_LOL",
        help="数据集目录路径 (默认: /mnt/f/datasets/kitti_LOL)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/retinex",
        help="训练输出目录 (默认: runs/retinex)",
    )

    parser.add_argument(
        "--mode",
        choices=["train", "validate", "predict"],
        default="train",
        help="运行模式 (默认: train)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="从检查点恢复训练 (指定路径或 'latest')",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="模型路径 (用于 validate/predict 模式)",
    )

    parser.add_argument(
        "--attention-backend",
        choices=["auto", "compile", "xformers", "native"],
        help="UNet 运行后端策略",
    )

    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead",
                 "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile 模式",
    )

    parser.add_argument(
        "--use-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="显式开启或关闭 torch.compile",
    )

    return parser


def find_accelerate() -> str:
    """查找 accelerate 可执行文件"""
    venv_accelerate = PROJECT_ROOT / ".venv" / "bin" / "accelerate"
    return str(venv_accelerate) if venv_accelerate.exists() else "accelerate"


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    _warn_if_cross_mounted_data(args.data_dir)

    # 显示配置摘要
    try:
        summary = load_preset_summary(args.config)
        print("=" * 70)
        print(f"配置: {summary['name']} - {summary['description']}")
        print(f"显存: {summary['target_vram_gb']} | 数据集: {args.data_dir}")
        print(f"输出: {args.output_dir}")
        print("=" * 70)
    except Exception:
        pass

    # 构建命令
    accelerate_bin = find_accelerate()
    main_script = PROJECT_ROOT / "main.py"

    cmd = [
        accelerate_bin,
        "launch",
        str(main_script),
        "--mode", args.mode,
        "--config", args.config,
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
    ]

    if args.resume:
        cmd.extend(["--resume", args.resume])
    if args.model_path:
        cmd.extend(["--model_path", args.model_path])
    if args.attention_backend:
        cmd.extend(["--attention_backend", args.attention_backend])
    if args.compile_mode:
        cmd.extend(["--torch_compile_mode", args.compile_mode])
    if args.use_torch_compile is True:
        cmd.append("--use_torch_compile")
    elif args.use_torch_compile is False:
        cmd.append("--no-use_torch_compile")

    # 执行
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode
    except KeyboardInterrupt:
        print("\n训练被中断", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
