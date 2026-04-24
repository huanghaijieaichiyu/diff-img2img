import argparse
import os
import signal
import subprocess
import sys
from dataclasses import dataclass

from utils.project_config import (
    load_config_defaults,
    print_runtime_summary,
    resolve_config_path,
)

MIN_EFFECTIVE_BATCH_SIZE = 16
CPU_BOUND_CROSS_MOUNT_DEFAULTS = {
    "prefetch_factor": 2,
    "decode_cache_size": 32,
    "validation_steps": 2000,
    "num_validation_images": 6,
}


@dataclass(frozen=True)
class TrainDefaults:
    report_to: str
    gradient_accumulation_steps: int
    checkpointing_steps: int
    validation_steps: int
    max_train_steps: int | None
    lr_scheduler: str
    lr_warmup_steps: int
    offset_noise: bool
    snr_gamma: float
    retinex_loss_weight: float
    tv_loss_weight: float
    retinex_consistency_weight: float
    retinex_exposure_weight: float
    grad_clip_norm: float
    offset_noise_scale: float
    freeze_decom_steps: int
    decom_warmup_steps: int
    joint_retinex_ramp_steps: int
    x0_loss_weight: float
    x0_loss_t_max: int
    x0_loss_warmup_steps: int
    residual_scale: float
    ema_decay: float
    prediction_type: str
    unet_layers_per_block: int
    unet_block_channels: list[int]
    unet_down_block_types: list[str]
    unet_up_block_types: list[str]
    conditioning_space: str
    inject_mode: str
    base_condition_channels: int
    decom_base_channels: int
    decom_variant: str
    condition_variant: str
    enable_torch_sdpa_memory_efficient_attention: bool
    num_workers: int
    semantic_backbone: str
    nr_metric: str
    # P0/P1 Improvements
    loss_weighting_scheme: str = "min_snr"  # "min_snr", "p2", "edm"
    use_uncertainty_weighting: bool = False
    frequency_loss_weight: float = 0.0
    edge_loss_weight: float = 0.0
    loss_balance_mode: str = "fixed"
    loss_balance_decay: float = 0.98
    loss_balance_warmup_steps: int = 0
    lpips_stage_enable: bool = False
    lpips_stage_start_ratio: float = 0.8


def _recommended_num_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


DEFAULT_TRAIN_CONFIG = TrainDefaults(
    report_to="tensorboard",
    gradient_accumulation_steps=4,
    checkpointing_steps=1000,
    validation_steps=500,
    max_train_steps=None,
    lr_scheduler="cosine_with_warmup",
    lr_warmup_steps=500,
    offset_noise=True,
    snr_gamma=5.0,
    loss_weighting_scheme="min_snr",
    use_uncertainty_weighting=False,
    retinex_loss_weight=0.1,
    tv_loss_weight=0.1,
    retinex_consistency_weight=1.0,
    retinex_exposure_weight=0.1,
    grad_clip_norm=5.0,
    offset_noise_scale=0.1,
    freeze_decom_steps=0,
    decom_warmup_steps=4000,
    joint_retinex_ramp_steps=2000,
    x0_loss_weight=0.2,
    x0_loss_t_max=200,
    x0_loss_warmup_steps=2000,
    residual_scale=0.5,
    ema_decay=0.9999,
    prediction_type="v_prediction",
    unet_layers_per_block=2,
    unet_block_channels=[32, 64, 128, 256, 512],
    unet_down_block_types=["DownBlock2D", "DownBlock2D",
                           "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
    unet_up_block_types=["AttnUpBlock2D", "AttnUpBlock2D",
                         "UpBlock2D", "UpBlock2D", "UpBlock2D"],
    conditioning_space="hvi_lite",
    inject_mode="film_pyramid",
    base_condition_channels=32,
    decom_base_channels=32,
    decom_variant="middle",
    condition_variant="middle",
    enable_torch_sdpa_memory_efficient_attention=True,
    num_workers=_recommended_num_workers(),
    semantic_backbone="none",
    nr_metric="none",
    frequency_loss_weight=0.02,
    edge_loss_weight=0.03,
    loss_balance_mode="ema",
    loss_balance_decay=0.98,
    loss_balance_warmup_steps=1000,
)


def _add_hidden_argument(parser: argparse.ArgumentParser, *name_or_flags, **kwargs):
    kwargs.setdefault("help", argparse.SUPPRESS)
    if "default" not in kwargs:
        kwargs["default"] = None
    parser.add_argument(*name_or_flags, **kwargs)


def _normalize_validation_metrics_arg(metrics):
    if metrics in (None, "", []):
        return None
    if isinstance(metrics, str):
        metrics = [part.strip() for part in metrics.replace(
            ",", " ").split() if part.strip()]
    normalized = []
    for metric in metrics:
        metric_name = str(metric).strip().lower()
        if metric_name:
            normalized.append(metric_name)
    return normalized or None


def _normalize_step_counts_arg(step_counts):
    if step_counts in (None, "", []):
        return None
    if isinstance(step_counts, str):
        step_counts = [part.strip() for part in step_counts.replace(
            ",", " ").split() if part.strip()]
    normalized = []
    for step_count in step_counts:
        value = int(step_count)
        if value > 0 and value not in normalized:
            normalized.append(value)
    return normalized or None


def _normalize_loss_weighting_scheme_arg(weighting_scheme):
    if weighting_scheme in (None, ""):
        return weighting_scheme
    normalized = str(weighting_scheme).strip().lower()
    if normalized == "mini_snr":
        return "min_snr"
    if normalized not in {"min_snr", "p2", "edm"}:
        raise ValueError(
            f"Unknown loss_weighting_scheme: {weighting_scheme}. Expected one of min_snr/mini_snr/p2/edm"
        )
    return normalized


def _normalize_lpips_stage_start_ratio_arg(ratio):
    if ratio in (None, ""):
        return ratio
    value = float(ratio)
    if not (0.0 <= value <= 1.0):
        raise ValueError("lpips_stage_start_ratio must be in [0, 1]")
    return value


def _looks_cross_mounted(path: str | None) -> bool:
    if not path:
        return False
    return os.path.abspath(path).startswith("/mnt/")


def _collect_explicit_cli_destinations(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest = {}
    for action in parser._actions:
        for option_string in action.option_strings:
            option_to_dest[option_string] = action.dest

    explicit_dests: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit_dests.add(dest)
    return explicit_dests


def _apply_cross_mount_throughput_defaults(args):
    if args.mode != "train":
        args.cross_mount_warning_paths = []
        args.cross_mount_tuning_applied = {}
        return

    slow_paths = [path for path in [args.data_dir] if _looks_cross_mounted(path)]
    args.cross_mount_warning_paths = slow_paths
    args.cross_mount_tuning_applied = {}
    if not slow_paths:
        return

    explicit_dests = getattr(args, "_explicit_cli_dests", set())
    for field_name, tuned_value in CPU_BOUND_CROSS_MOUNT_DEFAULTS.items():
        if field_name in explicit_dests:
            continue
        if getattr(args, field_name, None) != tuned_value:
            setattr(args, field_name, tuned_value)
            args.cross_mount_tuning_applied[field_name] = tuned_value


def apply_train_defaults(args):
    for key, value in DEFAULT_TRAIN_CONFIG.__dict__.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.device is None:
        args.device = "cuda"
    if args.port is None:
        args.port = 8501
    if args.benchmark_inference_steps is None:
        args.benchmark_inference_steps = [8, 20]
    if getattr(args, "prefetch_factor", None) is None:
        args.prefetch_factor = 4
    if getattr(args, "persistent_workers", None) is None:
        args.persistent_workers = args.num_workers > 0
    if getattr(args, "pin_memory", None) is None:
        args.pin_memory = True
    if getattr(args, "decode_cache_size", None) is None:
        args.decode_cache_size = 0
    if getattr(args, "opencv_threads_per_worker", None) is None:
        args.opencv_threads_per_worker = 1
    if getattr(args, "use_lpips", None) is None:
        args.use_lpips = True
    if getattr(args, "lpips_resize", None) in ("", None):
        args.lpips_resize = None
    if getattr(args, "wavelet_loss_weight", None) is None:
        args.wavelet_loss_weight = 0.0
    if getattr(args, "lpips_stage_enable", None) is None:
        args.lpips_stage_enable = False
    if getattr(args, "lpips_stage_start_ratio", None) is None:
        args.lpips_stage_start_ratio = 0.8
    args.lpips_stage_start_ratio = _normalize_lpips_stage_start_ratio_arg(
        getattr(args, "lpips_stage_start_ratio", None)
    )
    if getattr(args, "attention_backend", None) in ("", None):
        args.attention_backend = "auto"
    if getattr(args, "inject_mode", None) in ("", None):
        args.inject_mode = "concat_pyramid"
    if getattr(args, "enable_torch_sdpa_memory_efficient_attention", None) is None:
        args.enable_torch_sdpa_memory_efficient_attention = True
    if getattr(args, "use_torch_compile", None) is None:
        args.use_torch_compile = True
    if getattr(args, "torch_compile_mode", None) in ("", None):
        args.torch_compile_mode = "max-autotune-no-cudagraphs"
    if getattr(args, "allow_unsafe_compile_with_film", None) is None:
        args.allow_unsafe_compile_with_film = False
    args.loss_weighting_scheme = _normalize_loss_weighting_scheme_arg(
        getattr(args, "loss_weighting_scheme", None)
    )
    if getattr(args, "train_fast_validation", None) is None:
        args.train_fast_validation = True
    args.train_validation_metrics = _normalize_validation_metrics_arg(
        getattr(args, "train_validation_metrics", None))
    if args.train_validation_metrics is None:
        args.train_validation_metrics = ["psnr", "ssim"]
    args.train_validation_benchmark_steps = _normalize_step_counts_arg(
        getattr(args, "train_validation_benchmark_steps", None)
    )
    if args.train_validation_benchmark_steps is None:
        args.train_validation_benchmark_steps = list(
            dict.fromkeys([args.num_inference_steps]))

    _apply_cross_mount_throughput_defaults(args)
    if args.cross_mount_warning_paths:
        joined_paths = ", ".join(args.cross_mount_warning_paths)
        print(
            "[throughput-warning] dataset appears to be on a cross-mounted path. "
            f"Prefer a local Linux SSD for better throughput: {joined_paths}",
            flush=True,
        )
        if args.cross_mount_tuning_applied:
            applied = " ".join(
                f"{field_name}={value}" for field_name, value in args.cross_mount_tuning_applied.items()
            )
            print(
                f"[throughput-tuning] applied CPU-bound cross-mount defaults: {applied}",
                flush=True,
            )

    _validate_model_args(args)
    _attach_effective_batch_metadata(args)
    return args


def _validate_model_args(args):
    num_blocks = len(args.unet_block_channels)
    if len(args.unet_down_block_types) != num_blocks:
        raise ValueError(
            "unet_down_block_types length must match unet_block_channels length")
    if len(args.unet_up_block_types) != num_blocks:
        raise ValueError(
            "unet_up_block_types length must match unet_block_channels length")


def _attach_effective_batch_metadata(args):
    if args.mode != "train":
        return

    args.effective_batch_size = max(
        1, int(args.batch_size)) * max(1, int(args.gradient_accumulation_steps))
    args.min_effective_batch_size = MIN_EFFECTIVE_BATCH_SIZE


def _report_effective_batch(args):
    if args.mode != "train":
        return

    effective_batch_size = getattr(
        args,
        "effective_batch_size",
        max(1, int(args.batch_size)) *
        max(1, int(args.gradient_accumulation_steps)),
    )
    print(
        "[train-config] "
        f"batch_size={args.batch_size}, "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
        f"effective_batch={effective_batch_size}"
    )
    if effective_batch_size < MIN_EFFECTIVE_BATCH_SIZE:
        print(
            "[train-config][warning] "
            f"effective_batch={effective_batch_size} is below the recommended minimum of "
            f"{MIN_EFFECTIVE_BATCH_SIZE}. Consider increasing gradient_accumulation_steps."
        )


def _handle_termination_signal(_signum, _frame):
    raise KeyboardInterrupt


def get_args():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", type=str, default="small")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    config_defaults = load_config_defaults(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Diff-Img2Img Unified Engine")

    # User-facing parameters
    parser.add_argument(
        "--config", type=str, default=config_defaults["config"], help="YAML config path or preset name: small / middle / max")
    parser.add_argument("--mode", type=str, default="train", choices=[
                        "train", "predict", "validate", "ui"], help="Execution mode")
    parser.add_argument("--data_dir", type=str,
                        default="../datasets/kitti_LOL", help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="runs/exp1",
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pretrained model for predict/validate")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Path to input video for prediction mode")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint or 'latest'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Training and validation resolution")
    parser.add_argument(
        "--use_retinex",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Retinex decomposition",
    )
    parser.add_argument("--ema", dest="use_ema", action=argparse.BooleanOptionalAction,
                        default=True, help="Enable EMA for the trainable model modules.")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"], help="Mixed precision policy")
    # Hidden advanced compatibility parameters
    _add_hidden_argument(parser, "--device", type=str)
    _add_hidden_argument(parser, "--report_to", type=str)
    _add_hidden_argument(parser, "--log_interval", type=int)
    _add_hidden_argument(parser, "--num_inference_steps", type=int)
    _add_hidden_argument(parser, "--num_validation_images", type=int)
    _add_hidden_argument(parser, "--gradient_accumulation_steps", type=int)
    _add_hidden_argument(parser, "--checkpointing_steps", type=int)
    _add_hidden_argument(parser, "--checkpoints_total_limit", type=int)
    _add_hidden_argument(parser, "--validation_steps", type=int)
    _add_hidden_argument(parser, "--max_train_steps", type=int)
    _add_hidden_argument(parser, "--lr_scheduler", type=str)
    _add_hidden_argument(parser, "--lr_warmup_steps", type=int)
    _add_hidden_argument(parser, "--offset_noise",
                         action="store_true", default=None)
    _add_hidden_argument(parser, "--snr_gamma", type=float)
    _add_hidden_argument(parser, "--retinex_loss_weight", type=float)
    _add_hidden_argument(parser, "--tv_loss_weight", type=float)
    _add_hidden_argument(parser, "--retinex_consistency_weight", type=float)
    _add_hidden_argument(parser, "--retinex_exposure_weight", type=float)
    _add_hidden_argument(parser, "--grad_clip_norm", type=float)
    _add_hidden_argument(parser, "--offset_noise_scale", type=float)
    _add_hidden_argument(parser, "--freeze_decom_steps", type=int)
    _add_hidden_argument(parser, "--decom_warmup_steps", type=int)
    _add_hidden_argument(parser, "--joint_retinex_ramp_steps", type=int)
    _add_hidden_argument(parser, "--x0_loss_weight", type=float)
    _add_hidden_argument(parser, "--x0_loss_t_max", type=int)
    _add_hidden_argument(parser, "--x0_loss_warmup_steps", type=int)
    _add_hidden_argument(parser, "--residual_scale", type=float)
    _add_hidden_argument(parser, "--use_ema", dest="use_ema",
                         action="store_true", default=None)
    _add_hidden_argument(parser, "--ema_decay", type=float)
    _add_hidden_argument(parser, "--prediction_type", type=str)
    _add_hidden_argument(parser, "--unet_layers_per_block", type=int)
    _add_hidden_argument(parser, "--unet_block_channels", nargs="+", type=int)
    _add_hidden_argument(
        parser, "--unet_down_block_types", nargs="+", type=str)
    _add_hidden_argument(parser, "--unet_up_block_types", nargs="+", type=str)
    _add_hidden_argument(parser, "--conditioning_space",
                         type=str, choices=["hvi_lite", "rgb"])
    _add_hidden_argument(parser, "--inject_mode", type=str,
                         choices=["film_pyramid", "concat_pyramid"])
    _add_hidden_argument(parser, "--base_condition_channels", type=int)
    _add_hidden_argument(parser, "--decom_base_channels", type=int)
    _add_hidden_argument(parser, "--decom_variant", type=str,
                         choices=["small", "middle", "max", "naf", "naf_lite"])
    _add_hidden_argument(parser, "--condition_variant", type=str,
                         choices=["small", "small_v2", "middle", "max", "max_v2", "cross_attn"])
    _add_hidden_argument(parser, "--attention_backend",
                         type=str, choices=["auto", "compile", "sdpa", "native"])
    _add_hidden_argument(
        parser, "--enable_torch_sdpa_memory_efficient_attention", action="store_true", default=None)
    _add_hidden_argument(parser, "--use_torch_compile",
                         action=argparse.BooleanOptionalAction, default=None)
    _add_hidden_argument(parser, "--torch_compile_mode", type=str, choices=[
                         "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    _add_hidden_argument(parser, "--allow_unsafe_compile_with_film",
                         action=argparse.BooleanOptionalAction, default=None)
    _add_hidden_argument(
        parser, "--benchmark_inference_steps", nargs="+", type=int)
    _add_hidden_argument(parser, "--num_workers", type=int)
    _add_hidden_argument(parser, "--prefetch_factor", type=int)
    _add_hidden_argument(parser, "--persistent_workers",
                         action=argparse.BooleanOptionalAction, default=None)
    _add_hidden_argument(parser, "--pin_memory",
                         action=argparse.BooleanOptionalAction, default=None)
    _add_hidden_argument(parser, "--decode_cache_size", type=int)
    _add_hidden_argument(parser, "--opencv_threads_per_worker", type=int)
    _add_hidden_argument(parser, "--semantic_backbone",
                         type=str, choices=["none", "resnet18"])
    _add_hidden_argument(parser, "--nr_metric", type=str,
                         choices=["none", "niqe"])
    _add_hidden_argument(parser, "--port", type=int)
    # P0/P1 Improvement parameters
    _add_hidden_argument(parser, "--loss_weighting_scheme",
                         type=str, choices=["min_snr", "mini_snr", "p2", "edm"])
    _add_hidden_argument(parser, "--use_uncertainty_weighting",
                         action="store_true", default=None)
    _add_hidden_argument(parser, "--use_lpips",
                         action=argparse.BooleanOptionalAction, default=None)
    _add_hidden_argument(parser, "--lpips_resize", type=int)
    _add_hidden_argument(parser, "--lpips_stage_enable",
                         action=argparse.BooleanOptionalAction, default=None)
    _add_hidden_argument(parser, "--lpips_stage_start_ratio", type=float)
    _add_hidden_argument(parser, "--wavelet_loss_weight", type=float)
    _add_hidden_argument(parser, "--frequency_loss_weight", type=float)
    _add_hidden_argument(parser, "--edge_loss_weight", type=float)
    _add_hidden_argument(parser, "--loss_balance_mode", type=str, choices=["fixed", "ema", "uncertainty"])
    _add_hidden_argument(parser, "--loss_balance_decay", type=float)
    _add_hidden_argument(parser, "--loss_balance_warmup_steps", type=int)
    _add_hidden_argument(parser, "--train_fast_validation",
                         action=argparse.BooleanOptionalAction, default=None)
    _add_hidden_argument(
        parser, "--train_validation_metrics", nargs="+", type=str)
    _add_hidden_argument(
        parser, "--train_validation_benchmark_steps", nargs="+", type=int)

    parser.set_defaults(**config_defaults)
    explicit_cli_dests = _collect_explicit_cli_destinations(
        parser, sys.argv[1:])
    args = parser.parse_args()
    args._explicit_cli_dests = explicit_cli_dests
    args.config = resolve_config_path(args.config)
    return apply_train_defaults(args)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_termination_signal)
    try:
        args = get_args()
        _report_effective_batch(args)
        if args.mode != "ui":
            print_runtime_summary(args)

        if args.mode == "ui":
            print("🚀 Launching Diff-Img2Img Studio...")
            ui_path = os.path.join("ui", "app.py")
            if not os.path.exists(ui_path):
                print(f"Error: UI file not found at {ui_path}")
                sys.exit(1)

            cmd = [sys.executable, "-m", "streamlit", "run", ui_path]
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                print("\nUI Stopped.")
        else:
            from core.engine import DiffusionEngine

            engine = DiffusionEngine(args)

            if args.mode == "train":
                engine.train()
            elif args.mode == "validate":
                engine.validate()
            elif args.mode == "predict":
                engine.predict()
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", flush=True)
        sys.exit(2)
    except KeyboardInterrupt:
        print(
            "\n[interrupt] Received keyboard interrupt. Exiting gracefully.", flush=True)
        sys.exit(130)
