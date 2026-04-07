import argparse
import os
import signal
import subprocess
import sys
from dataclasses import dataclass

import yaml

from utils.project_config import (
    load_config_defaults,
    print_runtime_summary,
    resolve_config_path,
)

MIN_EFFECTIVE_BATCH_SIZE = 16


@dataclass(frozen=True)
class TrainProfileConfig:
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
    online_synthesis: bool
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
    enable_xformers_memory_efficient_attention: bool
    num_workers: int
    semantic_backbone: str
    nr_metric: str
    # P0/P1 Improvements
    loss_weighting_scheme: str = "min_snr"  # "min_snr", "p2", "edm"
    use_uncertainty_weighting: bool = False


def _recommended_num_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


TRAIN_PROFILES = {
    "auto": TrainProfileConfig(
        report_to="tensorboard",
        gradient_accumulation_steps=4,
        checkpointing_steps=1000,
        validation_steps=500,
        max_train_steps=None,
        lr_scheduler="cosine_with_warmup",  # P0: Use cosine annealing
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
        online_synthesis=False,
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
        unet_down_block_types=["DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
        unet_up_block_types=["AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"],
        conditioning_space="hvi_lite",
        inject_mode="film_pyramid",
        base_condition_channels=32,
        decom_base_channels=32,
        decom_variant="middle",
        condition_variant="middle",
        enable_xformers_memory_efficient_attention=True,
        num_workers=_recommended_num_workers(),
        semantic_backbone="resnet18",
        nr_metric="niqe",
    ),
    "debug_online": TrainProfileConfig(
        report_to="tensorboard",
        gradient_accumulation_steps=1,
        checkpointing_steps=500,
        validation_steps=250,
        max_train_steps=None,
        lr_scheduler="cosine_with_warmup",  # P0: Use cosine annealing
        lr_warmup_steps=200,
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
        online_synthesis=True,
        freeze_decom_steps=0,
        decom_warmup_steps=1000,
        joint_retinex_ramp_steps=1000,
        x0_loss_weight=0.2,
        x0_loss_t_max=200,
        x0_loss_warmup_steps=1000,
        residual_scale=0.5,
        ema_decay=0.9999,
        prediction_type="v_prediction",
        unet_layers_per_block=2,
        unet_block_channels=[32, 64, 128, 256, 512],
        unet_down_block_types=["DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
        unet_up_block_types=["AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"],
        conditioning_space="hvi_lite",
        inject_mode="film_pyramid",
        base_condition_channels=32,
        decom_base_channels=32,
        decom_variant="small",
        condition_variant="small",
        enable_xformers_memory_efficient_attention=True,
        num_workers=max(2, min(4, _recommended_num_workers())),
        semantic_backbone="none",
        nr_metric="none",
    ),
}

def _add_hidden_argument(parser: argparse.ArgumentParser, *name_or_flags, **kwargs):
    kwargs.setdefault("help", argparse.SUPPRESS)
    if "default" not in kwargs:
        kwargs["default"] = None
    parser.add_argument(*name_or_flags, **kwargs)


def _normalize_darker_ranges_arg(darker_ranges):
    if darker_ranges in (None, "", {}):
        return None
    if isinstance(darker_ranges, dict):
        return darker_ranges
    if isinstance(darker_ranges, str):
        parsed = yaml.safe_load(darker_ranges)
        if not isinstance(parsed, dict):
            raise ValueError("darker_ranges must decode to a dict.")
        return parsed
    raise TypeError(f"Unsupported darker_ranges type: {type(darker_ranges)!r}")


def apply_profile_defaults(args):
    profile = TRAIN_PROFILES[args.train_profile]
    for key, value in profile.__dict__.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.device is None:
        args.device = "cuda"
    if args.port is None:
        args.port = 8501
    if args.benchmark_inference_steps is None:
        args.benchmark_inference_steps = [8, 20]
    if getattr(args, "prepare_on_train", None) is None:
        args.prepare_on_train = True
    if getattr(args, "prepared_cache_dir", None) in ("", None):
        args.prepared_cache_dir = os.path.join(args.data_dir, ".prepared")
    if getattr(args, "offline_variant_count", None) is None:
        args.offline_variant_count = 3
    if getattr(args, "prepare_workers", None) is None:
        args.prepare_workers = _recommended_num_workers()
    if getattr(args, "prepare_force", None) is None:
        args.prepare_force = False
    if getattr(args, "synthesis_seed", None) is None:
        args.synthesis_seed = args.seed if args.seed is not None else 42
    args.darker_ranges = _normalize_darker_ranges_arg(getattr(args, "darker_ranges", None))
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

    # Legacy compatibility
    if getattr(args, "use_ema", None) is None:
        args.use_ema = True

    _validate_model_args(args)
    _validate_prepare_args(args)
    _attach_effective_batch_metadata(args)
    return args


def _validate_model_args(args):
    num_blocks = len(args.unet_block_channels)
    if len(args.unet_down_block_types) != num_blocks:
        raise ValueError("unet_down_block_types length must match unet_block_channels length")
    if len(args.unet_up_block_types) != num_blocks:
        raise ValueError("unet_up_block_types length must match unet_block_channels length")


def _attach_effective_batch_metadata(args):
    if args.mode != "train":
        return

    args.effective_batch_size = max(1, int(args.batch_size)) * max(1, int(args.gradient_accumulation_steps))
    args.min_effective_batch_size = MIN_EFFECTIVE_BATCH_SIZE


def _validate_prepare_args(args):
    if getattr(args, "offline_variant_count", 1) < 1:
        raise ValueError("offline_variant_count must be >= 1")
    if getattr(args, "prepare_workers", 1) < 1:
        raise ValueError("prepare_workers must be >= 1")
    darker_ranges = getattr(args, "darker_ranges", None)
    if darker_ranges is not None and not isinstance(darker_ranges, dict):
        raise ValueError("darker_ranges must be a dict or a JSON string that decodes to a dict")


def _ensure_prepared_training_manifest(args):
    from datasets.prepare_data import ensure_prepared_training_data

    print(
        "[prepare] "
        f"cache_dir={args.prepared_cache_dir} "
        f"variants={args.offline_variant_count} "
        f"workers={args.prepare_workers} "
        f"seed={args.synthesis_seed}",
        flush=True,
    )
    manifest_path, prepared = ensure_prepared_training_data(
        args.data_dir,
        args.prepared_cache_dir,
        variant_count=args.offline_variant_count,
        synthesis_seed=args.synthesis_seed,
        darker_ranges=args.darker_ranges,
        prepare_workers=args.prepare_workers,
        force=args.prepare_force,
        prepare_on_train=args.prepare_on_train,
    )
    args.train_manifest_path = manifest_path
    return manifest_path, prepared


def _report_effective_batch(args):
    if args.mode != "train":
        return

    effective_batch_size = getattr(
        args,
        "effective_batch_size",
        max(1, int(args.batch_size)) * max(1, int(args.gradient_accumulation_steps)),
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
    parser.add_argument("--config", type=str, default=config_defaults["config"], help="YAML config path or preset name: small / middle / max")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "prepare", "predict", "validate", "ui"], help="Execution mode")
    parser.add_argument("--data_dir", type=str, default="../datasets/kitti_LOL", help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="runs/exp1", help="Output directory for logs and checkpoints")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model for predict/validate")
    parser.add_argument("--video_path", type=str, default=None, help="Path to input video for prediction mode")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint or 'latest'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--resolution", type=int, default=256, help="Training and validation resolution")
    parser.add_argument(
        "--use_retinex",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Retinex decomposition",
    )
    parser.add_argument("--ema", dest="use_ema", action=argparse.BooleanOptionalAction, default=True, help="Enable EMA for the trainable model modules.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision policy")
    parser.add_argument("--train_profile", type=str, default="auto", choices=sorted(TRAIN_PROFILES.keys()), help="High-level training preset")
    parser.add_argument("--log_interval", type=int, default=10, help="How often to refresh training summaries")
    parser.add_argument("--num_inference_steps", type=int, default=8, help="Inference steps for prediction/validation")
    parser.add_argument("--num_validation_images", type=int, default=16, help="How many validation images to evaluate")
    parser.add_argument("--prepare_on_train", "--prepare-on-train", dest="prepare_on_train", action=argparse.BooleanOptionalAction, default=None, help="Automatically build the multi-variant prepared training cache before training.")
    parser.add_argument("--prepared_cache_dir", "--prepared-cache-dir", type=str, default=None, help="Directory that stores prepared multi-variant low-light training data.")
    parser.add_argument("--offline_variant_count", "--offline-variant-count", type=int, default=None, help="How many low-light variants to prepare for each training high-light image.")
    parser.add_argument("--prepare_workers", "--prepare-workers", type=int, default=None, help="Worker count for offline data preparation.")
    parser.add_argument("--prepare_force", "--prepare-force", dest="prepare_force", action=argparse.BooleanOptionalAction, default=None, help="Force rebuilding the prepared training cache.")
    parser.add_argument("--synthesis_seed", "--synthesis-seed", type=int, default=None, help="Base seed used for offline low-light synthesis.")
    parser.add_argument("--darker_ranges", "--darker-ranges", type=str, default=None, help="JSON/YAML dict overriding Darker parameter ranges during offline preparation.")

    # Hidden advanced compatibility parameters
    _add_hidden_argument(parser, "--device", type=str)
    _add_hidden_argument(parser, "--report_to", type=str)
    _add_hidden_argument(parser, "--gradient_accumulation_steps", type=int)
    _add_hidden_argument(parser, "--checkpointing_steps", type=int)
    _add_hidden_argument(parser, "--checkpoints_total_limit", type=int)
    _add_hidden_argument(parser, "--validation_steps", type=int)
    _add_hidden_argument(parser, "--max_train_steps", type=int)
    _add_hidden_argument(parser, "--lr_scheduler", type=str)
    _add_hidden_argument(parser, "--lr_warmup_steps", type=int)
    _add_hidden_argument(parser, "--offset_noise", action="store_true", default=None)
    _add_hidden_argument(parser, "--snr_gamma", type=float)
    _add_hidden_argument(parser, "--retinex_loss_weight", type=float)
    _add_hidden_argument(parser, "--tv_loss_weight", type=float)
    _add_hidden_argument(parser, "--retinex_consistency_weight", type=float)
    _add_hidden_argument(parser, "--retinex_exposure_weight", type=float)
    _add_hidden_argument(parser, "--grad_clip_norm", type=float)
    _add_hidden_argument(parser, "--offset_noise_scale", type=float)
    _add_hidden_argument(parser, "--online_synthesis", action="store_true", default=None)
    _add_hidden_argument(parser, "--freeze_decom_steps", type=int)
    _add_hidden_argument(parser, "--decom_warmup_steps", type=int)
    _add_hidden_argument(parser, "--joint_retinex_ramp_steps", type=int)
    _add_hidden_argument(parser, "--x0_loss_weight", type=float)
    _add_hidden_argument(parser, "--x0_loss_t_max", type=int)
    _add_hidden_argument(parser, "--x0_loss_warmup_steps", type=int)
    _add_hidden_argument(parser, "--residual_scale", type=float)
    _add_hidden_argument(parser, "--use_ema", dest="use_ema", action="store_true", default=None)
    _add_hidden_argument(parser, "--ema_decay", type=float)
    _add_hidden_argument(parser, "--prediction_type", type=str)
    _add_hidden_argument(parser, "--unet_layers_per_block", type=int)
    _add_hidden_argument(parser, "--unet_block_channels", nargs="+", type=int)
    _add_hidden_argument(parser, "--unet_down_block_types", nargs="+", type=str)
    _add_hidden_argument(parser, "--unet_up_block_types", nargs="+", type=str)
    _add_hidden_argument(parser, "--conditioning_space", type=str, choices=["hvi_lite", "rgb"])
    _add_hidden_argument(parser, "--inject_mode", type=str, choices=["film_pyramid"])
    _add_hidden_argument(parser, "--base_condition_channels", type=int)
    _add_hidden_argument(parser, "--decom_base_channels", type=int)
    _add_hidden_argument(parser, "--decom_variant", type=str, choices=["small", "middle", "max", "naf", "naf_lite"])
    _add_hidden_argument(parser, "--condition_variant", type=str, choices=["small", "small_v2", "middle", "max", "max_v2", "cross_attn"])
    _add_hidden_argument(parser, "--enable_xformers_memory_efficient_attention", action="store_true", default=None)
    _add_hidden_argument(parser, "--benchmark_inference_steps", nargs="+", type=int)
    _add_hidden_argument(parser, "--num_workers", type=int)
    _add_hidden_argument(parser, "--prefetch_factor", type=int)
    _add_hidden_argument(parser, "--persistent_workers", action=argparse.BooleanOptionalAction, default=None)
    _add_hidden_argument(parser, "--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    _add_hidden_argument(parser, "--decode_cache_size", type=int)
    _add_hidden_argument(parser, "--opencv_threads_per_worker", type=int)
    _add_hidden_argument(parser, "--semantic_backbone", type=str, choices=["none", "resnet18"])
    _add_hidden_argument(parser, "--nr_metric", type=str, choices=["none", "niqe"])
    _add_hidden_argument(parser, "--port", type=int)
    # P0/P1 Improvement parameters
    _add_hidden_argument(parser, "--loss_weighting_scheme", type=str, choices=["min_snr", "p2", "edm"])
    _add_hidden_argument(parser, "--use_uncertainty_weighting", action="store_true", default=None)
    _add_hidden_argument(parser, "--use_lpips", action=argparse.BooleanOptionalAction, default=None)
    _add_hidden_argument(parser, "--lpips_resize", type=int)
    _add_hidden_argument(parser, "--wavelet_loss_weight", type=float)

    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    args.config = resolve_config_path(args.config)
    return apply_profile_defaults(args)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_termination_signal)
    try:
        args = get_args()
        _report_effective_batch(args)
        if args.mode != "ui":
            print_runtime_summary(args)

        if args.mode == "prepare":
            manifest_path, prepared = _ensure_prepared_training_manifest(args)
            action = "Prepared" if prepared else "Validated"
            print(f"{action} multi-variant training cache: {manifest_path}")
        elif args.mode == "ui":
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
            if args.mode == "train":
                manifest_path, prepared = _ensure_prepared_training_manifest(args)
                if prepared:
                    print(f"[prepare] built multi-variant training cache at {manifest_path}")
                else:
                    print(f"[prepare] using existing multi-variant training cache at {manifest_path}")
            from core.engine import DiffusionEngine

            engine = DiffusionEngine(args)

            if args.mode == "train":
                engine.train()
            elif args.mode == "validate":
                engine.validate()
            elif args.mode == "predict":
                engine.predict()
    except ValueError as exc:
        print(f"Error: {exc}", flush=True)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[interrupt] Received keyboard interrupt. Exiting gracefully.", flush=True)
        sys.exit(130)
