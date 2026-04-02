import argparse
import os
import subprocess
import sys
from dataclasses import dataclass


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
    enable_xformers_memory_efficient_attention: bool
    num_workers: int
    semantic_backbone: str
    nr_metric: str


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
        lr_scheduler="constant_with_warmup",
        lr_warmup_steps=500,
        offset_noise=True,
        snr_gamma=5.0,
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
        lr_scheduler="constant_with_warmup",
        lr_warmup_steps=200,
        offset_noise=True,
        snr_gamma=5.0,
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


def _ensure_training_data_mode(args):
    if args.mode != "train":
        return

    if args.online_synthesis:
        return

    low_dir = os.path.join(args.data_dir, "our485", "low")
    if not os.path.isdir(low_dir):
        args.online_synthesis = True
        return

    supported_exts = (".png", ".jpg", ".jpeg", ".bmp")
    has_low_images = any(file_name.lower().endswith(supported_exts) for file_name in os.listdir(low_dir))
    if not has_low_images:
        args.online_synthesis = True


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

    # Legacy compatibility
    if getattr(args, "use_ema", None) is None:
        args.use_ema = True

    _ensure_training_data_mode(args)
    return args


def get_args():
    parser = argparse.ArgumentParser(description="Diff-Img2Img Unified Engine")

    # User-facing parameters
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict", "validate", "ui"], help="Execution mode")
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
    parser.add_argument("--use_retinex", action="store_true", help="Enable Retinex decomposition")
    parser.add_argument("--ema", dest="use_ema", action=argparse.BooleanOptionalAction, default=True, help="Enable EMA for the diffusion backbone.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision policy")
    parser.add_argument("--train_profile", type=str, default="auto", choices=sorted(TRAIN_PROFILES.keys()), help="High-level training preset")
    parser.add_argument("--log_interval", type=int, default=10, help="How often to refresh training summaries")
    parser.add_argument("--num_inference_steps", type=int, default=8, help="Inference steps for prediction/validation")
    parser.add_argument("--num_validation_images", type=int, default=16, help="How many validation images to evaluate")

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
    _add_hidden_argument(parser, "--enable_xformers_memory_efficient_attention", action="store_true", default=None)
    _add_hidden_argument(parser, "--benchmark_inference_steps", nargs="+", type=int)
    _add_hidden_argument(parser, "--num_workers", type=int)
    _add_hidden_argument(parser, "--semantic_backbone", type=str, choices=["none", "resnet18"])
    _add_hidden_argument(parser, "--nr_metric", type=str, choices=["none", "niqe"])
    _add_hidden_argument(parser, "--port", type=int)

    return apply_profile_defaults(parser.parse_args())


if __name__ == "__main__":
    args = get_args()

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
