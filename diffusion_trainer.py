import argparse
import os
import math
import logging
from pathlib import Path
import random
from PIL import Image

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
# from torchvision.transforms.functional import rgb_to_hsv # Replaced with local version
from utils.color_trans import RGB_HSV  # Import from local project utils

import diffusers
# Import ConfigMixin for type hinting
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# 引入 DDPMSchedulerOutput 用于类型提示
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from torchvision import transforms
from torcheval.metrics.functional import peak_signal_noise_ratio

from utils.misic import ssim, save_path
from datasets.data_set import LowLightDataset  # 假设数据集类可用
import lpips

# 检查 diffusers 版本
check_min_version("0.10.0")  # 示例版本，根据需要调整

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a conditional diffusion model training script for low-light enhancement.")
    parser.add_argument(
        "--data", type=str, default="../datasets/kitti_LOL", help="数据集根目录"
    )
    parser.add_argument(
        "--output", type=str, default="run_diffusion", help="所有输出 (模型, 日志等) 的根目录"
    )
    parser.add_argument("--overwrite_output",
                        action="store_true", help="是否覆盖输出目录")
    parser.add_argument("--cache", type=str, default=None, help="缓存目录")
    parser.add_argument("--seed", type=int,
                        default=random.randint(0, 1000000), help="随机种子")
    parser.add_argument(
        "--resolution", type=int, default=256, help="输入图像分辨率"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="训练和评估的批次大小"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps", type=int, default=None, help="如果设置，将覆盖 epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="是否启用梯度检查点"
    )
    parser.add_argument(
        "--lr", type=float, default=4e-4, help="优化器初始学习率"
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant_with_warmup",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help="学习率调度器类型"
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="学习率预热步数"
    )
    parser.add_argument(
        "--b1", type=float, default=0.9, help="AdamW 优化器的 beta1 参数"
    )
    parser.add_argument(
        "--b2", type=float, default=0.999, help="AdamW 优化器的 beta2 参数"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="AdamW 优化器的权重衰减"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-07, help="AdamW 优化器的 epsilon 参数"
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="最大梯度范数（用于梯度裁剪）"
    )
    parser.add_argument(
        "--mixed_precision", type=str, default='fp16', choices=["no", "fp16", "bf16"],
        help="是否使用混合精度训练。选择 'fp16' 或 'bf16' (需要 PyTorch >= 1.10)，或 'no' 关闭。"
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="是否启用 xformers 内存高效注意力"
    )
    parser.add_argument(
        "--checkpointing_steps", type=int, default=5000, help="每 N 步保存一次检查点(调整默认值)"
    )
    parser.add_argument(
        "--checkpoints_total_limit", type=int, default=5, help=("限制检查点总数。删除旧的检查点。(调整默认值)")
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="从哪个检查点恢复训练 ('latest' 或特定路径)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Dataloader 使用的工作线程数"
    )
    parser.add_argument(
        "--unet_layers_per_block", type=int, default=2, help="UNet 中每个块的 ResNet 层数"
    )
    parser.add_argument(
        "--unet_block_channels", nargs='+', type=int, default=[64, 64, 128, 128, 256, 512], help="UNet 各层级的通道数"
    )
    parser.add_argument(
        "--unet_down_block_types", nargs='+', type=str,
        default=["DownBlock2D", "DownBlock2D", "DownBlock2D",
                 "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
        help="UNet 下采样块类型"
    )
    parser.add_argument(
        "--unet_up_block_types", nargs='+', type=str,
        default=["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D",
                 "UpBlock2D", "UpBlock2D", "UpBlock2D"],
        help="UNet 上采样块类型"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="采样/推断步数"
    )
    parser.add_argument(
        "--num_validation_images", type=int, default=4, help="验证时生成的图像数量"
    )
    parser.add_argument(
        "--validation_epochs", type=int, default=5, help="每 N 个 epoch 运行一次验证 (将基于步数触发)"
    )
    # === 添加 LPIPS 损失权重 ===
    parser.add_argument(
        "--lambda_lpips", type=float, default=0.5, help="LPIPS 损失的权重"
    )
    # =========================
    # === 添加 Accelerate 日志报告目标 ===
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ===========================

    # === 添加轻量化 UNet 参数 ===
    parser.add_argument(
        "--lightweight_unet", action="store_true", help="使用轻量化的 UNet 配置进行快速测试"
    )
    # ===========================
    # === 添加 HSV 损失相关参数 ===
    parser.add_argument(
        "--use_hsv_loss", action="store_true", help="是否使用 HSV 加权损失替代 MSE 损失"
    )
    parser.add_argument(
        "--hsv_weights", nargs=3, type=float, default=[0.1, 0.1, 0.8],
        help="HSV 加权损失中 H, S, V 通道的权重，例如 0.1 0.1 0.8"
    )
    # =======================

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and args.mixed_precision == "fp16":
        logger.warning(
            "FP16 is not recommended for multi-GPU training. Setting mixed_precision to 'no'.")
        args.mixed_precision = "no"  # 多 GPU 不推荐 fp16

    # === 轻量化配置覆盖 ===
    if args.lightweight_unet:
        # logger.info("使用轻量化 UNet 配置...") # <-- 移动到 main 函数
        args.unet_layers_per_block = 1
        args.unet_block_channels = [64, 64, 128, 128]  # 减少层级和通道
        args.unet_down_block_types = [
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ]
        args.unet_up_block_types = [
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ]
    # ======================

    return args


def main():
    args = parse_args()

    # 日志目录固定在 output 下的 'logs'
    logging_dir = os.path.join(args.output, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # 设置日志 (Accelerator 初始化之后)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # 在 Accelerator 初始化后记录配置信息
    if args.lightweight_unet:
        logger.info("使用轻量化 UNet 配置...")
    if args.use_hsv_loss:  # Log HSV usage
        logger.info(
            f"使用 HSV 加权损失，权重 H: {args.hsv_weights[0]}, S: {args.hsv_weights[1]}, V: {args.hsv_weights[2]}")

    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"设置随机种子为: {args.seed}")

    # 创建输出目录
    if accelerator.is_main_process:
        save_path(args.output)  # Ensures output exists
        # if args.overwrite_output and os.path.exists(args.output): # This was handled by save_path
        #     shutil.rmtree(args.output)
        # os.makedirs(args.output, exist_ok=True)
        logger.info(f"Output directory {args.output} ensured/created.")

    # 初始化模型 - 使用解析后的参数
    logger.info("Initializing UNet model...")
    logger.info(f"  Layers per block: {args.unet_layers_per_block}")
    logger.info(f"  Block channels: {args.unet_block_channels}")
    logger.info(f"  Down blocks: {args.unet_down_block_types}")
    logger.info(f"  Up blocks: {args.unet_up_block_types}")
    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=6,  # 3 (noisy) + 3 (condition)
        out_channels=3,
        layers_per_block=args.unet_layers_per_block,
        block_out_channels=args.unet_block_channels,
        down_block_types=args.unet_down_block_types,
        up_block_types=args.unet_up_block_types,
    )

    if args.enable_xformers_memory_efficient_attention:
        try:
            model.enable_xformers_memory_efficient_attention()
            logger.info("启用 xformers 内存高效注意力")
        except Exception as e:
            logger.warning(f"无法启用 xformers 内存高效注意力: {e}. 继续执行而不使用 xformers。")

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # 初始化噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # 初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2),
        weight_decay=args.weight_decay,
        eps=args.epsilon,
    )

    # 数据处理
    preprocess = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # To [-1, 1]
        ]
    )
    eval_preprocess = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # To [-1, 1]
        ]
    )

    # 创建数据集和数据加载器
    try:
        train_dataset = LowLightDataset(
            image_dir=args.data, transform=preprocess, phase="train")
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        eval_dataset = LowLightDataset(
            image_dir=args.data, transform=eval_preprocess, phase="test")
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
        )
    except Exception as e:
        logger.error(
            f"加载数据集失败，请检查路径 '{args.data}' 和 LowLightDataset 实现: {e}")
        return

    # 初始化学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps if args.max_train_steps else len(
            train_dataloader) * args.epochs * args.gradient_accumulation_steps,
    )

    # 使用 Accelerate 准备
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # 初始化 HSV 转换器
    # Initialize and move to device
    rgb_hsv_converter = RGB_HSV().to(accelerator.device)

    # 初始化追踪器 (tensorboard)
    if accelerator.is_main_process:
        run_name = Path(
            args.output).name if args.output else "diffusion_conditional_run"
        config_to_log = vars(args).copy()
        for key in ["unet_block_channels", "unet_down_block_types", "unet_up_block_types", "hsv_weights"]:
            if key in config_to_log and isinstance(config_to_log[key], list):
                config_to_log[key] = ",".join(map(str, config_to_log[key]))
        accelerator.init_trackers(run_name, config=config_to_log)

    # 计算训练步数
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch
    else:
        args.epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch)

    # 断点续训逻辑
    total_batch_size = args.batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps
    global_step = 0
    first_epoch = 0

    if args.resume:
        if args.resume != "latest":
            path = os.path.basename(args.resume)
        else:
            dirs = os.listdir(args.output)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume}' does not exist. Starting a new training run."
            )
            args.resume = None
        else:
            checkpoint_path = os.path.join(args.output, path)
            accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
            try:
                accelerator.load_state(checkpoint_path)
                global_step = int(path.split("-")[1])
                accelerator.print(
                    f"Successfully loaded state from {checkpoint_path}. Resuming global step {global_step}.")
            except Exception as e:
                accelerator.print(
                    f"Failed to load state from {checkpoint_path}: {e}. Starting from scratch.")
                global_step = 0
            first_epoch = global_step // num_update_steps_per_epoch

    # === 初始化 LPIPS 模型 ===
    if accelerator.is_main_process:
        logger.info("Initializing LPIPS model...")
    try:
        lpips_model = lpips.LPIPS(net='alex').to(accelerator.device)
        lpips_model.eval()
        logger.info("LPIPS model initialized successfully.")
    except Exception as e:
        logger.error(
            f"Failed to initialize LPIPS model: {e}. LPIPS loss will not be used.")
        lpips_model = None
    # ========================

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num eval examples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Starting epoch = {first_epoch + 1}")
    logger.info(f"  Starting global step = {global_step}")

    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        # Initialize accumulators for individual loss components for logging
        accum_loss_recon_mse = 0.0
        accum_loss_hsv = 0.0
        accum_loss_lpips = 0.0

        for step, batch in enumerate(train_dataloader):
            low_light_images, clean_images = batch

            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config["num_train_timesteps"], (bsz,), device=clean_images.device
            ).int()  # Corrected to .int()

            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps
            )

            with accelerator.accumulate(model):
                model_input = torch.cat(
                    [noisy_images, low_light_images], dim=1)
                noise_pred = model(model_input, timesteps).sample

                loss_recon = torch.tensor(0.0).to(accelerator.device)
                current_step_loss_mse = 0.0
                current_step_loss_hsv = 0.0

                # Calculate pred_x0 (predicted denoised image)
                # This is needed for both HSV and LPIPS loss
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(
                    timesteps.device)
                sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt(
                ).view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_prod = (
                    1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
                pred_x0 = (noisy_images - sqrt_one_minus_alpha_prod *
                           noise_pred) / (sqrt_alpha_prod + 1e-8)

                if args.use_hsv_loss:
                    # Denormalize to [0, 1] for HSV conversion
                    pred_x0_0_1 = (pred_x0 / 2 + 0.5).clamp(0, 1)
                    clean_images_0_1 = (clean_images / 2 + 0.5).clamp(0, 1)

                    try:
                        pred_x0_hsv = rgb_hsv_converter.rgb_to_hsv(pred_x0_0_1)
                        clean_images_hsv = rgb_hsv_converter.rgb_to_hsv(
                            clean_images_0_1)

                        loss_h = F.mse_loss(
                            pred_x0_hsv[:, 0:1, :, :], clean_images_hsv[:, 0:1, :, :])
                        loss_s = F.mse_loss(
                            pred_x0_hsv[:, 1:2, :, :], clean_images_hsv[:, 1:2, :, :])
                        loss_v = F.mse_loss(
                            pred_x0_hsv[:, 2:3, :, :], clean_images_hsv[:, 2:3, :, :])

                        loss_recon = (args.hsv_weights[0] * loss_h +
                                      args.hsv_weights[1] * loss_s +
                                      args.hsv_weights[2] * loss_v)
                        current_step_loss_hsv = loss_recon.item()
                    except Exception as e:
                        logger.warning(
                            f"HSV conversion or loss calculation failed: {e}. Falling back to MSE loss on noise for this step.")
                        loss_recon = F.mse_loss(
                            noise_pred.float(), noise.float())
                        current_step_loss_mse = loss_recon.item()
                else:
                    loss_recon = F.mse_loss(noise_pred.float(), noise.float())
                    current_step_loss_mse = loss_recon.item()

                # Accumulate for logging per gradient update
                accum_loss_recon_mse += current_step_loss_mse
                accum_loss_hsv += current_step_loss_hsv

                loss_lpips_val = 0.0
                loss_lpips = torch.tensor(0.0).to(accelerator.device)
                if lpips_model is not None and args.lambda_lpips > 0:
                    # pred_x0 is already calculated
                    pred_x0_clamp = torch.clamp(pred_x0, -1.0, 1.0)
                    clean_images_clamp = torch.clamp(clean_images, -1.0, 1.0)

                    current_lpips = lpips_model(
                        pred_x0_clamp.float(), clean_images_clamp.float()).mean()
                    loss_lpips = current_lpips
                    loss_lpips_val = current_lpips.item()

                accum_loss_lpips += loss_lpips_val

                loss = loss_recon + args.lambda_lpips * loss_lpips

                gathered_loss = accelerator.gather(
                    loss.repeat(args.batch_size))
                if gathered_loss is not None:
                    if isinstance(gathered_loss, torch.Tensor):
                        avg_loss = gathered_loss.mean()
                        train_loss += avg_loss.item() / args.gradient_accumulation_steps
                    else:
                        logger.warning(
                            f"accelerator.gather returned unexpected type: {type(gathered_loss)}")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    current_lr = lr_scheduler.get_last_lr(
                    )[0] if lr_scheduler else args.lr

                    # Calculate average of accumulated losses for logging this step
                    log_mse = accum_loss_recon_mse / args.gradient_accumulation_steps
                    log_hsv = accum_loss_hsv / args.gradient_accumulation_steps
                    log_lpips = accum_loss_lpips / args.gradient_accumulation_steps

                    logs = {
                        "loss": train_loss,  # This is the accumulated total loss for the logging interval
                        "lr": current_lr,
                        "epoch": epoch + 1
                    }
                    postfix_dict = {
                        "loss": f"{train_loss:.4f}",
                        "lr": f"{current_lr:.6f}",
                        "epoch": epoch + 1
                    }

                    if args.use_hsv_loss:
                        logs["loss_hsv"] = log_hsv
                        postfix_dict["hsv"] = f"{log_hsv:.4f}"
                        if current_step_loss_mse > 0:  # Log MSE if HSV failed and fell back
                            logs["loss_mse_fallback"] = log_mse
                            postfix_dict["mse_fb"] = f"{log_mse:.4f}"
                    else:
                        logs["loss_mse"] = log_mse
                        postfix_dict["mse"] = f"{log_mse:.4f}"

                    if lpips_model is not None and args.lambda_lpips > 0:
                        logs["loss_lpips"] = log_lpips
                        postfix_dict["lpips"] = f"{log_lpips:.4f}"

                    progress_bar.set_postfix(**postfix_dict)
                    accelerator.log(logs, step=global_step)

                # Reset accumulators for the next logging interval
                train_loss = 0.0
                accum_loss_recon_mse = 0.0
                accum_loss_hsv = 0.0
                accum_loss_lpips = 0.0

                if accelerator.is_main_process:
                    if args.validation_epochs > 0 and num_update_steps_per_epoch > 0:
                        validation_steps = args.validation_epochs * num_update_steps_per_epoch
                        if global_step % validation_steps == 0 or global_step == args.max_train_steps:
                            logger.info(
                                f"Running validation at step {global_step} (Epoch {epoch+1})...")
                            unet = accelerator.unwrap_model(model)
                            unet.eval()

                            scheduler_config: dict = noise_scheduler.config
                            sampling_scheduler = DDPMScheduler(
                                **scheduler_config)
                            sampling_scheduler.set_timesteps(
                                args.num_inference_steps)

                            val_psnr_list = []
                            val_ssim_list = []
                            generated_images_pil = []
                            val_progress_bar = tqdm(
                                total=len(eval_dataloader), desc="Validation", leave=False, position=1)

                            for val_step, val_batch in enumerate(eval_dataloader):
                                low_light_images_val, clean_images_val = val_batch  # Renamed to avoid scope issues
                                # Renamed
                                batch_size_val = low_light_images_val.shape[0]

                                latents = torch.randn_like(
                                    clean_images_val, device=accelerator.device)
                                latents = latents * sampling_scheduler.init_noise_sigma

                                timesteps_to_iterate = sampling_scheduler.timesteps
                                # Renamed t
                                for t_val in tqdm(timesteps_to_iterate, leave=False, desc="Sampling", position=2):
                                    with torch.no_grad():
                                        model_input_val = torch.cat(  # Renamed
                                            [latents, low_light_images_val], dim=1)
                                        timestep_tensor_val = torch.tensor(  # Renamed
                                            [t_val] * batch_size_val, device=accelerator.device).long()
                                        noise_pred_val = unet(
                                            model_input_val, timestep_tensor_val).sample

                                        current_timestep_val = int(  # Renamed
                                            t_val.item() if isinstance(t_val, torch.Tensor) else t_val)
                                        step_output = sampling_scheduler.step(
                                            noise_pred_val, current_timestep_val, latents)

                                        if isinstance(step_output, DDPMSchedulerOutput):
                                            latents = step_output.prev_sample
                                        else:
                                            logger.warning(
                                                f"Unexpected type from scheduler step: {type(step_output)}. Using noise_pred as fallback.")
                                            latents = noise_pred_val

                                enhanced_images = latents
                                enhanced_images_0_1 = (
                                    enhanced_images / 2 + 0.5).clamp(0, 1)
                                clean_images_0_1_val = (
                                    clean_images_val / 2 + 0.5).clamp(0, 1)  # Renamed

                                try:
                                    current_psnr = peak_signal_noise_ratio(
                                        enhanced_images_0_1.cpu(), clean_images_0_1_val.cpu()).item()
                                    current_ssim = ssim(
                                        enhanced_images_0_1.cpu(), clean_images_0_1_val.cpu()).item()
                                    val_psnr_list.append(current_psnr)
                                    val_ssim_list.append(current_ssim)
                                    val_progress_bar.set_postfix(
                                        PSNR=f"{current_psnr:.2f}", SSIM=f"{current_ssim:.4f}")
                                except Exception as e:
                                    logger.error(f"计算指标时出错: {e}")
                                    val_psnr_list.append(float('nan'))
                                    val_ssim_list.append(float('nan'))

                                if len(generated_images_pil) < args.num_validation_images:
                                    num_to_save = min(
                                        batch_size_val, args.num_validation_images - len(generated_images_pil))
                                    for i in range(num_to_save):
                                        enhanced_pil = transforms.ToPILImage()(
                                            enhanced_images_0_1[i].cpu())
                                        low_light_pil = transforms.ToPILImage()(
                                            (low_light_images_val[i].cpu() / 2 + 0.5).clamp(0, 1))
                                        clean_pil = transforms.ToPILImage()(
                                            clean_images_0_1_val[i].cpu())

                                        total_width = low_light_pil.width * 3
                                        max_height = low_light_pil.height
                                        combined_image = Image.new(
                                            'RGB', (total_width, max_height))
                                        combined_image.paste(
                                            low_light_pil, (0, 0))
                                        combined_image.paste(
                                            enhanced_pil, (low_light_pil.width, 0))
                                        combined_image.paste(
                                            clean_pil, (low_light_pil.width * 2, 0))
                                        generated_images_pil.append(
                                            combined_image)
                                val_progress_bar.update(1)
                            val_progress_bar.close()

                            valid_psnr = [
                                p for p in val_psnr_list if not math.isnan(p)]
                            valid_ssim = [
                                s for s in val_ssim_list if not math.isnan(s)]
                            avg_psnr = sum(valid_psnr) / \
                                len(valid_psnr) if valid_psnr else 0.0
                            avg_ssim = sum(valid_ssim) / \
                                len(valid_ssim) if valid_ssim else 0.0

                            logger.info(
                                f"Step {global_step} Validation Results: Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}")

                            metrics_to_log = {
                                "val_psnr": avg_psnr, "val_ssim": avg_ssim}
                            tracker_key = "validation_enhanced_samples"
                            accelerator.log(metrics_to_log, step=global_step)

                            if generated_images_pil:
                                try:
                                    accelerator.log(
                                        {tracker_key: generated_images_pil}, step=global_step)
                                    logger.info(
                                        f"验证样本图像已记录到 tracker ({args.report_to})")
                                except Exception as e:
                                    logger.warning(f"无法将验证图像记录到 tracker: {e}")

                                sample = os.path.join(
                                    args.output, "validation_samples")
                                os.makedirs(sample, exist_ok=True)
                                for idx, img in enumerate(generated_images_pil):
                                    save_filename = os.path.join(
                                        sample, f"epoch-{epoch+1}_step-{global_step}_sample-{idx}.png")
                                    try:
                                        img.save(save_filename)
                                    except Exception as save_err:
                                        logger.error(
                                            f"保存验证样本图像失败 {save_filename}: {save_err}")
                                logger.info(f"验证样本图像已保存到本地目录 {sample}")

                            del unet
                            torch.cuda.empty_cache()
                            model.train()
                    elif global_step == args.max_train_steps:  # Ensure validation runs at the very end if not triggered by steps
                        logger.warning(
                            "Validation step calculation issue or validation_epochs <= 0. Running validation at the final step if it hasn't run.")
                        # Consider refactoring validation logic into a function to call here
                        pass

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save = os.path.join(
                        args.output, f"checkpoint-{global_step}")
                    accelerator.save_state(save)
                    logger.info(f"已保存检查点到 {save}")
                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [d for d in os.listdir(args.output) if d.startswith(
                                "checkpoint") and os.path.isdir(os.path.join(args.output, d))],
                            key=lambda x: int(x.split('-')[1])
                        )
                        if len(ckpts) > args.checkpoints_total_limit:
                            num_to_remove = len(ckpts) - \
                                args.checkpoints_total_limit
                            for old_ckpt in ckpts[:num_to_remove]:
                                old_ckpt_path = os.path.join(
                                    args.output, old_ckpt)
                                if os.path.isdir(old_ckpt_path):
                                    import shutil
                                    try:
                                        shutil.rmtree(old_ckpt_path)
                                        logger.info(
                                            f"已删除旧检查点: {old_ckpt_path}")
                                    except OSError as e:
                                        logger.error(
                                            f"删除检查点失败 {old_ckpt_path}: {e}")

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
        if global_step >= args.max_train_steps:
            logger.info("达到最大训练步数，停止训练。")
            break

    accelerator.end_training()
    logger.info("训练完成")

    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(model)
        save_path_final = os.path.join(args.output, "unet_final")
        unet.save_pretrained(save_path_final)
        logger.info(f"最终模型已保存到 {save_path_final}")


if __name__ == "__main__":
    main()
