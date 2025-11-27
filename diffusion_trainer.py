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

import diffusers
from diffusers import UNet2DModel
import torch.nn as nn
# 引入 DDPMScheduler 和 DPMSolver (用于加速验证)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.training_utils import EMAModel  # 引入 EMA

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from torchvision import transforms
from torcheval.metrics.functional import peak_signal_noise_ratio

from utils.misic import ssim, Save_path
# from models.common import ImageEncoder # 移除 ImageEncoder，因为改用 Concat
from datasets.data_set import LowLightDataset
import lpips

# 检查 diffusers 版本
check_min_version("0.10.0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Improved diffusion model training script for low-light enhancement.")
    parser.add_argument(
        "--data_dir", type=str, default="../datasets/kitti_LOL", help="数据集根目录"
    )
    parser.add_argument(
        "--output_dir", type=str, default="run_diffusion_improved", help="所有输出根目录"
    )
    parser.add_argument("--overwrite_output_dir",
                        action="store_true", help="是否覆盖输出目录")
    parser.add_argument("--seed", type=int,
                        default=random.randint(0, 1000000), help="随机种子")
    parser.add_argument(
        "--resolution", type=int, default=256, help="输入图像分辨率 (Random Crop size)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps", type=int, default=None, help="覆盖 epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="启用梯度检查点"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="学习率 (建议比原来稍低，因为结构变了)"
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant_with_warmup",
        choices=["linear", "cosine", "constant", "constant_with_warmup"], help="LR Scheduler"
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="LR Warmup"
    )
    parser.add_argument("--use_ema", action="store_true",
                        default=True, help="是否使用 EMA 模型 (默认推荐开启)")
    parser.add_argument("--ema_decay", type=float,
                        default=0.9999, help="EMA 衰减率")
    parser.add_argument(
        "--mixed_precision", type=str, default='fp16', choices=["no", "fp16", "bf16"],
        help="Mixed precision"
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="启用 xformers"
    )
    parser.add_argument(
        "--checkpointing_steps", type=int, default=5000, help="保存间隔"
    )
    parser.add_argument(
        "--checkpoints_total_limit", type=int, default=5, help="保留 Checkpoint 数量"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint 恢复路径"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Dataloader workers"
    )
    # === UNet 结构参数 (针对 Concat 优化) ===
    parser.add_argument(
        "--unet_layers_per_block", type=int, default=2, help="ResNet layers per block"
    )
    parser.add_argument(
        "--unet_block_channels", nargs='+', type=int, default=[64, 128, 128, 256, 256, 512],
        help="UNet channels"
    )
    # [改进] 引入 Attention Block 到深层，增强全局上下文建模能力
    parser.add_argument(
        "--unet_down_block_types", nargs='+', type=str,
        default=["DownBlock2D", "DownBlock2D", "DownBlock2D",
                 "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
        help="UNet Down Blocks (深层使用 AttnDownBlock2D)"
    )
    parser.add_argument(
        "--unet_up_block_types", nargs='+', type=str,
        default=["AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                 "UpBlock2D", "UpBlock2D", "UpBlock2D"],
        help="UNet Up Blocks (深层使用 AttnUpBlock2D)"
    )
    # ========================================
    parser.add_argument(
        "--num_inference_steps", type=int, default=20, help="验证时的采样步数 (使用 DPM-Solver 可以更少)"
    )
    parser.add_argument(
        "--prediction_type", type=str, default="v_prediction", choices=["epsilon", "v_prediction"],
        help="预测目标类型"
    )
    parser.add_argument(
        "--num_validation_images", type=int, default=4, help="验证生成数量"
    )
    parser.add_argument(
        "--validation_epochs", type=int, default=5, help="验证频率"
    )
    parser.add_argument(
        "--lambda_lpips", type=float, default=0.1, help="LPIPS 权重 (仅用于验证指标)"
    )
    parser.add_argument(
        "--lambda_fft", type=float, default=0.1, help="FFT 频域损失权重"
    )
    parser.add_argument(
        "--lambda_edge", type=float, default=0.5, help="Sobel 边缘损失权重 (针对 BEV 结构保持)"
    )
    parser.add_argument(
        "--report_to", type=str, default="tensorboard", help="日志报告目标"
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=5.0, help="Min-SNR Gamma"
    )

    args = parser.parse_args()
    return args


# === 辅助函数：SNR 计算 ===
def compute_snr(noise_scheduler, timesteps):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    if alphas_cumprod.device != timesteps.device:
        alphas_cumprod = alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
    return snr


def compute_min_snr_loss_weights(noise_scheduler, timesteps, snr_gamma=5.0):
    snr = compute_snr(noise_scheduler, timesteps)
    prediction_type = getattr(noise_scheduler.config,
                              "prediction_type", "epsilon")

    if prediction_type == "v_prediction":
        min_snr = torch.stack([snr, torch.ones_like(
            snr) * snr_gamma], dim=1).min(dim=1)[0]
        weights = min_snr / (snr + 1)
    elif prediction_type == "epsilon":
        gamma_over_snr = snr_gamma / (snr + 1e-8)
        weights = torch.stack(
            [torch.ones_like(gamma_over_snr), gamma_over_snr], dim=1).min(dim=1)[0]
    else:
        raise ValueError(f"不支持的预测类型: {prediction_type}")
    return weights.detach()

# === 辅助函数：Loss ===


def charbonnier_loss_elementwise(pred, target, eps=1e-3):
    """
    计算逐元素的 Charbonnier Loss (L1 的平滑变体)
    公式: sqrt((x-y)^2 + eps^2)
    """
    return torch.sqrt((pred - target)**2 + eps**2)


def fft_loss(pred, target):
    """
    计算频域损失 (FFT Loss)，用于强化边缘和高频细节恢复
    """
    pred_fft = torch.fft.rfft2(pred)
    target_fft = torch.fft.rfft2(target)
    return (pred_fft - target_fft).abs().mean()


def sobel_loss(pred, target):
    """
    计算 Sobel 边缘损失，增强几何结构稳定性 (针对 BEV 任务优化)
    """
    # 定义 Sobel 算子 (保持 device 一致)
    # shape: (out_channels, in_channels, kernel_size, kernel_size) -> (1, 1, 3, 3)
    # 这里假设输入是 RGB (batch, 3, H, W)，我们对每个通道分别计算然后平均
    device = pred.device
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=pred.dtype, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=pred.dtype, device=device).view(1, 1, 3, 3)

    # 为了适应 conv2d，reshape weight: output=3, input=1 (grouped conv)
    sobel_x = sobel_x.repeat(3, 1, 1, 1)
    sobel_y = sobel_y.repeat(3, 1, 1, 1)

    pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=3)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=3)
    pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)

    target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=3)
    target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=3)
    target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)

    return F.l1_loss(pred_grad, target_grad)

# ==============================


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        Save_path(args.output_dir)

    # === 1. 初始化模型 (Concat Architecture) ===
    logger.info(
        "Initializing UNet for Concat Conditioning (Input Channels=6)...")

    # 注意：不再需要 ImageEncoder
    model = UNet2DModel(
        sample_size=args.resolution,
        # 关键修改：输入通道 = 3 (Noisy) + 3 (Low Light Condition)
        in_channels=6,
        out_channels=3,
        layers_per_block=args.unet_layers_per_block,
        block_out_channels=args.unet_block_channels,
        down_block_types=tuple(args.unet_down_block_types),
        up_block_types=tuple(args.unet_up_block_types)
    )

    # === 2. 初始化 EMA 模型 ===
    if args.use_ema:
        logger.info("Initializing EMA model...")
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_decay,
            model_cls=UNet2DModel,
            model_config=model.config,
        )
        ema_model.to(accelerator.device)

    model.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # 初始化噪声调度器 (支持 v_prediction 配置)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type=args.prediction_type
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-08
    )

    # === 3. 数据增强优化 (Random Crop) ===
    # 训练：随机裁剪 + 翻转 (保留细节)
    train_preprocess = transforms.Compose([
        # 如果图像非常大，可以先 Resize 到稍大的尺寸，再 Crop
        # 这里假设直接 RandomCrop
        transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # 验证：Center Crop 或 Resize (保证尺寸一致性)
    eval_preprocess = transforms.Compose([
        transforms.Resize(args.resolution),  # 或者 CenterCrop
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    try:
        train_dataset = LowLightDataset(
            image_dir=args.data_dir, img_size=args.resolution, phase="train")
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        eval_dataset = LowLightDataset(
            image_dir=args.data_dir, img_size=args.resolution, phase="test")
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
        )
    except Exception as e:
        logger.error(f"Dataset init failed: {e}")
        return

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps if args.max_train_steps else len(
            train_dataloader) * args.epochs * args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Accelerate Tracker Init
    if accelerator.is_main_process:
        run_name = Path(args.output_dir).name
        # 创建配置的副本，以免修改原始 args
        config_to_log = vars(args).copy()

        # 遍历所有参数，如果是列表或元组，就转成字符串
        for key, value in config_to_log.items():
            if isinstance(value, (list, tuple)):
                config_to_log[key] = str(value)
        # 传入处理后的 config_to_log
        accelerator.init_trackers(run_name, config=config_to_log)

    # Calculate steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch
    else:
        args.epochs = math.ceil(args.max_train_steps /
                                num_update_steps_per_epoch)

    # Resume logic
    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume:
        if args.resume == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        else:
            path = os.path.basename(args.resume)

        if path is None:
            logger.info(
                f"Checkpoint '{args.resume}' not found. Starting a new training run."
            )
            args.resume = None
        else:
            # The checkpoint path expected by accelerator.load_state is the folder path
            if os.path.isabs(args.resume) and args.resume != "latest":
                checkpoint_path = args.resume
            else:
                checkpoint_path = os.path.join(args.output_dir, path)

            if not os.path.exists(checkpoint_path):
                logger.info(
                    f"Checkpoint '{checkpoint_path}' does not exist. Starting a new training run.")
                args.resume = None
            else:
                logger.info(f"Resuming from checkpoint {checkpoint_path}")
                accelerator.load_state(checkpoint_path)

                # Restore global_step
                try:
                    global_step = int(path.split("-")[-1])
                except ValueError:
                    global_step = 0

                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = global_step % num_update_steps_per_epoch

                # Restore EMA
                if args.use_ema:
                    ema_path = os.path.join(checkpoint_path, "ema_model.pth")
                    if os.path.exists(ema_path):
                        logger.info(f"Loading EMA model from {ema_path}")
                        ema_model.load_state_dict(torch.load(
                            ema_path, map_location=accelerator.device))
                    else:
                        logger.warning(f"EMA model not found at {ema_path}")

    # LPIPS Init (仅在验证时计算指标，避免训练开销，但模型需加载)
    if accelerator.is_main_process:
        logger.info("Initializing LPIPS (for validation)...")
    try:
        # 确保 lpips_model 在正确的设备上
        lpips_model = lpips.LPIPS(net='alex').to(accelerator.device).eval()
    except:
        logger.warning("LPIPS init failed, skipping LPIPS metric.")
        lpips_model = None

    # === 训练循环 ===
    logger.info("***** Running training *****")
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)

    for epoch in range(first_epoch, args.epochs):
        model.train()
        train_loss = 0.0

        if args.resume and epoch == first_epoch and resume_step > 0:
            # We need to skip the first `resume_step` batches
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            low_light_images, clean_images = batch

            # 采样噪声---高斯改进
            # 1. 生成标准高斯噪声
            noise = torch.randn_like(clean_images)

            # 2. 生成偏移量 (Batch, C, 1, 1)
            # 0.1 是推荐的偏移强度，太大会破坏分布
            offset_noise = torch.randn(
                clean_images.shape[0], clean_images.shape[1], 1, 1, device=clean_images.device)

            # 3. 将偏移加到噪声中
            noise = noise + 0.1 * offset_noise
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add Noise
            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # === 关键修改：Concat Conditioning ===
                # 并在通道维度拼接: (Batch, 6, H, W)
                model_input = torch.cat(
                    [noisy_images, low_light_images], dim=1)

                # 预测 (注意：encoder_hidden_states=None，因为我们用了 concat)
                model_output = model(model_input, timesteps).sample

                # 确定 Target (Epsilon 或 V-Prediction)
                if args.prediction_type == "epsilon":
                    target = noise
                elif args.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        clean_images, noise, timesteps)

                # === 关键修改：Min-SNR + Charbonnier Loss ===
                # 1. 计算权重
                snr_weights = compute_min_snr_loss_weights(
                    noise_scheduler, timesteps, args.snr_gamma)

                # 2. 计算 Element-wise Charbonnier Loss (替代 MSE)
                loss_elementwise = charbonnier_loss_elementwise(
                    model_output, target)

                # 3. 空间平均 (Batch, C, H, W) -> (Batch,)
                loss_per_sample = loss_elementwise.mean(dim=[1, 2, 3])

                # 4. 加权平均
                loss_main = (loss_per_sample * snr_weights).mean()

                # 5. [改进] 频域损失 (FFT Loss) & [新增] 边缘损失 (Sobel Loss)
                # 需要先恢复出 x0 (pred_original_sample)
                alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].view(
                    -1, 1, 1, 1)
                beta_prod_t = 1 - alpha_prod_t

                if args.prediction_type == "epsilon":
                    # x0 = (x_t - sqrt(1-alpha) * eps) / sqrt(alpha)
                    pred_original_sample = (
                        noisy_images - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
                elif args.prediction_type == "v_prediction":
                    # x0 = alpha * x_t - sigma * v
                    pred_original_sample = alpha_prod_t ** 0.5 * \
                        noisy_images - beta_prod_t ** 0.5 * model_output
                else:
                    pred_original_sample = None

                loss_fft_val = torch.tensor(0.0, device=accelerator.device)
                loss_edge_val = torch.tensor(0.0, device=accelerator.device)

                if pred_original_sample is not None:
                    # FFT Loss
                    if args.lambda_fft > 0:
                        loss_fft_val = fft_loss(
                            pred_original_sample, clean_images) * args.lambda_fft

                    # Sobel Edge Loss (针对 BEV 优化)
                    if args.lambda_edge > 0:
                        loss_edge_val = sobel_loss(
                            pred_original_sample, clean_images) * args.lambda_edge

                # LPIPS 在训练中禁用以节省开销，仅在验证中使用
                loss = loss_main + loss_fft_val + loss_edge_val

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 5.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                # === 关键修改：更新 EMA 模型 ===
                if args.use_ema:
                    ema_model.step(model.parameters())

                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss": loss_main.item(),
                        "loss_fft": loss_fft_val.item(),
                        "loss_edge": loss_edge_val.item(),
                        "lr": lr_scheduler.get_last_lr()[0]
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                # === 验证逻辑 (使用 EMA + DPMSolver + 详细指标计算) ===
                if accelerator.is_main_process and (global_step % (args.validation_epochs * num_update_steps_per_epoch) == 0):
                    logger.info("Running validation with EMA model...")

                    # 1. 获取验证模型 (EMA 或 原始)
                    if args.use_ema:
                        # 将 EMA 权重暂存到 unet 中进行推理
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())

                    unet_val = accelerator.unwrap_model(model)
                    unet_val.eval()

                    # 2. 使用 DPM-Solver 加速验证
                    val_scheduler = DPMSolverMultistepScheduler.from_config(
                        noise_scheduler.config)
                    val_scheduler.set_timesteps(
                        args.num_inference_steps)  # 步数少，速度快

                    # 准备验证结果保存目录
                    val_save_dir = os.path.join(
                        args.output_dir, "validation", f"step_{global_step}")
                    os.makedirs(val_save_dir, exist_ok=True)

                    total_psnr = 0.0
                    total_ssim = 0.0
                    total_lpips = 0.0
                    num_val_samples = 0

                    with torch.no_grad():
                        for val_idx, (val_low, val_clean) in enumerate(eval_dataloader):
                            if val_idx >= args.num_validation_images // args.batch_size:
                                break  # 限制验证数量

                            val_low = val_low.to(accelerator.device)
                            val_clean = val_clean.to(accelerator.device)
                            latents = torch.randn_like(val_low)

                            # DPM-Solver Init Noise Scaling
                            latents = latents * val_scheduler.init_noise_sigma

                            for t in tqdm(val_scheduler.timesteps, desc="DPM Sampling", leave=False):
                                # 构造输入: Concat
                                latent_model_input = val_scheduler.scale_model_input(
                                    latents, t)
                                model_input = torch.cat(
                                    [latent_model_input, val_low], dim=1)

                                noise_pred = unet_val(
                                    model_input, t).sample
                                latents = val_scheduler.step(
                                    noise_pred, t, latents).prev_sample

                            # 后处理: [-1, 1] -> [0, 1]
                            enhanced_images = (latents / 2 + 0.5).clamp(0, 1)
                            gt_images = (val_clean / 2 + 0.5).clamp(0, 1)
                            low_images = (val_low / 2 + 0.5).clamp(0, 1)

                            # 计算指标
                            # PSNR
                            # batch mean PSNR
                            batch_psnr = peak_signal_noise_ratio(
                                enhanced_images, gt_images, data_range=1.0)
                            total_psnr += batch_psnr.item() * val_low.shape[0]

                            # SSIM (使用 utils.misic 中的 ssim，或者 torchmetrics)
                            # 假设 ssim 接收 (N, C, H, W)
                            batch_ssim = ssim(
                                enhanced_images, gt_images).item()
                            total_ssim += batch_ssim * val_low.shape[0]

                            # LPIPS (如果已加载)
                            if lpips_model is not None:
                                # LPIPS model expects input in [-1, 1] usually, but check specific implementation.
                                # standard lpips expects [-1, 1].
                                # enhanced_images is [0, 1], so convert back to [-1, 1] for lpips
                                enh_norm = enhanced_images * 2.0 - 1.0
                                gt_norm = gt_images * 2.0 - 1.0
                                batch_lpips = lpips_model(
                                    enh_norm, gt_norm).mean().item()
                                total_lpips += batch_lpips * val_low.shape[0]

                            num_val_samples += val_low.shape[0]

                            # 保存图片 (仅保存第一个 batch 的前几张)
                            if val_idx == 0:
                                for i in range(min(4, enhanced_images.shape[0])):
                                    grid = torch.cat(
                                        [low_images[i], enhanced_images[i], gt_images[i]], dim=2)  # W 维度拼接
                                    transforms.ToPILImage()(grid).save(
                                        os.path.join(val_save_dir, f"val_{val_idx}_{i}.png"))

                    # 平均指标
                    if num_val_samples > 0:
                        avg_psnr = total_psnr / num_val_samples
                        avg_ssim = total_ssim / num_val_samples
                        avg_lpips = total_lpips / num_val_samples

                        logger.info(
                            f"Validation Step {global_step}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
                        accelerator.log({
                            "val/psnr": avg_psnr,
                            "val/ssim": avg_ssim,
                            "val/lpips": avg_lpips
                        }, step=global_step)

                    # 3. 恢复原始权重
                    if args.use_ema:
                        ema_model.restore(model.parameters())

                    unet_val.train()

                # === 保存 Checkpoint ===
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    # 如果使用了 EMA，建议保存 EMA 状态
                    if args.use_ema:
                        torch.save(ema_model.state_dict(), os.path.join(
                            save_path, "ema_model.pth"))

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    accelerator.end_training()

    # === 保存最终模型 ===
    if accelerator.is_main_process:
        if args.use_ema:
            logger.info("Saving EMA model as final model...")
            ema_model.copy_to(model.parameters())

        unet = accelerator.unwrap_model(model)
        unet.save_pretrained(os.path.join(args.output_dir, "unet_final"))
        logger.info("Training Finished.")


if __name__ == "__main__":
    main()
