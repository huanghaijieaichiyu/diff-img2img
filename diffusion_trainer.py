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
        "--data_dir", type=str, default="../datasets/kitti_LOL", help="数据集根目录"
    )
    parser.add_argument(
        "--output_dir", type=str, default="run_diffusion", help="所有输出 (模型, 日志等) 的根目录"
    )
    parser.add_argument("--overwrite_output_dir",
                        action="store_true", help="是否覆盖输出目录")
    parser.add_argument("--cache_dir", type=str, default=None, help="缓存目录")
    parser.add_argument("--seed", type=int,
                        default=random.randint(0, 1000000), help="随机种子")
    parser.add_argument(
        "--resolution", type=int, default=256, help="输入图像分辨率"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="训练和评估的批次大小"
    )
    parser.add_argument("--epochs", type=int, default=50)
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
        "--lambda_lpips", type=float, default=0.5, help="LPIPS 损失的权重"  # <-- 添加 lambda_lpips 参数
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

    # 日志目录固定在 output_dir 下的 'logs'
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
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

    # 在 Accelerator 初始化后记录轻量化配置信息
    if args.lightweight_unet:
        logger.info("使用轻量化 UNet 配置...")

    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"设置随机种子为: {args.seed}")  # 添加日志确认

    # 创建输出目录
    if accelerator.is_main_process:
        save_path(args.output_dir)
        logger.info(f"Overwriting output directory {args.output_dir}")

    # 初始化模型 - 使用解析后的参数
    logger.info("Initializing UNet model...")
    logger.info(f"  Layers per block: {args.unet_layers_per_block}")
    logger.info(f"  Block channels: {args.unet_block_channels}")
    logger.info(f"  Down blocks: {args.unet_down_block_types}")
    logger.info(f"  Up blocks: {args.unet_up_block_types}")
    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=6,  # 修改这里: 3 (noisy) + 3 (condition)
        out_channels=3,  # 预测噪声还是 3 通道
        layers_per_block=args.unet_layers_per_block,
        block_out_channels=args.unet_block_channels,
        down_block_types=args.unet_down_block_types,
        up_block_types=args.unet_up_block_types,
    )

    model.enable_xformers_memory_efficient_attention()
    logger.info("启用 xformers 内存高效注意力")

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # 初始化噪声调度器
    # DDPM 常用 1000 步, 尝试不同的 schedule
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

    # 数据处理 (保持不变，确保输出在 [-1, 1])
    preprocess = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    eval_preprocess = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # 创建数据集和数据加载器
    try:
        train_dataset = LowLightDataset(
            image_dir=args.data_dir, transform=preprocess, phase="train")
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        eval_dataset = LowLightDataset(
            image_dir=args.data_dir, transform=eval_preprocess, phase="test")
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
        )
    except Exception as e:
        logger.error(
            f"加载数据集失败，请检查路径 '{args.data_dir}' 和 LowLightDataset 实现: {e}")
        return  # 无法继续，退出

    # 初始化学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps if args.max_train_steps else len(
            train_dataloader) * args.epochs * args.gradient_accumulation_steps,
    )

    # 使用 Accelerate 准备
    # 注意 noise_scheduler 不需要 prepare
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler  # 添加 eval_dataloader
    )

    # 初始化追踪器 (tensorboard)
    if accelerator.is_main_process:
        run_name = Path(
            args.output_dir).name if args.output_dir else "diffusion_conditional_run"

        # 创建配置字典副本，并转换列表为字符串以兼容 TensorBoard hparams
        config_to_log = vars(args).copy()
        for key in ["unet_block_channels", "unet_down_block_types", "unet_up_block_types"]:
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

    # 断点续训逻辑 (基本保持不变)
    total_batch_size = args.batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps
    global_step = 0
    first_epoch = 0

    if args.resume:
        if args.resume != "latest":
            path = os.path.basename(args.resume)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume}' does not exist. Starting a new training run."
            )
            args.resume = None
        else:
            checkpoint_path = os.path.join(args.output_dir, path)
            accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
            # 使用 load_state 时，模型、优化器、调度器等的状态会自动恢复
            # 但需要手动恢复 global_step 和 epoch
            try:
                accelerator.load_state(checkpoint_path)
                # 假设 checkpoint 目录名包含 step 数
                global_step = int(path.split("-")[1])
                accelerator.print(
                    f"Successfully loaded state from {checkpoint_path}. Resuming global step {global_step}.")
            except Exception as e:
                accelerator.print(
                    f"Failed to load state from {checkpoint_path}: {e}. Starting from scratch.")
                global_step = 0  # 加载失败，从头开始

            # resume_global_step = global_step  # 修正：global_step 已经是优化器步数
            first_epoch = global_step // num_update_steps_per_epoch
            # resume_step 不再需要以这种方式计算，因为 dataloader 不会被跳过
            # accelerator.load_state 会处理学习率调度器的状态

    # === 初始化 LPIPS 模型 ===
    # 只在主进程打印加载信息，但所有进程都需要加载模型
    if accelerator.is_main_process:
        logger.info("Initializing LPIPS model...")
    try:
        # 使用预训练的 AlexNet
        lpips_model = lpips.LPIPS(net='alex').to(accelerator.device)
        # LPIPS 模型不需要训练
        lpips_model.eval()
        # 确保在混合精度下正确运行 (通常 LPIPS 在 fp32 下计算)
        # 不需要 accelerator.prepare，因为它不参与梯度计算和优化
        logger.info("LPIPS model initialized successfully.")
    except Exception as e:
        logger.error(
            f"Failed to initialize LPIPS model: {e}. LPIPS loss will not be used.")
        lpips_model = None  # 设置为 None，后续逻辑会跳过 LPIPS 计算
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

    # ==========================\n    # === 开始训练循环 ===\n    # ==========================
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.epochs):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            # low_light_images: 条件输入, clean_images: 目标 (Ground Truth)
            low_light_images, clean_images = batch

            # 采样噪声添加到干净图像
            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            # 为批次中的每个图像采样随机时间步
            # 确保 timesteps 是 LongTensor
            timesteps = torch.randint(
                0, noise_scheduler.config["num_train_timesteps"], (bsz,), device=clean_images.device
            ).long()

            # 根据噪声和时间步将噪声添加到干净图像中，得到噪声图像
            # Revert to .long() for timesteps as it's generally accepted
            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps  # 使用 .long()
            )

            with accelerator.accumulate(model):
                # 准备模型输入：拼接噪声图像和条件图像
                model_input = torch.cat(
                    [noisy_images, low_light_images], dim=1)

                # 预测噪声残差
                noise_pred = model(model_input, timesteps).sample

                # 1. 计算 MSE 损失 (预测噪声和实际添加噪声之间的 MSE)
                loss_mse = F.mse_loss(noise_pred.float(),
                                      noise.float())  # 确保 float 类型

                # 2. 计算 LPIPS 损失 (如果 LPIPS 模型成功加载)
                loss_lpips = torch.tensor(0.0).to(accelerator.device)  # 初始化为 0
                if lpips_model is not None and args.lambda_lpips > 0:
                    # 需要估算去噪后的图像 pred_x0
                    # 使用 scheduler 的 alphas_cumprod
                    # 确保 alphas_cumprod 在正确的设备上并且形状匹配
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
                        timesteps.device)
                    sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt(
                    ).view(-1, 1, 1, 1)
                    sqrt_one_minus_alpha_prod = (
                        1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)

                    # 计算 pred_x0
                    # pred_x0 = (noisy_images - sqrt(1 - alpha_prod_t) * noise_pred) / sqrt(alpha_prod_t)
                    pred_x0 = (noisy_images - sqrt_one_minus_alpha_prod *
                               noise_pred) / (sqrt_alpha_prod + 1e-8)

                    # 将 pred_x0 和 clean_images clamp 到 [-1, 1] 以确保 LPIPS 输入范围正确
                    pred_x0_clamp = torch.clamp(pred_x0, -1.0, 1.0)
                    clean_images_clamp = torch.clamp(clean_images, -1.0, 1.0)

                    # 计算 LPIPS 损失
                    # LPIPS 模型通常在 fp32 下运行更稳定
                    loss_lpips = lpips_model(
                        pred_x0_clamp.float(), clean_images_clamp.float()).mean()

                # 3. 合并损失
                loss = loss_mse + args.lambda_lpips * loss_lpips

                # 收集损失用于日志记录 (收集总损失)
                # accelerator.gather 返回 tensor (主进程) 或 None (其他进程)
                gathered_loss = accelerator.gather(
                    loss.repeat(args.batch_size))
                # .mean() 应该在 tensor 上调用
                if gathered_loss is not None:  # 只有主进程计算 avg_loss
                    # 确保 gathered_loss 是 tensor
                    if isinstance(gathered_loss, torch.Tensor):
                        avg_loss = gathered_loss.mean()
                        train_loss += avg_loss.item() / args.gradient_accumulation_steps
                    else:
                        # 处理 gather 返回非 Tensor 的情况 (理论上不应发生)
                        logger.warning(
                            f"accelerator.gather returned unexpected type: {type(gathered_loss)}")
                        avg_loss = torch.tensor(0.0)  # 设置默认值
                else:
                    avg_loss = torch.tensor(0.0)  # 其他进程设为0，避免错误

                # 反向传播
                accelerator.backward(loss)

                # 梯度裁剪
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 检查是否是同步和更新步骤
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # 主进程记录日志
                if accelerator.is_main_process:
                    current_lr = lr_scheduler.get_last_lr(
                    )[0] if lr_scheduler else args.lr
                    # --- 修改日志内容 ---
                    logs = {
                        "loss": train_loss,  # 这是累积的总损失
                        # 当前 step 的 mse loss
                        "loss_mse": loss_mse.item() / args.gradient_accumulation_steps if accelerator.is_main_process else 0.0,
                        "loss_lpips": loss_lpips.item() / args.gradient_accumulation_steps if accelerator.is_main_process and lpips_model is not None and args.lambda_lpips > 0 else 0.0,  # 当前 step 的 lpips loss
                        "lr": current_lr,
                        "epoch": epoch + 1
                    }
                    postfix_dict = {
                        "loss": f"{train_loss:.4f}",  # 显示累积总损失
                        "mse": f"{logs['loss_mse']:.4f}",  # 显示当前 mse
                        "lpips": f"{logs['loss_lpips']:.4f}",  # 显示当前 lpips
                        "lr": f"{current_lr:.6f}",
                        "epoch": epoch + 1
                    }
                    progress_bar.set_postfix(**postfix_dict)
                    accelerator.log(logs, step=global_step)
                    # --- 结束修改 ---
                train_loss = 0.0  # 重置累积损失

                # 定期验证 (基于 global_step)
                if accelerator.is_main_process:
                    # 验证频率可以基于步数，更精确
                    # 确保 validation_epochs > 0 且 num_update_steps_per_epoch > 0
                    if args.validation_epochs > 0 and num_update_steps_per_epoch > 0:
                        validation_steps = args.validation_epochs * num_update_steps_per_epoch
                        if global_step % validation_steps == 0 or global_step == args.max_train_steps:
                            logger.info(
                                f"Running validation at step {global_step} (Epoch {epoch+1})...")
                            unet = accelerator.unwrap_model(model)  # 获取原始模型
                            unet.eval()  # 设置为评估模式

                            # 获取未包装的 scheduler config
                            # noise_scheduler 是 accelerate prepare 之前的原始对象
                            scheduler_config: dict = noise_scheduler.config  # 添加类型提示
                            # 显式实例化验证调度器
                            sampling_scheduler = DDPMScheduler(
                                **scheduler_config)

                            # 设置采样步数
                            sampling_scheduler.set_timesteps(
                                args.num_inference_steps)

                            val_psnr_list = []
                            val_ssim_list = []
                            generated_images_pil = []
                            # 使用 eval_dataloader 进行验证
                            val_progress_bar = tqdm(
                                total=len(eval_dataloader), desc="Validation", leave=False, position=1)

                            for val_step, val_batch in enumerate(eval_dataloader):
                                low_light_images, clean_images = val_batch
                                batch_size = low_light_images.shape[0]

                                # 初始化随机噪声 latents
                                latents = torch.randn_like(
                                    clean_images, device=accelerator.device)
                                # scale the initial noise by the standard deviation required by the scheduler
                                latents = latents * sampling_scheduler.init_noise_sigma

                                # 条件采样循环
                                # 使用 sampling_scheduler 的时间步
                                timesteps_to_iterate = sampling_scheduler.timesteps
                                for t in tqdm(timesteps_to_iterate, leave=False, desc="Sampling", position=2):
                                    with torch.no_grad():
                                        # 准备模型输入: [b, 6, H, W]
                                        # latents 可能需要扩展以匹配模型预期（如果需要）
                                        model_input = torch.cat(
                                            [latents, low_light_images], dim=1)
                                        # timestep 需要是 LongTensor
                                        timestep_tensor = torch.tensor(
                                            [t] * batch_size, device=accelerator.device).long()
                                        noise_pred = unet(
                                            model_input, timestep_tensor).sample
                                        # 使用调度器计算上一步的样本
                                        # scheduler.step 需要 int timestep
                                        # 将 t 转换为 int
                                        current_timestep = int(
                                            t.item() if isinstance(t, torch.Tensor) else t)
                                        # DDPMScheduler.step 返回 dataclass DDPMSchedulerOutput
                                        step_output = sampling_scheduler.step(
                                            noise_pred, current_timestep, latents)
                                        # Explicitly check type before accessing attribute
                                        if isinstance(step_output, DDPMSchedulerOutput):
                                            latents = step_output.prev_sample
                                        else:
                                            # Handle unexpected return type (e.g., log warning, use default)
                                            logger.warning(
                                                f"Unexpected type from scheduler step: {type(step_output)}. Using noise_pred as fallback.")
                                            # Fallback strategy might depend on the scheduler
                                            latents = noise_pred  # Or some other fallback

                                # latents 现在是增强后的图像 [-1, 1]
                                enhanced_images = latents

                                # 将图像转换回 [0, 1] 以计算指标和保存
                                enhanced_images_0_1 = (
                                    enhanced_images / 2 + 0.5).clamp(0, 1)
                                clean_images_0_1 = (
                                    clean_images / 2 + 0.5).clamp(0, 1)

                                # 计算 PSNR 和 SSIM (在 CPU 上计算更安全)
                                try:
                                    current_psnr = peak_signal_noise_ratio(
                                        enhanced_images_0_1.cpu(), clean_images_0_1.cpu()).item()
                                    current_ssim = ssim(
                                        enhanced_images_0_1.cpu(), clean_images_0_1.cpu()).item()
                                    val_psnr_list.append(current_psnr)
                                    val_ssim_list.append(current_ssim)
                                    val_progress_bar.set_postfix(
                                        PSNR=f"{current_psnr:.2f}", SSIM=f"{current_ssim:.4f}")
                                except Exception as e:
                                    logger.error(f"计算指标时出错: {e}")
                                    # 可以选择跳过这个 batch 或记录 NaN
                                    val_psnr_list.append(float('nan'))
                                    val_ssim_list.append(float('nan'))

                                # 保存前 N 个验证图像用于可视化
                                if len(generated_images_pil) < args.num_validation_images:
                                    # 从 batch 中安全地选择图像进行可视化
                                    num_to_save = min(
                                        batch_size, args.num_validation_images - len(generated_images_pil))
                                    for i in range(num_to_save):
                                        enhanced_pil = transforms.ToPILImage()(
                                            enhanced_images_0_1[i].cpu())
                                        low_light_pil = transforms.ToPILImage()(
                                            (low_light_images[i].cpu() / 2 + 0.5).clamp(0, 1))
                                        clean_pil = transforms.ToPILImage()(
                                            clean_images_0_1[i].cpu())
                                        # 拼接图像 (low, enhanced, clean)
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

                            # 计算平均指标 (过滤掉 NaN 值)
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

                            # 记录指标和图像
                            metrics_to_log = {
                                "val_psnr": avg_psnr, "val_ssim": avg_ssim}
                            # (low, enhanced, clean)
                            tracker_key = "validation_enhanced_samples"

                            # 使用 accelerator.log 记录指标
                            accelerator.log(metrics_to_log, step=global_step)

                            # 记录图像到 tracker (tensorboard 或其他)
                            if generated_images_pil:  # 确保有图像可记录
                                try:
                                    # 直接让 accelerate 处理 PIL 图像列表
                                    accelerator.log(
                                        {tracker_key: generated_images_pil},
                                        step=global_step
                                    )
                                    logger.info(
                                        f"验证样本图像已记录到 tracker ({args.report_to})")
                                except Exception as e:
                                    logger.warning(f"无法将验证图像记录到 tracker: {e}")
                                    # 如果 tracker 记录失败，仍保存到本地文件
                                    # 这个本地保存的逻辑现在会移到外面，无论如何都执行

                                # --- 新增：总是保存验证样本到本地 ---
                                sample_dir = os.path.join(
                                    args.output_dir, "validation_samples")
                                os.makedirs(sample_dir, exist_ok=True)
                                for idx, img in enumerate(generated_images_pil):
                                    # 使用更详细的文件名，包含 epoch 和 step
                                    save_filename = os.path.join(
                                        sample_dir, f"epoch-{epoch+1}_step-{global_step}_sample-{idx}.png")
                                    try:
                                        img.save(save_filename)
                                    except Exception as save_err:
                                        logger.error(
                                            f"保存验证样本图像失败 {save_filename}: {save_err}")
                                logger.info(f"验证样本图像已保存到本地目录 {sample_dir}")
                                # --- 结束新增代码 ---

                            # 清理 GPU 缓存
                            del unet  # 删除对 unwrap 模型的引用
                            torch.cuda.empty_cache()
                            model.train()  # 验证结束后切回训练模式
                    else:  # validation_epochs <= 0 or num_update_steps_per_epoch <= 0
                        # Ensure validation runs at the end if step-based validation is disabled/problematic
                        if global_step == args.max_train_steps:
                            logger.warning(
                                "Validation step calculation issue or validation_epochs <= 0. Running validation at the final step.")
                            # ... (复制粘贴上面的验证逻辑或将其重构为函数) ...
                            pass  # 避免重复代码，实际应用中应重构为一个函数

                # 定期保存检查点 (检查点逻辑移到验证之后，确保保存的是 train 模式的模型状态)
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_dir = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_dir)
                    logger.info(f"已保存检查点到 {save_dir}")
                    # 管理检查点数量
                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [d for d in os.listdir(
                                args.output_dir) if d.startswith("checkpoint") and os.path.isdir(os.path.join(args.output_dir, d))],  # 确保是目录
                            key=lambda x: int(x.split('-')[1])
                        )
                        if len(ckpts) > args.checkpoints_total_limit:
                            num_to_remove = len(
                                ckpts) - args.checkpoints_total_limit
                            for old_ckpt in ckpts[:num_to_remove]:  # 删除最旧的
                                old_ckpt_path = os.path.join(
                                    args.output_dir, old_ckpt)
                                # 确保删除的是目录
                                if os.path.isdir(old_ckpt_path):
                                    import shutil
                                    try:
                                        shutil.rmtree(old_ckpt_path)
                                        logger.info(
                                            f"已删除旧检查点: {old_ckpt_path}")
                                    except OSError as e:
                                        logger.error(
                                            f"删除检查点失败 {old_ckpt_path}: {e}")

            # 超过最大步数则停止
            if global_step >= args.max_train_steps:
                break

        # epoch 结束
        accelerator.wait_for_everyone()

        # # Epoch 结束时的验证逻辑（如果需要）
        # if accelerator.is_main_process and (epoch + 1) % args.validation_epochs == 0:
        #      # 这里可以放入基于 epoch 的验证逻辑，如果不想基于 step
        #      pass

        # 超过最大步数则停止外部循环
        if global_step >= args.max_train_steps:
            logger.info("达到最大训练步数，停止训练。")
            break

    # 训练结束
    accelerator.end_training()
    logger.info("训练完成")

    # 保存最终模型
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(model)
        # 保存为 diffusers 格式
        save_path_final = os.path.join(args.output_dir, "unet_final")
        unet.save_pretrained(save_path_final)
        logger.info(f"最终模型已保存到 {save_path_final}")


if __name__ == "__main__":
    main()
