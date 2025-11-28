import argparse
import os
import math
import logging
from pathlib import Path
import random
import itertools

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from torcheval.metrics.functional import peak_signal_noise_ratio

from diffusers import UNet2DModel, DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.training_utils import EMAModel

import lpips

from utils.misic import ssim, Save_path
from datasets.data_set import LowLightDataset
from models.retinex import DecomNet

# Verify diffusers version
check_min_version("0.10.0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Improved diffusion model training script for low-light enhancement."
    )
    # Dataset & Output
    parser.add_argument("--data_dir", type=str, default="../datasets/kitti_LOL", help="Root dataset directory")
    parser.add_argument("--output_dir", type=str, default="run_diffusion_improved", help="Output directory")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite output directory")
    parser.add_argument("--seed", type=int, default=random.randint(0, 1000000), help="Random seed")
    
    # Training Config
    parser.add_argument("--resolution", type=int, default=256, help="Input resolution (Random Crop size)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Override epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup",
                        choices=["linear", "cosine", "constant", "constant_with_warmup"], help="LR Scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="LR Warmup")
    parser.add_argument("--mixed_precision", type=str, default='fp16', choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Enable xformers")
    
    # EMA
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA model")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay")

    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=5000, help="Save interval")
    parser.add_argument("--checkpoints_total_limit", type=int, default=5, help="Max checkpoints to keep")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers")

    # UNet Architecture
    parser.add_argument("--unet_layers_per_block", type=int, default=2, help="ResNet layers per block")
    parser.add_argument("--unet_block_channels", nargs='+', type=int, default=[64, 128, 128, 256, 256, 512], help="UNet channels")
    parser.add_argument("--unet_down_block_types", nargs='+', type=str,
                        default=["DownBlock2D", "DownBlock2D", "DownBlock2D",
                                 "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
                        help="UNet Down Blocks")
    parser.add_argument("--unet_up_block_types", nargs='+', type=str,
                        default=["AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                                 "UpBlock2D", "UpBlock2D", "UpBlock2D"],
                        help="UNet Up Blocks")

    # Validation & Inference
    parser.add_argument("--num_inference_steps", type=int, default=20, help="DPM-Solver sampling steps")
    parser.add_argument("--prediction_type", type=str, default="v_prediction", choices=["epsilon", "v_prediction"], help="Prediction target")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of validation images")
    parser.add_argument("--validation_epochs", type=int, default=5, help="Validation frequency")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Logging target")
    
    # Loss Config
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="Min-SNR Gamma")
    
    # Aux Network
    parser.add_argument("--use_retinex", action="store_true", help="Use Retinex decomposition network")

    args = parser.parse_args()
    return args

# === SNR Helper Functions ===
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
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")

    if prediction_type == "v_prediction":
        min_snr = torch.stack([snr, torch.ones_like(snr) * snr_gamma], dim=1).min(dim=1)[0]
        weights = min_snr / (snr + 1)
    elif prediction_type == "epsilon":
        gamma_over_snr = snr_gamma / (snr + 1e-8)
        weights = torch.stack([torch.ones_like(gamma_over_snr), gamma_over_snr], dim=1).min(dim=1)[0]
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")
    return weights.detach()

def charbonnier_loss_elementwise(pred, target, eps=1e-3):
    """
    Element-wise Charbonnier Loss (Smoothed L1)
    sqrt((x-y)^2 + eps^2)
    """
    return torch.sqrt((pred - target)**2 + eps**2)


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

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

    # === 1. Initialize Models ===
    # Input channels: 3 (Noisy) + [3 (Reflectance) + 1 (Illumination) if Retinex else 3 (Low Light)]
    input_channels = 7 if args.use_retinex else 6
    
    logger.info(f"Initializing UNet (Input Channels={input_channels})...")
    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=input_channels,
        out_channels=3,
        layers_per_block=args.unet_layers_per_block,
        block_out_channels=args.unet_block_channels,
        down_block_types=tuple(args.unet_down_block_types),
        up_block_types=tuple(args.unet_up_block_types)
    )

    decom_model = None
    if args.use_retinex:
        logger.info("Initializing Retinex Decomposition Network...")
        decom_model = DecomNet().to(accelerator.device)
        decom_model.train()

    # EMA Model
    if args.use_ema:
        logger.info("Initializing EMA model...")
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_decay,
            model_cls=UNet2DModel,
            model_config=model.config,
        )
        ema_model.to(accelerator.device)

    if args.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()
    
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type=args.prediction_type
    )

    # Optimizer
    params_to_optimize = model.parameters()
    if decom_model is not None:
        params_to_optimize = itertools.chain(model.parameters(), decom_model.parameters())

    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=args.lr, weight_decay=1e-2, eps=1e-08
    )

    # Data Transforms (handled in Dataset, implicitly assumed compatible)
    
    try:
        train_dataset = LowLightDataset(image_dir=args.data_dir, img_size=args.resolution, phase="train")
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        eval_dataset = LowLightDataset(image_dir=args.data_dir, img_size=args.resolution, phase="test")
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
        )
    except Exception as e:
        logger.error(f"Dataset init failed: {e}")
        return

    # LR Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps if args.max_train_steps else len(train_dataloader) * args.epochs * args.gradient_accumulation_steps,
    )

    # Prepare with Accelerator
    if decom_model is not None:
        model, decom_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, decom_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

    # Trackers
    if accelerator.is_main_process:
        run_name = Path(args.output_dir).name
        config_to_log = vars(args).copy()
        for key, value in config_to_log.items():
            if isinstance(value, (list, tuple)):
                config_to_log[key] = str(value)
        accelerator.init_trackers(run_name, config=config_to_log)

    # Training Steps Calculation
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch
    else:
        args.epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Resume Training
    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume:
        if args.resume == "latest":
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        else:
            path = os.path.basename(args.resume)

        if path is None:
            logger.info(f"Checkpoint '{args.resume}' not found. Starting new run.")
            args.resume = None
        else:
            if os.path.isabs(args.resume) and args.resume != "latest":
                checkpoint_path = args.resume
            else:
                checkpoint_path = os.path.join(args.output_dir, path)

            if not os.path.exists(checkpoint_path):
                logger.info(f"Checkpoint '{checkpoint_path}' missing. Starting new run.")
                args.resume = None
            else:
                logger.info(f"Resuming from {checkpoint_path}")
                accelerator.load_state(checkpoint_path)
                try:
                    global_step = int(path.split("-")[-1])
                except ValueError:
                    global_step = 0
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = global_step % num_update_steps_per_epoch

                if args.use_ema:
                    ema_path = os.path.join(checkpoint_path, "ema_model.pth")
                    if os.path.exists(ema_path):
                        ema_model.load_state_dict(torch.load(ema_path, map_location=accelerator.device))
    
    # Init LPIPS for validation only
    if accelerator.is_main_process:
        logger.info("Initializing LPIPS (for validation)...")
    try:
        lpips_model = lpips.LPIPS(net='alex').to(accelerator.device).eval()
    except:
        logger.warning("LPIPS init failed, skipping LPIPS metric.")
        lpips_model = None

    # === Training Loop ===
    logger.info("***** Running training *****")
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(first_epoch, args.epochs):
        model.train()
        
        if args.resume and epoch == first_epoch and resume_step > 0:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            low_light_images, clean_images = batch

            # 1. Noise Injection (with Offset Noise for better contrast)
            noise = torch.randn_like(clean_images)
            offset_noise = torch.randn(clean_images.shape[0], clean_images.shape[1], 1, 1, device=clean_images.device)
            noise = noise + 0.1 * offset_noise
            
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device).long()
            
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # 2. Conditioning
                if decom_model is not None:
                    # Retinex Decomposition (R, I)
                    low_light_01 = (low_light_images / 2 + 0.5).clamp(0, 1)
                    r_low, i_low = decom_model(low_light_01)
                    
                    # Normalize [0,1] -> [-1,1]
                    r_low_norm = r_low * 2.0 - 1.0
                    i_low_norm = i_low * 2.0 - 1.0
                    
                    model_input = torch.cat([noisy_images, r_low_norm, i_low_norm], dim=1)
                else:
                    # Standard Concat: Noisy + LowLight
                    model_input = torch.cat([noisy_images, low_light_images], dim=1)

                # 3. Prediction
                model_output = model(model_input, timesteps).sample

                # 4. Target Calculation
                if args.prediction_type == "epsilon":
                    target = noise
                elif args.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {args.prediction_type}")

                # 5. Loss Calculation (Min-SNR + Charbonnier)
                snr_weights = compute_min_snr_loss_weights(noise_scheduler, timesteps, args.snr_gamma)
                
                # Element-wise Charbonnier
                loss_elementwise = charbonnier_loss_elementwise(model_output, target)
                
                # Weighted Mean
                loss = (loss_elementwise.mean(dim=[1, 2, 3]) * snr_weights).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 5.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())

                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss": loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0]
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                # === Validation ===
                if accelerator.is_main_process and (global_step % (args.validation_epochs * num_update_steps_per_epoch) == 0):
                    logger.info("Running validation...")
                    
                    if args.use_ema:
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())

                    unet_val = accelerator.unwrap_model(model)
                    unet_val.eval()

                    val_scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config)
                    val_scheduler.set_timesteps(args.num_inference_steps)

                    val_save_dir = os.path.join(args.output_dir, "validation", f"step_{global_step}")
                    os.makedirs(val_save_dir, exist_ok=True)

                    total_psnr = 0.0
                    total_ssim = 0.0
                    total_lpips = 0.0
                    num_val_samples = 0

                    with torch.no_grad():
                        for val_idx, (val_low, val_clean) in enumerate(eval_dataloader):
                            if val_idx >= args.num_validation_images // args.batch_size:
                                break

                            val_low = val_low.to(accelerator.device)
                            val_clean = val_clean.to(accelerator.device)
                            latents = torch.randn_like(val_low) * val_scheduler.init_noise_sigma

                            for t in tqdm(val_scheduler.timesteps, desc="Sampling", leave=False):
                                latent_model_input = val_scheduler.scale_model_input(latents, t)
                                
                                if decom_model is not None:
                                    val_low_01 = (val_low / 2 + 0.5).clamp(0, 1)
                                    r_val, i_val = decom_model(val_low_01)
                                    r_val_norm = r_val * 2.0 - 1.0
                                    i_val_norm = i_val * 2.0 - 1.0
                                    model_input = torch.cat([latent_model_input, r_val_norm, i_val_norm], dim=1)
                                else:
                                    model_input = torch.cat([latent_model_input, val_low], dim=1)

                                noise_pred = unet_val(model_input, t).sample
                                latents = val_scheduler.step(noise_pred, t, latents).prev_sample

                            enhanced_images = (latents / 2 + 0.5).clamp(0, 1)
                            gt_images = (val_clean / 2 + 0.5).clamp(0, 1)
                            low_images = (val_low / 2 + 0.5).clamp(0, 1)

                            # Metrics
                            batch_psnr = peak_signal_noise_ratio(enhanced_images, gt_images, data_range=1.0)
                            total_psnr += batch_psnr.item() * val_low.shape[0]
                            
                            batch_ssim = ssim(enhanced_images, gt_images).item()
                            total_ssim += batch_ssim * val_low.shape[0]

                            if lpips_model is not None:
                                enh_norm = enhanced_images * 2.0 - 1.0
                                gt_norm = gt_images * 2.0 - 1.0
                                batch_lpips = lpips_model(enh_norm, gt_norm).mean().item()
                                total_lpips += batch_lpips * val_low.shape[0]

                            num_val_samples += val_low.shape[0]

                            # Save Images
                            if val_idx == 0:
                                for i in range(min(4, enhanced_images.shape[0])):
                                    grid = torch.cat([low_images[i], enhanced_images[i], gt_images[i]], dim=2)
                                    transforms.ToPILImage()(grid).save(os.path.join(val_save_dir, f"val_{val_idx}_{i}.png"))

                    if num_val_samples > 0:
                        avg_psnr = total_psnr / num_val_samples
                        avg_ssim = total_ssim / num_val_samples
                        avg_lpips = total_lpips / num_val_samples
                        logger.info(f"Val Step {global_step}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
                        accelerator.log({"val/psnr": avg_psnr, "val/ssim": avg_ssim, "val/lpips": avg_lpips}, step=global_step)

                    if args.use_ema:
                        ema_model.restore(model.parameters())
                    unet_val.train()

                # === Checkpointing ===
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if args.use_ema:
                        torch.save(ema_model.state_dict(), os.path.join(save_path, "ema_model.pth"))
                    if decom_model is not None:
                        unwrapped_decom = accelerator.unwrap_model(decom_model)
                        torch.save(unwrapped_decom.state_dict(), os.path.join(save_path, "decom_model.pth"))

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    accelerator.end_training()

    # === Save Final Model ===
    if accelerator.is_main_process:
        if args.use_ema:
            logger.info("Saving EMA model as final model...")
            ema_model.copy_to(model.parameters())

        unet = accelerator.unwrap_model(model)
        unet.save_pretrained(os.path.join(args.output_dir, "unet_final"))
        
        if decom_model is not None:
            unwrapped_decom = accelerator.unwrap_model(decom_model)
            torch.save(unwrapped_decom.state_dict(), os.path.join(args.output_dir, "unet_final", "decom_model.pth"))

        logger.info("Training Finished.")

if __name__ == "__main__":
    main()