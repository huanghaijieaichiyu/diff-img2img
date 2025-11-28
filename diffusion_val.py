import argparse
import os
import logging
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from torcheval.metrics.functional import peak_signal_noise_ratio
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
import lpips
from utils.misic import ssim
from datasets.data_set import LowLightDataset
from models.retinex import DecomNet

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Validation script for Diffusion Low-Light Enhancement")
    parser.add_argument("--data_dir", type=str, default="../datasets/kitti_LOL", help="Root dataset directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory (containing unet_final)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for results")
    parser.add_argument("--resolution", type=int, default=256, help="Input resolution")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="DPM-Solver sampling steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mixed_precision", type=str, default='fp16', choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--use_retinex", action="store_true", help="Use Retinex decomposition network")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    return parser.parse_args()

def main():
    args = parse_args()
    
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    set_seed(args.seed)

    # 1. Load Models
    logger.info(f"Loading models from {args.model_path}...")
    
    # UNet
    unet_path = os.path.join(args.model_path, "unet_final")
    if not os.path.exists(unet_path):
        # Fallback if user points to a checkpoint folder
        unet_path = args.model_path 
        
    try:
        unet = UNet2DModel.from_pretrained(unet_path, use_safetensors=True)
    except:
        unet = UNet2DModel.from_pretrained(unet_path, use_safetensors=False)
    
    unet.to(accelerator.device)
    unet.eval()

    # DecomNet
    decom_model = None
    if args.use_retinex:
        decom_model = DecomNet().to(accelerator.device)
        # Try loading decom model from the same directory
        decom_path = os.path.join(args.model_path, "unet_final", "decom_model.pth")
        if not os.path.exists(decom_path):
            decom_path = os.path.join(args.model_path, "decom_model.pth")
        
        if os.path.exists(decom_path):
            logger.info(f"Loading DecomNet from {decom_path}")
            decom_model.load_state_dict(torch.load(decom_path, map_location=accelerator.device))
        else:
            logger.warning(f"DecomNet weights not found at {decom_path}, using random init (Expect poor results if this was intended).")
        decom_model.eval()

    # LPIPS
    logger.info("Initializing LPIPS...")
    lpips_model = lpips.LPIPS(net='alex').to(accelerator.device).eval()

    # Scheduler
    # We use DPMSolver for fast inference
    scheduler = DPMSolverMultistepScheduler.from_config(unet.config)
    scheduler.set_timesteps(args.num_inference_steps)

    # 2. Data
    logger.info("Loading Dataset...")
    eval_dataset = LowLightDataset(image_dir=args.data_dir, img_size=args.resolution, phase="test")
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    unet, eval_dataloader = accelerator.prepare(unet, eval_dataloader)

    # 3. Evaluation Loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    num_samples = 0

    logger.info("Starting Inference...")
    
    for step, (val_low, val_clean) in enumerate(tqdm(eval_dataloader, disable=not accelerator.is_local_main_process)):
        val_low = val_low.to(accelerator.device)
        val_clean = val_clean.to(accelerator.device)
        
        # Init noise
        latents = torch.randn_like(val_low) * scheduler.init_noise_sigma
        
        with torch.no_grad():
            for t in scheduler.timesteps:
                latent_model_input = scheduler.scale_model_input(latents, t)
                
                if decom_model is not None:
                    val_low_01 = (val_low / 2 + 0.5).clamp(0, 1)
                    r_val, i_val = decom_model(val_low_01)
                    r_val_norm = r_val * 2.0 - 1.0
                    i_val_norm = i_val * 2.0 - 1.0
                    model_input = torch.cat([latent_model_input, r_val_norm, i_val_norm], dim=1)
                else:
                    model_input = torch.cat([latent_model_input, val_low], dim=1)

                noise_pred = unet(model_input, t).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        enhanced_images = (latents / 2 + 0.5).clamp(0, 1)
        gt_images = (val_clean / 2 + 0.5).clamp(0, 1)
        low_images = (val_low / 2 + 0.5).clamp(0, 1)

        # Metrics
        # PSNR
        batch_psnr = peak_signal_noise_ratio(enhanced_images, gt_images, data_range=1.0)
        total_psnr += batch_psnr.item() * val_low.shape[0]
        
        # SSIM
        batch_ssim = ssim(enhanced_images, gt_images).item()
        total_ssim += batch_ssim * val_low.shape[0]

        # LPIPS (inputs expected to be [-1, 1])
        enh_norm = enhanced_images * 2.0 - 1.0
        gt_norm = gt_images * 2.0 - 1.0
        batch_lpips = lpips_model(enh_norm, gt_norm).mean().item()
        total_lpips += batch_lpips * val_low.shape[0]

        num_samples += val_low.shape[0]

        # Save samples (first batch only or a few images)
        if step == 0:
            for i in range(min(8, enhanced_images.shape[0])):
                grid = torch.cat([low_images[i], enhanced_images[i], gt_images[i]], dim=2)
                save_path = os.path.join(args.output_dir, f"eval_sample_{i}.png")
                transforms.ToPILImage()(grid).save(save_path)

    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_lpips = total_lpips / num_samples

    logger.info("="*40)
    logger.info(f"Evaluation Results (N={num_samples})")
    logger.info(f"PSNR:  {avg_psnr:.4f}")
    logger.info(f"SSIM:  {avg_ssim:.4f}")
    logger.info(f"LPIPS: {avg_lpips:.4f}")
    logger.info("="*40)

    # Save metrics to text file
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"PSNR: {avg_psnr:.4f}\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"LPIPS: {avg_lpips:.4f}\n")

if __name__ == "__main__":
    main()
