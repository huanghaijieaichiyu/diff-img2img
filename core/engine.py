import os
import logging
import math
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from torcheval.metrics.functional import peak_signal_noise_ratio

from utils.misic import ssim, Save_path, compute_snr, compute_min_snr_loss_weights, charbonnier_loss_elementwise
from utils.loss import CompositeLoss
from datasets.data_set import LowLightDataset
from models.retinex import DecomNet
from models.diffusion import CombinedModel
from utils.video_writer import video_writer

logger = get_logger(__name__, log_level="INFO")

class DiffusionEngine:
    def __init__(self, args):
        self.args = args
        self.best_psnr = -1.0
        
        # 1. Initialize Accelerator
        logging_dir = os.path.join(args.output_dir, "logs")
        accelerator_project_config = ProjectConfiguration(
            project_dir=args.output_dir, logging_dir=logging_dir
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )

        # 2. Logging Setup
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)

        if args.seed is not None:
            set_seed(args.seed)
        
        if self.accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)

        # 3. Model Setup
        self._setup_models()
        
        # 4. Scheduler Setup
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=args.prediction_type
        )
        
        # 5. Loss Setup (Only needed for training, but harmless to init)
        self.criterion = CompositeLoss(device=self.accelerator.device).to(self.accelerator.device)

    def _setup_models(self):
        # Input channels: 3 (Noisy) + [3 (Reflectance) + 1 (Illumination) if Retinex else 3 (Low Light)]
        input_channels = 7 if self.args.use_retinex else 6

        logger.info(f"Initializing UNet (Input Channels={input_channels})...")
        self.unet = UNet2DModel(
            sample_size=self.args.resolution,
            in_channels=input_channels,
            out_channels=3,
            layers_per_block=self.args.unet_layers_per_block,
            block_out_channels=self.args.unet_block_channels,
            down_block_types=tuple(self.args.unet_down_block_types),
            up_block_types=tuple(self.args.unet_up_block_types)
        )

        if self.args.enable_xformers_memory_efficient_attention:
            if self.accelerator.is_main_process:
                logger.info("Enabling xformers memory efficient attention")
            self.unet.enable_xformers_memory_efficient_attention()

        self.decom_model = None
        if self.args.use_retinex:
            logger.info("Initializing Retinex Decomposition Network...")
            self.decom_model = DecomNet().to(self.accelerator.device)
        
        # Load Pretrained/Checkpoint if provided
        if self.args.model_path:
            self._load_checkpoint(self.args.model_path)

        # Create Combined Model for Training
        self.training_model = CombinedModel(self.unet, self.decom_model)
        
        # EMA Setup
        self.ema_model = None
        if self.args.use_ema:
            logger.info("Initializing EMA model...")
            self.ema_model = EMAModel(
                self.unet.parameters(),
                decay=self.args.ema_decay,
                model_cls=UNet2DModel,
                model_config=self.unet.config,
            )
            self.ema_model.to(self.accelerator.device)

    def _load_checkpoint(self, path):
        logger.info(f"Loading model from {path}...")
        
        # Load UNet
        unet_path = os.path.join(path, "unet_final")
        if not os.path.exists(unet_path):
            unet_path = path # Fallback
        
        try:
            # Load state dict directly if it's a full model save, or use from_pretrained
            self.unet = UNet2DModel.from_pretrained(unet_path, use_safetensors=True)
        except:
            try:
                self.unet = UNet2DModel.from_pretrained(unet_path, use_safetensors=False)
            except Exception as e:
                logger.warning(f"Could not load UNet from {unet_path}: {e}")

        self.unet.to(self.accelerator.device)

        # Load DecomNet
        if self.decom_model is not None:
            decom_path = os.path.join(path, "unet_final", "decom_model.pth")
            if not os.path.exists(decom_path):
                decom_path = os.path.join(path, "decom_model.pth")
            
            if os.path.exists(decom_path):
                self.decom_model.load_state_dict(torch.load(decom_path, map_location=self.accelerator.device))
                logger.info("DecomNet loaded.")
            else:
                logger.warning("DecomNet weights not found.")

    def train(self):
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.training_model.parameters(), lr=self.args.lr, weight_decay=1e-2, eps=1e-08
        )

        # Dataloaders
        train_dataset = LowLightDataset(
            image_dir=self.args.data_dir, img_size=self.args.resolution, phase="train")
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True
        )
        
        # LR Scheduler
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.epochs * num_update_steps_per_epoch
        
        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )

        # Prepare
        self.training_model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.training_model, optimizer, train_dataloader, lr_scheduler
        )

        # Trackers
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.args))
            # Filter configuration to satisfy TensorBoard requirements
            for k, v in list(tracker_config.items()):
                if not isinstance(v, (int, float, str, bool, torch.Tensor)):
                    if isinstance(v, list):
                        tracker_config[k] = str(v)
                    elif v is None:
                        tracker_config[k] = "None"
                    else:
                        del tracker_config[k]
            self.accelerator.init_trackers(Path(self.args.output_dir).name, config=tracker_config)

        # Loop
        logger.info("***** Running training *****")
        global_step = 0
        progress_bar = tqdm(range(global_step, self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)

        for epoch in range(self.args.epochs):
            self.training_model.train()
            for step, batch in enumerate(train_dataloader):
                loss, logs = self._train_step(batch, optimizer, lr_scheduler)
                
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if self.accelerator.is_main_process:
                        progress_bar.set_postfix(**logs)
                        self.accelerator.log(logs, step=global_step)

                    # Validation
                    if global_step % self.args.validation_steps == 0:
                        self.validate(step=global_step)

                    # Checkpoint
                    if global_step % self.args.checkpointing_steps == 0:
                        self._save_checkpoint(global_step)

                if global_step >= self.args.max_train_steps:
                    break
        
        self.accelerator.end_training()
        self._save_final_model()

    def _train_step(self, batch, optimizer, lr_scheduler):
        low_light_images, clean_images = batch
        
        # Noise Injection
        noise = torch.randn_like(clean_images)
        if self.args.offset_noise:
            offset_noise = torch.randn(clean_images.shape[0], clean_images.shape[1], 1, 1, device=clean_images.device)
            noise = noise + 0.1 * offset_noise

        bsz = clean_images.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device).long()
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        with self.accelerator.accumulate(self.training_model):
            model_pred, r_low, i_low = self.training_model(low_light_images, noisy_images, timesteps)

            # Target
            if self.args.prediction_type == "epsilon":
                target = noise
            else:
                target = self.noise_scheduler.get_velocity(clean_images, noise, timesteps)

            # Loss 1: Diffusion Loss
            snr_weights = compute_min_snr_loss_weights(self.noise_scheduler, timesteps, self.args.snr_gamma)
            loss_diff_elem = charbonnier_loss_elementwise(model_pred, target)
            loss_diffusion = (loss_diff_elem.mean(dim=[1, 2, 3]) * snr_weights).mean()

            # Loss 2: Composite Loss (Reconstruction)
            # Reconstruct x0
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(clean_images.device)
            alpha_prod_t = alphas_cumprod[timesteps][:, None, None, None]
            beta_prod_t = 1 - alpha_prod_t
            
            if self.args.prediction_type == "epsilon":
                pred_x0 = (noisy_images - beta_prod_t ** (0.5) * model_pred) / alpha_prod_t ** (0.5)
            else:
                pred_x0 = alpha_prod_t ** (0.5) * noisy_images - beta_prod_t ** (0.5) * model_pred
            
            loss_composite, loss_logs = self.criterion(pred_x0, clean_images)
            
            loss = loss_diffusion + loss_composite

            # Loss 3: Retinex
            logs = {
                "loss": loss.item(), 
                "l_diff": loss_diffusion.item(), 
                "l_comp": loss_composite.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }

            loss_retinex = 0.0
            if self.args.use_retinex and r_low is not None:
                low_light_01 = (low_light_images / 2 + 0.5).clamp(0, 1)
                clean_images_01 = (clean_images / 2 + 0.5).clamp(0, 1)
                loss_recon = F.l1_loss(r_low * i_low, low_light_01)
                loss_reflectance = F.l1_loss(r_low, clean_images_01)
                loss_tv = torch.mean(torch.abs(i_low[:, :, :-1, :] - i_low[:, :, 1:, :])) + \
                          torch.mean(torch.abs(i_low[:, :, :, :-1] - i_low[:, :, :, 1:]))
                loss_retinex = loss_recon + loss_reflectance + 0.1 * loss_tv
                loss += 0.1 * loss_retinex
                
                logs.update({
                    "l_ret": loss_retinex.item(),
                    "l_rec": loss_recon.item(),
                    "l_ref": loss_reflectance.item(),
                    "l_tv": loss_tv.item()
                })

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.training_model.parameters(), 5.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        return loss, logs

    def validate(self, step=None):
        logger.info(f"Running validation at step {step}...")
        
        # Setup Val Loader (Lazy load)
        eval_dataset = LowLightDataset(
            image_dir=self.args.data_dir, img_size=self.args.resolution, phase="test")
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers
        )
        
        # Unwrap
        unet = self.accelerator.unwrap_model(self.training_model).unet
        decom = self.accelerator.unwrap_model(self.training_model).decom_model
        
        if self.args.use_ema:
            self.ema_model.store(unet.parameters())
            self.ema_model.copy_to(unet.parameters())
        
        unet.eval()
        if decom: decom.eval()
        
        # Scheduler for Inference
        val_scheduler = DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config)
        val_scheduler.set_timesteps(self.args.num_inference_steps)

        total_psnr, total_ssim = 0.0, 0.0
        num_samples = 0
        
        # Prepare validation directory
        val_dir = os.path.join(self.args.output_dir, "validation")
        os.makedirs(val_dir, exist_ok=True)
        
        with torch.no_grad():
            for i, (val_low, val_clean) in enumerate(eval_dataloader):
                if i >= self.args.num_validation_images // self.args.batch_size: break
                
                val_low = val_low.to(self.accelerator.device)
                val_clean = val_clean.to(self.accelerator.device)
                
                enhanced = self._inference_step(val_low, unet, decom, val_scheduler)
                
                # Save Visualization for the first batch
                if i == 0:
                    # Normalize to [0, 1] for visualization (assuming inputs are [-1, 1] for diff logic, but let's check)
                    # Engine uses (x/2 + 0.5) so inputs seem to be [-1, 1]
                    vis_low = (val_low / 2 + 0.5).clamp(0, 1)
                    vis_clean = (val_clean / 2 + 0.5).clamp(0, 1)
                    vis_enhanced = enhanced # already [0, 1] from inference_step
                    
                    # Create Grid: Top=Low, Mid=Enhanced, Bot=Clean
                    grid = torch.cat([vis_low, vis_enhanced, vis_clean], dim=0)
                    grid_image = make_grid(grid, nrow=vis_low.shape[0], padding=2)
                    
                    save_image_path = os.path.join(val_dir, f"val_step_{step}.png")
                    transforms.ToPILImage()(grid_image).save(save_image_path)
                    logger.info(f"Saved validation grid to {save_image_path}")

                # Metrics (Images are [0, 1])
                gt_images = (val_clean / 2 + 0.5).clamp(0, 1)
                
                total_psnr += peak_signal_noise_ratio(enhanced, gt_images, data_range=1.0).item() * val_low.shape[0]
                total_ssim += ssim(enhanced, gt_images).item() * val_low.shape[0]
                num_samples += val_low.shape[0]
        
        if num_samples > 0:
            avg_psnr = total_psnr / num_samples
            avg_ssim = total_ssim / num_samples
            logger.info(f"Validation: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
            self.accelerator.log({"val/psnr": avg_psnr, "val/ssim": avg_ssim}, step=step)
            
            # Best Model Check
            if avg_psnr > self.best_psnr:
                self.best_psnr = avg_psnr
                logger.info(f"New Best Model found (PSNR: {avg_psnr:.4f}). Saving...")
                
                best_model_dir = os.path.join(self.args.output_dir, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)
                
                unet.save_pretrained(os.path.join(best_model_dir, "unet_best"))
                if decom:
                    torch.save(decom.state_dict(), os.path.join(best_model_dir, "decom_model_best.pth"))
        
        # Restore EMA
        if self.args.use_ema:
            self.ema_model.restore(unet.parameters())
        
        self.training_model.train()

    def predict(self):
        """
        Unified prediction method for both Image folders and Videos.
        """
        self.unet.eval()
        if self.decom_model: self.decom_model.eval()
        
        # Scheduler
        val_scheduler = DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config)
        val_scheduler.set_timesteps(self.args.num_inference_steps)
        
        if hasattr(self.args, 'video_path') and self.args.video_path:
            self._predict_video(val_scheduler)
        else:
            self._predict_image(val_scheduler)

    def _predict_image(self, scheduler):
        logger.info(f"Starting Image Prediction from {self.args.data_dir}")
        dataset = LowLightDataset(image_dir=self.args.data_dir, img_size=self.args.resolution, phase="test")
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        
        out_dir = os.path.join(self.args.output_dir, 'predictions')
        os.makedirs(out_dir, exist_ok=True)
        
        idx = 0
        to_pil = transforms.ToPILImage()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Handle both (low, clean) or just low
                low_light = batch[0] if isinstance(batch, (list, tuple)) else batch
                low_light = low_light.to(self.accelerator.device)
                
                enhanced = self._inference_step(low_light, self.unet, self.decom_model, scheduler)
                
                for i in range(enhanced.shape[0]):
                    img = to_pil(enhanced[i].cpu())
                    img.save(os.path.join(out_dir, f"enhanced_{idx:05d}.png"))
                    idx += 1
        logger.info(f"Saved results to {out_dir}")

    def _predict_video(self, scheduler):
        logger.info(f"Starting Video Prediction: {self.args.video_path}")
        if not os.path.exists(self.args.video_path):
            raise FileNotFoundError(f"Video not found: {self.args.video_path}")
            
        cap = cv2.VideoCapture(self.args.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_dir = os.path.join(self.args.output_dir, 'video_frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.args.resolution, self.args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        idx = 0
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Preprocess
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(frame_rgb).unsqueeze(0).to(self.accelerator.device)
                
                # Inference
                enhanced = self._inference_step(input_tensor, self.unet, self.decom_model, scheduler)
                
                # Save
                enhanced_np = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                enhanced_bgr = cv2.cvtColor((enhanced_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:06d}.png"), enhanced_bgr)
                
                idx += 1
                print(f"Processed frame {idx}", end='\r')
        
        cap.release()
        
        # Synthesize
        out_video = os.path.join(self.args.output_dir, "enhanced.mp4")
        video_writer(frames_dir, out_video, fps=fps)
        logger.info(f"\nVideo saved to {out_video}")

    def _inference_step(self, low_light, unet, decom, scheduler):
        latents = torch.randn_like(low_light) * scheduler.init_noise_sigma
        
        for t in scheduler.timesteps:
            latent_input = scheduler.scale_model_input(latents, t)
            
            if decom is not None:
                low_01 = (low_light / 2 + 0.5).clamp(0, 1)
                r, i = decom(low_01)
                model_input = torch.cat([latent_input, r * 2 - 1, i * 2 - 1], dim=1)
            else:
                model_input = torch.cat([latent_input, low_light], dim=1)
            
            noise_pred = unet(model_input, t).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
        return (latents / 2 + 0.5).clamp(0, 1)

    def _save_checkpoint(self, step):
        save_path = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        self.accelerator.save_state(save_path)
        if self.args.use_ema:
            torch.save(self.ema_model.state_dict(), os.path.join(save_path, "ema_model.pth"))

    def _save_final_model(self):
        if self.accelerator.is_main_process:
            unet = self.accelerator.unwrap_model(self.training_model).unet
            if self.args.use_ema:
                self.ema_model.copy_to(unet.parameters())
            unet.save_pretrained(os.path.join(self.args.output_dir, "unet_final"))
            if self.decom_model:
                torch.save(self.accelerator.unwrap_model(self.training_model).decom_model.state_dict(), 
                           os.path.join(self.args.output_dir, "unet_final", "decom_model.pth"))

