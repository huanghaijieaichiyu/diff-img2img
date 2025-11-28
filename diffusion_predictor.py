import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import time
import traceback
from typing import Optional

from diffusers import UNet2DModel, DDPMScheduler, DPMSolverMultistepScheduler

from datasets.data_set import LowLightDataset
from utils.video_writer import video_writer
from models.retinex import DecomNet

class BaseDiffusionPredictor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        print(f"Device: {self.device}")

        # 1. Load Model
        try:
            # Load UNet2DModel (must match training config)
            self.model = UNet2DModel.from_pretrained(args.model_path).to(self.device)
            self.model.eval()
            print(f"Model loaded from {args.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # 2. Load DecomNet (if Retinex used)
        self.decom_model = None
        if args.use_retinex:
            try:
                decom_path = os.path.join(args.model_path, "decom_model.pth")
                if not os.path.exists(decom_path):
                    decom_path = os.path.join(os.path.dirname(args.model_path), "decom_model.pth")

                if os.path.exists(decom_path):
                    self.decom_model = DecomNet().to(self.device)
                    self.decom_model.load_state_dict(torch.load(decom_path, map_location=self.device))
                    self.decom_model.eval()
                    print(f"DecomNet loaded: {decom_path}")
                else:
                    print(f"Warning: DecomNet not found at {decom_path}, but --use_retinex is set.")
            except Exception as e:
                print(f"Error loading DecomNet: {e}")

        # 3. Scheduler
        scheduler_config = {
            "num_train_timesteps": 1000,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": args.prediction_type,
            "algorithm_type": "dpmsolver++",
            "solver_order": 2,
        }

        if args.use_ddpm:
            self.scheduler = DDPMScheduler(**scheduler_config)
            print("Scheduler: DDPM (Standard/Slow)")
        else:
            self.scheduler = DPMSolverMultistepScheduler(**scheduler_config)
            self.scheduler.set_timesteps(args.num_inference_steps)
            print(f"Scheduler: DPM-Solver++ (Fast, {args.num_inference_steps} steps)")

        self.to_pil = transforms.ToPILImage()
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(args.output_dir, f"pred_{self.timestamp}")
        os.makedirs(os.path.join(self.output_path, 'predictions'), exist_ok=True)

    def _sample(self, low_light_batch: torch.Tensor) -> torch.Tensor:
        batch_size = low_light_batch.shape[0]
        low_light_batch = low_light_batch.to(self.device)

        # Initial Noise
        latents = torch.randn(
            (batch_size, 3, self.args.resolution, self.args.resolution),
            device=self.device,
            dtype=self.model.dtype
        ) * self.scheduler.init_noise_sigma

        for t in tqdm(self.scheduler.timesteps, desc="Sampling", leave=False, disable=self.args.disable_tqdm):
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # Condition Input Construction
            if self.decom_model is not None:
                # Retinex Mode: Input(7) = Noisy(3) + R(3) + I(1)
                # Assuming input is normalized [-1, 1], convert to [0, 1] for DecomNet
                low_light_01 = (low_light_batch / 2 + 0.5).clamp(0, 1)
                r_low, i_low = self.decom_model(low_light_01)
                
                # Normalize R, I back to [-1, 1] for UNet
                r_low_norm = r_low * 2.0 - 1.0
                i_low_norm = i_low * 2.0 - 1.0
                
                model_input = torch.cat([latent_model_input, r_low_norm, i_low_norm], dim=1)
            else:
                # Standard Mode: Input(6) = Noisy(3) + LowLight(3)
                model_input = torch.cat([latent_model_input, low_light_batch], dim=1)

            with torch.no_grad():
                noise_pred = self.model(model_input, t).sample

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Post-process: [-1, 1] -> [0, 1]
        return (latents / 2 + 0.5).clamp(0, 1)

class ImageDiffusionPredictor(BaseDiffusionPredictor):
    def __init__(self, args):
        super().__init__(args)
        self.dataset = LowLightDataset(
            image_dir=args.data_dir, img_size=args.resolution, phase="test"
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=True
        )

    def run(self):
        print(f"Starting Image Prediction on {len(self.dataset)} images...")
        out_dir = os.path.join(self.output_path, 'predictions')
        idx = 0
        
        for batch in tqdm(self.dataloader, desc="Predicting"):
            low_light = batch[0] if isinstance(batch, (list, tuple)) else batch
            enhanced = self._sample(low_light)

            for i in range(enhanced.shape[0]):
                img = self.to_pil(enhanced[i].cpu())
                img.save(os.path.join(out_dir, f"enhanced_{idx:05d}.png"))
                idx += 1
        
        print(f"Saved results to {out_dir}")

class VideoDiffusionPredictor(BaseDiffusionPredictor):
    def __init__(self, args):
        super().__init__(args)
        if not os.path.exists(args.video_path):
            raise FileNotFoundError(f"Video not found: {args.video_path}")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def run(self):
        print(f"Starting Video Prediction: {self.args.video_path}")
        cap = cv2.VideoCapture(self.args.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_dir = os.path.join(self.output_path, 'predictions')
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame_rgb).unsqueeze(0)
            
            # Sample
            enhanced = self._sample(input_tensor)
            
            # Save Frame
            enhanced_np = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            enhanced_bgr = cv2.cvtColor((enhanced_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Resize back to resolution (optional, currently fixed to resolution)
            cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:06d}.png"), enhanced_bgr)
            
            if self.args.display_video:
                display = cv2.resize(frame, (self.args.resolution, self.args.resolution))
                cv2.imshow('Original vs Enhanced', np.hstack([display, enhanced_bgr]))
                if cv2.waitKey(1) == ord('q'): break
            
            idx += 1
            print(f"Processed frame {idx}", end='\r')

        cap.release()
        cv2.destroyAllWindows()
        
        # Synthesize Video
        out_video = os.path.join(self.output_path, "enhanced.mp4")
        video_writer(frames_dir, out_video, fps=fps)
        print(f"\nVideo saved to {out_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['image', 'video'])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="diffusion_predictions")
    parser.add_argument("--data_dir", type=str, default="../datasets/kitti_LOL")
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "v_prediction"])
    parser.add_argument("--use_retinex", action="store_true")
    parser.add_argument("--use_ddpm", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--display_video", action="store_true")
    
    args = parser.parse_args()
    
    if args.mode == 'image':
        ImageDiffusionPredictor(args).run()
    else:
        VideoDiffusionPredictor(args).run()