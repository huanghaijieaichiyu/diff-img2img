# diffusion_predictor.py
import argparse
import os
import cv2  # 用于视频处理
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import time
import traceback

# Diffusers 相关导入
# [关键修改] 使用 UNet2DModel 替代 UNet2DConditionModel (因为没有 CrossAttention)
from diffusers.models.unets.unet_2d import UNet2DModel
# [关键修改] 引入 DPMSolverMultistepScheduler 用于快速推理 (20-50步)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from datasets.data_set import LowLightDataset
from utils.video_writer import video_writer


def save_output_path(base_path, model_type='predict'):
    """创建一个带有时间戳的唯一输出目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(
        base_path, f"diffusion_{model_type}_{timestamp}_{os.getpid()}")
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'predictions'), exist_ok=True)
    return path


class BaseDiffusionPredictor:
    """扩散模型预测器的基类"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        print(f"使用的设备: {self.device}")

        # === 1. 加载模型 ===
        # 训练时保存的是 UNet2DModel (Concat结构, in_channels=6)
        try:
            self.model = UNet2DModel.from_pretrained(
                args.model_path).to(self.device)
            self.model.eval()
            print(f"成功从 {args.model_path} 加载 UNet2DModel")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise

        # === 2. 初始化调度器 (Scheduler) ===
        # 为了加快推理速度，默认使用 DPMSolverMultistepScheduler (20-50步)
        # 必须确保 prediction_type 与训练时一致
        scheduler_config = {
            "num_train_timesteps": 1000,
            "beta_schedule": "squaredcos_cap_v2",  # 与训练保持一致
            # [关键] 支持 epsilon 或 v_prediction
            "prediction_type": args.prediction_type,
            # DPM-Solver 特定参数
            "algorithm_type": "dpmsolver++",
            "solver_order": 2,
        }

        if args.use_ddpm:
            # 如果强制使用 DDPM (慢，但作为基准)
            self.scheduler = DDPMScheduler(**scheduler_config)
            print("使用 DDPMScheduler (慢速标准采样)")
        else:
            # 默认使用 DPM-Solver++ (快速)
            self.scheduler = DPMSolverMultistepScheduler(**scheduler_config)
            print(
                f"使用 DPMSolverMultistepScheduler (快速采样, 预测类型: {args.prediction_type})")

        self.scheduler.set_timesteps(args.num_inference_steps)
        print(f"推断步数: {args.num_inference_steps}")

        self.to_pil = transforms.ToPILImage()

        self.output_path = save_output_path(
            args.output_dir, model_type='predict')
        print(f"预测结果将保存到: {self.output_path}")

    def _sample(self, low_light_condition_batch: torch.Tensor) -> torch.Tensor:
        """
        执行采样过程
        """
        batch_size = low_light_condition_batch.shape[0]

        # 1. 生成初始随机噪声
        latents = torch.randn(
            (batch_size, 3, self.args.resolution, self.args.resolution),
            device=self.device,
            dtype=self.model.dtype
        )

        # 缩放初始噪声 (DPM-Solver 需要)
        latents = latents * self.scheduler.init_noise_sigma

        low_light_condition_batch = low_light_condition_batch.to(
            self.device, dtype=self.model.dtype)

        # 2. 迭代采样
        for t in tqdm(self.scheduler.timesteps, desc="采样步骤", leave=False, disable=self.args.disable_tqdm):
            # 扩展 latents 以适应 classifier-free guidance (如果需要的话，这里暂时不需要)
            # scale_model_input 对 DPM-Solver 很重要
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            with torch.no_grad():
                # [关键逻辑] Concat Conditioning
                # 将 噪声Latent(3通道) 和 低光照条件图(3通道) 在维度1拼接 -> 6通道
                # 这与训练时的 model_input = torch.cat([noisy_images, low_light_images], dim=1) 对应
                model_input = torch.cat(
                    [latent_model_input, low_light_condition_batch], dim=1)

                # 预测噪声/速度
                # UNet2DModel 不需要 encoder_hidden_states
                noise_pred = self.model(model_input, t).sample

                # 调度器更新
                step_output = self.scheduler.step(noise_pred, t, latents)
                latents = step_output.prev_sample

        # 3. 后处理
        enhanced_images_0_1 = (latents / 2 + 0.5).clamp(0, 1)
        return enhanced_images_0_1

    def predict_images(self):
        raise NotImplementedError

    def predict_video(self):
        raise NotImplementedError


class ImageDiffusionPredictor(BaseDiffusionPredictor):
    def __init__(self, args):
        super().__init__(args)
        try:
            self.dataset: Dataset = LowLightDataset(
                image_dir=args.data_dir, img_size=args.resolution, phase="test"
            )
            if len(self.dataset) == 0:
                raise ValueError(f"No images found in {args.data_dir}")

            self.dataloader = DataLoader(
                self.dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            print(f"Loaded {len(self.dataset)} images.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def predict_images(self):
        print("开始图像预测...")
        img_index = 0
        output_dir = os.path.join(self.output_path, 'predictions')

        pbar = tqdm(self.dataloader, desc="Predicting")
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                low_light_batch = batch[0]
            else:
                low_light_batch = batch

            if not isinstance(low_light_batch, torch.Tensor):
                continue

            enhanced_batch = self._sample(low_light_batch)

            for i in range(enhanced_batch.shape[0]):
                enhanced_pil = self.to_pil(enhanced_batch[i].cpu())
                img_save_path = os.path.join(
                    output_dir, f"enhanced_{img_index:05d}.png")
                enhanced_pil.save(img_save_path)
                img_index += 1

            pbar.set_postfix({"Saved": img_index})

        print(f"Done. Results saved to {output_dir}")

    def predict_video(self):
        print("Image mode does not support video prediction.")


class VideoDiffusionPredictor(BaseDiffusionPredictor):
    def __init__(self, args):
        super().__init__(args)
        if not os.path.isfile(args.video_path):
            raise FileNotFoundError(f"Video not found: {args.video_path}")
        self.video_path = args.video_path
        self.frame_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def predict_images(self):
        print("Video mode does not support image prediction.")

    def predict_video(self):
        print(f"Start video prediction: {self.video_path}")
        frames_output_dir = os.path.join(self.output_path, 'predictions')

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_width = self.args.resolution
        out_height = self.args.resolution

        print(
            f"Video Info: {fps} FPS, {total_frames} Frames. Output Res: {out_width}x{out_height}")

        frame_count = 0
        pbar = tqdm(total=total_frames if total_frames >
                    0 else None, desc="Processing Video")
        frames_processed = False

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                # BGR -> RGB -> Transform -> (1, 3, H, W)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                low_light_tensor = self.frame_transform(frame_rgb).unsqueeze(0)

                # Sample
                enhanced_tensor = self._sample(low_light_tensor)

                # Post-process
                enhanced_np = enhanced_tensor.squeeze(
                    0).cpu().numpy().transpose(1, 2, 0)
                enhanced_uint8 = (enhanced_np * 255).clip(0,
                                                          255).astype(np.uint8)
                enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)

                # Resize check (in case output needs to strictly match)
                if enhanced_bgr.shape[:2] != (out_height, out_width):
                    enhanced_bgr = cv2.resize(
                        enhanced_bgr, (out_width, out_height))

                # Save
                cv2.imwrite(os.path.join(frames_output_dir,
                            f"frame_{frame_count:06d}.png"), enhanced_bgr)

                # Display
                if self.args.display_video:
                    display_orig = cv2.resize(
                        frame_bgr, (out_width, out_height))
                    combined = cv2.hconcat([display_orig, enhanced_bgr])
                    cv2.imshow("Original vs Diffusion", combined)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1
                frames_processed = True
                pbar.update(1)

        except Exception as e:
            print(f"Error at frame {frame_count}: {e}")
            traceback.print_exc()
        finally:
            pbar.close()
            cap.release()
            cv2.destroyAllWindows()

        if frames_processed:
            video_output_path = os.path.join(
                self.output_path, "enhanced_video.mp4")
            print("Synthesizing video...")
            video_writer(frames_output_dir, video_output_path, fps=fps)
            print(f"Done: {video_output_path}")


def parse_predict_args():
    parser = argparse.ArgumentParser(description="Diffusion Prediction Script")
    parser.add_argument("--mode", type=str, required=True,
                        choices=['image', 'video'])
    parser.add_argument("--model_path", type=str,
                        required=True, help="Path to unet folder")
    parser.add_argument("--output_dir", type=str,
                        default="diffusion_predictions")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int,
                        default=20, help="Default 20 for DPM-Solver")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_tqdm", action="store_true")

    # [关键参数] 预测类型: 必须与训练一致
    parser.add_argument("--prediction_type", type=str, default="epsilon",
                        choices=["epsilon", "v_prediction"],
                        help="Must match training! 'epsilon' (default) or 'v_prediction'")

    # 可选：强制使用 DDPM
    parser.add_argument("--use_ddpm", action="store_true",
                        help="Force use DDPMScheduler (slow)")

    parser.add_argument("--data_dir", type=str,
                        default="../datasets/kitti_LOL")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--display_video", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_predict_args()

    if args.mode == 'image':
        try:
            predictor = ImageDiffusionPredictor(args)
            predictor.predict_images()
        except Exception as e:
            traceback.print_exc()
    elif args.mode == 'video':
        try:
            predictor = VideoDiffusionPredictor(args)
            predictor.predict_video()
        except Exception as e:
            traceback.print_exc()
