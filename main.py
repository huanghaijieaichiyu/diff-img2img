import argparse
import os
import sys
import subprocess
from core.engine import DiffusionEngine

def get_args():
    parser = argparse.ArgumentParser(description="Diff-Img2Img Unified Engine")
    
    # Mode Selection
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict", "validate", "ui"], help="Execution mode")
    
    # Common Args
    parser.add_argument("--data_dir", type=str, default="../datasets/kitti_LOL")
    parser.add_argument("--output_dir", type=str, default="runs/exp1")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model for predict/validate/resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    
    # Train Args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--offset_noise", action="store_true", help="Use offset noise")
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    
    # Model Args
    parser.add_argument("--use_retinex", action="store_true")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--prediction_type", type=str, default="v_prediction")
    parser.add_argument("--unet_layers_per_block", type=int, default=2)
    parser.add_argument("--unet_block_channels", nargs='+', type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--unet_down_block_types", nargs='+', type=str, default=["DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"])
    parser.add_argument("--unet_up_block_types", nargs='+', type=str, default=["AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"])
    
    # Predict/Validate Args
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # UI specific (dummy to avoid crash if passed)
    parser.add_argument("--port", type=int, default=8501)

    return parser.parse_args()

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
        engine = DiffusionEngine(args)
        
        if args.mode == "train":
            engine.train()
        elif args.mode == "validate":
            engine.validate()
        elif args.mode == "predict":
            engine.predict()
