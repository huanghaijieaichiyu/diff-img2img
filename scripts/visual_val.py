import torch
import os
import sys

# Add root to sys.path if running standalone to find models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusers import UNet2DModel, DDPMScheduler
from models.retinex import DecomNet
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

def load_models(model_path, use_retinex, device):
    """
    Load the UNet and optional Retinex DecomNet models.
    """
    models = {}
    
    # 1. Load UNet
    # Try loading as a full pipeline or just UNet
    unet_path = os.path.join(model_path, "unet_final")
    if not os.path.exists(unet_path):
        # Fallback: try loading directly from model_path if unet_final doesn't exist
        unet_path = model_path
        
    try:
        unet = UNet2DModel.from_pretrained(unet_path).to(device)
        unet.eval()
        models["unet"] = unet
    except Exception as e:
        raise RuntimeError(f"Failed to load UNet from {unet_path}: {e}")

    # 2. Load DecomNet (if Retinex)
    if use_retinex:
        decom_path = os.path.join(unet_path, "decom_model.pth")
        # If not in unet folder, check root of checkpoint
        if not os.path.exists(decom_path):
            decom_path = os.path.join(model_path, "decom_model.pth")
            
        if os.path.exists(decom_path):
            decom_model = DecomNet().to(device)
            decom_model.load_state_dict(torch.load(decom_path, map_location=device))
            decom_model.eval()
            models["decom"] = decom_model
        else:
            print(f"Warning: Retinex enabled but decom_model.pth not found at {decom_path}. Proceeding without it (may fail).")
            models["decom"] = None

    # 3. Scheduler (Default to DPMSolver for fast inference)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(unet_path) # Try loading config
    if not scheduler:
        # Fallback config
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction" # Assumption based on training script
        )
    models["scheduler"] = scheduler
    
    return models

def run_inference(models, t_low, num_inference_steps, device):
    """
    Run diffusion inference to enhance the low light image.
    t_low: Input tensor (C, H, W) normalized to [-1, 1]
    """
    unet = models["unet"]
    decom = models.get("decom")
    scheduler = models["scheduler"]
    
    scheduler.set_timesteps(num_inference_steps)
    
    # Prepare input
    # Ensure batch dimension [1, C, H, W]
    if t_low.ndim == 3:
        t_low = t_low.unsqueeze(0)
    t_low = t_low.to(device)
    
    # Init Noise
    latents = torch.randn_like(t_low) * scheduler.init_noise_sigma
    
    with torch.no_grad():
        for t in scheduler.timesteps:
            latent_model_input = scheduler.scale_model_input(latents, t)
            
            if decom is not None:
                # Retinex Decomposition
                # Input to DecomNet should be [0, 1]
                low_light_01 = (t_low / 2 + 0.5).clamp(0, 1)
                r_low, i_low = decom(low_light_01)
                
                # Normalize R/I to [-1, 1] for UNet
                r_low_norm = r_low * 2.0 - 1.0
                i_low_norm = i_low * 2.0 - 1.0
                
                model_input = torch.cat([latent_model_input, r_low_norm, i_low_norm], dim=1)
            else:
                model_input = torch.cat([latent_model_input, t_low], dim=1)
            
            # Predict noise/velocity
            noise_pred = unet(model_input, t).sample
            
            # Step
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
    return latents

def tensor_to_pil(tensor):
    """
    Convert a tensor [-1, 1] to PIL Image.
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    image = tensor.cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

def plot_histogram(image):
    """
    Plot RGB histogram for a PIL image.
    Returns a PIL image of the plot.
    """
    img_np = np.array(image)
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['red', 'green', 'blue']
    
    for i, color in enumerate(colors):
        hist, bins = np.histogram(img_np[:, :, i], bins=256, range=(0, 256))
        ax.plot(bins[:-1], hist, color=color, alpha=0.7)
        
    ax.set_title("RGB Histogram")
    ax.set_xlim([0, 256])
    ax.axis('off')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
