import json
import os
import sys
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.conditioning import build_condition_adapter
from models.diffusion import CombinedModel
from models.retinex import build_decom_net


def _read_metadata(model_path):
    metadata_candidates = [
        os.path.join(model_path, "metadata.json"),
        os.path.join(model_path, "final_metadata.json"),
        os.path.join(model_path, "best_metadata.json"),
    ]
    for metadata_path in metadata_candidates:
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as handle:
                return json.load(handle)
    return {"args": {}}


def load_models(model_path, use_retinex, device):
    models = {}
    metadata = _read_metadata(model_path)
    args_dict = metadata.get("args", {})

    unet_path = os.path.join(model_path, "unet_final")
    if not os.path.exists(unet_path):
        unet_path = os.path.join(model_path, "unet_best")
    if not os.path.exists(unet_path):
        unet_path = model_path

    unet = UNet2DModel.from_pretrained(unet_path).to(device)
    unet.eval()
    models["unet"] = unet

    decom_model = None
    if use_retinex:
        decom_candidates = [
            os.path.join(model_path, "decom_model.pth"),
            os.path.join(model_path, "decom_model_best.pth"),
            os.path.join(unet_path, "decom_model.pth"),
        ]
        for decom_path in decom_candidates:
            if os.path.exists(decom_path):
                decom_model = build_decom_net(
                    variant=args_dict.get("decom_variant", "middle"),
                    base_channel=int(args_dict.get("decom_base_channels", 32)),
                ).to(device)
                decom_model.load_state_dict(torch.load(decom_path, map_location=device))
                decom_model.eval()
                break
    models["decom"] = decom_model

    condition_adapter = build_condition_adapter(
        variant=args_dict.get("condition_variant", "middle"),
        block_channels=args_dict.get("unet_block_channels", [32, 64, 128, 256, 512]),
        cond_out_channels=7,
        base_channels=int(args_dict.get("base_condition_channels", 32)),
        use_retinex=use_retinex,
        conditioning_space=args_dict.get("conditioning_space", "hvi_lite"),
    ).to(device)

    adapter_candidates = [
        os.path.join(model_path, "condition_adapter.pth"),
        os.path.join(model_path, "condition_adapter_best.pth"),
        os.path.join(unet_path, "condition_adapter.pth"),
    ]
    adapter_loaded = False
    for adapter_path in adapter_candidates:
        if os.path.exists(adapter_path):
            condition_adapter.load_state_dict(torch.load(adapter_path, map_location=device), strict=False)
            adapter_loaded = True
            break
    if not adapter_loaded:
        raise RuntimeError("Condition adapter weights were not found for visualization.")
    condition_adapter.eval()
    models["condition_adapter"] = condition_adapter

    try:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(unet_path)
    except Exception:
        scheduler = DPMSolverMultistepScheduler.from_config(
            DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule="squaredcos_cap_v2",
                prediction_type=args_dict.get("prediction_type", "v_prediction"),
            ).config
        )

    models["scheduler"] = scheduler
    models["conditioning_space"] = args_dict.get("conditioning_space", "hvi_lite")
    models["residual_scale"] = float(args_dict.get("residual_scale", 0.5))
    return models


def run_inference(models, t_low, num_inference_steps, device):
    unet = models["unet"]
    decom = models.get("decom")
    condition_adapter = models["condition_adapter"]
    scheduler = models["scheduler"]
    conditioning_space = models.get("conditioning_space", "hvi_lite")
    residual_scale = models.get("residual_scale", 0.5)
    inference_model = CombinedModel(
        unet,
        decom_model=decom,
        condition_adapter=condition_adapter,
        conditioning_space=conditioning_space,
    )

    scheduler.set_timesteps(num_inference_steps)

    if t_low.ndim == 3:
        t_low = t_low.unsqueeze(0)
    t_low = t_low.to(device)

    latents = torch.randn_like(t_low) * scheduler.init_noise_sigma

    with torch.no_grad():
        for timestep in scheduler.timesteps:
            latent_input = scheduler.scale_model_input(latents, timestep)
            model_input, aux = inference_model.build_model_input(t_low, latent_input)
            noise_pred = inference_model.run_unet(model_input, timestep, aux)
            latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    enhanced = (t_low + latents / max(residual_scale, 1e-6)).clamp(-1, 1)
    return enhanced


def tensor_to_pil(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    image = tensor.cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def plot_histogram(image):
    img_np = np.array(image)
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["red", "green", "blue"]

    for idx, color in enumerate(colors):
        hist, bins = np.histogram(img_np[:, :, idx], bins=256, range=(0, 256))
        ax.plot(bins[:-1], hist, color=color, alpha=0.7)

    ax.set_title("RGB Histogram")
    ax.set_xlim([0, 256])
    ax.axis("off")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer)
