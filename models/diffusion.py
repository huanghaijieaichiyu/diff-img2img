import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, unet, decom_model=None):
        super().__init__()
        self.unet = unet
        self.decom_model = decom_model

    def forward(self, low_light_images, noisy_images, timesteps):
        r_low, i_low = None, None
        if self.decom_model is not None:
            # Retinex Decomposition (R, I)
            low_light_01 = (low_light_images / 2 + 0.5).clamp(0, 1)
            r_low, i_low = self.decom_model(low_light_01)
            
            # Normalize [0,1] -> [-1,1]
            r_low_norm = r_low * 2.0 - 1.0
            i_low_norm = i_low * 2.0 - 1.0
            
            model_input = torch.cat([noisy_images, r_low_norm, i_low_norm], dim=1)
        else:
            model_input = torch.cat([noisy_images, low_light_images], dim=1)

        return self.unet(model_input, timesteps).sample, r_low, i_low
