import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import Conv, C2f, ConvTranspose, Concat


class DecomNet(nn.Module):
    """
    Upgraded Retinex Decomposition Network using a lightweight U-Net architecture.
    Leverages C2f blocks for efficient feature extraction and skip connections for detail preservation.

    Input: Low-light image (B, 3, H, W)
    Output: 
        - Reflectance (B, 3, H, W)
        - Illumination (B, 1, H, W)
    """

    def __init__(self, base_channel=32):
        super(DecomNet, self).__init__()
        
        # === Encoder ===
        # Level 0
        self.cv0 = Conv(3, base_channel, k=3, s=1)
        self.c2f0 = C2f(base_channel, base_channel, n=1, shortcut=True)
        
        # Level 1
        self.down1 = Conv(base_channel, base_channel * 2, k=3, s=2)
        self.c2f1 = C2f(base_channel * 2, base_channel * 2, n=1, shortcut=True)
        
        # Level 2 (Bottleneck)
        self.down2 = Conv(base_channel * 2, base_channel * 4, k=3, s=2)
        self.c2f2 = C2f(base_channel * 4, base_channel * 4, n=1, shortcut=True)
        
        # === Decoder ===
        # Up 1
        self.up1 = ConvTranspose(base_channel * 4, base_channel * 2, k=2, s=2)
        self.concat1 = Concat()
        self.c2f_up1 = C2f(base_channel * 4, base_channel * 2, n=1, shortcut=True) # 2*C from up + 2*C from skip
        
        # Up 0
        self.up0 = ConvTranspose(base_channel * 2, base_channel, k=2, s=2)
        self.concat0 = Concat()
        self.c2f_up0 = C2f(base_channel * 2, base_channel, n=1, shortcut=True) # 1*C from up + 1*C from skip

        # === Heads ===
        # Reflectance Branch
        self.r_head = nn.Sequential(
            Conv(base_channel, base_channel, k=3),
            nn.Conv2d(base_channel, 3, kernel_size=3, padding=1)
        )

        # Illumination Branch
        self.i_head = nn.Sequential(
            Conv(base_channel, base_channel, k=3),
            nn.Conv2d(base_channel, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder
        x0 = self.cv0(x)
        feat0 = self.c2f0(x0) # Skip 0
        
        x1 = self.down1(feat0)
        feat1 = self.c2f1(x1) # Skip 1
        
        x2 = self.down2(feat1)
        feat2 = self.c2f2(x2) # Bottleneck
        
        # Decoder
        up1 = self.up1(feat2)
        # Resize if necessary for concatenation (in case of odd dimensions)
        if up1.shape != feat1.shape:
            up1 = F.interpolate(up1, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        cat1 = self.concat1([up1, feat1])
        dec1 = self.c2f_up1(cat1)
        
        up0 = self.up0(dec1)
        if up0.shape != feat0.shape:
            up0 = F.interpolate(up0, size=feat0.shape[2:], mode='bilinear', align_corners=False)
        cat0 = self.concat0([up0, feat0])
        dec0 = self.c2f_up0(cat0)

        # Heads
        r = torch.sigmoid(self.r_head(dec0))
        i = torch.sigmoid(self.i_head(dec0))
        
        return r, i