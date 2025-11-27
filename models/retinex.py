import torch
import torch.nn as nn
import torch.nn.functional as F

class DecomNet(nn.Module):
    """
    一个轻量级的 Retinex 分解网络。
    输入: 低光照图像 (B, 3, H, W)
    输出: 
        - Reflectance (反射图): (B, 3, H, W) - 包含物体本身的颜色和纹理
        - Illumination (光照图): (B, 1, H, W) - 包含光照强度分布
    """
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # 浅层特征提取
        self.net = nn.Sequential(
            nn.Conv2d(3, channel, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # R 分支 (Reflectance) - 输出 3 通道
        self.conv_r = nn.Conv2d(channel, 3, kernel_size, padding=1)
        
        # I 分支 (Illumination) - 输出 1 通道
        self.conv_i = nn.Conv2d(channel, 1, kernel_size, padding=1)

    def forward(self, x):
        feat = self.net(x)
        
        # 使用 Sigmoid 限制输出在 [0, 1] 范围内
        r = torch.sigmoid(self.conv_r(feat))
        i = torch.sigmoid(self.conv_i(feat))
        return r, i
