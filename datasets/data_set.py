import numpy as np
import os
import random
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF  # 关键引用：用于函数式变换
from PIL import Image

from scripts.darker import Darker


class LowLightDataset(Dataset):
    def __init__(self, image_dir, img_size=256, phase="train",
                 online_synthesis: bool = False,
                 darker_ranges: dict = None):
        """
        Args:
            image_dir (string): 数据集根目录。
            img_size (int): 训练/测试时的图像分辨率。
            phase (string): "train", "test" 或 "predict"。
            online_synthesis (bool): 如果为 True，则在每次 __getitem__ 中
                对 high 图像实时随机合成低光照退化，而非读取预生成的 low 图像。
                仅在 phase="train" 时有效。
            darker_ranges (dict): 自定义退化参数范围，传递给 Darker。
        """
        self.image_dir = image_dir
        self.img_size = img_size
        self.phase = phase
        self.online_synthesis = online_synthesis and (phase == "train")

        self.data = []

        # 初始化在线合成引擎
        self.darker = None
        if self.online_synthesis:
            self.darker = Darker(randomize=True, param_ranges=darker_ranges)

        # === 1. 构建文件列表 ===
        if phase == "predict":
            # 预测模式：直接加载目录下所有图片
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
            if os.path.exists(image_dir):
                for root, _, files in os.walk(image_dir):
                    for f in files:
                        if f.lower().endswith(valid_exts):
                            self.data.append(os.path.join(root, f))
            self.data.sort()
            if not self.data:
                print(f"警告: 在 {image_dir} 中未找到图片。")
        else:
            if phase == "train":
                subset = "our485"
            elif phase == "test":
                subset = "eval15"
            else:
                raise ValueError("phase must be 'train', 'test' or 'predict'")

            if subset == "eval15":
                subset_dir = os.path.join(image_dir, "eval15")
            elif subset == "our485":
                subset_dir = os.path.join(image_dir, "our485")

            if os.path.exists(subset_dir):
                high_dir = os.path.join(subset_dir, "high")
                low_dir = os.path.join(subset_dir, "low")
                # 过滤图片文件
                image_names = [f for f in os.listdir(high_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                image_names.sort()

                for img_name in image_names:
                    high_path = os.path.join(high_dir, img_name)
                    if self.online_synthesis:
                        # 在线合成模式：只需要 high 图像路径
                        self.data.append((None, high_path))
                    else:
                        low_path = os.path.join(low_dir, img_name)
                        if os.path.exists(low_path):
                            self.data.append((low_path, high_path))
            else:
                print(f"警告: 目录 {subset_dir} 不存在，数据集为空。")

    def __len__(self):
        return len(self.data)

    def _online_degrade(self, high_pil: Image.Image) -> Image.Image:
        """使用 Darker 引擎实时合成低光照退化图像。"""
        # PIL -> BGR numpy
        high_np = cv2.cvtColor(np.array(high_pil), cv2.COLOR_RGB2BGR)
        # 应用随机退化
        low_np = self.darker.degrade_single(high_np)
        # BGR numpy -> PIL
        low_pil = Image.fromarray(cv2.cvtColor(low_np, cv2.COLOR_BGR2RGB))
        return low_pil

    def __getitem__(self, idx):
        if self.phase == "predict":
            img_path = self.data[idx]
            img = Image.open(img_path).convert("RGB")
            # 预测时统一 Resize
            img = TF.resize(img, (self.img_size, self.img_size))
            img = TF.to_tensor(img)
            img = TF.normalize(img, [0.5], [0.5])
            return img

        low_img_path, high_img_path = self.data[idx]

        # === 2. 加载图像 ===
        high_img = Image.open(high_img_path).convert("RGB")

        if self.online_synthesis:
            # 在线合成：对 high 图像实时退化
            low_img = self._online_degrade(high_img)
        else:
            low_img = Image.open(low_img_path).convert("RGB")

        # === 3. 同步数据增强 ===
        if self.phase == "train":
            # A. 随机裁剪 (Random Crop)
            i, j, h, w = transforms.RandomCrop.get_params(
                low_img, output_size=(self.img_size, self.img_size))

            low_img = TF.crop(low_img, i, j, h, w)
            high_img = TF.crop(high_img, i, j, h, w)

            # B. 随机水平翻转 (Random Horizontal Flip)
            if random.random() > 0.5:
                low_img = TF.hflip(low_img)
                high_img = TF.hflip(high_img)

        else:
            low_img = TF.resize(low_img, (self.img_size, self.img_size))
            high_img = TF.resize(high_img, (self.img_size, self.img_size))

        # === 4. 转为 Tensor 并归一化 ===
        low_img = TF.to_tensor(low_img)
        high_img = TF.to_tensor(high_img)

        # 归一化到 [-1, 1] (Diffusion 模型标准)
        low_img = TF.normalize(low_img, [0.5], [0.5])
        high_img = TF.normalize(high_img, [0.5], [0.5])

        return low_img, high_img


# 示例用法
if __name__ == '__main__':
    data_dir = "../datasets/kitti_LOL"

    try:
        # 在线合成模式测试
        train_dataset = LowLightDataset(
            image_dir=data_dir, img_size=256, phase="train",
            online_synthesis=True)

        if len(train_dataset) > 0:
            train_loader = DataLoader(
                train_dataset, batch_size=2, shuffle=True)

            print("检查第一个 Batch 的数据对齐情况 (在线合成模式)...")
            for low, high in train_loader:
                print(
                    f"Low shape: {low.shape}, Range: [{low.min():.2f}, {low.max():.2f}]")
                print(
                    f"High shape: {high.shape}, Range: [{high.min():.2f}, {high.max():.2f}]")

                transforms.ToPILImage()(
                    low[0]/2+0.5).save("debug_crop_low.png")
                transforms.ToPILImage()(
                    high[0]/2+0.5).save("debug_crop_high.png")
                print("已保存 debug_crop_low.png 和 debug_crop_high.png。")
                break
        else:
            print(f"在 {data_dir} 未找到数据。")

    except Exception as e:
        print(f"发生错误: {e}")
