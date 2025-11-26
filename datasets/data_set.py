import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF  # 关键引用：用于函数式变换
from PIL import Image


class LowLightDataset(Dataset):
    def __init__(self, image_dir, img_size=256, phase="train"):
        """
        Args:
            image_dir (string): 数据集根目录。
            img_size (int): 训练/测试时的图像分辨率。
            phase (string): "train" 或 "test"。
        """
        self.image_dir = image_dir
        self.img_size = img_size
        self.phase = phase

        self.data = []

        # === 1. 构建文件列表 (保持原逻辑不变) ===
        if phase == "train":
            subset = "our485"
        elif phase == "test":
            subset = "eval15"
        else:
            raise ValueError("phase must be 'train' or 'test'")

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
                low_path = os.path.join(low_dir, img_name)
                high_path = os.path.join(high_dir, img_name)
                # 简单检查低光文件是否存在
                if os.path.exists(low_path):
                    self.data.append((low_path, high_path))
        else:
            print(f"警告: 目录 {subset_dir} 不存在，数据集为空。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        low_img_path, high_img_path = self.data[idx]

        # === 2. 加载图像 ===
        # 必须确保转换为 RGB
        low_img = Image.open(low_img_path).convert("RGB")
        high_img = Image.open(high_img_path).convert("RGB")

        # === 3. 同步数据增强 (关键修改) ===
        if self.phase == "train":
            # A. 随机裁剪 (Random Crop)
            # 获取随机裁剪参数
            i, j, h, w = transforms.RandomCrop.get_params(
                low_img, output_size=(self.img_size, self.img_size))

            # 对两张图应用【完全相同】的裁剪参数
            low_img = TF.crop(low_img, i, j, h, w)
            high_img = TF.crop(high_img, i, j, h, w)

            # B. 随机水平翻转 (Random Horizontal Flip)
            if random.random() > 0.5:
                low_img = TF.hflip(low_img)
                high_img = TF.hflip(high_img)

        else:
            # 测试/验证阶段：统一 Resize 或 CenterCrop，不做随机操作
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
    # 替换成你的真实路径
    data_dir = "../datasets/kitti_LOL"

    # 只需要传入 img_size，内部会自动处理同步变换
    try:
        train_dataset = LowLightDataset(
            image_dir=data_dir, img_size=256, phase="train")

        # 检查是否加载成功
        if len(train_dataset) > 0:
            train_loader = DataLoader(
                train_dataset, batch_size=2, shuffle=True)

            print("检查第一个 Batch 的数据对齐情况...")
            for low, high in train_loader:
                print(
                    f"Low shape: {low.shape}, Range: [{low.min():.2f}, {low.max():.2f}]")
                print(
                    f"High shape: {high.shape}, Range: [{high.min():.2f}, {high.max():.2f}]")

                # 简单的肉眼验证：保存下来看看裁剪是否一致
                transforms.ToPILImage()(
                    low[0]/2+0.5).save("debug_crop_low.png")
                transforms.ToPILImage()(
                    high[0]/2+0.5).save("debug_crop_high.png")
                print("已保存 debug_crop_low.png 和 debug_crop_high.png，请手动检查内容是否对应。")
                break
        else:
            print(f"在 {data_dir} 未找到数据。")

    except Exception as e:
        print(f"发生错误: {e}")
