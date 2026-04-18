# Torch 低照度退化重构方案

目标是把训练数据的低照度退化从 OpenCV / numpy 主导的 CPU 管线，逐步迁移到 torch 张量算子，减少 Python 循环、减少 numpy<->torch 往返，并让离线准备和在线增强保持同一套语义。

## 分阶段方案

1. 先把几何类增强全部迁移到 torch
- 随机裁剪、resize、hflip、色彩抖动优先改成 `torch` / `torchvision.transforms.functional`。
- 训练样本直接保持为 `torch.Tensor`，只在最开始做一次解码。
- 所有随机数使用 `torch.Generator` 或 worker 级别的随机种子，保证 DataLoader 多 worker 可复现。

2. 把低照度退化拆成可组合的 torch 模块
- 亮度压缩：用逐像素乘法、gamma、曝光映射或分段 LUT 的 torch 实现。
- 噪声：高斯噪声、泊松噪声、读出噪声用 `torch.randn_like`、`torch.poisson` 和参数化采样实现。
- 模糊：用固定卷积核或可配置核的 `conv2d` 实现运动模糊 / 轻微散焦。
- 颜色偏移：用 3x3 颜色矩阵或通道级仿射变换实现白平衡漂移、色温偏移。
- 压缩伪影：如果需要 JPEG 伪影，优先放到离线脚本，在线阶段避免反复 CPU 编码。

3. 给退化模块做成函数式流水线
- 每个退化步骤写成 `forward(image, generator=None)` 的 `nn.Module` 或纯函数。
- 通过参数字典控制概率、范围和顺序，而不是在训练代码里硬编码。
- 在离线构建脚本里复用同一套模块，确保 offline cache 和 online synthesis 行为一致。

4. 兼容现有项目结构
- 先保留 `datasets/prepare_data.py` 作为离线构建入口。
- 再把 `scripts/darker.py` 的核心逻辑分解成 torch 算子实现。
- 等 torch 版本稳定后，再考虑移除 OpenCV 退化路径，只保留解码与少量边界处理。

## 推荐迁移顺序

- 第一步：裁剪 / resize / flip / normalize。
- 第二步：亮度、gamma、噪声。
- 第三步：模糊、颜色偏移、局部遮挡。
- 第四步：压缩伪影与更复杂的组合退化。

## 实际收益

- 减少 numpy 和 OpenCV 的转换开销。
- 更容易在 CPU worker 间并行。
- 更容易后续把部分退化迁移到 GPU，或者在离线阶段批量生成。
- 让增强逻辑和训练逻辑使用同一套 torch 代码，减少语义漂移。
