# 训练启动指南

简洁的训练启动方式，只需指定配置文件和数据集路径。所有详细配置从 YAML 配置文件加载。

## 快速开始

### Python 脚本

```bash
# 使用默认配置（small，6GB显存）
python start_train.py

# 指定配置和数据集
python start_train.py --config middle --data-dir /path/to/dataset

# 自定义输出目录
python start_train.py --config middle --data-dir /path/to/dataset --output-dir runs/exp1

# 从检查点恢复
python start_train.py --config middle --resume latest

# 验证模式
python start_train.py --mode validate --model-path runs/retinex/best_model
```

### Bash 脚本

```bash
# 使用默认配置
./start_train.sh

# 指定配置和数据集
./start_train.sh middle /path/to/dataset

# 指定配置、数据集和输出目录
./start_train.sh middle /path/to/dataset runs/exp1

# 验证模式
RUN_MODE=validate MODEL_PATH=runs/retinex/best_model ./start_train.sh middle

# 从检查点恢复
RESUME=latest ./start_train.sh middle
```

## 可用配置

项目提供三个预设配置：

| 配置 | 显存需求 | 描述 |
|------|---------|------|
| `small` | 6GB | 轻量级配置，适合消费级GPU |
| `middle` | 8GB | 官方推荐配置，平衡性能和质量 |
| `max` | 64GB | 最高质量配置，需要专业级GPU |

## 配置文件

所有训练参数都在配置文件中定义：

```
configs/train/
├── small.yaml    # 6GB 配置
├── middle.yaml   # 8GB 配置
└── max.yaml      # 64GB 配置
```

### 配置文件结构

```yaml
meta:                    # 元信息
  name: middle
  target_vram_gb: 8
  description: 官方推荐配置

runtime:                 # 运行时配置
  mixed_precision: fp16
  resolution: 256
  num_workers: 4

model:                   # 模型配置
  architecture_scale: middle
  use_retinex: true
  unet_block_channels: [32, 64, 128, 256]

optimization:            # 优化配置
  batch_size: 2
  gradient_accumulation_steps: 4
  lr: 1.0e-4

loss:                    # 损失函数配置
  retinex_loss_weight: 0.1
  tv_loss_weight: 0.1
  x0_loss_weight: 0.18

schedule:                # 训练计划
  checkpointing_steps: 1000
  validation_steps: 500

evaluation:              # 评估配置
  num_inference_steps: 8
  semantic_backbone: resnet18
```

## 修改配置

### 方法 1: 编辑现有配置

直接编辑配置文件：

```bash
vim configs/train/middle.yaml
```

### 方法 2: 创建自定义配置

```bash
# 复制现有配置
cp configs/train/middle.yaml configs/train/my_config.yaml

# 编辑配置
vim configs/train/my_config.yaml

# 使用自定义配置
python start_train.py --config configs/train/my_config.yaml
```

### 常见配置修改

**调整批次大小和梯度累积：**

```yaml
optimization:
  batch_size: 4              # 增加批次大小
  gradient_accumulation_steps: 2  # 减少梯度累积
```

**调整学习率：**

```yaml
optimization:
  lr: 2.0e-4                 # 提高学习率
  lr_scheduler: cosine_with_warmup
  lr_warmup_steps: 500
```

**调整损失权重：**

```yaml
loss:
  retinex_loss_weight: 0.2   # 增加 Retinex 损失权重
  tv_loss_weight: 0.05       # 减少 TV 损失权重
  x0_loss_weight: 0.25       # 调整 x0 损失权重
```

**调整验证频率：**

```yaml
schedule:
  checkpointing_steps: 500   # 更频繁保存检查点
  validation_steps: 250      # 更频繁验证
```

## 命令行参数

### Python 脚本参数

```
--config CONFIG          配置文件 (small/middle/max 或 YAML 路径)
--data-dir DIR           数据集目录
--output-dir DIR         输出目录
--mode MODE              运行模式 (train/validate/predict)
--resume PATH            从检查点恢复 (路径或 'latest')
--model-path PATH        模型路径 (用于 validate/predict)
```

### Bash 脚本参数

```bash
./start_train.sh [CONFIG] [DATA_DIR] [OUTPUT_DIR]
```

环境变量：
- `RUN_MODE`: 运行模式 (train/validate/predict)
- `RESUME`: 检查点路径
- `MODEL_PATH`: 模型路径

## 常见场景

### 训练新模型

```bash
python start_train.py --config middle --data-dir /path/to/dataset
```

### 从检查点继续训练

```bash
python start_train.py --config middle --resume runs/retinex/checkpoint-5000
```

或恢复最新检查点：

```bash
python start_train.py --config middle --resume latest
```

### 验证模型

```bash
python start_train.py \
    --mode validate \
    --config middle \
    --model-path runs/retinex/best_model
```

### 使用自定义配置

```bash
python start_train.py \
    --config configs/train/my_config.yaml \
    --data-dir /path/to/dataset
```

## 多GPU训练

脚本自动使用 `accelerate` 进行分布式训练。首次使用前需要配置：

```bash
accelerate config
```

按提示选择：
- 单机多GPU或多机多GPU
- GPU数量
- 混合精度设置

配置完成后，正常使用启动脚本即可自动启用多GPU训练。

## 监控训练

### TensorBoard

```bash
tensorboard --logdir runs/retinex/logs
```

### 训练状态

训练过程中会生成以下文件：

```
runs/retinex/
├── logs/                    # TensorBoard 日志
├── training_metrics.csv     # 训练指标
├── training_status.json     # 当前状态
├── validation/              # 验证图像
├── checkpoint-*/            # 检查点
└── best_model/              # 最佳模型
```

## 常见问题

### Q: 如何选择合适的配置？

根据GPU显存选择：
- 6-8GB: `small`
- 8-12GB: `middle`
- 24GB+: `max`

### Q: 如何调整训练参数？

编辑配置文件，修改对应的参数。所有参数都在配置文件中定义。

### Q: 训练中断后如何恢复？

```bash
python start_train.py --config middle --resume latest
```

### Q: 如何使用自己的数据集？

将数据集组织为以下结构：

```
dataset/
├── train/
│   ├── low/     # 低光照图像
│   └── high/    # 正常光照图像
└── test/
    ├── low/
    └── high/
```

然后指定数据集路径：

```bash
python start_train.py --config middle --data-dir /path/to/dataset
```

### Q: 如何修改模型架构？

编辑配置文件的 `model` 部分。例如修改 UNet 通道数：

```yaml
model:
  unet_block_channels: [64, 128, 256, 512]  # 增加通道数
```

## 设计理念

这个简化版本遵循以下原则：

1. **配置文件优先**：所有训练参数在配置文件中定义
2. **命令行简洁**：只指定必要的路径（配置、数据集、输出）
3. **易于维护**：修改配置只需编辑 YAML 文件
4. **版本控制友好**：配置文件可以纳入 Git 管理

这样的设计让训练配置更加清晰、可复现、易于分享。
