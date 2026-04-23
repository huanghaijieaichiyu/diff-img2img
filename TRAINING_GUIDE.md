# 训练启动指南

训练参数入口现在统一到 `main.py`。`start_train.py` 是一个**极简 train-only 启动器**：只负责选择训练 preset、指定数据目录/输出目录，并通过 `accelerate` 启动 `main.py --mode train`。

## 1. 启动训练

推荐方式：

```bash
python3 start_train.py \
  --config middle \
  --data-dir /path/to/dataset \
  --output-dir runs/exp
```

如需恢复训练：

```bash
python3 start_train.py \
  --config middle \
  --data-dir /path/to/dataset \
  --output-dir runs/exp \
  --resume latest
```

训练直接读取数据集中的 `our485/low` 与 `our485/high` 源图对，不会生成 `.prepared` 缓存或在线合成图片。

## 2. 数据集结构

```text
<data_dir>/
  our485/
    low/
    high/
  eval15/
    low/
    high/
```

## 3. 验证模型

```bash
python3 main.py \
  --mode validate \
  --config middle \
  --model_path runs/exp/best_model \
  --data_dir /path/to/dataset \
  --output_dir runs/exp/full_eval
```

## 4. 自定义配置

```bash
cp configs/train/middle.yaml configs/train/my_config.yaml

python3 start_train.py \
  --config my_config.yaml \
  --data-dir /path/to/dataset \
  --output-dir runs/my_config_exp
```

## 5. 说明

- `main.py` 是权威运行时入口
- `start_train.py` 只保留 train 启动职责
- validate/predict/ui 等非训练模式请直接调用 `main.py`
- 训练相关的公开 CLI 已收紧，数据生成/实验性/非核心选项不再作为常规训练入口暴露
- 旧的数据生成入口已从训练链路移除，训练只消费已有源图对
