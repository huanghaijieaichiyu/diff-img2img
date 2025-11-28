# Diff-Img2Img Studio: 低光照图像增强 (基于扩散模型)

[![模型下载](https://img.shields.io/badge/模型下载-天翼云盘-blue?style=flat-square&logo=icloud)](https://cloud.189.cn/web/share?code=AJ7fUzBbuUzm) (访问码: q2u9)
[![GitHub Repo stars](https://img.shields.io/github/stars/yourusername/INR2RGB?style=social)](https://github.com/yourusername/INR2RGB)

这是一个基于 PyTorch 和 Diffusers 库的深度学习项目，**核心利用条件扩散模型 (Conditional Diffusion Model)** 进行低光照图像增强。项目旨在将低光照、噪声严重的图像恢复为清晰、正常的图像。

本项目提供了一个功能强大的 **Web UI (Diff-Img2Img Studio)**，集成了数据准备、训练、评估和可视化的全流程。

## ✨ 项目特点

- **全流程 Web UI**: 提供基于 Streamlit 的交互式界面，覆盖从数据合成到模型训练、评估和可视化的所有环节。
- **核心技术**: 基于 `diffusers` 库实现的条件扩散模型 (Conditional DDPM)。
- **Retinex 理论**: 结合 Retinex 理论进行图像分解（反射率/光照），辅助扩散模型训练，提升增强效果。
- **物理数据合成**: 内置 `Darker` 引擎，可基于物理模型（Gamma 校正、线性衰减、车灯模拟、噪声注入）将正常图像合成为低光照图像，解决数据匮乏问题。
- **分布式训练**: 集成 Hugging Face `accelerate`，支持单机多卡、混合精度 (FP16/BF16) 训练。
- **实时监控**: 训练过程中支持实时查看终端日志、Loss 曲线和学习率变化。
- **多语言支持**: 界面支持 **简体中文** 和 **English** 切换。
- **一键打包**: 提供脚本将项目打包为独立的可执行文件 (EXE/Linux Binary)。

## 🖼️ 效果展示

|          输入（低光照）          |      输出（扩散模型增强后）      |
| :------------------------------: | :------------------------------: |
| ![低光照图像](examples/fake.png) | ![增强后图像](examples/real.png) |
|   _(示例输入)_   |   _(示例真值)_   |

## 🛠️ 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (强烈推荐用于加速训练和预测)

## 🚀 安装步骤

1.  **克隆仓库:**

    ```bash
    git clone https://github.com/yourusername/INR2RGB.git
    cd INR2RGB
    ```

2.  **创建并激活虚拟环境 (推荐):**

    ```bash
    # 使用 conda
    conda create -n diff-img2img python=3.10
    conda activate diff-img2img

    # 或者使用 venv
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate   # Windows
    ```

3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Web UI 使用指南 (推荐)

本项目提供了一站式的 Web 界面，无需记忆复杂的命令行参数。

1.  **启动 Studio:**
    ```bash
    streamlit run app.py
    ```
    浏览器将自动打开 `http://localhost:8501`。

2.  **功能模块:**
    *   **🏠 Home (主页)**: 项目概览与环境安装指引。
    *   **🌑 Dataset Preparation (数据集准备)**: 使用 `Darker` 引擎将正常光照数据集（如 KITTI）转换为低光照训练对。支持调节 Gamma、噪声、车灯强度等物理参数。
    *   **⚙️ Configuration (配置)**: 设置 `accelerate` 分布式训练参数（GPU 数量、混合精度等）。
    *   **🚂 Training (训练)**:
        *   配置训练参数（Epochs, Batch Size, LR 等）。
        *   **实时监控**: 在界面上直接查看训练日志输出、Loss 曲线和学习率曲线。
    *   **📊 Evaluation (评估)**: 在测试集上计算 PSNR, SSIM, LPIPS 等指标。
    *   **🎨 Visualization (可视化)**: 加载训练好的模型，对单张图像进行增强并对比显示。

## 💻 命令行使用指南

如果你更喜欢使用命令行，本项目依然支持完整的 CLI 操作。

### 1. 训练扩散模型

```bash
accelerate launch diffusion_trainer.py \
    --data_dir ../datasets/kitti_LOL \
    --output_dir run_diffusion_experiment \
    --resolution 256 \
    --batch_size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --mixed_precision fp16 \
    --use_retinex
```

### 2. 预测与评估

```bash
# 预测并计算指标
python diffusion_val.py \
    --model_path run_diffusion_experiment \
    --data_dir ../datasets/kitti_LOL \
    --output_dir eval_results \
    --use_retinex
```

### 3. 单图推理 (可视化脚本)

```bash
# 需自行编写或使用 Visual UI
python visual_val.py ...
```

## 📦 打包与发布

支持将应用打包为独立可执行文件，方便在没有 Python 环境的机器上运行。

**Linux:**
```bash
chmod +x build_executable.sh
./build_executable.sh
```

**Windows:**
```powershell
pip install pyinstaller
pyinstaller build.spec --clean --noconfirm
```

构建产物将位于 `dist/DiffImg2ImgStudio` 目录。

## 📂 数据集

推荐使用以下数据集进行训练：
- **LOL Dataset**: [下载链接](https://daooshee.github.io/BMVC2018website/)
- **国内网盘镜像 (Quark)**: [点击跳转](https://pan.quark.cn/s/1867c35697db) (提取码: **ZUWn**)

请确保数据集结构如下：
```
dataset_root/
    train/
        high/ (正常光照)
        low/  (低光照)
    test/
        high/
        low/
```

## 📄 许可证

[MIT License](LICENSE)

## 🤝 联系方式

如有任何问题，请提交 GitHub Issues 或联系：huangxiaohai99@126.com