# Diff-Img2Img Studio: Low-Light Image Enhancement (Based on Diffusion Models)

[![Model Download](https://img.shields.io/badge/Model_Download-Tianyi_Cloud-blue?style=flat-square&logo=icloud)](https://cloud.189.cn/web/share?code=AJ7fUzBbuUzm) (Access Code: q2u9)
[![GitHub Repo stars](https://img.shields.io/github/stars/yourusername/INR2RGB?style=social)](https://github.com/yourusername/INR2RGB)

This is a deep learning project based on PyTorch and the Diffusers library, **utilizing Conditional Diffusion Models** for low-light image enhancement. The project aims to restore low-light, noisy images to clear, normally lit images.

This project provides a powerful **Web UI (Diff-Img2Img Studio)** that integrates the full workflow of data preparation, training, evaluation, and visualization.

## ✨ Features

- **Full-Stack Web UI**: Interactive interface based on Streamlit, covering all steps from data synthesis to model training, evaluation, and visualization.
- **Core Technology**: Conditional Denoising Diffusion Probabilistic Models (Conditional DDPM) based on `diffusers`.
- **Retinex Theory**: Decomposes images into Reflectance/Illumination to assist diffusion model training and improve enhancement results.
- **Physics-Based Data Synthesis**: Built-in `Darker` engine synthesizes low-light images from normal images using physical models (Gamma correction, linear attenuation, headlight simulation, noise injection) to address data scarcity.
- **Distributed Training**: Integrates Hugging Face `accelerate`, supporting single-machine multi-GPU and mixed precision (FP16/BF16) training.
- **Real-Time Monitoring**: View training logs, loss curves, and learning rate changes directly on the interface during training.
- **Multi-Language Support**: Interface supports switching between **English** and **Simplified Chinese**.
- **One-Click Packaging**: Scripts provided to package the project into standalone executables (EXE/Linux Binary).

## 🖼️ Demo

|           Input (Low Light)           |     Output (Diffusion Enhanced)      |
| :-----------------------------------: | :----------------------------------: |
| ![Low Light Image](examples/fake.png) | ![Enhanced Image](examples/real.png) |
|           _(Example Input)_           |      _(Example Ground Truth)_        |

## 🛠️ Environment Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (Strongly recommended for accelerated training and prediction)

## 🚀 Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/INR2RGB.git
    cd INR2RGB
    ```

2.  **Create and activate a virtual environment (Recommended):**

    ```bash
    # Using conda
    conda create -n diff-img2img python=3.10
    conda activate diff-img2img

    # Or using venv
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate   # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Web UI Usage Guide (Recommended)

This project provides a one-stop Web interface, eliminating the need to memorize complex command-line arguments.

1.  **Start the Studio:**
    ```bash
    streamlit run app.py
    ```
    The browser will automatically open `http://localhost:8501`.

2.  **Functional Modules:**
    *   **🏠 Home**: Project overview and installation guide.
    *   **🌑 Dataset Preparation**: Use the `Darker` engine to convert normal light datasets (like KITTI) into low-light training pairs. Supports adjusting Gamma, noise, headlight intensity, etc.
    *   **⚙️ Configuration**: Configure `accelerate` distributed training parameters (GPU count, mixed precision, etc.).
    *   **🚂 Training**:
        *   Configure training parameters (Epochs, Batch Size, LR, etc.).
        *   **Real-time Monitoring**: View training log output, Loss curves, and Learning Rate curves directly on the interface.
    *   **📊 Evaluation**: Calculate PSNR, SSIM, LPIPS metrics on the test set.
    *   **🎨 Visualization**: Load trained models to enhance single images and display comparisons.

## 💻 Command Line Usage Guide

If you prefer using the command line, this project still supports full CLI operations.

### 1. Train Diffusion Model

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

### 2. Prediction & Evaluation

```bash
# Predict and calculate metrics
python diffusion_val.py \
    --model_path run_diffusion_experiment \
    --data_dir ../datasets/kitti_LOL \
    --output_dir eval_results \
    --use_retinex
```

### 3. Single Image Inference (Visualization Script)

```bash
# Needs to be written manually or use the Visual UI
python visual_val.py ...
```

## 📦 Packaging & Release

Supports packaging the application into a standalone executable for running on machines without a Python environment.

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

The build artifacts will be located in the `dist/DiffImg2ImgStudio` directory.

## 📂 Datasets

Recommended datasets for training:
- **LOL Dataset**: [Download Link](https://daooshee.github.io/BMVC2018website/)
- **Quark Drive Mirror (China)**: [Click to Visit](https://pan.quark.cn/s/1867c35697db) (Access Code: **ZUWn**)

Please ensure the dataset structure is as follows:
```
dataset_root/
    train/
        high/ (Normal Light)
        low/  (Low Light)
    test/
        high/
        low/
```

## 📄 License

[MIT License](LICENSE)

## 🤝 Contact

For any questions, please submit GitHub Issues or contact: huangxiaohai99@126.com