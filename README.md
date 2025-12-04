# Diff-Img2Img Studio: Low-Light Image Enhancement

[![Model Download](https://img.shields.io/badge/Model%20Download-Cloud-blue?style=flat-square&logo=icloud)](https://cloud.189.cn/web/share?code=AJ7fUzBbuUzm) (Access Code: q2u9)
[![GitHub Repo stars](https://img.shields.io/github/stars/yourusername/INR2RGB?style=social)](https://github.com/yourusername/INR2RGB)

This is a Deep Learning project based on **PyTorch** and **Diffusers**, focusing on low-light image enhancement using **Conditional Diffusion Models**. It integrates **Retinex Theory** to decompose images into reflectance and illumination components for more stable training and better structural preservation.

The project features a unified engine and a comprehensive **Web UI (Diff-Img2Img Studio)** for data preparation, training, evaluation, and visualization.

## ✨ Key Features

-   **Unified Engine**: A robust `DiffusionEngine` (in `core/engine.py`) handling training, validation, and inference with `accelerate` support.
-   **Web UI Studio**: A Streamlit-based dashboard covering the entire workflow:
    -   **Dataset Synthesis**: Physics-based low-light simulation (Gamma, Noise, Headlights) via `Darker` engine.
    -   **Training**: Real-time monitoring of loss and learning rates.
    -   **Evaluation**: PSNR, SSIM, and LPIPS metrics.
    -   **Visualization**: Side-by-side comparison of enhanced images.
-   **Advanced Loss Functions**: Combines **Charbonnier Loss** (Pixel), **SSIM Loss** (Structure), **Edge Loss**, and **Frequency Loss**.
-   **Retinex-Diffusion**: Decomposes low-light images to guide the diffusion process.

## 🖼️ Gallery

|          Input (Low Light)          |      Output (Enhanced)      |
| :------------------------------: | :------------------------------: |
| ![Low Light](examples/fake.png) | ![Enhanced](examples/real.png) |

## 🛠️ Requirements

-   Python 3.8+
-   PyTorch 2.0+
-   CUDA (Recommended)

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Diff-Img2Img.git
    cd Diff-Img2Img
    ```

2.  **Create environment:**
    ```bash
    conda create -n diff-img2img python=3.10
    conda activate diff-img2img
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage (Web UI)

The recommended way to use the project is via the Web UI.

```bash
# Launch the Studio
python main.py --mode ui
```
*(Or directly: `streamlit run ui/app.py`)*

Open your browser at `http://localhost:8501`.

## 💻 Usage (CLI)

You can also use the unified `main.py` entry point for all operations.

### 1. Training

```bash
accelerate launch main.py --mode train \
    --data_dir ../datasets/kitti_LOL \
    --output_dir runs/experiment_1 \
    --resolution 256 \
    --batch_size 4 \
    --epochs 50 \
    --use_retinex
```

### 2. Prediction (Inference)

**Single Image / Folder:**
```bash
python main.py --mode predict \
    --model_path runs/experiment_1 \
    --data_dir ../datasets/test_images \
    --output_dir predictions \
    --use_retinex
```

**Video:**
```bash
python main.py --mode predict \
    --model_path runs/experiment_1 \
    --video_path input_video.mp4 \
    --output_dir video_results \
    --use_retinex
```

### 3. Validation

```bash
python main.py --mode validate \
    --model_path runs/experiment_1 \
    --data_dir ../datasets/kitti_LOL \
    --use_retinex
```

## 📂 Project Structure

```
/
├── core/               # Core engine and logic
│   └── engine.py       # DiffusionEngine class
├── models/             # Neural network architectures
│   ├── diffusion.py    # Conditional UNet wrapper
│   └── retinex.py      # DecomNet for Retinex decomposition
├── ui/                 # Streamlit Web UI
│   └── app.py
├── scripts/            # Utility scripts
│   ├── darker.py       # Data synthesis engine
│   └── visual_val.py   # Visualization helpers
├── datasets/           # Data loading logic
├── utils/              # Helper functions (Loss, Metrics)
├── main.py             # Unified entry point
└── legacy/             # Old standalone scripts (deprecated)
```

## 📄 License

[MIT License](LICENSE)

## 🤝 Contact

For issues, please submit a GitHub Issue or contact: huangxiaohai99@126.com
