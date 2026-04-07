import streamlit as st
import json
import os
import subprocess
import signal
import shlex
import time
from pathlib import Path
import shutil
import sys
import threading
import queue
import pandas as pd
import plotly.express as px
import torch
import yaml
from html import escape
from concurrent.futures import ProcessPoolExecutor

# Import project modules
sys.path.append(os.getcwd())

# --- Localization ---
LANGUAGES = {
    "English": "en",
    "简体中文": "zh"
}

TRANSLATIONS = {
    "en": {
        "nav_title": "Navigation",
        "home": "Home",
        "dataset": "Dataset Preparation",
        "training": "Training",
        "evaluation": "Evaluation",
        "visualization": "Visualization",
        "configuration": "Configuration",
        "app_title": "🌌 Diff-Img2Img Studio",
        "home_kicker": "Prepared-cache workflow for low-light restoration",
        "home_subtitle": "🚀 Next-Gen Low-Light Enhancement",
        "home_desc": "Diff-Img2Img combines Retinex decomposition, diffusion restoration, and an offline prepared-cache pipeline to make low-light enhancement runs easier to reproduce and monitor.",
        "key_features": "🌟 Key Features",
        "feature_1": "**Physics-based Data Synthesis**: Uses `Darker` engine to simulate realistic low-light degradation (Gamma, Noise, Headlights).",
        "feature_2": "**Retinex-Diffusion Architecture**: Decomposes images into Reflectance/Illumination for stable diffusion training.",
        "feature_3": "**Full-Stack Workflow**: Prepare cache, launch training, monitor logs, evaluate runs, and inspect outputs in one place.",
        "example_results": "🖼️ Example Results",
        "synthesized_input": "Synthesized Low-Light (Input)",
        "ground_truth": "Ground Truth (Reference)",
        "home_tip": "💡 **Tip:** Navigate using the sidebar to start your workflow.",
        "workflow_header": "Workflow",
        "workflow_prepare_title": "Prepare The Offline Cache",
        "workflow_prepare_desc": "Generate multi-variant low-light samples from `our485/high`, inspect cache health, and rebuild only when settings change.",
        "workflow_train_title": "Launch Preset Training",
        "workflow_train_desc": "Use the official `small`, `middle`, or `max` configs, preview the final command, and launch with prepared-cache settings attached.",
        "workflow_eval_title": "Validate And Visualize",
        "workflow_eval_desc": "Read metrics, compare outputs on `eval15`, and keep qualitative checks close to the training loop.",
        "quick_start": "Quick Start",
        "train_command": "Training Command",
        "ui_command": "UI Command",
        "preset_overview": "Preset Overview",
        "preset_summary": "Preset Summary",
        "preset_description": "Preset Description",
        "target_vram": "Target VRAM",
        "preset_resolution": "Preset Resolution",
        "effective_batch": "Effective Batch",
        "recommended_effective_batch": "Recommended minimum effective batch",
        "command_preview": "Command Preview",
        "start_training_hint": "Launch training from the Configuration tab to start logging metrics here.",
        "monitoring_note": "The monitor reads `training_metrics.csv` and `training_status.json` from the active output directory.",
        "waiting_for_logs": "Waiting for logs...",
        "auto_refresh": "🔄 Auto Refresh (5s)",
        "auto_refresh_help": "Automatically refresh metrics and logs every 5 seconds.",
        "waiting_for_data_points": "Waiting for data points...",
        "initializing_metrics_log": "Initializing metrics log...",
        "evaluation_failed": "Evaluation failed.",
        "dependencies_missing": "Dependencies missing (`scripts/visual_val.py`).",
        "load_failed": "Model load failed",
        "displaying": "Displaying",
        "dataset_path_invalid": "Dataset path is invalid or missing `eval15/low`.",
        "dataset_header": "🌑 Dataset Preparation",
        "dataset_sub": "Build and manage the prepared multi-variant low-light cache used by training.",
        "how_it_works": "ℹ️ How it works",
        "darker_desc": """Training now uses a prepared offline cache instead of on-the-fly synthesis:
        1.  Read clean images from `our485/high`.
        2.  Regenerate `N` low-light variants per image with the **Darker** engine.
        3.  Save them under `.prepared/our485/low`.
        4.  Write `train_manifest.jsonl` + `prepare_meta.json` and let training consume only that cache.""",
        "io_settings": "📁 I/O Settings",
        "raw_dataset": "Raw Dataset (High Light)",
        "split": "Split",
        "physics_params": "🎛️ Physics Params",
        "gamma": "Gamma",
        "linear_att": "Linear Attenuation",
        "effects": "✨ Effects",
        "headlight": "Headlight Intensity",
        "noise": "Noise Level (Sigma)",
        "saturation": "Saturation",
        "blue_shift": "Blue Shift",
        "beam_width": "Beam Width",
        "start_synthesis": "⚡ Start Synthesis",
        "processing": "Processing Dataset...",
        "synthesis_complete": "✅ Dataset synthesis complete!",
        "train_dashboard": "🚂 Training Dashboard",
        "running": "RUNNING",
        "stop": "🛑 STOP",
        "idle": "IDLE",
        "config": "⚙️ Configuration",
        "monitoring": "📈 Monitoring",
        "paths": "📍 Paths",
        "dataset_root": "Dataset Root",
        "exp_name": "Experiment Name (Output)",
        "resume": "Resume From",
        "model": "🧠 Model",
        "use_retinex": "Use Retinex Decomposition",
        "model_scale": "Model Scale",
        "resolution": "Image Resolution",
        "hyperparams": "⚡ Hyperparameters",
        "epochs": "Epochs",
        "batch_size": "Batch Size",
        "lr": "Learning Rate",
        "train_profile": "Training Profile",
        "train_desc": "Launch preset training, inspect the resolved command, and monitor logs plus metrics from the current output directory.",
        "train_tip": "💡 **Tip:** Training now auto-prepares a 3-variant offline low-light cache when prepared data is missing. Ensure your GPU has enough VRAM for Batch Size > 1 at 512px.",
        "launch_train": "🚀 Launch Training",
        "train_launched": "Training launched! Switch to 'Monitoring' tab.",
        "prepare_dataset": "🧪 Prepare Dataset",
        "prepare_running": "Preparing offline multi-variant dataset cache...",
        "prepare_complete": "✅ Prepared dataset cache is ready.",
        "prepare_failed": "Dataset preparation failed.",
        "prepared_settings": "🗂️ Prepared Cache",
        "prepared_cache_dir": "Prepared Cache Dir",
        "variant_count": "Variant Count",
        "prepare_workers": "Prepare Workers",
        "synthesis_seed": "Synthesis Seed",
        "prepare_force": "Force Rebuild Cache",
        "darker_ranges": "Darker Ranges (YAML/JSON)",
        "darker_ranges_help": "Optional override for Darker parameter ranges. Leave blank to use defaults.",
        "prepared_status": "Prepared Cache Status",
        "status_ready": "READY",
        "status_missing": "MISSING",
        "status_stale": "STALE",
        "status_preparing": "PREPARING",
        "status_interrupted": "INTERRUPTED",
        "status_invalid": "INVALID",
        "high_images": "High Images",
        "expected_entries": "Expected Entries",
        "manifest_entries": "Manifest Entries",
        "meta_variant_count": "Cached Variants",
        "completed_entries": "Completed Entries",
        "cache_dir_label": "Cache Dir",
        "manual_prepare_tip": "Use this to build the prepared cache ahead of training, or let training auto-build it on launch.",
        "prepare_stdout": "Prepare Output",
        "prepare_needed": "Prepared cache is missing or stale. Training will rebuild it before the first epoch.",
        "prepare_ready": "Prepared cache matches the current variant-count/seed settings.",
        "prepare_resume": "Prepared cache was interrupted previously. Relaunching prepare will resume missing variants only.",
        "process_finished": "⚠️ Process finished or stopped unexpectedly.",
        "refresh_charts": "🔄 Refresh Charts",
        "terminal_output": "💻 Terminal Output",
        "eval_header": "📊 Evaluation",
        "eval_desc": "Calculate quantitative metrics (PSNR, SSIM, LPIPS) on the test set.",
        "model_ckpt": "Model Checkpoint",
        "output_folder": "Output Folder",
        "run_eval": "▶️ Run Evaluation",
        "calc_metrics": "Calculating metrics...",
        "eval_complete": "Evaluation Complete!",
        "results": "Results",
        "vis_header": "🎨 Visualization",
        "vis_desc": "Inspect checkpoints on `eval15` and compare low-light, enhanced, and reference images.",
        "vis_config": "Config",
        "reload_models": "Reload Models",
        "select_image": "Select Test Image",
        "running_diff": "Running Diffusion...",
        "enhanced": "Enhanced",
        "dataset_recommendation": "⬇️ Dataset Recommendations",
        "dataset_link_text": "Recommended datasets for training:",
        "accelerate_config_header": "🚀 Accelerate Configuration",
        "accelerate_config_desc": "Configure Hugging Face Accelerate for distributed training.",
        "save_config": "💾 Save Configuration",
        "config_saved": "✅ Configuration saved to `accelerate_config.yaml`",
        "setup_header": "🛠️ Setup & Installation",
        "setup_desc": "Ensure you have the required environment:",
        "install_dependencies": "Install Dependencies",
        "config_tip": "This file is written to `accelerate_config.yaml` and will be picked up by the training launcher.",
    },
    "zh": {
        "nav_title": "导航",
        "home": "主页",
        "dataset": "数据集准备",
        "training": "训练",
        "evaluation": "评估",
        "visualization": "可视化",
        "configuration": "配置",
        "app_title": "🌌 Diff-Img2Img 工作室",
        "home_kicker": "面向低光增强的 prepared-cache 训练工作流",
        "home_subtitle": "🚀 下一代低光照增强",
        "home_desc": "Diff-Img2Img 将 Retinex 分解、扩散恢复和离线 prepared-cache 数据流程组合在一起，让低光增强训练更稳定、更容易复现。",
        "key_features": "🌟 主要特性",
        "feature_1": "**基于物理的数据合成**：使用 `Darker` 引擎模拟真实的低光照退化（Gamma，噪声，车灯）。",
        "feature_2": "**Retinex-扩散架构**：将图像分解为反射率/光照，以进行稳定的扩散训练。",
        "feature_3": "**全栈工作流**：在一个界面里完成缓存准备、训练启动、日志监控、评估与可视化。",
        "example_results": "🖼️ 示例结果",
        "synthesized_input": "合成低光照 (输入)",
        "ground_truth": "地面实况 (参考)",
        "home_tip": "💡 **提示：** 使用侧边栏导航开始您的工作流。",
        "workflow_header": "工作流",
        "workflow_prepare_title": "准备离线缓存",
        "workflow_prepare_desc": "从 `our485/high` 生成多变体低光样本，检查缓存健康状态，并在参数变化时按需重建。",
        "workflow_train_title": "启动官方预设训练",
        "workflow_train_desc": "使用 `small`、`middle` 或 `max` 官方配置，先预览最终命令，再携带 prepared-cache 参数启动训练。",
        "workflow_eval_title": "评估与可视化",
        "workflow_eval_desc": "读取指标、查看 `eval15` 可视化结果，把定量和定性检查都放在训练闭环里。",
        "quick_start": "快速开始",
        "train_command": "训练命令",
        "ui_command": "界面命令",
        "preset_overview": "预设概览",
        "preset_summary": "预设摘要",
        "preset_description": "预设说明",
        "target_vram": "目标显存",
        "preset_resolution": "预设分辨率",
        "effective_batch": "有效批次",
        "recommended_effective_batch": "建议的最小有效批次",
        "command_preview": "命令预览",
        "start_training_hint": "请先在“配置”标签页启动训练，这里才会开始显示日志和指标。",
        "monitoring_note": "监控页会读取当前输出目录下的 `training_metrics.csv` 与 `training_status.json`。",
        "waiting_for_logs": "正在等待日志输出...",
        "auto_refresh": "🔄 自动刷新 (5 秒)",
        "auto_refresh_help": "每 5 秒自动刷新日志和图表。",
        "waiting_for_data_points": "正在等待训练数据点...",
        "initializing_metrics_log": "正在初始化指标日志...",
        "evaluation_failed": "评估失败。",
        "dependencies_missing": "缺少依赖（`scripts/visual_val.py`）。",
        "load_failed": "模型加载失败",
        "displaying": "当前显示",
        "dataset_path_invalid": "数据集路径无效，或缺少 `eval15/low`。",
        "dataset_header": "🌑 数据集准备",
        "dataset_sub": "构建并管理训练实际使用的 prepared 多变体低光缓存。",
        "how_it_works": "ℹ️ 工作原理",
        "darker_desc": """训练现在使用 prepared 离线缓存，而不是读取时在线合成：
        1.  从 `our485/high` 读取高光图像。
        2.  用 **Darker** 引擎为每张图重新生成 `N` 个低光变体。
        3.  结果保存到 `.prepared/our485/low`。
        4.  同时写出 `train_manifest.jsonl` 和 `prepare_meta.json`，训练阶段只消费这一份缓存。""",
        "io_settings": "📁 I/O 设置",
        "raw_dataset": "原始数据集 (高光)",
        "split": "分割",
        "physics_params": "🎛️ 物理参数",
        "gamma": "Gamma",
        "linear_att": "线性衰减",
        "effects": "✨ 特效",
        "headlight": "车灯强度",
        "noise": "噪声水平 (Sigma)",
        "saturation": "饱和度",
        "blue_shift": "蓝移",
        "beam_width": "光束宽度",
        "start_synthesis": "⚡ 开始合成",
        "processing": "正在处理数据集...",
        "synthesis_complete": "✅ 数据集合成完成！",
        "train_dashboard": "🚂 训练仪表板",
        "running": "运行中",
        "stop": "🛑 停止",
        "idle": "空闲",
        "config": "⚙️ 配置",
        "monitoring": "📈 监控",
        "paths": "📍 路径",
        "dataset_root": "数据集根目录",
        "exp_name": "实验名称 (输出)",
        "resume": "恢复自",
        "model": "🧠 模型",
        "use_retinex": "使用 Retinex 分解",
        "model_scale": "模型规模",
        "resolution": "图像分辨率",
        "hyperparams": "⚡ 超参数",
        "epochs": "轮数 (Epochs)",
        "batch_size": "批次大小",
        "lr": "学习率",
        "train_profile": "训练配置",
        "train_desc": "用官方预设启动训练，先检查最终命令，再在当前输出目录上持续查看日志和指标。",
        "train_tip": "💡 **提示：** 如果缺少 prepared 数据，训练会先自动生成 3 变体离线低光缓存。也请确保您的 GPU 显存足以在 512px 下支持批次大小 > 1。",
        "launch_train": "🚀 启动训练",
        "train_launched": "训练已启动！切换到 '监控' 标签。",
        "prepare_dataset": "🧪 准备数据集",
        "prepare_running": "正在准备离线多变体数据缓存...",
        "prepare_complete": "✅ Prepared 数据缓存已就绪。",
        "prepare_failed": "数据准备失败。",
        "prepared_settings": "🗂️ Prepared 缓存",
        "prepared_cache_dir": "Prepared 缓存目录",
        "variant_count": "变体数量",
        "prepare_workers": "准备并发数",
        "synthesis_seed": "合成种子",
        "prepare_force": "强制重建缓存",
        "darker_ranges": "Darker 参数范围 (YAML/JSON)",
        "darker_ranges_help": "可选：覆盖 Darker 参数范围。留空则使用默认值。",
        "prepared_status": "Prepared 缓存状态",
        "status_ready": "已就绪",
        "status_missing": "缺失",
        "status_stale": "过期",
        "status_preparing": "准备中",
        "status_interrupted": "已中断",
        "status_invalid": "无效",
        "high_images": "高光图数量",
        "expected_entries": "期望条目数",
        "manifest_entries": "Manifest 条目数",
        "meta_variant_count": "缓存变体数",
        "completed_entries": "已完成条目数",
        "cache_dir_label": "缓存目录",
        "manual_prepare_tip": "可以先手动构建 prepared 缓存，也可以在启动训练时让系统自动补齐。",
        "prepare_stdout": "准备输出",
        "prepare_needed": "Prepared 缓存缺失或已过期。训练启动时会在首个 epoch 前自动重建。",
        "prepare_ready": "Prepared 缓存与当前变体数/种子设置一致。",
        "prepare_resume": "Prepared 缓存上次被中断。重新执行 prepare 时只会补齐缺失变体。",
        "process_finished": "⚠️ 进程已结束或异常停止。",
        "refresh_charts": "🔄 刷新图表",
        "terminal_output": "💻 终端输出",
        "eval_header": "📊 评估",
        "eval_desc": "在测试集上计算定量指标 (PSNR, SSIM, LPIPS)。",
        "model_ckpt": "模型检查点",
        "output_folder": "输出文件夹",
        "run_eval": "▶️ 运行评估",
        "calc_metrics": "正在计算指标...",
        "eval_complete": "评估完成！",
        "results": "结果",
        "vis_header": "🎨 可视化",
        "vis_desc": "在 `eval15` 上检查 checkpoint，并对比低光输入、增强结果和参考图像。",
        "vis_config": "配置",
        "reload_models": "重新加载模型",
        "select_image": "选择测试图像",
        "running_diff": "正在运行扩散...",
        "enhanced": "增强后",
        "dataset_recommendation": "⬇️ 数据集推荐",
        "dataset_link_text": "推荐的训练数据集：",
        "accelerate_config_header": "🚀 Accelerate 配置",
        "accelerate_config_desc": "配置 Hugging Face Accelerate 以进行分布式训练。",
        "save_config": "💾 保存配置",
        "config_saved": "✅ 配置已保存到 `accelerate_config.yaml`",
        "setup_header": "🛠️ 安装与设置",
        "setup_desc": "确保您已准备好运行环境：",
        "install_dependencies": "安装依赖",
        "config_tip": "该配置会写入 `accelerate_config.yaml`，训练启动时会自动读取。",
    }
}



# --- Page Configuration ---
st.set_page_config(
    page_title="Diff-Img2Img Studio",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:wght@600;700&display=swap');

    :root {
        --bg: #f5f1e8;
        --surface: rgba(255, 252, 246, 0.92);
        --surface-strong: rgba(255, 250, 242, 0.98);
        --line: rgba(102, 79, 43, 0.16);
        --text: #1e2b2b;
        --muted: #5c6868;
        --accent: #0f766e;
        --accent-soft: rgba(15, 118, 110, 0.10);
        --highlight: #c38b2f;
        --shadow: 0 18px 40px rgba(48, 44, 31, 0.10);
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(195, 139, 47, 0.16), transparent 28%),
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 24%),
            linear-gradient(180deg, #f8f4ec 0%, #f2ede2 100%);
    }

    /* Global Fonts */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', 'Avenir Next', 'Helvetica Neue', sans-serif;
        color: var(--text);
    }
    
    /* Headings */
    h1, h2, h3 {
        font-family: 'Source Serif 4', Georgia, serif;
        font-weight: 700;
        color: var(--text);
    }
    
    /* Buttons */
    .stButton button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease-in-out;
        padding: 0.6rem 1rem;
        background: linear-gradient(135deg, #146d67 0%, #0f766e 100%);
        color: #fff;
        box-shadow: 0 10px 24px rgba(15, 118, 110, 0.18);
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(15, 118, 110, 0.24);
    }
    
    /* Inputs & Selects */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 12px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.75);
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: rgba(15, 118, 110, 0.6);
        box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.18);
    }
    
    /* Expanders & Cards */
    div[data-testid="stExpander"] {
        border-radius: 18px;
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        background-color: var(--surface);
        margin-bottom: 1rem;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(250,246,238,0.92) 100%);
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid var(--line);
        text-align: center;
        transition: transform 0.2s;
        box-shadow: 0 10px 24px rgba(48, 44, 31, 0.06);
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
        border-color: rgba(15, 118, 110, 0.20);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(250, 246, 237, 0.96) 0%, rgba(241, 235, 222, 0.94) 100%);
        border-right: 1px solid var(--line);
    }
    
    /* Custom Classes */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 16px;
        box-shadow: var(--shadow);
        text-align: center;
        margin-bottom: 1rem;
    }

    .hero-panel {
        background:
            linear-gradient(135deg, rgba(18, 104, 97, 0.96) 0%, rgba(17, 83, 78, 0.94) 55%, rgba(36, 50, 48, 0.96) 100%);
        color: #f8f6f0;
        border-radius: 24px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 24px 48px rgba(26, 39, 38, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.78rem;
        color: rgba(248, 246, 240, 0.72);
        margin-bottom: 0.55rem;
    }

    .hero-title {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 2rem;
        line-height: 1.12;
        margin: 0;
        color: #fff8eb;
    }

    .hero-desc {
        color: rgba(248, 246, 240, 0.88);
        margin: 0.75rem 0 0;
        max-width: 58rem;
        line-height: 1.6;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(255, 248, 235, 0.12);
        border: 1px solid rgba(255, 248, 235, 0.16);
        color: #fff8eb;
        font-size: 0.9rem;
    }

    .info-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1rem 0.95rem;
        min-height: 180px;
        box-shadow: 0 12px 26px rgba(48, 44, 31, 0.06);
    }

    .info-card h4 {
        margin: 0 0 0.5rem;
        font-family: 'Source Serif 4', Georgia, serif;
        color: var(--text);
        font-size: 1.12rem;
    }

    .info-card p {
        margin: 0;
        color: var(--muted);
        line-height: 1.6;
    }

    .preset-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,243,234,0.92) 100%);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.8rem;
    }

    .preset-card strong {
        color: var(--text);
        font-size: 1rem;
    }

    .preset-card p {
        margin: 0.45rem 0 0;
        color: var(--muted);
        line-height: 1.5;
    }

    .section-note {
        color: var(--muted);
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'training_pid' not in st.session_state:
    st.session_state.training_pid = None
if 'training_pgid' not in st.session_state:
    st.session_state.training_pgid = None
if 'training_log_file' not in st.session_state:
    st.session_state.training_log_file = None
if 'training_csv_file' not in st.session_state:
    st.session_state.training_csv_file = None
if 'training_status_file' not in st.session_state:
    st.session_state.training_status_file = None
if 'language' not in st.session_state:
    st.session_state.language = "en"

# --- Sidebar Language Selector ---
with st.sidebar:
    st.image("https://img.icons8.com/color/48/000000/google-translate.png", width=30)
    selected_lang_name = st.selectbox(
        "Language / 语言", 
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.values()).index(st.session_state.language)
    )
    st.session_state.language = LANGUAGES[selected_lang_name]
    lang = st.session_state.language

def t(key):
    """Helper to get translated string"""
    return TRANSLATIONS[lang].get(key, key)

# --- Helper Functions ---
def list_folders(path):
    try:
        path = os.path.abspath(path)
        if not os.path.exists(path): return []
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')]
    except: return []

def folder_selector(label, key, default_path="."):
    base_path = st.text_input(label, value=default_path, key=key+"_input", help=f"Enter the full path for {label}")
    if os.path.exists(base_path):
        subdirs = list_folders(base_path)
        if subdirs:
            selected_subdir = st.selectbox(f"📂 Subfolders in '{os.path.basename(base_path)}'", [""] + subdirs, key=key+"_select")
            if selected_subdir: return os.path.join(base_path, selected_subdir)
    return base_path

def read_log_file(file_path, num_lines=100):
    if not file_path or not os.path.exists(file_path): return t("waiting_for_logs")
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            return "".join(lines[-num_lines:])
    except Exception as e: return f"Error reading log: {e}"

def read_status_file(file_path):
    if not file_path or not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def clear_training_state():
    st.session_state.training_pid = None
    st.session_state.training_pgid = None


def recommended_prepare_workers():
    cpu_count = os.cpu_count() or 4
    return max(1, min(8, cpu_count // 2))


def count_image_files(path):
    if not path or not os.path.exists(path):
        return 0
    count = 0
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                count += 1
    return count


def resolve_cache_dir(data_dir, prepared_cache_dir):
    if prepared_cache_dir:
        return os.path.abspath(prepared_cache_dir)
    return os.path.abspath(os.path.join(data_dir, ".prepared"))


def summarize_prepared_cache(data_dir, prepared_cache_dir, variant_count, synthesis_seed):
    cache_dir = resolve_cache_dir(data_dir, prepared_cache_dir)
    high_dir = os.path.join(data_dir, "our485", "high")
    manifest_path = os.path.join(cache_dir, "train_manifest.jsonl")
    meta_path = os.path.join(cache_dir, "prepare_meta.json")

    high_images = count_image_files(high_dir)
    expected_entries = high_images * int(variant_count)
    manifest_entries = 0
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest_entries = sum(1 for line in handle if line.strip())

    meta = read_status_file(meta_path)
    status = "missing"
    if meta:
        status = str(meta.get("status", "invalid")).lower()
        if status not in {"ready", "preparing", "interrupted"}:
            status = "invalid"
    if os.path.exists(manifest_path) and not meta:
        status = "stale"
    if meta and status == "ready":
        if int(meta.get("variant_count", -1)) != int(variant_count):
            status = "stale"
        elif int(meta.get("synthesis_seed", -1)) != int(synthesis_seed):
            status = "stale"
        elif expected_entries > 0 and manifest_entries != expected_entries:
            status = "stale"
    elif meta and status == "invalid":
        status = "stale"

    return {
        "cache_dir": cache_dir,
        "manifest_path": manifest_path,
        "meta_path": meta_path,
        "high_images": high_images,
        "expected_entries": expected_entries,
        "manifest_entries": manifest_entries,
        "meta": meta,
        "status": status,
    }


def render_prepared_cache_summary(summary):
    status_map = {
        "ready": ("success", t("status_ready"), t("prepare_ready")),
        "missing": ("warning", t("status_missing"), t("prepare_needed")),
        "stale": ("warning", t("status_stale"), t("prepare_needed")),
        "preparing": ("info", t("status_preparing"), t("prepare_needed")),
        "interrupted": ("warning", t("status_interrupted"), t("prepare_resume")),
        "invalid": ("warning", t("status_invalid"), t("prepare_needed")),
    }
    level, label, message = status_map.get(summary["status"], ("warning", summary["status"].upper(), t("prepare_needed")))
    if level == "success":
        st.success(f"{t('prepared_status')}: {label}")
    elif level == "info":
        st.info(f"{t('prepared_status')}: {label}")
    else:
        st.warning(f"{t('prepared_status')}: {label}")
    st.caption(message)

    top_cols = st.columns(4)
    top_cols[0].metric(t("high_images"), summary["high_images"])
    top_cols[1].metric(t("expected_entries"), summary["expected_entries"])
    top_cols[2].metric(t("manifest_entries"), summary["manifest_entries"])
    cached_variants = summary["meta"].get("variant_count", "-") if summary["meta"] else "-"
    top_cols[3].metric(t("meta_variant_count"), cached_variants)
    if summary["meta"]:
        st.caption(
            f"{t('completed_entries')}: {summary['meta'].get('completed_entries', '-')}"
        )
    st.caption(f"{t('cache_dir_label')}: `{summary['cache_dir']}`")


def load_preset_summary(preset_name):
    config_path = Path("configs/train") / f"{preset_name}.yaml"
    summary = {
        "name": preset_name,
        "config_path": str(config_path),
        "description": "",
        "target_vram_gb": "-",
        "resolution": "-",
        "batch_size": "-",
        "gradient_accumulation_steps": 1,
        "effective_batch": "-",
    }
    if not config_path.exists():
        return summary

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return summary

    meta = data.get("meta", {})
    runtime = data.get("runtime", {})
    optimization = data.get("optimization", {})
    batch_size = int(optimization.get("batch_size", 1))
    grad_acc = int(optimization.get("gradient_accumulation_steps", 1))
    summary.update({
        "name": meta.get("name", preset_name),
        "description": meta.get("description", ""),
        "target_vram_gb": meta.get("target_vram_gb", "-"),
        "resolution": runtime.get("resolution", "-"),
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_acc,
        "effective_batch": batch_size * grad_acc,
    })
    return summary


def render_intro_card(title, subtitle, badges=None, kicker=None):
    badges = badges or []
    badges_html = "".join(
        f"<span class='hero-badge'>{escape(str(badge))}</span>" for badge in badges if badge
    )
    kicker = kicker or "Diff-Img2Img"
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-kicker">{escape(kicker)}</div>
            <div class="hero-title">{escape(title)}</div>
            <p class="hero-desc">{escape(subtitle)}</p>
            {f"<div class='hero-badges'>{badges_html}</div>" if badges_html else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_train_command(
    model_scale,
    data_dir,
    output_dir,
    resolution,
    batch_size,
    epochs,
    lr,
    train_profile,
    variant_count,
    prepare_workers,
    synthesis_seed,
    use_retinex,
    resume,
    prepared_cache_dir,
    prepare_force,
    darker_ranges_text,
):
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch", "main.py", "--mode", "train",
        "--config", f"configs/train/{model_scale}.yaml",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--resolution", str(resolution),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--train_profile", train_profile,
        "--offline_variant_count", str(variant_count),
        "--prepare_workers", str(prepare_workers),
        "--synthesis_seed", str(synthesis_seed),
    ]

    if use_retinex:
        cmd.append("--use_retinex")
    if resume:
        cmd.extend(["--resume", resume])
    if prepared_cache_dir:
        cmd.extend(["--prepared_cache_dir", prepared_cache_dir])
    if prepare_force:
        cmd.append("--prepare_force")
    if darker_ranges_text.strip():
        cmd.extend(["--darker_ranges", darker_ranges_text])
    return cmd


def build_prepare_command(config_path, data_dir, prepared_cache_dir, variant_count, prepare_workers, synthesis_seed, prepare_force, darker_ranges_text):
    cmd = [
        sys.executable, "main.py", "--mode", "prepare",
        "--config", config_path,
        "--data_dir", data_dir,
        "--offline_variant_count", str(variant_count),
        "--prepare_workers", str(prepare_workers),
        "--synthesis_seed", str(synthesis_seed),
    ]
    if prepared_cache_dir:
        cmd.extend(["--prepared_cache_dir", prepared_cache_dir])
    if prepare_force:
        cmd.append("--prepare_force")
    if darker_ranges_text.strip():
        cmd.extend(["--darker_ranges", darker_ranges_text])
    return cmd

# --- Pages ---

def home_page():
    render_intro_card(
        t("app_title"),
        t("home_desc"),
        badges=[t("dataset"), t("training"), t("evaluation"), t("visualization")],
        kicker=t("home_kicker"),
    )

    st.markdown(f"### {t('workflow_header')}")
    flow_cols = st.columns(3)
    cards = [
        (t("workflow_prepare_title"), t("workflow_prepare_desc")),
        (t("workflow_train_title"), t("workflow_train_desc")),
        (t("workflow_eval_title"), t("workflow_eval_desc")),
    ]
    for col, (title, desc) in zip(flow_cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>{escape(title)}</h4>
                    <p>{escape(desc)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    quick_col, preset_col = st.columns([1.15, 0.85])
    with quick_col:
        st.markdown(f"### {t('quick_start')}")
        st.code(
            "MODEL_SIZE=middle \\\n"
            "DATA_DIR=/path/to/dataset \\\n"
            "OUTPUT_DIR=runs/middle_exp \\\n"
            "TRAIN_PROFILE=auto \\\n"
            "bash start_train.sh",
            language="bash",
        )
        st.caption(t("monitoring_note"))
        with st.expander(t("setup_header")):
            st.markdown(t("setup_desc"))
            st.markdown(f"**{t('install_dependencies')}:**")
            st.code("python3 -m pip install -r requirements.txt", language="bash")
            st.markdown(f"**{t('ui_command')}:**")
            st.code("python3 main.py --mode ui", language="bash")

    with preset_col:
        st.markdown(f"### {t('preset_overview')}")
        for preset_name in ["small", "middle", "max"]:
            preset = load_preset_summary(preset_name)
            st.markdown(
                f"""
                <div class="preset-card">
                    <strong>{escape(preset_name)}</strong>
                    <p>{escape(str(preset['description']))}</p>
                    <p>{escape(t('target_vram'))}: {escape(str(preset['target_vram_gb']))} GB ·
                    {escape(t('preset_resolution'))}: {escape(str(preset['resolution']))} ·
                    {escape(t('effective_batch'))}: {escape(str(preset['effective_batch']))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(f"### {t('example_results')}")
    c1, c2 = st.columns(2)
    fake_path = "examples/fake.png"
    real_path = "examples/real.png"

    with c1:
        if os.path.exists(fake_path):
            st.image(fake_path, caption=t("synthesized_input"), use_container_width=True)
        else:
            st.info("Example image (fake.png) not found.")

    with c2:
        if os.path.exists(real_path):
            st.image(real_path, caption=t("ground_truth"), use_container_width=True)
        else:
            st.info("Example image (real.png) not found.")

    st.info(t("home_tip"))

def dataset_page():
    render_intro_card(
        t("dataset_header"),
        t("dataset_sub"),
        badges=["our485/high", ".prepared", "train_manifest.jsonl"],
    )
    
    with st.expander(t("how_it_works"), expanded=False):
        st.markdown(t("darker_desc"))
        
    # --- Dataset Recommendations ---
    st.info(f"""
    **{t("dataset_recommendation")}**
    {t("dataset_link_text")}
    *   **LOL Dataset**: [Download Link](https://daooshee.github.io/BMVC2018website/)
    *   **Quark Drive (China)**: [https://pan.quark.cn/s/1867c35697db](https://pan.quark.cn/s/1867c35697db) (Code: **ZUWn**)
    """)

    with st.form("prepare_cache_form"):
        c1, c2 = st.columns(2)

        with c1:
            st.subheader(t("io_settings"))
            data_dir = folder_selector(t("dataset_root"), "prepare_data_dir", "../datasets/kitti_LOL")
            prepared_cache_dir = st.text_input(t("prepared_cache_dir"), value="", placeholder="../datasets/kitti_LOL/.prepared")
            model_scale_options = ["small", "middle", "max"]
            model_scale = st.selectbox(t("model_scale"), model_scale_options, index=1)

        with c2:
            st.subheader(t("prepared_settings"))
            variant_count = st.number_input(t("variant_count"), min_value=1, max_value=8, value=3)
            prepare_workers = st.number_input(t("prepare_workers"), min_value=1, max_value=64, value=recommended_prepare_workers())
            synthesis_seed = st.number_input(t("synthesis_seed"), min_value=0, value=42)
            prepare_force = st.checkbox(t("prepare_force"), value=False)

        darker_ranges_text = st.text_area(
            t("darker_ranges"),
            value="",
            height=180,
            placeholder="gamma: [1.5, 4.0]\nlinear_attenuation: [0.25, 0.7]\nnoise_sigma_read: [2.0, 15.0]",
            help=t("darker_ranges_help"),
        )
        st.info(t("manual_prepare_tip"))
        prepare_btn = st.form_submit_button(t("prepare_dataset"), type="primary")

    cache_summary = summarize_prepared_cache(data_dir, prepared_cache_dir, variant_count, synthesis_seed)
    render_prepared_cache_summary(cache_summary)
    preview_cmd = build_prepare_command(
        config_path=f"configs/train/{model_scale}.yaml",
        data_dir=data_dir,
        prepared_cache_dir=prepared_cache_dir,
        variant_count=variant_count,
        prepare_workers=prepare_workers,
        synthesis_seed=synthesis_seed,
        prepare_force=prepare_force,
        darker_ranges_text=darker_ranges_text,
    )
    with st.expander(t("command_preview"), expanded=False):
        st.code(shlex.join(preview_cmd), language="bash")

    if prepare_btn:
        if not os.path.exists(data_dir):
            st.error(f"❌ Directory not found: {data_dir}")
            return

        config_path = f"configs/train/{model_scale}.yaml"
        cmd = build_prepare_command(
            config_path=config_path,
            data_dir=data_dir,
            prepared_cache_dir=prepared_cache_dir,
            variant_count=variant_count,
            prepare_workers=prepare_workers,
            synthesis_seed=synthesis_seed,
            prepare_force=prepare_force,
            darker_ranges_text=darker_ranges_text,
        )
        with st.spinner(t("prepare_running")):
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            st.success(t("prepare_complete"))
            render_prepared_cache_summary(summarize_prepared_cache(data_dir, prepared_cache_dir, variant_count, synthesis_seed))
        else:
            st.error(t("prepare_failed"))
        with st.expander(t("prepare_stdout"), expanded=result.returncode != 0):
            st.code((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""), language="bash")

def training_page():
    render_intro_card(
        t("train_dashboard"),
        t("train_desc"),
        badges=["small", "middle", "max"],
    )
    
    # --- Sidebar Status ---
    with st.sidebar:
        st.divider()
        if st.session_state.training_pid:
            st.success(f"{t('running')} (PID: {st.session_state.training_pid})")
            if st.button(t("stop"), type="primary"):
                try:
                    training_pgid = st.session_state.get("training_pgid")
                    if training_pgid:
                        os.killpg(training_pgid, signal.SIGTERM)
                    else:
                        os.kill(st.session_state.training_pid, signal.SIGTERM)
                    clear_training_state()
                    st.rerun()
                except:
                    clear_training_state()
        else:
            st.info(t("idle"))

    # --- Configuration Tabs ---
    tab_config, tab_monitor = st.tabs([t("config"), t("monitoring")])
    
    with tab_config:
        with st.form("train_conf"):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(t("paths"))
                data_dir = folder_selector(t("dataset_root"), "train_data", "../datasets/kitti_LOL")
                output_dir = st.text_input(t("exp_name"), value="run_diffusion_experiment", help="Folder where logs and checkpoints will be saved.")
                resume = st.text_input(t("resume"), placeholder="e.g. 'latest' or checkpoint-1000", help="Leave empty to start new training.")
                prepared_cache_dir = st.text_input(t("prepared_cache_dir"), value="", placeholder="../datasets/kitti_LOL/.prepared")
                
                st.subheader(t("model"))
                use_retinex = st.checkbox(t("use_retinex"), value=True, help="Highly recommended for low-light tasks.")
                model_scale_options = ["small", "middle", "max"]
                model_scale = st.selectbox(t("model_scale"), model_scale_options, index=1)
                preset_summary = load_preset_summary(model_scale)
                default_resolution = int(preset_summary["resolution"]) if str(preset_summary["resolution"]).isdigit() else 256
                res = st.selectbox(t("resolution"), [256, 320, 512], index=[256, 320, 512].index(default_resolution))
            
            with c2:
                st.subheader(t("hyperparams"))
                epochs = st.number_input(t("epochs"), value=50, min_value=1)
                batch_size = st.number_input(t("batch_size"), value=4, min_value=1)
                lr = st.number_input(t("lr"), value=1e-4, format="%.1e", step=1e-5)
                train_profile_options = ["auto", "debug_online"]
                train_profile = st.selectbox(t("train_profile"), train_profile_options, index=0)
                st.subheader(t("prepared_settings"))
                variant_count = st.number_input(t("variant_count"), min_value=1, max_value=8, value=3)
                prepare_workers = st.number_input(t("prepare_workers"), min_value=1, max_value=64, value=recommended_prepare_workers())
                synthesis_seed = st.number_input(t("synthesis_seed"), min_value=0, value=42)
                prepare_force = st.checkbox(t("prepare_force"), value=False)
                darker_ranges_text = st.text_area(
                    t("darker_ranges"),
                    value="",
                    height=180,
                    placeholder="gamma: [1.5, 4.0]\nlinear_attenuation: [0.25, 0.7]",
                    help=t("darker_ranges_help"),
                )
                
            st.info(t("train_tip"))
            button_cols = st.columns(2)
            prepare_btn = button_cols[0].form_submit_button(t("prepare_dataset"))
            train_btn = button_cols[1].form_submit_button(t("launch_train"), type="primary")

        preset_summary = load_preset_summary(model_scale)
        cache_summary = summarize_prepared_cache(data_dir, prepared_cache_dir, variant_count, synthesis_seed)
        effective_batch = int(batch_size) * int(preset_summary.get("gradient_accumulation_steps", 1))
        summary_cols = st.columns(4)
        summary_cols[0].metric(t("target_vram"), f"{preset_summary['target_vram_gb']} GB")
        summary_cols[1].metric(t("preset_resolution"), str(preset_summary["resolution"]))
        summary_cols[2].metric(t("effective_batch"), effective_batch)
        summary_cols[3].metric(t("prepared_status"), t(f"status_{cache_summary['status']}"))
        st.caption(f"{t('preset_description')}: {preset_summary['description']}")

        render_prepared_cache_summary(cache_summary)
        if effective_batch < 16:
            st.warning(f"{t('recommended_effective_batch')}: 16")
        else:
            st.caption(f"{t('recommended_effective_batch')}: 16")

        train_cmd_preview = build_train_command(
            model_scale=model_scale,
            data_dir=data_dir,
            output_dir=output_dir,
            resolution=res,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            train_profile=train_profile,
            variant_count=variant_count,
            prepare_workers=prepare_workers,
            synthesis_seed=synthesis_seed,
            use_retinex=use_retinex,
            resume=resume,
            prepared_cache_dir=prepared_cache_dir,
            prepare_force=prepare_force,
            darker_ranges_text=darker_ranges_text,
        )
        with st.expander(t("command_preview"), expanded=False):
            st.code(shlex.join(train_cmd_preview), language="bash")

        if prepare_btn:
            if not os.path.exists(data_dir):
                st.error(f"❌ Directory not found: {data_dir}")
            else:
                prepare_cmd = build_prepare_command(
                    config_path=f"configs/train/{model_scale}.yaml",
                    data_dir=data_dir,
                    prepared_cache_dir=prepared_cache_dir,
                    variant_count=variant_count,
                    prepare_workers=prepare_workers,
                    synthesis_seed=synthesis_seed,
                    prepare_force=prepare_force,
                    darker_ranges_text=darker_ranges_text,
                )
                with st.spinner(t("prepare_running")):
                    result = subprocess.run(prepare_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    st.success(t("prepare_complete"))
                    render_prepared_cache_summary(summarize_prepared_cache(data_dir, prepared_cache_dir, variant_count, synthesis_seed))
                else:
                    st.error(t("prepare_failed"))
                with st.expander(t("prepare_stdout"), expanded=result.returncode != 0):
                    st.code((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""), language="bash")
    
    # --- Launch Logic ---
    if train_btn and not st.session_state.training_pid:
        cmd = build_train_command(
            model_scale=model_scale,
            data_dir=data_dir,
            output_dir=output_dir,
            resolution=res,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            train_profile=train_profile,
            variant_count=variant_count,
            prepare_workers=prepare_workers,
            synthesis_seed=synthesis_seed,
            use_retinex=use_retinex,
            resume=resume,
            prepared_cache_dir=prepared_cache_dir,
            prepare_force=prepare_force,
            darker_ranges_text=darker_ranges_text,
        )
        
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "train_ui_log.txt")
        csv_file = os.path.join(output_dir, "training_metrics.csv")
        status_file = os.path.join(output_dir, "training_status.json")
        
        st.session_state.training_log_file = log_file
        st.session_state.training_csv_file = csv_file
        st.session_state.training_status_file = status_file
        
        try:
            with open(log_file, "w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                )
            st.session_state.training_pid = process.pid
            if hasattr(os, "getpgid"):
                try:
                    st.session_state.training_pgid = os.getpgid(process.pid)
                except OSError:
                    st.session_state.training_pgid = None
            st.success(t("train_launched"))
            time.sleep(1) # Wait for startup
            st.rerun()
        except Exception as e:
            st.error(f"Launch failed: {e}")

    # --- Monitoring Tab ---
    with tab_monitor:
        if not st.session_state.training_pid and not st.session_state.training_csv_file:
            st.info(t("start_training_hint"))
        else:
            # Auto-refresh toggle
            col_controls = st.container()
            with col_controls:
                auto_refresh = st.toggle(t("auto_refresh"), value=False, help=t("auto_refresh_help"))
                st.caption(t("monitoring_note"))
                
            # Layout: Left (Logs), Right (Charts)
            c_logs, c_charts = st.columns([1, 1])
            
            with c_logs:
                st.subheader(t("terminal_output"))
                # Use a container with fixed height for logs to behave like a terminal window
                log_container = st.container(height=600) 
                logs = read_log_file(st.session_state.training_log_file, num_lines=200)
                log_container.code(logs, language="bash", line_numbers=True)

            with c_charts:
                st.subheader(t("results")) # Metrics
                csv_path = st.session_state.training_csv_file
                status = read_status_file(st.session_state.training_status_file)
                if status:
                    metrics_cols = st.columns(4)
                    metrics_cols[0].metric("Phase", str(status.get("phase", "-")))
                    metrics_cols[1].metric("Loss", f"{status.get('loss', 0.0):.4f}" if isinstance(status.get("loss"), (int, float)) else "-")
                    metrics_cols[2].metric("Samples/s", f"{status.get('samples_per_sec', 0.0):.2f}" if isinstance(status.get("samples_per_sec"), (int, float)) else "-")
                    metrics_cols[3].metric("GPU Reserved", f"{status.get('gpu_reserved_gb', 0.0):.2f} GB" if isinstance(status.get("gpu_reserved_gb"), (int, float)) else "-")

                    resource_cols = st.columns(4)
                    resource_cols[0].metric("Data Time", f"{status.get('data_time', 0.0):.3f} s" if isinstance(status.get("data_time"), (int, float)) else "-")
                    resource_cols[1].metric("Compute Time", f"{status.get('compute_time', 0.0):.3f} s" if isinstance(status.get("compute_time"), (int, float)) else "-")
                    resource_cols[2].metric("CPU %", f"{status.get('cpu_percent', 0.0):.1f}" if isinstance(status.get("cpu_percent"), (int, float)) else "-")
                    resource_cols[3].metric("CPU RSS", f"{status.get('cpu_rss_gb', 0.0):.2f} GB" if isinstance(status.get("cpu_rss_gb"), (int, float)) else "-")

                    val_cols = st.columns(3)
                    val_cols[0].metric("Val PSNR", f"{status.get('val_psnr', 0.0):.3f}" if isinstance(status.get("val_psnr"), (int, float)) else "-")
                    val_cols[1].metric("Val SSIM", f"{status.get('val_ssim', 0.0):.4f}" if isinstance(status.get("val_ssim"), (int, float)) else "-")
                    val_cols[2].metric("Val LPIPS", f"{status.get('val_lpips', 0.0):.4f}" if isinstance(status.get("val_lpips"), (int, float)) else "-")

                if csv_path and os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            fig_loss = px.line(df, x='step', y='loss', title='Training Loss', template="plotly_white")
                            fig_loss.update_traces(line_color='#FF4B4B')
                            st.plotly_chart(fig_loss, use_container_width=True)

                            fig_lr = px.line(df, x='step', y='lr', title='Learning Rate', template="plotly_white")
                            fig_lr.update_traces(line_color='#0068C9')
                            st.plotly_chart(fig_lr, use_container_width=True)

                            if "samples_per_sec" in df.columns:
                                fig_throughput = px.line(df, x='step', y='samples_per_sec', title='Throughput (samples/s)', template="plotly_white")
                                fig_throughput.update_traces(line_color='#059669')
                                st.plotly_chart(fig_throughput, use_container_width=True)

                            if {"data_time", "compute_time"}.issubset(df.columns):
                                time_df = df[["step", "data_time", "compute_time"]].melt(id_vars="step", var_name="component", value_name="seconds")
                                fig_timing = px.line(time_df, x='step', y='seconds', color='component', title='Data vs Compute Time', template="plotly_white")
                                st.plotly_chart(fig_timing, use_container_width=True)
                        else:
                            st.warning(t("waiting_for_data_points"))
                    except Exception as e:
                        st.caption(f"Reading metrics... ({e})")
                else:
                    st.info(t("initializing_metrics_log"))

            # Auto-refresh logic
            if auto_refresh and st.session_state.training_pid:
                time.sleep(5)
                st.rerun()
            
            # Check process status
            if st.session_state.training_pid:
                try:
                    training_pgid = st.session_state.get("training_pgid")
                    if training_pgid:
                        os.killpg(training_pgid, 0)
                    else:
                        os.kill(st.session_state.training_pid, 0)
                except OSError:
                    st.warning(t("process_finished"))
                    clear_training_state()
                    st.rerun()

def evaluation_page():
    render_intro_card(t("eval_header"), t("eval_desc"), badges=["PSNR", "SSIM", "LPIPS"])
    
    with st.form("eval_form"):
        c1, c2 = st.columns(2)
        with c1:
            model_path = folder_selector(t("model_ckpt"), "eval_model", "run_diffusion_experiment")
            data_dir = folder_selector(t("dataset_root"), "eval_data", "../datasets/kitti_LOL")
        with c2:
            out_dir = st.text_input(t("output_folder"), value="eval_results")
            use_retinex = st.checkbox(t("use_retinex"), value=True)
        
        run_eval = st.form_submit_button(t("run_eval"))
        
    if run_eval:
        cmd = [
            sys.executable, "main.py", "--mode", "validate",
            "--model_path", model_path,
            "--data_dir", data_dir,
            "--output_dir", out_dir
        ]
        if use_retinex: cmd.append("--use_retinex")
        
        with st.spinner(t("calc_metrics")):
            result = subprocess.run(cmd, capture_output=True, text=True)
            
        if result.returncode == 0:
            st.balloons()
            st.success(t("eval_complete"))
            
            metric_file = os.path.join(out_dir, "metrics.txt")
            if os.path.exists(metric_file):
                with open(metric_file, 'r') as f:
                    metrics = f.readlines()
                
                # Display as cards
                st.subheader(t("results"))
                m_cols = st.columns(3)
                for line in metrics:
                    if ":" in line:
                        key, val = line.split(":")
                        try:
                            numeric_val = float(val.strip())
                        except ValueError:
                            continue
                        if "PSNR" in key: m_cols[0].metric("PSNR", f"{numeric_val:.2f} dB")
                        elif "SSIM" in key: m_cols[1].metric("SSIM", f"{numeric_val:.4f}")
                        elif "LPIPS" in key: m_cols[2].metric("LPIPS", f"{numeric_val:.4f}")
        else:
            st.error(t("evaluation_failed"))
            st.code(result.stderr)

def visualization_page():
    render_intro_card(t("vis_header"), t("vis_desc"))
    
    # Import visual logic dynamically
    try:
        from scripts.visual_val import load_models, run_inference, tensor_to_pil, plot_histogram
    except ImportError:
        st.error(t("dependencies_missing"))
        return

    with st.sidebar:
        st.subheader(t("vis_config"))
        model_path = folder_selector(t("model_ckpt"), "vis_model", "run_diffusion_experiment")
        use_retinex = st.checkbox(t("use_retinex"), value=True)
        if st.button(t("reload_models")):
            st.cache_resource.clear()

    # Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        models = load_models(model_path, use_retinex, device)
    except Exception as e:
        st.error(f"{t('load_failed')}: {e}")
        return

    # Select Data
    data_path = folder_selector(t("dataset_root"), "vis_data", "../datasets/kitti_LOL")
    test_low = os.path.join(data_path, "eval15", "low")
    
    if os.path.exists(test_low):
        files = sorted([f for f in os.listdir(test_low) if f.endswith(('.png', '.jpg'))])
        sel_file = st.selectbox(t("select_image"), files)
        
        if sel_file:
            from PIL import Image
            from torchvision import transforms
            
            # Paths
            low_p = os.path.join(test_low, sel_file)
            high_p = os.path.join(data_path, "eval15", "high", sel_file)
            
            # Load & Process
            img_low = Image.open(low_p).convert("RGB")
            img_high = Image.open(high_p).convert("RGB")
            
            tf = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
            
            t_low = tf(img_low)
            
            with st.spinner(t("running_diff")):
                t_out = run_inference(models, t_low, 20, device)
            
            img_out = tensor_to_pil(t_out)
            
            # Layout
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(img_low, caption=t("synthesized_input"), use_container_width=True)
            with c2:
                st.image(img_out, caption=t("enhanced"), use_container_width=True)
            with c3:
                st.image(img_high, caption=t("ground_truth"), use_container_width=True)
            
            st.info(f"{t('displaying')}: {sel_file}")
    else:
        st.warning(t("dataset_path_invalid"))

def configuration_page():
    render_intro_card(t("accelerate_config_header"), t("accelerate_config_desc"))

    # Check for existing config
    config_path = "accelerate_config.yaml"
    default_config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "NO",
        "mixed_precision": "fp16",
        "use_cpu": False,
        "dynamo_backend": "no",
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    default_config.update(loaded_config)
        except:
            pass

    with st.form("accelerate_config_form"):
        c1, c2 = st.columns(2)
        with c1:
            compute_env = st.selectbox("Compute Environment", ["LOCAL_MACHINE", "AWS_SAGEMAKER_1_GPU", "AWS_SAGEMAKER_MULTI_GPU"], index=0 if default_config["compute_environment"] == "LOCAL_MACHINE" else 0)
            dist_type = st.selectbox("Distributed Type", ["NO", "MULTI_GPU", "MULTI_CPU"], index=0 if default_config["distributed_type"] == "NO" else (1 if default_config["distributed_type"] == "MULTI_GPU" else 2))
            mixed_precision = st.selectbox("Mixed Precision", ["no", "fp16", "bf16"], index=["no", "fp16", "bf16"].index(default_config.get("mixed_precision", "no")))
        
        with c2:
            dynamo = st.selectbox("Dynamo Backend (Torch 2.0+)", ["no", "eager", "inductor"], index=["no", "eager", "inductor"].index(default_config.get("dynamo_backend", "no")))
            num_processes = st.number_input("Num Processes (GPUs)", value=1, min_value=1, help="For distributed training.")
            use_cpu = st.checkbox("Force CPU", value=default_config.get("use_cpu", False))

        save = st.form_submit_button(t("save_config"))

    if save:
        new_config = {
            "compute_environment": compute_env,
            "distributed_type": dist_type,
            "mixed_precision": mixed_precision,
            "use_cpu": use_cpu,
            "dynamo_backend": dynamo,
            "num_processes": num_processes,
            "machine_rank": 0,
            "num_machines": 1,
            "rdzv_backend": "static",
            "same_network": True,
            "main_training_function": "main",
        }
        
        if dist_type == "MULTI_GPU":
            new_config["distributed_type"] = "MULTI_GPU"
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)
            st.success(t("config_saved"))
        except Exception as e:
            st.error(f"Error saving config: {e}")
            
    st.info(t("config_tip"))

# --- Main Nav ---
with st.sidebar:
    st.title(t("nav_title"))
    # Define mapping between localized names and internal keys
    nav_options = {
        t("home"): "Home",
        t("dataset"): "Dataset Preparation",
        t("training"): "Training",
        t("evaluation"): "Evaluation",
        t("visualization"): "Visualization",
        t("configuration"): "Configuration",
    }
    selected_nav = st.radio("", list(nav_options.keys()), label_visibility="collapsed")
    page = nav_options[selected_nav]

if page == "Home": home_page()
elif page == "Dataset Preparation": dataset_page()
elif page == "Training": training_page()
elif page == "Evaluation": evaluation_page()
elif page == "Visualization": visualization_page()
elif page == "Configuration": configuration_page()
