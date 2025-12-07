import streamlit as st
import os
import subprocess
import signal
import time
from pathlib import Path
import shutil
import sys
import threading
import queue
import pandas as pd
import plotly.express as px
import yaml
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
        "home_subtitle": "🚀 Next-Gen Low-Light Enhancement",
        "home_desc": "**Diff-Img2Img** is a comprehensive framework leveraging **Diffusion Models** and **Retinex Theory** to restore extreme low-light images with realistic noise handling.",
        "key_features": "🌟 Key Features",
        "feature_1": "**Physics-based Data Synthesis**: Uses `Darker` engine to simulate realistic low-light degradation (Gamma, Noise, Headlights).",
        "feature_2": "**Retinex-Diffusion Architecture**: Decomposes images into Reflectance/Illumination for stable diffusion training.",
        "feature_3": "**Full-Stack Workflow**: From data generation to visual evaluation in one integrated dashboard.",
        "example_results": "🖼️ Example Results",
        "synthesized_input": "Synthesized Low-Light (Input)",
        "ground_truth": "Ground Truth (Reference)",
        "home_tip": "💡 **Tip:** Navigate using the sidebar to start your workflow.",
        "dataset_header": "🌑 Dataset Preparation",
        "dataset_sub": "Generate synthetic low-light training data from normal datasets.",
        "how_it_works": "ℹ️ How it works",
        "darker_desc": """The **Darker** engine applies a physics-based degradation model:
        1.  **Gamma Correction**: Non-linear darkening.
        2.  **Linear Attenuation**: Simulating low exposure.
        3.  **Headlight Simulation**: Adding localized light sources using masks.
        4.  **Sensor Noise**: Gaussian/Poisson noise injection.""",
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
        "resolution": "Image Resolution",
        "hyperparams": "⚡ Hyperparameters",
        "epochs": "Epochs",
        "batch_size": "Batch Size",
        "lr": "Learning Rate",
        "val_freq": "Validation Frequency (Epochs)",
        "train_tip": "💡 **Tip:** Ensure your GPU has enough VRAM for Batch Size > 1 at 512px.",
        "launch_train": "🚀 Launch Training",
        "train_launched": "Training launched! Switch to 'Monitoring' tab.",
        "refresh_charts": "🔄 Refresh Charts",
        "retinex_loss_weight": "Retinex Loss Weight",
        "tv_loss_weight": "TV Loss Weight (Retinex)",
        "grad_clip_norm": "Gradient Clip Norm",
        "offset_noise_scale": "Offset Noise Scale",
        "snr_gamma": "Min-SNR Gamma",
        "adv_hyperparams": "🛠️ Advanced Hyperparameters",
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
        "home_subtitle": "🚀 下一代低光照增强",
        "home_desc": "**Diff-Img2Img** 是一个利用 **扩散模型** 和 **Retinex 理论** 来恢复极低光照图像并处理真实噪声的综合框架。",
        "key_features": "🌟 主要特性",
        "feature_1": "**基于物理的数据合成**：使用 `Darker` 引擎模拟真实的低光照退化（Gamma，噪声，车灯）。",
        "feature_2": "**Retinex-扩散架构**：将图像分解为反射率/光照，以进行稳定的扩散训练。",
        "feature_3": "**全栈工作流**：从数据生成到可视化评估的一个集成仪表板。",
        "example_results": "🖼️ 示例结果",
        "synthesized_input": "合成低光照 (输入)",
        "ground_truth": "地面实况 (参考)",
        "home_tip": "💡 **提示：** 使用侧边栏导航开始您的工作流。",
        "dataset_header": "🌑 数据集准备",
        "dataset_sub": "从正常数据集生成合成低光照训练数据。",
        "how_it_works": "ℹ️ 工作原理",
        "darker_desc": """**Darker** 引擎应用基于物理的退化模型：
        1.  **Gamma 校正**：非线性变暗。
        2.  **线性衰减**：模拟低曝光。
        3.  **车灯模拟**：使用掩码添加局部光源。
        4.  **传感器噪声**：高斯/泊松噪声注入。""",
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
        "resolution": "图像分辨率",
        "hyperparams": "⚡ 超参数",
        "epochs": "轮数 (Epochs)",
        "batch_size": "批次大小",
        "lr": "学习率",
        "val_freq": "验证频率 (Epochs)",
        "train_tip": "💡 **提示：** 确保您的 GPU 显存足以在 512px 下支持批次大小 > 1。",
        "launch_train": "🚀 启动训练",
        "train_launched": "训练已启动！切换到 '监控' 标签。",
        "refresh_charts": "🔄 刷新图表",
        "retinex_loss_weight": "Retinex 损失权重",
        "tv_loss_weight": "TV 损失权重 (Retinex)",
        "grad_clip_norm": "梯度裁剪范数",
        "offset_noise_scale": "偏移噪声缩放 (Offset Noise Scale)",
        "snr_gamma": "Min-SNR Gamma",
        "adv_hyperparams": "🛠️ 高级超参数",
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
    /* Global Fonts */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headings */
    h1, h2, h3 {
        font-weight: 700;
        color: #1f2937;
    }
    
    /* Buttons */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease-in-out;
        padding: 0.6rem 1rem;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Inputs & Selects */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2);
    }
    
    /* Expanders & Cards */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        background-color: #ffffff;
        margin-bottom: 1rem;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f3f4f6;
        text-align: center;
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
        border-color: #d1d5db;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Custom Classes */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'training_pid' not in st.session_state:
    st.session_state.training_pid = None
if 'training_log_file' not in st.session_state:
    st.session_state.training_log_file = None
if 'training_csv_file' not in st.session_state:
    st.session_state.training_csv_file = None
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
    if not file_path or not os.path.exists(file_path): return t("Waiting for logs...")
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            return "".join(lines[-num_lines:])
    except Exception as e: return f"Error reading log: {e}"

# --- Pages ---

def home_page():
    st.title(t("app_title"))
    
    st.markdown(f"""
    ### {t("home_subtitle")}
    {t("home_desc")}
    
    #### {t("key_features")}
    *   {t("feature_1")}
    *   {t("feature_2")}
    *   {t("feature_3")}
    """)
    
    st.divider()

    # --- Setup & Installation Guide ---
    with st.expander(t("setup_header")):
        st.markdown(t("setup_desc"))
        st.markdown(f"**{t('install_dependencies')}:**")
        st.code("pip install -r requirements.txt")
        st.markdown("---") # Separator

    st.subheader(t("example_results"))
    c1, c2 = st.columns(2)
    
    # Load example images safely
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
    st.header(t("dataset_header"))
    st.markdown(t("dataset_sub"))
    
    with st.expander(t("how_it_works"), expanded=False):
        st.markdown(t("darker_desc"))
        
    # --- Dataset Recommendations ---
    st.info(f"""
    **{t("dataset_recommendation")}**
    {t("dataset_link_text")}
    *   **LOL Dataset**: [Download Link](https://daooshee.github.io/BMVC2018website/)
    *   **Quark Drive (China)**: [https://pan.quark.cn/s/1867c35697db](https://pan.quark.cn/s/1867c35697db) (Code: **ZUWn**)
    """)

    with st.form("darker_config"):
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader(t("io_settings"))
            data_dir = folder_selector(t("raw_dataset"), "data_dir", "../datasets/kitti_raw")
            phase = st.selectbox(t("split"), ["train", "test", "eval"], help="Which dataset split to process")
            
            st.subheader(t("physics_params"))
            gamma = st.slider(t("gamma"), 1.0, 10.0, 6.5, help="Controls global darkening curve.")
            linear_attenuation = st.slider(t("linear_att"), 0.0, 1.0, 0.15, help="Simulates exposure time reduction.")
            
        with c2:
            st.subheader(t("effects"))
            headlight_boost = st.slider(t("headlight"), 0.0, 2.0, 0.9, help="Brightness of simulated light sources.")
            noise_sigma = st.slider(t("noise"), 0.0, 50.0, 8.0, help="Amount of sensor noise added.")
            saturation = st.slider(t("saturation"), 0.0, 1.0, 0.5, help="Color washout in dark areas.")
            color_shift = st.slider(t("blue_shift"), 0.0, 0.5, 0.15, help="Purkinje effect simulation.")
            beam_width = st.slider(t("beam_width"), 0.1, 1.0, 0.7, help="Size of the light beam mask.")
            
        submit_darker = st.form_submit_button(t("start_synthesis"))
        
    if submit_darker:
        if not os.path.exists(data_dir):
            st.error(f"❌ Directory not found: {data_dir}")
            return

        st.status(t("processing"), expanded=True)
        try:
            # Dynamic import to ensure fresh load
            from scripts.darker import Darker
            mask_params = {"center_y_factor": 0.9, "beam_width_factor": beam_width, "falloff_sharpness": 2.5}
            dk = Darker(data_dir=data_dir, phase=phase, gamma=gamma, linear_attenuation=linear_attenuation)
            dk.process_images(
                mask_params=mask_params, 
                headlight_boost=headlight_boost,
                noise_sigma=noise_sigma,
                saturation_factor=saturation,
                color_shift_factor=color_shift
            )
            st.success(t("synthesis_complete"))
        except Exception as e:
            st.error(f"Error: {e}")

def training_page():
    st.header(t("train_dashboard"))
    
    # --- Sidebar Status ---
    with st.sidebar:
        st.divider()
        if st.session_state.training_pid:
            st.success(f"{t('running')} (PID: {st.session_state.training_pid})")
            if st.button(t("stop"), type="primary"):
                try:
                    os.kill(st.session_state.training_pid, signal.SIGTERM)
                    st.session_state.training_pid = None
                    st.rerun()
                except:
                    st.session_state.training_pid = None
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
                
                st.subheader(t("model"))
                use_retinex = st.checkbox(t("use_retinex"), value=True, help="Highly recommended for low-light tasks.")
                res = st.selectbox(t("resolution"), [256, 512], index=0)
            
            with c2:
                st.subheader(t("hyperparams"))
                epochs = st.number_input(t("epochs"), value=50, min_value=1)
                batch_size = st.number_input(t("batch_size"), value=4, min_value=1)
                lr = st.number_input(t("lr"), value=1e-4, format="%.1e", step=1e-5)
                val_freq = st.number_input(t("val_freq"), value=5, min_value=1)
            
            with st.expander(t("adv_hyperparams"), expanded=False):
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    retinex_loss_w = st.number_input(t("retinex_loss_weight"), value=0.1, step=0.01, format="%.2f")
                    tv_loss_w = st.number_input(t("tv_loss_weight"), value=0.1, step=0.01, format="%.2f")
                with ac2:
                    grad_clip = st.number_input(t("grad_clip_norm"), value=5.0, step=0.1, format="%.1f")
                    offset_noise = st.number_input(t("offset_noise_scale"), value=0.1, step=0.01, format="%.2f")
                with ac3:
                    snr_gamma = st.number_input(t("snr_gamma"), value=5.0, step=0.5, format="%.1f")
                
            st.info(t("train_tip"))
            train_btn = st.form_submit_button(t("launch_train"), type="primary")
    
    # --- Launch Logic ---
    if train_btn and not st.session_state.training_pid:
        cmd = [
            sys.executable, "main.py", "--mode", "train",
            "--data_dir", data_dir,
            "--output_dir", output_dir,
            "--resolution", str(res),
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--validation_steps", "500",
            "--retinex_loss_weight", str(retinex_loss_w),
            "--tv_loss_weight", str(tv_loss_w),
            "--grad_clip_norm", str(grad_clip),
            "--offset_noise_scale", str(offset_noise),
            "--snr_gamma", str(snr_gamma)
        ]

        # Enable offset noise flag if scale > 0
        if offset_noise > 0:
            cmd.append("--offset_noise")

        if use_retinex: cmd.append("--use_retinex")
        if resume: 
             # main.py expects --model_path for resume or handle resume internally?
             # engine.py loads from model_path.
             # If resume is 'latest', we need logic.
             # main.py has --model_path. 
             # I will map resume to model_path if it looks like a path.
             cmd.extend(["--model_path", resume])
        
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "train_ui_log.txt")
        csv_file = os.path.join(output_dir, "training_metrics.csv")
        
        st.session_state.training_log_file = log_file
        st.session_state.training_csv_file = csv_file
        
        try:
            # Start subprocess
            process = subprocess.Popen(cmd, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
            st.session_state.training_pid = process.pid
            st.success(t("train_launched"))
            time.sleep(1) # Wait for startup
            st.rerun()
        except Exception as e:
            st.error(f"Launch failed: {e}")

    # --- Monitoring Tab ---
    with tab_monitor:
        if not st.session_state.training_pid and not st.session_state.training_csv_file:
            st.info("Start training to see metrics.")
        else:
            # Auto-refresh toggle
            col_controls = st.container()
            with col_controls:
                auto_refresh = st.toggle("🔄 Auto Refresh (5s)", value=False, help="Automatically refresh metrics and logs every 5 seconds.")
                
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
                if csv_path and os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            # Loss Chart
                            fig_loss = px.line(df, x='step', y='loss', title='Training Loss', template="plotly_white")
                            fig_loss.update_traces(line_color='#FF4B4B')
                            st.plotly_chart(fig_loss, use_container_width=True)
                            
                            # LR Chart
                            fig_lr = px.line(df, x='step', y='lr', title='Learning Rate', template="plotly_white")
                            fig_lr.update_traces(line_color='#0068C9')
                            st.plotly_chart(fig_lr, use_container_width=True)
                        else:
                            st.warning(t("Waiting for data points..."))
                    except Exception as e:
                        st.caption(f"Reading metrics... ({e})")
                else:
                    st.info(t("Initializing metrics log..."))

            # Auto-refresh logic
            if auto_refresh and st.session_state.training_pid:
                time.sleep(5)
                st.rerun()
            
            # Check process status
            if st.session_state.training_pid:
                try:
                    os.kill(st.session_state.training_pid, 0)
                except OSError:
                    st.warning(t("⚠️ Process finished or stopped unexpectedly."))
                    st.session_state.training_pid = None
                    st.rerun()

def evaluation_page():
    st.header(t("eval_header"))
    st.markdown(t("eval_desc"))
    
    with st.form("eval_form"):
        c1, c2 = st.columns(2)
        with c1:
            model_path = folder_selector(t("model_ckpt"), "eval_model", "run_diffusion_experiment")
            data_dir = folder_selector(t("dataset"), "eval_data", "../datasets/kitti_LOL")
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
                        val = float(val.strip())
                        if "PSNR" in key: m_cols[0].metric("PSNR", f"{val:.2f} dB")
                        elif "SSIM" in key: m_cols[1].metric("SSIM", f"{val:.4f}")
                        elif "LPIPS" in key: m_cols[2].metric("LPIPS", f"{val:.4f}")
        else:
            st.error("Evaluation Failed")
            st.code(result.stderr)

def visualization_page():
    st.header(t("vis_header"))
    
    # Import visual logic dynamically
    try:
        from scripts.visual_val import load_models, run_inference, tensor_to_pil, plot_histogram
    except ImportError:
        st.error("Dependencies missing (scripts/visual_val.py).")
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
        st.error(f"Load failed: {e}")
        return

    # Select Data
    data_path = folder_selector(t("dataset"), "vis_data", "../datasets/kitti_LOL")
    test_low = os.path.join(data_path, "test", "low")
    
    if os.path.exists(test_low):
        files = sorted([f for f in os.listdir(test_low) if f.endswith(('.png', '.jpg'))])
        sel_file = st.selectbox(t("select_image"), files)
        
        if sel_file:
            from PIL import Image
            from torchvision import transforms
            
            # Paths
            low_p = os.path.join(test_low, sel_file)
            high_p = os.path.join(data_path, "test", "high", sel_file)
            
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
        st.warning(t("Dataset path invalid."))

def configuration_page():
    st.header(t("accelerate_config_header"))
    st.markdown(t("accelerate_config_desc"))

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
            
    st.info("💡 **Tip:** This config file (`accelerate_config.yaml`) will be used when launching training.")

# --- Main Nav ---
with st.sidebar:
    st.title(t("nav_title"))
    # Define mapping between localized names and internal keys
    nav_options = {
        t("home"): "Home",
        t("dataset"): "Dataset Preparation",
        t("configuration"): "Configuration",
        t("training"): "Training",
        t("evaluation"): "Evaluation",
        t("visualization"): "Visualization"
    }
    selected_nav = st.radio("", list(nav_options.keys()), label_visibility="collapsed")
    page = nav_options[selected_nav]

if page == "Home": home_page()
elif page == "Dataset Preparation": dataset_page()
elif page == "Configuration": configuration_page()
elif page == "Training": training_page()
elif page == "Evaluation": evaluation_page()
elif page == "Visualization": visualization_page()