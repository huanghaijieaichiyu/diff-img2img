# 📦 Packaging & Release Guide

This guide explains how to package **Diff-Img2Img Studio** as a standalone executable and release it to GitHub.

## 1. Prerequisites

Ensure you have the project dependencies installed:
```bash
pip install -r requirements.txt
pip install pyinstaller
```

## 2. Building the Executable

### Linux / macOS
Run the automated build script:
```bash
chmod +x build_executable.sh
./build_executable.sh
```

### Windows
1.  Open a terminal (PowerShell or CMD).
2.  Run PyInstaller with the spec file:
    ```powershell
    pyinstaller build.spec --clean --noconfirm
    ```

**Output:**
The executable will be generated in the `dist/DiffImg2ImgStudio` folder.
- **Linux:** `dist/DiffImg2ImgStudio/DiffImg2ImgStudio`
- **Windows:** `dist/DiffImg2ImgStudio/DiffImg2ImgStudio.exe`

## 3. Releasing to GitHub

### Automated (Linux/macOS)
If you have the [GitHub CLI (`gh`)](https://cli.github.com/) installed and authenticated:
```bash
chmod +x release.sh
./release.sh
```

### Manual
1.  **Archive the Build:**
    - **Linux:** `tar -czf DiffImg2ImgStudio.tar.gz -C dist DiffImg2ImgStudio`
    - **Windows:** Zip the `dist/DiffImg2ImgStudio` folder.
2.  **Create Release:**
    - Go to your GitHub repository -> **Releases** -> **Draft a new release**.
    - Tag version (e.g., `v1.0.0`).
    - Upload the archive file.
    - Publish!

## ⚠️ Important Notes
- **Size:** The executable will be large (likely >2GB) because it bundles PyTorch, CUDA libraries, and other dependencies.
- **Startup:** The app launches a local web server. On first run, it might take a moment to unpack dependencies to a temporary folder.
- **Accelerate Config:** The packaged app uses `accelerate_config.yaml` in its working directory if present.
