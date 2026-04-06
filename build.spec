# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# Collect all necessary data for Streamlit and other packages
datas = []
binaries = []
hiddenimports = ['streamlit', 'streamlit.runtime.scriptrunner.magic_funcs']

# Collect data for key libraries
packages_to_collect = ['streamlit', 'altair', 'pillow', 'pandas', 'numpy', 'torch', 'diffusers', 'accelerate', 'timm', 'cv2', 'scipy', 'matplotlib']

for package in packages_to_collect:
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

# Add custom project files
datas += [('app.py', '.'), ('diffusion_trainer.py', '.'), ('diffusion_val.py', '.'), ('darker.py', '.'), ('visual_val.py', '.')]
datas += [('models/', 'models/'), ('utils/', 'utils/'), ('datasets/', 'datasets/')]
datas += [('examples/', 'examples/')]

block_cipher = None

a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DiffImg2ImgStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DiffImg2ImgStudio',
)
