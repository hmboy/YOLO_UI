# -*- mode: python ; coding: utf-8 -*-
"""
YOLO_UI PyInstaller 规格：打包为目录版 exe（不含 .py 源码）。
用法: pyinstaller YOLO_UI.spec --noconfirm
产物: dist/YOLO_UI/YOLO_UI.exe
"""

import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None
try:
    project_root = os.path.abspath(SPECPATH)
except NameError:
    project_root = os.path.abspath(os.getcwd())

datas = []
binaries = []
hiddenimports = [
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.QtSvg',
    'cv2',
    'numpy',
    'yaml',
    'PIL',
    'ultralytics',
    'torch',
    'torchvision',
    'matplotlib',
    'scipy',
]

# 只读资源：图标等（不要打进用户 settings.json，避免泄露本机路径）
assets = os.path.join(project_root, 'ui', 'assets')
if os.path.isdir(assets):
    datas.append((assets, os.path.join('ui', 'assets')))

# ultralytics / torch 相关数据与隐式依赖
for pkg in ('ultralytics', 'torch', 'torchvision'):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f'[spec] collect_all({pkg}) skipped: {e}')

try:
    datas += collect_data_files('ultralytics')
except Exception:
    pass

a = Analysis(
    [os.path.join(project_root, 'main.py')],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'pytest',
        'IPython',
        'notebook',
        'jupyter',
    ],
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
    name='YOLO_UI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # torch/CUDA 建议关 UPX，避免启动异常
    console=False,  # 无黑色控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='YOLO_UI',
)
