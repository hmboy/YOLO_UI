@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM 优先使用本机 yolotrain 环境（含 torch/ultralytics）
set "PY=A:\ProgramData\anaconda3\envs\yolotrain\python.exe"
if not exist "%PY%" set "PY=python"

echo ========================================
echo  YOLO_UI 打包 (PyInstaller, 目录版 exe)
echo  使用: %PY%
echo ========================================
echo.

"%PY%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
  echo [1/3] 安装 PyInstaller...
  "%PY%" -m pip install -U pyinstaller
) else (
  echo [1/3] PyInstaller 已安装
)

echo [2/3] 开始打包（体积大、耗时长，请耐心等待）...
"%PY%" -m PyInstaller YOLO_UI.spec --noconfirm --clean
if errorlevel 1 (
  echo.
  echo 打包失败。请确认当前环境已能正常 python main.py 运行。
  pause
  exit /b 1
)

echo [3/3] 完成
echo.
echo 输出目录: %cd%\dist\YOLO_UI\
echo 入口文件: %cd%\dist\YOLO_UI\YOLO_UI.exe
echo.
echo 分发: 拷贝整个 dist\YOLO_UI 文件夹到目标机（不要只拷 exe）
echo.
pause
