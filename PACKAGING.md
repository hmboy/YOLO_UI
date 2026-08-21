# YOLO_UI 打包说明（不附带源码的 Windows exe）

## 怎么打

1. 在**已经能正常运行** `python main.py` 的环境里打开终端  
2. 双击或运行：

```bat
build_exe.bat
```

3. 产物在：`dist\YOLO_UI\`  
   - 入口：`YOLO_UI.exe`  
   - 旁边有 `_internal` 等依赖目录  

## 怎么发到别的机台

- **整夹拷贝** `dist\YOLO_UI`（不要只拷一个 exe）  
- 目标机：Windows 10/11  
- 要用 GPU：显卡驱动需匹配打包时的 PyTorch/CUDA  
- 模型权重 `.pt`、数据集路径在目标机上重新选  

## 源码是否泄露

- 打包结果里**没有 `.py` 源文件**，业务逻辑在字节码/二进制归档中  
- 这不是军事级加密；专业逆向仍可能分析字节码  
- 若需要更强保护，可再考虑 Nuitka 编译或商业加固（可另做）  

## 注意

- 体积通常很大（含 torch，常到数 GB）  
- 首次打包很慢  
- 不要把本机带路径的 `data\settings.json` 打进包；程序会在 exe 旁自动生成  
