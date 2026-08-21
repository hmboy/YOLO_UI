"""应用路径：开发运行与 PyInstaller 冻结 exe 共用。"""
import os
import sys


def is_frozen() -> bool:
    return bool(getattr(sys, 'frozen', False))


def app_root() -> str:
    """
    可写的应用根目录：
    - 源码运行：项目根目录
    - 冻结 exe：exe 所在目录（settings、日志等写在这里）
    """
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_root() -> str:
    """
    只读资源根目录（ui/assets 等）：
    - 源码运行：项目根目录
    - 冻结 exe：PyInstaller 解包目录 (_MEIPASS) 或 exe 旁
    """
    if is_frozen():
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            return meipass
        return os.path.dirname(os.path.abspath(sys.executable))
    return app_root()


def data_dir() -> str:
    path = os.path.join(app_root(), 'data')
    os.makedirs(path, exist_ok=True)
    return path


def assets_dir() -> str:
    return os.path.join(resource_root(), 'ui', 'assets')
