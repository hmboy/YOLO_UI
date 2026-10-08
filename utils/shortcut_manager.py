from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence


class ShortcutManager:
    """管理应用程序快捷键。"""

    def __init__(self, main_window):
        self.main_window = main_window
        self.shortcuts = {}
        self.setup_shortcuts()

    def setup_shortcuts(self):
        self.add_shortcut("Ctrl+Q", self.main_window.close, "退出应用程序")
        self.add_shortcut("F1", self.show_help, "显示帮助")
        self.add_shortcut("Ctrl+,", self.main_window.open_settings, "打开设置")
        self.add_shortcut("F11", self.toggle_fullscreen, "切换全屏模式")

    def add_shortcut(self, key_sequence, callback, description):
        shortcut = QShortcut(QKeySequence(key_sequence), self.main_window)
        shortcut.activated.connect(callback)
        self.shortcuts[key_sequence] = {
            "shortcut": shortcut,
            "description": description,
            "callback": callback,
        }

    def show_help(self):
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea, QWidget, QGridLayout,
        )

        help_dialog = QDialog(self.main_window)
        help_dialog.setWindowTitle("键盘快捷键")
        help_dialog.setMinimumSize(420, 320)

        layout = QVBoxLayout(help_dialog)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QGridLayout(scroll_content)

        scroll_layout.addWidget(QLabel("<h2>可用的键盘快捷键</h2>"), 0, 0, 1, 2)
        row = 1
        for key, info in self.shortcuts.items():
            scroll_layout.addWidget(QLabel(f"<b>{key}</b>"), row, 0)
            scroll_layout.addWidget(QLabel(info["description"]), row, 1)
            row += 1

        # 标注页内置快捷键提示
        for key, desc in (
            ("A / ←", "上一张（当前筛选）"),
            ("D / →", "下一张（当前筛选）"),
            ("N", "下一张未标注"),
            ("1-9", "切换类别"),
            ("Ctrl+S", "保存标注"),
            ("Ctrl+Z", "撤销"),
            ("Delete", "删除选中框；焦点在图像列表时删除图像"),
            ("Ctrl+点击", "多选标注框"),
            ("Alt+点击", "切换重叠层"),
            ("空白拖拽", "画矩形框"),
            ("Shift+拖拽", "在已有框上叠画"),
            ("多边形", "左键加点，双击/Enter/右键完成，Esc取消"),
            ("滚轮", "缩放"),
            ("中键 / 空格+拖", "平移画布"),
        ):
            scroll_layout.addWidget(QLabel(f"<b>{key}</b>"), row, 0)
            scroll_layout.addWidget(QLabel(desc), row, 1)
            row += 1

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(help_dialog.accept)
        layout.addWidget(close_button)
        help_dialog.exec_()

    def toggle_fullscreen(self):
        if self.main_window.isFullScreen():
            self.main_window.showNormal()
        else:
            self.main_window.showFullScreen()
