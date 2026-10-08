import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QMessageBox, QApplication,
    QDialog, QDialogButtonBox, QAction,
)
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QFont

from ui.components.settings_tab import SettingsTab
from ui.components.annotation_tab import AnnotationTab
from utils.terminal_redirect import TerminalManager
from utils.theme_manager import ThemeManager
from utils.app_paths import assets_dir


class MainWindow(QMainWindow):
    """缺陷标注主窗口（设置从菜单打开）。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 缺陷标注工具")
        self.setMinimumSize(1200, 700)

        self.set_app_icon()
        ThemeManager.initialize()

        self.settings_tab = SettingsTab()
        self.apply_saved_theme()
        if not self._restore_window_geometry():
            screen = QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                w = max(1280, int(avail.width() * 0.85))
                h = max(800, int(avail.height() * 0.85))
                self.resize(min(w, avail.width()), min(h, avail.height()))
                frame = self.frameGeometry()
                frame.moveCenter(avail.center())
                self.move(frame.topLeft())
            else:
                self.resize(1280, 800)

        self.terminal_manager = TerminalManager()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.annotation_tab = AnnotationTab()
        main_layout.addWidget(self.annotation_tab)

        self._setup_menu()
        self.setup_connections()
        self.setFont(QFont("Segoe UI", 13))
        self.statusBar().showMessage(
            'Del 删除 · Ctrl+Z 撤销 · A/D 切图 · 滚轮缩放 · 空格拖拽 · F1 全部快捷键'
        )
        self.setup_terminal_redirection()
        self.load_default_settings()
        self.force_apply_theme()

    def _setup_menu(self):
        menu = self.menuBar().addMenu('工具')
        settings_action = QAction('设置...', self)
        settings_action.setShortcut('Ctrl+,')
        settings_action.triggered.connect(self.open_settings)
        menu.addAction(settings_action)

    def open_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('设置')
        dlg.resize(920, 720)
        layout = QVBoxLayout(dlg)
        layout.addWidget(self.settings_tab)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        buttons.clicked.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec_()
        # 避免随对话框销毁
        self.settings_tab.setParent(self)

    def force_apply_theme(self):
        app = QApplication.instance()
        theme = self.settings_tab.settings.get('theme', 'light')
        if theme == 'light':
            ThemeManager.apply_light_theme(app)
        elif theme == 'dark':
            ThemeManager.apply_dark_theme(app)
        else:
            ThemeManager.apply_tech_theme(app)
        self.annotation_tab.apply_theme()
        self.update()
        self.annotation_tab.update()

    def _restore_window_geometry(self) -> bool:
        geom = self.settings_tab.settings.get('window_geometry')
        if not isinstance(geom, (list, tuple)) or len(geom) != 4:
            return False
        try:
            x, y, w, h = [int(v) for v in geom]
        except (TypeError, ValueError):
            return False
        if w < 800 or h < 500:
            return False
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(max(w, 1200), avail.width())
            h = min(max(h, 700), avail.height())
            x = max(avail.x(), min(x, avail.x() + avail.width() - 120))
            y = max(avail.y(), min(y, avail.y() + avail.height() - 80))
        self.setGeometry(x, y, w, h)
        return True

    def _persist_window_layout(self):
        g = self.geometry()
        self.settings_tab.settings['window_geometry'] = [
            g.x(), g.y(), g.width(), g.height(),
        ]
        sizes = self.annotation_tab.get_splitter_sizes()
        if sizes:
            self.settings_tab.settings['splitter_sizes'] = sizes
        try:
            self.settings_tab.save_settings(show_message=False)
        except Exception as e:
            print(f"保存窗口布局失败: {e}")

    def set_app_icon(self):
        icon_path = os.path.join(assets_dir(), "app_icon.svg")
        try:
            app_icon = QIcon(icon_path)
            self.setWindowIcon(app_icon)
            QApplication.instance().setWindowIcon(app_icon)
        except Exception as e:
            print(f"设置应用图标时出错: {str(e)}")

    def setup_connections(self):
        self.settings_tab.settings_updated.connect(self.annotation_tab.update_settings)
        self.settings_tab.theme_changed.connect(self.on_theme_changed)

    def on_theme_changed(self, theme):
        self.force_apply_theme()
        print(f"主题已变更为: {theme}")

    def apply_saved_theme(self):
        theme = self.settings_tab.settings.get('theme', 'light')
        app = QApplication.instance()
        if theme == 'light':
            ThemeManager.apply_light_theme(app)
        elif theme == 'dark':
            ThemeManager.apply_dark_theme(app)
        else:
            ThemeManager.apply_tech_theme(app)

    def setup_terminal_redirection(self):
        try:
            self.terminal_manager.start_redirection()
        except Exception as e:
            print(f"设置终端重定向时出错: {str(e)}")

    def closeEvent(self, event):
        busy = (
            (hasattr(self.annotation_tab, 'is_training') and self.annotation_tab.is_training)
            or (hasattr(self.annotation_tab, 'is_detecting') and self.annotation_tab.is_detecting)
        )
        unsaved = (
            hasattr(self.annotation_tab, 'has_unsaved_changes')
            and self.annotation_tab.has_unsaved_changes()
        )
        if busy or unsaved:
            if busy and unsaved:
                message = "有进程正在运行，且标注尚未保存。确定要退出吗？"
            elif busy:
                message = "有进程正在运行。确定要退出吗？"
            else:
                message = "标注有未保存更改。确定要退出吗？"
            reply = QMessageBox.question(
                self, '确认退出', message,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        self._persist_window_layout()
        self.terminal_manager.stop_redirection()
        event.accept()

    def load_default_settings(self):
        try:
            self.annotation_tab.update_settings(self.settings_tab.settings)
            self.annotation_tab.restore_splitter_sizes()
            self.annotation_tab.refresh_recent_menu()
            try:
                self.annotation_tab.restore_last_project()
            except Exception as restore_err:
                print(f"自动打开上次项目失败: {restore_err}")
            print("默认设置已加载。")
        except Exception as e:
            import traceback
            print(f"加载默认设置时出错: {str(e)}")
            print(traceback.format_exc())
