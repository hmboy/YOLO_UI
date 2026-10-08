import os
import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QGroupBox, QCheckBox, QMessageBox, QFormLayout, QTabWidget,
    QApplication, QLabel,
)
from PyQt5.QtCore import pyqtSignal
from utils.theme_manager import ThemeManager
from utils.app_paths import data_dir


class SettingsTab(QWidget):
    """应用设置：硬件 / 主题 / 标注页默认超参。"""

    settings_updated = pyqtSignal(dict)
    theme_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.settings = self._default_settings()
        self.settings_file = os.path.join(data_dir(), 'settings.json')
        self.load_settings_from_file()
        self.setup_ui()
        self.apply_settings_to_ui()

    @staticmethod
    def _default_settings():
        return {
            'default_model': 'yolo12n',
            'default_batch_size': 4,
            'default_img_size': 1280,
            'default_conf_thresh': 0.25,
            'default_iou_thresh': 0.45,
            'use_gpu': True,
            'gpu_device': 0,
            'theme': 'light',
            'last_annotation_project': '',
            'recent_annotation_projects': [],
            'window_geometry': [],
            'splitter_sizes': [],
            'roi_enabled': False,
            'roi_x1': 0.0,
            'roi_y1': 0.0,
            'roi_x2': 1.0,
            'roi_y2': 1.0,
            'roi_preview_image': '',
            'roi_view_only': False,
        }

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        settings_tabs = QTabWidget()

        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)

        hardware_group = QGroupBox('硬件')
        hardware_layout = QFormLayout()
        self.use_gpu_check = QCheckBox('使用 GPU')
        self.gpu_device_spin = QSpinBox()
        self.gpu_device_spin.setRange(0, 8)
        hardware_layout.addRow(self.use_gpu_check)
        hardware_layout.addRow('GPU 设备:', self.gpu_device_spin)
        hardware_group.setLayout(hardware_layout)

        ui_group = QGroupBox('界面')
        ui_layout = QFormLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(['浅色主题', '深色主题', '科技感主题'])
        ui_layout.addRow('主题:', self.theme_combo)
        ui_group.setLayout(ui_layout)

        general_layout.addWidget(hardware_group)
        general_layout.addWidget(ui_group)

        defaults_tab = QWidget()
        defaults_layout = QFormLayout(defaults_tab)
        hint = QLabel('以下默认值在打开新项目（或项目无已存训练参数）时填入标注页。')
        hint.setWordWrap(True)
        defaults_layout.addRow(hint)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            'yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x',
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
            'yolo11m', 'yolo11l',
            'yolo11m-seg', 'yolo11l-seg',
            'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
        ])
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 8192)
        self.img_size_spin.setSingleStep(32)
        self.conf_thresh_spin = QDoubleSpinBox()
        self.conf_thresh_spin.setRange(0.05, 0.95)
        self.conf_thresh_spin.setSingleStep(0.05)
        self.iou_thresh_spin = QDoubleSpinBox()
        self.iou_thresh_spin.setRange(0.1, 1.0)
        self.iou_thresh_spin.setSingleStep(0.05)

        defaults_layout.addRow('默认模型:', self.model_combo)
        defaults_layout.addRow('默认批次:', self.batch_size_spin)
        defaults_layout.addRow('默认尺寸:', self.img_size_spin)
        defaults_layout.addRow('默认检测置信度:', self.conf_thresh_spin)
        defaults_layout.addRow('默认检测 IoU:', self.iou_thresh_spin)

        settings_tabs.addTab(general_tab, '通用')
        settings_tabs.addTab(defaults_tab, '标注默认')

        button_layout = QHBoxLayout()
        self.save_btn = QPushButton('保存设置')
        self.save_btn.setMinimumHeight(40)
        self.reset_btn = QPushButton('恢复默认')
        self.reset_btn.setMinimumHeight(40)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.reset_btn)

        main_layout.addWidget(settings_tabs)
        main_layout.addLayout(button_layout)
        self.connect_signals()

    def connect_signals(self):
        self.save_btn.clicked.connect(self.save_settings)
        self.reset_btn.clicked.connect(self.reset_settings)
        self.use_gpu_check.toggled.connect(self.gpu_device_spin.setEnabled)
        self.theme_combo.currentIndexChanged.connect(self.apply_theme)

    def apply_theme(self, index):
        app = QApplication.instance()
        if index == 0:
            ThemeManager.apply_light_theme(app)
            self.settings['theme'] = 'light'
        elif index == 1:
            ThemeManager.apply_dark_theme(app)
            self.settings['theme'] = 'dark'
        else:
            ThemeManager.apply_tech_theme(app)
            self.settings['theme'] = 'tech'
        self.theme_changed.emit(self.settings['theme'])

    def save_settings(self, show_message=True):
        self.settings['default_model'] = self.model_combo.currentText()
        self.settings['default_batch_size'] = self.batch_size_spin.value()
        self.settings['default_img_size'] = self.img_size_spin.value()
        self.settings['default_conf_thresh'] = self.conf_thresh_spin.value()
        self.settings['default_iou_thresh'] = self.iou_thresh_spin.value()
        self.settings['use_gpu'] = self.use_gpu_check.isChecked()
        self.settings['gpu_device'] = self.gpu_device_spin.value()

        theme_index = self.theme_combo.currentIndex()
        self.settings['theme'] = ('light', 'dark', 'tech')[min(theme_index, 2)]

        for key, default in (
            ('last_annotation_project', ''),
            ('recent_annotation_projects', []),
            ('window_geometry', []),
            ('splitter_sizes', []),
        ):
            if key not in self.settings:
                self.settings[key] = default
        if not isinstance(self.settings.get('recent_annotation_projects'), list):
            self.settings['recent_annotation_projects'] = []
        if not isinstance(self.settings.get('window_geometry'), list):
            self.settings['window_geometry'] = []
        if not isinstance(self.settings.get('splitter_sizes'), list):
            self.settings['splitter_sizes'] = []

        self.settings['roi_enabled'] = bool(self.settings.get('roi_enabled', False))
        self.settings['roi_x1'] = float(self.settings.get('roi_x1', 0.0))
        self.settings['roi_y1'] = float(self.settings.get('roi_y1', 0.0))
        self.settings['roi_x2'] = float(self.settings.get('roi_x2', 1.0))
        self.settings['roi_y2'] = float(self.settings.get('roi_y2', 1.0))
        self.settings['roi_preview_image'] = self.settings.get('roi_preview_image', '') or ''
        self.settings['roi_view_only'] = bool(self.settings.get('roi_view_only', False))

        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
            self.settings_updated.emit(self.settings)
            if show_message:
                QMessageBox.information(self, '设置已保存', '设置已成功保存。')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'保存设置失败: {str(e)}')

    def load_settings_from_file(self):
        if not os.path.exists(self.settings_file):
            return
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            for key, value in loaded.items():
                if key in self.settings:
                    self.settings[key] = value
        except Exception as e:
            print(f'加载设置文件时出错: {str(e)}')

    def apply_settings_to_ui(self):
        if not hasattr(self, 'model_combo'):
            return
        index = self.model_combo.findText(self.settings['default_model'])
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.batch_size_spin.setValue(int(self.settings['default_batch_size']))
        self.img_size_spin.setValue(int(self.settings['default_img_size']))
        self.conf_thresh_spin.setValue(float(self.settings['default_conf_thresh']))
        self.iou_thresh_spin.setValue(float(self.settings['default_iou_thresh']))
        self.use_gpu_check.setChecked(bool(self.settings['use_gpu']))
        self.gpu_device_spin.setValue(int(self.settings['gpu_device']))
        self.gpu_device_spin.setEnabled(bool(self.settings['use_gpu']))

        theme = self.settings.get('theme', 'light')
        theme_index = {'light': 0, 'dark': 1}.get(theme, 2)
        self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentIndex(theme_index)
        self.theme_combo.blockSignals(False)

    def load_settings(self):
        self.load_settings_from_file()
        self.apply_settings_to_ui()

    def reset_settings(self):
        reply = QMessageBox.question(
            self, '确认重置',
            '确定将所有设置重置为默认值吗？（最近项目与窗口布局也会清空）',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.settings = self._default_settings()
        self.apply_settings_to_ui()
        self.save_settings(show_message=True)
