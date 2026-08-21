import os
import re
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QGroupBox, QMessageBox, QTextEdit, QLineEdit, QDoubleSpinBox, QSpinBox,
    QListWidget, QListWidgetItem, QSplitter, QComboBox, QInputDialog,
    QAbstractItemView, QProgressBar, QCheckBox, QFormLayout, QRadioButton,
    QButtonGroup, QScrollArea, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, QEvent, QThread
from PyQt5.QtGui import QColor

from utils.annotation_manager import AnnotationManager, BBox, Polygon
from utils.training_worker import TrainingWorker
from utils.detect_all_worker import DetectAllWorker
from ui.components.annotation_canvas import AnnotationCanvas


class AnnotationTab(QWidget):
    """缺陷标注 Tab：画框标注、类别管理、导出、一键训练并检测。"""

    DEFAULT_DEFECT_CLASSES = ['划痕', '凹坑', '污渍', '裂纹', '异物', '其他']
    _ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07')
    _NOISE_RE = re.compile(
        r'(━|░|█|\r|it/s|GPU_mem|Transferred |Freezing layer|Overriding model|'
        r'New https://|ping:|Fast image access|Scanning |New cache created|'
        r'optimizer:|Image sizes |Using \d+ dataloader|Logging results to|'
        r'Starting training for|engine[/\\]trainer|from\s+n\s+params)',
        re.IGNORECASE,
    )

    def __init__(self):
        super().__init__()
        self.manager = AnnotationManager()
        self._dirty = False
        self._auto_save_timer = QTimer()
        self._auto_save_timer.setSingleShot(True)
        self._auto_save_timer.timeout.connect(self._save_current)
        self._settings_save_timer = QTimer()
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.timeout.connect(self._persist_train_settings)

        self.is_training = False
        self.is_detecting = False
        self._pipeline_active = False
        self._best_weights = ''
        self._detection_results = {}  # image_path -> [det, ...]
        self._settings = {}
        self._log_font_size = 8
        self._log_compact_max = 280
        self._log_enlarge_min = 380
        self._loading_settings = False

        self.training_worker = None
        self.training_thread = None
        self.detect_worker = None
        self.detect_thread = None

        self.setup_ui()
        self.setup_connections()
        self._install_shortcut_filter()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        toolbar = QHBoxLayout()
        self.open_project_btn = QPushButton('打开/创建项目')
        self.import_images_btn = QPushButton('导入图像')
        self.export_btn = QPushButton('导出数据集')
        self.export_btn.setStyleSheet('font-weight: bold;')
        toolbar.addWidget(self.open_project_btn)
        toolbar.addWidget(self.import_images_btn)
        toolbar.addWidget(self.export_btn)
        toolbar.addStretch()
        self.project_label = QLabel('未打开项目')
        self.project_label.setStyleSheet('color: gray;')
        toolbar.addWidget(self.project_label)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        filter_layout = QVBoxLayout()
        filter_top = QHBoxLayout()
        filter_top.addWidget(QLabel('筛选:'))
        self.filter_logic_combo = QComboBox()
        self.filter_logic_combo.addItems(['与', '或'])
        self.filter_logic_combo.setToolTip(
            '与：同时满足下方所有非「全部」条件\n'
            '或：满足任一非「全部」条件即可'
        )
        self.filter_logic_combo.setMaximumWidth(100)
        filter_top.addWidget(self.filter_logic_combo)
        filter_top.addStretch(1)
        filter_layout.addLayout(filter_top)

        filter_conds = QHBoxLayout()
        self.filter_status_combo = QComboBox()
        self.filter_status_combo.addItems(['状态:全部', '已标注', '未标注'])
        self.filter_status_combo.setToolTip('标注状态')
        self.filter_detect_combo = QComboBox()
        self.filter_detect_combo.addItems(['检测:全部', '有检测', '无检测'])
        self.filter_detect_combo.setToolTip('是否有检测结果')
        self.filter_split_combo = QComboBox()
        self.filter_split_combo.addItems(['划分:全部', '训练', '验证', '仅标注', '未分配'])
        self.filter_split_combo.setToolTip('数据划分')
        filter_conds.addWidget(self.filter_status_combo)
        filter_conds.addWidget(self.filter_detect_combo)
        filter_conds.addWidget(self.filter_split_combo)
        filter_layout.addLayout(filter_conds)
        left_layout.addLayout(filter_layout)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        left_layout.addWidget(self.image_list)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel('分配:'))
        self.split_train_btn = QPushButton('训练')
        self.split_val_btn = QPushButton('验证')
        self.split_mark_btn = QPushButton('仅标注')
        self.split_clear_btn = QPushButton('清除')
        for btn in (self.split_train_btn, self.split_val_btn, self.split_mark_btn, self.split_clear_btn):
            btn.setMaximumHeight(26)
            split_row.addWidget(btn)
        left_layout.addLayout(split_row)
        self.split_hint = QLabel('多选图像后点上方按钮分配；导出/训练按分配结果')
        self.split_hint.setStyleSheet('color: gray; font-size: 12px;')
        self.split_hint.setWordWrap(True)
        left_layout.addWidget(self.split_hint)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton('上一张')
        self.next_btn = QPushButton('下一张')
        self.image_index_label = QLabel('0/0')
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.image_index_label, alignment=Qt.AlignCenter)
        nav_layout.addWidget(self.next_btn)
        left_layout.addLayout(nav_layout)
        splitter.addWidget(left_panel)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        canvas_toolbar = QHBoxLayout()
        canvas_toolbar.addWidget(QLabel('当前类别:'))
        self.class_combo = QComboBox()
        canvas_toolbar.addWidget(self.class_combo, stretch=1)

        canvas_toolbar.addWidget(QLabel('标注:'))
        self.tool_bbox_radio = QRadioButton('矩形框')
        self.tool_polygon_radio = QRadioButton('多边形')
        self.tool_bbox_radio.setChecked(True)
        self.draw_tool_group = QButtonGroup(self)
        self.draw_tool_group.addButton(self.tool_bbox_radio)
        self.draw_tool_group.addButton(self.tool_polygon_radio)
        canvas_toolbar.addWidget(self.tool_bbox_radio)
        canvas_toolbar.addWidget(self.tool_polygon_radio)

        self.delete_box_btn = QPushButton('删除选中 (Del)')
        canvas_toolbar.addWidget(self.delete_box_btn)
        self.show_gt_check = QCheckBox('显示标注')
        self.show_gt_check.setChecked(True)
        self.show_det_check = QCheckBox('显示检测')
        self.show_det_check.setChecked(True)
        self.roi_view_only_check = QCheckBox('只显示ROI')
        self.roi_view_only_check.setToolTip('勾选后切换图像时仅显示全局 ROI 区域内容')
        canvas_toolbar.addWidget(self.show_gt_check)
        canvas_toolbar.addWidget(self.show_det_check)
        canvas_toolbar.addWidget(self.roi_view_only_check)
        center_layout.addLayout(canvas_toolbar)

        self.canvas = AnnotationCanvas()
        self.canvas.set_manager(self.manager)
        self.canvas.set_owner(self)
        self.canvas.setMinimumSize(640, 480)
        center_layout.addWidget(self.canvas, stretch=1)

        self.hint_label = QLabel(
            '点选标注可删除(Del) | Ctrl+点击多选 | 空白处拖拽画框 | '
            '多边形: 左键加点，双击/Enter/右键完成，Esc取消 | '
            '滚轮缩放 | 中键/空格+拖拽平移 | ←→切图 | 1-9类别'
        )
        self.hint_label.setStyleSheet('color: gray; font-size: 12px;')
        self.hint_label.setWordWrap(True)
        center_layout.addWidget(self.hint_label)
        splitter.addWidget(center_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setFrameShape(QScrollArea.NoFrame)
        right_scroll_content = QWidget()
        scroll_layout = QVBoxLayout(right_scroll_content)
        scroll_layout.setContentsMargins(0, 0, 4, 0)
        scroll_layout.setSpacing(8)

        class_group = QGroupBox('缺陷类别管理')
        class_group_layout = QVBoxLayout(class_group)
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(110)
        class_group_layout.addWidget(self.class_list)
        class_btn_layout = QHBoxLayout()
        self.add_class_btn = QPushButton('添加')
        self.rename_class_btn = QPushButton('重命名')
        self.remove_class_btn = QPushButton('删除')
        class_btn_layout.addWidget(self.add_class_btn)
        class_btn_layout.addWidget(self.rename_class_btn)
        class_btn_layout.addWidget(self.remove_class_btn)
        class_group_layout.addLayout(class_btn_layout)
        scroll_layout.addWidget(class_group)

        stats_group = QGroupBox('统计信息')
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel('暂无数据')
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        scroll_layout.addWidget(stats_group)

        train_group = QGroupBox('训练并检测')
        train_form = QFormLayout(train_group)
        train_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.train_model_combo = QComboBox()
        self.train_model_combo.addItems([
            'yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x',
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
            'yolo11m', 'yolo11l',
            'yolo11m-seg', 'yolo11l-seg',
        ])
        self.train_epochs_spin = QSpinBox()
        self.train_epochs_spin.setRange(1, 10000)
        self.train_epochs_spin.setValue(50)
        self.train_batch_spin = QSpinBox()
        self.train_batch_spin.setRange(1, 128)
        self.train_batch_spin.setValue(4)
        self.train_imgsz_spin = QSpinBox()
        self.train_imgsz_spin.setRange(32, 8192)
        self.train_imgsz_spin.setSingleStep(32)
        self.train_imgsz_spin.setValue(1280)
        self.train_lr_spin = QDoubleSpinBox()
        self.train_lr_spin.setRange(0.00001, 0.1)
        self.train_lr_spin.setDecimals(5)
        self.train_lr_spin.setSingleStep(0.001)
        self.train_lr_spin.setValue(0.01)

        init_box = QGroupBox('模型初始化')
        init_layout = QVBoxLayout(init_box)
        self.train_init_group = QButtonGroup(self)
        self.use_pretrained_radio = QRadioButton('使用预训练权重')
        self.from_scratch_radio = QRadioButton('从头训练（无预训练）')
        self.custom_weights_radio = QRadioButton('使用自定义权重')
        self.train_init_group.addButton(self.use_pretrained_radio)
        self.train_init_group.addButton(self.from_scratch_radio)
        self.train_init_group.addButton(self.custom_weights_radio)
        self.use_pretrained_radio.setChecked(True)
        init_layout.addWidget(self.use_pretrained_radio)
        init_layout.addWidget(self.from_scratch_radio)
        init_layout.addWidget(self.custom_weights_radio)

        weights_row = QHBoxLayout()
        self.train_weights_edit = QLineEdit()
        self.train_weights_edit.setReadOnly(True)
        self.train_weights_edit.setPlaceholderText('自定义 .pt 权重路径')
        self.train_weights_btn = QPushButton('浏览...')
        self.train_weights_edit.setEnabled(False)
        self.train_weights_btn.setEnabled(False)
        weights_row.addWidget(self.train_weights_edit)
        weights_row.addWidget(self.train_weights_btn)
        init_layout.addLayout(weights_row)

        self.fine_tuning_check = QCheckBox('微调模式（冻结骨干，仅训练检测头）')
        self.fine_tuning_check.setChecked(False)
        init_layout.addWidget(self.fine_tuning_check)

        self.detect_conf_spin = QDoubleSpinBox()
        self.detect_conf_spin.setRange(0.05, 0.95)
        self.detect_conf_spin.setSingleStep(0.05)
        self.detect_conf_spin.setValue(0.25)

        train_form.addRow('模型:', self.train_model_combo)
        train_form.addRow('轮数:', self.train_epochs_spin)
        train_form.addRow('批次:', self.train_batch_spin)
        train_form.addRow('尺寸:', self.train_imgsz_spin)
        train_form.addRow('学习率:', self.train_lr_spin)
        train_form.addRow(init_box)
        train_form.addRow('检测置信度:', self.detect_conf_spin)

        # 全局训练 ROI（对所有图按比例裁剪）
        roi_box = QGroupBox('全局训练 ROI')
        roi_layout = QVBoxLayout(roi_box)
        self.roi_enabled_check = QCheckBox('启用（训练时对所有图像裁剪）')
        roi_btn_row = QHBoxLayout()
        self.roi_pick_btn = QPushButton('框选 ROI...')
        self.roi_clear_btn = QPushButton('清除')
        roi_btn_row.addWidget(self.roi_pick_btn)
        roi_btn_row.addWidget(self.roi_clear_btn)
        roi_layout.addWidget(self.roi_enabled_check)
        roi_layout.addLayout(roi_btn_row)
        train_form.addRow(roi_box)

        train_btn_row = QHBoxLayout()
        self.train_detect_btn = QPushButton('开始训练并检测')
        self.train_detect_btn.setStyleSheet('font-weight: bold;')
        self.stop_pipeline_btn = QPushButton('停止')
        self.stop_pipeline_btn.setEnabled(False)
        self.redetect_btn = QPushButton('仅重新检测')
        train_btn_row.addWidget(self.train_detect_btn)
        train_btn_row.addWidget(self.stop_pipeline_btn)
        train_form.addRow(train_btn_row)
        train_form.addRow(self.redetect_btn)

        self.pipeline_progress = QProgressBar()
        self.pipeline_progress.setRange(0, 100)
        self.pipeline_progress.setValue(0)
        self.pipeline_progress.setFormat('%v / %m')
        self.pipeline_progress.setTextVisible(True)
        self.pipeline_status = QLabel('空闲')
        self.pipeline_status.setWordWrap(True)
        train_form.addRow(self.pipeline_progress)
        train_form.addRow(self.pipeline_status)
        scroll_layout.addWidget(train_group)

        self.box_list = QGroupBox('当前图像框')
        box_list_layout = QVBoxLayout(self.box_list)
        self.annotation_list = QListWidget()
        self.annotation_list.setMaximumHeight(120)
        box_list_layout.addWidget(self.annotation_list)
        scroll_layout.addWidget(self.box_list)
        scroll_layout.setAlignment(Qt.AlignTop)

        right_scroll.setWidget(right_scroll_content)
        right_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # stretch=0：上方设置区按内容高度，避免撑出空白把日志顶下去
        right_layout.addWidget(right_scroll, stretch=0)

        log_header = QHBoxLayout()
        log_header.addWidget(QLabel('日志'))
        log_header.addStretch(1)
        self.log_font_down_btn = QPushButton('A-')
        self.log_font_down_btn.setMinimumWidth(56)
        self.log_font_down_btn.setFixedHeight(32)
        self.log_font_down_btn.setToolTip('缩小日志字体')
        self.log_font_up_btn = QPushButton('A+')
        self.log_font_up_btn.setMinimumWidth(56)
        self.log_font_up_btn.setFixedHeight(32)
        self.log_font_up_btn.setToolTip('放大日志字体')
        for btn in (self.log_font_down_btn, self.log_font_up_btn):
            f = btn.font()
            f.setPointSize(11)
            f.setBold(True)
            btn.setFont(f)
        self.log_enlarge_check = QCheckBox('放大日志')
        self.log_enlarge_check.setChecked(True)
        self.log_enlarge_check.setToolTip('扩大日志区域高度，便于查看训练输出')
        log_header.addWidget(self.log_font_down_btn)
        log_header.addWidget(self.log_font_up_btn)
        log_header.addWidget(self.log_enlarge_check)
        right_layout.addLayout(log_header)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(self._log_enlarge_min)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        font = self.log_text.font()
        font.setPointSize(self._log_font_size)
        font.setFamily('Consolas')
        self.log_text.setFont(font)
        right_layout.addWidget(self.log_text, stretch=1)
        splitter.addWidget(right_panel)

        splitter.setSizes([220, 680, 360])
        layout.addWidget(splitter, stretch=1)

        self.setFocusPolicy(Qt.StrongFocus)
        self.image_list.setFocusPolicy(Qt.ClickFocus)
        self.annotation_list.setFocusPolicy(Qt.ClickFocus)
        self.class_list.setFocusPolicy(Qt.ClickFocus)

    def _install_shortcut_filter(self):
        self.installEventFilter(self)
        for child in self.findChildren(QWidget):
            child.installEventFilter(self)

    def _is_text_editing(self, widget) -> bool:
        if widget is None:
            return False
        if isinstance(widget, (QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox)):
            return True
        parent = widget.parentWidget()
        while parent and parent is not self:
            if isinstance(parent, (QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox)):
                return True
            parent = parent.parentWidget()
        return False

    def _is_active_tab(self) -> bool:
        if not self.isVisible():
            return False
        return self.isVisibleTo(self.window())

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and self._is_active_tab():
            if self._is_text_editing(obj if isinstance(obj, QWidget) else None):
                return super().eventFilter(obj, event)
            if obj is self.canvas or (isinstance(obj, QWidget) and self.canvas.isAncestorOf(obj)):
                key = event.key()
                if key in (Qt.Key_Space, Qt.Key_Escape, Qt.Key_Delete,
                           Qt.Key_Plus, Qt.Key_Equal, Qt.Key_Minus, Qt.Key_0):
                    return super().eventFilter(obj, event)
            if self._handle_global_shortcut(event):
                return True
        return super().eventFilter(obj, event)

    def _handle_global_shortcut(self, event) -> bool:
        key = event.key()
        mods = event.modifiers()

        if mods & Qt.ControlModifier and key == Qt.Key_S:
            self._save_current()
            self.log('已保存当前标注')
            return True

        if mods & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            return False

        if Qt.Key_1 <= key <= Qt.Key_9:
            idx = key - Qt.Key_1
            if idx < self.class_combo.count():
                self.class_combo.setCurrentIndex(idx)
            return True

        if key in (Qt.Key_Left, Qt.Key_Up, Qt.Key_A):
            self.prev_image()
            return True
        if key in (Qt.Key_Right, Qt.Key_Down, Qt.Key_D):
            self.next_image()
            return True
        if key == Qt.Key_Delete:
            self._delete_annotation()
            return True
        return False

    def _delete_annotation(self):
        """删除画布选中项；若无选中则删除列表当前行。"""
        if self.canvas.delete_selected():
            return
        row = self.annotation_list.currentRow()
        if 0 <= row < len(self.canvas.box_items):
            self.canvas.select_only(self.canvas.box_items[row])
            self.canvas.delete_selected()

    def setup_connections(self):
        self.open_project_btn.clicked.connect(self.open_project)
        self.import_images_btn.clicked.connect(self.import_images)
        self.export_btn.clicked.connect(self.export_dataset)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.image_list.currentRowChanged.connect(self.on_image_selected)
        self.filter_logic_combo.currentIndexChanged.connect(self.refresh_image_list)
        self.filter_status_combo.currentIndexChanged.connect(self.refresh_image_list)
        self.filter_detect_combo.currentIndexChanged.connect(self.refresh_image_list)
        self.filter_split_combo.currentIndexChanged.connect(self.refresh_image_list)
        self.class_combo.currentIndexChanged.connect(self.on_class_changed)
        self.add_class_btn.clicked.connect(self.add_class)
        self.rename_class_btn.clicked.connect(self.rename_class)
        self.remove_class_btn.clicked.connect(self.remove_class)
        self.delete_box_btn.clicked.connect(self._delete_annotation)
        self.annotation_list.currentRowChanged.connect(self.on_annotation_selected)
        self.canvas.boxes_changed.connect(self.on_boxes_changed)
        self.show_gt_check.toggled.connect(self._on_overlay_toggled)
        self.show_det_check.toggled.connect(self._on_overlay_toggled)
        self.roi_view_only_check.toggled.connect(self._on_roi_view_only_toggled)

        self.split_train_btn.clicked.connect(lambda: self._assign_selected_split('train'))
        self.split_val_btn.clicked.connect(lambda: self._assign_selected_split('val'))
        self.split_mark_btn.clicked.connect(lambda: self._assign_selected_split('mark'))
        self.split_clear_btn.clicked.connect(lambda: self._assign_selected_split(''))

        self.train_detect_btn.clicked.connect(self.start_train_and_detect)
        self.stop_pipeline_btn.clicked.connect(self.stop_pipeline)
        self.redetect_btn.clicked.connect(self.start_redetect_only)
        self.train_weights_btn.clicked.connect(self._browse_train_weights)
        self.use_pretrained_radio.toggled.connect(self._on_train_init_changed)
        self.from_scratch_radio.toggled.connect(self._on_train_init_changed)
        self.custom_weights_radio.toggled.connect(self._on_train_init_changed)
        self.roi_pick_btn.clicked.connect(self.open_roi_picker)
        self.roi_clear_btn.clicked.connect(self.clear_roi)
        self.roi_enabled_check.toggled.connect(self._on_roi_enabled_toggled)
        self.log_enlarge_check.toggled.connect(self._toggle_log_enlarge)
        self.log_enlarge_check.toggled.connect(self._schedule_settings_save)
        self.log_font_up_btn.clicked.connect(lambda: self._change_log_font(1))
        self.log_font_down_btn.clicked.connect(lambda: self._change_log_font(-1))
        self.tool_bbox_radio.toggled.connect(self._on_draw_tool_changed)
        self.tool_polygon_radio.toggled.connect(self._on_draw_tool_changed)
        self.train_model_combo.currentTextChanged.connect(self._on_model_changed_for_tool)

        for w in (
            self.train_model_combo, self.train_epochs_spin, self.train_batch_spin,
            self.train_imgsz_spin, self.train_lr_spin, self.detect_conf_spin,
            self.fine_tuning_check, self.train_weights_edit,
        ):
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self._schedule_settings_save)
            elif isinstance(w, QCheckBox):
                w.toggled.connect(self._schedule_settings_save)
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(self._schedule_settings_save)
            else:
                w.valueChanged.connect(self._schedule_settings_save)

    def log(self, msg: str):
        """写入干净日志：去 ANSI、过滤进度条/刷屏行，带时间戳。"""
        if msg is None:
            return
        text = self._ANSI_RE.sub('', str(msg))
        text = text.replace('\r', '\n')
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if self._NOISE_RE.search(line):
                continue
            # 过长的参数 dump
            if line.startswith('agnostic_nms=') or 'ultralytics.nn.modules' in line:
                continue
            if len(line) > 400:
                line = line[:400] + '...'
            ts = datetime.now().strftime('%H:%M:%S')
            self.log_text.append(f'[{ts}] {line}')

    def clear_terminal(self):
        self.log_text.clear()

    def _toggle_log_enlarge(self, checked: bool):
        if checked:
            self.log_text.setMaximumHeight(16777215)
            self.log_text.setMinimumHeight(self._log_enlarge_min)
            self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        else:
            self.log_text.setMinimumHeight(0)
            self.log_text.setMaximumHeight(self._log_compact_max)
            self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def _change_log_font(self, delta: int):
        self._log_font_size = max(8, min(28, self._log_font_size + delta))
        font = self.log_text.font()
        font.setPointSize(self._log_font_size)
        self.log_text.setFont(font)
        self._schedule_settings_save()

    def _on_draw_tool_changed(self, checked=False):
        if not checked:
            return
        tool = 'polygon' if self.tool_polygon_radio.isChecked() else 'bbox'
        self.canvas.set_draw_tool(tool)
        self._schedule_settings_save()

    def _on_model_changed_for_tool(self, model_name: str):
        """选择 seg 模型时自动切到多边形标注。"""
        if '-seg' in (model_name or '').lower():
            if not self.tool_polygon_radio.isChecked():
                self.tool_polygon_radio.setChecked(True)
                self.log('已切换到多边形标注（seg 模型）')
        self._schedule_settings_save()

    def _export_task_for_model(self) -> str:
        model = self.train_model_combo.currentText().lower()
        return 'segment' if '-seg' in model else 'detect'

    def _browse_train_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择自定义权重', '', '模型文件 (*.pt);;所有文件 (*)'
        )
        if path:
            self.train_weights_edit.setText(path)
            self._schedule_settings_save()

    def _on_train_init_changed(self, _checked=False):
        is_custom = self.custom_weights_radio.isChecked()
        self.train_weights_edit.setEnabled(is_custom)
        self.train_weights_btn.setEnabled(is_custom)
        if not is_custom:
            # 保留路径文本以便切回时仍可见，但禁用编辑
            pass
        can_finetune = self.use_pretrained_radio.isChecked() or (
            self.custom_weights_radio.isChecked() and bool(self.train_weights_edit.text().strip())
        )
        self.fine_tuning_check.setEnabled(can_finetune)
        if self.fine_tuning_check.isChecked() and not can_finetune:
            self.fine_tuning_check.setChecked(False)
        self._schedule_settings_save()

    def _refresh_roi_ui(self):
        if not hasattr(self, 'roi_enabled_check'):
            return
        self.roi_enabled_check.blockSignals(True)
        self.roi_enabled_check.setChecked(bool(self._settings.get('roi_enabled', False)))
        self.roi_enabled_check.blockSignals(False)
        if hasattr(self, 'roi_view_only_check'):
            self.roi_view_only_check.blockSignals(True)
            self.roi_view_only_check.setChecked(bool(self._settings.get('roi_view_only', False)))
            self.roi_view_only_check.blockSignals(False)
        self._apply_view_roi_to_canvas(reload=False)

    def _current_roi_norm(self):
        roi = (
            float(self._settings.get('roi_x1', 0.0)),
            float(self._settings.get('roi_y1', 0.0)),
            float(self._settings.get('roi_x2', 1.0)),
            float(self._settings.get('roi_y2', 1.0)),
        )
        is_full = (
            abs(roi[0]) < 1e-6 and abs(roi[1]) < 1e-6
            and abs(roi[2] - 1.0) < 1e-6 and abs(roi[3] - 1.0) < 1e-6
        )
        return None if is_full else roi

    def _apply_view_roi_to_canvas(self, reload=True):
        """根据勾选状态设置画布预览裁剪。"""
        view_only = bool(self._settings.get('roi_view_only', False))
        roi = self._current_roi_norm() if view_only else None
        if view_only and roi is None:
            # 勾选了但未设置 ROI，不裁剪
            self.canvas.set_view_roi(None)
        else:
            self.canvas.set_view_roi(roi)
        if reload and self.manager.current_index >= 0:
            self.load_image_at(self.manager.current_index)

    def _on_roi_view_only_toggled(self, checked):
        self._settings['roi_view_only'] = bool(checked)
        if checked and self._current_roi_norm() is None:
            self.log('请先框选全局 ROI，再开启「只显示ROI」')
        self._persist_roi_to_settings()
        self._apply_view_roi_to_canvas(reload=True)

    def _persist_roi_to_settings(self):
        """把 ROI 写回全局 settings.json。"""
        settings_tab = self._find_settings_tab()
        if settings_tab is None:
            return
        for key in (
            'roi_enabled', 'roi_x1', 'roi_y1', 'roi_x2', 'roi_y2',
            'roi_preview_image', 'roi_view_only',
        ):
            settings_tab.settings[key] = self._settings.get(key, settings_tab.settings.get(key))
        try:
            settings_tab.save_settings(show_message=False)
        except Exception:
            pass

    def _on_roi_enabled_toggled(self, checked):
        self._settings['roi_enabled'] = bool(checked)
        self._persist_roi_to_settings()

    def _roi_preview_candidate(self) -> str:
        preview = self._settings.get('roi_preview_image', '') or ''
        if preview and os.path.isfile(preview):
            return preview
        cur = self.manager.current_image_path
        if cur and os.path.isfile(cur):
            return cur
        if self.manager.image_paths:
            return self.manager.image_paths[0]
        return ''

    def open_roi_picker(self):
        from ui.components.roi_picker_dialog import RoiPickerDialog
        # 优先用全局设置里已保存的 ROI，避免内存状态不同步
        settings_tab = self._find_settings_tab()
        src = settings_tab.settings if settings_tab else self._settings
        roi = (
            float(src.get('roi_x1', self._settings.get('roi_x1', 0.0))),
            float(src.get('roi_y1', self._settings.get('roi_y1', 0.0))),
            float(src.get('roi_x2', self._settings.get('roi_x2', 1.0))),
            float(src.get('roi_y2', self._settings.get('roi_y2', 1.0))),
        )
        is_full = (
            abs(roi[0]) < 1e-6 and abs(roi[1]) < 1e-6
            and abs(roi[2] - 1.0) < 1e-6 and abs(roi[3] - 1.0) < 1e-6
        )
        roi_arg = None if is_full else roi
        preview = self._roi_preview_candidate()
        if settings_tab and settings_tab.settings.get('roi_preview_image'):
            p = settings_tab.settings['roi_preview_image']
            if p and os.path.isfile(p):
                preview = p
        dlg = RoiPickerDialog(self, image_path=preview, roi_norm=roi_arg)
        if dlg.exec_() != dlg.Accepted:
            return
        roi_norm, image_path = dlg.result_roi()
        if not roi_norm:
            return
        x1, y1, x2, y2 = roi_norm
        self._settings['roi_x1'] = float(x1)
        self._settings['roi_y1'] = float(y1)
        self._settings['roi_x2'] = float(x2)
        self._settings['roi_y2'] = float(y2)
        self._settings['roi_preview_image'] = image_path or ''
        self._settings['roi_enabled'] = True
        self._refresh_roi_ui()
        self._persist_roi_to_settings()
        self._apply_view_roi_to_canvas(reload=True)
        w, h = x2 - x1, y2 - y1
        self.log(
            f'已设置全局 ROI: 左上角 ({x1:.4f}, {y1:.4f}) 宽×高 ({w:.4f} × {h:.4f})'
        )

    def clear_roi(self):
        self._settings['roi_enabled'] = False
        self._settings['roi_x1'] = 0.0
        self._settings['roi_y1'] = 0.0
        self._settings['roi_x2'] = 1.0
        self._settings['roi_y2'] = 1.0
        self._settings['roi_preview_image'] = ''
        self._refresh_roi_ui()
        self._persist_roi_to_settings()
        self._apply_view_roi_to_canvas(reload=True)
        self.log('已清除全局 ROI')

    def _collect_train_settings(self) -> dict:
        if self.use_pretrained_radio.isChecked():
            init_mode = 'pretrained'
        elif self.from_scratch_radio.isChecked():
            init_mode = 'scratch'
        else:
            init_mode = 'custom'
        return {
            'model': self.train_model_combo.currentText(),
            'epochs': self.train_epochs_spin.value(),
            'batch_size': self.train_batch_spin.value(),
            'img_size': self.train_imgsz_spin.value(),
            'learning_rate': self.train_lr_spin.value(),
            'init_mode': init_mode,
            'custom_weights': self.train_weights_edit.text().strip(),
            'fine_tuning': self.fine_tuning_check.isChecked(),
            'detect_conf': self.detect_conf_spin.value(),
            'log_enlarge': self.log_enlarge_check.isChecked(),
            'log_font_size': self._log_font_size,
            'draw_tool': 'polygon' if self.tool_polygon_radio.isChecked() else 'bbox',
        }

    def _apply_train_settings(self, settings: dict):
        if not settings:
            return
        self._loading_settings = True
        try:
            model = settings.get('model')
            if model:
                idx = self.train_model_combo.findText(model)
                if idx >= 0:
                    self.train_model_combo.setCurrentIndex(idx)
            if 'epochs' in settings:
                self.train_epochs_spin.setValue(int(settings['epochs']))
            if 'batch_size' in settings:
                self.train_batch_spin.setValue(int(settings['batch_size']))
            if 'img_size' in settings:
                self.train_imgsz_spin.setValue(int(settings['img_size']))
            if 'learning_rate' in settings:
                self.train_lr_spin.setValue(float(settings['learning_rate']))
            if 'detect_conf' in settings:
                self.detect_conf_spin.setValue(float(settings['detect_conf']))

            init_mode = settings.get('init_mode', 'pretrained')
            if init_mode == 'scratch':
                self.from_scratch_radio.setChecked(True)
            elif init_mode == 'custom':
                self.custom_weights_radio.setChecked(True)
            else:
                self.use_pretrained_radio.setChecked(True)

            weights = settings.get('custom_weights', '')
            if weights:
                self.train_weights_edit.setText(weights)
            self.fine_tuning_check.setChecked(bool(settings.get('fine_tuning', False)))
            self._on_train_init_changed()

            # 默认最小字体；仅当用户曾调大过才恢复
            if 'log_font_size' in settings:
                self._log_font_size = max(8, min(28, int(settings['log_font_size'])))
            else:
                self._log_font_size = 8
            font = self.log_text.font()
            font.setPointSize(self._log_font_size)
            self.log_text.setFont(font)
            if 'log_enlarge' in settings:
                self.log_enlarge_check.setChecked(bool(settings['log_enlarge']))
            else:
                self.log_enlarge_check.setChecked(True)
            self._toggle_log_enlarge(self.log_enlarge_check.isChecked())
            draw_tool = settings.get('draw_tool', 'bbox')
            if draw_tool == 'polygon':
                self.tool_polygon_radio.setChecked(True)
            else:
                self.tool_bbox_radio.setChecked(True)
            self.canvas.set_draw_tool(draw_tool)
        finally:
            self._loading_settings = False

    def _schedule_settings_save(self, *_args):
        if self._loading_settings or not self.manager.project_dir:
            return
        self._settings_save_timer.start(400)

    def _persist_train_settings(self):
        if not self.manager.project_dir:
            return
        self.manager.save_train_settings(self._collect_train_settings())

    def has_unsaved_changes(self) -> bool:
        return bool(self._dirty)

    def update_settings(self, settings):
        self._settings = dict(settings or {})
        self._refresh_roi_ui()
        # 仅在尚未从项目加载时用全局默认值填充
        if self.manager.project_dir and self.manager.train_settings:
            return
        model = self._settings.get('default_model')
        if model:
            idx = self.train_model_combo.findText(model)
            if idx >= 0:
                self.train_model_combo.setCurrentIndex(idx)
        if 'default_batch_size' in self._settings:
            self.train_batch_spin.setValue(int(self._settings['default_batch_size']))
        if 'default_img_size' in self._settings:
            self.train_imgsz_spin.setValue(int(self._settings['default_img_size']))
        if 'default_conf_thresh' in self._settings:
            self.detect_conf_spin.setValue(float(self._settings['default_conf_thresh']))
        if 'default_learning_rate' in self._settings:
            self.train_lr_spin.setValue(float(self._settings['default_learning_rate']))

    def open_project(self):
        folder = QFileDialog.getExistingDirectory(self, '选择或创建项目目录')
        if not folder:
            return
        self.open_project_path(folder)

    def open_project_path(self, folder: str, remember: bool = True) -> bool:
        """打开指定项目目录；成功返回 True。"""
        if not folder or not os.path.isdir(folder):
            return False

        images_dir = os.path.join(folder, 'images')
        if os.path.exists(images_dir):
            self.manager.open_folder(folder)
        else:
            # 自动恢复时：仅打开已有项目，不弹创建确认
            project_file = os.path.join(folder, 'project.json')
            if os.path.isfile(project_file) or os.path.isdir(os.path.join(folder, 'labels')):
                self.manager.open_folder(folder)
            elif remember:
                reply = QMessageBox.question(
                    self, '创建新项目',
                    f'将在以下目录创建标注项目:\n{folder}\n\n默认缺陷类别: {", ".join(self.DEFAULT_DEFECT_CLASSES)}',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if reply != QMessageBox.Yes:
                    return False
                self.manager.init_project(folder, self.DEFAULT_DEFECT_CLASSES)
            else:
                return False

        self.project_label.setText(folder)
        self.project_label.setStyleSheet('')
        self._apply_train_settings(self.manager.train_settings)
        self._persist_train_settings()
        self._detection_results = {}
        self._best_weights = self._find_latest_model(folder)
        self.refresh_class_ui()
        self.refresh_image_list()
        self.log(f'已打开项目: {folder}')
        if self.manager.train_settings:
            self.log('已加载项目保存的训练参数')
        if self._best_weights:
            self.log(f'已定位最近模型: {self._best_weights}')
        if remember:
            self._remember_last_project(folder)
        return True

    def _find_latest_model(self, project_dir: str = None) -> str:
        """在项目 models/ 下查找最新的 best*.pt。"""
        root = project_dir or self.manager.project_dir
        if not root:
            return ''
        models_root = os.path.join(root, 'models')
        if not os.path.isdir(models_root):
            return ''
        newest = ''
        newest_mtime = -1.0
        for dirpath, _, filenames in os.walk(models_root):
            for name in filenames:
                lower = name.lower()
                if lower == 'best.pt' or (lower.startswith('best_') and lower.endswith('.pt')):
                    path = os.path.join(dirpath, name)
                    try:
                        mtime = os.path.getmtime(path)
                    except OSError:
                        continue
                    if mtime > newest_mtime:
                        newest_mtime = mtime
                        newest = path
        return newest

    def _remember_last_project(self, folder: str):
        self._settings['last_annotation_project'] = folder
        settings_tab = self._find_settings_tab()
        if settings_tab is None:
            return
        settings_tab.settings['last_annotation_project'] = folder
        try:
            settings_tab.save_settings(show_message=False)
        except Exception:
            pass

    def _find_settings_tab(self):
        parent = self.parentWidget()
        while parent is not None:
            if hasattr(parent, 'settings_tab'):
                return parent.settings_tab
            parent = parent.parentWidget()
        return None

    def restore_last_project(self):
        """启动时自动打开上次项目。"""
        folder = (self._settings or {}).get('last_annotation_project', '')
        if not folder:
            settings_tab = self._find_settings_tab()
            if settings_tab:
                folder = settings_tab.settings.get('last_annotation_project', '')
        if folder and os.path.isdir(folder):
            ok = self.open_project_path(folder, remember=True)
            if ok:
                self.log(f'已自动打开上次项目: {folder}')
            return ok
        return False

    def import_images(self):
        if not self.manager.project_dir:
            QMessageBox.warning(self, '提示', '请先打开或创建项目')
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, '选择图像文件', '',
            '图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)'
        )
        if not files:
            return
        count = self.manager.import_images(files)
        self.refresh_image_list()
        self.log(f'已导入 {count} 张图像')
        if self.manager.current_index < 0 and self.manager.image_paths:
            self.load_image_at(0)

    def refresh_class_ui(self):
        self.class_combo.blockSignals(True)
        self.class_list.clear()
        self.class_combo.clear()
        for i, name in enumerate(self.manager.classes):
            color = self.manager.class_color(i)
            item = QListWidgetItem(f'{i}: {name}')
            item.setForeground(QColor(color))
            self.class_list.addItem(item)
            self.class_combo.addItem(f'{i}: {name}', i)
        self.class_combo.blockSignals(False)
        self.update_stats()

    def _image_matches_filters(self, path: str) -> bool:
        """按状态/检测/划分条件，结合与或逻辑判断是否显示。"""
        annotated = self.manager.is_annotated(path)
        has_det = bool(self._detection_results.get(path))
        split = self.manager.get_split(path) or ''

        status_idx = self.filter_status_combo.currentIndex()
        detect_idx = self.filter_detect_combo.currentIndex()
        split_idx = self.filter_split_combo.currentIndex()
        use_and = self.filter_logic_combo.currentIndex() == 0

        checks = []
        if status_idx == 1:  # 已标注
            checks.append(annotated)
        elif status_idx == 2:  # 未标注
            checks.append(not annotated)

        if detect_idx == 1:  # 有检测
            checks.append(has_det)
        elif detect_idx == 2:  # 无检测
            checks.append(not has_det)

        if split_idx == 1:
            checks.append(split == 'train')
        elif split_idx == 2:
            checks.append(split == 'val')
        elif split_idx == 3:
            checks.append(split == 'mark')
        elif split_idx == 4:
            checks.append(not split)

        if not checks:
            return True
        if use_and:
            return all(checks)
        return any(checks)

    def refresh_image_list(self):
        self.image_list.blockSignals(True)
        current_path = self.manager.current_image_path
        self.image_list.clear()
        selected_rows = []
        for i, path in enumerate(self.manager.image_paths):
            if not self._image_matches_filters(path):
                continue

            annotated = self.manager.is_annotated(path)
            has_det = bool(self._detection_results.get(path))
            split = self.manager.get_split(path)

            name = os.path.basename(path)
            split_tag = {
                'train': '[训]',
                'val': '[验]',
                'mark': '[标]',
            }.get(split, '[未]')
            prefix = '✓ ' if annotated else '○ '
            if has_det:
                prefix = '◎ ' + prefix
            item = QListWidgetItem(f'{split_tag} {prefix}{name}')
            item.setData(Qt.UserRole, i)
            if split == 'train':
                item.setForeground(QColor('#2E7D32'))
            elif split == 'val':
                item.setForeground(QColor('#1565C0'))
            elif split == 'mark':
                item.setForeground(QColor('#6A1B9A'))
            elif has_det:
                item.setForeground(QColor('#1E90FF'))
            elif annotated:
                item.setForeground(QColor('#48F90A'))
            self.image_list.addItem(item)
            if current_path and path == current_path:
                selected_rows.append(self.image_list.count() - 1)
        self.image_list.blockSignals(False)
        if selected_rows:
            self.image_list.setCurrentRow(selected_rows[0])
        self.update_index_label()
        self.update_stats()

    def update_index_label(self):
        total = len(self.manager.image_paths)
        current = self.manager.current_index + 1 if self.manager.current_index >= 0 else 0
        self.image_index_label.setText(f'{current}/{total}')

    def update_stats(self):
        stats = self.manager.get_statistics()
        det_imgs = sum(1 for v in self._detection_results.values() if v)
        det_boxes = sum(len(v) for v in self._detection_results.values())
        sc = stats.get('split_counts') or {}
        lines = [
            f"图像: {stats['annotated_images']}/{stats['total_images']} 已标注",
            f"未标注: {stats['unannotated_images']}",
            f"划分: 训{sc.get('train', 0)} / 验{sc.get('val', 0)} / "
            f"标{sc.get('mark', 0)} / 未{sc.get('unassigned', 0)}",
            f"框/多边形: {stats.get('bbox_count', 0)}/{stats.get('polygon_count', 0)}",
            f"检测: {det_imgs} 张 / {det_boxes} 框",
        ]
        for i, name in enumerate(self.manager.classes):
            count = stats['class_counts'].get(i, 0)
            lines.append(f'  {name}: {count} 个')
        self.stats_label.setText('\n'.join(lines))

    def _selected_image_paths(self):
        paths = []
        for item in self.image_list.selectedItems():
            idx = item.data(Qt.UserRole)
            if isinstance(idx, int) and 0 <= idx < len(self.manager.image_paths):
                paths.append(self.manager.image_paths[idx])
        return paths

    def _assign_selected_split(self, split: str):
        paths = self._selected_image_paths()
        if not paths and self.manager.current_image_path:
            paths = [self.manager.current_image_path]
        if not paths:
            QMessageBox.information(self, '提示', '请先在左侧列表选中图像')
            return
        n = self.manager.set_splits(paths, split)
        label = AnnotationManager.SPLIT_LABELS.get(split or '', '未分配')
        self.log(f'已将 {len(paths)} 张设为「{label}」')
        self.refresh_image_list()
        # 当前图若在选中集合内，立即更新画布角标
        cur = self.manager.current_image_path
        if cur:
            self.canvas.set_split_badge(self.manager.get_split(cur))
        if n == 0 and paths:
            pass

    def on_image_selected(self, row: int):
        if row < 0:
            return
        item = self.image_list.item(row)
        if item is None:
            return
        index = item.data(Qt.UserRole)
        self.load_image_at(index)

    def load_image_at(self, index: int):
        if self._dirty:
            self._save_current()
        if index < 0 or index >= len(self.manager.image_paths):
            return
        self.manager.current_index = index
        path = self.manager.image_paths[index]
        boxes = self.manager.load_annotations(path)
        detections = self._detection_results.get(path, [])
        self.canvas.show_annotations = self.show_gt_check.isChecked()
        self.canvas.show_detections = self.show_det_check.isChecked()
        # 切换图像时按勾选状态裁剪显示 ROI
        view_only = bool(self._settings.get('roi_view_only', False))
        self.canvas.set_view_roi(self._current_roi_norm() if view_only else None)
        self.canvas.load_image(path, boxes, detections)
        self.canvas.set_split_badge(self.manager.get_split(path))
        self._dirty = False
        self.update_annotation_list(boxes, detections)
        self.update_index_label()
        self._sync_list_selection(index)

    def _sync_list_selection(self, index: int):
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            if item.data(Qt.UserRole) == index:
                self.image_list.blockSignals(True)
                self.image_list.setCurrentRow(row)
                self.image_list.blockSignals(False)
                break

    def prev_image(self):
        if self.manager.current_index > 0:
            self.load_image_at(self.manager.current_index - 1)

    def next_image(self):
        if self.manager.current_index < len(self.manager.image_paths) - 1:
            self.load_image_at(self.manager.current_index + 1)

    def on_boxes_changed(self):
        self._dirty = True
        self._auto_save_timer.start(500)
        boxes = self.canvas.get_boxes()
        path = self.manager.current_image_path
        detections = self._detection_results.get(path, []) if path else []
        self.update_annotation_list(boxes, detections)
        self.update_stats()

    def update_annotation_list(self, boxes, detections=None):
        self.annotation_list.blockSignals(True)
        self.annotation_list.clear()
        for i, box in enumerate(boxes):
            name = self.manager.classes[box.class_id] if box.class_id < len(self.manager.classes) else f'class{box.class_id}'
            self.annotation_list.addItem(f'标注 #{i + 1} [{name}]')
        for i, det in enumerate(detections or []):
            name = det.get('class_name', f"class{det.get('class_id', 0)}")
            conf = float(det.get('confidence', 0))
            self.annotation_list.addItem(f'检测 #{i + 1} [{name}] {conf:.2f}')
        self.annotation_list.blockSignals(False)

    def on_annotation_selected(self, row: int):
        for item in self.canvas.box_items:
            item.setSelected(False)
        if 0 <= row < len(self.canvas.box_items):
            self.canvas.box_items[row].setSelected(True)

    def _save_current(self):
        if not self.manager.current_image_path:
            # 无当前图时仍保存训练参数
            self._persist_train_settings()
            return
        boxes = self.canvas.get_boxes()
        if boxes:
            self.manager.save_annotations(self.manager.current_image_path, boxes)
        else:
            self.manager.delete_label(self.manager.current_image_path)
        self._dirty = False
        self._persist_train_settings()
        self.refresh_image_list()

    def on_class_changed(self, index: int):
        if index < 0:
            return
        class_id = self.class_combo.currentData()
        if class_id is None:
            class_id = index
        self.canvas.set_current_class(class_id)
        self.canvas.change_selected_class(class_id)

    def add_class(self):
        name, ok = QInputDialog.getText(self, '添加类别', '缺陷类别名称:')
        if ok and name:
            if self.manager.add_class(name):
                self.refresh_class_ui()
                self.log(f'已添加类别: {name}')
            else:
                QMessageBox.warning(self, '错误', '类别名称无效或已存在')

    def rename_class(self):
        row = self.class_list.currentRow()
        if row < 0:
            return
        old_name = self.manager.classes[row]
        name, ok = QInputDialog.getText(self, '重命名类别', '新名称:', text=old_name)
        if ok and name:
            if self.manager.rename_class(row, name):
                self.refresh_class_ui()
                self.log(f'类别已重命名: {old_name} → {name}')

    def remove_class(self):
        row = self.class_list.currentRow()
        if row < 0:
            return
        name = self.manager.classes[row]
        reply = QMessageBox.question(
            self, '确认删除',
            f'确定删除类别 "{name}" 吗？\n该类别所有标注将被移除。',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.manager.remove_class(row)
            self.refresh_class_ui()
            if self.manager.current_image_path:
                boxes = self.manager.load_annotations(self.manager.current_image_path)
                self.canvas.set_boxes(boxes)
            self.log(f'已删除类别: {name}')

    def export_dataset(self):
        if not self.manager.project_dir:
            QMessageBox.warning(self, '提示', '请先打开项目')
            return
        if self._dirty:
            self._save_current()
        else:
            self._persist_train_settings()

        output_dir = os.path.join(self.manager.project_dir, 'dataset')

        try:
            task = self._export_task_for_model()
            result = self.manager.export_dataset(output_dir, val_ratio=0.2, task=task)
            task_label = 'segment (多边形)' if task == 'segment' else 'detect (矩形框)'
            mode_label = '按分配' if result.get('mode') == 'assigned' else '自动划分'
            extra = ''
            if result.get('mark_skipped'):
                extra += f"\n仅标注跳过: {result['mark_skipped']} 张"
            if result.get('unassigned_count'):
                extra += f"\n未分配未纳入: {result['unassigned_count']} 张"
            msg = (
                f"导出成功!\n\n"
                f"目录: {result['output_dir']}\n"
                f"方式: {mode_label}\n"
                f"训练集: {result['train_count']} 张\n"
                f"验证集: {result['val_count']} 张\n"
                f"格式: {task_label}\n"
                f"配置: {result['yaml_path']}"
                f"{extra}\n\n"
                f"可直接在「训练」Tab 中使用此数据集。"
            )
            self.log(
                f"数据集已导出到 {result['output_dir']} "
                f"({task_label}, {mode_label}, "
                f"训{result['train_count']}/验{result['val_count']})"
            )
            QMessageBox.information(self, '导出成功', msg)
            return result
        except ValueError as e:
            QMessageBox.warning(self, '导出失败', str(e))
        except Exception as e:
            QMessageBox.critical(self, '导出失败', str(e))
        return None

    def _on_overlay_toggled(self, _checked=False):
        if self.manager.current_index >= 0:
            self.load_image_at(self.manager.current_index)

    def _set_pipeline_busy(self, busy: bool):
        self.train_detect_btn.setEnabled(not busy)
        self.redetect_btn.setEnabled(not busy)
        self.stop_pipeline_btn.setEnabled(busy)
        self.export_btn.setEnabled(not busy)

    def start_train_and_detect(self):
        if self._pipeline_active or self.is_training or self.is_detecting:
            return
        if not self.manager.project_dir:
            QMessageBox.warning(self, '提示', '请先打开项目')
            return
        if not self.manager.image_paths:
            QMessageBox.warning(self, '提示', '请先导入图像')
            return

        annotated = sum(1 for p in self.manager.image_paths if self.manager.is_annotated(p))
        if annotated < 1:
            QMessageBox.warning(self, '提示', '至少需要 1 张已标注图像才能训练')
            return

        if self._dirty:
            self._save_current()
        else:
            self._persist_train_settings()

        # 解析初始化方式
        if self.custom_weights_radio.isChecked():
            model_weights = self.train_weights_edit.text().strip()
            if not model_weights or not os.path.isfile(model_weights):
                QMessageBox.warning(self, '提示', '请先选择有效的自定义权重文件')
                return
            pretrained = False
        elif self.use_pretrained_radio.isChecked():
            model_weights = None
            pretrained = True
        else:
            model_weights = None
            pretrained = False

        fine_tuning = self.fine_tuning_check.isChecked()
        if fine_tuning and not (pretrained or model_weights):
            fine_tuning = False

        init_desc = (
            '预训练' if pretrained else
            (f'自定义权重 {os.path.basename(model_weights)}' if model_weights else '从头训练')
        )
        reply = QMessageBox.question(
            self, '确认',
            f'将导出数据集并训练 {self.train_model_combo.currentText()}\n'
            f'轮数={self.train_epochs_spin.value()}  批次={self.train_batch_spin.value()}  '
            f'尺寸={self.train_imgsz_spin.value()}  lr={self.train_lr_spin.value()}\n'
            f'初始化: {init_desc}'
            f'{" | 微调" if fine_tuning else ""}\n'
            f'完成后对全部 {len(self.manager.image_paths)} 张图检测。\n\n继续？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        dataset_dir = os.path.join(self.manager.project_dir, 'dataset')
        try:
            export_result = self.manager.export_dataset(
                dataset_dir, val_ratio=0.2,
                task=self._export_task_for_model(),
            )
        except Exception as e:
            QMessageBox.critical(self, '导出失败', str(e))
            return

        train_images = os.path.join(dataset_dir, 'images', 'train')
        val_images = os.path.join(dataset_dir, 'images', 'val')
        train_labels = os.path.join(dataset_dir, 'labels', 'train')
        val_labels = os.path.join(dataset_dir, 'labels', 'val')
        # 训练产物落到 models/<时间戳>/，不覆盖旧模型
        import time as _time
        stamp = _time.strftime('%Y%m%d-%H%M%S')
        models_dir = os.path.join(self.manager.project_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        run_out = os.path.join(models_dir, stamp)

        epochs = self.train_epochs_spin.value()
        self.clear_terminal()
        self._pipeline_active = True
        self.is_training = True
        self._set_pipeline_busy(True)
        self.pipeline_progress.setRange(0, max(1, epochs))
        self.pipeline_progress.setValue(0)
        self.pipeline_progress.setFormat('%v / %m epoch')
        self.pipeline_status.setText('正在训练...')
        self.log(
            f"开始训练: {self.train_model_combo.currentText()} | "
            f"epochs={epochs} batch={self.train_batch_spin.value()} "
            f"imgsz={self.train_imgsz_spin.value()} lr={self.train_lr_spin.value()} | "
            f"{init_desc}{' | 微调' if fine_tuning else ''} | "
            f"train={export_result['train_count']} val={export_result['val_count']} | "
            f"输出: {run_out}"
        )
        if self._settings.get('roi_enabled'):
            self.log(
                f"全局 ROI 已启用: "
                f"({float(self._settings.get('roi_x1', 0)):.4f}, "
                f"{float(self._settings.get('roi_y1', 0)):.4f}) → "
                f"({float(self._settings.get('roi_x2', 1)):.4f}, "
                f"{float(self._settings.get('roi_y2', 1)):.4f})"
            )

        self.training_worker = TrainingWorker(
            model_type=self.train_model_combo.currentText(),
            train_dir=train_images,
            val_dir=val_images if os.path.isdir(val_images) else train_images,
            output_dir=models_dir,
            project_name=stamp,
            dataset_format='YOLO',
            batch_size=self.train_batch_spin.value(),
            epochs=epochs,
            img_size=self.train_imgsz_spin.value(),
            learning_rate=self.train_lr_spin.value(),
            pretrained=pretrained,
            model_weights=model_weights,
            fine_tuning=fine_tuning,
            train_labels_dir=train_labels,
            val_labels_dir=val_labels if os.path.isdir(val_labels) else train_labels,
            use_gpu=self._settings.get('use_gpu', True),
            gpu_device=self._settings.get('gpu_device', 0),
            export_onnx=True,
            roi_enabled=bool(self._settings.get('roi_enabled', False)),
            roi_norm=(
                float(self._settings.get('roi_x1', 0.0)),
                float(self._settings.get('roi_y1', 0.0)),
                float(self._settings.get('roi_x2', 1.0)),
                float(self._settings.get('roi_y2', 1.0)),
            ),
        )
        self.training_thread = QThread()
        self.training_worker.moveToThread(self.training_thread)
        self.training_worker.progress_update.connect(self.pipeline_progress.setValue)
        self.training_worker.log_update.connect(self.log)
        self.training_worker.best_weights_ready.connect(self._on_best_weights)
        self.training_worker.training_complete.connect(self._on_train_complete)
        self.training_worker.training_stopped.connect(self._on_train_stopped)
        self.training_worker.training_error.connect(self._on_train_error)
        self.training_thread.started.connect(self.training_worker.run)
        self.training_thread.start()

    def _on_best_weights(self, path: str):
        self._best_weights = path
        self.log(f'已记录最佳权重: {path}')

    def _cleanup_training(self):
        self.is_training = False
        if self.training_thread:
            self.training_thread.quit()
            self.training_thread.wait(3000)
        self.training_thread = None
        self.training_worker = None

    def _on_train_complete(self):
        self._cleanup_training()
        if not self._pipeline_active:
            return
        if not self._best_weights or not os.path.isfile(self._best_weights):
            self._pipeline_active = False
            self._set_pipeline_busy(False)
            self.pipeline_status.setText('训练完成但未找到 best.pt')
            QMessageBox.warning(self, '训练完成', '训练结束，但未找到 best.pt，无法继续检测。')
            return
        self.pipeline_status.setText('训练完成，开始检测全部图像...')
        self._start_detect(self._best_weights)

    def _on_train_stopped(self):
        self._cleanup_training()
        self._pipeline_active = False
        self._set_pipeline_busy(False)
        if self._best_weights and os.path.isfile(self._best_weights):
            self.pipeline_status.setText(f'已停止（已保存模型: {os.path.basename(self._best_weights)}）')
            self.log(f'训练已停止，模型已保存: {self._best_weights}')
            onnx_path = os.path.splitext(self._best_weights)[0] + '.onnx'
            if os.path.isfile(onnx_path):
                self.log(f'ONNX: {onnx_path}')
        else:
            self.pipeline_status.setText('已停止')
            self.log('训练已停止')

    def _on_train_error(self, err: str):
        self._cleanup_training()
        self._pipeline_active = False
        self._set_pipeline_busy(False)
        self.pipeline_status.setText('训练失败')
        self.log(f'训练错误: {err}')
        QMessageBox.critical(self, '训练失败', err)

    def start_redetect_only(self):
        if self._pipeline_active or self.is_training or self.is_detecting:
            return
        if not self.manager.image_paths:
            QMessageBox.warning(self, '提示', '没有可检测的图像')
            return
        weights = self._best_weights
        if not weights or not os.path.isfile(weights):
            # 优先使用项目内最新模型
            weights = self._find_latest_model()
        if not weights or not os.path.isfile(weights):
            start_dir = ''
            if self.manager.project_dir:
                start_dir = os.path.join(self.manager.project_dir, 'models')
                if not os.path.isdir(start_dir):
                    start_dir = self.manager.project_dir
            weights, _ = QFileDialog.getOpenFileName(
                self, '选择检测模型', start_dir, '模型文件 (*.pt *.onnx);;所有文件 (*)'
            )
            if not weights:
                return
            self._best_weights = weights
        else:
            self._best_weights = weights
        self._pipeline_active = True
        self._set_pipeline_busy(True)
        self.pipeline_status.setText('正在检测...')
        self._start_detect(weights)

    def _start_detect(self, model_path: str):
        self.is_detecting = True
        self.pipeline_progress.setRange(0, 100)
        self.pipeline_progress.setValue(0)
        self.pipeline_progress.setFormat('%p%')
        detect_out = os.path.join(self.manager.project_dir, 'detect_results')
        self.detect_worker = DetectAllWorker(
            model_path=model_path,
            image_paths=self.manager.image_paths,
            conf_thresh=self.detect_conf_spin.value(),
            iou_thresh=float(self._settings.get('default_iou_thresh', 0.45)),
            img_size=self.train_imgsz_spin.value(),
            output_dir=detect_out,
            class_names=list(self.manager.classes),
            roi_enabled=bool(self._settings.get('roi_enabled', False)),
            roi_norm=(
                float(self._settings.get('roi_x1', 0.0)),
                float(self._settings.get('roi_y1', 0.0)),
                float(self._settings.get('roi_x2', 1.0)),
                float(self._settings.get('roi_y2', 1.0)),
            ),
        )
        self.detect_thread = QThread()
        self.detect_worker.moveToThread(self.detect_thread)
        self.detect_worker.progress_update.connect(self.pipeline_progress.setValue)
        self.detect_worker.log_update.connect(self.log)
        self.detect_worker.detection_complete.connect(self._on_detect_complete)
        self.detect_worker.detection_error.connect(self._on_detect_error)
        self.detect_thread.started.connect(self.detect_worker.run)
        self.detect_thread.start()

    def _cleanup_detect(self):
        self.is_detecting = False
        if self.detect_thread:
            self.detect_thread.quit()
            self.detect_thread.wait(3000)
        self.detect_thread = None
        self.detect_worker = None

    def _on_detect_complete(self, results: dict):
        self._cleanup_detect()
        self._pipeline_active = False
        self._set_pipeline_busy(False)
        self._detection_results = results or {}
        detected_imgs = sum(1 for v in self._detection_results.values() if v)
        total_boxes = sum(len(v) for v in self._detection_results.values())
        self.pipeline_progress.setValue(100)
        self.pipeline_status.setText(f'完成: {detected_imgs} 张有检测 / {total_boxes} 框')
        self.show_det_check.setChecked(True)
        self.filter_status_combo.setCurrentIndex(0)
        self.filter_detect_combo.setCurrentIndex(0)
        self.filter_split_combo.setCurrentIndex(0)
        self.filter_logic_combo.setCurrentIndex(0)
        self.refresh_image_list()
        if self.manager.current_index >= 0:
            self.load_image_at(self.manager.current_index)
        elif self.manager.image_paths:
            self.load_image_at(0)
        QMessageBox.information(
            self, '检测完成',
            f'训练与检测已完成。\n\n'
            f'有检测结果的图像: {detected_imgs}\n'
            f'检测框总数: {total_boxes}\n\n'
            f'虚线框为检测结果，可切换「显示检测/显示标注」。\n'
            f'结果图保存在项目 detect_results/ 目录。'
        )

    def _on_detect_error(self, err: str):
        self._cleanup_detect()
        self._pipeline_active = False
        self._set_pipeline_busy(False)
        self.pipeline_status.setText('检测失败')
        self.log(f'检测错误: {err}')
        QMessageBox.critical(self, '检测失败', err)

    def stop_pipeline(self):
        if self.is_training and self.training_worker:
            self.training_worker.stop()
            self.log('正在停止训练...')
        if self.is_detecting and self.detect_worker:
            self.detect_worker.stop()
            self.log('正在停止检测...')
        self.pipeline_status.setText('正在停止...')

    def keyPressEvent(self, event):
        if self._handle_global_shortcut(event):
            event.accept()
            return
        super().keyPressEvent(event)
