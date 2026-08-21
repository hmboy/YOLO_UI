"""全局训练 ROI 框选对话框：拖拽框选，并支持左上角/宽高手动输入。"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFormLayout,
    QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QMessageBox, QSizePolicy, QSpinBox, QWidget,
    QMenu, QAction,
)
from PyQt5.QtCore import Qt, QRectF, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QPen, QColor, QBrush, QPainter, QWheelEvent, QMouseEvent


class _RoiTempRect(QGraphicsRectItem):
    def __init__(self, rect):
        super().__init__(rect)
        self.setZValue(10)

    def paint(self, painter, option, widget=None):
        pen = QPen(QColor('#00E5FF'), 2, Qt.DashLine)
        painter.setPen(pen)
        fill = QColor('#00E5FF')
        fill.setAlpha(40)
        painter.setBrush(QBrush(fill))
        painter.drawRect(self.rect())


class RoiCanvas(QGraphicsView):
    """单框 ROI 选择画布。"""

    roi_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setAlignment(Qt.AlignCenter)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 480)

        self._pixmap_item = None
        self.image_width = 0
        self.image_height = 0
        self._drawing = False
        self._start = None
        self._temp = None
        self._roi_item = None
        # 归一化 ROI: (x1, y1, x2, y2) 或 None
        self.roi_norm = None

    def reset_view(self):
        """还原缩放，使图像填满画布（保持比例）。"""
        if self.image_width <= 0 or self.image_height <= 0:
            return
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def load_image(self, path: str) -> bool:
        pix = QPixmap(path)
        if pix.isNull():
            return False
        self.scene.clear()
        self._pixmap_item = None
        self._temp = None
        self._roi_item = None
        self._pixmap_item = QGraphicsPixmapItem(pix)
        self.scene.addItem(self._pixmap_item)
        self.image_width = pix.width()
        self.image_height = pix.height()
        self.scene.setSceneRect(QRectF(pix.rect()))
        # 布局完成后再 fit，避免打开时画布尺寸未就绪导致图像偏小
        QTimer.singleShot(0, self.reset_view)
        QTimer.singleShot(50, self.reset_view)
        if self.roi_norm:
            self._draw_roi_from_norm(self.roi_norm)
        self.roi_changed.emit()
        return True

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        reset_action = QAction('还原图像', self)
        reset_action.setToolTip('重置缩放，使图像重新填满画布')
        reset_action.triggered.connect(self.reset_view)
        menu.addAction(reset_action)
        menu.exec_(event.globalPos())

    def set_roi_norm(self, roi, emit=True):
        self.roi_norm = roi
        if self.image_width > 0 and self.image_height > 0 and roi:
            self._draw_roi_from_norm(roi)
        elif not roi and self._roi_item:
            self.scene.removeItem(self._roi_item)
            self._roi_item = None
        if emit:
            self.roi_changed.emit()

    def set_roi_pixel(self, x, y, w, h, emit=True):
        """按像素设置 ROI（左上角 + 宽高）。"""
        if self.image_width <= 0 or self.image_height <= 0:
            return
        x = max(0, min(int(x), self.image_width - 1))
        y = max(0, min(int(y), self.image_height - 1))
        w = max(1, min(int(w), self.image_width - x))
        h = max(1, min(int(h), self.image_height - y))
        x1 = x / self.image_width
        y1 = y / self.image_height
        x2 = (x + w) / self.image_width
        y2 = (y + h) / self.image_height
        self.set_roi_norm((x1, y1, x2, y2), emit=emit)

    def roi_pixel(self):
        """返回 (x, y, w, h) 像素；无 ROI 返回 None。"""
        if not self.roi_norm or self.image_width <= 0 or self.image_height <= 0:
            return None
        x1, y1, x2, y2 = self.roi_norm
        x = int(round(x1 * self.image_width))
        y = int(round(y1 * self.image_height))
        w = max(1, int(round((x2 - x1) * self.image_width)))
        h = max(1, int(round((y2 - y1) * self.image_height)))
        return x, y, w, h

    def clear_roi(self):
        self.roi_norm = None
        if self._roi_item:
            self.scene.removeItem(self._roi_item)
            self._roi_item = None
        self.roi_changed.emit()

    def _draw_roi_from_norm(self, roi):
        x1, y1, x2, y2 = roi
        w, h = self.image_width, self.image_height
        rect = QRectF(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h)
        if self._roi_item:
            self.scene.removeItem(self._roi_item)
        self._roi_item = _RoiTempRect(rect.normalized())
        self.scene.addItem(self._roi_item)

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._pixmap_item:
            scene_pos = self.mapToScene(event.pos())
            self._drawing = True
            self._start = scene_pos
            if self._temp:
                self.scene.removeItem(self._temp)
            self._temp = _RoiTempRect(QRectF(scene_pos, scene_pos))
            self.scene.addItem(self._temp)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drawing and self._temp and self._start:
            scene_pos = self.mapToScene(event.pos())
            self._temp.setRect(QRectF(self._start, scene_pos).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._drawing and event.button() == Qt.LeftButton and self._temp:
            self._drawing = False
            rect = self._temp.rect().normalized()
            self.scene.removeItem(self._temp)
            self._temp = None
            img_rect = QRectF(0, 0, self.image_width, self.image_height)
            rect = rect.intersected(img_rect)
            if rect.width() > 5 and rect.height() > 5 and self.image_width > 0:
                x1 = max(0.0, rect.left() / self.image_width)
                y1 = max(0.0, rect.top() / self.image_height)
                x2 = min(1.0, rect.right() / self.image_width)
                y2 = min(1.0, rect.bottom() / self.image_height)
                self.roi_norm = (x1, y1, x2, y2)
                self._draw_roi_from_norm(self.roi_norm)
                self.roi_changed.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class RoiPickerDialog(QDialog):
    """在样例图上框选全局训练 ROI，可查看/手动输入左上角与宽高。"""

    def __init__(self, parent=None, image_path='', roi_norm=None):
        super().__init__(parent)
        self.setWindowTitle('框选训练 ROI')
        self.resize(1732, 1751)
        self.setMinimumSize(960, 700)
        self._image_path = image_path or ''
        self._updating_spins = False

        layout = QVBoxLayout(self)
        tip = QLabel(
            '在样例图上拖拽框选 ROI，或在下方手动输入左上角与宽高（像素）。'
            '保存为相对比例，训练时对所有图像按相同比例裁剪。'
            '滚轮缩放；右键可「还原图像」。'
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        path_row = QHBoxLayout()
        self.path_label = QLabel(self._image_path or '未选择样例图')
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        browse_btn = QPushButton('选择样例图...')
        browse_btn.clicked.connect(self._browse_image)
        path_row.addWidget(self.path_label, 1)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        self.canvas = RoiCanvas()
        layout.addWidget(self.canvas, 1)

        info = QWidget()
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(0, 0, 0, 0)
        self.size_label = QLabel('图像尺寸: -')
        info_layout.addWidget(self.size_label)

        form = QFormLayout()
        spin_row = QHBoxLayout()
        self.x_spin = QSpinBox()
        self.y_spin = QSpinBox()
        self.w_spin = QSpinBox()
        self.h_spin = QSpinBox()
        for spin, name in (
            (self.x_spin, '左上 X'),
            (self.y_spin, '左上 Y'),
            (self.w_spin, '宽度'),
            (self.h_spin, '高度'),
        ):
            spin.setRange(0, 99999)
            spin.setMaximumWidth(110)
            spin.setEnabled(False)
            cell = QHBoxLayout()
            cell.addWidget(QLabel(name))
            cell.addWidget(spin)
            spin_row.addLayout(cell)
        spin_row.addStretch()
        form.addRow('位置 (像素):', spin_row)
        info_layout.addLayout(form)

        self.norm_label = QLabel('归一化: 未框选')
        self.norm_label.setStyleSheet('color: gray;')
        info_layout.addWidget(self.norm_label)
        layout.addWidget(info)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton('清除框选')
        clear_btn.clicked.connect(self._clear)
        apply_btn = QPushButton('应用输入')
        apply_btn.setToolTip('将下方手动输入的像素值应用到画布')
        apply_btn.clicked.connect(self._apply_spins)
        ok_btn = QPushButton('确定')
        ok_btn.clicked.connect(self._accept)
        cancel_btn = QPushButton('取消')
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(apply_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self.canvas.roi_changed.connect(self._sync_spins_from_canvas)

        if roi_norm and len(roi_norm) == 4:
            self.canvas.roi_norm = tuple(float(v) for v in roi_norm)

        if self._image_path:
            self._load_current()
        else:
            self._sync_spins_from_canvas()

        # 初始加载完成后再连接，避免 setRange/setValue 把恢复的 ROI 冲掉
        for spin in (self.x_spin, self.y_spin, self.w_spin, self.h_spin):
            spin.valueChanged.connect(self._on_spin_changed)

    def showEvent(self, event):
        super().showEvent(event)
        # 窗口真正显示后再填满画布
        QTimer.singleShot(0, self.canvas.reset_view)
        QTimer.singleShot(80, self.canvas.reset_view)

    def _configure_spin_ranges(self):
        w = max(1, self.canvas.image_width)
        h = max(1, self.canvas.image_height)
        self.x_spin.setRange(0, max(0, w - 1))
        self.y_spin.setRange(0, max(0, h - 1))
        self.w_spin.setRange(1, w)
        self.h_spin.setRange(1, h)
        enabled = self.canvas.image_width > 0
        for spin in (self.x_spin, self.y_spin, self.w_spin, self.h_spin):
            spin.setEnabled(enabled)
        self.size_label.setText(
            f'图像尺寸: {self.canvas.image_width} × {self.canvas.image_height}'
            if enabled else '图像尺寸: -'
        )

    def _sync_spins_from_canvas(self):
        # 必须先挡住 valueChanged：setRange 钳位时会触发，否则会把已恢复的 ROI 覆盖成初始值
        self._updating_spins = True
        try:
            for spin in (self.x_spin, self.y_spin, self.w_spin, self.h_spin):
                spin.blockSignals(True)
            self._configure_spin_ranges()
            pix = self.canvas.roi_pixel()
            if not pix:
                self.x_spin.setValue(0)
                self.y_spin.setValue(0)
                if self.canvas.image_width > 0:
                    self.w_spin.setValue(min(100, max(1, self.canvas.image_width)))
                    self.h_spin.setValue(min(100, max(1, self.canvas.image_height)))
                self.norm_label.setText('归一化: 未框选')
            else:
                x, y, w, h = pix
                self.x_spin.setValue(x)
                self.y_spin.setValue(y)
                self.w_spin.setValue(w)
                self.h_spin.setValue(h)
                roi = self.canvas.roi_norm
                if roi:
                    x1, y1, x2, y2 = roi
                    self.norm_label.setText(
                        f'归一化: 左上({x1:.4f}, {y1:.4f})  '
                        f'宽高({x2 - x1:.4f}, {y2 - y1:.4f})'
                    )
        finally:
            for spin in (self.x_spin, self.y_spin, self.w_spin, self.h_spin):
                spin.blockSignals(False)
            self._updating_spins = False

    def _on_spin_changed(self, _value=None):
        if self._updating_spins or self.canvas.image_width <= 0:
            return
        self._apply_spins()

    def _apply_spins(self):
        if self.canvas.image_width <= 0:
            return
        x = self.x_spin.value()
        y = self.y_spin.value()
        w = self.w_spin.value()
        h = self.h_spin.value()
        w = max(1, min(w, self.canvas.image_width - x))
        h = max(1, min(h, self.canvas.image_height - y))
        self._updating_spins = True
        try:
            for spin in (self.x_spin, self.y_spin, self.w_spin, self.h_spin):
                spin.blockSignals(True)
            self.w_spin.setValue(w)
            self.h_spin.setValue(h)
        finally:
            for spin in (self.x_spin, self.y_spin, self.w_spin, self.h_spin):
                spin.blockSignals(False)
            self._updating_spins = False
        # emit=False 避免再次 sync 造成循环；手动刷新标签
        self.canvas.set_roi_pixel(x, y, w, h, emit=False)
        roi = self.canvas.roi_norm
        if roi:
            x1, y1, x2, y2 = roi
            self.norm_label.setText(
                f'归一化: 左上({x1:.4f}, {y1:.4f})  '
                f'宽高({x2 - x1:.4f}, {y2 - y1:.4f})'
            )

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择样例图', self._image_path or '',
            '图像 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;所有文件 (*)'
        )
        if path:
            self._image_path = path
            self.path_label.setText(path)
            self._load_current()

    def _load_current(self):
        if not self._image_path:
            return
        # 保留已有 roi_norm，load_image 后重绘
        if not self.canvas.load_image(self._image_path):
            QMessageBox.warning(self, '错误', f'无法加载图像:\n{self._image_path}')
            return
        self._sync_spins_from_canvas()

    def _clear(self):
        self.canvas.clear_roi()
        self._sync_spins_from_canvas()

    def _accept(self):
        if self.canvas.image_width > 0 and self.canvas.roi_norm is not None:
            # 已有框选时，把可能改过的像素输入写回
            self._apply_spins()
        elif self.canvas.image_width > 0 and self.w_spin.isEnabled():
            # 仅手动输入、尚未画框
            self._apply_spins()
        if not self.canvas.roi_norm:
            QMessageBox.information(self, '提示', '请先框选或输入 ROI。')
            return
        if not self._image_path:
            QMessageBox.information(self, '提示', '请先选择样例图。')
            return
        self.accept()

    def result_roi(self):
        """返回 (roi_norm_tuple, image_path)。"""
        return self.canvas.roi_norm, self._image_path
