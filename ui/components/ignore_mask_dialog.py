"""项目级 ignore mask 多边形框选对话框。"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem,
    QMessageBox, QSizePolicy, QMenu, QAction,
)
from PyQt5.QtCore import Qt, QRectF, QPointF, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QPixmap, QPen, QColor, QBrush, QPainter, QWheelEvent, QMouseEvent, QPolygonF,
    QPalette,
)

from utils.ignore_mask import normalize_polygons


class _MaskPolyItem(QGraphicsPolygonItem):
    def __init__(self, polygon: QPolygonF):
        super().__init__(polygon)
        self.setZValue(10)
        pen = QPen(QColor('#FF6D00'), 2, Qt.DashLine)
        self.setPen(pen)
        fill = QColor('#FF6D00')
        fill.setAlpha(70)
        self.setBrush(QBrush(fill))


class IgnoreMaskCanvas(QGraphicsView):
    """左键加点，双击/回车闭合；右键撤销点或菜单。"""

    mask_changed = pyqtSignal()

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
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._pixmap_item = None
        self.image_width = 0
        self.image_height = 0
        self.polygons = []  # list[list[(nx,ny)]]
        self._draft = []  # current points in scene coords
        self._poly_items = []
        self._draft_item = None

    def reset_view(self):
        if self.image_width <= 0 or self.image_height <= 0:
            return
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def load_image(self, path: str) -> bool:
        pix = QPixmap(path)
        if pix.isNull():
            return False
        self.scene.clear()
        self._pixmap_item = QGraphicsPixmapItem(pix)
        self.scene.addItem(self._pixmap_item)
        self.image_width = pix.width()
        self.image_height = pix.height()
        self.scene.setSceneRect(QRectF(pix.rect()))
        self._poly_items = []
        self._draft_item = None
        self._draft = []
        self._redraw_polys()
        QTimer.singleShot(0, self.reset_view)
        QTimer.singleShot(50, self.reset_view)
        self.mask_changed.emit()
        return True

    def set_polygons(self, polygons, emit=True):
        self.polygons = normalize_polygons(polygons)
        self._draft = []
        self._redraw_polys()
        if emit:
            self.mask_changed.emit()

    def clear_mask(self):
        self.polygons = []
        self._draft = []
        self._redraw_polys()
        self.mask_changed.emit()

    def _scene_to_norm(self, pt: QPointF):
        if self.image_width <= 0 or self.image_height <= 0:
            return (0.0, 0.0)
        x = min(1.0, max(0.0, pt.x() / self.image_width))
        y = min(1.0, max(0.0, pt.y() / self.image_height))
        return (x, y)

    def _norm_to_scene(self, x, y) -> QPointF:
        return QPointF(x * self.image_width, y * self.image_height)

    def _redraw_polys(self):
        for item in self._poly_items:
            self.scene.removeItem(item)
        self._poly_items = []
        if self._draft_item:
            self.scene.removeItem(self._draft_item)
            self._draft_item = None
        for poly in self.polygons:
            qpoly = QPolygonF([self._norm_to_scene(x, y) for x, y in poly])
            item = _MaskPolyItem(qpoly)
            self.scene.addItem(item)
            self._poly_items.append(item)
        self._update_draft_item()

    def _update_draft_item(self, cursor=None):
        if self._draft_item:
            self.scene.removeItem(self._draft_item)
            self._draft_item = None
        pts = list(self._draft)
        if cursor is not None and pts:
            pts = pts + [cursor]
        if len(pts) < 2:
            return
        qpoly = QPolygonF(pts)
        item = _MaskPolyItem(qpoly)
        pen = QPen(QColor('#FFAB40'), 2, Qt.DotLine)
        item.setPen(pen)
        self.scene.addItem(item)
        self._draft_item = item

    def _finish_draft(self):
        if len(self._draft) < 3:
            return False
        poly = [self._scene_to_norm(p) for p in self._draft]
        self.polygons.append(poly)
        self._draft = []
        self._redraw_polys()
        self.mask_changed.emit()
        return True

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        if self._draft:
            undo = QAction('撤销上一点', self)
            undo.triggered.connect(self._undo_point)
            menu.addAction(undo)
            finish = QAction('闭合当前多边形', self)
            finish.triggered.connect(self._finish_draft)
            menu.addAction(finish)
        reset = QAction('还原图像缩放', self)
        reset.triggered.connect(self.reset_view)
        menu.addAction(reset)
        clear = QAction('清除全部 mask', self)
        clear.triggered.connect(self.clear_mask)
        menu.addAction(clear)
        menu.exec_(event.globalPos())

    def _undo_point(self):
        if self._draft:
            self._draft.pop()
            self._update_draft_item()

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._finish_draft()
            return
        if event.key() == Qt.Key_Escape:
            self._draft = []
            self._update_draft_item()
            return
        if event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            if self._draft:
                self._undo_point()
            elif self.polygons:
                self.polygons.pop()
                self._redraw_polys()
                self.mask_changed.emit()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.image_width > 0:
            if event.modifiers() & Qt.ControlModifier:
                self._finish_draft()
                return
            sp = self.mapToScene(event.pos())
            sp.setX(min(max(sp.x(), 0), self.image_width))
            sp.setY(min(max(sp.y(), 0), self.image_height))
            self._draft.append(sp)
            self._update_draft_item()
            return
        if event.button() == Qt.RightButton and self._draft:
            self._undo_point()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._finish_draft()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._draft:
            sp = self.mapToScene(event.pos())
            self._update_draft_item(sp)
        super().mouseMoveEvent(event)


class IgnoreMaskDialog(QDialog):
    def __init__(self, parent=None, image_path='', polygons=None):
        super().__init__(parent)
        self.setWindowTitle('框选忽略区域 (Ignore Mask)')
        self.resize(1200, 900)
        self.setMinimumSize(900, 650)
        self._image_path = image_path or ''

        layout = QVBoxLayout(self)
        tip = QLabel(
            '在样例图上左键逐点画多边形，双击或 Enter 闭合；可画多块。'
            '训练/检测前会把这些区域涂黑。滚轮缩放；Ctrl+Z 撤销。'
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

        self.canvas = IgnoreMaskCanvas()
        layout.addWidget(self.canvas, 1)

        self.info_label = QLabel('多边形: 0')
        muted = self.palette().color(QPalette.Disabled, QPalette.WindowText).name()
        self.info_label.setStyleSheet(f'color: {muted};')
        layout.addWidget(self.info_label)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton('清除全部')
        clear_btn.clicked.connect(self.canvas.clear_mask)
        ok_btn = QPushButton('确定')
        ok_btn.clicked.connect(self._accept)
        cancel_btn = QPushButton('取消')
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self.canvas.mask_changed.connect(self._sync_info)
        if polygons:
            self.canvas.polygons = normalize_polygons(polygons)
        if self._image_path:
            self._load_current()
        else:
            self._sync_info()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.canvas.reset_view)

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择样例图', self._image_path or '',
            'Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)',
        )
        if path:
            self._image_path = path
            self._load_current()

    def _load_current(self):
        self.path_label.setText(self._image_path)
        if not self.canvas.load_image(self._image_path):
            QMessageBox.warning(self, '错误', f'无法加载图像:\n{self._image_path}')
            return
        # load_image 会清空并重画；保留已设 polygons
        self.canvas._redraw_polys()
        self._sync_info()

    def _sync_info(self):
        n = len(self.canvas.polygons)
        draft = len(self.canvas._draft)
        extra = f'（绘制中 {draft} 点）' if draft else ''
        self.info_label.setText(f'多边形: {n}{extra}')

    def _accept(self):
        if self.canvas._draft and len(self.canvas._draft) >= 3:
            self.canvas._finish_draft()
        if not self._image_path:
            QMessageBox.information(self, '提示', '请先选择样例图')
            return
        if not self.canvas.polygons:
            QMessageBox.information(self, '提示', '请至少画一个多边形（≥3 点）')
            return
        self.accept()

    def result_mask(self):
        """返回 (polygons, image_path)。"""
        return normalize_polygons(self.canvas.polygons), self._image_path
