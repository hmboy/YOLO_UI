from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem,
    QGraphicsPolygonItem, QGraphicsSimpleTextItem,
    QMenu, QAction, QDialog, QVBoxLayout, QSizePolicy, QApplication, QLabel,
)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPoint, QPointF, QTimer
from PyQt5.QtGui import (
    QPixmap, QPen, QColor, QBrush, QPainter, QPainterPath, QWheelEvent, QMouseEvent,
    QKeyEvent, QFont, QPolygonF,
)

from utils.annotation_manager import BBox, Polygon, AnnotationManager


def _parse_color(hex_color: str) -> QColor:
    c = QColor(hex_color)
    if not c.isValid():
        c = QColor('#FF3838')
    return c


class BBoxItem(QGraphicsRectItem):
    """可交互的标注框图形项（视觉仅边框，整框可点选）。"""

    kind = 'bbox'

    def __init__(self, x1, y1, x2, y2, class_id, color, index):
        super().__init__(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        self.class_id = class_id
        self.box_index = index
        self._color = color
        self.setPen(QPen(Qt.NoPen))
        # 极低透明度画刷：保证整框可命中，paint 仍只画边框
        self.setBrush(QBrush(QColor(0, 0, 0, 1)))
        self.setFlags(
            QGraphicsRectItem.ItemIsSelectable |
            QGraphicsRectItem.ItemIsMovable |
            QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setCacheMode(QGraphicsRectItem.NoCache)
        self.setZValue(1)
        self.setCursor(Qt.PointingHandCursor)

    def shape(self):
        """扩大命中范围，避免只能点到细边框。"""
        path = QPainterPath()
        r = self.rect()
        m = max(10.0, min(r.width(), r.height()) * 0.08)
        path.addRect(r.adjusted(-m, -m, m, m))
        return path

    def update_color(self, color: str) -> None:
        self._color = color
        self.update()

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemSelectedChange:
            self.update()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        color = _parse_color(self._color)
        border_w = 3 if self.isSelected() else 2
        pen = QPen(color, border_w)
        if self.isSelected():
            pen.setStyle(Qt.DashLine)
            fill = QColor(color)
            fill.setAlpha(45)
            painter.setBrush(QBrush(fill))
        else:
            painter.setBrush(Qt.NoBrush)
        painter.setPen(pen)
        painter.drawRect(self.rect())


class PolygonItem(QGraphicsPolygonItem):
    """可交互的多边形标注项（YOLO-seg）。"""

    kind = 'polygon'

    def __init__(self, points, class_id, color, index):
        poly = QPolygonF([QPointF(float(x), float(y)) for x, y in points])
        super().__init__(poly)
        self.class_id = class_id
        self.box_index = index
        self._color = color
        self.setPen(QPen(Qt.NoPen))
        fill = _parse_color(color)
        fill.setAlpha(40)
        self.setBrush(QBrush(fill))
        self.setFlags(
            QGraphicsPolygonItem.ItemIsSelectable |
            QGraphicsPolygonItem.ItemIsMovable |
            QGraphicsPolygonItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setCacheMode(QGraphicsPolygonItem.NoCache)
        self.setZValue(1)
        self.setCursor(Qt.PointingHandCursor)

    def shape(self):
        path = QPainterPath()
        path.addPolygon(self.polygon())
        # 边线附近也易点选
        stroker = QPainterPath()
        try:
            from PyQt5.QtGui import QPainterPathStroker
            s = QPainterPathStroker()
            s.setWidth(14)
            stroker = s.createStroke(path)
            path = path.united(stroker)
        except Exception:
            pass
        return path

    def update_color(self, color: str) -> None:
        self._color = color
        fill = _parse_color(color)
        fill.setAlpha(40)
        self.setBrush(QBrush(fill))
        self.update()

    def itemChange(self, change, value):
        if change == QGraphicsPolygonItem.ItemSelectedChange:
            self.update()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        color = _parse_color(self._color)
        border_w = 3 if self.isSelected() else 2
        pen = QPen(color, border_w)
        if self.isSelected():
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(55 if self.isSelected() else 35)
        painter.setBrush(QBrush(fill))
        painter.drawPolygon(self.polygon())


class TempRectItem(QGraphicsRectItem):
    """拖拽过程中的临时框，仅虚线边框。"""

    def __init__(self, rect, color):
        super().__init__(rect)
        self._color = color
        self.setPen(QPen(Qt.NoPen))
        self.setBrush(QBrush(Qt.NoBrush))
        self.setCacheMode(QGraphicsRectItem.NoCache)

    def paint(self, painter, option, widget=None):
        pen = QPen(_parse_color(self._color), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self.rect())


class TempPolygonItem(QGraphicsPolygonItem):
    """绘制中的临时多边形。"""

    def __init__(self, color):
        super().__init__()
        self._color = color
        self.setZValue(3)

    def set_points(self, points, cursor=None):
        pts = [QPointF(float(x), float(y)) for x, y in points]
        if cursor is not None:
            pts.append(QPointF(float(cursor.x()), float(cursor.y())))
        self.setPolygon(QPolygonF(pts))
        self.update()

    def paint(self, painter, option, widget=None):
        color = _parse_color(self._color)
        pen = QPen(color, 2, Qt.DashLine)
        painter.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(30)
        painter.setBrush(QBrush(fill))
        painter.drawPolygon(self.polygon())
        painter.setBrush(QBrush(color))
        for p in self.polygon():
            painter.drawEllipse(p, 3, 3)


class DetectionItem(QGraphicsRectItem):
    """只读检测框：虚线边框 + 类别/置信度标签。"""

    def __init__(self, x1, y1, x2, y2, class_id, color, label_text):
        super().__init__(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        self.class_id = class_id
        self._color = color
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, False)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setZValue(2)
        self._label = QGraphicsSimpleTextItem(label_text, self)
        self._label.setBrush(QBrush(_parse_color(color)))
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self._label.setFont(font)
        self._label.setPos(self.rect().x(), max(0, self.rect().y() - 16))

    def paint(self, painter, option, widget=None):
        color = _parse_color(self._color)
        pen = QPen(color, 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self.rect())


class AnnotationCanvas(QGraphicsView):
    """图像标注画布：矩形框 / 多边形、缩放、平移、全屏。"""

    boxes_changed = pyqtSignal()
    box_selected = pyqtSignal(int)

    MODE_VIEW = 0
    MODE_DRAW = 1
    TOOL_BBOX = 'bbox'
    TOOL_POLYGON = 'polygon'

    MIN_ZOOM = 0.05
    MAX_ZOOM = 20.0
    ZOOM_STEP = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor('#2b2b2b')))
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setOptimizationFlags(QGraphicsView.DontSavePainterState)
        self.setCacheMode(QGraphicsView.CacheNone)
        self.setStyleSheet('QGraphicsView { border: none; background: #2b2b2b; }')
        self.viewport().setAutoFillBackground(False)
        self.scene.setBackgroundBrush(QBrush(Qt.NoBrush))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.manager: AnnotationManager = None
        self.pixmap_item: QGraphicsPixmapItem = None
        self.box_items: list = []
        self.detection_items: list = []
        self.image_path: str = None
        self.image_width = 0
        self.image_height = 0
        # 全图尺寸与 ROI 预览裁剪偏移（标注仍按全图坐标存盘）
        self.full_image_width = 0
        self.full_image_height = 0
        self.roi_offset_x = 0
        self.roi_offset_y = 0
        self.view_roi_norm = None  # (x1,y1,x2,y2) 归一化；None 表示显示全图
        self.show_detections = True
        self.show_annotations = True

        self.mode = self.MODE_DRAW
        self.draw_tool = self.TOOL_BBOX
        self.current_class_id = 0
        self.drawing = False
        self.start_point = None
        self.temp_rect: TempRectItem = None

        self._poly_points = []
        self._temp_poly: TempPolygonItem = None
        self._poly_drawing = False

        self._panning = False
        self._pan_last_pos = QPoint()
        self._space_pressed = False

        self._fs_dialog = None
        self._pre_fs_parent = None
        self._pre_fs_layout = None
        self._pre_fs_index = -1
        self._owner = None

        # 图像划分标记（固定在视口左上角，不随缩放变小）
        self._split_badge = QLabel(self.viewport())
        self._split_badge.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._split_badge.hide()
        self._split_badge_text = ''

    def set_owner(self, owner) -> None:
        self._owner = owner

    def set_manager(self, manager: AnnotationManager) -> None:
        self.manager = manager

    def set_split_badge(self, split: str) -> None:
        """在图像上显示划分：train/val/mark/空。"""
        split = (split or '').strip().lower()
        styles = {
            'train': ('训练集', '#1B5E20', '#C8E6C9'),
            'val': ('验证集', '#0D47A1', '#BBDEFB'),
            'mark': ('仅标注', '#4A148C', '#E1BEE7'),
        }
        if split not in styles:
            self._split_badge_text = ''
            self._split_badge.hide()
            return
        text, fg, bg = styles[split]
        self._split_badge_text = text
        self._split_badge.setText(f'  {text}  ')
        self._split_badge.setStyleSheet(
            f'QLabel {{'
            f'  color: {fg};'
            f'  background: {bg};'
            f'  border: 2px solid {fg};'
            f'  border-radius: 4px;'
            f'  padding: 4px 10px;'
            f'  font-size: 16px;'
            f'  font-weight: bold;'
            f'}}'
        )
        self._split_badge.adjustSize()
        self._split_badge.move(12, 12)
        self._split_badge.show()
        self._split_badge.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._split_badge.isVisible():
            self._split_badge.move(12, 12)
            self._split_badge.raise_()

    def set_current_class(self, class_id: int) -> None:
        self.current_class_id = class_id

    def set_mode(self, mode: int) -> None:
        self.mode = mode

    def set_draw_tool(self, tool: str) -> None:
        """切换标注工具：bbox / polygon。"""
        tool = (tool or self.TOOL_BBOX).lower()
        if tool not in (self.TOOL_BBOX, self.TOOL_POLYGON):
            tool = self.TOOL_BBOX
        if tool != self.draw_tool:
            self._cancel_polygon()
            if self.drawing and self.temp_rect:
                self.scene.removeItem(self.temp_rect)
                self.temp_rect = None
                self.drawing = False
        self.draw_tool = tool

    def current_zoom(self) -> float:
        return self.transform().m11()

    def zoom_by(self, factor: float) -> None:
        new_zoom = self.current_zoom() * factor
        if new_zoom < self.MIN_ZOOM or new_zoom > self.MAX_ZOOM:
            return
        self.scale(factor, factor)

    def zoom_in(self) -> None:
        self.zoom_by(self.ZOOM_STEP)

    def zoom_out(self) -> None:
        self.zoom_by(1 / self.ZOOM_STEP)

    def reset_view(self) -> None:
        if not self.pixmap_item:
            return
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def set_view_roi(self, roi_norm) -> None:
        """设置预览裁剪 ROI（归一化）；传 None 关闭。不自动重载图像。"""
        if roi_norm and len(roi_norm) == 4:
            x1, y1, x2, y2 = (float(v) for v in roi_norm)
            x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
            y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
            if x2 - x1 < 0.005 or y2 - y1 < 0.005:
                self.view_roi_norm = None
            else:
                self.view_roi_norm = (x1, y1, x2, y2)
        else:
            self.view_roi_norm = None

    def _roi_crop_rect(self, full_w: int, full_h: int):
        """返回 (ox, oy, ow, oh) 像素裁剪框；无 ROI 返回 None。"""
        if not self.view_roi_norm or full_w <= 0 or full_h <= 0:
            return None
        x1, y1, x2, y2 = self.view_roi_norm
        ox = int(round(x1 * full_w))
        oy = int(round(y1 * full_h))
        ox2 = int(round(x2 * full_w))
        oy2 = int(round(y2 * full_h))
        ox = max(0, min(ox, full_w - 1))
        oy = max(0, min(oy, full_h - 1))
        ox2 = max(ox + 1, min(ox2, full_w))
        oy2 = max(oy + 1, min(oy2, full_h))
        return ox, oy, ox2 - ox, oy2 - oy

    def _ann_full_to_view(self, ann):
        """全图归一化标注 → 当前视图（可能已裁剪）归一化标注；完全在 ROI 外则返回 None。"""
        fw = self.full_image_width or self.image_width
        fh = self.full_image_height or self.image_height
        vw, vh = self.image_width, self.image_height
        ox, oy = self.roi_offset_x, self.roi_offset_y
        if vw <= 0 or vh <= 0:
            return None
        if isinstance(ann, Polygon):
            pts = [(x * fw - ox, y * fh - oy) for x, y in ann.points]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if max(xs) < 0 or max(ys) < 0 or min(xs) > vw or min(ys) > vh:
                return None
            return Polygon.from_pixel_points(ann.class_id, pts, vw, vh)
        x1, y1, x2, y2 = ann.to_pixel(fw, fh)
        x1 -= ox
        y1 -= oy
        x2 -= ox
        y2 -= oy
        if x2 < 0 or y2 < 0 or x1 > vw or y1 > vh:
            return None
        x1 = max(0, min(vw, x1))
        y1 = max(0, min(vh, y1))
        x2 = max(0, min(vw, x2))
        y2 = max(0, min(vh, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        return BBox.from_pixel(ann.class_id, x1, y1, x2, y2, vw, vh)

    def _ann_view_to_full(self, ann):
        """视图坐标标注 → 全图归一化标注。"""
        fw = self.full_image_width or self.image_width
        fh = self.full_image_height or self.image_height
        vw, vh = self.image_width, self.image_height
        ox, oy = self.roi_offset_x, self.roi_offset_y
        if isinstance(ann, Polygon):
            pts = [(x * vw + ox, y * vh + oy) for x, y in ann.points]
            return Polygon.from_pixel_points(ann.class_id, pts, fw, fh)
        x1, y1, x2, y2 = ann.to_pixel(vw, vh)
        return BBox.from_pixel(
            ann.class_id, x1 + ox, y1 + oy, x2 + ox, y2 + oy, fw, fh
        )

    def load_image(self, image_path: str, boxes: list, detections: list = None) -> None:
        self.image_path = image_path
        self.scene.clear()
        self.box_items.clear()
        self.detection_items.clear()
        self.temp_rect = None
        self._temp_poly = None
        self.drawing = False
        self._poly_drawing = False
        self._poly_points = []
        self._panning = False

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            from PyQt5.QtGui import QImage
            image = QImage(image_path)
            if image.isNull():
                return
            pixmap = QPixmap.fromImage(image)

        self.full_image_width = pixmap.width()
        self.full_image_height = pixmap.height()
        crop = self._roi_crop_rect(self.full_image_width, self.full_image_height)
        if crop:
            ox, oy, ow, oh = crop
            pixmap = pixmap.copy(ox, oy, ow, oh)
            self.roi_offset_x, self.roi_offset_y = ox, oy
        else:
            self.roi_offset_x = 0
            self.roi_offset_y = 0

        self.image_width = pixmap.width()
        self.image_height = pixmap.height()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.pixmap_item.setZValue(0)
        self.scene.setSceneRect(0, 0, self.image_width, self.image_height)

        if self.show_annotations:
            for i, ann in enumerate(boxes or []):
                view_ann = self._ann_full_to_view(ann)
                if view_ann is None:
                    continue
                item = self._add_ann_item(view_ann, i)
                if item:
                    item.setZValue(1)

        if detections and self.show_detections:
            self.set_detections(detections, clear_existing=False)

        self.reset_view()
        QTimer.singleShot(0, self.reset_view)
        # 恢复划分角标（scene.clear 不影响 viewport 上的 QLabel）
        if self.manager and self.image_path:
            self.set_split_badge(self.manager.get_split(self.image_path))
        else:
            self.set_split_badge('')

    def set_detections(self, detections: list, clear_existing: bool = True) -> None:
        if clear_existing:
            for item in self.detection_items:
                self.scene.removeItem(item)
            self.detection_items.clear()

        if not self.show_detections:
            return

        for det in detections or []:
            class_id = int(det.get('class_id', 0))
            color = self.manager.class_color(class_id) if self.manager else '#00E5FF'
            name = det.get('class_name') or f'class{class_id}'
            conf = float(det.get('confidence', 0))
            label = f'{name} {conf:.2f}'
            x1 = float(det['x1']) - self.roi_offset_x
            y1 = float(det['y1']) - self.roi_offset_y
            x2 = float(det['x2']) - self.roi_offset_x
            y2 = float(det['y2']) - self.roi_offset_y
            if x2 < 0 or y2 < 0 or x1 > self.image_width or y1 > self.image_height:
                continue
            item = DetectionItem(x1, y1, x2, y2, class_id, color, label)
            self.scene.addItem(item)
            self.detection_items.append(item)

    def clear_detections(self) -> None:
        for item in self.detection_items:
            self.scene.removeItem(item)
        self.detection_items.clear()

    def _add_ann_item(self, ann, index: int):
        color = self.manager.class_color(ann.class_id) if self.manager else '#FF3838'
        if isinstance(ann, Polygon):
            pts = ann.to_pixel_points(self.image_width, self.image_height)
            item = PolygonItem(pts, ann.class_id, color, index)
        else:
            x1, y1, x2, y2 = ann.to_pixel(self.image_width, self.image_height)
            item = BBoxItem(x1, y1, x2, y2, ann.class_id, color, index)
        self.scene.addItem(item)
        self.box_items.append(item)
        return item

    def _add_box_item(self, box: BBox, index: int):
        return self._add_ann_item(box, index)

    def get_boxes(self) -> list:
        """返回当前图像全部标注（BBox 或 Polygon），坐标为全图归一化。"""
        anns = []
        for item in self.box_items:
            if getattr(item, 'kind', 'bbox') == 'polygon':
                poly = item.polygon()
                pos = item.pos()
                points = [(p.x() + pos.x(), p.y() + pos.y()) for p in poly]
                view_ann = Polygon.from_pixel_points(
                    item.class_id, points, self.image_width, self.image_height
                )
            else:
                rect = item.rect()
                pos = item.pos()
                x1 = int(rect.x() + pos.x())
                y1 = int(rect.y() + pos.y())
                x2 = int(rect.x() + rect.width() + pos.x())
                y2 = int(rect.y() + rect.height() + pos.y())
                view_ann = BBox.from_pixel(
                    item.class_id, x1, y1, x2, y2,
                    self.image_width, self.image_height
                )
            anns.append(self._ann_view_to_full(view_ann))
        return anns

    def set_boxes(self, boxes: list) -> None:
        for item in self.box_items:
            self.scene.removeItem(item)
        self.box_items.clear()
        for i, ann in enumerate(boxes):
            view_ann = self._ann_full_to_view(ann)
            if view_ann is None:
                continue
            item = self._add_ann_item(view_ann, i)
            if item:
                item.setZValue(1)
        self.boxes_changed.emit()

    def delete_selected(self) -> bool:
        selected = [item for item in self.box_items if item.isSelected()]
        if not selected:
            return False
        for item in selected:
            self.scene.removeItem(item)
            self.box_items.remove(item)
        self.boxes_changed.emit()
        return True

    def select_only(self, target) -> None:
        """只选中指定标注项。"""
        for item in self.box_items:
            item.setSelected(item is target)
        if target is not None:
            self.box_selected.emit(self.box_items.index(target) if target in self.box_items else -1)

    def _ann_item_at(self, view_pos: QPoint):
        """在点击位置附近查找标注项（优先整框命中，其次扩大搜索半径）。"""
        scene_pos = self.mapToScene(view_pos)
        for item in self.scene.items(scene_pos):
            if getattr(item, 'kind', None) in ('bbox', 'polygon') and item in self.box_items:
                return item

        # 放大搜索：按屏幕像素半径找附近标注，缩放小时也易选中
        radius = 14
        search = self.mapToScene(
            view_pos.x() - radius, view_pos.y() - radius,
            radius * 2, radius * 2,
        ).boundingRect()
        candidates = []
        for item in self.scene.items(search, Qt.IntersectsItemShape):
            if getattr(item, 'kind', None) in ('bbox', 'polygon') and item in self.box_items:
                br = item.sceneBoundingRect()
                cx, cy = br.center().x(), br.center().y()
                dist = (cx - scene_pos.x()) ** 2 + (cy - scene_pos.y()) ** 2
                candidates.append((dist, item))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        return None

    def change_selected_class(self, class_id: int) -> None:
        changed = False
        for item in self.box_items:
            if item.isSelected():
                item.class_id = class_id
                color = self.manager.class_color(class_id) if self.manager else '#FF3838'
                item.update_color(color)
                changed = True
        if changed:
            self.boxes_changed.emit()

    def _cancel_polygon(self) -> None:
        if self._temp_poly is not None:
            self.scene.removeItem(self._temp_poly)
            self._temp_poly = None
        self._poly_points = []
        self._poly_drawing = False

    def _finish_polygon(self) -> bool:
        if len(self._poly_points) < 3:
            self._cancel_polygon()
            return False
        poly = Polygon.from_pixel_points(
            self.current_class_id, list(self._poly_points),
            self.image_width, self.image_height
        )
        self._cancel_polygon()
        item = self._add_ann_item(poly, len(self.box_items))
        if item:
            item.setZValue(1)
        self.boxes_changed.emit()
        return True

    def _start_pan(self, pos: QPoint) -> None:
        self._panning = True
        self._pan_last_pos = pos
        self.setCursor(Qt.ClosedHandCursor)

    def _stop_pan(self) -> None:
        self._panning = False
        self.setCursor(Qt.ArrowCursor)

    def _pan_move(self, pos: QPoint) -> None:
        if not self._panning:
            return
        delta = pos - self._pan_last_pos
        self._pan_last_pos = pos
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    def _is_pan_button(self, event: QMouseEvent) -> bool:
        return (
            event.button() == Qt.MiddleButton
            or (event.button() == Qt.LeftButton and self._space_pressed)
        )

    def _clamp_scene_pos(self, scene_pos: QPointF) -> QPointF:
        x = max(0.0, min(float(self.image_width), scene_pos.x()))
        y = max(0.0, min(float(self.image_height), scene_pos.y()))
        return QPointF(x, y)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.pixmap_item:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self.ZOOM_STEP if delta > 0 else 1 / self.ZOOM_STEP
        self.zoom_by(factor)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._is_pan_button(event) and self.pixmap_item:
            self._start_pan(event.pos())
            event.accept()
            return

        if event.button() == Qt.RightButton and self._poly_drawing:
            self._finish_polygon()
            event.accept()
            return

        if event.button() == Qt.LeftButton and self.mode == self.MODE_DRAW and self.pixmap_item and not self._space_pressed:
            # 正在画多边形时继续加点，不切换选中
            if not self._poly_drawing:
                hit = self._ann_item_at(event.pos())
                if hit is not None:
                    # 点到已有标注：选中/拖动，而不是新建
                    modifiers = event.modifiers()
                    if modifiers & Qt.ControlModifier:
                        hit.setSelected(not hit.isSelected())
                    else:
                        self.select_only(hit)
                    super().mousePressEvent(event)
                    event.accept()
                    return

            scene_pos = self._clamp_scene_pos(self.mapToScene(event.pos()))
            if not self.scene.sceneRect().contains(scene_pos):
                super().mousePressEvent(event)
                return

            if self.draw_tool == self.TOOL_POLYGON:
                color = self.manager.class_color(self.current_class_id) if self.manager else '#FF3838'
                if not self._poly_drawing:
                    self._poly_drawing = True
                    self._poly_points = [(scene_pos.x(), scene_pos.y())]
                    self._temp_poly = TempPolygonItem(color)
                    self.scene.addItem(self._temp_poly)
                    self._temp_poly.set_points(self._poly_points)
                else:
                    self._poly_points.append((scene_pos.x(), scene_pos.y()))
                    self._temp_poly.set_points(self._poly_points)
                event.accept()
                return

            self.drawing = True
            self.start_point = scene_pos
            color = self.manager.class_color(self.current_class_id) if self.manager else '#FF3838'
            self.temp_rect = TempRectItem(QRectF(self.start_point, self.start_point), color)
            self.temp_rect.setZValue(2)
            self.scene.addItem(self.temp_rect)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            self._pan_move(event.pos())
            event.accept()
            return
        if self._poly_drawing and self._temp_poly is not None:
            scene_pos = self._clamp_scene_pos(self.mapToScene(event.pos()))
            self._temp_poly.set_points(self._poly_points, scene_pos)
            event.accept()
            return
        if self.drawing and self.temp_rect and self.start_point:
            scene_pos = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, scene_pos).normalized()
            self.temp_rect.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._stop_pan()
            event.accept()
            return

        if self.drawing and event.button() == Qt.LeftButton and self.temp_rect:
            rect = self.temp_rect.rect()
            self.scene.removeItem(self.temp_rect)
            self.temp_rect = None
            self.drawing = False

            if rect.width() > 5 and rect.height() > 5:
                x1 = int(rect.x())
                y1 = int(rect.y())
                x2 = int(rect.x() + rect.width())
                y2 = int(rect.y() + rect.height())
                box = BBox.from_pixel(
                    self.current_class_id, x1, y1, x2, y2,
                    self.image_width, self.image_height
                )
                item = self._add_box_item(box, len(self.box_items))
                item.setZValue(1)
                self.boxes_changed.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self.pixmap_item:
            if self._poly_drawing:
                if len(self._poly_points) >= 2:
                    self._poly_points = self._poly_points[:-1]
                self._finish_polygon()
                event.accept()
                return
            self.toggle_fullscreen()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:
        if self._poly_drawing:
            self._finish_polygon()
            event.accept()
            return
        menu = QMenu(self)
        zoom_in_action = QAction('放大', self)
        zoom_out_action = QAction('缩小', self)
        reset_action = QAction('还原图像', self)
        fit_action = QAction('适应窗口', self)
        fullscreen_action = QAction('全屏', self)

        zoom_in_action.triggered.connect(self.zoom_in)
        zoom_out_action.triggered.connect(self.zoom_out)
        reset_action.triggered.connect(self.reset_view)
        fit_action.triggered.connect(self.reset_view)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)

        if self._fs_dialog and self._fs_dialog.isVisible():
            fullscreen_action.setText('退出全屏')

        menu.addAction(zoom_in_action)
        menu.addAction(zoom_out_action)
        menu.addSeparator()
        menu.addAction(reset_action)
        menu.addAction(fit_action)
        menu.addSeparator()
        menu.addAction(fullscreen_action)
        menu.exec_(event.globalPos())

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            if self._poly_drawing:
                self._cancel_polygon()
                event.accept()
                return
            if self._fs_dialog and self._fs_dialog.isVisible():
                self.exit_fullscreen()
                event.accept()
                return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self._poly_drawing:
            self._finish_polygon()
            event.accept()
            return
        if event.key() == Qt.Key_Delete:
            self.delete_selected()
            event.accept()
            return
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal) and event.modifiers() & Qt.ControlModifier:
            self.zoom_in()
            event.accept()
            return
        if event.key() == Qt.Key_Minus and event.modifiers() & Qt.ControlModifier:
            self.zoom_out()
            event.accept()
            return
        if event.key() == Qt.Key_0 and event.modifiers() & Qt.ControlModifier:
            self.reset_view()
            event.accept()
            return
        if self._owner is not None and hasattr(self._owner, '_handle_global_shortcut'):
            if self._owner._handle_global_shortcut(event):
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            if not self._panning:
                self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def toggle_fullscreen(self) -> None:
        if self._fs_dialog and self._fs_dialog.isVisible():
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def enter_fullscreen(self) -> None:
        if not self.pixmap_item or (self._fs_dialog and self._fs_dialog.isVisible()):
            return

        self._fs_dialog = QDialog(None)
        self._fs_dialog.setWindowTitle('标注全屏 (双击或 Esc 退出)')
        self._fs_dialog.setWindowFlags(
            Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self._fs_dialog.setStyleSheet('QDialog { background: #000000; }')

        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.geometry()
            self._fs_dialog.setGeometry(geo)

        layout = QVBoxLayout(self._fs_dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._pre_fs_parent = self.parentWidget()
        self._pre_fs_layout = self._pre_fs_parent.layout() if self._pre_fs_parent else None
        if self._pre_fs_layout:
            self._pre_fs_index = self._pre_fs_layout.indexOf(self)
            self._pre_fs_layout.removeWidget(self)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self, 1)

        self._fs_dialog.finished.connect(self._on_fullscreen_closed)
        self._fs_dialog.showFullScreen()
        self._fs_dialog.raise_()
        self._fs_dialog.activateWindow()
        self.show()
        self.setFocus(Qt.OtherFocusReason)
        if self._owner is not None:
            self.installEventFilter(self._owner)
            self.viewport().installEventFilter(self._owner)
        QTimer.singleShot(0, self.reset_view)
        QTimer.singleShot(50, self.reset_view)

    def exit_fullscreen(self) -> None:
        if self._fs_dialog:
            self._fs_dialog.close()

    def _on_fullscreen_closed(self) -> None:
        if self._pre_fs_layout and self._pre_fs_parent:
            if self._pre_fs_index >= 0:
                self._pre_fs_layout.insertWidget(self._pre_fs_index, self)
            else:
                self._pre_fs_layout.addWidget(self)
        self._fs_dialog = None
        self._pre_fs_parent = None
        self._pre_fs_layout = None
        self._pre_fs_index = -1
        self.reset_view()
