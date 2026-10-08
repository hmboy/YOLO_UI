"""图像列表项：状态色点图标与文案。"""
import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap, QPainter, QIcon
from PyQt5.QtWidgets import QListWidgetItem


def status_icon(annotated: bool, has_det: bool) -> QIcon:
    """左绿/灰=标注，右蓝/浅灰=检测。"""
    pix = QPixmap(16, 16)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor('#43A047' if annotated else '#BDBDBD'))
    p.drawEllipse(0, 4, 8, 8)
    p.setBrush(QColor('#1E88E5' if has_det else '#E0E0E0'))
    p.drawEllipse(8, 4, 8, 8)
    p.end()
    return QIcon(pix)


def make_image_list_item(
    path: str,
    index: int,
    *,
    annotated: bool,
    has_det: bool,
    split: str,
    img_set: str,
    split_labels: dict = None,
) -> QListWidgetItem:
    name = os.path.basename(path)
    split_tag = {
        'train': '[训]',
        'val': '[验]',
        'mark': '[标]',
    }.get(split or '', '[未]')
    if img_set:
        short = img_set if len(img_set) <= 10 else img_set[:9] + '…'
        set_tag = f'#{short} '
    else:
        set_tag = ''
    item = QListWidgetItem(f'{split_tag} {set_tag}{name}')
    item.setIcon(status_icon(annotated, has_det))
    item.setData(Qt.UserRole, index)
    tip_parts = [
        f"标注: {'已标注' if annotated else '未标注'}",
        f"检测: {'有' if has_det else '无'}",
    ]
    if img_set:
        tip_parts.append(f'集合: {img_set}')
    if split:
        labels = split_labels or {}
        tip_parts.append(f"划分: {labels.get(split, split)}")
    item.setToolTip('\n'.join(tip_parts))
    if split == 'train':
        item.setForeground(QColor('#2E7D32'))
    elif split == 'val':
        item.setForeground(QColor('#1565C0'))
    elif split == 'mark':
        item.setForeground(QColor('#6A1B9A'))
    elif img_set:
        item.setForeground(QColor('#E65100'))
    return item
