import os
import json
import glob
import random
import shutil
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union

import yaml

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

CLASS_COLORS = [
    '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231',
    '#48F90A', '#92CC17', '#3DDB86', '#1E90FF', '#001F3F',
    '#0074D9', '#7FDBFF', '#85144B', '#F012BE', '#B10DC9',
    '#111111', '#AAAAAA', '#39CCCC', '#01FF70', '#FFDC00',
]

Annotation = Union['BBox', 'Polygon']


@dataclass
class BBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    @classmethod
    def from_yolo_line(cls, line: str) -> Optional['BBox']:
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        try:
            return cls(
                class_id=int(float(parts[0])),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
        except ValueError:
            return None

    def to_pixel(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        cx = self.x_center * img_w
        cy = self.y_center * img_h
        w = self.width * img_w
        h = self.height * img_h
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return x1, y1, x2, y2

    @classmethod
    def from_pixel(cls, class_id: int, x1: int, y1: int, x2: int, y2: int,
                   img_w: int, img_h: int) -> 'BBox':
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        return cls(class_id, cx, cy, w / img_w, h / img_h)

    def to_polygon(self) -> 'Polygon':
        """将矩形框转为 4 点多边形（YOLO-seg 可用）。"""
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        return Polygon(self.class_id, [
            (max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))),
            (max(0.0, min(1.0, x2)), max(0.0, min(1.0, y1))),
            (max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))),
            (max(0.0, min(1.0, x1)), max(0.0, min(1.0, y2))),
        ])


@dataclass
class Polygon:
    """YOLO segmentation 多边形标注（归一化坐标）。"""
    class_id: int
    points: List[Tuple[float, float]] = field(default_factory=list)

    def to_yolo_line(self) -> str:
        coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in self.points)
        return f'{self.class_id} {coords}'

    @classmethod
    def from_yolo_line(cls, line: str) -> Optional['Polygon']:
        parts = line.strip().split()
        # class_id + 至少 3 个点 (6 个数) => 最少 7 个 token
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            return None
        try:
            class_id = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            if len(points) < 3:
                return None
            return cls(class_id, points)
        except ValueError:
            return None

    def to_pixel_points(self, img_w: int, img_h: int) -> List[Tuple[float, float]]:
        return [(x * img_w, y * img_h) for x, y in self.points]

    @classmethod
    def from_pixel_points(cls, class_id: int, points: List[Tuple[float, float]],
                          img_w: int, img_h: int) -> 'Polygon':
        norm = []
        for x, y in points:
            nx = max(0.0, min(1.0, float(x) / max(img_w, 1)))
            ny = max(0.0, min(1.0, float(y) / max(img_h, 1)))
            norm.append((nx, ny))
        return cls(class_id, norm)

    def to_bbox(self) -> BBox:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        return BBox(self.class_id, (x1 + x2) / 2, (y1 + y2) / 2, w, h)


def parse_yolo_annotation(line: str) -> Optional[Annotation]:
    """解析 YOLO 标签行：5 列为 bbox，>=7 列为 polygon。"""
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) == 5:
        return BBox.from_yolo_line(line)
    if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
        return Polygon.from_yolo_line(line)
    return None


def annotation_to_task_line(ann: Annotation, task: str = 'detect') -> str:
    """按任务类型写出标签行。segment 时 bbox 会转为矩形多边形。"""
    task = (task or 'detect').lower()
    if task in ('segment', 'seg'):
        if isinstance(ann, Polygon):
            return ann.to_yolo_line()
        return ann.to_polygon().to_yolo_line()
    # detect
    if isinstance(ann, BBox):
        return ann.to_yolo_line()
    return ann.to_bbox().to_yolo_line()


class AnnotationManager:
    """管理标注项目：类别、图像列表、YOLO 标签读写与数据集导出。"""

    PROJECT_FILE = 'project.json'
    CLASSES_FILE = 'classes.txt'
    SPLIT_TRAIN = 'train'
    SPLIT_VAL = 'val'
    SPLIT_MARK = 'mark'
    SPLIT_LABELS = {
        'train': '训练',
        'val': '验证',
        'mark': '仅标注',
        '': '未分配',
    }

    def __init__(self):
        self.project_dir: Optional[str] = None
        self.images_dir: Optional[str] = None
        self.labels_dir: Optional[str] = None
        self.classes: List[str] = []
        self.image_paths: List[str] = []
        self.current_index: int = -1
        self.train_settings: Dict = {}
        # basename -> 'train' | 'val' | 'mark' | ''
        self.splits: Dict[str, str] = {}

    @property
    def current_image_path(self) -> Optional[str]:
        if 0 <= self.current_index < len(self.image_paths):
            return self.image_paths[self.current_index]
        return None

    def class_color(self, class_id: int) -> str:
        return CLASS_COLORS[class_id % len(CLASS_COLORS)]

    def init_project(self, project_dir: str, default_classes: Optional[List[str]] = None) -> None:
        self.project_dir = os.path.abspath(project_dir)
        self.images_dir = os.path.join(self.project_dir, 'images')
        self.labels_dir = os.path.join(self.project_dir, 'labels')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        project_file = os.path.join(self.project_dir, self.PROJECT_FILE)
        classes_file = os.path.join(self.project_dir, self.CLASSES_FILE)

        if os.path.exists(project_file):
            with open(project_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.classes = meta.get('classes', [])
            self.train_settings = meta.get('train_settings', {}) or {}
            self.splits = dict(meta.get('splits', {}) or {})
        elif os.path.exists(classes_file):
            self.classes = self._load_classes_file(classes_file)
            self.train_settings = {}
            self.splits = {}
        else:
            self.classes = default_classes or ['缺陷']
            self.train_settings = {}
            self.splits = {}
            self._save_classes()

        self._save_project_meta()
        self.refresh_image_list()

    def _folder_has_images(self, folder: str) -> bool:
        for name in os.listdir(folder):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                return True
        return False

    def open_folder(self, folder: str) -> None:
        """打开已有图像目录，自动推断 labels 与 classes。"""
        folder = os.path.abspath(folder)
        images_sub = os.path.join(folder, 'images')

        if os.path.basename(folder) == 'images':
            self.project_dir = os.path.dirname(folder)
            self.images_dir = folder
            self.labels_dir = os.path.join(self.project_dir, 'labels')
        elif os.path.isdir(images_sub):
            self.project_dir = folder
            self.images_dir = images_sub
            self.labels_dir = os.path.join(folder, 'labels')
        elif self._folder_has_images(folder):
            self.project_dir = folder
            self.images_dir = folder
            sibling_labels = os.path.join(os.path.dirname(folder), 'labels', os.path.basename(folder))
            local_labels = os.path.join(folder, 'labels')
            if os.path.exists(local_labels):
                self.labels_dir = local_labels
            elif os.path.exists(sibling_labels):
                self.labels_dir = sibling_labels
            else:
                self.labels_dir = os.path.join(folder, 'labels')
        else:
            self.init_project(folder)
            return

        os.makedirs(self.labels_dir, exist_ok=True)
        classes_file = os.path.join(self.project_dir, self.CLASSES_FILE)
        project_file = os.path.join(self.project_dir, self.PROJECT_FILE)
        if os.path.exists(project_file):
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                self.train_settings = meta.get('train_settings', {}) or {}
                self.splits = dict(meta.get('splits', {}) or {})
                if meta.get('classes') and not self.classes:
                    self.classes = meta['classes']
            except Exception:
                self.train_settings = {}
                self.splits = {}
        else:
            self.splits = {}
        if os.path.exists(classes_file):
            self.classes = self._load_classes_file(classes_file)
        elif not self.classes:
            self.classes = self._infer_classes_from_labels() or ['缺陷']
            self._save_classes()

        self._save_project_meta()
        self.refresh_image_list()

    def _load_classes_file(self, path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def _save_classes(self) -> None:
        if not self.project_dir:
            return
        path = os.path.join(self.project_dir, self.CLASSES_FILE)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.classes))
            if self.classes:
                f.write('\n')

    def _save_project_meta(self) -> None:
        if not self.project_dir:
            return
        meta = {
            'classes': self.classes,
            'images_dir': self.images_dir,
            'labels_dir': self.labels_dir,
            'train_settings': self.train_settings or {},
            'splits': self.splits or {},
        }
        with open(os.path.join(self.project_dir, self.PROJECT_FILE), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def save_train_settings(self, settings: Dict) -> None:
        """保存训练参数到 project.json。"""
        self.train_settings = dict(settings or {})
        self._save_project_meta()

    def _split_key(self, image_path: str) -> str:
        return os.path.basename(image_path)

    def get_split(self, image_path: str) -> str:
        """返回 train / val / mark / ''(未分配)。"""
        key = self._split_key(image_path)
        val = (self.splits.get(key) or '').strip().lower()
        if val in (self.SPLIT_TRAIN, self.SPLIT_VAL, self.SPLIT_MARK):
            return val
        return ''

    def set_split(self, image_path: str, split: str) -> None:
        key = self._split_key(image_path)
        split = (split or '').strip().lower()
        if split not in (self.SPLIT_TRAIN, self.SPLIT_VAL, self.SPLIT_MARK):
            self.splits.pop(key, None)
        else:
            self.splits[key] = split
        self._save_project_meta()

    def set_splits(self, image_paths: List[str], split: str) -> int:
        """批量设置划分，返回修改数量。"""
        split = (split or '').strip().lower()
        count = 0
        for path in image_paths:
            key = self._split_key(path)
            if split not in (self.SPLIT_TRAIN, self.SPLIT_VAL, self.SPLIT_MARK):
                if key in self.splits:
                    self.splits.pop(key, None)
                    count += 1
            else:
                if self.splits.get(key) != split:
                    self.splits[key] = split
                    count += 1
        if count:
            self._save_project_meta()
        return count

    def split_counts(self) -> Dict[str, int]:
        counts = {'train': 0, 'val': 0, 'mark': 0, 'unassigned': 0}
        for path in self.image_paths:
            s = self.get_split(path)
            if s == self.SPLIT_TRAIN:
                counts['train'] += 1
            elif s == self.SPLIT_VAL:
                counts['val'] += 1
            elif s == self.SPLIT_MARK:
                counts['mark'] += 1
            else:
                counts['unassigned'] += 1
        return counts

    def refresh_image_list(self) -> None:
        if not self.images_dir or not os.path.exists(self.images_dir):
            self.image_paths = []
            return
        paths = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(self.images_dir, f'*{ext}')))
            paths.extend(glob.glob(os.path.join(self.images_dir, f'*{ext.upper()}')))
        self.image_paths = sorted(set(paths), key=lambda p: os.path.basename(p).lower())
        if self.current_index >= len(self.image_paths):
            self.current_index = len(self.image_paths) - 1

    def import_images(self, file_paths: List[str], copy: bool = True) -> int:
        if not self.images_dir:
            return 0
        count = 0
        for src in file_paths:
            ext = os.path.splitext(src)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            dst = os.path.join(self.images_dir, os.path.basename(src))
            if os.path.abspath(src) == os.path.abspath(dst):
                count += 1
                continue
            if copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)
            count += 1
        self.refresh_image_list()
        return count

    def label_path_for_image(self, image_path: str) -> str:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.labels_dir, f'{basename}.txt')

    def load_annotations(self, image_path: str) -> List[Annotation]:
        label_path = self.label_path_for_image(image_path)
        if not os.path.exists(label_path):
            return []
        anns = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                ann = parse_yolo_annotation(line)
                if ann is not None:
                    anns.append(ann)
        return anns

    def save_annotations(self, image_path: str, annotations: List[Annotation]) -> None:
        if not self.labels_dir:
            return
        os.makedirs(self.labels_dir, exist_ok=True)
        label_path = self.label_path_for_image(image_path)
        with open(label_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                f.write(ann.to_yolo_line() + '\n')

    def delete_label(self, image_path: str) -> None:
        label_path = self.label_path_for_image(image_path)
        if os.path.exists(label_path):
            os.remove(label_path)

    def is_annotated(self, image_path: str) -> bool:
        label_path = self.label_path_for_image(image_path)
        return os.path.exists(label_path) and os.path.getsize(label_path) > 0

    def get_statistics(self) -> Dict:
        total = len(self.image_paths)
        annotated = sum(1 for p in self.image_paths if self.is_annotated(p))
        class_counts = {i: 0 for i in range(len(self.classes))}
        polygon_count = 0
        bbox_count = 0
        for img_path in self.image_paths:
            for ann in self.load_annotations(img_path):
                if ann.class_id in class_counts:
                    class_counts[ann.class_id] += 1
                if isinstance(ann, Polygon):
                    polygon_count += 1
                else:
                    bbox_count += 1
        return {
            'total_images': total,
            'annotated_images': annotated,
            'unannotated_images': total - annotated,
            'class_counts': class_counts,
            'bbox_count': bbox_count,
            'polygon_count': polygon_count,
            'split_counts': self.split_counts(),
        }

    def add_class(self, name: str) -> bool:
        name = name.strip()
        if not name or name in self.classes:
            return False
        self.classes.append(name)
        self._save_classes()
        self._save_project_meta()
        return True

    def remove_class(self, index: int) -> bool:
        if index < 0 or index >= len(self.classes) or len(self.classes) <= 1:
            return False
        self.classes.pop(index)
        self._save_classes()
        self._save_project_meta()
        self._reindex_labels_after_class_removal(index)
        return True

    def rename_class(self, index: int, new_name: str) -> bool:
        new_name = new_name.strip()
        if index < 0 or index >= len(self.classes) or not new_name:
            return False
        if new_name in self.classes and self.classes.index(new_name) != index:
            return False
        self.classes[index] = new_name
        self._save_classes()
        self._save_project_meta()
        return True

    def _reindex_labels_after_class_removal(self, removed_index: int) -> None:
        for img_path in self.image_paths:
            anns = self.load_annotations(img_path)
            updated = []
            for ann in anns:
                if ann.class_id == removed_index:
                    continue
                if ann.class_id > removed_index:
                    ann.class_id -= 1
                updated.append(ann)
            if updated:
                self.save_annotations(img_path, updated)
            else:
                self.delete_label(img_path)

    def _infer_classes_from_labels(self) -> Optional[List[str]]:
        max_id = -1
        if not self.labels_dir or not os.path.exists(self.labels_dir):
            return None
        for txt in glob.glob(os.path.join(self.labels_dir, '*.txt')):
            with open(txt, 'r', encoding='utf-8') as f:
                for line in f:
                    ann = parse_yolo_annotation(line)
                    if ann:
                        max_id = max(max_id, ann.class_id)
        if max_id < 0:
            return None
        return [f'class{i}' for i in range(max_id + 1)]

    def export_dataset(self, output_dir: str, val_ratio: float = 0.2,
                       seed: int = 42, task: str = 'detect') -> Dict:
        """导出标准 YOLO 数据集结构，可直接用于训练。

        优先按用户分配的 train/val 导出；仅标注(mark)不参与。
        若尚未分配任何 train/val，则对已标注样本按 val_ratio 随机划分。
        task='segment' 时会把 bbox 转成矩形 polygon，便于 -seg 模型直接训练。
        """
        if not self.project_dir or not self.image_paths:
            raise ValueError('没有可导出的图像')

        output_dir = os.path.abspath(output_dir)
        annotated_images = [p for p in self.image_paths if self.is_annotated(p)]
        if not annotated_images:
            raise ValueError('没有已标注的图像，请先完成标注')

        train_set: List[str] = []
        val_set: List[str] = []
        mark_skipped = 0
        unassigned_used = 0
        mode = 'assigned'

        explicit_train = [p for p in annotated_images if self.get_split(p) == self.SPLIT_TRAIN]
        explicit_val = [p for p in annotated_images if self.get_split(p) == self.SPLIT_VAL]
        mark_set = [p for p in annotated_images if self.get_split(p) == self.SPLIT_MARK]
        unassigned = [
            p for p in annotated_images
            if self.get_split(p) not in (self.SPLIT_TRAIN, self.SPLIT_VAL, self.SPLIT_MARK)
        ]
        mark_skipped = len(mark_set)

        if explicit_train or explicit_val:
            train_set = list(explicit_train)
            val_set = list(explicit_val)
            if not train_set:
                raise ValueError('请至少将一张已标注图像分配为「训练」')
            if not val_set:
                # YOLO 需要验证集：复用第一张训练图
                val_set = [train_set[0]]
            if unassigned:
                unassigned_used = 0  # 不自动纳入，提醒用户
        else:
            mode = 'auto'
            pool = [p for p in annotated_images if self.get_split(p) != self.SPLIT_MARK]
            if not pool:
                raise ValueError('没有可用于训练的已标注图像（全部为「仅标注」）')
            random.seed(seed)
            shuffled = pool.copy()
            random.shuffle(shuffled)
            if len(shuffled) == 1:
                train_set = shuffled
                val_set = list(shuffled)
            else:
                val_count = max(1, int(len(shuffled) * val_ratio))
                val_set = shuffled[:val_count]
                train_set = shuffled[val_count:]
                if not train_set:
                    train_set = [val_set[0]]

        # 清空并写出
        for split, paths in [('train', train_set), ('val', list(val_set))]:
            img_out = os.path.join(output_dir, 'images', split)
            lbl_out = os.path.join(output_dir, 'labels', split)
            if os.path.isdir(img_out):
                shutil.rmtree(img_out)
            if os.path.isdir(lbl_out):
                shutil.rmtree(lbl_out)
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(lbl_out, exist_ok=True)
            for src in paths:
                fname = os.path.basename(src)
                shutil.copy2(src, os.path.join(img_out, fname))
                anns = self.load_annotations(src)
                label_name = os.path.splitext(fname)[0] + '.txt'
                with open(os.path.join(lbl_out, label_name), 'w', encoding='utf-8') as f:
                    for ann in anns:
                        f.write(annotation_to_task_line(ann, task) + '\n')

        classes_file = os.path.join(output_dir, 'classes.txt')
        with open(classes_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.classes) + '\n')

        yaml_path = os.path.join(output_dir, 'data.yaml')
        yaml_data = {
            'path': output_dir.replace('\\', '/'),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.classes),
            'names': self.classes,
        }
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        return {
            'output_dir': output_dir,
            'train_count': len(train_set),
            'val_count': len(val_set),
            'yaml_path': yaml_path,
            'task': task,
            'mode': mode,
            'mark_skipped': mark_skipped,
            'unassigned_count': len(unassigned) if (explicit_train or explicit_val) else 0,
        }
