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
    '#FF3838', '#1E90FF', '#48F90A', '#F012BE', '#FFDC00',
    '#FF701F', '#B10DC9', '#39CCCC', '#0074D9', '#FF9D97',
    '#92CC17', '#7FDBFF', '#85144B', '#CFD231', '#FFB21D',
    '#3DDB86', '#01FF70', '#AAAAAA', '#001F3F', '#111111',
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
        # basename -> set name（单集合归属；空/缺失表示未分组）
        self.image_sets: Dict[str, str] = {}
        # 项目级 ignore mask（归一化多边形列表）
        self.ignore_mask_enabled: bool = False
        self.ignore_mask_polygons: List[List[Tuple[float, float]]] = []
        self.ignore_mask_preview: str = ''
        # path -> annotations; None stats means dirty
        self._ann_cache: Dict[str, List[Annotation]] = {}
        self._stats_cache: Optional[Dict] = None

    @staticmethod
    def _normalize_set_name(name: str) -> str:
        return (name or '').strip()

    def _load_image_sets_from_meta(self, meta: Dict) -> None:
        raw = meta.get('image_sets') or {}
        if not isinstance(raw, dict):
            raw = {}
        cleaned: Dict[str, str] = {}
        for key, val in raw.items():
            k = str(key).strip()
            if not k:
                continue
            if isinstance(val, (list, tuple)):
                # 兼容未来多标签：取第一个非空
                name = ''
                for item in val:
                    name = self._normalize_set_name(str(item))
                    if name:
                        break
            else:
                name = self._normalize_set_name(str(val))
            if name:
                cleaned[k] = name
        self.image_sets = cleaned

    @property
    def current_image_path(self) -> Optional[str]:
        if 0 <= self.current_index < len(self.image_paths):
            return self.image_paths[self.current_index]
        return None

    def class_color(self, class_id: int) -> str:
        return CLASS_COLORS[class_id % len(CLASS_COLORS)]

    def _load_ignore_mask_from_meta(self, meta: Dict) -> None:
        from utils.ignore_mask import normalize_polygons
        raw = meta.get('ignore_mask') or {}
        if not isinstance(raw, dict):
            raw = {}
        self.ignore_mask_enabled = bool(raw.get('enabled', False))
        self.ignore_mask_polygons = normalize_polygons(raw.get('polygons') or [])
        self.ignore_mask_preview = (raw.get('preview_image') or '') or ''
        if not self.ignore_mask_polygons:
            self.ignore_mask_enabled = False

    def set_ignore_mask(self, enabled: bool, polygons, preview_image: str = '') -> None:
        from utils.ignore_mask import normalize_polygons
        self.ignore_mask_polygons = normalize_polygons(polygons)
        self.ignore_mask_enabled = bool(enabled) and bool(self.ignore_mask_polygons)
        self.ignore_mask_preview = preview_image or ''
        self._save_project_meta()

    def get_ignore_mask(self) -> Dict:
        return {
            'enabled': bool(self.ignore_mask_enabled and self.ignore_mask_polygons),
            'polygons': [list(map(list, poly)) for poly in self.ignore_mask_polygons],
            'preview_image': self.ignore_mask_preview or '',
        }

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
            self._load_image_sets_from_meta(meta)
            self._load_ignore_mask_from_meta(meta)
        elif os.path.exists(classes_file):
            self.classes = self._load_classes_file(classes_file)
            self.train_settings = {}
            self.splits = {}
            self.image_sets = {}
            self.ignore_mask_enabled = False
            self.ignore_mask_polygons = []
            self.ignore_mask_preview = ''
        else:
            self.classes = default_classes or ['缺陷']
            self.train_settings = {}
            self.splits = {}
            self.image_sets = {}
            self.ignore_mask_enabled = False
            self.ignore_mask_polygons = []
            self.ignore_mask_preview = ''
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
                self._load_image_sets_from_meta(meta)
                if meta.get('classes') and not self.classes:
                    self.classes = meta['classes']
                self._load_ignore_mask_from_meta(meta)
            except Exception:
                self.train_settings = {}
                self.splits = {}
                self.image_sets = {}
                self.ignore_mask_enabled = False
                self.ignore_mask_polygons = []
                self.ignore_mask_preview = ''
        else:
            self.splits = {}
            self.image_sets = {}
            self.ignore_mask_enabled = False
            self.ignore_mask_polygons = []
            self.ignore_mask_preview = ''
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
            'image_sets': self.image_sets or {},
            'ignore_mask': self.get_ignore_mask(),
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
        self._stats_cache = None

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
            self._stats_cache = None
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

    def get_set(self, image_path: str) -> str:
        """返回图像所属集合名；未分组返回空字符串。"""
        key = self._split_key(image_path)
        return self._normalize_set_name(self.image_sets.get(key, ''))

    def set_image_set(self, image_path: str, set_name: str) -> None:
        key = self._split_key(image_path)
        name = self._normalize_set_name(set_name)
        if not name:
            self.image_sets.pop(key, None)
        else:
            self.image_sets[key] = name
        self._save_project_meta()
        self._stats_cache = None

    def set_image_sets(self, image_paths: List[str], set_name: str) -> int:
        """批量设置集合（空字符串表示清除），返回修改数量。"""
        name = self._normalize_set_name(set_name)
        count = 0
        for path in image_paths:
            key = self._split_key(path)
            if not name:
                if key in self.image_sets:
                    self.image_sets.pop(key, None)
                    count += 1
            else:
                if self.image_sets.get(key) != name:
                    self.image_sets[key] = name
                    count += 1
        if count:
            self._save_project_meta()
            self._stats_cache = None
        return count

    def list_set_names(self) -> List[str]:
        """当前项目中已使用的集合名（排序）。"""
        names = {
            self._normalize_set_name(v)
            for v in self.image_sets.values()
            if self._normalize_set_name(v)
        }
        return sorted(names, key=lambda s: s.lower())

    def set_counts(self) -> Dict[str, int]:
        """各集合图像数；含 ungrouped。"""
        counts: Dict[str, int] = {'ungrouped': 0}
        for path in self.image_paths:
            name = self.get_set(path)
            if not name:
                counts['ungrouped'] += 1
            else:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def images_in_set(self, set_name: str) -> List[str]:
        """返回属于指定集合的图像路径；set_name 为空表示未分组。"""
        name = self._normalize_set_name(set_name)
        if not name:
            return [p for p in self.image_paths if not self.get_set(p)]
        return [p for p in self.image_paths if self.get_set(p) == name]

    def rename_set(self, old_name: str, new_name: str) -> int:
        """重命名集合，返回受影响图像数。"""
        old = self._normalize_set_name(old_name)
        new = self._normalize_set_name(new_name)
        if not old or not new:
            raise ValueError('集合名不能为空')
        if old == new:
            return 0
        if new in self.list_set_names() and new != old:
            raise ValueError(f'集合「{new}」已存在')
        count = 0
        for key, val in list(self.image_sets.items()):
            if self._normalize_set_name(val) == old:
                self.image_sets[key] = new
                count += 1
        if count:
            self._save_project_meta()
            self._stats_cache = None
        return count

    def delete_set(self, set_name: str) -> int:
        """删除集合：解除其下所有图像归属，返回受影响数量。"""
        name = self._normalize_set_name(set_name)
        if not name:
            return 0
        count = 0
        for key, val in list(self.image_sets.items()):
            if self._normalize_set_name(val) == name:
                self.image_sets.pop(key, None)
                count += 1
        if count:
            self._save_project_meta()
            self._stats_cache = None
        return count

    def prune_empty_sets(self) -> int:
        """清理指向已不存在图像的集合记录，返回删除条数。"""
        if not self.image_paths and not self.image_sets:
            return 0
        alive = {self._split_key(p) for p in self.image_paths}
        stale = [k for k in self.image_sets if k not in alive]
        for k in stale:
            self.image_sets.pop(k, None)
        if stale:
            self._save_project_meta()
            self._stats_cache = None
        return len(stale)

    @staticmethod
    def set_export_dirname(set_filter: Optional[str] = None) -> str:
        """全集导出用 dataset；子集用 dataset_<集合名>（文件系统安全）。"""
        if not set_filter:
            return 'dataset'
        if set_filter == '__ungrouped__':
            return 'dataset_ungrouped'
        name = AnnotationManager._normalize_set_name(str(set_filter))
        bad = '<>:"/\\|?*'
        safe = ''.join('_' if (c in bad or ord(c) < 32) else c for c in name)
        safe = safe.strip(' .') or 'set'
        if len(safe) > 80:
            safe = safe[:80].rstrip(' .')
        return f'dataset_{safe}'

    def refresh_image_list(self) -> None:
        if not self.images_dir or not os.path.exists(self.images_dir):
            self.image_paths = []
            self._ann_cache.clear()
            self._stats_cache = None
            return
        paths = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(self.images_dir, f'*{ext}')))
            paths.extend(glob.glob(os.path.join(self.images_dir, f'*{ext.upper()}')))
        self.image_paths = sorted(set(paths), key=lambda p: os.path.basename(p).lower())
        if self.current_index >= len(self.image_paths):
            self.current_index = len(self.image_paths) - 1
        alive = set(self.image_paths)
        self._ann_cache = {k: v for k, v in self._ann_cache.items() if k in alive}
        self._stats_cache = None
        self.prune_empty_sets()

    def import_images(
        self,
        file_paths: List[str],
        copy: bool = True,
        set_name: str = '',
    ) -> List[str]:
        """导入图像，返回目标路径列表；可选批量挂到集合。"""
        if not self.images_dir:
            return []
        imported: List[str] = []
        for src in file_paths:
            ext = os.path.splitext(src)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            dst = os.path.join(self.images_dir, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                if copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)
            imported.append(dst)
        self.refresh_image_list()
        name = self._normalize_set_name(set_name)
        if name and imported:
            self.set_image_sets(imported, name)
        return imported

    def delete_images(self, image_paths: List[str]) -> int:
        """永久删除图像文件及对应标签/划分/集合，返回成功数。"""
        targets = []
        seen = set()
        for path in image_paths or []:
            if not path or path in seen:
                continue
            seen.add(path)
            if os.path.isfile(path) or path in self.image_paths:
                targets.append(path)
        if not targets:
            return 0

        first_idx = -1
        for path in targets:
            try:
                i = self.image_paths.index(path)
            except ValueError:
                continue
            if first_idx < 0 or i < first_idx:
                first_idx = i

        deleted = 0
        meta_changed = False
        for path in targets:
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    continue
            label_path = self.label_path_for_image(path)
            if os.path.isfile(label_path):
                try:
                    os.remove(label_path)
                except OSError:
                    pass
            self._ann_cache.pop(path, None)
            key = self._split_key(path)
            if key in self.splits:
                self.splits.pop(key, None)
                meta_changed = True
            if key in self.image_sets:
                self.image_sets.pop(key, None)
                meta_changed = True
            deleted += 1

        if not deleted:
            return 0
        if meta_changed:
            self._save_project_meta()
        self._stats_cache = None
        self.refresh_image_list()
        if self.image_paths:
            if first_idx < 0:
                first_idx = 0
            self.current_index = min(first_idx, len(self.image_paths) - 1)
        else:
            self.current_index = -1
        return deleted

    def label_path_for_image(self, image_path: str) -> str:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.labels_dir, f'{basename}.txt')

    def load_annotations(self, image_path: str) -> List[Annotation]:
        cached = self._ann_cache.get(image_path)
        if cached is not None:
            return cached
        label_path = self.label_path_for_image(image_path)
        if not os.path.exists(label_path):
            self._ann_cache[image_path] = []
            return self._ann_cache[image_path]
        anns = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                ann = parse_yolo_annotation(line)
                if ann is not None:
                    anns.append(ann)
        self._ann_cache[image_path] = anns
        return anns

    def save_annotations(self, image_path: str, annotations: List[Annotation]) -> None:
        if not self.labels_dir:
            return
        os.makedirs(self.labels_dir, exist_ok=True)
        label_path = self.label_path_for_image(image_path)
        with open(label_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                f.write(ann.to_yolo_line() + '\n')
        self._ann_cache[image_path] = list(annotations)
        self._stats_cache = None

    def delete_label(self, image_path: str) -> None:
        label_path = self.label_path_for_image(image_path)
        if os.path.exists(label_path):
            os.remove(label_path)
        self._ann_cache[image_path] = []
        self._stats_cache = None

    def is_annotated(self, image_path: str) -> bool:
        cached = self._ann_cache.get(image_path)
        if cached is not None:
            return bool(cached)
        label_path = self.label_path_for_image(image_path)
        return os.path.isfile(label_path) and os.path.getsize(label_path) > 0

    def count_label_instances(self, image_path: str) -> int:
        return len(self.load_annotations(image_path))

    def get_statistics(self) -> Dict:
        if self._stats_cache is not None:
            return self._stats_cache
        total = len(self.image_paths)
        annotated = 0
        class_counts = {i: 0 for i in range(len(self.classes))}
        polygon_count = 0
        bbox_count = 0
        for img_path in self.image_paths:
            anns = self.load_annotations(img_path)
            if anns:
                annotated += 1
            for ann in anns:
                if ann.class_id in class_counts:
                    class_counts[ann.class_id] += 1
                if isinstance(ann, Polygon):
                    polygon_count += 1
                else:
                    bbox_count += 1
        self._stats_cache = {
            'total_images': total,
            'annotated_images': annotated,
            'unannotated_images': total - annotated,
            'class_counts': class_counts,
            'bbox_count': bbox_count,
            'polygon_count': polygon_count,
            'split_counts': self.split_counts(),
            'set_counts': self.set_counts(),
        }
        return self._stats_cache

    def add_class(self, name: str) -> bool:
        name = name.strip()
        if not name or name in self.classes:
            return False
        self.classes.append(name)
        self._save_classes()
        self._save_project_meta()
        self._stats_cache = None
        return True

    def remove_class(self, index: int) -> bool:
        if index < 0 or index >= len(self.classes) or len(self.classes) <= 1:
            return False
        self.classes.pop(index)
        self._save_classes()
        self._save_project_meta()
        self._reindex_labels_after_class_removal(index)
        self._stats_cache = None
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

    def export_dataset(
        self,
        output_dir: str,
        val_ratio: float = 0.2,
        seed: int = 42,
        task: str = 'detect',
        set_filter: Optional[str] = None,
    ) -> Dict:
        """导出标准 YOLO 数据集结构，可直接用于训练。

        优先按用户分配的 train/val 导出；仅标注(mark)不参与。
        若尚未分配任何 train/val，则对已标注样本按 val_ratio 随机划分。
        task='segment' 时会把 bbox 转成矩形 polygon，便于 -seg 模型直接训练。
        set_filter: None=全部；'__ungrouped__'=未分组；其它=指定集合名。
        """
        if not self.project_dir or not self.image_paths:
            raise ValueError('没有可导出的图像')

        output_dir = os.path.abspath(output_dir)
        pool_paths = self.image_paths
        set_label = ''
        if set_filter == '__ungrouped__':
            pool_paths = self.images_in_set('')
            set_label = '未分组'
        elif set_filter:
            name = self._normalize_set_name(set_filter)
            pool_paths = self.images_in_set(name)
            set_label = name
            if not pool_paths:
                raise ValueError(f'集合「{name}」中没有图像')

        annotated_images = [p for p in pool_paths if self.is_annotated(p)]
        if not annotated_images:
            if set_label:
                raise ValueError(f'集合「{set_label}」中没有已标注的图像')
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

        # 再过滤：标签文件存在但解析不出框的不算
        train_set = [p for p in train_set if self.load_annotations(p)]
        val_set = [p for p in val_set if self.load_annotations(p)]
        if not train_set:
            raise ValueError('没有带有效标注框的训练图像（标签可能损坏或为空）')
        if not val_set:
            val_set = [train_set[0]]

        # 清空并写出
        burn_mask = bool(self.ignore_mask_enabled and self.ignore_mask_polygons)
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
                anns = self.load_annotations(src)
                if not anns:
                    continue
                fname = os.path.basename(src)
                dst_img = os.path.join(img_out, fname)
                self._export_image_file(src, dst_img, burn_mask=burn_mask)
                # 若写出变成 png，标签仍按原 stem
                label_name = os.path.splitext(fname)[0] + '.txt'
                with open(os.path.join(lbl_out, label_name), 'w', encoding='utf-8') as f:
                    for ann in anns:
                        f.write(annotation_to_task_line(ann, task) + '\n')

        # 以实际写出的文件数为准（跳过空标签）
        def _count_split(split: str) -> Tuple[int, int]:
            img_out = os.path.join(output_dir, 'images', split)
            lbl_out = os.path.join(output_dir, 'labels', split)
            n_img = len([
                n for n in os.listdir(img_out)
                if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS
            ]) if os.path.isdir(img_out) else 0
            n_inst = 0
            if os.path.isdir(lbl_out):
                for name in os.listdir(lbl_out):
                    if not name.endswith('.txt'):
                        continue
                    with open(os.path.join(lbl_out, name), 'r', encoding='utf-8') as f:
                        n_inst += sum(1 for line in f if line.strip())
            return n_img, n_inst

        train_count, train_inst = _count_split('train')
        val_count, val_inst = _count_split('val')
        if train_count < 1 or train_inst < 1:
            raise ValueError('导出后训练集没有有效标签，请检查标注文件')
        if val_count < 1 or val_inst < 1:
            raise ValueError(
                '导出后验证集没有有效标签（YOLO 无法算指标）。\n'
                '请给验证集分配带框/多边形的图像，或清除划分后让系统自动划分。'
            )

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
            'train_count': train_count,
            'val_count': val_count,
            'train_instances': train_inst,
            'val_instances': val_inst,
            'yaml_path': yaml_path,
            'task': task,
            'mode': mode,
            'mark_skipped': mark_skipped,
            'unassigned_count': len(unassigned) if (explicit_train or explicit_val) else 0,
            'ignore_mask_applied': burn_mask,
            'set_filter': set_label or None,
        }

    def _export_image_file(self, src: str, dst: str, burn_mask: bool = False) -> None:
        """写出导出图像；启用 ignore mask 时涂黑后写入。"""
        if not burn_mask:
            shutil.copy2(src, dst)
            return
        import cv2
        from utils.ignore_mask import apply_to_bgr

        img = cv2.imread(src)
        if img is None:
            shutil.copy2(src, dst)
            return
        apply_to_bgr(img, self.ignore_mask_polygons, (0, 0, 0))
        if cv2.imwrite(dst, img):
            return
        # 部分扩展名写失败时回退 png（标签按 stem 匹配）
        stem = os.path.splitext(os.path.basename(dst))[0]
        cv2.imwrite(os.path.join(os.path.dirname(dst), stem + '.png'), img)


if __name__ == '__main__':
    import tempfile
    root = tempfile.mkdtemp()
    m = AnnotationManager()
    m.init_project(root, ['a', 'b'])
    img = os.path.join(m.images_dir, 't.jpg')
    open(img, 'wb').close()
    m.refresh_image_list()
    assert not m.is_annotated(img)
    m.save_annotations(img, [BBox(0, 0.5, 0.5, 0.2, 0.2)])
    assert m.is_annotated(img)
    s1 = m.get_statistics()
    s2 = m.get_statistics()
    assert s1 is s2 and s1['annotated_images'] == 1
    m.delete_label(img)
    assert not m.is_annotated(img)
    assert m.get_statistics()['annotated_images'] == 0
    m.set_split(img, 'train')
    m.set_image_set(img, 'batch1')
    assert m.delete_images([img]) == 1
    assert not os.path.isfile(img)
    assert img not in m.image_paths
    assert 't.jpg' not in m.splits and 't.jpg' not in m.image_sets
    print('ok')
