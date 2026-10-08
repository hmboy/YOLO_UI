"""项目级检测结果缓存：重启后恢复蓝点与虚线框。"""
import json
import os
from typing import Dict, List


CACHE_NAME = 'detections.json'


def _cache_path(project_dir: str) -> str:
    return os.path.join(project_dir, CACHE_NAME)


def _rel_key(project_dir: str, path: str) -> str:
    try:
        return os.path.relpath(path, project_dir).replace('\\', '/')
    except ValueError:
        return os.path.basename(path)


def save_detections(project_dir: str, results: Dict[str, list]) -> None:
    if not project_dir:
        return
    payload = {}
    for path, dets in (results or {}).items():
        if not path:
            continue
        payload[_rel_key(project_dir, path)] = list(dets or [])
    os.makedirs(project_dir, exist_ok=True)
    with open(_cache_path(project_dir), 'w', encoding='utf-8') as f:
        json.dump({'version': 1, 'detections': payload}, f, ensure_ascii=False)


def load_detections(project_dir: str, image_paths: List[str] = None) -> Dict[str, list]:
    """返回 abs_path -> [det, ...]。无缓存或坏文件则 {}。"""
    if not project_dir:
        return {}
    fp = _cache_path(project_dir)
    if not os.path.isfile(fp):
        return {}
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    raw = data.get('detections') if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return {}

    by_rel = {str(k).replace('\\', '/'): (v if isinstance(v, list) else []) for k, v in raw.items()}
    out = {}
    paths = image_paths or []
    if paths:
        for p in paths:
            key = _rel_key(project_dir, p)
            if key in by_rel:
                out[p] = by_rel[key]
            else:
                base = os.path.basename(p)
                for rk, dets in by_rel.items():
                    if rk.endswith('/' + base) or rk == base:
                        out[p] = dets
                        break
    else:
        for key, dets in by_rel.items():
            out[os.path.normpath(os.path.join(project_dir, key))] = dets
    return out


if __name__ == '__main__':
    import tempfile
    root = tempfile.mkdtemp()
    img = os.path.join(root, 'images', 'a.jpg')
    os.makedirs(os.path.dirname(img), exist_ok=True)
    open(img, 'w').close()
    dets = {img: [{'class_id': 0, 'confidence': 0.9, 'x1': 1, 'y1': 2, 'x2': 3, 'y2': 4}]}
    save_detections(root, dets)
    loaded = load_detections(root, [img])
    assert img in loaded and loaded[img][0]['class_id'] == 0
    print('detection_store ok')
