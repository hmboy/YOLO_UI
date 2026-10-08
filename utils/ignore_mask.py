"""项目级 ignore mask：归一化多边形，训练/检测前涂黑固定干扰区。"""
from typing import List, Sequence, Tuple

Point = Tuple[float, float]
Polygon = List[Point]


def normalize_polygons(raw) -> List[Polygon]:
    """清洗为 [[(x,y), ...], ...]，坐标钳到 0~1，至少 3 点。"""
    out: List[Polygon] = []
    if not raw:
        return out
    for poly in raw:
        pts: Polygon = []
        for p in poly or []:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                x = min(1.0, max(0.0, float(p[0])))
                y = min(1.0, max(0.0, float(p[1])))
            except (TypeError, ValueError):
                continue
            pts.append((x, y))
        if len(pts) >= 3:
            out.append(pts)
    return out


def fingerprint(polygons: Sequence[Polygon]) -> str:
    parts = []
    for poly in polygons or []:
        parts.append(';'.join(f'{x:.5f},{y:.5f}' for x, y in poly))
    return '|'.join(parts)


def apply_to_bgr(img, polygons: Sequence[Polygon], color=(0, 0, 0)):
    """在 BGR 图上填充归一化多边形，返回同一数组（原地修改）。"""
    import cv2
    import numpy as np

    polys = normalize_polygons(polygons)
    if img is None or not polys:
        return img
    h, w = img.shape[:2]
    if w < 1 or h < 1:
        return img
    for poly in polys:
        pts = np.array(
            [[int(round(x * w)), int(round(y * h))] for x, y in poly],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            cv2.fillPoly(img, [pts], color)
    return img


if __name__ == '__main__':
    polys = normalize_polygons([
        [[0, 0], [1, 0], [0.5]],  # 第三点无效 → 整块丢弃
        [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]],
    ])
    assert len(polys) == 1 and len(polys[0]) == 4
    assert fingerprint(polys)
    try:
        import numpy as np
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        apply_to_bgr(img, polys, (0, 0, 0))
        assert img[15, 15, 0] == 0 and img[90, 90, 0] == 255
    except ImportError:
        pass
    print('ok')
