"""标注页：训练完成后对全部图像批量检测。"""
import os
import traceback
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal


class DetectAllWorker(QObject):
    """对图像列表跑 YOLO 检测，返回结构化结果。"""

    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    detection_complete = pyqtSignal(dict)  # {image_path: [det, ...]}
    detection_error = pyqtSignal(str)

    def __init__(self, model_path: str, image_paths: List[str],
                 conf_thresh: float = 0.25, iou_thresh: float = 0.45,
                 img_size: int = 640, output_dir: str = None,
                 class_names: List[str] = None,
                 roi_enabled: bool = False, roi_norm: Optional[Tuple[float, float, float, float]] = None):
        super().__init__()
        self.model_path = model_path
        self.image_paths = list(image_paths)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.output_dir = output_dir
        self.class_names = class_names or []
        self.roi_enabled = bool(roi_enabled)
        if roi_norm and len(roi_norm) == 4:
            self.roi_norm = tuple(float(v) for v in roi_norm)
        else:
            self.roi_norm = (0.0, 0.0, 1.0, 1.0)
        self.should_stop = False

    def stop(self):
        self.should_stop = True
        self.log_update.emit('收到停止信号，正在结束检测...')

    def _roi_is_full_frame(self) -> bool:
        x1, y1, x2, y2 = self.roi_norm
        return (
            abs(x1) < 1e-6 and abs(y1) < 1e-6
            and abs(x2 - 1.0) < 1e-6 and abs(y2 - 1.0) < 1e-6
        )

    def _crop_roi(self, img):
        """按归一化 ROI 裁剪；返回 (crop, ox, oy)。无 ROI 时 ox=oy=0。"""
        if not self.roi_enabled or self._roi_is_full_frame():
            return img, 0, 0
        h, w = img.shape[:2]
        x1, y1, x2, y2 = self.roi_norm
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        ox = int(round(x1 * w))
        oy = int(round(y1 * h))
        ox2 = int(round(x2 * w))
        oy2 = int(round(y2 * h))
        ox = max(0, min(ox, w - 1))
        oy = max(0, min(oy, h - 1))
        ox2 = max(ox + 1, min(ox2, w))
        oy2 = max(oy + 1, min(oy2, h))
        return img[oy:oy2, ox:ox2].copy(), ox, oy

    def run(self):
        try:
            if not self.image_paths:
                self.detection_error.emit('没有可检测的图像')
                return
            if not os.path.isfile(self.model_path):
                self.detection_error.emit(f'模型文件不存在: {self.model_path}')
                return

            from ultralytics import YOLO
            import cv2

            self.log_update.emit(f'加载模型: {self.model_path}')
            model = YOLO(self.model_path)

            use_roi = self.roi_enabled and not self._roi_is_full_frame()
            if use_roi:
                x1, y1, x2, y2 = self.roi_norm
                self.log_update.emit(
                    f'检测使用训练 ROI 裁剪: ({x1:.4f}, {y1:.4f}) → ({x2:.4f}, {y2:.4f})'
                )

            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)

            results_map: Dict[str, list] = {}
            total = len(self.image_paths)

            for i, img_path in enumerate(self.image_paths):
                if self.should_stop:
                    self.log_update.emit('检测已停止')
                    break

                self.log_update.emit(f'检测 ({i + 1}/{total}): {os.path.basename(img_path)}')
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        self.log_update.emit(f'  跳过无法读取的图像')
                        results_map[img_path] = []
                        continue

                    source, ox, oy = self._crop_roi(img)
                    preds = model.predict(
                        source=source,
                        conf=self.conf_thresh,
                        iou=self.iou_thresh,
                        imgsz=self.img_size,
                        verbose=False,
                    )
                except Exception as e:
                    self.log_update.emit(f'  跳过失败图像: {e}')
                    results_map[img_path] = []
                    continue

                detections = []
                if preds:
                    result = preds[0]
                    names = getattr(result, 'names', None) or {}
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                        for j in range(len(xyxy)):
                            cls_id = int(clss[j])
                            conf = float(confs[j])
                            # 裁剪坐标系 → 全图像素坐标
                            x1 = float(xyxy[j][0]) + ox
                            y1 = float(xyxy[j][1]) + oy
                            x2 = float(xyxy[j][2]) + ox
                            y2 = float(xyxy[j][3]) + oy
                            label = (
                                self.class_names[cls_id]
                                if cls_id < len(self.class_names)
                                else names.get(cls_id, f'class{cls_id}')
                            )
                            detections.append({
                                'class_id': cls_id,
                                'class_name': label,
                                'confidence': conf,
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            })

                    if self.output_dir:
                        try:
                            # 在全图上画框，便于对照标注
                            vis = img.copy()
                            for det in detections:
                                p1 = (int(det['x1']), int(det['y1']))
                                p2 = (int(det['x2']), int(det['y2']))
                                cv2.rectangle(vis, p1, p2, (0, 229, 255), 2)
                                tag = f"{det['class_name']} {det['confidence']:.2f}"
                                cv2.putText(
                                    vis, tag, (p1[0], max(16, p1[1] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1, cv2.LINE_AA,
                                )
                            if use_roi:
                                h, w = img.shape[:2]
                                rx1 = int(round(self.roi_norm[0] * w))
                                ry1 = int(round(self.roi_norm[1] * h))
                                rx2 = int(round(self.roi_norm[2] * w))
                                ry2 = int(round(self.roi_norm[3] * h))
                                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1)
                            out_path = os.path.join(self.output_dir, f'result_{os.path.basename(img_path)}')
                            cv2.imwrite(out_path, vis)
                        except Exception:
                            pass

                results_map[img_path] = detections
                self.progress_update.emit(int((i + 1) / total * 100))

            if not self.should_stop:
                detected_imgs = sum(1 for v in results_map.values() if v)
                total_boxes = sum(len(v) for v in results_map.values())
                self.log_update.emit(
                    f'检测完成: {detected_imgs}/{len(results_map)} 张有目标, 共 {total_boxes} 个框'
                )
                self.detection_complete.emit(results_map)

        except Exception as e:
            self.detection_error.emit(f'{e}\n{traceback.format_exc()}')
