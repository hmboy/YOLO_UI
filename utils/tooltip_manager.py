from PyQt5.QtWidgets import QToolTip
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont


class TooltipManager:
    """管理 UI 元素的工具提示。"""

    @staticmethod
    def apply_tooltips(window):
        QToolTip.setFont(QFont('Segoe UI', 11))
        if getattr(window, 'settings_tab', None):
            TooltipManager.apply_settings_tab_tooltips(window.settings_tab)
        if getattr(window, 'annotation_tab', None):
            TooltipManager.apply_annotation_tab_tooltips(window.annotation_tab)

    @staticmethod
    def apply_settings_tab_tooltips(tab):
        if not tab:
            return
        tips = {
            'use_gpu_check': '启用后训练/检测优先使用 GPU',
            'gpu_device_spin': 'CUDA 设备编号，通常为 0',
            'save_btn': '保存当前设置',
            'reset_btn': '重置所有设置为默认值',
            'theme_combo': '选择应用程序主题',
            'model_combo': '新项目默认 YOLO 模型',
            'batch_size_spin': '新项目默认训练批次',
            'img_size_spin': '新项目默认训练/检测尺寸',
            'conf_thresh_spin': '新项目默认检测置信度（与标注页同步）',
            'iou_thresh_spin': '新项目默认检测 IoU（与标注页同步）',
        }
        for attr, tip in tips.items():
            w = getattr(tab, attr, None)
            if w is not None:
                w.setToolTip(tip)

    @staticmethod
    def apply_annotation_tab_tooltips(tab):
        if not tab:
            return
        tips = {
            'open_project_btn': '打开已有项目目录，或在空目录创建新项目',
            'recent_btn': '最近打开的项目（最多 5 个）',
            'import_images_btn': '导入图像到当前项目（可挂到集合）',
            'export_btn': (
                '导出标准 YOLO 数据集（images/labels + data.yaml）；'
                '全集写入 dataset/，按集合导出写入 dataset_<集合名>/'
            ),
            'set_name_combo': '输入集合名后点「加入」；右键可重命名/删除集合',
            'set_assign_btn': '将选中图像加入当前集合',
            'set_clear_btn': '清除选中图像的集合归属',
            'split_train_btn': '将选中图像划为训练集',
            'split_val_btn': '将选中图像划为验证集',
            'split_mark_btn': '仅标注，不参与训练导出',
            'split_clear_btn': '清除选中图像的 train/val/mark 划分',
            'class_combo': '当前画框使用的缺陷类别（快捷键 1-9）',
            'delete_box_btn': '删除画布上选中的标注框（Delete）',
            'accept_det_btn': '把检测框写入标注（追加）。左键=当前图；Shift+点击=全部有检测的图',
            'view_menu_btn': '显示/隐藏标注、检测叠加层，以及只显示 ROI',
            'train_detect_btn': '导出数据集 → 训练 → 对全部图像重新检测',
            'redetect_btn': '用最近 best.pt（或自定义权重）仅重新检测，不训练',
            'stop_pipeline_btn': '停止当前训练或检测进程',
            'detect_conf_spin': '检测置信度阈值',
            'detect_iou_spin': 'NMS IoU 阈值',
            'roi_enabled_check': '启用后训练前按归一化 ROI 裁剪全部图像并重映射标签',
            'roi_pick_btn': '在样例图上拖拽框选全局 ROI（相对比例，对所有图生效）',
            'roi_view_only_check': '勾选后切换图像时仅显示全局 ROI 区域；标注仍按全图坐标保存',
            'ignore_mask_check': '启用后训练/检测前涂黑项目级固定干扰区',
            'ignore_mask_pick_btn': '框选 Ignore Mask 多边形',
        }
        for attr, tip in tips.items():
            w = getattr(tab, attr, None)
            if w is not None:
                w.setToolTip(tip)

    @staticmethod
    def show_temporary_tooltip(widget, message, duration=3000):
        position = widget.mapToGlobal(widget.rect().topRight())
        QToolTip.showText(position, message, widget)
        QTimer.singleShot(duration, lambda: QToolTip.hideText())
