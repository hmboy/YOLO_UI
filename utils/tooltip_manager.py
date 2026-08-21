from PyQt5.QtWidgets import QWidget, QToolTip
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont

class TooltipManager:
    """管理UI元素的工具提示"""
    
    @staticmethod
    def apply_tooltips(window):
        """为主窗口的UI元素添加工具提示"""
        QToolTip.setFont(QFont('Segoe UI', 11))
        
        TooltipManager.apply_training_tab_tooltips(window.training_tab)
        TooltipManager.apply_testing_tab_tooltips(window.testing_tab)
        TooltipManager.apply_inference_tab_tooltips(window.inference_tab)
        TooltipManager.apply_settings_tab_tooltips(window.settings_tab)
        if hasattr(window, 'annotation_tab'):
            TooltipManager.apply_annotation_tab_tooltips(window.annotation_tab)
    
    @staticmethod
    def apply_training_tab_tooltips(tab):
        """为训练标签页添加工具提示"""
        if not tab:
            return
            
        if hasattr(tab, 'train_images_edit'):
            tab.train_images_edit.setToolTip("训练图像目录（如 dataset/images/train）")
        if hasattr(tab, 'train_labels_edit'):
            tab.train_labels_edit.setToolTip("训练标签目录（如 dataset/labels/train）")
        if hasattr(tab, 'val_images_edit'):
            tab.val_images_edit.setToolTip("验证图像目录（如 dataset/images/val）")
        if hasattr(tab, 'val_labels_edit'):
            tab.val_labels_edit.setToolTip("验证标签目录（如 dataset/labels/val）")
        if hasattr(tab, 'model_combo'):
            tab.model_combo.setToolTip("选择要训练的模型架构")
        if hasattr(tab, 'epochs_spin'):
            tab.epochs_spin.setToolTip("训练轮数。越大通常效果越好，但耗时更长")
        if hasattr(tab, 'batch_size_spin'):
            tab.batch_size_spin.setToolTip("批次大小。越大越快，但更吃显存")
        if hasattr(tab, 'img_size_spin'):
            tab.img_size_spin.setToolTip("训练图像尺寸。高分辨率小缺陷可试 1280/1920")
        if hasattr(tab, 'start_btn'):
            tab.start_btn.setToolTip("开始训练模型")
        if hasattr(tab, 'stop_btn'):
            tab.stop_btn.setToolTip("停止当前训练（无法从断点恢复）")
    
    @staticmethod
    def apply_testing_tab_tooltips(tab):
        """为测试标签页添加工具提示"""
        if not tab:
            return
            
        if hasattr(tab, 'test_images_edit'):
            tab.test_images_edit.setToolTip("测试图像目录")
        if hasattr(tab, 'test_labels_edit'):
            tab.test_labels_edit.setToolTip("测试标签目录（用于计算 mAP）")
        if hasattr(tab, 'model_path_edit'):
            tab.model_path_edit.setToolTip("训练好的模型权重文件（.pt）")
        if hasattr(tab, 'conf_thresh_spin'):
            tab.conf_thresh_spin.setToolTip("检测置信度阈值。越高误报越少，漏报可能增多")
        if hasattr(tab, 'start_btn'):
            tab.start_btn.setToolTip("开始评估模型性能")
        if hasattr(tab, 'stop_btn'):
            tab.stop_btn.setToolTip("停止当前测试")
    
    @staticmethod
    def apply_inference_tab_tooltips(tab):
        """为推理标签页添加工具提示"""
        if not tab:
            return
            
        if hasattr(tab, 'input_edit'):
            tab.input_edit.setToolTip("推理源：图像、视频或文件夹路径")
        if hasattr(tab, 'model_path_edit'):
            tab.model_path_edit.setToolTip("用于推理的模型权重文件")
        if hasattr(tab, 'start_btn'):
            tab.start_btn.setToolTip("开始推理")
        if hasattr(tab, 'stop_btn'):
            tab.stop_btn.setToolTip("停止当前推理")
    
    @staticmethod
    def apply_settings_tab_tooltips(tab):
        """为设置标签页添加工具提示"""
        if not tab:
            return
            
        if hasattr(tab, 'use_gpu_check'):
            tab.use_gpu_check.setToolTip("启用后训练/推理优先使用 GPU")
        if hasattr(tab, 'gpu_device_spin'):
            tab.gpu_device_spin.setToolTip("CUDA 设备编号，通常为 0")
        if hasattr(tab, 'output_dir_edit'):
            tab.output_dir_edit.setToolTip("默认输出目录")
        if hasattr(tab, 'save_btn'):
            tab.save_btn.setToolTip("保存当前设置<br><b>快捷键:</b> Ctrl+S")
        if hasattr(tab, 'reset_btn'):
            tab.reset_btn.setToolTip("重置所有设置为默认值")
        if hasattr(tab, 'theme_combo'):
            tab.theme_combo.setToolTip("选择应用程序主题")
    
    @staticmethod
    def apply_annotation_tab_tooltips(tab):
        """为缺陷标注标签页添加工具提示"""
        if not tab:
            return
        if hasattr(tab, 'export_btn'):
            tab.export_btn.setToolTip("导出标准 YOLO 数据集（images/labels + data.yaml）")
        if hasattr(tab, 'val_ratio_spin'):
            tab.val_ratio_spin.setToolTip("导出时验证集占比")
        if hasattr(tab, 'class_combo'):
            tab.class_combo.setToolTip("当前画框使用的缺陷类别（快捷键 1-9）")
        if hasattr(tab, 'roi_enabled_check'):
            tab.roi_enabled_check.setToolTip(
                "启用后，训练前会按归一化 ROI 裁剪全部训练/验证图像，并同步重映射标签"
            )
        if hasattr(tab, 'roi_pick_btn'):
            tab.roi_pick_btn.setToolTip(
                "在当前/样例图上拖拽框选全局 ROI（相对比例，对所有图生效）"
            )
        if hasattr(tab, 'roi_view_only_check'):
            tab.roi_view_only_check.setToolTip(
                "勾选后切换图像时仅显示全局 ROI 区域；标注仍按全图坐标保存"
            )    
    @staticmethod
    def show_temporary_tooltip(widget, message, duration=3000):
        """显示临时的工具提示"""
        position = widget.mapToGlobal(widget.rect().topRight())
        QToolTip.showText(position, message, widget)
        QTimer.singleShot(duration, lambda: QToolTip.hideText())
