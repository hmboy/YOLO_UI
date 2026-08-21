import os
import sys
import time
import threading
import torch
from PyQt5.QtCore import QObject, pyqtSignal

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

class TrainingWorker(QObject):
    """Worker class to handle YOLO model training in a separate thread."""
    
    # Signal definitions
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    training_complete = pyqtSignal()
    training_stopped = pyqtSignal()
    training_error = pyqtSignal(str)
    best_weights_ready = pyqtSignal(str)  # 训练成功后发出 best.pt 路径

    def __init__(self, model_type, train_dir, val_dir, output_dir, project_name,
                 dataset_format, batch_size, epochs, img_size, learning_rate, pretrained,
                 model_weights=None, fine_tuning=False,
                 train_labels_dir=None, val_labels_dir=None,
                 use_gpu=True, gpu_device=0, export_onnx=True,
                 roi_enabled=False, roi_norm=None):
        """
        Initialize the training worker with parameters.
        
        Args:
            model_type (str): YOLO model type (e.g., 'yolov8n')
            train_dir (str): Path to training data directory
            val_dir (str): Path to validation data directory
            output_dir (str): Path to save output results
            project_name (str): Project name for output organization
            dataset_format (str): Dataset format ('COCO' or 'VOC')
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            img_size (int): Image size for training
            learning_rate (float): Learning rate
            pretrained (bool): Whether to use pretrained weights
            model_weights (str, optional): Path to custom model weights for initialization
            fine_tuning (bool): Whether to freeze backbone layers and only train detection head
            export_onnx (bool): Whether to export ONNX after training
            roi_enabled (bool): Whether to crop all images with a global ROI before training
            roi_norm (tuple|list|None): Normalized ROI (x1, y1, x2, y2) in 0-1
        """
        super().__init__()
        self.model_type = model_type
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.output_dir = output_dir
        self.project_name = project_name
        self.dataset_format = dataset_format
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.pretrained = pretrained
        self.model_weights = model_weights
        self.fine_tuning = fine_tuning
        self.train_labels_dir = train_labels_dir or ''
        self.val_labels_dir = val_labels_dir or ''
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.export_onnx = export_onnx
        self.roi_enabled = bool(roi_enabled)
        if roi_norm and len(roi_norm) == 4:
            self.roi_norm = tuple(float(v) for v in roi_norm)
        else:
            self.roi_norm = (0.0, 0.0, 1.0, 1.0)

        self._stop_event = threading.Event()
        self._trainer_ref = None  # Reference to the trainer object for direct access
        self._process_ref = None  # Reference to any training process that might be running
    
    def _check_internet_connection(self):
        """
        Check for internet connectivity by attempting to connect to known servers.
        
        Returns:
            bool: True if internet is available, False otherwise
        """
        try:
            # Try to connect to Google's DNS server (should work in most countries)
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            try:
                # Try to connect to Baidu (for users in China)
                socket.create_connection(("220.181.38.148", 80), timeout=3)
                return True
            except OSError:
                pass
        
        # Alternative method: try to resolve a known domain
        try:
            socket.gethostbyname("google.com")
            return True
        except:
            try:
                socket.gethostbyname("baidu.com")
                return True
            except:
                pass
        
        return False

    def _is_valid_weights_file(self, path: str) -> bool:
        """检查 .pt 权重文件是否有效（排除占位文本文件）。"""
        try:
            return os.path.isfile(path) and os.path.getsize(path) >= 1024
        except OSError:
            return False

    def run(self):
        """Run the training process."""
        try:
            self.log_update.emit(f"Starting training with {self.model_type}")
            self.log_update.emit(f"Dataset format: {self.dataset_format}")
            self.log_update.emit(f"Batch size: {self.batch_size}, Image size: {self.img_size}")
            self.log_update.emit(f"Learning rate: {self.learning_rate}, Epochs: {self.epochs}")
            if self.roi_enabled and not self._roi_is_full_frame():
                x1, y1, x2, y2 = self.roi_norm
                self.log_update.emit(f"Global ROI: ({x1:.4f}, {y1:.4f}) -> ({x2:.4f}, {y2:.4f})")
            
            # Check internet connectivity for model downloading
            has_internet = self._check_internet_connection()
            if not has_internet and self.pretrained and not self.model_weights:
                self.log_update.emit("警告：检测到没有互联网连接。若本地没有预训练模型文件，将自动切换到从头训练模式")
            
            # 预加载YOLO模型，这可以避免在训练时重复下载权重
            if not self.model_weights and self.pretrained:
                model_cache_dir = os.path.join(self.output_dir, "model_cache")
                os.makedirs(model_cache_dir, exist_ok=True)
                model_file = os.path.join(model_cache_dir, f"{self.model_type}.pt")
                
                # Check for model in multiple locations with priority
                model_found = False
                
                # Check locations to look for model files
                possible_locations = [
                    # 1. Current directory (highest priority)
                    f"{self.model_type}.pt",
                    # 2. Cache directory
                    model_file,
                    # 3. Common model directories
                    os.path.join("models", f"{self.model_type}.pt"),
                    os.path.join("weights", f"{self.model_type}.pt"),
                    os.path.join("pretrained", f"{self.model_type}.pt"),
                    # 4. Application directory
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"{self.model_type}.pt")
                ]
                
                # Look for model file in possible locations
                for location in possible_locations:
                    if self._is_valid_weights_file(location):
                        self.log_update.emit(f"找到本地模型权重: {location}")
                        # Copy to cache if not already there
                        if location != model_file:
                            try:
                                import shutil
                                shutil.copy(location, model_file)
                                self.log_update.emit(f"已复制模型权重到缓存目录: {model_file}")
                            except Exception as e:
                                self.log_update.emit(f"复制模型到缓存失败，将直接使用原始文件: {str(e)}")
                        self.model_weights = location
                        model_found = True
                        break
                
                # If model not found locally, prepare for download
                if not model_found:
                    self.log_update.emit(f"本地未找到模型文件 {self.model_type}.pt，将尝试自动下载")
                    self.model_weights = None
            
            # 全局 ROI：训练前物化裁剪数据集，并对标签坐标重映射
            self._apply_global_roi_if_needed()

            # Create data.yaml file based on dataset format
            yaml_path = self._create_dataset_yaml()
            self._ensure_dataset_ready(yaml_path)

            # Check GPU availability
            device = self._check_gpu()
            self.log_update.emit(f"Using device: {device}")
            
            # Import YOLO after checking environment
            # This is done inside the run method to avoid importing in the main thread
            try:
                from ultralytics import YOLO
                # 尝试获取ultralytics版本
                import ultralytics
                ultralytics_version = getattr(ultralytics, '__version__', 'unknown')
                self.log_update.emit(f"Ultralytics YOLO imported successfully (version: {ultralytics_version})")
                
                # 检测是否可以导入Callback类
                has_callback_class = False
                try:
                    from ultralytics.utils.callbacks.base import Callback
                    has_callback_class = True
                    self.log_update.emit("Callback class available")
                except ImportError:
                    self.log_update.emit("Callback class not available, will use function-based callbacks")
                    has_callback_class = False
                
            except ImportError as e:
                self.training_error.emit(f"Failed to import ultralytics: {str(e)}")
                return
            
            # Initialize the model
            try:
                model_task = 'segment' if '-seg' in self.model_type.lower() else (
                    'obb' if '-obb' in self.model_type.lower() else 'detect'
                )

                if self.model_weights:
                    # Use specified model weights
                    self.log_update.emit(f"正在加载指定模型权重: {self.model_weights}")
                    model = YOLO(self.model_weights)
                    self.log_update.emit(f"成功加载模型权重: {self.model_weights}")
                elif self.pretrained:
                    # Use pretrained weights
                    self.log_update.emit(f"正在加载预训练权重: {self.model_type} (task={model_task})")
                    
                    # Check if it's a YOLO12 model
                    if 'yolo12' in self.model_type.lower():
                        self.log_update.emit(f"检测到YOLO12模型类型: {self.model_type}")
                        try:
                            # For YOLO12, need to specify the correct task
                            model = YOLO(f"{self.model_type}.pt", task=model_task)
                            self.log_update.emit(f"成功加载预训练YOLO12模型: {self.model_type}")
                        except Exception as e:
                            error = str(e)
                            self.log_update.emit(f"加载YOLO12预训练模型失败: {error}")
                            
                            # Check if it's a network issue
                            if "not online" in error.lower() or "download failure" in error.lower():
                                self.log_update.emit("检测到网络连接问题，尝试从头开始训练模型")
                                # Fall back to training from scratch
                                model = YOLO(f"{self.model_type}.yaml", task=model_task)
                                self.log_update.emit(f"已从头初始化YOLO12模型: {self.model_type}")
                            else:
                                # Re-raise the exception if it's not a network issue
                                raise
                    else:
                        # Handle YOLOv5/YOLOv8/YOLO11 models
                        try:
                            # Standard model loading
                            model = YOLO(f"{self.model_type}.pt", task=model_task)
                            self.log_update.emit(f"成功加载预训练模型: {self.model_type}")
                        except Exception as e:
                            error = str(e)
                            self.log_update.emit(f"加载预训练模型失败: {error}")
                            
                            # Check if it's a network issue
                            if "not online" in error.lower() or "download failure" in error.lower():
                                self.log_update.emit("检测到网络连接问题，尝试从头开始训练模型")
                                # Fall back to training from scratch
                                model = YOLO(f"{self.model_type}.yaml", task=model_task)
                                self.log_update.emit(f"已从头初始化模型: {self.model_type}")
                            else:
                                # Re-raise the exception if it's not a network issue
                                raise
                else:
                    # For training from scratch, use the yaml file of the model architecture
                    self.log_update.emit(f"将从头开始训练模型: {self.model_type} (task={model_task})")
                    
                    # Check for model YAML file in multiple locations
                    yaml_found = False
                    yaml_file = None
                    
                    # Possible locations for YAML architecture files
                    yaml_locations = [
                        f"{self.model_type}.yaml",  # Current directory
                        os.path.join("models", f"{self.model_type}.yaml"),
                        os.path.join("configs", f"{self.model_type}.yaml"),
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    f"{self.model_type}.yaml")
                    ]
                    
                    for location in yaml_locations:
                        if os.path.exists(location):
                            yaml_file = location
                            yaml_found = True
                            self.log_update.emit(f"找到本地模型配置文件: {yaml_file}")
                            break
                    
                    try:
                        if yaml_found:
                            model = YOLO(yaml_file, task=model_task)
                        else:
                            model = YOLO(f"{self.model_type}.yaml", task=model_task)
                        self.log_update.emit(f"已从头初始化模型: {self.model_type}")
                    except Exception as e:
                        error = str(e)
                        self.log_update.emit(f"加载模型配置文件失败: {error}")
                        
                        # Try to handle common architecture file errors
                        if "Cannot find" in error or "No such file" in error:
                            # Try to fall back to similar model sizes
                            fallback_model = None
                            
                            # Map model types to fallbacks
                            if self.model_type.endswith("n") or self.model_type.endswith("n-seg"):
                                fallback_model = "yolo11n-seg" if model_task == 'segment' else "yolov8n"
                            elif self.model_type.endswith("s") or self.model_type.endswith("s-seg"):
                                fallback_model = "yolo11s-seg" if model_task == 'segment' else "yolov8s"
                            elif self.model_type.endswith("m") or self.model_type.endswith("m-seg"):
                                fallback_model = "yolo11m-seg" if model_task == 'segment' else "yolov8m"
                            elif self.model_type.endswith("l") or self.model_type.endswith("l-seg"):
                                fallback_model = "yolo11l-seg" if model_task == 'segment' else "yolov8l"
                            elif self.model_type.endswith("x") or self.model_type.endswith("x-seg"):
                                fallback_model = "yolo11x-seg" if model_task == 'segment' else "yolov8x"
                            
                            if fallback_model:
                                self.log_update.emit(f"尝试使用替代模型架构: {fallback_model}")
                                model = YOLO(f"{fallback_model}.yaml", task=model_task)
                                self.log_update.emit(f"已使用替代模型架构: {fallback_model}")
                            else:
                                # Last resort fallback
                                fb = "yolo11m-seg.yaml" if model_task == 'segment' else "yolov8n.yaml"
                                self.log_update.emit(f"使用标准架构作为后备方案: {fb}")
                                model = YOLO(fb, task=model_task)
                        else:
                            # Re-raise if not a missing file issue
                            raise
                
                # Log model information
                task_name = getattr(model, 'task', model_task)
                self.log_update.emit(f"模型任务类型: {task_name}")
                
                # Apply fine-tuning mode if requested
                if self.fine_tuning:
                    self.log_update.emit("启用微调模式: 冻结检测头之前的所有参数，仅更新检测头参数")
                    
                    # Access model's pytorch module
                    pytorch_model = model.model
                    
                    # First, we'll freeze all parameters
                    for param in pytorch_model.parameters():
                        param.requires_grad = False
                        
                    # Then, unfreeze only the detection head layers
                    # For YOLOv8, the detection head is in the 'model.model.model[-1]' (detection module)
                    detection_head = None
                    
                    # YOLOv8 models may have different structures, so try different paths
                    try:
                        # yolo12 specific structure handling
                        if 'yolo12' in self.model_type:
                            self.log_update.emit("检测到yolo12结构，正在识别检测头...")
                            
                            # yolo12 has a different structure than YOLOv8
                            # Try to identify the detection head by name or position
                            if hasattr(pytorch_model, 'model'):
                                if hasattr(pytorch_model.model, 'detect'):
                                    # Direct detect module
                                    detection_head = pytorch_model.model.detect
                                    self.log_update.emit("找到yolo12检测头: model.detect")
                                elif hasattr(pytorch_model.model, 'head'):
                                    # Head module
                                    detection_head = pytorch_model.model.head
                                    self.log_update.emit("找到yolo12检测头: model.head")
                                else:
                                    # Try to locate by position (last modules)
                                    layers = list(pytorch_model.model.children())
                                    # Assume the last 1-2 modules are detection related
                                    detection_head = layers[-1]
                                    self.log_update.emit(f"使用yolo12最后一层作为检测头: {detection_head.__class__.__name__}")
                            else:
                                self.log_update.emit("无法识别yolo12结构，将尝试常规方法")
                        
                        # Standard YOLOv8 structure: typically the last module is the detection head
                        elif hasattr(pytorch_model, 'model') and hasattr(pytorch_model.model, 'model'):
                            detection_head = pytorch_model.model.model[-1]
                            self.log_update.emit("检测到YOLOv8结构，已找到检测头")
                        else:
                            # For other structures, try to identify the detection head by name
                            for name, module in pytorch_model.named_children():
                                if 'detect' in name.lower() or 'head' in name.lower():
                                    detection_head = module
                                    self.log_update.emit(f"根据名称找到检测头: {name}")
                                    break
                            
                            # If we still can't find it, try last layer as fallback
                            if detection_head is None:
                                # Get the last layer as fallback
                                layers = list(pytorch_model.children())
                                detection_head = layers[-1]
                                self.log_update.emit("使用模型最后一层作为检测头进行微调")
                        
                        # Unfreeze the detection head parameters
                        if detection_head:
                            for param in detection_head.parameters():
                                param.requires_grad = True
                            
                            # Count trainable parameters
                            trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
                            total_params = sum(p.numel() for p in pytorch_model.parameters())
                            self.log_update.emit(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,}")
                            self.log_update.emit(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")
                        else:
                            self.log_update.emit("警告: 无法找到检测头，微调模式可能无效")
                    except Exception as e:
                        error_msg = f"设置微调模式时出错: {str(e)}"
                        self.log_update.emit(error_msg)
                        self.log_update.emit("将继续训练但微调设置可能未成功应用")
                
            except Exception as e:
                self.training_error.emit(f"Failed to initialize model: {str(e)}")
                return
            
            # Start training
            try:
                run_name = time.strftime("%Y%m%d-%H%M%S")
                self.run_stamp = run_name
                self.effective_run_name = self._make_run_name(run_name)
                metrics_root = os.path.join(self.output_dir, self.effective_run_name)
                self.log_update.emit(f"本次训练输出目录: {metrics_root}")

                stop_flag = threading.Event()

                def find_metrics_file():
                    # 只读本次 run 目录，避免扫到历史 results.csv
                    csv_path = os.path.join(metrics_root, 'results.csv')
                    if os.path.isfile(csv_path):
                        return csv_path
                    return None

                def _format_epoch_line(metrics: dict) -> str:
                    try:
                        ep = int(float(metrics.get('epoch', 0)))
                        if ep < 1:
                            ep = 1
                    except (TypeError, ValueError):
                        ep = '?'
                    parts = [f'Epoch {ep}/{self.epochs}']
                    for key, short in (
                        ('train/box_loss', 'box'),
                        ('train/cls_loss', 'cls'),
                        ('train/dfl_loss', 'dfl'),
                        ('metrics/precision(B)', 'P'),
                        ('metrics/recall(B)', 'R'),
                        ('metrics/mAP50(B)', 'mAP50'),
                        ('metrics/mAP50-95(B)', 'mAP50-95'),
                        ('metrics/precision', 'P'),
                        ('metrics/recall', 'R'),
                        ('metrics/mAP50', 'mAP50'),
                        ('metrics/mAP50-95', 'mAP50-95'),
                    ):
                        if key not in metrics:
                            continue
                        try:
                            parts.append(f'{short}={float(metrics[key]):.3f}')
                        except (TypeError, ValueError):
                            pass
                    seen = set()
                    uniq = []
                    for p in parts:
                        k = p.split('=', 1)[0]
                        if k in seen:
                            continue
                        seen.add(k)
                        uniq.append(p)
                    return ' | '.join(uniq)

                def progress_monitor():
                    self.progress_update.emit(0)
                    metrics_file = None
                    last_metrics_time = 0
                    last_emitted_epoch = -1
                    train_started_at = time.time()

                    while not stop_flag.is_set() and not self._stop_event.is_set():
                        if metrics_file is None:
                            cand = find_metrics_file()
                            if cand and os.path.getmtime(cand) >= train_started_at - 1:
                                metrics_file = cand

                        if metrics_file and os.path.exists(metrics_file):
                            current_time = os.path.getmtime(metrics_file)
                            if current_time > last_metrics_time:
                                last_metrics_time = current_time
                                try:
                                    with open(metrics_file, 'r', encoding='utf-8', errors='ignore') as f:
                                        lines = f.readlines()
                                    if len(lines) > 1:
                                        header = [h.strip() for h in lines[0].strip().split(',')]
                                        values = lines[-1].strip().split(',')
                                        metrics = {}
                                        for i, key in enumerate(header):
                                            if i < len(values):
                                                try:
                                                    metrics[key] = float(values[i])
                                                except ValueError:
                                                    metrics[key] = values[i]
                                        if 'epoch' in metrics:
                                            try:
                                                epoch = int(float(metrics['epoch']))
                                                if epoch < 1:
                                                    epoch = 1
                                                elif epoch > self.epochs:
                                                    epoch = self.epochs
                                                if epoch != last_emitted_epoch:
                                                    last_emitted_epoch = epoch
                                                    self.progress_update.emit(epoch)
                                                    self.log_update.emit(_format_epoch_line(metrics))
                                            except (TypeError, ValueError):
                                                pass
                                except Exception as e:
                                    self.log_update.emit(f"读取指标文件出错: {str(e)}")
                        time.sleep(1.0)

                start_time = time.time()
                self.log_update.emit("启动进度监控...")
                monitor_thread = threading.Thread(target=progress_monitor)
                monitor_thread.daemon = True
                monitor_thread.start()

                class StdoutCapture:
                    """透传 stdout 到原始终端；不再二次写入 UI（避免刷屏/重复）。"""

                    def __init__(self, worker):
                        self.worker = worker
                        self.original_stdout = sys.stdout

                    def write(self, text):
                        try:
                            self.original_stdout.write(text)
                            self.original_stdout.flush()
                        except Exception:
                            pass

                    def flush(self):
                        try:
                            self.original_stdout.flush()
                        except Exception:
                            pass

                    def __enter__(self):
                        sys.stdout = self
                        return self

                    def __exit__(self, exc_type, exc_val, exc_tb):
                        sys.stdout = self.original_stdout

                # 创建捕获实例
                stdout_capture = StdoutCapture(self)
                
                # 创建通用回调函数，支持任何版本的ultralytics
                def on_train_batch_end_fn(trainer=None):
                    # 在每个训练批次结束时检查停止标志
                    if self._stop_event.is_set():
                        self.log_update.emit("检测到停止信号，正在中断训练...")
                        if trainer:
                            self._trainer_ref = trainer  # Store reference to trainer
                            # 尝试停止训练循环
                            if hasattr(trainer, 'epoch_progress'):
                                try:
                                    trainer.epoch_progress.close()  # 关闭进度条
                                except:
                                    pass
                            if hasattr(trainer, 'stop'):
                                trainer.stop = True
                        return False  # 返回False以停止训练循环
                    return True
                
                def on_train_epoch_end_fn(trainer=None):
                    # 每个 epoch 结束时更新进度（当前轮 / 总轮数）
                    try:
                        if trainer is not None and hasattr(trainer, 'epoch'):
                            # trainer.epoch 为 0-based
                            current = int(trainer.epoch) + 1
                        else:
                            current = 0
                        total = int(getattr(trainer, 'epochs', self.epochs) if trainer else self.epochs)
                        current = max(0, min(current, total or self.epochs))
                        self.progress_update.emit(current)
                    except Exception:
                        pass

                    if self._stop_event.is_set():
                        self.log_update.emit("检测到停止信号，正在中断训练...")
                        if trainer:
                            self._trainer_ref = trainer
                            if hasattr(trainer, 'stop'):
                                trainer.stop = True
                        return False
                    return True
                
                # 添加新的回调函数，用于设置进程参考
                def on_train_start_fn(trainer=None):
                    if trainer:
                        self._trainer_ref = trainer  # Store reference to trainer
                        self.log_update.emit("训练开始")
                    self.progress_update.emit(0)
                    import threading
                    self._process_ref = threading.current_thread()
                    return True
                
                train_args = {
                    'data': yaml_path,
                    'epochs': self.epochs,
                    'batch': self.batch_size,
                    'imgsz': self.img_size,
                    'project': self.output_dir,
                    'name': self.effective_run_name,
                    'lr0': self.learning_rate,
                    'device': device,
                    'exist_ok': True,
                    'deterministic': False,
                    'workers': 0,
                    'verbose': False,
                    'plots': False,
                    'amp': True,
                    'rect': True,
                    # 关闭早停：默认 patience=100 会在指标不提升时提前结束
                    'patience': 0,
                }

                self.log_update.emit("配置训练参数和回调...")
                self.log_update.emit(
                    f"加速选项: amp=True, rect=True | 早停已关闭 (patience=0)，将训练满 {self.epochs} 轮"
                )
                model.add_callback("on_train_start", on_train_start_fn)
                model.add_callback("on_train_batch_end", on_train_batch_end_fn)
                model.add_callback("on_train_epoch_end", on_train_epoch_end_fn)

                with stdout_capture:
                    self.log_update.emit("开始训练，第一个epoch可能较慢，因为需要进行初始化和缓存")
                    results = model.train(**train_args)

                stop_flag.set()

                if self._stop_event.is_set():
                    self.log_update.emit("训练被用户中止")
                    self._finalize_after_stop(model, results)
                    self.training_stopped.emit()
                elif results is None:
                    self.training_error.emit("训练未返回结果")
                else:
                    self.log_update.emit("训练成功完成!")
                    self.progress_update.emit(self.epochs)
                    self._emit_final_metrics(results, model)

                    best_path = self._locate_best_weights(model, results)
                    if best_path:
                        self.log_update.emit(f"最佳权重: {best_path}")
                        published = self._publish_models_to_project(best_path)
                        if published.get('best_pt'):
                            best_path = published['best_pt']
                            self.log_update.emit(f"已保存到项目: {best_path}")
                        if published.get('best_onnx'):
                            self.log_update.emit(f"ONNX 模型: {published['best_onnx']}")
                        self.best_weights_ready.emit(best_path)
                    else:
                        self.log_update.emit("警告: 未找到 best.pt")

                    self.training_complete.emit()
                
            except Exception as e:
                stop_flag.set()
                if self._stop_event.is_set():
                    self.log_update.emit("训练已被用户中止")
                    try:
                        self._finalize_after_stop(model if 'model' in locals() else None,
                                                  results if 'results' in locals() else None)
                    except Exception as finalize_err:
                        self.log_update.emit(f"中止后整理模型失败: {finalize_err}")
                    self.training_stopped.emit()
                else:
                    self.training_error.emit(f"训练错误: {str(e)}")
        
        except Exception as e:
            self.training_error.emit(f"意外错误: {str(e)}")
    
    def stop(self):
        """Stop the training process immediately."""
        self._stop_event.set()
        self.log_update.emit("收到停止信号，立即中断训练...")
        
        # Attempt to terminate the training more aggressively
        if self._trainer_ref is not None:
            try:
                # Try all possible ways to forcibly stop the trainer
                if hasattr(self._trainer_ref, 'stop'):
                    self._trainer_ref.stop = True
                if hasattr(self._trainer_ref, 'epoch_progress') and hasattr(self._trainer_ref.epoch_progress, 'close'):
                    self._trainer_ref.epoch_progress.close()
                if hasattr(self._trainer_ref, 'stopper') and hasattr(self._trainer_ref.stopper, 'run'):
                    self._trainer_ref.stopper.possible_stop = True
                self.log_update.emit("已发送终止信号到训练器")
            except Exception as e:
                self.log_update.emit(f"尝试终止训练器时出错: {str(e)}")
        
        # If we have a training process, attempt to terminate it more forcefully
        if self._process_ref is not None:
            try:
                import signal
                import ctypes
                import os
                
                if hasattr(self._process_ref, 'terminate'):
                    self._process_ref.terminate()
                    self.log_update.emit("已强制终止训练进程")
                elif isinstance(self._process_ref, threading.Thread) and self._process_ref.is_alive():
                    # This is a more aggressive approach for Python threads
                    if hasattr(threading, '_async_raise'):
                        threading._async_raise(self._process_ref.ident, SystemExit)
                    self.log_update.emit("已尝试强制终止训练线程")
            except Exception as e:
                self.log_update.emit(f"尝试强制终止训练时出错: {str(e)}")

    def _finalize_after_stop(self, model, results) -> None:
        """用户停止训练时：若已有 best/last 权重，仍复制到项目并导出 ONNX。"""
        best_path = self._locate_best_weights(model, results)
        if not best_path:
            # 再找 last.pt 作为兜底
            run_name = getattr(self, 'effective_run_name', None) or self.project_name or ''
            candidates = []
            if run_name:
                candidates.append(os.path.join(self.output_dir, run_name, 'weights', 'last.pt'))
                candidates.append(os.path.join(self.output_dir, run_name, 'weights', 'best.pt'))
            candidates.append(os.path.join(self.output_dir, 'weights', 'last.pt'))
            for path in candidates:
                if path and os.path.isfile(path):
                    best_path = os.path.abspath(path)
                    break
        if not best_path:
            self.log_update.emit("中止时尚未生成可用权重，跳过 ONNX 导出")
            return

        self.log_update.emit(f"中止前已有权重，正在整理到项目并导出 ONNX: {best_path}")
        try:
            published = self._publish_models_to_project(best_path)
            if published.get('best_pt'):
                self.log_update.emit(f"已保存到项目: {published['best_pt']}")
                self.best_weights_ready.emit(published['best_pt'])
            if published.get('best_onnx'):
                self.log_update.emit(f"ONNX 模型: {published['best_onnx']}")
            elif self.export_onnx:
                self.log_update.emit("未能生成 ONNX（请检查导出依赖或日志）")
        except Exception as e:
            self.log_update.emit(f"中止后导出模型失败: {e}")

    def _make_run_name(self, stamp: str) -> str:
        """生成带时间戳的唯一 run 名称，避免覆盖旧模型。"""
        import re
        base = (self.project_name or '').strip()
        if not base:
            return stamp
        # 调用方已传入纯时间戳时不再二次追加
        if re.fullmatch(r'\d{8}-\d{6}', base):
            return base
        if base.endswith('_' + stamp) or base.endswith(stamp):
            return base
        return f"{base}_{stamp}"

    def _check_gpu(self):
        """Check if CUDA is available and return appropriate device."""
        if self.use_gpu and torch.cuda.is_available():
            device_id = max(0, min(int(self.gpu_device), torch.cuda.device_count() - 1))
            return device_id
        if not self.use_gpu:
            self.log_update.emit("设置中已禁用 GPU，使用 CPU 训练")
        else:
            self.log_update.emit("未检测到 CUDA，使用 CPU 训练")
        return 'cpu'

    def _locate_best_weights(self, model, results) -> str:
        """定位训练产出的 best.pt。"""
        candidates = []
        try:
            if results is not None and hasattr(results, 'save_dir') and results.save_dir:
                candidates.append(os.path.join(str(results.save_dir), 'weights', 'best.pt'))
        except Exception:
            pass
        try:
            trainer = getattr(model, 'trainer', None)
            if trainer is not None and getattr(trainer, 'best', None):
                candidates.append(str(trainer.best))
            if trainer is not None and getattr(trainer, 'save_dir', None):
                candidates.append(os.path.join(str(trainer.save_dir), 'weights', 'best.pt'))
        except Exception:
            pass

        run_name = getattr(self, 'effective_run_name', None) or self.project_name or ''
        if run_name:
            candidates.append(os.path.join(self.output_dir, run_name, 'weights', 'best.pt'))
            candidates.append(os.path.join(self.output_dir, run_name, 'weights', 'last.pt'))
        candidates.append(os.path.join(self.output_dir, 'weights', 'best.pt'))

        for path in candidates:
            if path and os.path.isfile(path):
                return os.path.abspath(path)

        # 兜底：在输出目录下找最新 best.pt
        newest = None
        newest_mtime = -1
        for root, _, files in os.walk(self.output_dir):
            if 'best.pt' in files:
                path = os.path.join(root, 'best.pt')
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                if mtime > newest_mtime:
                    newest_mtime = mtime
                    newest = path
        return os.path.abspath(newest) if newest else ''

    def _emit_final_metrics(self, results, model) -> None:
        """训练结束时打印最终 metric。"""
        self.log_update.emit("=" * 50)
        self.log_update.emit("最终 Metrics:")
        printed = False

        # 1) results_dict（ultralytics 常见）
        results_dict = None
        for obj in (results, getattr(results, 'metrics', None), getattr(model, 'metrics', None)):
            if obj is None:
                continue
            if hasattr(obj, 'results_dict') and obj.results_dict:
                results_dict = dict(obj.results_dict)
                break
            if isinstance(obj, dict):
                results_dict = dict(obj)
                break

        if results_dict:
            key_map = [
                ('metrics/precision(B)', 'precision'),
                ('metrics/recall(B)', 'recall'),
                ('metrics/mAP50(B)', 'mAP50'),
                ('metrics/mAP50-95(B)', 'mAP50-95'),
                ('metrics/precision', 'precision'),
                ('metrics/recall', 'recall'),
                ('metrics/mAP50', 'mAP50'),
                ('metrics/mAP50-95', 'mAP50-95'),
            ]
            seen = set()
            for key, label in key_map:
                if key in results_dict and label not in seen:
                    try:
                        self.log_update.emit(f"  {label}: {float(results_dict[key]):.4f}")
                        seen.add(label)
                        printed = True
                    except (TypeError, ValueError):
                        pass
            # 补充打印其余 numeric metrics
            for key, val in results_dict.items():
                if not str(key).startswith('metrics/'):
                    continue
                short = str(key).split('/', 1)[-1]
                if short.rstrip('(B)') in seen or short in seen:
                    continue
                try:
                    self.log_update.emit(f"  {key}: {float(val):.4f}")
                    printed = True
                except (TypeError, ValueError):
                    pass

        # 2) box 属性兜底
        if not printed:
            box = None
            for obj in (results, getattr(results, 'metrics', None), getattr(model, 'metrics', None)):
                if obj is not None and hasattr(obj, 'box'):
                    box = obj.box
                    break
            if box is not None:
                for attr, label in (
                    ('mp', 'precision'),
                    ('mr', 'recall'),
                    ('map50', 'mAP50'),
                    ('map', 'mAP50-95'),
                ):
                    if hasattr(box, attr):
                        try:
                            self.log_update.emit(f"  {label}: {float(getattr(box, attr)):.4f}")
                            printed = True
                        except (TypeError, ValueError):
                            pass

        # 3) results.csv 最后一行
        if not printed:
            csv_path = None
            save_dir = getattr(results, 'save_dir', None) if results is not None else None
            search_roots = []
            if save_dir:
                search_roots.append(str(save_dir))
            run_name = self.project_name or ''
            if run_name:
                search_roots.append(os.path.join(self.output_dir, run_name))
            search_roots.append(self.output_dir)
            for root in search_roots:
                candidate = os.path.join(root, 'results.csv')
                if os.path.isfile(candidate):
                    csv_path = candidate
                    break
                if os.path.isdir(root):
                    for dirpath, _, filenames in os.walk(root):
                        if 'results.csv' in filenames:
                            csv_path = os.path.join(dirpath, 'results.csv')
                            break
                if csv_path:
                    break
            if csv_path:
                try:
                    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                    if len(lines) >= 2:
                        headers = [h.strip() for h in lines[0].split(',')]
                        values = [v.strip() for v in lines[-1].split(',')]
                        interesting = (
                            'metrics/precision(B)', 'metrics/recall(B)',
                            'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                            'metrics/precision', 'metrics/recall',
                            'metrics/mAP50', 'metrics/mAP50-95',
                            'precision', 'recall', 'mAP50', 'mAP50-95',
                        )
                        for key in interesting:
                            if key in headers:
                                idx = headers.index(key)
                                if idx < len(values):
                                    try:
                                        self.log_update.emit(f"  {key}: {float(values[idx]):.4f}")
                                        printed = True
                                    except ValueError:
                                        pass
                        self.log_update.emit(f"  (来源: {csv_path})")
                except Exception as e:
                    self.log_update.emit(f"  读取 results.csv 失败: {e}")

        if not printed:
            self.log_update.emit("  (未能解析到 metric，请查看 results.csv)")
        self.log_update.emit("=" * 50)

    def _publish_models_to_project(self, best_path: str) -> dict:
        """将 best.pt / ONNX 整理到带时间戳的项目输出目录，避免覆盖旧模型。"""
        import shutil
        from ultralytics import YOLO

        published = {'best_pt': '', 'best_onnx': '', 'last_pt': '', 'run_dir': ''}
        if not best_path or not os.path.isfile(best_path):
            return published

        stamp = getattr(self, 'run_stamp', None) or time.strftime("%Y%m%d-%H%M%S")
        run_name = getattr(self, 'effective_run_name', None) or self._make_run_name(stamp)
        run_root = os.path.join(self.output_dir, run_name)
        os.makedirs(run_root, exist_ok=True)
        weights_dir = os.path.join(run_root, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        published['run_dir'] = os.path.abspath(run_root)

        # 时间戳文件名，同时保留 best.pt 便于兼容
        stamped_pt_name = f'best_{stamp}.pt'
        stamped_onnx_name = f'best_{stamp}.onnx'
        stamped_last_name = f'last_{stamp}.pt'

        dest_best = os.path.join(weights_dir, 'best.pt')
        dest_best_flat = os.path.join(run_root, 'best.pt')
        dest_best_stamped = os.path.join(run_root, stamped_pt_name)
        try:
            if os.path.abspath(best_path) != os.path.abspath(dest_best):
                shutil.copy2(best_path, dest_best)
            src_for_flat = dest_best if os.path.isfile(dest_best) else best_path
            shutil.copy2(src_for_flat, dest_best_flat)
            shutil.copy2(src_for_flat, dest_best_stamped)
            published['best_pt'] = os.path.abspath(dest_best_stamped)
        except Exception as e:
            self.log_update.emit(f"复制 best.pt 失败: {e}")
            published['best_pt'] = os.path.abspath(best_path)

        # last.pt（若存在）一并放入项目
        src_last = os.path.join(os.path.dirname(best_path), 'last.pt')
        if os.path.isfile(src_last):
            try:
                dest_last = os.path.join(weights_dir, 'last.pt')
                if os.path.abspath(src_last) != os.path.abspath(dest_last):
                    shutil.copy2(src_last, dest_last)
                last_src = dest_last if os.path.isfile(dest_last) else src_last
                shutil.copy2(last_src, os.path.join(run_root, 'last.pt'))
                shutil.copy2(last_src, os.path.join(run_root, stamped_last_name))
                published['last_pt'] = os.path.abspath(os.path.join(run_root, stamped_last_name))
            except Exception as e:
                self.log_update.emit(f"复制 last.pt 失败: {e}")

        if not self.export_onnx:
            return published

        try:
            self.log_update.emit(f"正在导出 ONNX 格式到: {run_root}")
            export_model = YOLO(published['best_pt'] or best_path)
            onnx_path = export_model.export(
                format="onnx",
                imgsz=self.img_size,
                dynamic=False,
                simplify=True,
            )
            if not onnx_path:
                guess = os.path.splitext(published['best_pt'] or best_path)[0] + '.onnx'
                onnx_path = guess if os.path.isfile(guess) else ''

            if onnx_path and os.path.isfile(str(onnx_path)):
                onnx_path = os.path.abspath(str(onnx_path))
                dest_onnx = os.path.join(run_root, 'best.onnx')
                dest_onnx_stamped = os.path.join(run_root, stamped_onnx_name)
                dest_onnx_weights = os.path.join(weights_dir, 'best.onnx')
                if os.path.abspath(onnx_path) != os.path.abspath(dest_onnx):
                    shutil.copy2(onnx_path, dest_onnx)
                onnx_src = dest_onnx if os.path.isfile(dest_onnx) else onnx_path
                shutil.copy2(onnx_src, dest_onnx_stamped)
                shutil.copy2(onnx_src, dest_onnx_weights)
                published['best_onnx'] = os.path.abspath(dest_onnx_stamped)
            else:
                self.log_update.emit("ONNX 导出完成但未找到输出文件")
        except Exception as export_err:
            self.log_update.emit(f"ONNX 导出跳过: {export_err}")

        return published

    def _has_images(self, directory: str) -> bool:
        if not directory or not os.path.isdir(directory):
            return False
        try:
            for name in os.listdir(directory):
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    return True
        except OSError:
            return False
        return False

    def _swap_labels_to_images(self, path: str) -> str:
        """将 labels 目录路径替换为对应的 images 目录路径。"""
        if not path:
            return path
        normalized = path.replace('\\', '/')
        if '/labels/' not in normalized and not normalized.endswith('/labels'):
            return path
        candidate = normalized.replace('/labels/', '/images/')
        if candidate.endswith('/labels'):
            candidate = candidate[:-len('/labels')] + '/images'
        candidate = candidate.replace('/', os.sep)
        return os.path.normpath(candidate)

    def _resolve_image_dir(self, directory: str, split_name: str) -> str:
        """解析并修正单个 train/val 图像目录。"""
        directory = os.path.abspath(self._normalize_path(directory))
        if not os.path.isdir(directory):
            return directory

        if self._has_images(directory):
            return directory

        alt = self._swap_labels_to_images(directory)
        if alt != directory and os.path.isdir(alt) and self._has_images(alt):
            self.log_update.emit(
                f"自动修正{split_name}路径: 标签目录 -> 图像目录\n  {directory}\n  -> {alt}"
            )
            return alt

        parent = os.path.dirname(directory)
        grand = os.path.dirname(parent)
        split = os.path.basename(directory)
        if os.path.basename(parent) == 'labels' and split in ('train', 'val', 'test'):
            candidate = os.path.join(grand, 'images', split)
            if os.path.isdir(candidate) and self._has_images(candidate):
                self.log_update.emit(
                    f"自动修正{split_name}路径:\n  {directory}\n  -> {candidate}"
                )
                return candidate

        return directory

    def _resolve_yolo_dataset_config(self) -> dict:
        """根据 UI 填写的目录生成正确的 YOLO data.yaml 配置。"""
        train_dir = self._resolve_image_dir(self.train_dir, '训练集')
        val_dir = self._resolve_image_dir(self.val_dir, '验证集')

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"训练图像目录不存在: {train_dir}")
        if not self._has_images(train_dir):
            raise FileNotFoundError(
                f"训练目录中没有图像文件: {train_dir}\n"
                "请确认「训练图像目录」指向 images/train，而不是 labels/train。"
            )

        if not os.path.isdir(val_dir) or not self._has_images(val_dir):
            self.log_update.emit(f"验证图像目录不可用，将使用训练集作为验证集: {train_dir}")
            val_dir = train_dir

        dataset_root = self._find_common_parent(train_dir, val_dir)
        train_rel = os.path.relpath(train_dir, dataset_root).replace('\\', '/')
        val_rel = os.path.relpath(val_dir, dataset_root).replace('\\', '/')

        config = {
            'path': dataset_root.replace('\\', '/'),
            'train': train_rel,
            'val': val_rel,
        }
        self.log_update.emit(
            f"数据集配置:\n  path: {config['path']}\n  train: {config['train']}\n  val: {config['val']}"
        )
        return config

    def _write_yolo_yaml(self, yaml_path: str, config: dict, class_names: list) -> None:
        import yaml
        data = {
            'path': config['path'],
            'train': config['train'],
            'val': config['val'],
            'nc': len(class_names),
            'names': class_names,
        }
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def _yaml_split_path(self, data: dict, key: str) -> str:
        base = data.get('path', '')
        rel = data.get(key, '')
        if not rel:
            return ''
        if os.path.isabs(rel):
            return self._normalize_path(rel)
        return self._normalize_path(os.path.join(base, rel))

    def _yaml_paths_valid(self, yaml_path: str) -> bool:
        import yaml
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            train_path = self._yaml_split_path(data, 'train')
            return bool(train_path and os.path.isdir(train_path) and self._has_images(train_path))
        except Exception:
            return False

    def _ensure_dataset_ready(self, yaml_path: str) -> None:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        train_path = self._yaml_split_path(data, 'train')
        val_path = self._yaml_split_path(data, 'val')

        if not train_path or not os.path.isdir(train_path) or not self._has_images(train_path):
            raise FileNotFoundError(
                f"训练图像路径无效: {train_path or '(空)'}\n"
                "请检查训练 Tab 中的「训练图像目录」是否指向 images/train。"
            )

        if not val_path or not os.path.isdir(val_path) or not self._has_images(val_path):
            self.log_update.emit(f"验证图像路径无效，将改用训练集: {train_path}")
            data['val'] = data.get('train')
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def _create_dataset_yaml(self):
        """
        Create the dataset YAML file based on the selected format.
        
        Returns:
            str: Path to the created YAML file
        """
        self.log_update.emit("准备训练数据配置...")
        
        os.makedirs(os.path.join(self.output_dir, "datasets"), exist_ok=True)
        yaml_path = os.path.join(self.output_dir, "datasets", "data.yaml")
        
        # 检查是否存在缓存的YAML文件，而且训练目录没有变化
        cache_info_path = os.path.join(self.output_dir, "datasets", "cache_info.txt")
        if os.path.exists(yaml_path) and os.path.exists(cache_info_path):
            try:
                with open(cache_info_path, 'r', encoding='utf-8') as f:
                    cached = f.read().strip()
                if cached == self._cache_fingerprint().strip():
                    if self._yaml_paths_valid(yaml_path):
                        self.log_update.emit("使用缓存的数据集配置...")
                        return yaml_path
                    self.log_update.emit("缓存的数据集配置无效，重新生成...")
            except Exception:
                pass

        if self.dataset_format == "YOLO":
            config = self._resolve_yolo_dataset_config()
            class_names = self._get_class_names()
            self._write_yolo_yaml(yaml_path, config, class_names)
            self._validate_yaml(yaml_path)
            self.log_update.emit(f"已创建YOLO格式数据集配置文件: {yaml_path}")
            try:
                with open(cache_info_path, 'w', encoding='utf-8') as f:
                    f.write(self._cache_fingerprint())
            except Exception as e:
                self.log_update.emit(f"保存缓存信息失败: {str(e)}")
            return yaml_path

        # COCO / VOC：同样按图像目录解析相对路径，避免 path/train 写错
        try:
            config = self._resolve_yolo_dataset_config()
        except Exception as e:
            self.log_update.emit(f"解析数据集路径失败，回退到简单路径: {e}")
            train_dir = self._normalize_path(self.train_dir)
            val_dir = self._normalize_path(self.val_dir)
            if not os.path.exists(val_dir):
                val_dir = train_dir
            dataset_root = self._find_common_parent(train_dir, val_dir) or os.path.dirname(train_dir)
            config = {
                'path': dataset_root.replace('\\', '/'),
                'train': os.path.relpath(train_dir, dataset_root).replace('\\', '/'),
                'val': os.path.relpath(val_dir, dataset_root).replace('\\', '/'),
            }

        class_names = self._get_class_names()
        self._write_yolo_yaml(yaml_path, config, class_names)
        self._validate_yaml(yaml_path)
        self.log_update.emit(f"已创建 {self.dataset_format} 数据集配置文件: {yaml_path}")

        try:
            with open(cache_info_path, 'w', encoding='utf-8') as f:
                f.write(self._cache_fingerprint())
            self.log_update.emit("已缓存数据集信息，下次训练将加速初始化")
        except Exception as e:
            self.log_update.emit(f"保存缓存信息失败: {str(e)}")

        return yaml_path

    def _cache_fingerprint(self) -> str:
        """缓存指纹：包含图像/标签目录与 ROI，避免变更后误用旧 yaml。"""
        roi = self.roi_norm if self.roi_enabled else (0.0, 0.0, 1.0, 1.0)
        return '\n'.join([
            self.train_dir or '',
            self.val_dir or '',
            self.train_labels_dir or '',
            self.val_labels_dir or '',
            self.dataset_format or '',
            f'roi_enabled={int(bool(self.roi_enabled))}',
            f'roi={roi[0]:.6f},{roi[1]:.6f},{roi[2]:.6f},{roi[3]:.6f}',
        ])

    def _roi_is_full_frame(self) -> bool:
        x1, y1, x2, y2 = self.roi_norm
        return (
            abs(x1) < 1e-6 and abs(y1) < 1e-6
            and abs(x2 - 1.0) < 1e-6 and abs(y2 - 1.0) < 1e-6
        )

    def _guess_labels_dir(self, images_dir: str, explicit_labels: str = '') -> str:
        """根据图像目录推断对应 labels 目录。"""
        if explicit_labels and os.path.isdir(explicit_labels):
            return explicit_labels
        images_dir = self._normalize_path(images_dir)
        # .../images/train -> .../labels/train
        parent = os.path.dirname(images_dir)
        split = os.path.basename(images_dir)
        if os.path.basename(parent) == 'images':
            candidate = os.path.join(os.path.dirname(parent), 'labels', split)
            if os.path.isdir(candidate):
                return candidate
        # 同级 labels
        sibling = os.path.join(os.path.dirname(images_dir), 'labels')
        if os.path.isdir(sibling):
            return sibling
        # 图像目录内直接放 txt
        return images_dir

    def _list_image_files(self, directory: str):
        files = []
        if not directory or not os.path.isdir(directory):
            return files
        for name in sorted(os.listdir(directory)):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                files.append(os.path.join(directory, name))
        return files

    def _remap_yolo_label_line(self, line: str, img_w: int, img_h: int,
                               rx1: int, ry1: int, rx2: int, ry2: int) -> str:
        """将一行 YOLO 标签映射到 ROI 裁剪坐标系；无有效交集则返回空串。"""
        parts = line.strip().split()
        if len(parts) < 5:
            return ''
        crop_w = max(rx2 - rx1, 1)
        crop_h = max(ry2 - ry1, 1)
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            return ''

        # detect: class cx cy w h
        if len(parts) == 5:
            try:
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                return ''
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            ix1 = max(x1, rx1)
            iy1 = max(y1, ry1)
            ix2 = min(x2, rx2)
            iy2 = min(y2, ry2)
            if ix2 - ix1 < 2 or iy2 - iy1 < 2:
                return ''
            ncx = ((ix1 + ix2) / 2 - rx1) / crop_w
            ncy = ((iy1 + iy2) / 2 - ry1) / crop_h
            nw = (ix2 - ix1) / crop_w
            nh = (iy2 - iy1) / crop_h
            ncx = min(max(ncx, 0.0), 1.0)
            ncy = min(max(ncy, 0.0), 1.0)
            nw = min(max(nw, 1e-6), 1.0)
            nh = min(max(nh, 1e-6), 1.0)
            return f'{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}'

        # segment: class x1 y1 x2 y2 ...
        if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
            try:
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                return ''
            pts = []
            for i in range(0, len(coords), 2):
                px = coords[i] * img_w
                py = coords[i + 1] * img_h
                # clamp 到 ROI
                px = min(max(px, rx1), rx2)
                py = min(max(py, ry1), ry2)
                pts.append(((px - rx1) / crop_w, (py - ry1) / crop_h))
            # 丢弃完全塌缩的多边形
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if max(xs) - min(xs) < 1e-4 or max(ys) - min(ys) < 1e-4:
                return ''
            body = ' '.join(f'{x:.6f} {y:.6f}' for x, y in pts)
            return f'{cls_id} {body}'

        return ''

    def _crop_split_with_roi(self, images_dir: str, labels_dir: str, out_images: str, out_labels: str,
                             rx_n, ry_n, rw_n, rh_n) -> int:
        """裁剪一个 split，返回成功写出的图像数。"""
        import cv2

        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)
        count = 0
        images = self._list_image_files(images_dir)
        total = len(images)
        for idx, img_path in enumerate(images):
            if self._stop_event.is_set():
                raise RuntimeError('训练已停止')
            img = cv2.imread(img_path)
            if img is None:
                self.log_update.emit(f'跳过无法读取的图像: {img_path}')
                continue
            h, w = img.shape[:2]
            rx1 = int(round(rx_n * w))
            ry1 = int(round(ry_n * h))
            rx2 = int(round((rx_n + rw_n) * w))
            ry2 = int(round((ry_n + rh_n) * h))
            rx1 = max(0, min(rx1, w - 1))
            ry1 = max(0, min(ry1, h - 1))
            rx2 = max(rx1 + 1, min(rx2, w))
            ry2 = max(ry1 + 1, min(ry2, h))
            cropped = img[ry1:ry2, rx1:rx2]
            fname = os.path.basename(img_path)
            stem = os.path.splitext(fname)[0]
            out_img = os.path.join(out_images, fname)
            if not cv2.imwrite(out_img, cropped):
                # 部分扩展名写失败时回退 png
                out_img = os.path.join(out_images, stem + '.png')
                cv2.imwrite(out_img, cropped)

            src_label = os.path.join(labels_dir, stem + '.txt')
            out_label = os.path.join(out_labels, stem + '.txt')
            lines_out = []
            if os.path.isfile(src_label):
                try:
                    with open(src_label, 'r', encoding='utf-8') as f:
                        for line in f:
                            mapped = self._remap_yolo_label_line(line, w, h, rx1, ry1, rx2, ry2)
                            if mapped:
                                lines_out.append(mapped)
                except Exception as e:
                    self.log_update.emit(f'读取标签失败 {src_label}: {e}')
            with open(out_label, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines_out) + ('\n' if lines_out else ''))

            count += 1
            if total and (idx + 1) % max(1, total // 10) == 0:
                self.log_update.emit(f'ROI 裁剪进度: {idx + 1}/{total}')
        return count

    def _apply_global_roi_if_needed(self):
        """若启用全局 ROI，生成裁剪数据集并切换 train/val 目录。"""
        if not self.roi_enabled or self._roi_is_full_frame():
            if self.roi_enabled and self._roi_is_full_frame():
                self.log_update.emit('ROI 为全图，跳过裁剪')
            return

        x1, y1, x2, y2 = self.roi_norm
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if x2 - x1 < 0.01 or y2 - y1 < 0.01:
            raise ValueError(f'ROI 过小，无法用于训练: ({x1}, {y1})-({x2}, {y2})')

        self.roi_norm = (x1, y1, x2, y2)
        rw, rh = x2 - x1, y2 - y1
        self.log_update.emit(
            f'启用全局 ROI 裁剪: ({x1:.4f}, {y1:.4f}) → ({x2:.4f}, {y2:.4f})'
        )

        train_img = self._resolve_image_dir(self.train_dir, '训练集')
        val_img = self._resolve_image_dir(self.val_dir, '验证集')
        if not os.path.isdir(val_img) or not self._has_images(val_img):
            val_img = train_img

        train_lbl = self._guess_labels_dir(train_img, self.train_labels_dir)
        val_lbl = self._guess_labels_dir(val_img, self.val_labels_dir)

        out_root = os.path.join(self.output_dir, 'datasets', 'roi_cropped')
        # 清理旧裁剪结果，避免残留
        import shutil
        if os.path.isdir(out_root):
            shutil.rmtree(out_root, ignore_errors=True)

        train_out_img = os.path.join(out_root, 'images', 'train')
        train_out_lbl = os.path.join(out_root, 'labels', 'train')
        val_out_img = os.path.join(out_root, 'images', 'val')
        val_out_lbl = os.path.join(out_root, 'labels', 'val')

        n_train = self._crop_split_with_roi(
            train_img, train_lbl, train_out_img, train_out_lbl, x1, y1, rw, rh
        )
        if n_train <= 0:
            raise RuntimeError('ROI 裁剪后训练集为空，请检查图像目录与 ROI 设置')

        same_val = os.path.normpath(val_img) == os.path.normpath(train_img)
        if same_val:
            n_val = n_train
            val_out_img = train_out_img
            val_out_lbl = train_out_lbl
            self.log_update.emit('验证集与训练集相同，复用裁剪结果')
        else:
            n_val = self._crop_split_with_roi(
                val_img, val_lbl, val_out_img, val_out_lbl, x1, y1, rw, rh
            )
            if n_val <= 0:
                self.log_update.emit('ROI 裁剪后验证集为空，将使用训练集作为验证集')
                val_out_img = train_out_img
                val_out_lbl = train_out_lbl

        self.train_dir = train_out_img
        self.val_dir = val_out_img
        self.train_labels_dir = train_out_lbl
        self.val_labels_dir = val_out_lbl
        self.log_update.emit(
            f'ROI 裁剪完成: train={n_train} val={n_val if not same_val else n_train} → {out_root}'
        )

    def _update_paths_in_yaml(self, src_yaml, dst_yaml):
        """更新YAML文件中的路径以适应当前环境"""
        try:
            config = self._resolve_yolo_dataset_config()
            class_names = self._get_class_names()
            self._write_yolo_yaml(dst_yaml, config, class_names)
            self._validate_yaml(dst_yaml)
        except Exception as e:
            self.log_update.emit(f"更新YAML文件失败: {str(e)}，将创建新文件")
            config = self._resolve_yolo_dataset_config()
            class_names = self._get_class_names()
            self._write_yolo_yaml(dst_yaml, config, class_names)
            self._validate_yaml(dst_yaml)
    
    def _normalize_path(self, path):
        """规范化路径，确保路径格式正确"""
        # 替换多个连续的斜杠为单个斜杠
        path = path.replace('///', '/').replace('//', '/')
        
        # 确保Windows路径使用正确的斜杠格式
        if os.name == 'nt':
            path = path.replace('/', '\\')
            # 修复可能出现的多个反斜杠问题
            while '\\\\' in path:
                path = path.replace('\\\\', '\\')
        
        return path
    
    def _validate_yaml(self, yaml_path):
        """验证YAML文件中的路径是否有效"""
        import yaml
        try:
            # 读取YAML文件
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # 检查基础路径是否存在
            base_path = data.get('path', '')
            if not base_path or not os.path.exists(base_path):
                self.log_update.emit(f"警告: YAML中的基础路径不存在: {base_path}")
                
                # 尝试修复基础路径
                if 'train' in data:
                    train_rel = data['train']
                    possible_base = self._find_valid_base_path(train_rel)
                    if possible_base:
                        data['path'] = possible_base
                        self.log_update.emit(f"自动修复基础路径为: {possible_base}")
                        
                        # 重新写入YAML
                        with open(yaml_path, 'w') as f:
                            yaml.dump(data, f, default_flow_style=False)
            
            if 'path' in data and os.path.exists(data['path']):
                if 'train' in data:
                    train_path = self._yaml_split_path(data, 'train')
                    if not os.path.exists(train_path) or not self._has_images(train_path):
                        self.log_update.emit(f"警告: 训练路径无效: {train_path}")
                        alt = self._swap_labels_to_images(train_path)
                        if alt != train_path and os.path.isdir(alt) and self._has_images(alt):
                            dataset_root = self._find_common_parent(alt, alt)
                            data['path'] = dataset_root.replace('\\', '/')
                            data['train'] = os.path.relpath(alt, dataset_root).replace('\\', '/')
                            self.log_update.emit(f"自动修复训练路径: {data['train']}")

                if 'val' in data:
                    val_path = self._yaml_split_path(data, 'val')
                    if not os.path.exists(val_path) or not self._has_images(val_path):
                        self.log_update.emit(f"警告: 验证路径无效: {val_path}")
                        alt = self._swap_labels_to_images(val_path)
                        if alt != val_path and os.path.isdir(alt) and self._has_images(alt):
                            dataset_root = data.get('path', self._find_common_parent(alt, alt))
                            data['val'] = os.path.relpath(alt, dataset_root).replace('\\', '/')
                            self.log_update.emit(f"自动修复验证路径: {data['val']}")
                        elif 'train' in data:
                            data['val'] = data['train']
                            self.log_update.emit(f"自动设置验证集与训练集相同: {data['train']}")

                with open(yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        except Exception as e:
            self.log_update.emit(f"验证YAML文件失败: {str(e)}")
    
    def _find_valid_base_path(self, train_rel):
        """尝试找到有效的基础路径"""
        # 从当前工作目录开始
        cwd = os.getcwd()
        if os.path.exists(os.path.join(cwd, train_rel)):
            return cwd
        
        # 从训练目录的父级目录尝试
        train_dir = self._normalize_path(self.train_dir)
        parent_dir = os.path.dirname(train_dir)
        if os.path.exists(os.path.join(parent_dir, train_rel)):
            return parent_dir
        
        # 从训练目录本身尝试
        if os.path.basename(train_dir) == train_rel:
            return os.path.dirname(train_dir)
        
        return None
    
    def _get_class_names(self):
        """
        Extract class names from the dataset.
        
        Returns:
            list: List of class names
        """
        if self.dataset_format == "YOLO":
            # 对于YOLO格式，尝试从classes.txt或data.yaml文件中获取类名
            class_names = self._get_yolo_class_names()
        elif self.dataset_format == "COCO":
            # For COCO, we would parse the annotations JSON file
            class_names = self._get_coco_class_names()
        elif self.dataset_format == "VOC":
            # For VOC, we would look for the labels in the annotation XML files
            class_names = self._get_voc_class_names()
        else:
            class_names = ['class0', 'class1']
        
        # 如果没有找到类名，使用默认名称
        if not class_names:
            self.log_update.emit("警告: 无法确定类名，使用默认值")
            class_names = ['class0', 'class1']
        
        return class_names
    
    def _dataset_search_roots(self):
        """返回用于查找 classes.txt / data.yaml 的目录列表。"""
        roots = []
        for path in (self.train_dir, self.val_dir, self.train_labels_dir, self.val_labels_dir):
            if path and os.path.isdir(path):
                roots.append(path)
        train_dir = self._normalize_path(self.train_dir)
        if os.path.basename(train_dir) in ('train', 'val') and os.path.basename(os.path.dirname(train_dir)) == 'images':
            base = os.path.dirname(os.path.dirname(train_dir))
            roots.extend([
                base,
                os.path.join(base, 'labels', os.path.basename(train_dir)),
            ])
        deduped = []
        for root in roots:
            norm = os.path.normpath(root)
            if norm not in deduped:
                deduped.append(norm)
        return deduped

    def _get_yolo_class_names(self):
        """从YOLO格式数据集中提取类名"""
        class_names = []
        search_roots = self._dataset_search_roots()

        possible_yaml_paths = []
        possible_class_files = []
        for root in search_roots:
            possible_yaml_paths.extend([
                os.path.join(root, 'data.yaml'),
                os.path.join(os.path.dirname(root), 'data.yaml'),
                os.path.join(os.path.dirname(os.path.dirname(root)), 'data.yaml'),
            ])
            possible_class_files.extend([
                os.path.join(root, 'classes.txt'),
                os.path.join(os.path.dirname(root), 'classes.txt'),
                os.path.join(os.path.dirname(os.path.dirname(root)), 'classes.txt'),
            ])

        for yaml_path in dict.fromkeys(possible_yaml_paths):
            if os.path.exists(yaml_path):
                try:
                    import yaml
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    if data and 'names' in data:
                        names = data['names']
                        if isinstance(names, dict):
                            names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else x)]
                        self.log_update.emit(f"从YAML文件加载类名: {yaml_path}")
                        return list(names)
                except Exception as e:
                    self.log_update.emit(f"从YAML读取类名失败: {str(e)}")

        for class_file in dict.fromkeys(possible_class_files):
            if os.path.exists(class_file):
                try:
                    with open(class_file, 'r', encoding='utf-8') as f:
                        class_names = [line.strip() for line in f if line.strip()]
                    if class_names:
                        self.log_update.emit(f"从classes.txt加载类名: {class_file}")
                        return class_names
                except Exception as e:
                    self.log_update.emit(f"从classes.txt读取类名失败: {str(e)}")

        label_dirs = [p for p in (self.train_labels_dir, self.val_labels_dir) if p and os.path.isdir(p)]
        if not label_dirs:
            for root in search_roots:
                if root.replace('\\', '/').endswith('/labels') or '\\labels\\' in root:
                    label_dirs.append(root)
                elif os.path.basename(root) in ('train', 'val') and os.path.basename(os.path.dirname(root)) == 'labels':
                    label_dirs.append(root)

        try:
            txt_dirs = list(dict.fromkeys(label_dirs))
            for txt_dir in txt_dirs:
                if not os.path.isdir(txt_dir):
                    continue
                max_class_id = -1
                for name in os.listdir(txt_dir):
                    if not name.endswith('.txt'):
                        continue
                    with open(os.path.join(txt_dir, name), 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                max_class_id = max(max_class_id, int(float(parts[0])))
                if max_class_id >= 0:
                    self.log_update.emit(f"从标签目录推断类名: {txt_dir}")
                    return [f'class{i}' for i in range(max_class_id + 1)]
        except Exception as e:
            self.log_update.emit(f"从标签文件推断类名失败: {str(e)}")

        return []

    def _get_coco_class_names(self):
        """从COCO格式数据集中提取类名"""
        try:
            import json
            
            # Look for annotations file
            ann_file = None
            for file in os.listdir(self.train_dir):
                if file.endswith('.json') and ('annotations' in file or 'instances' in file):
                    ann_file = os.path.join(self.train_dir, file)
                    break
            
            if ann_file:
                with open(ann_file, 'r') as f:
                    coco_data = json.load(f)
                
                # Extract category names
                if 'categories' in coco_data:
                    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
                    class_names = [cat['name'] for cat in categories]
                    return class_names
        
        except Exception as e:
            self.log_update.emit(f"Error extracting COCO class names: {str(e)}")
        
        return []
    
    def _get_voc_class_names(self):
        """从VOC格式数据集中提取类名"""
        try:
            import xml.etree.ElementTree as ET
            
            # Get a list of all XML files
            xml_files = []
            for root, _, files in os.walk(self.train_dir):
                for file in files:
                    if file.endswith('.xml'):
                        xml_files.append(os.path.join(root, file))
            
            if not xml_files:
                self.log_update.emit("未找到VOC XML标注文件")
                return []
            
            # Extract unique class names from XML files
            class_names = set()
            for xml_file in xml_files:  # 解析全部 XML，避免漏类
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.findall('.//object'):
                    name = obj.find('name').text
                    class_names.add(name)
            
            return sorted(list(class_names))
        
        except Exception as e:
            self.log_update.emit(f"Error extracting VOC class names: {str(e)}")
        
        return []
    
    def _find_common_parent(self, path1, path2):
        """找到两个路径的共同父目录"""
        path1 = os.path.abspath(path1)
        path2 = os.path.abspath(path2)
        
        # 将路径拆分为组件
        parts1 = path1.split(os.sep)
        parts2 = path2.split(os.sep)
        
        # 找到共同的前缀
        common_parts = []
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                common_parts.append(p1)
            else:
                break
        
        # 构建共同父路径
        if common_parts:
            common_path = os.sep.join(common_parts)
            # 在Windows上，确保包含驱动器号
            if os.name == 'nt' and not common_path.endswith(':'):
                common_path += os.sep
            return common_path
        
        # 如果没有共同部分，返回根目录
        return os.path.dirname(path1) 