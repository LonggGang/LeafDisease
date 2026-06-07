"""
YOLO-LeafNet detector implementation.
Inherits from BaseDetector and encapsulates the customized YOLOv8s model.
"""
import os
import time
import yaml
import logging
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

try:
    import ultralytics
    import ultralytics.nn.tasks
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2f, A2C2f
    from ultralytics.utils.ops import xywh2xyxy
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.torch_utils import get_flops
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    C2f = nn.Module  # fallback stub for definition
    A2C2f = nn.Module

from src.architectures.detectors.base_detector import BaseDetector

logger = logging.getLogger("YOLOLeafNet")


class C2f_BNDropout(nn.Module):
    """
    Custom C2f block appended with Batch Normalization and Dropout.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, p: float = 0.5, g: int = 1, e: float = 0.5):
        super().__init__()
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics library is required to instantiate C2f_BNDropout.")
        self.c2f = C2f(c1, c2, n, shortcut, g, e)
        self.bn = nn.BatchNorm2d(c2)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c2f(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class A2C2f_BNDropout(nn.Module):
    """
    Custom A2C2f block appended with Batch Normalization and Dropout.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1, residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5, g: int = 1, shortcut: bool = True, p: float = 0.5):
        super().__init__()
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics library is required to instantiate A2C2f_BNDropout.")
        self.a2c2f = A2C2f(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.bn = nn.BatchNorm2d(c2)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a2c2f(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class YOLOLeafNetDetector(BaseDetector):
    """
    YOLO-LeafNet Object Detector.
    Wraps the Ultralytics YOLOv8 library and modifies the backbone
    to include a Batch Normalization and a Dropout layer before SPPF.
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        Args:
            cfg: Configuration dictionary containing:
                - num_classes: int
                - input_size: int
                - pretrained: bool
                - dropout_rate: float
        """
        super().__init__()
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Failed to import 'ultralytics'. Please install it using "
                "'pip install ultralytics' before running YOLO-LeafNet."
            )

        self.cfg = cfg
        self.num_classes = cfg.get("num_classes", 11)
        self.input_size = cfg.get("input_size", 640)
        self.pretrained = cfg.get("pretrained", True)
        self.dropout_rate = cfg.get("dropout_rate", 0.5)

        # 1. Register custom module in ultralytics tasks namespace (optional backup)
        self._register_custom_modules()

        # 2. Build the YOLO model structure
        self.model = self._build_yolo_model()

    def _register_custom_modules(self) -> None:
        """Registers C2f_BNDropout and A2C2f_BNDropout in ultralytics tasks namespace."""
        logger.info("Registering custom C2f_BNDropout and A2C2f_BNDropout modules in ultralytics tasks namespace.")
        ultralytics.nn.tasks.C2f_BNDropout = C2f_BNDropout
        ultralytics.nn.tasks.A2C2f_BNDropout = A2C2f_BNDropout

    def _build_yolo_model(self) -> YOLO:
        """
        Dynamically configures and builds the YOLOv12s model.
        If architecture is 'yolo_leafnet', it builds a customized YOLOv12s backbone
        and injects custom A2C2f_BNDropout at layer 8.
        Otherwise, it instantiates standard YOLO models (e.g., yolov12s, yolov8s, yolov8n).
        """
        arch = self.cfg.get("architecture", "yolo_leafnet")
        
        if arch != "yolo_leafnet":
            logger.info(f"Building standard YOLO model: {arch}")
            # Ensure model weight file name matches format expected by Ultralytics
            if not (arch.endswith(".yaml") or arch.endswith(".pt")):
                # Check if we should load pretrained or random weights config
                model_str = f"{arch}.pt" if self.pretrained else f"{arch}.yaml"
            else:
                model_str = arch
                
            model = YOLO(model_str)
            return model

        # Logic for customized yolo_leafnet
        # Read the template yolo_leafnet configuration
        arch_config_path = "configs/yolo_leafnet.yaml"
        if not os.path.exists(arch_config_path):
            raise FileNotFoundError(f"Base architecture config not found: {arch_config_path}")

        with open(arch_config_path, "r", encoding="utf-8") as f:
            yolo_cfg = yaml.safe_load(f)

        # Override class count
        yolo_cfg["nc"] = self.num_classes

        # Create checkpoints dir if not exists
        os.makedirs("checkpoints", exist_ok=True)
        config_path = "checkpoints/yolo_leafnet_config.yaml"
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(yolo_cfg, f)
            
        logger.info(f"Wrote configuration to {config_path}")

        # Initialize the model structure
        model = YOLO(config_path)

        # Load pretrained weights from yolov12s.pt if requested (prior to replacing layer 8)
        if self.pretrained:
            try:
                logger.info("Loading pretrained weights from yolov12s.pt...")
                standard_model = YOLO("yolov12s.pt")
                pretrained_dict = standard_model.model.state_dict()
                
                # Filter out parameters with size mismatch
                model_dict = model.model.state_dict()
                filtered_dict = {
                    k: v for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                model.model.load_state_dict(filtered_dict, strict=False)
                logger.info("Successfully loaded pretrained yolov12s weights (excluding mismatched head layers).")
            except Exception as e:
                logger.warning(
                    f"Could not load pretrained weights from yolov12s.pt: {e}. "
                    "Falling back to random weights initialization."
                )

        # Now, replace backbone layer 8 with our custom layer in PyTorch model
        standard_layer = model.model.model[8]
        
        # Check standard_layer type
        is_a2c2f = False
        if hasattr(standard_layer, "cv1") and hasattr(standard_layer, "cv2") and hasattr(standard_layer, "m"):
            c1 = standard_layer.cv1.conv.in_channels
            c2 = standard_layer.cv2.conv.out_channels
            n = len(standard_layer.m)
            
            # Identify if it is A2C2f
            is_a2c2f = isinstance(standard_layer.m[0], nn.Sequential) if len(standard_layer.m) > 0 else False
            
            if is_a2c2f:
                # Extract A2C2f parameters
                first_ablock = standard_layer.m[0][0]
                area = first_ablock.attn.area
                mlp_ratio = float(first_ablock.mlp[0].conv.out_channels) / first_ablock.mlp[0].conv.in_channels
                residual = standard_layer.gamma is not None
                e = float(standard_layer.cv1.conv.out_channels) / c2
                
                custom_layer = A2C2f_BNDropout(
                    c1, c2, n=n, a2=True, area=area, residual=residual,
                    mlp_ratio=mlp_ratio, e=e, p=self.dropout_rate
                )
                custom_layer_type = "src.architectures.detectors.yolo_leafnet.A2C2f_BNDropout"
            else:
                # fallback/backward compatibility with C2f if someone passes YOLOv8 structure
                shortcut = standard_layer.m[0].add if len(standard_layer.m) > 0 else False
                custom_layer = C2f_BNDropout(c1, c2, n=n, shortcut=shortcut, p=self.dropout_rate)
                custom_layer_type = "src.architectures.detectors.yolo_leafnet.C2f_BNDropout"
        else:
            raise ValueError(f"Unexpected module type for layer 8: {type(standard_layer)}")

        # Transfer pretrained weights from standard module to our nested module
        if self.pretrained:
            try:
                if is_a2c2f:
                    custom_layer.a2c2f.load_state_dict(standard_layer.state_dict())
                    logger.info("Transferred pretrained weights from standard A2C2f to A2C2f_BNDropout.a2c2f.")
                else:
                    custom_layer.c2f.load_state_dict(standard_layer.state_dict())
                    logger.info("Transferred pretrained weights from standard C2f to C2f_BNDropout.c2f.")
            except Exception as e:
                logger.warning(f"Could not transfer weights for layer 8: {e}")
                
        # Copy index and metadata properties required by the parser
        custom_layer.i = standard_layer.i
        custom_layer.f = standard_layer.f
        custom_layer.type = custom_layer_type
        
        # Replace the module in PyTorch DetectionModel's Sequential container
        model.model.model[8] = custom_layer
        logger.info(f"Successfully injected custom layer at backbone layer 8.")

        return model

    def forward(self, x: torch.Tensor) -> Any:
        """Standard PyTorch forward pass."""
        return self.model.model(x)

    def apply_nms(self, predictions: Any, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Any:
        """Applies Non-Maximum Suppression to filter bounding boxes."""
        return non_max_suppression(predictions, conf_thres=conf_threshold, iou_thres=iou_threshold)

    def decode_boxes(self, raw_boxes: Any) -> Any:
        """Decodes model-specific box outputs from xywh to xyxy format."""
        return xywh2xyxy(raw_boxes)

    def filter_by_confidence(self, boxes: Any, conf_threshold: float) -> Any:
        """Filters out boxes below the given confidence threshold."""
        if isinstance(boxes, torch.Tensor) and boxes.ndim == 2:
            return boxes[boxes[:, 4] >= conf_threshold]
        return boxes

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        End-to-end inference from a raw image file path.
        Returns:
            {
                "label": str,
                "confidence": float,
                "boxes": list[dict],
                "inference_ms": float
            }
        """
        start_time = time.time()
        
        results = self.model.predict(
            source=image_path,
            imgsz=self.input_size,
            conf=0.25,
            verbose=False
        )
        
        elapsed_ms = (time.time() - start_time) * 1000.0
        
        if not results or len(results) == 0:
            return {
                "label": "healthy",
                "confidence": 0.0,
                "boxes": [],
                "inference_ms": elapsed_ms
            }

        result = results[0]
        boxes_list = []
        
        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                
                boxes_list.append({
                    "box": xyxy,
                    "label": cls_name,
                    "confidence": conf
                })

        main_label = "healthy"
        main_conf = 0.0
        if len(boxes_list) > 0:
            best_box = max(boxes_list, key=lambda x: x["confidence"])
            main_label = best_box["label"]
            main_conf = best_box["confidence"]

        if hasattr(result, "speed") and isinstance(result.speed, dict):
            speed_dict = result.speed
            inference_ms = speed_dict.get("preprocess", 0.0) + speed_dict.get("inference", 0.0) + speed_dict.get("postprocess", 0.0)
        else:
            inference_ms = elapsed_ms

        return {
            "label": main_label,
            "confidence": main_conf,
            "boxes": boxes_list,
            "inference_ms": inference_ms
        }

    def get_complexity(self) -> Dict[str, float]:
        """
        Returns model complexity.
        Returns:
            {
                "params_M": float,
                "flops_G": float
            }
        """
        params_M = sum(p.numel() for p in self.model.model.parameters()) / 1e6

        flops_G = 0.0
        try:
            flops_G = float(get_flops(self.model.model, imgsz=self.input_size))
        except Exception as e:
            logger.warning(f"Could not calculate FLOPs: {e}")

        return {
            "params_M": params_M,
            "flops_G": flops_G
        }
