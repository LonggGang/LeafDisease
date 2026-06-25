"""detector yolo de phat hien benh la"""
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
    C2f = nn.Module
    A2C2f = nn.Module

from src.architectures.detectors.base_detector import BaseDetector

logger = logging.getLogger("YOLOLeafNet")


class C2f_BNDropout(nn.Module):
    """khoi c2f kem theo batch norm va dropout"""
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
    """khoi a2c2f kem theo batch norm va dropout"""
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
    """lop phat hien doi tuong yolo leafnet"""

    def __init__(self, cfg: Dict[str, Any]):
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

        # dang ky module tu dinh nghia
        self._register_custom_modules()

        # dung model yolo
        self.model = self._build_yolo_model()

    def _register_custom_modules(self) -> None:
        """dang ky module vao ultralytics"""
        logger.info("Registering custom C2f_BNDropout and A2C2f_BNDropout modules in ultralytics tasks namespace.")
        ultralytics.nn.tasks.C2f_BNDropout = C2f_BNDropout
        ultralytics.nn.tasks.A2C2f_BNDropout = A2C2f_BNDropout

    def _build_yolo_model(self) -> YOLO:
        """tao model yolo tu config"""
        arch = self.cfg.get("architecture", "yolo_leafnet")
        
        if arch not in ["yolo_leafnet", "yolo_leafnet_v8"]:
            logger.info(f"Building standard YOLO model: {arch}")
            # sua lai ten file weights
            if not (arch.endswith(".yaml") or arch.endswith(".pt")):
                # check load weights nao
                model_str = f"{arch}.pt" if self.pretrained else f"{arch}.yaml"
            else:
                model_str = arch
                
            model = YOLO(model_str)
            return model

        # logic cho yolo leafnet tu dinh nghia
        # doc config co san
        if arch == "yolo_leafnet_v8":
            arch_config_path = "configs/yolo_leafnet_v8.yaml"
            default_weights = "yolov8s.pt"
        else:
            arch_config_path = "configs/yolo_leafnet.yaml"
            default_weights = "yolov12s.pt"

        if not os.path.exists(arch_config_path):
            raise FileNotFoundError(f"Base architecture config not found: {arch_config_path}")

        with open(arch_config_path, "r", encoding="utf-8") as f:
            yolo_cfg = yaml.safe_load(f)

        # ghi de so class
        yolo_cfg["nc"] = self.num_classes

        # tao folder checkpoints
        os.makedirs("checkpoints", exist_ok=True)
        config_path = "checkpoints/yolo_leafnet_config.yaml"
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(yolo_cfg, f)
            
        logger.info(f"Wrote configuration to {config_path}")

        # khoi tao model
        model = YOLO(config_path)

        # load weights pretrained neu can
        if self.pretrained:
            try:
                logger.info(f"Loading pretrained weights from {default_weights}...")
                standard_model = YOLO(default_weights)
                pretrained_dict = standard_model.model.state_dict()
                
                # loc parameters bi lech size
                model_dict = model.model.state_dict()
                filtered_dict = {
                    k: v for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                model.model.load_state_dict(filtered_dict, strict=False)
                logger.info(f"Successfully loaded pretrained {default_weights} weights (excluding mismatched head layers).")
            except Exception as e:
                logger.warning(
                    f"Could not load pretrained weights from {default_weights}: {e}. "
                    "Falling back to random weights initialization."
                )

        # thay the layer 8 bang khoi tu dinh nghia
        standard_layer = model.model.model[8]
        
        # kiem tra kieu layer
        is_a2c2f = False
        if hasattr(standard_layer, "cv1") and hasattr(standard_layer, "cv2") and hasattr(standard_layer, "m"):
            c1 = standard_layer.cv1.conv.in_channels
            c2 = standard_layer.cv2.conv.out_channels
            n = len(standard_layer.m)
            
            # kiem tra co phai a2c2f khong
            is_a2c2f = isinstance(standard_layer.m[0], nn.Sequential) if len(standard_layer.m) > 0 else False
            
            if is_a2c2f:
                # lay cac tham so cua a2c2f
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
                # backup neu la c2f
                shortcut = standard_layer.m[0].add if len(standard_layer.m) > 0 else False
                custom_layer = C2f_BNDropout(c1, c2, n=n, shortcut=shortcut, p=self.dropout_rate)
                custom_layer_type = "src.architectures.detectors.yolo_leafnet.C2f_BNDropout"
        else:
            raise ValueError(f"Unexpected module type for layer 8: {type(standard_layer)}")

        # copy weights tu module cu sang module moi
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
                
        # copy metadata cho parser cua yolo
        custom_layer.i = standard_layer.i
        custom_layer.f = standard_layer.f
        custom_layer.type = custom_layer_type
        
        # thay the module vao model
        model.model.model[8] = custom_layer
        logger.info(f"Successfully injected custom layer at backbone layer 8.")

        return model

    def forward(self, x: torch.Tensor) -> Any:
        """chay forward qua mang neural"""
        return self.model.model(x)

    def apply_nms(self, predictions: Any, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Any:
        """ap dung nms de loc hop"""
        return non_max_suppression(predictions, conf_thres=conf_threshold, iou_thres=iou_threshold)

    def decode_boxes(self, raw_boxes: Any) -> Any:
        """chuyen hop tu xywh sang xyxy"""
        return xywh2xyxy(raw_boxes)

    def filter_by_confidence(self, boxes: Any, conf_threshold: float) -> Any:
        """loc cac hop duoi nguong tin cay"""
        if isinstance(boxes, torch.Tensor) and boxes.ndim == 2:
            return boxes[boxes[:, 4] >= conf_threshold]
        return boxes

    def predict(self, image_path: str) -> Dict[str, Any]:
        """du doan anh tu duong dan"""
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
                xyxy = box.xyxy[0].tolist()
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
        """lay do phuc tap cua model"""
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
