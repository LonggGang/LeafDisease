"""khoi tao model va component"""
from typing import Dict, Any
from src.architectures.base import BaseModel

def build_model(cfg: Dict[str, Any]) -> BaseModel:
    """ham khoi tao cac loai model"""
    task = cfg.get("task", "classification")
    arch = cfg.get("architecture", "")

    if task == "detection":
        # ho tro yolo cac thu
        if arch == "yolo_leafnet" or "yolo" in arch.lower():
            from src.architectures.detectors.yolo_leafnet import YOLOLeafNetDetector
            return YOLOLeafNetDetector(cfg)
        else:
            raise ValueError(f"Unknown detection architecture: {arch}")
    elif task == "classification":
        if arch == "advanced_cnn":
            from src.architectures.classifiers.advanced_cnn_classifier import _register_classifier
            return _register_classifier(cfg)
        elif arch == "v2plantnet":
            from src.architectures.classifiers.v2plantnet_classifier import _register_classifier
            return _register_classifier(cfg)
        else:
            raise ValueError(f"Unknown classification architecture: {arch}")
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'classification' or 'detection'.")
