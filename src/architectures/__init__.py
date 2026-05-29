"""
Architectures package containing models and components.
"""
from typing import Dict, Any
from src.architectures.base import BaseModel

def build_model(cfg: Dict[str, Any]) -> BaseModel:
    """
    Factory function to construct and build neural network architectures.
    
    Args:
        cfg: Model configuration dictionary containing task type, architecture name, etc.
        
    Returns:
        An instance of BaseModel (either a classifier or detector).
    """
    task = cfg.get("task", "classification")
    arch = cfg.get("architecture", "")

    if task == "detection":
        if arch == "yolo_leafnet":
            from src.architectures.detectors.yolo_leafnet import YOLOLeafNetDetector
            return YOLOLeafNetDetector(cfg)
        else:
            raise ValueError(f"Unknown detection architecture: {arch}")
    elif task == "classification":
        raise NotImplementedError("Classification architectures are not implemented yet in the factory.")
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'classification' or 'detection'.")
