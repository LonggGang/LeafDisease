"""
Concrete implementation of the evaluation pipeline for YOLO-LeafNet.
Wraps the Ultralytics YOLO validation execution to calculate standard object detection metrics.
"""
import logging
from typing import Dict, Any
import torch

from src.training.evaluator import BaseEvaluator
from src.architectures.detectors.yolo_leafnet import YOLOLeafNetDetector

logger = logging.getLogger("YOLOLeafNetEvaluator")


class YOLOLeafNetEvaluator(BaseEvaluator):
    """
    Evaluator class for YOLO-LeafNet.
    Runs validation on a given dataset split and collects standard detection metrics.
    """

    def __init__(self, detector: YOLOLeafNetDetector, data_path: str, split: str = "val"):
        """
        Args:
            detector: An instance of YOLOLeafNetDetector.
            data_path: Path to dataset.yaml.
            split: Dataset split to evaluate on ('val' or 'test').
        """
        self.detector = detector
        self.data_path = data_path
        self.split = split

    def evaluate(self) -> Dict[str, Any]:
        """
        Computes evaluation metrics including mAP, speed (inference time), and complexity.
        
        Returns:
            A dictionary containing:
                - mAP50: float (mean Average Precision at IoU 0.5)
                - mAP50_95: float (mean Average Precision at IoU 0.5:0.95)
                - preprocess_ms: float (preprocessing time per image in ms)
                - inference_ms: float (raw model forward pass time per image in ms)
                - postprocess_ms: float (NMS and postprocessing time per image in ms)
                - params_M: float (parameter count in millions)
                - flops_G: float (floating point operations in billions)
        """
        logger.info(f"Evaluating model on split '{self.split}' using dataset {self.data_path}...")
        
        device = "0" if torch.cuda.is_available() else "cpu"
        
        # Run validation using Ultralytics
        results = self.detector.model.val(
            data=self.data_path,
            split=self.split,
            device=device,
            verbose=False
        )
        
        # Extract complexity metrics
        complexity = self.detector.get_complexity()
        
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "postprocess_ms": 0.0,
            "params_M": complexity.get("params_M", 0.0),
            "flops_G": complexity.get("flops_G", 0.0)
        }
        
        if results is not None:
            # Ultralytics results.results_dict contains metrics keys:
            # metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
            res_dict = results.results_dict
            precision = float(res_dict.get("metrics/precision(B)", 0.0))
            recall = float(res_dict.get("metrics/recall(B)", 0.0))
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1
            metrics["mAP50"] = float(res_dict.get("metrics/mAP50(B)", 0.0))
            metrics["mAP50_95"] = float(res_dict.get("metrics/mAP50-95(B)", 0.0))
            
            # Extract speed / latencies (in milliseconds per image)
            if hasattr(results, "speed") and isinstance(results.speed, dict):
                metrics["preprocess_ms"] = float(results.speed.get("preprocess", 0.0))
                metrics["inference_ms"] = float(results.speed.get("inference", 0.0))
                metrics["postprocess_ms"] = float(results.speed.get("postprocess", 0.0))
                
        logger.info(f"Evaluation completed. Results: {metrics}")
        return metrics
