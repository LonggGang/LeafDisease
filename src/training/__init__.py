"""
Training package containing trainers and evaluators.
"""
from src.training.trainer import BaseTrainer
from src.training.evaluator import BaseEvaluator
from src.training.yolo_trainer import YOLOLeafNetTrainer
from src.training.yolo_evaluator import YOLOLeafNetEvaluator
from src.training.cnn_trainer import CNNClassifierTrainer
from src.training.cnn_evaluator import CNNClassifierEvaluator

__all__ = [
    "BaseTrainer",
    "BaseEvaluator",
    "YOLOLeafNetTrainer",
    "YOLOLeafNetEvaluator",
    "CNNClassifierTrainer",
    "CNNClassifierEvaluator",
]
