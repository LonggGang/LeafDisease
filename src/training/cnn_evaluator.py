"""
Concrete implementation of the evaluation pipeline for CNN plant disease classifiers.
Computes standard classification metrics, latency, and complexity.
"""
import os
import logging
import time
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.evaluator import BaseEvaluator
from src.dataprocessing.dataset import PlantDiseaseDataset
from src.dataprocessing.augmentation import PlantDiseaseTransform

logger = logging.getLogger("CNNClassifierEvaluator")


class CNNClassifierEvaluator(BaseEvaluator):
    """
    Evaluator class for PyTorch CNN Classifiers.
    Runs validation on a given dataset and collects accuracy and latency metrics.
    """

    def __init__(self, model: nn.Module, data_path: str, split: str = "val"):
        """
        Args:
            model: PyTorch classification model.
            data_path: Path to dataset folder (containing class folders).
            split: Dataset split to evaluate on.
        """
        self.model = model
        self.data_path = data_path
        self.split = split

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Build dataloader
        self.dataloader = self._build_dataloader()
        self.criterion = nn.CrossEntropyLoss()

    def _build_dataloader(self) -> DataLoader:
        """Prepares DataLoader for evaluation."""
        # Determine subset dir
        subset_dir = self.data_path
        if self.split in ["val", "test", "train"]:
            candidates = [self.split]
            if self.split == "val":
                candidates.extend(["validation", "valid"])
            
            for cand in candidates:
                candidate_dir = os.path.join(self.data_path, cand)
                if os.path.isdir(candidate_dir):
                    subset_dir = candidate_dir
                    break

        # Build transform
        dataset_type = getattr(self.model, "dataset_type", "PlantDoc")
        transform_factory = PlantDiseaseTransform(dataset_type=dataset_type, task="classification")
        val_transform = transform_factory.build_val_transforms()

        # Build dataset
        dataset = PlantDiseaseDataset(subset_dir, transform=val_transform)
        
        return DataLoader(
            dataset,
            batch_size=1,  # Batch size 1 for exact latency measurement
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    def evaluate(self) -> Dict[str, Any]:
        """
        Computes accuracy, loss, latency, and complexity metrics.
        
        Returns:
            A dictionary containing:
                - accuracy: float
                - val_loss: float
                - inference_ms: float (average raw forward pass latency per image)
                - params_M: float (total model parameters in millions)
                - trainable_params_M: float (trainable model parameters in millions)
        """
        import os  # Ensure os is imported inside helper
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        total_time = 0.0

        logger.info(f"Evaluating classifier on split '{self.split}'...")

        with torch.no_grad():
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Time the forward pass
                t0 = time.perf_counter()
                outputs = self.model(images)
                total_time += (time.perf_counter() - t0)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        val_loss = running_loss / total if total > 0 else 0.0
        avg_latency_ms = (total_time / total) * 1000.0 if total > 0 else 0.0

        # Get complexity
        complexity = {}
        if hasattr(self.model, "get_complexity"):
            complexity = self.model.get_complexity()
        else:
            total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            complexity = {"params_M": total_params, "trainable_params_M": total_params}

        results = {
            "accuracy": accuracy,
            "val_loss": val_loss,
            "inference_ms": avg_latency_ms,
            "params_M": complexity.get("params_M", 0.0),
            "trainable_params_M": complexity.get("trainable_params_M", 0.0)
        }
        
        logger.info(f"Evaluation results: {results}")
        return results
