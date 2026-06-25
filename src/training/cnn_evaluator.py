"""lop dung de evaluate model cnn phan loai"""
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
    """lop evaluator cho model cnn phan loai"""

    def __init__(self, model: nn.Module, data_path: str, split: str = "val"):
        """khoi tao evaluator voi model va split datapath"""
        self.model = model
        self.data_path = data_path
        self.split = split

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # tao dataloader
        self.dataloader = self._build_dataloader()
        self.criterion = nn.CrossEntropyLoss()

    def _build_dataloader(self) -> DataLoader:
        """tao dataloader doc du lieu de evaluate"""
        # xac dinh thu muc du lieu con
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

        # tao phep bien doi anh
        dataset_type = getattr(self.model, "dataset_type", "PlantDoc")
        transform_factory = PlantDiseaseTransform(dataset_type=dataset_type, task="classification")
        val_transform = transform_factory.build_val_transforms()

        # tao dataset
        dataset = PlantDiseaseDataset(subset_dir, transform=val_transform)
        
        return DataLoader(
            dataset,
            batch_size=1,  # batch size 1 de do thoi gian chuan
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    def evaluate(self) -> Dict[str, Any]:
        """danh gia model va lay cac thong so thoi gian va dung luong"""
        import os
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        total_time = 0.0

        logger.info(f"Evaluating classifier on split '{self.split}'...")

        with torch.no_grad():
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                # do thoi gian chay forward
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

        # lay do phuc tap cua model
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
