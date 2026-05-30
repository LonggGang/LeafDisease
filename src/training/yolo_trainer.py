"""
Concrete implementation of the training loop for YOLO-LeafNet.
Wraps the Ultralytics YOLO training execution to maintain high performance.
"""
import os
import logging
from typing import Dict, Any, Optional
import torch

from src.training.trainer import BaseTrainer
from src.architectures.detectors.yolo_leafnet import YOLOLeafNetDetector

logger = logging.getLogger("YOLOLeafNetTrainer")


class YOLOLeafNetTrainer(BaseTrainer):
    """
    Trainer class for YOLO-LeafNet.
    Delegates core training execution to the Ultralytics engine.
    """

    def __init__(
        self,
        detector: YOLOLeafNetDetector,
        cfg_train: Dict[str, Any],
        data_path: Optional[str] = None
    ):
        """
        Args:
            detector: An instance of YOLOLeafNetDetector.
            cfg_train: Training hyperparameters dictionary.
            data_path: Optional path to dataset.yaml. If not provided, will read from cfg_train.
        """
        self.detector = detector
        self.cfg_train = cfg_train
        
        # Determine dataset configuration file
        self.data_path = data_path or cfg_train.get("data", "data/dataset.yaml")
        if not self.data_path:
            raise ValueError(
                "Dataset configuration path (data) must be specified "
                "either in train.yaml or passed directly."
            )

    def train_one_epoch(self) -> Dict[str, float]:
        """
        Standard PyTorch train one epoch.
        Note: Ultralytics manages its own internal epoch loop for efficiency.
        Calling this separately is not recommended as it incurs setup overhead.
        """
        logger.warning(
            "Calling train_one_epoch individually is discouraged with Ultralytics. "
            "Please use run() to execute the full optimized training loop."
        )
        # Fallback dry-run of a single epoch
        results = self.detector.model.train(
            data=self.data_path,
            epochs=1,
            batch=self.cfg_train.get("batch_size", 16),
            lr0=self.cfg_train.get("lr", 0.01),
            optimizer=self.cfg_train.get("optimizer", "AdamW"),
            device=self.get_device_string(),
            verbose=False
        )
        # Parse basic metrics if available
        metrics = {}
        if results and hasattr(results, "results_dict"):
            metrics = {k: float(v) for k, v in results.results_dict.items()}
        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Runs validation and returns performance metrics (mAP, etc.).
        """
        logger.info("Running validation loop...")
        results = self.detector.model.val(
            data=self.data_path,
            device=self.get_device_string(),
            verbose=False
        )
        
        metrics = {}
        if results and hasattr(results, "results_dict"):
            metrics = {k: float(v) for k, v in results.results_dict.items()}
            logger.info(f"Validation metrics: {metrics}")
        return metrics

    def run(self) -> None:
        """
        Runs the full optimized training process.
        """
        logger.info("Starting YOLO-LeafNet training process...")
        
        # Map optimizer names to format expected by Ultralytics
        optimizer = self.cfg_train.get("optimizer", "AdamW")
        
        # Determine scheduler settings
        cos_lr = self.cfg_train.get("scheduler", "cosine") == "cosine"
        
        checkpoint_dir = self.cfg_train.get("checkpoint_dir", "checkpoints/")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Call Ultralytics native train method
        # This automatically handles AMP, DFL loss, Box loss, Class loss, checkpointers, and loggers.
        self.detector.model.train(
            data=self.data_path,
            epochs=self.cfg_train.get("epochs", 10),
            batch=self.cfg_train.get("batch_size", 16),
            lr0=self.cfg_train.get("lr", 0.01),
            weight_decay=self.cfg_train.get("weight_decay", 0.0005),
            optimizer=optimizer,
            cos_lr=cos_lr,
            patience=self.cfg_train.get("early_stopping_patience", 5),
            project=checkpoint_dir,
            name="yolo_leafnet",
            device=self.get_device_string(),
            exist_ok=True,
            val=True
        )
        
        logger.info("Training process completed successfully.")

    def save_checkpoint(self, path: str) -> None:
        """
        Saves the model state dict to path.
        """
        logger.info(f"Saving checkpoint state dict to {path}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.detector.model.model.state_dict(), path)

    def get_device_string(self) -> str:
        """Helper to get PyTorch device string."""
        if torch.cuda.is_available():
            return "0"  # Use first GPU
        return "cpu"
