"""lop dung de train model yolo leafnet"""
import os
import logging
from typing import Dict, Any, Optional
import torch

from src.training.trainer import BaseTrainer
from src.architectures.detectors.yolo_leafnet import YOLOLeafNetDetector

logger = logging.getLogger("YOLOLeafNetTrainer")


class YOLOLeafNetTrainer(BaseTrainer):
    """lop trainer cho yolo leafnet"""

    def __init__(
        self,
        detector: YOLOLeafNetDetector,
        cfg_train: Dict[str, Any],
        data_path: Optional[str] = None
    ):
        """khoi tao trainer voi detector va config"""
        self.detector = detector
        self.cfg_train = cfg_train
        
        # xac dinh duong dan file dataset
        self.data_path = data_path or cfg_train.get("data", "data/dataset.yaml")
        if not self.data_path:
            raise ValueError(
                "Dataset configuration path (data) must be specified "
                "either in train.yaml or passed directly."
            )

    def train_one_epoch(self) -> Dict[str, float]:
        """train mot epoch (it khi dung vi ultralytics tu quan ly)"""
        logger.warning(
            "Calling train_one_epoch individually is discouraged with Ultralytics. "
            "Please use run() to execute the full optimized training loop."
        )
        # chay thu mot epoch de test
        results = self.detector.model.train(
            data=self.data_path,
            epochs=1,
            batch=self.cfg_train.get("batch_size", 16),
            lr0=self.cfg_train.get("lr", 0.01),
            optimizer=self.cfg_train.get("optimizer", "AdamW"),
            device=self.get_device_string(),
            verbose=False
        )
        # doc ket qua metric neu co
        metrics = {}
        if results and hasattr(results, "results_dict"):
            metrics = {k: float(v) for k, v in results.results_dict.items()}
        return metrics

    def validate(self) -> Dict[str, float]:
        """chay validate tinh cac metric map"""
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
        """chay toan bo chuong trinh training"""
        logger.info("Starting YOLO-LeafNet training process...")
        
        # chon optimizer
        optimizer = self.cfg_train.get("optimizer", "AdamW")
        
        # thiet lap scheduler
        cos_lr = self.cfg_train.get("scheduler", "cosine") == "cosine"
        
        checkpoint_dir = self.cfg_train.get("checkpoint_dir", "checkpoints/")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # tham so train co ban
        train_args = {
            "data": self.data_path,
            "epochs": self.cfg_train.get("epochs", 10),
            "batch": self.cfg_train.get("batch_size", 16),
            "lr0": self.cfg_train.get("lr", 0.01),
            "weight_decay": self.cfg_train.get("weight_decay", 0.0005),
            "optimizer": optimizer,
            "cos_lr": cos_lr,
            "patience": self.cfg_train.get("early_stopping_patience", 5),
            "project": checkpoint_dir,
            "name": "yolo_leafnet",
            "device": self.get_device_string(),
            "exist_ok": True,
            "val": True,
            "resume": self.cfg_train.get("resume", False)
        }

        # them cac tham so phu tu config
        for k, v in self.cfg_train.items():
            if k not in ["data", "epochs", "batch_size", "lr", "scheduler", "early_stopping_patience", "checkpoint_dir", "resume", "log_backend"]:
                train_args[k] = v

        # goi ham train cua ultralytics
        self.detector.model.train(**train_args)
        
        logger.info("Training process completed successfully.")

    def save_checkpoint(self, path: str) -> None:
        """luu lai trong so cua model"""
        logger.info(f"Saving checkpoint state dict to {path}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.detector.model.model.state_dict(), path)

    def get_device_string(self) -> str:
        """lay thong tin thiet bi gpu hoac cpu"""
        if torch.cuda.is_available():
            return "0"  # dung gpu dau tien
        return "cpu"
