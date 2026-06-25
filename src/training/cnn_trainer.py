"""lop trainer cho cac model phan loai cnn"""
import os
import logging
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.training.trainer import BaseTrainer
from src.dataprocessing.dataset import PlantDiseaseDataset
from src.dataprocessing.augmentation import PlantDiseaseTransform
from src.dataprocessing.sampler import get_balanced_sampler

logger = logging.getLogger("CNNClassifierTrainer")


class CNNClassifierTrainer(BaseTrainer):
    """lop trainer de train model cnn phan loai"""

    def __init__(
        self,
        model: nn.Module,
        cfg_train: Dict[str, Any],
        data_path: str,
        cfg_aug: Optional[Dict[str, Any]] = None
    ):
        """khoi tao trainer voi model va configs"""
        self.model = model
        self.cfg_train = cfg_train
        self.data_path = data_path
        self.cfg_aug = cfg_aug or {}

        # dat thiet bi cuda hoac cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # tao dataloader
        self.train_loader, self.val_loader = self._build_dataloaders()

        # tao loss, optimizer va scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # theo doi accuracy cao nhat
        self.best_acc = 0.0

    def _build_dataloaders(self):
        """tao bo doc du lieu dataloader"""
        batch_size = self.cfg_train.get("batch_size", 32)
        
        # xac dinh thu muc du lieu
        train_dir = os.path.join(self.data_path, "train")
        val_dir = os.path.join(self.data_path, "val")
        if not os.path.isdir(val_dir):
            val_dir = os.path.join(self.data_path, "validation")
        if not os.path.isdir(val_dir):
            val_dir = os.path.join(self.data_path, "valid")

        # tao cac phep bien doi anh
        dataset_type = self.cfg_train.get("dataset_type", "PlantDoc")
        transform_factory = PlantDiseaseTransform(dataset_type=dataset_type, task="classification")
        train_transform = transform_factory.build_train_transforms()
        val_transform = transform_factory.build_val_transforms()

        # kiem tra cau truc thu muc
        if os.path.isdir(train_dir):
            logger.info(f"Loading train dataset from {train_dir}")
            train_dataset = PlantDiseaseDataset(train_dir, transform=train_transform)
            
            if os.path.isdir(val_dir):
                logger.info(f"Loading validation dataset from {val_dir}")
                val_dataset = PlantDiseaseDataset(val_dir, transform=val_transform)
            else:
                logger.info("Validation directory not found. Splitting training dataset 80/20.")
                train_len = int(0.8 * len(train_dataset))
                val_len = len(train_dataset) - train_len
                # chia nho dataset ra
                train_subset, val_subset = random_split(train_dataset, [train_len, val_len])
                
                # class wrapper de lay dung transform
                class SubsetWithTransform(torch.utils.data.Dataset):
                    def __init__(self, subset, transform):
                        self.subset = subset
                        self.transform = transform
                    def __getitem__(self, index):
                        x, y = self.subset[index]
                        # lay anh va nhan
                        return x, y
                    def __len__(self):
                        return len(self.subset)
                
                train_dataset = train_subset
                val_dataset = val_subset
        else:
            # neu data_path la thu muc goc
            logger.info(f"Loading dataset directly from {self.data_path}")
            full_dataset = PlantDiseaseDataset(self.data_path, transform=train_transform)
            
            train_len = int(0.8 * len(full_dataset))
            val_len = len(full_dataset) - train_len
            train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

        # tao sampler can bang class
        sampler = None
        if hasattr(train_dataset, "samples") and hasattr(train_dataset, "classes"):
            try:
                sampler = get_balanced_sampler(train_dataset)
                logger.info("Applying Class-Balanced Weighted Random Sampler.")
            except Exception as e:
                logger.warning(f"Could not apply balanced sampler: {e}. Falling back to standard shuffling.")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4 if os.name != 'nt' else 0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if os.name != 'nt' else 0,
            pin_memory=True
        )

        # log ra so luong mau
        logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        return train_loader, val_loader

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """tao bo toi uu hoa optimizer"""
        opt_name = self.cfg_train.get("optimizer", "AdamW").upper()
        lr = self.cfg_train.get("lr", 0.001)
        wd = self.cfg_train.get("weight_decay", 0.0005)

        # kiem tra backbone de set lr rieng
        backbone_trainable = False
        if hasattr(self.model, "backbone"):
            backbone_trainable = any(p.requires_grad for p in self.model.backbone.parameters())

        if backbone_trainable:
            unfreeze_lr = self.cfg_train.get("unfreeze_lr", lr / 10.0)
            logger.info("Building optimizer")
            
            backbone_params = []
            head_params = []
            
            backbone_param_ids = set(id(p) for p in self.model.backbone.parameters())
            
            for p in self.model.parameters():
                if p.requires_grad:
                    if id(p) in backbone_param_ids:
                        backbone_params.append(p)
                    else:
                        head_params.append(p)
            
            param_groups = [
                {"params": backbone_params, "lr": unfreeze_lr},
                {"params": head_params, "lr": lr}
            ]
        else:
            # dung optimizer mot muc lr
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            param_groups = [{"params": trainable_params, "lr": lr}]

        if opt_name == "SGD":
            return torch.optim.SGD(param_groups, weight_decay=wd, momentum=0.9)
        elif opt_name == "ADAM":
            return torch.optim.Adam(param_groups, weight_decay=wd)
        else:
            return torch.optim.AdamW(param_groups, weight_decay=wd)

    def _build_scheduler(self, epochs: Optional[int] = None) -> Optional[Any]:
        """tao bo dieu chinh learning rate scheduler"""
        sched_name = self.cfg_train.get("scheduler", "cosine").lower()
        t_max = epochs if epochs is not None else self.cfg_train.get("epochs", 10)
        
        if sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)
        elif sched_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", patience=3)
        return None

    def train_one_epoch(self) -> Dict[str, float]:
        """train model trong dung mot epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return {"loss": epoch_loss, "acc": epoch_acc}

    def validate(self) -> Dict[str, float]:
        """chay validation tinh loss va accuracy"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total

        return {"loss": val_loss, "acc": val_acc}

    def run(self) -> None:
        """chay toan bo qua trinh training"""
        epochs = self.cfg_train.get("epochs", 10)
        checkpoint_dir = self.cfg_train.get("checkpoint_dir", "checkpoints/")
        patience = self.cfg_train.get("early_stopping_patience", 5)
        
        unfreeze_epoch = self.cfg_train.get("unfreeze_epoch", None)
        unfreeze_layers = self.cfg_train.get("unfreeze_layers", 30)
        has_unfrozen = False
        
        logger.info(f"Starting classification training on {self.device} for {epochs} epochs...")
        if unfreeze_epoch is not None:
            logger.info(f"Dynamic unfreezing scheduled at Epoch {unfreeze_epoch} (unfreeze {unfreeze_layers} layers).")
        
        best_epoch = 0
        no_improvement_epochs = 0

        for epoch in range(1, epochs + 1):
            if unfreeze_epoch is not None and epoch >= unfreeze_epoch and not has_unfrozen:
                if hasattr(self.model, "unfreeze_backbone"):
                    logger.info(f"--- Epoch {epoch}: Dynamic Unfreezing Triggered ---")
                    self.model.unfreeze_backbone(unfreeze_layers)
                    
                    self.optimizer = self._build_optimizer()
                    
                    remaining_epochs = epochs - epoch + 1
                    self.scheduler = self._build_scheduler(remaining_epochs)
                    
                    has_unfrozen = True

            t0 = time.time()
            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()
            elapsed = time.time() - t0

            # Learning rate step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["acc"])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} - Acc: {train_metrics['acc']*100:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} - Acc: {val_metrics['acc']*100:.2f}% | "
                f"LR: {current_lr:.6f} | Took {elapsed:.1f}s"
            )

            # kiem tra xem co phai model tot nhat khong
            if val_metrics["acc"] > self.best_acc:
                self.best_acc = val_metrics["acc"]
                best_epoch = epoch
                no_improvement_epochs = 0
                
                # luu checkpoint tot nhat
                best_path = os.path.join(checkpoint_dir, "best_classifier.pth")
                self.save_checkpoint(best_path)
                logger.info(f"New best validation accuracy: {self.best_acc*100:.2f}%. Saved to {best_path}")
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                    break

        logger.info(f"Training finished. Best Val Acc: {self.best_acc*100:.2f}% at epoch {best_epoch}")

    def save_checkpoint(self, path: str) -> None:
        """luu checkpoint cua model phan loai"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
                "class_names": getattr(self.model, "class_names", [])
            },
            path
        )
