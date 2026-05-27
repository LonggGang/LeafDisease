"""
Abstract base class for PyTorch datasets handling plant leaf images.
"""
import abc
import os
import copy
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


class BasePlantDataset(Dataset, abc.ABC):
    """
    Abstract base dataset for classification and detection tasks.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Returns a single sample from the dataset."""
        pass

    @abc.abstractmethod
    def load_annotations(self) -> None:
        """Loads and parses dataset annotations."""
        pass


class PlantDiseaseDataset(BasePlantDataset):
    """
    Generalized dataset class that handles taxonomy unification, split isolation,
    and automated minority class balancing (over-sampling) for training.
    """
    
    def __init__(
        self, 
        root_dir: str, 
        split: str = "train", 
        transform: Optional[Callable] = None, 
        class_mapping: Optional[Dict[str, str]] = None,
        n_train_budget: int = 1000,
        seed: int = 42
    ):
        """
        Args:
            root_dir: Root directory containing class folders.
            split: "train", "val", or "test".
            transform: Transformations to apply (from augmentation.py).
            class_mapping: Dictionary to map irregular folder names to unified names.
            n_train_budget: Target number of samples per class for over-sampling (train split only).
            seed: Random seed for reproducibility in balancing.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.class_mapping = class_mapping or {}
        self.n_train_budget = n_train_budget
        
        # Internal state
        self.samples: List[Dict[str, Any]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.classes: List[str] = []
        
        # Initialize
        random.seed(seed)
        self.load_annotations()
        
        # Apply class balancing strictly to the train split
        if self.split == "train":
            self._balance_classes()
            
    def load_annotations(self) -> None:
        """
        Scans root directory, applies syntactic unification, and loads file paths.
        Assumes directory structure: root_dir/class_name/image.jpg
        """
        if not self.root_dir.exists():
            return
            
        raw_classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        
        # Determine unified unique classes
        unified_set = set()
        for c in raw_classes:
            unified_name = self.class_mapping.get(c, c)
            unified_set.add(unified_name)
            
        self.classes = sorted(list(unified_set))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load sample paths
        for raw_class in raw_classes:
            unified_name = self.class_mapping.get(raw_class, raw_class)
            label_idx = self.class_to_idx[unified_name]
            class_dir = self.root_dir / raw_class
            
            # Sort paths to ensure deterministic order across different OS file systems
            img_paths = sorted(class_dir.glob("*.*"))
            for img_path in img_paths:
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append({
                        "path": str(img_path),
                        "label_idx": label_idx,
                        "unified_name": unified_name,
                        "is_aug": False # Tracking flag to prevent leakage
                    })

    def _balance_classes(self) -> None:
        """
        Applies targeted over-sampling to minority classes up to n_train_budget.
        Strictly confined to the Train split to prevent validation/test leakage.
        """
        class_groups: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(self.classes))}
        
        for sample in self.samples:
            class_groups[sample["label_idx"]].append(sample)
            
        balanced_samples = []
        
        for label_idx, items in class_groups.items():
            if len(items) == 0:
                continue # Skip empty classes entirely
                
            balanced_samples.extend(items) # Add all real samples
            
            # Oversample if under budget
            num_real = len(items)
            deficit = self.n_train_budget - num_real
            
            if deficit > 0:
                # Duplicate randomly with replacement
                for _ in range(deficit):
                    duplicated_sample = copy.deepcopy(random.choice(items))
                    duplicated_sample["is_aug"] = True # Flag as artificial pad
                    balanced_samples.append(duplicated_sample)
                    
        self.samples = balanced_samples
        # Shuffle train set after balancing
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        sample_info = self.samples[idx]
        img_path = sample_info["path"]
        label = sample_info["label_idx"]
        
        # Load image preserving original channels
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback for corrupted images
            image = Image.new("RGB", (256, 256), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
