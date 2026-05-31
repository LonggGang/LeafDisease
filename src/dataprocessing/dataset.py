"""
Abstract base class for PyTorch datasets handling plant leaf images.
"""
import abc
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
    Pure dataset class that handles file I/O and taxonomy unification.
    Class balancing is deferred to the DataLoader via WeightedRandomSampler.
    """
    
    def __init__(
        self, 
        root_dir: str, 
        transform: Optional[Callable] = None, 
        class_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            root_dir: Root directory containing class folders.
            transform: Transformations to apply (from augmentation.py).
            class_mapping: Dictionary to map irregular folder names to unified names.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_mapping = class_mapping or {}
        
        # Internal state
        self.samples: List[Dict[str, Any]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.classes: List[str] = []
        
        self.load_annotations()
            
    def load_annotations(self) -> None:
        """
        Scans root directory, applies syntactic unification, and loads file paths.
        Assumes directory structure: root_dir/class_name/image.jpg
        Or, if root_dir contains images/ and labels/ subdirectories, parses YOLO detection split.
        """
        if not self.root_dir.exists():
            return
            
        # Tự động phát hiện cấu trúc YOLO detection dataset
        images_dir = self.root_dir / "images"
        labels_dir = self.root_dir / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            self._load_yolo_annotations(images_dir, labels_dir)
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
                        "unified_name": unified_name
                    })

    def _load_yolo_annotations(self, images_dir: Path, labels_dir: Path) -> None:
        """Loads annotations from YOLO detection format split directory."""
        # Find data.yaml in root_dir or parent directories
        data_yaml_path = None
        search_dirs = [self.root_dir, self.root_dir.parent, self.root_dir.parent.parent]
        for d in search_dirs:
            for name in ["data.yaml", "dataset.yaml", "data.yml", "dataset.yml"]:
                p = d / name
                if p.is_file():
                    data_yaml_path = p
                    break
            if data_yaml_path:
                break
                
        if not data_yaml_path:
            raise FileNotFoundError(f"Could not locate data.yaml or dataset.yaml in any parent directories of {self.root_dir}")
            
        import yaml
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
            
        class_names = yaml_data.get("names")
        if not class_names:
            raise ValueError(f"No class names found in {data_yaml_path}")
            
        # Convert dict to list if names is a dict
        if isinstance(class_names, dict):
            class_names = [class_names[i] for i in sorted(class_names.keys())]
            
        # Determine unified unique classes
        unified_set = set()
        for c in class_names:
            unified_name = self.class_mapping.get(c, c)
            unified_set.add(unified_name)
            
        self.classes = sorted(list(unified_set))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load sample paths and their boxes
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        img_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in img_extensions])
        
        for img_path in img_paths:
            lbl_file = labels_dir / f"{img_path.stem}.txt"
            if not lbl_file.exists():
                continue
                
            # Read label annotations
            try:
                with open(lbl_file, "r", encoding="utf-8") as lf:
                    lines = lf.readlines()
            except Exception:
                continue
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                try:
                    class_idx = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    continue
                    
                if class_idx >= len(class_names):
                    continue
                    
                raw_class = class_names[class_idx]
                unified_name = self.class_mapping.get(raw_class, raw_class)
                label_idx = self.class_to_idx[unified_name]
                
                # We store the normalized bounding box coordinates to crop in __getitem__
                self.samples.append({
                    "path": str(img_path),
                    "box": (x_center, y_center, width, height),
                    "label_idx": label_idx,
                    "unified_name": unified_name
                })

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
            
        # Crop if a box is specified in sample_info
        if "box" in sample_info:
            try:
                x_center, y_center, width, height = sample_info["box"]
                img_w, img_h = image.size
                
                x1 = max(0, int((x_center - width / 2) * img_w))
                y1 = max(0, int((y_center - height / 2) * img_h))
                x2 = min(img_w, int((x_center + width / 2) * img_w))
                y2 = min(img_h, int((y_center + height / 2) * img_h))
                
                # Only crop if it's a valid crop size
                if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                    image = image.crop((x1, y1, x2, y2))
            except Exception:
                pass  # Fallback: use original image if crop fails
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
