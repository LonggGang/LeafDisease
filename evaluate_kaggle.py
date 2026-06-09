"""
Self-contained script to evaluate AdvancedCNN, V2PlantNet, and YOLO models on PlantDoc.
Calculates Confusion Matrices and exports them into a single combined image.
"""
import os
import time
import argparse
import sys
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from src.dataprocessing.dataset import PlantDiseaseDataset

# Optional packages
try:
    import yaml
except ImportError:
    print("Warning: PyYAML is required. Install using: pip install pyyaml")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# =====================================================================
# 1. Classification Model Architectures
# =====================================================================

# --- V2PlantNet ---
class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class V2PlantNet(nn.Module):
    def __init__(self, num_classes: int = 38):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = 224
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(32, 64, stride=1),
            DepthwiseSeparableBlock(64, 64, stride=1),
            DepthwiseSeparableBlock(64, 64, stride=1),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(64, 128, stride=2),
            *[DepthwiseSeparableBlock(128, 128, stride=1) for _ in range(6)]
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(128, 256, stride=2),
            DepthwiseSeparableBlock(256, 256, stride=1),
            DepthwiseSeparableBlock(256, 256, stride=1),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(0.45),
            nn.Linear(256, self.num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


# --- AdvancedCNN ---
class SEBlockWithLN(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        se_units = max(channels // reduction_ratio, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, se_units, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.ln = nn.LayerNorm(se_units)
        self.fc2 = nn.Linear(se_units, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        z = self.squeeze(x).view(B, C)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.ln(z)
        z = self.fc2(z)
        z = self.sigmoid(z).view(B, C, 1, 1)
        return x * z


class GroupedDepthwiseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, num_groups: int = 4):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_relu = nn.ReLU(inplace=True)
        
        actual_groups = self._safe_groups(in_channels, out_channels, num_groups)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, groups=actual_groups, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.pw_relu = nn.ReLU(inplace=True)
    @staticmethod
    def _safe_groups(in_ch: int, out_ch: int, num_groups: int) -> int:
        for g in range(num_groups, 0, -1):
            if in_ch % g == 0 and out_ch % g == 0:
                return g
        return 1
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_relu(self.dw_bn(self.dw_conv(x)))
        x = self.pw_relu(self.pw_bn(self.pw_conv(x)))
        return x


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, num_groups: int = 4, reduction_ratio: int = 16):
        super().__init__()
        self.main = GroupedDepthwiseBlock(in_channels, out_channels, stride=stride, num_groups=num_groups)
        self.se = SEBlockWithLN(out_channels, reduction_ratio=reduction_ratio)
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=stride, padding=0, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        out = self.se(out)
        shortcut = self.skip(x) if self.skip is not None else x
        return out + shortcut


class AdvancedCNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 30, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = 224
        
        mobilenet = models.mobilenet_v2(pretrained=False)
        self.backbone = mobilenet.features
        
        self.block1 = ResidualSEBlock(1280, 256, stride=1)
        self.block2 = ResidualSEBlock(256, 128, stride=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(128, 512)
        self.swish1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.swish2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_rate / 2)
        
        self.classifier = nn.Linear(256, self.num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x).flatten(1)
        x = self.drop1(self.swish1(self.fc1(x)))
        x = self.drop2(self.swish2(self.fc2(x)))
        return self.classifier(x)


# =====================================================================
# 2. PlantDoc and PlantVillage Class Lists & Mappings
# =====================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode for plotting on servers
import matplotlib.pyplot as plt

plantdoc_classes = [
    "Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf spot",
    "Bell_pepper leaf", "Blueberry leaf", "Cherry leaf", "Corn Gray leaf spot",
    "Corn leaf blight", "Corn rust leaf", "Peach leaf", "Potato leaf early blight",
    "Potato leaf late blight", "Potato leaf", "Raspberry leaf", "Soyabean leaf",
    "Soybean leaf", "Squash Powdery mildew leaf", "Strawberry leaf", "Tomato Early blight leaf",
    "Tomato Septoria leaf spot", "Tomato leaf bacterial spot", "Tomato leaf late blight",
    "Tomato leaf mosaic virus", "Tomato leaf yellow virus", "Tomato leaf", "Tomato mold leaf",
    "Tomato two spotted spider mites leaf", "grape leaf black rot", "grape leaf"
]

plantvillage_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def clean_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").lower()

def map_pv_name_to_pd_name(pv_name: str) -> Optional[str]:
    pv_name_lower = pv_name.lower().replace("_", " ").replace(",", "")
    if "apple" in pv_name_lower:
        if "scab" in pv_name_lower: return "Apple Scab Leaf"
        if "rust" in pv_name_lower: return "Apple rust leaf"
        return "Apple leaf"
    if "blueberry" in pv_name_lower:
        return "Blueberry leaf"
    if "cherry" in pv_name_lower:
        return "Cherry leaf"
    if "corn" in pv_name_lower:
        if "rust" in pv_name_lower or "common" in pv_name_lower: return "Corn rust leaf"
        if "gray" in pv_name_lower: return "Corn Gray leaf spot"
        if "blight" in pv_name_lower: return "Corn leaf blight"
        return None
    if "grape" in pv_name_lower:
        if "black rot" in pv_name_lower: return "grape leaf black rot"
        return "grape leaf"
    if "peach" in pv_name_lower:
        return "Peach leaf"
    if "pepper" in pv_name_lower or "bell" in pv_name_lower:
        if "spot" in pv_name_lower or "bacterial" in pv_name_lower: return "Bell_pepper leaf spot"
        return "Bell_pepper leaf"
    if "potato" in pv_name_lower:
        if "early" in pv_name_lower: return "Potato leaf early blight"
        if "late" in pv_name_lower: return "Potato leaf late blight"
        return "Potato leaf"
    if "raspberry" in pv_name_lower:
        return "Raspberry leaf"
    if "soybean" in pv_name_lower or "soyabean" in pv_name_lower:
        return "Soybean leaf"
    if "squash" in pv_name_lower:
        return "Squash Powdery mildew leaf"
    if "strawberry" in pv_name_lower:
        return "Strawberry leaf"
    if "tomato" in pv_name_lower:
        if "bacterial" in pv_name_lower: return "Tomato leaf bacterial spot"
        if "early" in pv_name_lower: return "Tomato Early blight leaf"
        if "late" in pv_name_lower: return "Tomato leaf late blight"
        if "mold" in pv_name_lower: return "Tomato mold leaf"
        if "septoria" in pv_name_lower: return "Tomato Septoria leaf spot"
        if "spider" in pv_name_lower or "mite" in pv_name_lower: return "Tomato two spotted spider mites leaf"
        if "yellow" in pv_name_lower: return "Tomato leaf yellow virus"
        if "mosaic" in pv_name_lower: return "Tomato leaf mosaic virus"
        return "Tomato leaf"
    return None

def normalize_pv_name(name: str) -> str:
    """Canonical normalization of PlantVillage class names to resolve spelling/format differences."""
    name = name.replace("(including_sour)", "").replace("(maize)", "")
    name = name.replace(" ", "").replace("_", "").replace("-", "").replace(",", "").replace("(", "").replace(")", "")
    return name.lower().strip()

def prepare_yolo_yaml(data_path: str) -> str:
    """
    Validates a YOLO dataset configuration file. If the file is missing 'train' or 'val' keys,
    it dynamically generates a corrected YAML file in scratch/temp_yolo_dataset_pv.yaml.
    """
    if not data_path:
        return data_path
        
    from pathlib import Path
    path_obj = Path(data_path)
    if not path_obj.exists():
        return data_path
        
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Could not parse YAML at {data_path}: {e}")
        return data_path
        
    if not yaml_data or not isinstance(yaml_data, dict):
        return data_path
        
    # Check if 'train' or 'val' keys are missing
    has_train = "train" in yaml_data
    has_val = "val" in yaml_data or "valid" in yaml_data
    
    if has_train and has_val:
        return data_path
        
    print(f"Dataset YAML {data_path} is missing 'train' or 'val' keys. Creating a temporary corrected configuration...")
    
    corrected_data = yaml_data.copy()
    yaml_dir = path_obj.parent.resolve()
    
    if "path" not in corrected_data:
        corrected_data["path"] = str(yaml_dir).replace("\\", "/")
        
    # Check images directory
    images_dir = yaml_dir / "images"
    train_rel = "images"
    val_rel = "images"
    
    if images_dir.is_dir():
        if (images_dir / "train").is_dir():
            train_rel = "images/train"
        if (images_dir / "val").is_dir():
            val_rel = "images/val"
        elif (images_dir / "valid").is_dir():
            val_rel = "images/valid"
            
    corrected_data["train"] = train_rel
    corrected_data["val"] = val_rel
    corrected_data["test"] = val_rel
    
    temp_dir = Path("scratch")
    os.makedirs(temp_dir, exist_ok=True)
    temp_yaml_path = temp_dir / "temp_yolo_dataset_pv.yaml"
    
    with open(temp_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(corrected_data, f, default_flow_style=False, sort_keys=False)
        
    resolved_path = str(temp_yaml_path.resolve()).replace("\\", "/")
    print(f"Dynamically generated corrected YOLO dataset configuration at: {resolved_path}")
    return resolved_path

def prepare_yolo_dataset_subset(data_path: str, fraction: float = 1.0) -> Tuple[str, Optional[str]]:
    """
    If fraction < 1.0, creates a temporary directory with a subset of images and labels,
    and returns a path to a generated dataset YAML pointing to it.
    Otherwise, returns the path to the standard corrected dataset YAML.
    """
    if not data_path:
        return data_path, None
        
    from pathlib import Path
    path_obj = Path(data_path)
    if not path_obj.exists():
        return data_path, None
        
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Could not parse YAML at {data_path}: {e}")
        return data_path, None
        
    if not yaml_data or not isinstance(yaml_data, dict):
        return data_path, None
        
    names = yaml_data.get("names", {})
    nc = yaml_data.get("nc", len(names))
    
    yaml_dir = path_obj.parent.resolve()
    orig_img_dir = yaml_dir / "images"
    orig_lbl_dir = yaml_dir / "labels"
    
    if not orig_img_dir.exists() or not orig_lbl_dir.exists():
        # Fallback to standard check
        prepared = prepare_yolo_yaml(data_path)
        return prepared, None
        
    if fraction >= 1.0:
        prepared = prepare_yolo_yaml(data_path)
        return prepared, None
        
    print(f"Preparing temporary YOLO dataset subset (fraction={fraction})...")
    import random
    import shutil
    random.seed(42)
    
    temp_dir = Path("scratch/temp_yolo_subset")
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    temp_img_dir = temp_dir / "images"
    temp_lbl_dir = temp_dir / "labels"
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_lbl_dir, exist_ok=True)
    
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
    all_images = [f for f in os.listdir(orig_img_dir) 
                  if os.path.isfile(orig_img_dir / f) and os.path.splitext(f)[1] in img_extensions]
                  
    if not all_images:
        print("Warning: No images found, falling back to full dataset.")
        prepared = prepare_yolo_yaml(data_path)
        return prepared, None
        
    sample_size = max(1, int(len(all_images) * fraction))
    sampled_images = random.sample(all_images, sample_size)
    
    print(f"Copying {len(sampled_images)} images and labels for subset evaluation...")
    for img_name in sampled_images:
        shutil.copy2(orig_img_dir / img_name, temp_img_dir / img_name)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        if (orig_lbl_dir / lbl_name).exists():
            shutil.copy2(orig_lbl_dir / lbl_name, temp_lbl_dir / lbl_name)
            
    corrected_data = {
        "path": str(temp_dir.resolve()).replace("\\", "/"),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": names,
        "nc": nc
    }
    
    temp_yaml_path = temp_dir / "temp_yolo_dataset_pv.yaml"
    with open(temp_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(corrected_data, f, default_flow_style=False, sort_keys=False)
        
    resolved_yaml = str(temp_yaml_path.resolve()).replace("\\", "/")
    print(f"Subset dataset config created at: {resolved_yaml}")
    return resolved_yaml, str(temp_dir.resolve())

# =====================================================================

# =====================================================================
# 3. Helper Functions for Evaluation and Confusion Matrix Generation
# =====================================================================

def evaluate_classifier_cm(model: nn.Module, checkpoint_path: str, data_dir: str, device: torch.device, model_type: str = "advanced_cnn", dataset_name: str = "PlantDoc", fraction: float = 1.0) -> np.ndarray:
    """Evaluates classification model and returns its aligned confusion matrix."""
    print(f"Loading classifier checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Loading classification dataset from {data_dir}...")
    dataset = PlantDiseaseDataset(data_dir, transform=val_transforms)
    dataset_classes = dataset.classes
    
    if fraction < 1.0:
        num_samples = max(1, int(len(dataset) * fraction))
        generator = torch.Generator().manual_seed(42)
        dataset, _ = torch.utils.data.random_split(dataset, [num_samples, len(dataset) - num_samples], generator=generator)
        print(f"Subsampled classification dataset to {num_samples} samples (fraction={fraction}).")
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    y_true = []
    y_pred = []
    
    if dataset_name == "PlantDoc":
        clean_pd_classes = [clean_name(c) for c in plantdoc_classes]
        sorted_pd_classes = sorted(plantdoc_classes)
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                outputs = model(images)
                _, preds = outputs.max(dim=1)
                
                for label, pred in zip(labels, preds):
                    true_name = dataset_classes[label.item()]
                    try:
                        true_idx = clean_pd_classes.index(clean_name(true_name))
                    except ValueError:
                        continue  # skip if name doesn't match
                    
                    pred_idx_val = pred.item()
                    if model_type == "advanced_cnn":
                        pred_name = sorted_pd_classes[pred_idx_val]
                    else:  # v2plantnet has 38 classes
                        pred_name_pv = plantvillage_classes[pred_idx_val]
                        pred_name = map_pv_name_to_pd_name(pred_name_pv)
                    
                    if pred_name and clean_name(pred_name) in clean_pd_classes:
                        pred_idx = clean_pd_classes.index(clean_name(pred_name))
                    else:
                        pred_idx = 30  # background / unmapped
                        
                    y_true.append(true_idx)
                    y_pred.append(pred_idx)
                    
        # Compute 31x31 confusion matrix
        cm = np.zeros((31, 31), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm
        
    else:  # PlantVillage
        # Build mapping index from dataset_classes to plantvillage_classes
        folder_to_pv_idx = {}
        for folder_name in dataset_classes:
            normalized_folder = normalize_pv_name(folder_name)
            match_idx = None
            for pv_idx, pv_class in enumerate(plantvillage_classes):
                if normalize_pv_name(pv_class) == normalized_folder:
                    match_idx = pv_idx
                    break
            if match_idx is not None:
                folder_to_pv_idx[folder_name] = match_idx
            else:
                print(f"Warning: No match found for folder {folder_name}")
                folder_to_pv_idx[folder_name] = 38 # background / unmapped
                
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                outputs = model(images)
                _, preds = outputs.max(dim=1)
                
                for label, pred in zip(labels, preds):
                    true_name = dataset_classes[label.item()]
                    true_idx = folder_to_pv_idx.get(true_name, 38)
                    
                    pred_idx_val = pred.item()
                    if pred_idx_val < len(dataset_classes):
                        pred_folder_name = dataset_classes[pred_idx_val]
                        pred_idx = folder_to_pv_idx.get(pred_folder_name, 38)
                    else:
                        pred_idx = 38
                        
                    y_true.append(true_idx)
                    y_pred.append(pred_idx)
                    
        # Compute 39x39 confusion matrix
        cm = np.zeros((39, 39), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

def evaluate_yolo_cm(checkpoint_path: str, dataset_yaml: str, device: str, conf: float = 0.25) -> np.ndarray:
    """Evaluates YOLO model and returns its confusion matrix."""
    print(f"Loading YOLO model from {checkpoint_path}...")
    model = YOLO(checkpoint_path)
    results = model.val(
        data=dataset_yaml,
        split='test',
        device=device,
        verbose=False,
        conf=conf
    )
    # Get the matrix
    cm = results.confusion_matrix.matrix
    return cm

# =====================================================================
# 4. Main Execution Block
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Leaf Disease Models and Generate Confusion Matrices")
    parser.add_argument("--dataset", type=str, default="PlantDoc", choices=["PlantDoc", "PlantVillage"], help="Dataset to evaluate on")
    parser.add_argument("--class-data", type=str, default=None, help="Path to classification test set (overrides default)")
    parser.add_argument("--od-yaml", type=str, default=None, help="Path to object detection YAML config (overrides default)")
    parser.add_argument("--output-img", type=str, default=None, help="Path to save the combined image (overrides default)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--fraction", type=float, default=None, help="Fraction of dataset to evaluate (defaults to 0.02 on CPU for PlantVillage, 1.0 otherwise)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for object detection evaluation (default: 0.25)")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Base directory for checkpoints (default: 'checkpoints')")
    parser.add_argument("--ckpt-advance", type=str, default=None, help="Path to AdvancedCNN checkpoint (overrides default)")
    parser.add_argument("--ckpt-plantnet", type=str, default=None, help="Path to V2PlantNet checkpoint (overrides default)")
    parser.add_argument("--ckpt-yolov8s-leafnet", type=str, default=None, help="Path to YOLO-LeafNet (v8s) checkpoint (overrides default)")
    parser.add_argument("--ckpt-yolov12s", type=str, default=None, help="Path to YOLOv12s checkpoint (overrides default)")
    parser.add_argument("--ckpt-yolov12s-leafnet", type=str, default=None, help="Path to YOLOv12s-LeafNet checkpoint (overrides default)")
    parser.add_argument("--ckpt-yolov5s", type=str, default=None, help="Path to YOLOv5s checkpoint (overrides default)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    # Determine fraction to evaluate
    if args.fraction is not None:
        fraction = args.fraction
    else:
        if device.type == "cpu" and args.dataset == "PlantVillage":
            fraction = 0.02
        else:
            fraction = 1.0
            
    # Set default values based on --dataset if not overridden
    base_dir = args.checkpoints_dir
    
    if args.dataset == "PlantDoc":
        class_data = args.class_data if args.class_data else "data/PlantDoc/test"
        od_yaml = args.od_yaml if args.od_yaml else "data/PlantDoc/data.yaml"
        output_img = args.output_img if args.output_img else "combined_confusion_matrix.png"
        num_classes = 30
        classes_list = plantdoc_classes
        title_dataset = "PlantDoc Dataset"
        
        default_paths = {
            "AdvancedCNN": f"{base_dir}/main_models_plantdoc/cnn_Advance_finetune_2/best_classifier.pth",
            "V2PlantNet": f"{base_dir}/main_models_plantdoc/plantnet_finetune/best_classifier.pth",
            "YOLO-LeafNet (v8s)": f"{base_dir}/main_models_plantdoc/yolo_leafnet_yolov8s/weights/best.pt",
            "YOLOv12s": f"{base_dir}/main_models_plantdoc/yolov12s/weights/best.pt",
            "YOLOv12s-LeafNet": f"{base_dir}/main_models_plantdoc/yolov12s_leafnet_plantdoc/weights/best.pt",
            "YOLOv5s": f"{base_dir}/main_models_plantdoc/yolov5s/weights/best.pt"
        }
    else:  # PlantVillage
        class_data = args.class_data if args.class_data else "data/PlantVillage/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"
        od_yaml = args.od_yaml if args.od_yaml else "data/PlantVillage_OD/PlantVillage_for_object_detection/Dataset/classes.yaml"
        output_img = args.output_img if args.output_img else "combined_confusion_matrix_plantvillage.png"
        num_classes = 38
        classes_list = plantvillage_classes
        title_dataset = "PlantVillage Dataset"
        
        default_paths = {
            "AdvancedCNN": f"{base_dir}/main_models_plantVillage/cnn_Advance_pretrain_2/best_classifier.pth",
            "V2PlantNet": f"{base_dir}/main_models_plantVillage/plantnet_pretrain/best_classifier.pth",
            "YOLO-LeafNet (v8s)": f"{base_dir}/main_models_plantVillage/yolov8s_leafnet_plant_village/weights/best.pt",
            "YOLOv12s": f"{base_dir}/main_models_plantVillage/yolov12s/best.pt",
            "YOLOv12s-LeafNet": f"{base_dir}/main_models_plantVillage/yolov12s_leafnet_plant_village/weights/best.pt",
            "YOLOv5s": f"{base_dir}/main_models_plantVillage/yolov5s_plant_village/weights/best.pt"
        }
        
    overrides = {
        "AdvancedCNN": args.ckpt_advance,
        "V2PlantNet": args.ckpt_plantnet,
        "YOLO-LeafNet (v8s)": args.ckpt_yolov8s_leafnet,
        "YOLOv12s": args.ckpt_yolov12s,
        "YOLOv12s-LeafNet": args.ckpt_yolov12s_leafnet,
        "YOLOv5s": args.ckpt_yolov5s
    }
    
    checkpoints = {}
    for name, path in default_paths.items():
        task = "detection" if "YOLO" in name else "classification"
        model_type = "advanced_cnn" if name == "AdvancedCNN" else ("v2plantnet" if name == "V2PlantNet" else "")
        final_path = overrides[name] if overrides[name] is not None else path
        checkpoints[name] = (final_path, task, model_type)
        
    print(f"Using device: {device}")
    print(f"Dataset mode: {args.dataset}")
    print(f"Evaluation fraction: {fraction}")
    print(f"Classification Data: {class_data}")
    print(f"Object Detection YAML: {od_yaml}")
    print(f"Output Image Path: {output_img}")
    
    # Check if directories exist
    if not os.path.exists(class_data):
        print(f"Error: Classification data folder {class_data} not found.")
        sys.exit(1)
    if not os.path.exists(od_yaml):
        print(f"Error: Object Detection YAML file {od_yaml} not found.")
        sys.exit(1)
        
    # Dynamically generate corrected YOLO yaml or subset yaml if fraction < 1.0
    prepared_yaml, temp_subset_dir = prepare_yolo_dataset_subset(od_yaml, fraction=fraction)
    
    cm_results = {}
    
    # Define matrix size including background
    matrix_dim = num_classes + 1
    
    for name, (path, task, model_type) in checkpoints.items():
        print(f"\n==============================================")
        print(f"Evaluating {name}")
        print(f"==============================================")
        if not os.path.exists(path):
            print(f"Warning: Checkpoint not found at {path}. Skipping this model.")
            # Create a dummy confusion matrix for layout consistency
            cm_results[name] = np.zeros((matrix_dim, matrix_dim), dtype=np.int64)
            continue
            
        try:
            if task == "classification":
                if model_type == "advanced_cnn":
                    model = AdvancedCNNClassifier(num_classes=num_classes)
                else:
                    model = V2PlantNet(num_classes=38)  # V2PlantNet was trained on 38 classes in both checkpoints!
                    
                cm = evaluate_classifier_cm(model, path, class_data, device, model_type, dataset_name=args.dataset, fraction=fraction)
            else:
                # YOLO val runs on split 'test'
                # Convert torch device to YOLO compatible format
                device_yolo = "0" if device.type == "cuda" else "cpu"
                cm = evaluate_yolo_cm(path, prepared_yaml, device_yolo, conf=args.conf)
                
            # If the returned cm size is different from matrix_dim, pad/crop it
            if cm.shape[0] != matrix_dim or cm.shape[1] != matrix_dim:
                print(f"Warning: Matrix dimension mismatch for {name}: got {cm.shape}, expected ({matrix_dim}, {matrix_dim}). Resizing matrix...")
                new_cm = np.zeros((matrix_dim, matrix_dim), dtype=np.int64)
                min_r = min(cm.shape[0], matrix_dim)
                min_c = min(cm.shape[1], matrix_dim)
                new_cm[:min_r, :min_c] = cm[:min_r, :min_c]
                cm = new_cm
                
            cm_results[name] = cm
            print(f"Successfully evaluated {name}. Matrix shape: {cm.shape}")
        except Exception as e:
            print(f"Failed to evaluate {name}: {e}")
            import traceback
            traceback.print_exc()
            cm_results[name] = np.zeros((matrix_dim, matrix_dim), dtype=np.int64)
            
    # Clean up temp files and subset directory
    if temp_subset_dir and os.path.exists(temp_subset_dir):
        import shutil
        try:
            shutil.rmtree(temp_subset_dir, ignore_errors=True)
            print("Cleaned up temporary YOLO subset directory.")
        except Exception:
            pass
    elif prepared_yaml != od_yaml and os.path.exists(prepared_yaml):
        try:
            os.remove(prepared_yaml)
            print("Cleaned up temporary YOLO configuration file.")
        except Exception:
            pass

    # Plotting Combined Confusion Matrix Image
    print("\nGenerating combined confusion matrix plot...")
    fig, axes = plt.subplots(2, 3, figsize=(24, 16) if args.dataset == "PlantVillage" else (22, 14))
    axes = axes.ravel()
    
    # Add background class label
    labels_list = classes_list + ["background"]
    
    for i, (name, cm) in enumerate(cm_results.items()):
        ax = axes[i]
        
        # Calculate normalized matrix (row sum = 1.0)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.where(row_sums > 0, cm.astype('float') / row_sums, 0.0)
        
        im = ax.imshow(cm_norm, cmap='Blues', interpolation='nearest', vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=16, fontweight='bold', pad=10)
        
        ax.set_xticks(range(matrix_dim))
        ax.set_yticks(range(matrix_dim))
        
        # Only show text labels on the outer edge axes to avoid clutter
        # Label font size is smaller for PlantVillage to prevent overlap
        lbl_size = 5 if args.dataset == "PlantVillage" else 7
        
        if i in [0, 3]:
            ax.set_ylabel("True Label", fontsize=12)
            ax.set_yticklabels(labels_list, fontsize=lbl_size)
        else:
            ax.set_yticklabels([])
            
        if i in [3, 4, 5]:
            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_xticklabels(labels_list, fontsize=lbl_size, rotation=90)
        else:
            ax.set_xticklabels([])
            
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.suptitle(f"Confusion Matrix Comparison on {title_dataset}", fontsize=22, fontweight='bold', y=0.96)
    
    # Save image
    output_abs_path = os.path.abspath(output_img)
    plt.savefig(output_abs_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined confusion matrix image at: {output_abs_path}")

if __name__ == "__main__":
    main()
