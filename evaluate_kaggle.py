"""
Self-contained script to evaluate AdvancedCNN, V2PlantNet, and YOLOv12s models on Kaggle.
Calculates Accuracy, Recall, Precision, F1-score, Inference Time, and Model Complexity.
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
from torchvision.datasets import ImageFolder

# Optional packages
try:
    import yaml
except ImportError:
    print("Warning: PyYAML is required for YOLO config generation. Install using: pip install pyyaml")

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

try:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =====================================================================
# 1. Classification Model Architectures (Reproduced from src)
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
    def __init__(self, num_classes: int = 38, dropout_rate: float = 0.3):
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
# 2. Helper Functions for Evaluation
# =====================================================================

def evaluate_classifier(model: nn.Module, data_dir: str, batch_size: int, device: torch.device) -> Dict[str, Any]:
    """Runs full validation of a classification model and calculates metrics."""
    # Define validation transforms
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading dataset from {data_dir}...")
    dataset = ImageFolder(data_dir, transform=val_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    # Latency tracking (using single batch inference to measure raw forward time)
    latencies = []
    
    print("Evaluating classifier...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            # Latency measurement on a per-batch basis, normalized to per-image
            t0 = time.perf_counter()
            outputs = model(images)
            dt = (time.perf_counter() - t0) * 1000.0 / images.size(0) # ms per image
            latencies.append(dt)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            if (i+1) % 50 == 0:
                print(f"Processed batch {i+1}/{len(dataloader)}")
                
    # Calculate metrics
    avg_latency = sum(latencies) / len(latencies)
    
    if SKLEARN_AVAILABLE:
        acc = accuracy_score(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    else:
        # Simple manual fallback if sklearn is missing
        correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
        acc = correct / len(all_targets) if all_targets else 0
        prec, rec, f1 = 0.0, 0.0, 0.0
        print("Warning: scikit-learn is not installed. Recall, Precision, and F1 calculations are skipped.")

    # Complexity
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    
    flops_G = "N/A"
    if THOP_AVAILABLE:
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            flops_G = flops / 1e9
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
            
    return {
        "Acc": acc,
        "Recall": rec,
        "Precision": prec,
        "F1": f1,
        "Inference_ms": avg_latency,
        "Params_M": params_M,
        "GFLOPs": flops_G
    }


def create_temp_yolo_yaml(images_dir: str, labels_dir: str, classes_yaml_path: str) -> str:
    """Generates a temporary dataset configuration yaml for YOLO."""
    with open(classes_yaml_path, "r", encoding="utf-8") as f:
        classes_data = yaml.safe_load(f)
        
    names = classes_data.get("names", {})
    # Convert list to dict if needed
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
        
    images_dir_abs = os.path.abspath(images_dir)
    parent_dir = os.path.dirname(images_dir_abs)
    folder_name = os.path.basename(images_dir_abs)
    
    # Check if the folder structure is already split (e.g. images/val or images/train)
    grandparent_dir = os.path.dirname(parent_dir)
    parent_folder_name = os.path.basename(parent_dir)
    
    if folder_name in ["val", "test", "train"] and parent_folder_name == "images":
        base_path = grandparent_dir.replace("\\", "/")
        val_path = f"images/{folder_name}"
    else:
        base_path = parent_dir.replace("\\", "/")
        val_path = folder_name
        
    yolo_data = {
        "path": base_path,
        "train": val_path,  # Fallback to same folder for training if needed
        "val": val_path,
        "names": names,
        "nc": len(names)
    }
    
    temp_yaml_path = "temp_yolo_dataset_kaggle.yaml"
    with open(temp_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yolo_data, f, default_flow_style=False, sort_keys=False)
        
    print(f"Generated temporary YOLO config at: {os.path.abspath(temp_yaml_path)}")
    return temp_yaml_path


def evaluate_yolo(ckpt_path: str, dataset_yaml: str, device: str) -> Dict[str, Any]:
    """Runs YOLO validation and returns standard metrics."""
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics library is required to evaluate YOLO models.")
        
    print(f"Loading YOLO model from {ckpt_path}...")
    model = YOLO(ckpt_path)
    
    # Run built-in validation
    print("Running YOLO validation...")
    results = model.val(
        data=dataset_yaml,
        split='val',
        device=device,
        verbose=False
    )
    
    res_dict = results.results_dict
    mAP50 = float(res_dict.get("metrics/mAP50(B)", 0.0))
    mAP50_95 = float(res_dict.get("metrics/mAP50-95(B)", 0.0))
    precision = float(res_dict.get("metrics/precision(B)", 0.0))
    recall = float(res_dict.get("metrics/recall(B)", 0.0))
    
    # F1-score calculation
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Latency (preprocess + inference + postprocess)
    preprocess_ms = results.speed.get("preprocess", 0.0)
    inference_ms = results.speed.get("inference", 0.0)
    postprocess_ms = results.speed.get("postprocess", 0.0)
    total_latency = preprocess_ms + inference_ms + postprocess_ms
    
    # Complexity
    params_M = sum(p.numel() for p in model.model.parameters()) / 1e6
    
    flops_G = "N/A"
    try:
        flops_G = float(get_flops(model.model, imgsz=640))
    except Exception as e:
        print(f"Could not calculate FLOPs for YOLO: {e}")
        
    return {
        "Acc": "N/A",  # Detection doesn't have standard classification accuracy
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "Inference_ms": total_latency,
        "Params_M": params_M,
        "GFLOPs": flops_G
    }


# =====================================================================
# 3. Main Execution Block
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Kaggle Evaluation Script for Leaf Disease Models")
    parser.add_argument("--class-data", type=str, default=None, help="Path to classification validation dataset directory (e.g. valid/)")
    parser.add_argument("--od-images", type=str, default=None, help="Path to object detection images directory (e.g. Dataset/images/)")
    parser.add_argument("--od-labels", type=str, default=None, help="Path to object detection labels directory (e.g. Dataset/labels/)")
    parser.add_argument("--od-classes-yaml", type=str, default=None, help="Path to classes.yaml for object detection")
    parser.add_argument("--ckpt-advance", type=str, default=None, help="Path to AdvancedCNN best_classifier.pth")
    parser.add_argument("--ckpt-plantnet", type=str, default=None, help="Path to V2PlantNet best_classifier.pth")
    parser.add_argument("--ckpt-yolo", type=str, default=None, help="Path to YOLOv12 best.pt")
    parser.add_argument("--od-split-val-ratio", type=float, default=0.0, help="If > 0, dynamically splits a random portion of unsplit YOLO images for validation on-the-fly")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--dry-run", action="store_true", help="Perform a quick dry run with dummy data to verify model configuration")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    if args.dry_run:
        print("=== DRY RUN MODE ===")
        # Instantiate and test AdvancedCNN
        try:
            print("Instantiating AdvancedCNN...")
            model = AdvancedCNNClassifier(num_classes=38)
            x = torch.randn(1, 3, 224, 224)
            y = model(x)
            print(f"AdvancedCNN Output shape: {y.shape} (Expected: [1, 38])")
            print("AdvancedCNN OK!")
        except Exception as e:
            print(f"AdvancedCNN Dry-run Failed: {e}")
            
        # Instantiate and test V2PlantNet
        try:
            print("Instantiating V2PlantNet...")
            model = V2PlantNet(num_classes=38)
            x = torch.randn(1, 3, 224, 224)
            y = model(x)
            print(f"V2PlantNet Output shape: {y.shape} (Expected: [1, 38])")
            print("V2PlantNet OK!")
        except Exception as e:
            print(f"V2PlantNet Dry-run Failed: {e}")
            
        # Check YOLO import
        if ULTRALYTICS_AVAILABLE:
            print("Ultralytics and get_flops imported successfully!")
        else:
            print("Ultralytics library NOT available.")
        sys.exit(0)
        
    results = {}
    
    # 1. Evaluate AdvancedCNN
    if args.ckpt_advance and args.class_data:
        print("\n==============================================")
        print("Evaluating AdvancedCNN Model")
        print("==============================================")
        try:
            model = AdvancedCNNClassifier(num_classes=38)
            checkpoint = torch.load(args.ckpt_advance, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            
            res = evaluate_classifier(model, args.class_data, args.batch_size, device)
            results["AdvancedCNN"] = res
            print("AdvancedCNN Evaluation Done!")
        except Exception as e:
            print(f"Failed to evaluate AdvancedCNN: {e}")
            
    # 2. Evaluate V2PlantNet
    if args.ckpt_plantnet and args.class_data:
        print("\n==============================================")
        print("Evaluating V2PlantNet Model")
        print("==============================================")
        try:
            model = V2PlantNet(num_classes=38)
            checkpoint = torch.load(args.ckpt_plantnet, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            
            res = evaluate_classifier(model, args.class_data, args.batch_size, device)
            results["V2PlantNet"] = res
            print("V2PlantNet Evaluation Done!")
        except Exception as e:
            print(f"Failed to evaluate V2PlantNet: {e}")
            
    # 3. Evaluate YOLOv12s
    if args.ckpt_yolo and args.od_images and args.od_classes_yaml:
        print("\n==============================================")
        print("Evaluating YOLOv12s Model")
        print("==============================================")
        temp_split_dir = None
        eval_images_dir = args.od_images
        
        if args.od_labels:
            eval_labels_dir = args.od_labels
        else:
            # Smart auto-detection of sibling labels directory (supports both split and unsplit paths)
            img_dir_abs = os.path.abspath(args.od_images)
            parent_dir = os.path.dirname(img_dir_abs)
            folder_name = os.path.basename(img_dir_abs)
            
            grandparent_dir = os.path.dirname(parent_dir)
            parent_folder_name = os.path.basename(parent_dir)
            
            if folder_name in ["val", "test", "train"] and parent_folder_name == "images":
                # E.g. path/images/val -> path/labels/val
                eval_labels_dir = os.path.join(grandparent_dir, "labels", folder_name)
            elif folder_name == "images":
                # E.g. path/images -> path/labels
                eval_labels_dir = os.path.join(parent_dir, "labels")
            else:
                # Fallback to sibling folder
                eval_labels_dir = os.path.join(parent_dir, "labels")
        
        try:
            # Create a dynamic train/val split if requested
            if args.od_split_val_ratio > 0.0:
                print(f"Dynamic splitting enabled. Preparing a random {args.od_split_val_ratio*100:.1f}% validation subset...")
                import random
                import shutil
                random.seed(42)
                
                temp_split_dir = os.path.abspath("temp_val_split_dir")
                temp_img_dir = os.path.join(temp_split_dir, "images")
                temp_lbl_dir = os.path.join(temp_split_dir, "labels")
                os.makedirs(temp_img_dir, exist_ok=True)
                os.makedirs(temp_lbl_dir, exist_ok=True)
                
                image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
                all_images = [f for f in os.listdir(args.od_images) 
                              if os.path.isfile(os.path.join(args.od_images, f)) and os.path.splitext(f)[1] in image_extensions]
                
                if not all_images:
                    raise ValueError(f"No images found in {args.od_images} for dynamic splitting.")
                    
                sample_size = max(1, int(len(all_images) * args.od_split_val_ratio))
                sampled_images = random.sample(all_images, sample_size)
                
                print(f"Creating symlinks/copies of {len(sampled_images)} validation images & labels in temporary workspace...")
                link_count = 0
                for img_name in sampled_images:
                    src_img = os.path.join(args.od_images, img_name)
                    dst_img = os.path.join(temp_img_dir, img_name)
                    
                    lbl_name = os.path.splitext(img_name)[0] + ".txt"
                    src_lbl = os.path.join(eval_labels_dir, lbl_name)
                    dst_lbl = os.path.join(temp_lbl_dir, lbl_name)
                    
                    if os.path.exists(src_lbl):
                        try:
                            # Symlink is instant and takes 0 extra disk space
                            if sys.platform == "win32":
                                shutil.copy2(src_img, dst_img)
                                shutil.copy2(src_lbl, dst_lbl)
                            else:
                                os.symlink(src_img, dst_img)
                                os.symlink(src_lbl, dst_lbl)
                            link_count += 1
                        except Exception:
                            # Fallback if symlinks fail (e.g. due to permissions)
                            shutil.copy2(src_img, dst_img)
                            shutil.copy2(src_lbl, dst_lbl)
                            link_count += 1
                            
                print(f"Prepared {link_count} validation pairs.")
                eval_images_dir = temp_img_dir
                eval_labels_dir = temp_lbl_dir
            
            # Create a temp yaml configuration file
            temp_yaml = create_temp_yolo_yaml(eval_images_dir, eval_labels_dir, args.od_classes_yaml)
            
            res = evaluate_yolo(args.ckpt_yolo, temp_yaml, args.device)
            results["YOLOv12s"] = res
            print("YOLOv12s Evaluation Done!")
            
            # Clean up temp file
            if os.path.exists(temp_yaml):
                os.remove(temp_yaml)
        except Exception as e:
            print(f"Failed to evaluate YOLOv12s: {e}")
        finally:
            # Always clean up dynamic split folder
            if temp_split_dir and os.path.exists(temp_split_dir):
                import shutil
                print("Cleaning up temporary validation split folder...")
                shutil.rmtree(temp_split_dir, ignore_errors=True)
            
    # Print Results Table
    if not results:
        print("\nNo models were evaluated. Please provide checkpoints and datasets paths.")
        print("Use --help to see all available arguments.")
        return
        
    print("\n\n========================================================")
    print("EVALUATION RESULTS REPORT")
    print("========================================================")
    
    # Format and display a markdown table
    print("\n| Model | Acc | Recall | Precision | F1 | Inference Time (ms) | Params (M) | GFLOPs | mAP@50 | mAP@50-95 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, r in results.items():
        acc = f"{r['Acc']*100:.2f}%" if isinstance(r["Acc"], float) else r["Acc"]
        rec = f"{r['Recall']*100:.2f}%" if isinstance(r["Recall"], float) else r["Recall"]
        prec = f"{r['Precision']*100:.2f}%" if isinstance(r["Precision"], float) else r["Precision"]
        f1 = f"{r['F1']*100:.2f}%" if isinstance(r["F1"], float) else r["F1"]
        
        map50 = f"{r['mAP50']*100:.2f}%" if "mAP50" in r else "-"
        map50_95 = f"{r['mAP50_95']*100:.2f}%" if "mAP50_95" in r else "-"
        
        flops = f"{r['GFLOPs']:.4f}" if isinstance(r["GFLOPs"], float) else r["GFLOPs"]
        
        print(f"| {name} | {acc} | {rec} | {prec} | {f1} | {r['Inference_ms']:.2f} ms | {r['Params_M']:.2f} M | {flops} | {map50} | {map50_95} |")
        
    print("\n========================================================")


if __name__ == "__main__":
    main()
