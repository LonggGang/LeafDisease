"""
Utility script to parse a YOLO detection dataset and crop bounding boxes
to generate a folder-structured classification dataset.
"""
import os
import yaml
import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CropDataset")

def resolve_split_dirs(data_yaml_path: str, split: str):
    """
    Finds the images and labels directory for a given split using heuristics.
    Returns:
        (images_dir_path, labels_dir_path) or (None, None)
    """
    yaml_path = Path(data_yaml_path)
    yaml_dir = yaml_path.parent
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    
    # 1. Try to get relative path from YAML
    # Keys might be 'train', 'val', 'test'
    rel_path = yaml_data.get(split)
    if not rel_path and split == "valid":
        rel_path = yaml_data.get("val")
    elif not rel_path and split == "val":
        rel_path = yaml_data.get("valid")
        
    candidates = []
    if rel_path:
        # Candidate 1: relative path from yaml directory
        candidates.append(yaml_dir / rel_path)
        # Candidate 2: relative path from yaml directory, but ignoring parent navigation if it's nested
        candidates.append(yaml_dir / rel_path.lstrip("../").lstrip("./"))
        
    # Heuristics based on standard Roboflow/YOLO formats
    # Candidate 3: yaml_dir / split / images
    candidates.append(yaml_dir / split / "images")
    # Candidate 4: yaml_dir / val/images (if split is valid)
    if split in ["val", "valid"]:
        candidates.append(yaml_dir / "val" / "images")
        candidates.append(yaml_dir / "valid" / "images")
        
    # Check candidates
    images_dir = None
    for cand in candidates:
        cand_resolved = cand.resolve()
        if cand_resolved.is_dir():
            images_dir = cand_resolved
            break
            
    if not images_dir:
        logger.warning(f"Could not locate images directory for split '{split}'")
        return None, None
        
    # Locate labels directory corresponding to images directory
    # Standard YOLO has images/ and labels/ at the same level
    labels_dir = images_dir.parent / "labels"
    if not labels_dir.is_dir():
        # Fallback: check if labels is in the same folder with images (unlikely but possible)
        labels_dir = images_dir
        
    logger.info(f"Resolved paths for '{split}': images={images_dir}, labels={labels_dir}")
    return images_dir, labels_dir

def crop_yolo_dataset(data_yaml_path: str, output_dir: str):
    """
    Parses YOLO annotations and crops bounding boxes to create a classification dataset.
    """
    yaml_path = Path(data_yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
        
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
        
    # Get class names
    class_names = yaml_data.get("names")
    if not class_names:
        raise ValueError("No class 'names' found in data.yaml")
        
    # Convert dict to list if names is a dict
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names.keys())]
        
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    splits = ["train", "valid", "test"]
    
    total_cropped = 0
    
    for split in splits:
        img_dir, lbl_dir = resolve_split_dirs(data_yaml_path, split)
        if not img_dir or not lbl_dir:
            continue
            
        # Create output directories for each class in this split
        # We standardise the split folder name to 'train' and 'val' for the classification trainer
        out_split = "val" if split in ["val", "valid"] else split
        
        # Load image paths
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]
        
        logger.info(f"Processing {len(img_files)} images in '{split}' split...")
        
        split_cropped = 0
        
        for img_file in img_files:
            # Find corresponding label file
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                continue
                
            # Load image
            try:
                img = Image.open(img_file).convert("RGB")
            except Exception as e:
                logger.warning(f"Could not load image {img_file}: {e}")
                continue
                
            img_w, img_h = img.size
            
            # Read label annotations
            with open(lbl_file, "r", encoding="utf-8") as lf:
                lines = lf.readlines()
                
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
                    logger.warning(f"Class index {class_idx} out of range in {lbl_file}")
                    continue
                    
                class_name = class_names[class_idx]
                
                # Sanitize class name for folder safety
                class_name_clean = class_name.replace(" ", "_").replace("/", "_")
                
                # Convert normalized coordinates to pixel coordinates
                w = width * img_w
                h = height * img_h
                x1 = max(0, int((x_center - width / 2) * img_w))
                y1 = max(0, int((y_center - height / 2) * img_h))
                x2 = min(img_w, int((x_center + width / 2) * img_w))
                y2 = min(img_h, int((y_center + height / 2) * img_h))
                
                # Filter out extremely small boxes or invalid coordinates
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue
                    
                # Crop
                cropped_img = img.crop((x1, y1, x2, y2))
                
                # Save
                class_out_dir = output_path / out_split / class_name_clean
                os.makedirs(class_out_dir, exist_ok=True)
                
                crop_name = f"{img_file.stem}_crop_{idx}.jpg"
                cropped_img.save(class_out_dir / crop_name)
                
                split_cropped += 1
                total_cropped += 1
                
        logger.info(f"Split '{split}': Cropped {split_cropped} bounding boxes.")
        
    logger.info(f"Done! Created classification dataset at '{output_dir}'. Total cropped images: {total_cropped}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop YOLO dataset for Classification")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    crop_yolo_dataset(args.data, args.output)
