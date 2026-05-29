import os
import shutil
import argparse
import random
from pathlib import Path

def split_dataset(raw_dir: str, out_dir: str, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Physical Folder Splitting: Copies raw dataset images into train/val/test folders.
    Guarantees isolation at the file level before DataLoader initializes.
    """
    print(f"Initializing Dataset Split with seed {seed}...")
    random.seed(seed)
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Source directory '{raw_dir}' does not exist.")
        
    splits = ['train', 'val', 'test']
    out_paths = {split: Path(out_dir) / split for split in splits}
    
    # Create base output directories
    for split_path in out_paths.values():
        split_path.mkdir(parents=True, exist_ok=True)
        
    classes = [d for d in raw_path.iterdir() if d.is_dir()]
    
    if not classes:
        print(f"Warning: No class directories found in {raw_dir}")
        return

    print(f"Found {len(classes)} classes. Beginning split (Train: {train_ratio}, Val: {val_ratio})...")

    for cls_dir in classes:
        cls_name = cls_dir.name
        # Note: If filenames are highly sequential (source-linked), a grouped split logic 
        # should replace this random.shuffle to prevent Source-Level Leakage.
        images = sorted(list(cls_dir.glob("*.*")))
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits_dict = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }
        
        for split_name, imgs in splits_dict.items():
            dest_dir = out_paths[split_name] / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for img in imgs:
                # Copy file to new destination
                shutil.copy2(img, dest_dir / img.name)
                
        print(f" - Processed {cls_name}: Train({len(splits_dict['train'])}), Val({len(splits_dict['val'])}), Test({len(splits_dict['test'])})")

    print(f"\n✅ Splitting complete! Check '{out_dir}/' for your processed splits.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split raw dataset into train/val/test")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to unorganized raw dataset folder")
    parser.add_argument("--out_dir", type=str, default="data", help="Output base directory (default: 'data')")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    
    args = parser.parse_args()
    split_dataset(args.raw_dir, args.out_dir, args.train_ratio, args.val_ratio)
