import logging
from torch.utils.data import DataLoader

from src.dataprocessing.dataset import PlantDiseaseDataset
from src.dataprocessing.augmentation import PlantDiseaseTransform
from src.dataprocessing.sampler import get_balanced_sampler

logger = logging.getLogger("DataBuilder")

def build_dataloaders(cfg_train: dict, cfg_aug: dict):
    """
    Factory function to build the complete data pipeline.
    Reads YAML configurations and outputs ready-to-use DataLoaders.
    """
    logger.info("Initializing Data Pipeline Factory...")

    # 1. Initialize Transforms dynamically using configs
    transform_factory = PlantDiseaseTransform(
        dataset_type=cfg_train.get("dataset_type", "PlantDoc"),
        task=cfg_train.get("task", "classification"),
        cfg_aug=cfg_aug
    )
    
    train_transforms = transform_factory.build_train_transforms()
    val_transforms = transform_factory.build_val_transforms()
    
    # 2. Initialize Datasets
    train_dir = cfg_train.get("train_dir", "data/train")
    val_dir = cfg_train.get("val_dir", "data/val")
    
    logger.info(f"Loading Train Dataset from {train_dir}...")
    train_dataset = PlantDiseaseDataset(root_dir=train_dir, transform=train_transforms)
    
    logger.info(f"Loading Val Dataset from {val_dir}...")
    val_dataset = PlantDiseaseDataset(root_dir=val_dir, transform=val_transforms)
    
    # 3. Initialize Sampler (Train only)
    logger.info("Building Class Balancing Sampler...")
    train_sampler = get_balanced_sampler(train_dataset)
    
    # Extract DataLoader settings from config
    batch_size = cfg_train.get("batch_size", 32)
    num_workers = cfg_train.get("num_workers", 4)
    
    # 4. Create DataLoaders
    logger.info(f"Constructing DataLoaders (Batch Size: {batch_size}, Workers: {num_workers})")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True # Speeds up transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Strict rule: NEVER shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
