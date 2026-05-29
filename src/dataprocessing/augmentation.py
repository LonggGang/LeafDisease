"""
Abstract base class for building data augmentation pipelines.
"""
import abc
from typing import Any

import torch
import torchvision.transforms as T


class BaseTransform(abc.ABC):
    """
    Abstract base class defining the interface for data transforms.
    """

    @abc.abstractmethod
    def build_train_transforms(self) -> Any:
        """Constructs and returns the training augmentation pipeline."""
        pass

    @abc.abstractmethod
    def build_val_transforms(self) -> Any:
        """Constructs and returns the validation augmentation pipeline."""
        pass


class PlantDiseaseTransform(BaseTransform):
    """
    Concrete implementation of image processing and augmentation for plant disease datasets.
    Implements conditional logic based on dataset characteristics (IDADP vs PlantDoc).
    """
    def __init__(self, dataset_type: str = "PlantDoc", task: str = "classification", cfg_aug: dict = None):
        self.dataset_type = dataset_type
        self.task = task
        self.cfg_aug = cfg_aug or {}
        
        # ImageNet standardization is mathematically standard for most pre-trained CNNs
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def _get_base_resize(self):
        """
        Structural preprocessing: Upscale to 256x256 using texture-aware interpolation.
        """
        # IDADP requires Nearest-Neighbor to preserve fine-grained fungal networks.
        # PlantDoc and others can use standard Bilinear.
        interp = T.InterpolationMode.NEAREST if self.dataset_type.upper() == "IDADP" else T.InterpolationMode.BILINEAR
        return T.Resize((256, 256), interpolation=interp)

    def build_train_transforms(self) -> Any:
        """
        Dynamic augmentation factory for the Train split.
        Applies conditional adversarial and lighting strategies.
        """
        if self.task.lower() == "detection":
            raise NotImplementedError(
                "Standard torchvision transforms destroy bounding boxes during random crops. "
                "Object Detection requires a bounding-box aware library like Albumentations."
            )
            
        transforms_list = []
        
        # 1. Base Structural Resize (to 256)
        transforms_list.append(self._get_base_resize())
        
        # Extract hyperparameters from config (with defaults)
        train_cfg = self.cfg_aug.get("train", {})
        resize_dim = train_cfg.get("resize", 224)
        h_flip_p = train_cfg.get("horizontal_flip", 0.5)
        v_flip_p = train_cfg.get("vertical_flip", 0.5)
        
        jitter_cfg = train_cfg.get("color_jitter", {})
        brightness = jitter_cfg.get("brightness", 0.2)
        contrast = jitter_cfg.get("contrast", 0.2)
        saturation = jitter_cfg.get("saturation", 0.2)

        # 2. Spatial Invariance (Random Crop down to resize_dim & Flips)
        transforms_list.extend([
            T.RandomCrop(resize_dim),
            T.RandomHorizontalFlip(p=h_flip_p),
            T.RandomVerticalFlip(p=v_flip_p),
            T.RandomRotation(degrees=15)
        ])
        
        # 3. Illumination Simulation (Color Jitter)
        # Disable hue jittering for IDADP to preserve specific diagnostic disease colors.
        hue_shift = 0.0 if self.dataset_type.upper() == "IDADP" else 0.1
        transforms_list.append(
            T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue_shift)
        )
        
        # 4. Tensor Formatting
        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])
        
        # 5. Adversarial Occlusion (Random Erasing - Must be applied after ToTensor)
        # Highly aggressive for PlantDoc; disabled for IDADP to protect small fungal lesions.
        if self.dataset_type.upper() != "IDADP":
            transforms_list.append(T.RandomErasing(p=0.15, scale=(0.02, 0.1)))
            
        return T.Compose(transforms_list)

    def build_val_transforms(self) -> Any:
        """
        Pure structural preprocessing for Validation/Test/Inference (No augmentation).
        Safe for Mobile/IoT deployment edge inference.
        """
        val_cfg = self.cfg_aug.get("val", {})
        resize_dim = val_cfg.get("resize", 224)
        
        transforms_list = [
            self._get_base_resize(),
            T.CenterCrop(resize_dim), # Deterministic center crop for evaluation
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ]
        return T.Compose(transforms_list)
