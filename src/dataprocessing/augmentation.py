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
    def __init__(self, dataset_type: str = "PlantDoc", task: str = "classification"):
        self.dataset_type = dataset_type
        self.task = task
        
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
        
        # 2. Spatial Invariance (Random Crop down to 224x224 & Flips)
        transforms_list.extend([
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15)
        ])
        
        # 3. Illumination Simulation (Color Jitter)
        # Disable hue jittering for IDADP to preserve specific diagnostic disease colors.
        hue_shift = 0.0 if self.dataset_type.upper() == "IDADP" else 0.1
        transforms_list.append(
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=hue_shift)
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
        transforms_list = [
            self._get_base_resize(),
            T.CenterCrop(224), # Deterministic center crop for evaluation
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ]
        return T.Compose(transforms_list)
