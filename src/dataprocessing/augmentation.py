"""khoi tao cac buoc bien doi anh"""
import abc
from typing import Any

import torch
import torchvision.transforms as T


class BaseTransform(abc.ABC):
    """lop cha cho cac phep bien doi anh"""

    @abc.abstractmethod
    def build_train_transforms(self) -> Any:
        """tao cac phep bien doi cho tap train"""
        pass

    @abc.abstractmethod
    def build_val_transforms(self) -> Any:
        """tao cac phep bien doi cho tap val"""
        pass


class PlantDiseaseTransform(BaseTransform):
    """class trien khai cac phep bien doi anh cho tung tap du lieu"""
    def __init__(self, dataset_type: str = "PlantDoc", task: str = "classification"):
        self.dataset_type = dataset_type
        self.task = task
        
        # lay mean va std chuan cua imagenet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def _get_base_resize(self):
        """chuyen kich thuoc anh ve mac dinh"""
        # idadp thi dung nearest neighbor
        interp = T.InterpolationMode.NEAREST if self.dataset_type.upper() == "IDADP" else T.InterpolationMode.BILINEAR
        return T.Resize((256, 256), interpolation=interp)

    def build_train_transforms(self) -> Any:
        """tao cac phep bien doi cho tap train"""
        if self.task.lower() == "detection":
            raise NotImplementedError(
                "Standard torchvision transforms destroy bounding boxes during random crops. "
                "Object Detection requires a bounding-box aware library like Albumentations."
            )
            
        transforms_list = []
        
        # 1. resize ve 256
        transforms_list.append(self._get_base_resize())
        
        # 2. crop va lat anh ngau nhien
        transforms_list.extend([
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15)
        ])
        
        # 3. chinh mau sac anh
        hue_shift = 0.0 if self.dataset_type.upper() == "IDADP" else 0.1
        transforms_list.append(
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=hue_shift)
        )
        
        # 4. chuyen thanh tensor
        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])
        
        # 5. xoa mot vung anh ngau nhien
        if self.dataset_type.upper() != "IDADP":
            transforms_list.append(T.RandomErasing(p=0.15, scale=(0.02, 0.1)))
            
        return T.Compose(transforms_list)

    def build_val_transforms(self) -> Any:
        """tao cac phep bien doi cho tap val"""
        transforms_list = [
            self._get_base_resize(),
            T.CenterCrop(224), # crop o giua anh de test
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ]
        return T.Compose(transforms_list)
