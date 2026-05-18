"""
Abstract base class for building data augmentation pipelines.
"""
import abc
from typing import Any

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
