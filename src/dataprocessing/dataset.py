"""
Abstract base class for PyTorch datasets handling plant leaf images.
"""
import abc
import torch.utils.data
from typing import Any

class BasePlantDataset(torch.utils.data.Dataset, abc.ABC):
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
