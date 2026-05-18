"""
Abstract base class for the training loop and validation logic.
"""
import abc
from typing import Dict, Any

class BaseTrainer(abc.ABC):
    """
    Abstract base class for training models.
    """

    @abc.abstractmethod
    def train_one_epoch(self) -> Dict[str, float]:
        """Trains the model for a single epoch."""
        pass

    @abc.abstractmethod
    def validate(self) -> Dict[str, float]:
        """Runs the validation loop and computes metrics."""
        pass

    @abc.abstractmethod
    def run(self) -> None:
        """Executes the complete training process."""
        pass

    @abc.abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Saves the model checkpoint."""
        pass
