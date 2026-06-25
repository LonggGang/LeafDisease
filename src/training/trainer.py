"""lop cha cho tat ca trainer"""
import abc
from typing import Dict, Any

class BaseTrainer(abc.ABC):
    """lop cha de viet cac lop train sau nay"""

    @abc.abstractmethod
    def train_one_epoch(self) -> Dict[str, float]:
        """train model trong mot epoch"""
        pass

    @abc.abstractmethod
    def validate(self) -> Dict[str, float]:
        """validate model va tinh metric"""
        pass

    @abc.abstractmethod
    def run(self) -> None:
        """chay toan bo qua trinh train"""
        pass

    @abc.abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """luu checkpoint cua model"""
        pass
