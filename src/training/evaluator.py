"""lop cha de danh gia model"""
import abc
from typing import Dict, Any

class BaseEvaluator(abc.ABC):
    """lop cha de viet cac lop evaluate sau nay"""

    @abc.abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """tinh toan tat ca cac metric danh gia"""
        pass
