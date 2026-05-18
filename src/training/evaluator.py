"""
Abstract base class for model evaluation and metrics computation.
"""
import abc
from typing import Dict, Any

class BaseEvaluator(abc.ABC):
    """
    Abstract base class for evaluating models.
    """

    @abc.abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """
        Computes all evaluation metrics including accuracy/mAP, FPS, and FLOPs.
        Returns a metrics dictionary.
        """
        pass
