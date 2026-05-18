"""
Abstract base class defining the contract for all models (classifiers and detectors).
"""
import abc
from typing import Dict, Any, Optional

class BaseModel(abc.ABC):
    """
    Abstract base class for all neural network models in the project.
    """

    @abc.abstractmethod
    def forward(self, x: Any) -> Any:
        """Standard PyTorch forward pass."""
        pass

    @abc.abstractmethod
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        End-to-end inference from a raw image file path.
        Returns:
            {
                "label": str,
                "confidence": float,
                "boxes": list[dict] | None,  # None for classifiers
                "inference_ms": float
            }
        """
        pass

    @abc.abstractmethod
    def get_complexity(self) -> Dict[str, float]:
        """
        Returns:
            {
                "params_M": float,
                "flops_G": float
            }
        """
        pass

    def export_onnx(self, output_path: str) -> None:
        """Optional but encouraged for edge deployment."""
        raise NotImplementedError("ONNX export not implemented for this model.")
