"""
Base classifier extending BaseModel for all classification architectures.
"""
import abc
from typing import Dict, Any

from src.architectures.base import BaseModel

class BaseClassifier(BaseModel):
    """
    Abstract base class for all classification models.
    """

    @abc.abstractmethod
    def postprocess_logits(self, logits: Any) -> Any:
        """Converts raw logits to probabilities or class indices."""
        pass

    @abc.abstractmethod
    def map_index_to_label(self, class_idx: int) -> str:
        """Maps an integer class index to its string label."""
        pass

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        End-to-end inference from a raw image file path.
        
        This method defines the shared prediction logic for all classifiers.
        Concrete input loading and timing would be integrated here.
        
        Returns:
            {
                "label": str,
                "confidence": float,
                "boxes": None,  # None for classifiers
                "inference_ms": float
            }
        """
        # Expected pipeline:
        # 1. Load and preprocess image
        # 2. logits = self.forward(image_tensor)
        # 3. probs, class_idx = self.postprocess_logits(logits)
        # 4. label = self.map_index_to_label(class_idx)
        raise NotImplementedError("Predict pipeline needs concrete image loading utilities.")
