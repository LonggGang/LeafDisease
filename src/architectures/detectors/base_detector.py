"""
Base detector extending BaseModel for all object detection architectures.
"""
import abc
from typing import Dict, Any, List

from src.architectures.base import BaseModel

class BaseDetector(BaseModel):
    """
    Abstract base class for all detection models.
    """

    @abc.abstractmethod
    def apply_nms(self, predictions: Any) -> Any:
        """Applies Non-Maximum Suppression to filter bounding boxes."""
        pass

    @abc.abstractmethod
    def decode_boxes(self, raw_boxes: Any) -> Any:
        """Decodes model-specific box outputs to absolute image coordinates."""
        pass

    @abc.abstractmethod
    def filter_by_confidence(self, boxes: Any, conf_threshold: float) -> Any:
        """Filters out boxes below the given confidence threshold."""
        pass

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        End-to-end inference from a raw image file path.
        
        This method defines the shared prediction logic for all detectors.
        Concrete input loading and timing would be integrated here.
        
        Returns:
            {
                "label": str,
                "confidence": float,
                "boxes": list[dict],
                "inference_ms": float
            }
        """
        # Expected pipeline:
        # 1. Load and preprocess image
        # 2. raw_preds = self.forward(image_tensor)
        # 3. decoded_boxes = self.decode_boxes(raw_preds)
        # 4. nms_boxes = self.apply_nms(decoded_boxes)
        # 5. final_boxes = self.filter_by_confidence(nms_boxes, conf_threshold)
        raise NotImplementedError("Predict pipeline needs concrete image loading utilities.")
