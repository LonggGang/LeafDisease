"""lop detector co ban"""
import abc
from typing import Dict, Any, List

from src.architectures.base import BaseModel

class BaseDetector(BaseModel):
    """lop cha cho cac phat hien doi tuong"""

    @abc.abstractmethod
    def apply_nms(self, predictions: Any) -> Any:
        """ap dung non-maximum suppression"""
        pass

    @abc.abstractmethod
    def decode_boxes(self, raw_boxes: Any) -> Any:
        """giai ma toa do hop"""
        pass

    @abc.abstractmethod
    def filter_by_confidence(self, boxes: Any, conf_threshold: float) -> Any:
        """loc bot hop qua nguong tin cay"""
        pass

    def predict(self, image_path: str) -> Dict[str, Any]:
        """du doan anh tu duong dan"""
        # load va xu ly anh
        # chay model de lay box tho
        # giai ma cac hop prediction
        # ap dung nms
        # loc hop tin cay
        raise NotImplementedError("Predict pipeline needs concrete image loading utilities.")

