"""lop classifier co ban"""
import abc
from typing import Dict, Any

from src.architectures.base import BaseModel

class BaseClassifier(BaseModel):
    """lop cha cho cac phan loai"""

    @abc.abstractmethod
    def postprocess_logits(self, logits: Any) -> Any:
        """bien doi ket qua logits"""
        pass

    @abc.abstractmethod
    def map_index_to_label(self, class_idx: int) -> str:
        """chuyen chi so sang ten nhan"""
        pass

    def predict(self, image_path: str) -> Dict[str, Any]:
        """du doan anh tu duong dan"""
        # load va xu ly anh
        # chay model de lay logits
        # lay xac suat va nhan
        # lay nhan cuoi cung
        raise NotImplementedError("Predict pipeline needs concrete image loading utilities.")

