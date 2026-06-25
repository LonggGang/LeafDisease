"""lop cha cho tat ca model"""
import abc
from typing import Dict, Any, Optional

class BaseModel(abc.ABC):
    """lop cha de viet tiep cac model khac"""

    @abc.abstractmethod
    def forward(self, x: Any) -> Any:
        """chay model qua mang neural"""
        pass

    @abc.abstractmethod
    def predict(self, image_path: str) -> Dict[str, Any]:
        """du doan anh tu duong dan"""
        pass

    @abc.abstractmethod
    def get_complexity(self) -> Dict[str, float]:
        """lay so tham so va flops"""
        pass

    def export_onnx(self, output_path: str) -> None:
        """xuat model ra file onnx"""
        raise NotImplementedError("ONNX export not implemented for this model.")

