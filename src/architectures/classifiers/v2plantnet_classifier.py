"""mang v2plantnet sieu nhe"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import time
from PIL import Image
import torchvision.transforms as T

import abc
from typing import Dict, Any, Optional

from src.architectures.base import BaseModel
from src.architectures.classifiers.base_classifier import BaseClassifier


class DepthwiseSeparableBlock(nn.Module):
    """khoi depthwise separable conv cho nhe"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            # depthwise conv
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # pointwise conv
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class V2PlantNet(nn.Module, BaseClassifier):
    """lop de train va du doan v2plantnet"""

    def __init__(self, cfg_or_num_classes: Any = 38, class_map: Optional[Dict[int, str]] = None):
        super().__init__()

        if isinstance(cfg_or_num_classes, dict):
            cfg = cfg_or_num_classes
            self.num_classes = cfg.get("num_classes", 38)
            self.class_names = cfg.get("class_names", [])
            if self.class_names:
                self.class_map = {i: name for i, name in enumerate(self.class_names)}
            else:
                self.class_map = {i: f"class_{i}" for i in range(self.num_classes)}
            self.input_size = cfg.get("input_size", 224)
        else:
            self.num_classes = cfg_or_num_classes
            self.class_map = class_map or {i: f"class_{i}" for i in range(self.num_classes)}
            self.class_names = [self.class_map[i] for i in sorted(self.class_map.keys())]
            self.input_size = 224

        # cac lop khoi dau
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        # stage 1
        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(32, 64, stride=1),
            DepthwiseSeparableBlock(64, 64, stride=1),
            DepthwiseSeparableBlock(64, 64, stride=1),
        )

        # stage 2
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(64, 128, stride=2),

            *[
                DepthwiseSeparableBlock(128, 128, stride=1)
                for _ in range(6)
            ]
        )

        # stage 3
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(128, 256, stride=2),

            DepthwiseSeparableBlock(256, 256, stride=1),
            DepthwiseSeparableBlock(256, 256, stride=1),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, 256),

            nn.Dropout(0.45),

            nn.Linear(256, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def postprocess_logits(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=1)
        confidence, class_idx = torch.max(probs, dim=1)
        return probs, class_idx, confidence

    def map_index_to_label(self, class_idx: int) -> str:
        if self.class_names and class_idx < len(self.class_names):
            return self.class_names[class_idx]
        if self.class_map and class_idx in self.class_map:
            return self.class_map[class_idx]
        return str(class_idx)

    def predict(self, image_path: str):
        self.eval()
        device = next(self.parameters()).device

        transform = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
        ])

        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            logits = self.forward(tensor)
        inference_ms = (time.perf_counter() - start) * 1000

        _, class_idx, confidence = self.postprocess_logits(logits.cpu())
        label = self.map_index_to_label(class_idx.item())

        return {
            "label": label,
            "confidence": confidence.item() if isinstance(confidence, torch.Tensor) else float(confidence),
            "boxes": None,
            "inference_ms": inference_ms
        }

    def get_complexity(self) -> Dict[str, float]:
        self.eval()
        device = next(self.parameters()).device
        
        # dem parameter
        params = sum(
            p.numel()
            for p in self.parameters()
        )
        params_M = params / 1_000_000

        # tao du lieu gia
        dummy = torch.randn(1, 3, self.input_size, self.input_size).to(device)

        # tinh flops
        flops, _ = profile(
            self,
            inputs=(dummy,),
            verbose=False
        )
        flops_G = flops / 1_000_000_000

        return {
            "params_M": round(params_M, 3),
            "flops_G": round(flops_G, 6)
        }


def _register_classifier(cfg: Dict[str, Any]) -> V2PlantNet:
    """helper dang ky model voi factory"""
    return V2PlantNet(cfg)


if __name__ == "__main__":
    # chay thu model
    model = V2PlantNet(
        cfg_or_num_classes=38,
        class_map = {
        i: f"class_{i}"
        for i in range(38)}
    )

    print(model.get_complexity())
