# !pip install thop
"""
Architecture: Input -> Conv + BN + ReLU + Depthwise Separable Blcoks -> Global Average Pooling -> Dense -> Dropout -> Classifier
"""

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

"""
DepthwiseSeparableBlock is a lightweight convolution block that replaces a standard convolution with two operations:
Depthwise Convolution – applies a separate 3×3 filter to each input channel to extract spatial features.
Pointwise Convolution (1×1) – combines information across channels and produces the desired number of output channels.
This design significantly reduces the number of parameters and computations compared to a standard convolution while maintaining good feature extraction capability, making it suitable for lightweight models such as V2PlantNet.
"""
class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            # Depthwise
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

            # Pointwise
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

    def __init__(self, num_classes: int, class_map: Dict[int, str]):
        super().__init__()

        self.class_map = class_map
        self.input_size = 224
        # Initial layers
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

        # Stage 1
        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(32, 64, stride=1),
            DepthwiseSeparableBlock(64, 64, stride=1),
            DepthwiseSeparableBlock(64, 64, stride=1),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(64, 128, stride=2),

            *[
                DepthwiseSeparableBlock(128, 128, stride=1)
                for _ in range(6)
            ]
        )

        # Stage 3
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

            nn.Linear(256, num_classes)
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

        return self.class_map[class_idx]

    def predict(self, image_path: str):

        transform = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
        ])

        image = Image.open(image_path).convert("RGB")

        tensor = transform(image).unsqueeze(0)

        self.eval()

        start = time.perf_counter()

        with torch.no_grad():
            logits = self.forward(tensor)

        inference_ms = (time.perf_counter() - start) * 1000

        _, class_idx, confidence = self.postprocess_logits(logits)

        label = self.map_index_to_label(class_idx.item())

        return {
            "label": label,
            "confidence": confidence,
            "boxes": None,
            "inference_ms": inference_ms
      }
    def get_complexity(self) -> Dict[str, float]:
        self.eval()
        # Parameter count
        params = sum(
            p.numel()
            for p in self.parameters()
        )

        params_M = params / 1_000_000

        # Dummy input
        dummy = torch.randn(1, 3, self.input_size, self.input_size)

        # FLOPs calculation
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
"""
Initialize to get parameters
"""
if __name__ == "__main__":
    model = V2PlantNet(
        num_classes=38,
        class_map = {
        i: f"class_{i}"
        for i in range(38)}
    )

    print(model.get_complexity())
