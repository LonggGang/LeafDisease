"""model cnn xin xo de phan loai benh la"""

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from src.architectures.base import BaseModel
from src.architectures.classifiers.base_classifier import BaseClassifier


class SEBlockWithLN(nn.Module):
    """khoi se de recalibrate dac trung"""

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        se_units = max(channels // reduction_ratio, 1)

        self.squeeze = nn.AdaptiveAvgPool2d(1)          # squeeze dac trung
        self.fc1     = nn.Linear(channels, se_units, bias=False)
        self.relu    = nn.ReLU(inplace=True)
        self.ln      = nn.LayerNorm(se_units)           # dung layer norm cho de
        self.fc2     = nn.Linear(se_units, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape

        # squeeze chieu
        z = self.squeeze(x).view(B, C)                  

        # tinh toan excitation
        z = self.fc1(z)
        z = self.relu(z)
        z = self.ln(z)
        z = self.fc2(z)
        z = self.sigmoid(z)

        # recalibrate lai weight
        z = z.view(B, C, 1, 1)
        return x * z


class GroupedDepthwiseBlock(nn.Module):
    """khoi depthwise separable conv chia group"""

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, num_groups: int = 4):
        super().__init__()

        # depthwise conv size 3
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.dw_bn   = nn.BatchNorm2d(in_channels)
        self.dw_relu = nn.ReLU(inplace=True)

        # grouped pointwise conv size 1
        actual_groups = self._safe_groups(in_channels, out_channels, num_groups)
        self.pw_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, padding=0,
            groups=actual_groups, bias=False
        )
        self.pw_bn   = nn.BatchNorm2d(out_channels)
        self.pw_relu = nn.ReLU(inplace=True)

    @staticmethod
    def _safe_groups(in_ch: int, out_ch: int, num_groups: int) -> int:
        """tim so group lon nhat ma chia het"""
        for g in range(num_groups, 0, -1):
            if in_ch % g == 0 and out_ch % g == 0:
                return g
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_relu(self.dw_bn(self.dw_conv(x)))
        x = self.pw_relu(self.pw_bn(self.pw_conv(x)))
        return x


class ResidualSEBlock(nn.Module):
    """khoi residual ket hop se"""

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, num_groups: int = 4,
                 reduction_ratio: int = 16):
        super().__init__()

        # duong chinh
        self.main = GroupedDepthwiseBlock(
            in_channels, out_channels, stride=stride, num_groups=num_groups
        )
        self.se = SEBlockWithLN(out_channels, reduction_ratio=reduction_ratio)

        # duong tat neu khac chieu
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=1, stride=stride, padding=0,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # chay qua main
        out = self.main(x)
        out = self.se(out)

        # chay shortcut
        shortcut = self.skip(x) if self.skip is not None else x

        return out + shortcut


class AdvancedCNNClassifier(BaseClassifier, nn.Module):
    """bo phan loai anh dung mobilenetv2 va se blocks"""

    BACKBONE_OUT_CHANNELS = 1280

    def __init__(self, cfg: Dict[str, Any]):
        nn.Module.__init__(self)

        self.num_classes  = cfg.get("num_classes", 38)
        self.dropout_rate = cfg.get("dropout_rate", 0.3)
        self.class_names: List[str] = cfg.get("class_names", [])
        input_size        = cfg.get("input_size", 224)

        # backbone mobilenetv2
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if cfg.get("pretrained", True) else None
        mobilenet = models.mobilenet_v2(weights=weights)

        # chi lay feature thoi
        self.backbone = mobilenet.features

        # dong bang backbone de train classifier truoc
        self.freeze_backbone()

        # cac block residual va se kem theo
        C = self.BACKBONE_OUT_CHANNELS
        self.block1 = ResidualSEBlock(C,   256, stride=1)
        self.block2 = ResidualSEBlock(256, 128, stride=1)

        # global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # cac lop fully connected lay ket qua
        self.fc1      = nn.Linear(128, 512)
        self.swish1   = nn.SiLU()
        self.drop1    = nn.Dropout(self.dropout_rate)

        self.fc2      = nn.Linear(512, 256)
        self.swish2   = nn.SiLU()
        self.drop2    = nn.Dropout(self.dropout_rate / 2)

        # lop cuoi ra 38 class
        self.classifier = nn.Linear(256, self.num_classes)

        self.preprocess = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def freeze_backbone(self) -> None:
        """dong bang backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers_to_unfreeze: int = 30) -> None:
        """mo bang backbone de finetune"""
        all_layers = list(self.backbone.children())
        for layer in all_layers:
            for param in layer.parameters():
                param.requires_grad = False

        layers_to_unfreeze = all_layers[-num_layers_to_unfreeze:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """chay forward qua mang neural"""
        # qua backbone
        x = self.backbone(x)

        # qua residual blocks
        x = self.block1(x)
        x = self.block2(x)

        # qua gap
        x = self.gap(x)
        x = x.flatten(1)

        # qua cac lop fc
        x = self.drop1(self.swish1(self.fc1(x)))
        x = self.drop2(self.swish2(self.fc2(x)))

        return self.classifier(x)

    def postprocess_logits(self, logits: torch.Tensor):
        """lay class index tu logits"""
        probs = F.softmax(logits, dim=-1)
        class_idx = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, class_idx].item())
        return confidence, class_idx

    def map_index_to_label(self, class_idx: int) -> str:
        if self.class_names and class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return str(class_idx)

    def predict(self, image_path: str) -> Dict[str, Any]:
        """du doan anh tu duong dan"""
        self.eval()
        device = next(self.parameters()).device

        img = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.forward(tensor)
        inference_ms = (time.perf_counter() - t0) * 1000

        confidence, class_idx = self.postprocess_logits(logits)
        label = self.map_index_to_label(class_idx)

        return {
            "label":        label,
            "confidence":   confidence,
            "boxes":        None,
            "inference_ms": inference_ms,
        }

    def get_complexity(self) -> Dict[str, float]:
        """tinh so tham so"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "params_M":           total / 1e6,
            "trainable_params_M": trainable / 1e6,
        }

    def export_onnx(self, output_path: str) -> None:
        """xuat model ra file onnx"""
        self.eval()
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            self, dummy, output_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=11,
        )
        print(f"Exported ONNX model to: {output_path}")


def _register_classifier(cfg: Dict[str, Any]) -> AdvancedCNNClassifier:
    """dang ky bo phan loai voi factory"""
    return AdvancedCNNClassifier(cfg)


if __name__ == "__main__":
    cfg = {
        "num_classes":  38,
        "input_size":   224,
        "dropout_rate": 0.3,
        "pretrained":   False,
    }

    model = AdvancedCNNClassifier(cfg)
    model.eval()

    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy)

    print("=" * 55)
    print("Advanced CNN Classifier — PyTorch Smoke Test")
    print("=" * 55)
    print(f"Input  shape : {tuple(dummy.shape)}")
    print(f"Output shape : {tuple(logits.shape)}  (expected: [2, 38])")
    complexity = model.get_complexity()
    print(f"Total params : {complexity['params_M']:.2f} M")
    print(f"Trainable    : {complexity['trainable_params_M']:.2f} M  (backbone frozen)")
    print("=" * 55)

    # unfreeze roi test thu xem chay duoc khong
    model.unfreeze_backbone(num_layers_to_unfreeze=5)
    complexity2 = model.get_complexity()
    print(f"Trainable (after unfreeze 5 layers): {complexity2['trainable_params_M']:.2f} M")
    print("Smoke test PASSED ✓")
