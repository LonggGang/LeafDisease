"""
Advanced Lightweight CNN - Plant Disease Classifier (PyTorch)
=============================================================
Convert từ TensorFlow/Keras sang PyTorch, tích hợp vào cấu trúc project.

Paper: "Enhancing plant disease detection through deep learning:
        a Depthwise CNN with squeeze and excitation integration
        and residual skip connections"

Architecture:
    MobileNetV2 backbone (pretrained) → Modified Residual Blocks with SE → GAP → FC → Softmax

Usage:
    cfg = {
        "num_classes": 38,
        "input_size": 224,
        "dropout_rate": 0.3,
        "pretrained": True,
    }
    model = AdvancedCNNClassifier(cfg)
"""

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from src.architectures.base import BaseModel
from src.architectures.classifiers.base_classifier import BaseClassifier


# ─────────────────────────────────────────────────────────────────────────────
# 1. Squeeze-and-Excitation Block (với LayerNorm thay GroupNorm)
#    Paper section 3.3.2
# ─────────────────────────────────────────────────────────────────────────────

class SEBlockWithLN(nn.Module):
    """
    Squeeze-and-Excitation block với Layer Normalization.

    Công thức:
        z     = GlobalAvgPool(x)          # Squeeze
        s     = σ(LN(W2(δ(W1(z)))))     # Excitation
        x_hat = x ⊙ reshape(s)           # Recalibrate
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        se_units = max(channels // reduction_ratio, 1)

        self.squeeze = nn.AdaptiveAvgPool2d(1)          # (B, C, H, W) → (B, C, 1, 1)
        self.fc1     = nn.Linear(channels, se_units, bias=False)
        self.relu    = nn.ReLU(inplace=True)
        self.ln      = nn.LayerNorm(se_units)           # tương đương GroupNorm trong paper
        self.fc2     = nn.Linear(se_units, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape

        # Squeeze
        z = self.squeeze(x).view(B, C)                  # (B, C)

        # Excitation
        z = self.fc1(z)
        z = self.relu(z)
        z = self.ln(z)
        z = self.fc2(z)
        z = self.sigmoid(z)

        # Recalibrate
        z = z.view(B, C, 1, 1)
        return x * z


# ─────────────────────────────────────────────────────────────────────────────
# 2. Grouped Depthwise Separable Convolution
#    Paper section 3.3.1
# ─────────────────────────────────────────────────────────────────────────────

class GroupedDepthwiseBlock(nn.Module):
    """
    Modified Depthwise Separable Conv với grouped pointwise convolution.

    Với G=4 groups, cost giảm từ H×W×Cin×Cout xuống H×W×Cin×Cout/G.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, num_groups: int = 4):
        super().__init__()

        # Depthwise conv 3×3
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.dw_bn   = nn.BatchNorm2d(in_channels)
        self.dw_relu = nn.ReLU(inplace=True)

        # Grouped pointwise conv 1×1
        # Đảm bảo in_channels và out_channels chia hết cho num_groups
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
        """Tìm số group lớn nhất mà cả in_ch và out_ch đều chia hết."""
        for g in range(num_groups, 0, -1):
            if in_ch % g == 0 and out_ch % g == 0:
                return g
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_relu(self.dw_bn(self.dw_conv(x)))
        x = self.pw_relu(self.pw_bn(self.pw_conv(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. Residual Block với SE Module
#    Paper section 3.3.3 — y = DwiseConv(x) + SE(main_path)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualSEBlock(nn.Module):
    """
    Bottleneck residual block kết hợp SE module.

    Main path : GroupedDepthwiseSeparableConv → SE
    Skip path : DepthwiseConv1×1 + Conv1×1 (nếu dimensions thay đổi)
    Output    : main_path + skip_path
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, num_groups: int = 4,
                 reduction_ratio: int = 16):
        super().__init__()

        # Main path
        self.main = GroupedDepthwiseBlock(
            in_channels, out_channels, stride=stride, num_groups=num_groups
        )
        self.se = SEBlockWithLN(out_channels, reduction_ratio=reduction_ratio)

        # Skip connection (chỉ cần khi shape thay đổi)
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=1, stride=stride, padding=0,
                          groups=in_channels, bias=False),   # dwise 1×1
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),        # match channels
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path: grouped dwise → SE recalibration
        out = self.main(x)
        out = self.se(out)

        # Skip path
        shortcut = self.skip(x) if self.skip is not None else x

        return out + shortcut


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full Model
#    Paper section 3.3
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedCNNClassifier(BaseClassifier, nn.Module):
    """
    Advanced Lightweight CNN cho plant disease detection.

    Architecture:
        MobileNetV2 (frozen → fine-tune) →
        ResidualSEBlock(256) →
        ResidualSEBlock(128) →
        GlobalAvgPool →
        Dense(512, Swish, Dropout) →
        Dense(256, Swish, Dropout) →
        Softmax(num_classes)

    Args:
        cfg: dict với các key:
            - num_classes   (int,   default=38)
            - input_size    (int,   default=224)
            - dropout_rate  (float, default=0.3)
            - pretrained    (bool,  default=True)
            - class_names   (list,  optional)
    """

    # MobileNetV2 output channels = 1280
    BACKBONE_OUT_CHANNELS = 1280

    def __init__(self, cfg: Dict[str, Any]):
        nn.Module.__init__(self)

        self.num_classes  = cfg.get("num_classes", 38)
        self.dropout_rate = cfg.get("dropout_rate", 0.3)
        self.class_names: List[str] = cfg.get("class_names", [])
        input_size        = cfg.get("input_size", 224)

        # ── Backbone: MobileNetV2 ──────────────────────────────────────────
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if cfg.get("pretrained", True) else None
        mobilenet = models.mobilenet_v2(weights=weights)

        # Chỉ lấy features (bỏ classifier head)
        self.backbone = mobilenet.features           # output: (B, 1280, H/32, W/32)

        # Freeze toàn bộ backbone ban đầu (phase 1: feature extraction)
        self.freeze_backbone()

        # ── Modified Residual Blocks với SE ──────────────────────────────
        C = self.BACKBONE_OUT_CHANNELS
        self.block1 = ResidualSEBlock(C,   256, stride=1)
        self.block2 = ResidualSEBlock(256, 128, stride=1)

        # ── Global Average Pooling ────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)           # (B, 128, H, W) → (B, 128, 1, 1)

        # ── Fully Connected Layers (Swish + L2 reg + Dropout) ────────────
        # Paper section 3.3.4: f(x) = Swish(Wf·x + bf) + λ‖Wf‖²
        # L2 reg được handle qua weight_decay của optimizer
        self.fc1      = nn.Linear(128, 512)
        self.swish1   = nn.SiLU()                   # SiLU = Swish
        self.drop1    = nn.Dropout(self.dropout_rate)

        self.fc2      = nn.Linear(512, 256)
        self.swish2   = nn.SiLU()
        self.drop2    = nn.Dropout(self.dropout_rate / 2)

        # ── Output: Softmax 38 classes ────────────────────────────────────
        self.classifier = nn.Linear(256, self.num_classes)

        # Preprocessing transforms (ImageNet normalization)
        self.preprocess = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    # ── Backbone freeze / unfreeze helpers ───────────────────────────────

    def freeze_backbone(self) -> None:
        """Phase 1: Đóng băng backbone, chỉ train các layer mới thêm."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers_to_unfreeze: int = 30) -> None:
        """
        Phase 2: Mở đóng băng các layer cuối backbone để fine-tune.

        Args:
            num_layers_to_unfreeze: Số layer cuối cần unfreeze (đếm từ cuối).
        """
        all_layers = list(self.backbone.children())
        for layer in all_layers:
            for param in layer.parameters():
                param.requires_grad = False  # reset trước

        # Unfreeze các layer cuối
        layers_to_unfreeze = all_layers[-num_layers_to_unfreeze:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) - ảnh đã normalize
        Returns:
            logits: (B, num_classes)
        """
        # Backbone (MobileNetV2 features)
        x = self.backbone(x)                         # (B, 1280, 7, 7) với input 224

        # Modified Residual Blocks với SE
        x = self.block1(x)                           # (B, 256, 7, 7)
        x = self.block2(x)                           # (B, 128, 7, 7)

        # Global Average Pooling
        x = self.gap(x)                              # (B, 128, 1, 1)
        x = x.flatten(1)                             # (B, 128)

        # FC Layers
        x = self.drop1(self.swish1(self.fc1(x)))     # (B, 512)
        x = self.drop2(self.swish2(self.fc2(x)))     # (B, 256)

        return self.classifier(x)                    # (B, num_classes) — raw logits

    # ── BaseClassifier interface ──────────────────────────────────────────

    def postprocess_logits(self, logits: torch.Tensor):
        """Converts logits → (probabilities, class_index)."""
        probs = F.softmax(logits, dim=-1)
        class_idx = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, class_idx].item())
        return confidence, class_idx

    def map_index_to_label(self, class_idx: int) -> str:
        if self.class_names and class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return str(class_idx)

    def predict(self, image_path: str) -> Dict[str, Any]:
        """End-to-end inference từ đường dẫn ảnh."""
        self.eval()
        device = next(self.parameters()).device

        img = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(device)  # (1, 3, H, W)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.forward(tensor)
        inference_ms = (time.perf_counter() - t0) * 1000

        confidence, class_idx = self.postprocess_logits(logits)
        label = self.map_index_to_label(class_idx)

        return {
            "label":        label,
            "confidence":   confidence,
            "boxes":        None,        # classifier không có bbox
            "inference_ms": inference_ms,
        }

    def get_complexity(self) -> Dict[str, float]:
        """Đếm params (triệu)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "params_M":           total / 1e6,
            "trainable_params_M": trainable / 1e6,
        }

    def export_onnx(self, output_path: str) -> None:
        """Export model sang ONNX để deploy edge."""
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


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tích hợp vào factory (thêm vào src/architectures/__init__.py)
# ─────────────────────────────────────────────────────────────────────────────

def _register_classifier(cfg: Dict[str, Any]) -> AdvancedCNNClassifier:
    """
    Factory helper — gọi từ build_model() trong src/architectures/__init__.py.

    Thêm vào __init__.py:
        elif task == "classification":
            if arch == "advanced_cnn":
                from src.architectures.classifiers.advanced_cnn_classifier import _register_classifier
                return _register_classifier(cfg)
    """
    return AdvancedCNNClassifier(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = {
        "num_classes":  38,
        "input_size":   224,
        "dropout_rate": 0.3,
        "pretrained":   False,   # False để test không cần download weights
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

    # Unfreeze và kiểm tra lại
    model.unfreeze_backbone(num_layers_to_unfreeze=5)
    complexity2 = model.get_complexity()
    print(f"Trainable (after unfreeze 5 layers): {complexity2['trainable_params_M']:.2f} M")
    print("Smoke test PASSED ✓")
