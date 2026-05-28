import unittest
import os
import torch
import yaml
from PIL import Image

from src.architectures import build_model
from src.architectures.detectors.yolo_leafnet import YOLOLeafNetDetector, C2f_BNDropout


class TestYOLOLeafNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the configuration
        with open("configs/model.yaml", "r", encoding="utf-8") as f:
            cls.cfg = yaml.safe_load(f)
        
        # Ensure we are testing the yolo_leafnet detector
        cls.cfg["task"] = "detection"
        cls.cfg["architecture"] = "yolo_leafnet"
        cls.cfg["pretrained"] = False  # Set to False to speed up tests and avoid internet downloads

    def test_1_instantiation(self):
        """Test if the YOLOLeafNetDetector can be instantiated via the build_model factory."""
        model = build_model(self.cfg)
        self.assertIsInstance(model, YOLOLeafNetDetector)
        self.assertEqual(model.num_classes, self.cfg["num_classes"])
        self.assertEqual(model.input_size, self.cfg["input_size"])

    def test_2_architecture_structure(self):
        """Verify that C2f_BNDropout block and SPPF are positioned correctly in the backbone."""
        detector = build_model(self.cfg)
        pytorch_model = detector.model.model  # The underlying PyTorch DetectionModel
        
        # Check that layer 8 is our custom C2f_BNDropout block
        layer_8 = pytorch_model.model[8]
        self.assertIsInstance(layer_8, C2f_BNDropout)
        self.assertIsInstance(layer_8.bn, torch.nn.BatchNorm2d)
        self.assertIsInstance(layer_8.dropout, torch.nn.Dropout)
        self.assertEqual(layer_8.dropout.p, self.cfg.get("dropout_rate", 0.5))

        # Check that layer 9 is the SPPF layer
        from ultralytics.nn.modules import SPPF
        layer_9 = pytorch_model.model[9]
        self.assertIsInstance(layer_9, SPPF)

    def test_3_complexity(self):
        """Verify that the model complexity is within expectations (~11.1M parameters)."""
        detector = build_model(self.cfg)
        complexity = detector.get_complexity()
        
        self.assertIn("params_M", complexity)
        self.assertIn("flops_G", complexity)
        
        # YOLOv8s has ~11.1M parameters. Our modifications add BN and Dropout (very few additional parameters).
        # We expect the parameter count to be around 11.1M to 11.3M.
        params_M = complexity["params_M"]
        self.assertTrue(11.0 <= params_M <= 11.5, f"Expected params_M to be ~11.1M, got {params_M:.2f}M")
        
        # Verify that FLOPs is computed or returned as float
        self.assertIsInstance(complexity["flops_G"], float)

    def test_4_forward_pass(self):
        """Test model forward pass with a dummy input tensor."""
        detector = build_model(self.cfg)
        
        # YOLOv8 expects float tensor of shape (batch_size, channels, height, width)
        dummy_input = torch.randn(1, 3, self.cfg["input_size"], self.cfg["input_size"])
        
        # Run forward pass (set to eval mode to get inference output format)
        detector.model.model.eval()
        with torch.no_grad():
            output = detector.forward(dummy_input)
            
        # Standard YOLOv8 detection forward pass in eval mode returns a tuple/tensor of predictions
        # Shape of predictions is typically (batch_size, num_classes + 4, num_anchors)
        if isinstance(output, tuple):
            output = output[0]
            
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], self.cfg["num_classes"] + 4)

    def test_5_predict_pipeline(self):
        """Test the end-to-end predict pipeline using a dummy image file."""
        detector = build_model(self.cfg)
        
        # Create a dummy image
        dummy_img_path = "test_dummy.jpg"
        img = Image.new("RGB", (640, 640), color="green")
        img.save(dummy_img_path)
        
        try:
            # Run prediction
            result = detector.predict(dummy_img_path)
            
            # Verify output dictionary format
            self.assertIn("label", result)
            self.assertIn("confidence", result)
            self.assertIn("boxes", result)
            self.assertIn("inference_ms", result)
            
            self.assertIsInstance(result["label"], str)
            self.assertIsInstance(result["confidence"], float)
            self.assertIsInstance(result["boxes"], list)
            self.assertIsInstance(result["inference_ms"], float)
            
        finally:
            # Clean up dummy image
            if os.path.exists(dummy_img_path):
                os.remove(dummy_img_path)


if __name__ == "__main__":
    unittest.main()
