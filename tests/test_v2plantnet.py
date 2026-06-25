import unittest
import os
import torch
from PIL import Image

from src.architectures import build_model
from src.architectures.classifiers.v2plantnet_classifier import V2PlantNet


class TestV2PlantNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = {
            "task": "classification",
            "architecture": "v2plantnet",
            "num_classes": 38,
            "class_names": [f"class_{i}" for i in range(38)],
            "input_size": 224
        }

    def test_1_instantiation(self):
        """Test if V2PlantNet can be instantiated via build_model factory."""
        model = build_model(self.cfg)
        self.assertIsInstance(model, V2PlantNet)
        self.assertEqual(model.num_classes, self.cfg["num_classes"])
        self.assertEqual(model.input_size, self.cfg["input_size"])

    def test_2_complexity(self):
        """Verify that the model complexity returns valid values."""
        model = build_model(self.cfg)
        complexity = model.get_complexity()
        
        self.assertIn("params_M", complexity)
        self.assertIn("flops_G", complexity)
        self.assertEqual(complexity["params_M"], 0.379)
        self.assertEqual(complexity["flops_G"], 0.182741)

    def test_3_forward_pass(self):
        """Test model forward pass with dummy input."""
        model = build_model(self.cfg)
        dummy_input = torch.randn(2, 3, self.cfg["input_size"], self.cfg["input_size"])
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            
        self.assertEqual(output.shape, (2, self.cfg["num_classes"]))

    def test_4_predict_pipeline(self):
        """Test predictability using a dummy image path."""
        model = build_model(self.cfg)
        
        dummy_img_path = "test_dummy_v2plantnet.jpg"
        img = Image.new("RGB", (256, 256), color="green")
        img.save(dummy_img_path)
        
        try:
            result = model.predict(dummy_img_path)
            self.assertIn("label", result)
            self.assertIn("confidence", result)
            self.assertIn("boxes", result)
            self.assertIn("inference_ms", result)
            
            self.assertIsInstance(result["label"], str)
            self.assertIsInstance(result["confidence"], float)
            self.assertIsNone(result["boxes"])
            self.assertIsInstance(result["inference_ms"], float)
        finally:
            if os.path.exists(dummy_img_path):
                os.remove(dummy_img_path)


if __name__ == "__main__":
    unittest.main()
