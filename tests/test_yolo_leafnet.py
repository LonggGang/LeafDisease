import unittest
import os
import torch
import yaml
from PIL import Image

from src.architectures import build_model
from src.architectures.detectors.yolo_leafnet import YOLOLeafNetDetector, C2f_BNDropout, A2C2f_BNDropout


class TestYOLOLeafNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # doc file config
        with open("configs/model.yaml", "r", encoding="utf-8") as f:
            cls.cfg = yaml.safe_load(f)
        
        # cau hinh cho dung kieu yolo leafnet
        cls.cfg["task"] = "detection"
        cls.cfg["architecture"] = "yolo_leafnet"
        cls.cfg["pretrained"] = False  # tat pretrained de test nhanh hon

    def test_1_instantiation(self):
        """test xem co khoi tao duoc model detector khong"""
        model = build_model(self.cfg)
        self.assertIsInstance(model, YOLOLeafNetDetector)
        self.assertEqual(model.num_classes, self.cfg["num_classes"])
        self.assertEqual(model.input_size, self.cfg["input_size"])

    def test_2_architecture_structure(self):
        """test xem khoi layer 8 co dung khong"""
        detector = build_model(self.cfg)
        pytorch_model = detector.model.model
        
        # check layer 8 co phai hang xin tu che khong
        layer_8 = pytorch_model.model[8]
        self.assertIsInstance(layer_8, A2C2f_BNDropout)
        self.assertIsInstance(layer_8.bn, torch.nn.BatchNorm2d)
        self.assertIsInstance(layer_8.dropout, torch.nn.Dropout)
        self.assertEqual(layer_8.dropout.p, self.cfg.get("dropout_rate", 0.5))

    def test_3_complexity(self):
        """test so luong tham so co hop ly khong"""
        detector = build_model(self.cfg)
        complexity = detector.get_complexity()
        
        self.assertIn("params_M", complexity)
        self.assertIn("flops_G", complexity)
        
        # check dung luong khoang 9.3 trieu tham so
        params_M = complexity["params_M"]
        self.assertTrue(9.0 <= params_M <= 9.5, f"Expected params_M to be ~9.3M, got {params_M:.2f}M")
        
        # check thong so flops
        self.assertIsInstance(complexity["flops_G"], float)

    def test_4_forward_pass(self):
        """test chay thu mot lan forward xem co loi khong"""
        detector = build_model(self.cfg)
        
        # nhan tensor dang batch channels height width
        dummy_input = torch.randn(1, 3, self.cfg["input_size"], self.cfg["input_size"])
        
        # chay forward mode eval
        detector.model.model.eval()
        with torch.no_grad():
            output = detector.forward(dummy_input)
            
        # yolo tra ve toa do predictions
        if isinstance(output, tuple):
            output = output[0]
            
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], self.cfg["num_classes"] + 4)

    def test_5_predict_pipeline(self):
        """test toan bo pipeline predict anh"""
        detector = build_model(self.cfg)
        
        # tao mot anh gia
        dummy_img_path = "test_dummy.jpg"
        img = Image.new("RGB", (640, 640), color="green")
        img.save(dummy_img_path)
        
        try:
            # chay du doan
            result = detector.predict(dummy_img_path)
            
            # check format ket qua
            self.assertIn("label", result)
            self.assertIn("confidence", result)
            self.assertIn("boxes", result)
            self.assertIn("inference_ms", result)
            
            self.assertIsInstance(result["label"], str)
            self.assertIsInstance(result["confidence"], float)
            self.assertIsInstance(result["boxes"], list)
            self.assertIsInstance(result["inference_ms"], float)
            
        finally:
            # xoa anh gia sau khi test
            if os.path.exists(dummy_img_path):
                os.remove(dummy_img_path)


if __name__ == "__main__":
    unittest.main()
