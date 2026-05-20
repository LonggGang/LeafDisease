"""
Main entry point for the plant leaf disease detection system.
Handles training, evaluation, and single-image inference.
"""

import os
import yaml
import argparse
import logging
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("LeafDiseaseMain")


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Plant Leaf Disease Detection Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "predict"],
        help="Pipeline execution mode: 'train', 'eval' (evaluate on test set), or 'predict' (single image inference)"
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="configs/model.yaml",
        help="Path to the model architecture configuration YAML file"
    )
    parser.add_argument(
        "--train_cfg",
        type=str,
        default="configs/train.yaml",
        help="Path to the training hyperparameters configuration YAML file"
    )
    parser.add_argument(
        "--augment_cfg",
        type=str,
        default="configs/augment.yaml",
        help="Path to the data augmentation configuration YAML file"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the target image file for single-image prediction (required in 'predict' mode)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint file (.pth) to load for evaluation or prediction"
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    # 1. Load configurations
    cfg_model = load_config(args.model_cfg)
    cfg_train = load_config(args.train_cfg)
    cfg_aug = load_config(args.augment_cfg)

    # Đảm bảo thư mục lưu checkpoint tồn tại
    checkpoint_dir = cfg_train.get("checkpoint_dir", "checkpoints/")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Xử lý logic theo chế độ chạy (Mode)
    if args.mode == "train":
        logger.info("Starting TRAINING mode...")

        # Import các modules từ src (sẽ ném lỗi nếu chưa được implement concrete class)
        try:
            # Nhắc nhở người dùng nếu chưa triển khai cụ thể các class
            from src.dataprocessing.augmentation import BaseTransform
            from src.dataprocessing.dataset import BasePlantDataset
            from src.training.trainer import BaseTrainer
            # Lưu ý: Khi người dùng code cụ thể, các import này sẽ là các class concrete như:
            # from src.dataprocessing.augmentation import PlantAugmentation
            # from src.dataprocessing.dataset import PotatoLeafDataset
            # from src.training.trainer import PlantTrainer
            # và import hàm factory build_model
            # from src.architectures import build_model
        except ImportError as e:
            logger.error(f"Failed to import src modules. Have they been implemented yet? Details: {e}")
            raise

        logger.info("Building data augmentation and datasets...")
        # Minh họa luồng dữ liệu:
        # train_transforms = PlantAugmentation(cfg_aug, split="train")
        # val_transforms = PlantAugmentation(cfg_aug, split="val")
        #
        # train_dataset = PotatoLeafDataset(csv_path="data/splits/train.csv", transforms=train_transforms)
        # val_dataset = PotatoLeafDataset(csv_path="data/splits/val.csv", transforms=val_transforms)
        #
        # train_loader = DataLoader(train_dataset, batch_size=cfg_train["batch_size"], shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=cfg_train["batch_size"], shuffle=False)
        #
        # logger.info(f"Building model: {cfg_model['architecture']}")
        # model = build_model(cfg_model)
        #
        # trainer = PlantTrainer(model, train_loader, val_loader, cfg_train)
        # trainer.run()
        
        logger.warning(
            "Triển khai giả lập thành công! Hãy cài đặt cụ thể các lớp kế thừa "
            "trong thư mục `src/` trước khi tiến hành chạy huấn luyện thực tế."
        )

    elif args.mode == "eval":
        logger.info("Starting EVALUATION mode...")
        if not args.checkpoint:
            logger.warning("No checkpoint provided via --checkpoint. Evaluation might run on random weights!")

        try:
            from src.training.evaluator import BaseEvaluator
            # Concrete implementation:
            # from src.training.evaluator import PlantEvaluator
            # from src.architectures import build_model
        except ImportError as e:
            logger.error(f"Failed to import evaluator modules. Details: {e}")
            raise

        # Minh họa luồng đánh giá:
        # test_transforms = PlantAugmentation(cfg_aug, split="val")
        # test_dataset = PotatoLeafDataset(csv_path="data/splits/test.csv", transforms=test_transforms)
        # test_loader = DataLoader(test_dataset, batch_size=cfg_train["batch_size"], shuffle=False)
        #
        # model = build_model(cfg_model)
        # if args.checkpoint:
        #     model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        #
        # evaluator = PlantEvaluator(model, test_loader)
        # results = evaluator.evaluate()
        # logger.info(f"Evaluation Results: {results}")

        logger.warning("Đã phác thảo luồng Evaluation. Hãy triển khai lớp Evaluator cụ thể để chạy.")

    elif args.mode == "predict":
        logger.info("Starting PREDICTION mode...")
        if not args.image_path:
            raise ValueError("You must specify --image_path in predict mode.")
        if not args.checkpoint:
            logger.warning("No checkpoint provided via --checkpoint. Model will run using default weights.")

        try:
            # Concrete model build
            # from src.architectures import build_model
            pass
        except ImportError as e:
            logger.error(f"Failed to import model building factory. Details: {e}")
            raise

        logger.info(f"Running inference on image: {args.image_path}")
        # Minh họa luồng dự đoán ảnh đơn lẻ:
        # model = build_model(cfg_model)
        # if args.checkpoint:
        #     model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        # model.eval()
        #
        # output = model.predict(args.image_path)
        # logger.info(f"Prediction output: {output}")

        logger.warning("Đã phác thảo luồng Predict. Hãy hoàn thiện model.predict() để lấy dự đoán của ảnh.")


def main():
    args = parse_args()
    try:
        run_pipeline(args)
    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
        exit(1)


if __name__ == "__main__":
    main()
