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
        "--data",
        type=str,
        default=None,
        help="Path to the standard YOLO dataset.yaml config file (overrides train config)"
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
    # CLI Overrides for model & training
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["classification", "detection"],
        help="Override project task type ('classification' or 'detection')"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="Override model architecture (e.g. yolov8n, yolov8s, yolo_leafnet)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Override number of classes"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Override pretrained flag ('true' or 'false')"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=None,
        help="Override model input image size (imgsz)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="Override optimizer (e.g. AdamW, SGD)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override checkpoint directory"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint specified by --checkpoint"
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    # 1. Load configurations
    cfg_model = load_config(args.model_cfg)
    cfg_train = load_config(args.train_cfg)
    cfg_aug = load_config(args.augment_cfg)

    # Apply CLI overrides to model and training configs
    if args.task:
        cfg_model["task"] = args.task
    if args.architecture:
        cfg_model["architecture"] = args.architecture
    if args.num_classes is not None:
        cfg_model["num_classes"] = args.num_classes
    if args.pretrained is not None:
        cfg_model["pretrained"] = args.pretrained.lower() in ("true", "1", "yes")
    if args.input_size is not None:
        cfg_model["input_size"] = args.input_size

    if args.epochs is not None:
        cfg_train["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg_train["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg_train["lr"] = args.lr
    if args.optimizer:
        cfg_train["optimizer"] = args.optimizer
    if args.checkpoint_dir:
        cfg_train["checkpoint_dir"] = args.checkpoint_dir
    if args.resume:
        cfg_train["resume"] = True

    # Đảm bảo thư mục lưu checkpoint tồn tại
    checkpoint_dir = cfg_train.get("checkpoint_dir", "checkpoints/")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Xử lý logic theo chế độ chạy (Mode)
    task = cfg_model.get("task", "classification")

    if task == "detection":
        if args.mode == "train":
            logger.info("Starting DETECTION TRAINING mode...")

            from src.architectures import build_model
            from src.training import YOLOLeafNetTrainer

            data_path = args.data or cfg_train.get("data", "data/dataset.yaml")
            logger.info(f"Building detection model: {cfg_model['architecture']}")
            model = build_model(cfg_model)

            if args.checkpoint:
                if args.resume:
                    logger.info(f"Resuming detection training from checkpoint: {args.checkpoint}")
                    from ultralytics import YOLO
                    model.model = YOLO(args.checkpoint)
                else:
                    logger.info(f"Loading weights from checkpoint {args.checkpoint} to initialize new training/fine-tuning run...")
                    if args.checkpoint.endswith(".pt"):
                        try:
                            from ultralytics import YOLO
                            model.model = YOLO(args.checkpoint)
                            logger.info("Successfully loaded native YOLO checkpoint weights.")
                        except Exception as e:
                            logger.warning(f"Could not load native YOLO checkpoint: {e}. Trying state dict loading.")
                            state_dict = torch.load(args.checkpoint, map_location="cpu")
                            if isinstance(state_dict, dict) and "model" in state_dict:
                                state_dict = state_dict["model"]
                            model.model.model.load_state_dict(state_dict, strict=False)
                    else:
                        state_dict = torch.load(args.checkpoint, map_location="cpu")
                        if isinstance(state_dict, dict) and "model" in state_dict:
                            state_dict = state_dict["model"]
                        model.model.model.load_state_dict(state_dict, strict=False)
                        logger.info("Successfully loaded state dict weights.")

            trainer = YOLOLeafNetTrainer(model, cfg_train, data_path=data_path)
            trainer.run()

        elif args.mode == "eval":
            logger.info("Starting DETECTION EVALUATION mode...")

            from src.architectures import build_model
            from src.training import YOLOLeafNetEvaluator

            data_path = args.data or cfg_train.get("data", "data/dataset.yaml")
            logger.info(f"Building detection model: {cfg_model['architecture']}")
            model = build_model(cfg_model)

            if args.checkpoint:
                logger.info(f"Loading checkpoint from {args.checkpoint}...")
                if args.checkpoint.endswith(".pt"):
                    try:
                        from ultralytics import YOLO
                        model.model = YOLO(args.checkpoint)
                        logger.info("Successfully loaded native YOLO checkpoint.")
                    except Exception as e:
                        logger.warning(f"Could not load native YOLO checkpoint: {e}. Trying state dict loading.")
                        state_dict = torch.load(args.checkpoint, map_location="cpu")
                        if isinstance(state_dict, dict) and "model" in state_dict:
                            state_dict = state_dict["model"]
                        model.model.model.load_state_dict(state_dict, strict=False)
                else:
                    state_dict = torch.load(args.checkpoint, map_location="cpu")
                    if isinstance(state_dict, dict) and "model" in state_dict:
                        state_dict = state_dict["model"]
                    model.model.model.load_state_dict(state_dict, strict=False)
                    logger.info("Successfully loaded state dict checkpoint.")
            else:
                logger.warning("No checkpoint provided via --checkpoint. Evaluation might run on random/default weights!")

            evaluator = YOLOLeafNetEvaluator(model, data_path=data_path, split="val")
            results = evaluator.evaluate()
            logger.info(f"Evaluation Results: {results}")

        elif args.mode == "predict":
            logger.info("Starting DETECTION PREDICTION mode...")
            if not args.image_path:
                raise ValueError("You must specify --image_path in predict mode.")

            from src.architectures import build_model
            model = build_model(cfg_model)

            if args.checkpoint:
                logger.info(f"Loading checkpoint from {args.checkpoint}...")
                if args.checkpoint.endswith(".pt"):
                    try:
                        from ultralytics import YOLO
                        model.model = YOLO(args.checkpoint)
                        logger.info("Successfully loaded native YOLO checkpoint.")
                    except Exception as e:
                        logger.warning(f"Could not load native YOLO checkpoint: {e}. Trying state dict loading.")
                        state_dict = torch.load(args.checkpoint, map_location="cpu")
                        if isinstance(state_dict, dict) and "model" in state_dict:
                            state_dict = state_dict["model"]
                        model.model.model.load_state_dict(state_dict, strict=False)
                else:
                    state_dict = torch.load(args.checkpoint, map_location="cpu")
                    if isinstance(state_dict, dict) and "model" in state_dict:
                        state_dict = state_dict["model"]
                    model.model.model.load_state_dict(state_dict, strict=False)
                    logger.info("Successfully loaded state dict checkpoint.")
            else:
                logger.warning("No checkpoint provided via --checkpoint. Model will run using default weights.")

            logger.info(f"Running inference on image: {args.image_path}")
            output = model.predict(args.image_path)
            logger.info(f"Prediction output: {output}")

    elif task == "classification":
        data_path = args.data or cfg_train.get("data", "data/")
        # If a dataset YAML file is passed, extract the root directory path
        if data_path.endswith(".yaml") or data_path.endswith(".yml"):
            try:
                import yaml
                with open(data_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f)
                if yaml_data and "path" in yaml_data:
                    data_path = yaml_data["path"]
                    logger.info(f"Parsed classification root directory from YAML: {data_path}")
            except Exception as e:
                logger.warning(f"Could not parse YAML for classification path: {e}. Using raw path: {data_path}")

        if args.mode == "train":
            logger.info("Starting CLASSIFICATION TRAINING mode...")

            from src.architectures import build_model
            from src.training import CNNClassifierTrainer

            logger.info(f"Building classification model: {cfg_model['architecture']}")
            model = build_model(cfg_model)

            if args.checkpoint:
                logger.info(f"Loading weights from checkpoint {args.checkpoint} to initialize new training/fine-tuning run...")
                checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
                
                if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                    state_dict = checkpoint_data["model_state_dict"]
                else:
                    state_dict = checkpoint_data
                
                # Tự động loại bỏ layer classifier cuối cùng nếu số lượng class khác nhau
                if "classifier.weight" in state_dict:
                    ckpt_classes = state_dict["classifier.weight"].shape[0]
                    model_classes = model.classifier.weight.shape[0]
                    if ckpt_classes != model_classes:
                        logger.info(f"Class mismatch detected: checkpoint has {ckpt_classes} classes, "
                                    f"but model has {model_classes} classes. Dropping classifier head weights for transfer learning.")
                        state_dict.pop("classifier.weight", None)
                        state_dict.pop("classifier.bias", None)
                
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                logger.info(f"Successfully loaded classifier weights (strict=False). "
                            f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

            trainer = CNNClassifierTrainer(model, cfg_train, data_path=data_path, cfg_aug=cfg_aug)
            trainer.run()

        elif args.mode == "eval":
            logger.info("Starting CLASSIFICATION EVALUATION mode...")

            from src.architectures import build_model
            from src.training import CNNClassifierEvaluator

            logger.info(f"Building classification model: {cfg_model['architecture']}")
            model = build_model(cfg_model)

            if args.checkpoint:
                logger.info(f"Loading checkpoint from {args.checkpoint}...")
                checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
                if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                    model.load_state_dict(checkpoint_data["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint_data)
                logger.info("Successfully loaded classifier checkpoint.")
            else:
                logger.warning("No checkpoint provided via --checkpoint. Evaluation will run on random/default weights!")

            evaluator = CNNClassifierEvaluator(model, data_path=data_path, split="val")
            results = evaluator.evaluate()
            logger.info(f"Evaluation Results: {results}")

        elif args.mode == "predict":
            logger.info("Starting CLASSIFICATION PREDICTION mode...")
            if not args.image_path:
                raise ValueError("You must specify --image_path in predict mode.")

            from src.architectures import build_model
            model = build_model(cfg_model)

            if args.checkpoint:
                logger.info(f"Loading checkpoint from {args.checkpoint}...")
                checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
                if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                    model.load_state_dict(checkpoint_data["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint_data)
                logger.info("Successfully loaded classifier checkpoint.")
            else:
                logger.warning("No checkpoint provided via --checkpoint. Model will run using default weights.")

            logger.info(f"Running inference on image: {args.image_path}")
            output = model.predict(args.image_path)
            logger.info(f"Prediction output: {output}")


def main():
    args = parse_args()
    try:
        run_pipeline(args)
    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
        exit(1)


if __name__ == "__main__":
    main()
