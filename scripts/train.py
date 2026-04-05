"""
scripts/train.py

Entry point for training YOLO or RT-DETR.

Usage:
    python scripts/train.py --config configs/yolo/yolov8_bdd100k.yaml
    python scripts/train.py --config configs/rtdetr/rtdetr_kitti.yaml
    python scripts/train.py --config configs/yolo/yolov8_bdd100k.yaml model.variant=yolov8x
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf

from models.model_factory import build_model
from utils.logging import setup_logger, init_wandb
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train vehicle detection model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf overrides, e.g. model.variant=yolov8x")
    return parser.parse_args()


def load_config(config_path: str, overrides: list):
    base = OmegaConf.load("configs/base.yaml")
    model_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base, model_cfg)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    logger = setup_logger(cfg.project.run_name)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    set_seed(cfg.training.seed)

    if cfg.logging.use_wandb:
        init_wandb(cfg)

    logger.info(f"Building model: {cfg.model.name} / {cfg.model.variant}")
    model = build_model(cfg)
    logger.info(str(model))

    # ── For Ultralytics models, delegate to their internal trainer ───────
    # Ultralytics expects a YOLO-format dataset.yaml.
    # Use scripts/convert_to_yolo_format.py to prepare datasets first.
    dataset_yaml = Path(cfg.data.data_root) / cfg.data.dataset / "dataset.yaml"

    if not dataset_yaml.exists():
        logger.error(
            f"Dataset YAML not found: {dataset_yaml}\n"
            f"Run: python scripts/prepare_dataset.py --dataset {cfg.data.dataset}"
        )
        sys.exit(1)

    logger.info(f"Starting training on {cfg.data.dataset.upper()} ...")
    model.train_model(cfg, str(dataset_yaml))
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
