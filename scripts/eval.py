"""
scripts/eval.py

Evaluate a trained checkpoint on a test dataset.

Usage:
    python scripts/eval.py \
        --config configs/yolo/yolov8_bdd100k.yaml \
        --checkpoint runs/yolov8l_bdd100k/weights/best.pt \
        --dataset kitti           # optionally evaluate on a different dataset
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from models.model_factory import build_model
from evaluation.metrics import DetectionEvaluator
from utils.logging import setup_logger
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset",    type=str, default=None,
                        help="Override dataset (for cross-dataset eval)")
    parser.add_argument("--split",      type=str, default="test")
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--iou",        type=float, default=0.45)
    parser.add_argument("overrides",    nargs="*")
    return parser.parse_args()


def get_dataloader(cfg, split):
    dataset_name = cfg.data.dataset
    root = Path(cfg.data.data_root) / dataset_name

    if dataset_name == "kitti":
        from data.kitti.kitti_dataset import KITTIDataset, collate_fn
        from data.transforms import build_transforms
        ds = KITTIDataset(root=root, split=split,
                          img_size=cfg.data.img_size,
                          transforms=build_transforms("val", cfg),
                          classes_filter=cfg.data.classes)
    elif dataset_name == "bdd100k":
        from data.bdd100k.bdd100k_dataset import BDD100KDataset
        from data.kitti.kitti_dataset import collate_fn
        from data.transforms import build_transforms
        ds = BDD100KDataset(root=root, split=split,
                            img_size=cfg.data.img_size,
                            transforms=build_transforms("val", cfg),
                            classes_filter=cfg.data.classes)
    elif dataset_name == "ua_detrac":
        from data.ua_detrac.ua_detrac_dataset import UADETRACDataset
        from data.kitti.kitti_dataset import collate_fn
        from data.transforms import build_transforms
        ds = UADETRACDataset(root=root, split=split,
                             img_size=cfg.data.img_size,
                             transforms=build_transforms("val", cfg),
                             classes_filter=cfg.data.classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=1, shuffle=False,
                      num_workers=cfg.data.num_workers,
                      collate_fn=collate_fn)


def main():
    args = parse_args()
    base = OmegaConf.load("configs/base.yaml")
    cfg = OmegaConf.merge(base, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if args.dataset:
        cfg.data.dataset = args.dataset

    logger = setup_logger("eval")
    set_seed(cfg.training.seed)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    cfg.model.checkpoint = args.checkpoint
    model = build_model(cfg)

    device = cfg.training.device
    evaluator = DetectionEvaluator(
        num_classes=len(cfg.data.classes),
        class_names=cfg.data.classes,
    )

    loader = get_dataloader(cfg, args.split)
    logger.info(f"Evaluating on {cfg.data.dataset} / {args.split} ({len(loader)} images)")

    for images, targets, image_ids, _ in tqdm(loader, desc="Evaluating"):
        predictions = model.predict(
            images, conf=args.conf, iou=args.iou, device=device
        )
        evaluator.update(predictions, targets, image_ids)

    metrics = evaluator.compute()
    evaluator.print_summary(metrics)

    # Save results
    import json
    out_path = Path(args.checkpoint).parent / f"eval_{cfg.data.dataset}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
