"""
scripts/prepare_dataset.py

Converts KITTI / BDD100K / UA-DETRAC into YOLO-format layout
so that Ultralytics' trainer can consume them directly.

Output structure:
    data/<dataset>/
        images/train/   ← symlinks or copies
        images/val/
        labels/train/   ← .txt files (YOLO format: class cx cy w h normalized)
        dataset.yaml

Usage:
    python scripts/prepare_dataset.py --dataset bdd100k --data_root data/
    python scripts/prepare_dataset.py --dataset kitti   --data_root data/
    python scripts/prepare_dataset.py --dataset ua_detrac --data_root data/
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from tqdm import tqdm

CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]


def xyxy_norm_to_xywh_norm(box):
    """[x1,y1,x2,y2] normalized → [cx,cy,w,h] normalized."""
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    w  = box[2] - box[0]
    h  = box[3] - box[1]
    return cx, cy, w, h


def write_yolo_labels(label_dir: Path, image_id: str,
                      boxes: np.ndarray, labels: np.ndarray):
    label_path = label_dir / f"{image_id}.txt"
    with open(label_path, "w") as f:
        for box, cls_id in zip(boxes, labels):
            cx, cy, w, h = xyxy_norm_to_xywh_norm(box)
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def prepare_kitti(data_root: Path):
    from data.kitti.kitti_dataset import KITTIDataset
    from omegaconf import OmegaConf

    cfg_stub = OmegaConf.create({"data": {"img_size": 640, "classes": CLASSES}})
    root = data_root / "kitti"

    for split in ("train", "val"):
        ds = KITTIDataset(root=root, split=split, classes_filter=CLASSES)
        img_out = root / "images" / split
        lbl_out = root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(len(ds)), desc=f"KITTI {split}"):
            item = ds[i]
            src = ds.image_paths[i]
            dst = img_out / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            write_yolo_labels(lbl_out, item["image_id"],
                              item["boxes"].numpy(), item["labels"].numpy())

    _write_dataset_yaml(root, CLASSES)
    print(f"KITTI prepared at {root}")


def prepare_bdd100k(data_root: Path):
    from data.bdd100k.bdd100k_dataset import BDD100KDataset

    root = data_root / "bdd100k"
    for split in ("train", "val"):
        ds = BDD100KDataset(root=root, split=split, classes_filter=CLASSES)
        img_out = root / "images" / split
        lbl_out = root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(len(ds)), desc=f"BDD100K {split}"):
            item = ds[i]
            src = ds.image_paths[i]
            dst = img_out / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            write_yolo_labels(lbl_out, item["image_id"],
                              item["boxes"].numpy(), item["labels"].numpy())

    _write_dataset_yaml(root, CLASSES)
    print(f"BDD100K prepared at {root}")


def _write_dataset_yaml(root: Path, classes):
    cfg = {
        "path":  str(root.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(classes),
        "names": classes,
    }
    with open(root / "dataset.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   required=True,
                        choices=["kitti", "bdd100k", "ua_detrac"])
    parser.add_argument("--data_root", default="data/")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.dataset == "kitti":
        prepare_kitti(data_root)
    elif args.dataset == "bdd100k":
        prepare_bdd100k(data_root)
    else:
        print("UA-DETRAC preparation is video-based — use the tracking pipeline directly.")


if __name__ == "__main__":
    main()
