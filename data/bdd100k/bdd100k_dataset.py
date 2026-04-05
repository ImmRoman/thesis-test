"""
data/bdd100k/bdd100k_dataset.py

BDD100K Detection dataset loader.

Expected directory layout:
    data_root/
        images/
            100k/
                train/   ← .jpg images
                val/
                test/
        labels/
            det_20/
                det_train.json
                det_val.json

Annotation format: BDD100K JSON (COCO-like but custom schema).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.base_dataset import VehicleDetectionDataset


class BDD100KDataset(VehicleDetectionDataset):

    CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle",
               "pedestrian", "rider", "traffic light", "traffic sign"]

    # Map BDD100K category names → unified thesis classes
    CLASS_MAP = {
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        "motorcycle": "motorcycle",
        "bicycle": "bicycle",
        "pedestrian": "pedestrian",
        "rider": "pedestrian",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        transforms=None,
        classes_filter: Optional[List[str]] = None,
    ):
        super().__init__(root, split, img_size, transforms, classes_filter)

    def _load_image_paths(self) -> None:
        img_dir = self.root / "images" / "100k" / self.split
        self.image_paths = sorted(img_dir.glob("*.jpg"))

    def _load_annotations(self) -> None:
        split_name = "val" if self.split == "val" else self.split
        label_file = self.root / "labels" / "det_20" / f"det_{split_name}.json"

        self.annotations = {}

        if not label_file.exists():
            print(f"[BDD100K] Warning: label file not found at {label_file}")
            return

        with open(label_file) as f:
            data = json.load(f)

        for frame in data:
            name = Path(frame["name"]).stem
            boxes, labels = [], []

            for obj in frame.get("labels") or []:
                cat = obj.get("category", "")
                mapped = self.CLASS_MAP.get(cat)
                if mapped is None or mapped not in self.class_to_id:
                    continue
                b = obj["box2d"]
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_to_id[mapped])

            self.annotations[name] = {
                "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
                "labels": np.array(labels, dtype=np.int64),
            }

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        stem = img_path.stem

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        ann = self.annotations.get(stem, {"boxes": np.zeros((0, 4), np.float32),
                                          "labels": np.zeros(0, np.int64)})
        boxes = ann["boxes"].copy()
        labels = ann["labels"].copy()

        if len(boxes) > 0:
            boxes = self._xyxy_to_normalized(boxes, orig_w, orig_h)

        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes.tolist(),
                labels=labels.tolist(),
            )
            img = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.array(transformed["labels"], dtype=np.int64)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return {
            "image": img_tensor,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": stem,
            "orig_size": (orig_h, orig_w),
        }


def build_bdd100k_dataloaders(cfg) -> Dict:
    from data.transforms import build_transforms
    from data.kitti.kitti_dataset import collate_fn

    root = Path(cfg.data.data_root) / "bdd100k"
    common = dict(root=root, img_size=cfg.data.img_size,
                  classes_filter=cfg.data.classes)

    datasets = {
        "train": BDD100KDataset(split="train", transforms=build_transforms("train", cfg), **common),
        "val":   BDD100KDataset(split="val",   transforms=build_transforms("val",   cfg), **common),
    }

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.data.batch_size if split == "train" else 1,
            shuffle=(split == "train"),
            num_workers=cfg.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    return loaders
