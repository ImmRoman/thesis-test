"""
data/kitti/kitti_dataset.py

KITTI Object Detection dataset loader.

Expected directory layout:
    data_root/
        training/
            image_2/          ← left color images (.png)
            label_2/          ← annotation .txt files
        testing/
            image_2/

KITTI label format (one object per line):
    type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry score
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.base_dataset import VehicleDetectionDataset


class KITTIDataset(VehicleDetectionDataset):

    CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting",
               "Cyclist", "Tram", "Misc"]

    # Map KITTI names → unified thesis classes
    CLASS_MAP = {
        "Car": "car",
        "Van": "car",
        "Truck": "truck",
        "Tram": "bus",
        "Cyclist": "bicycle",
        "Pedestrian": "pedestrian",
        "Person_sitting": "pedestrian",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        transforms=None,
        classes_filter: Optional[List[str]] = None,
        # KITTI has no official val split — use a fraction of training
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.val_fraction = val_fraction
        self.seed = seed
        super().__init__(root, split, img_size, transforms, classes_filter)

    def _load_image_paths(self) -> None:
        img_dir = self.root / "training" / "image_2"
        all_paths = sorted(img_dir.glob("*.png"))

        # Split train/val deterministically
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(all_paths))
        n_val = max(1, int(len(all_paths) * self.val_fraction))

        if self.split == "test":
            test_dir = self.root / "testing" / "image_2"
            self.image_paths = sorted(test_dir.glob("*.png"))
        elif self.split == "val":
            self.image_paths = [all_paths[i] for i in indices[:n_val]]
        else:  # train
            self.image_paths = [all_paths[i] for i in indices[n_val:]]

    def _load_annotations(self) -> None:
        label_dir = self.root / "training" / "label_2"
        self.annotations = {}

        for img_path in self.image_paths:
            stem = img_path.stem
            label_path = label_dir / f"{stem}.txt"
            boxes, labels = [], []

            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        obj_type = parts[0]
                        mapped = self.CLASS_MAP.get(obj_type)
                        if mapped is None or mapped not in self.class_to_id:
                            continue
                        x1, y1, x2, y2 = map(float, parts[4:8])
                        boxes.append([x1, y1, x2, y2])
                        labels.append(self.class_to_id[mapped])

            self.annotations[stem] = {
                "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
                "labels": np.array(labels, dtype=np.int64),
            }

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        stem = img_path.stem

        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        ann = self.annotations[stem]
        boxes = ann["boxes"].copy()
        labels = ann["labels"].copy()

        # Normalize boxes
        if len(boxes) > 0:
            boxes = self._xyxy_to_normalized(boxes, orig_w, orig_h)

        # Apply transforms (albumentations-compatible)
        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes.tolist(),
                labels=labels.tolist(),
            )
            img = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.array(transformed["labels"], dtype=np.int64)

        # To tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return {
            "image": img_tensor,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": stem,
            "orig_size": (orig_h, orig_w),
        }


def build_kitti_dataloaders(cfg) -> Dict:
    """Convenience factory — returns {train, val, test} DataLoaders."""
    from data.transforms import build_transforms

    root = Path(cfg.data.data_root) / "kitti"
    common = dict(root=root, img_size=cfg.data.img_size,
                  classes_filter=cfg.data.classes)

    datasets = {
        "train": KITTIDataset(split="train", transforms=build_transforms("train", cfg), **common),
        "val":   KITTIDataset(split="val",   transforms=build_transforms("val",   cfg), **common),
        "test":  KITTIDataset(split="test",  transforms=build_transforms("val",   cfg), **common),
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


def collate_fn(batch):
    """Custom collate to handle variable-length box lists."""
    images = torch.stack([b["image"] for b in batch])
    targets = [{"boxes": b["boxes"], "labels": b["labels"]} for b in batch]
    image_ids = [b["image_id"] for b in batch]
    orig_sizes = [b["orig_size"] for b in batch]
    return images, targets, image_ids, orig_sizes
