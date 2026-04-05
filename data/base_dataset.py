"""
data/base_dataset.py
Abstract base class for all dataset loaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np


class VehicleDetectionDataset(Dataset, ABC):
    """
    Base class for KITTI, UA-DETRAC, BDD100K.

    Each subclass must implement:
        - _load_image_paths()
        - _load_annotations()
        - __len__()
        - __getitem__()

    Returns items as dicts:
        {
            "image":   torch.Tensor [3, H, W],
            "boxes":   torch.Tensor [N, 4]  (x1y1x2y2, normalized),
            "labels":  torch.Tensor [N]     (int class ids),
            "image_id": str,
            "orig_size": (H, W)
        }
    """

    # Override in subclass with dataset-specific class list
    CLASSES: List[str] = []

    def __init__(
        self,
        root: str,
        split: str = "train",       # train | val | test
        img_size: int = 640,
        transforms=None,
        classes_filter: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.transforms = transforms
        self.classes_filter = classes_filter or self.CLASSES

        # Build class → id mapping (filtered)
        self.class_to_id = {c: i for i, c in enumerate(self.classes_filter)}

        self.image_paths: List[Path] = []
        self.annotations: Dict = {}

        self._load_image_paths()
        self._load_annotations()

    @abstractmethod
    def _load_image_paths(self) -> None:
        """Populate self.image_paths."""
        ...

    @abstractmethod
    def _load_annotations(self) -> None:
        """Populate self.annotations keyed by image_id."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        ...

    # ── Helpers ──────────────────────────────────────────────────────────

    def _xyxy_to_normalized(
        self, boxes: np.ndarray, img_w: int, img_h: int
    ) -> np.ndarray:
        """Convert absolute xyxy boxes to normalized xyxy [0,1]."""
        boxes = boxes.astype(np.float32).copy()
        boxes[:, [0, 2]] /= img_w
        boxes[:, [1, 3]] /= img_h
        return np.clip(boxes, 0.0, 1.0)

    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert xywh → xyxy."""
        out = boxes.copy()
        out[:, 2] = boxes[:, 0] + boxes[:, 2]
        out[:, 3] = boxes[:, 1] + boxes[:, 3]
        return out

    def get_num_classes(self) -> int:
        return len(self.classes_filter)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split}, "
            f"n_images={len(self)}, "
            f"classes={self.classes_filter})"
        )
