"""
models/yolo/yolo_model.py

Thin wrapper around Ultralytics YOLO for consistent API
with the RT-DETR wrapper.

Supports: YOLOv8n/s/m/l/x, YOLOv9c/e, YOLOv10n/s/m/b/l/x
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Ultralytics provides the YOLO class directly
from ultralytics import YOLO


class YOLODetector(nn.Module):
    """
    Wrapper that exposes a unified interface:
        forward(images) → List[Dict[boxes, scores, labels]]

    For training, Ultralytics' own trainer is used via .train().
    This wrapper is mainly used for inference and evaluation.
    """

    VARIANT_WEIGHTS = {
        # YOLOv8
        "yolov8n": "yolov8n.pt",
        "yolov8s": "yolov8s.pt",
        "yolov8m": "yolov8m.pt",
        "yolov8l": "yolov8l.pt",
        "yolov8x": "yolov8x.pt",
        # YOLOv9
        "yolov9c": "yolov9c.pt",
        "yolov9e": "yolov9e.pt",
        # YOLOv10 — NMS-free
        "yolov10n": "yolov10n.pt",
        "yolov10s": "yolov10s.pt",
        "yolov10m": "yolov10m.pt",
        "yolov10b": "yolov10b.pt",
        "yolov10l": "yolov10l.pt",
        "yolov10x": "yolov10x.pt",
    }

    def __init__(
        self,
        variant: str = "yolov8l",
        num_classes: int = 6,
        pretrained: bool = True,
        checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes

        if checkpoint:
            self.model = YOLO(checkpoint)
        elif pretrained:
            weights = self.VARIANT_WEIGHTS.get(variant, f"{variant}.pt")
            self.model = YOLO(weights)
        else:
            self.model = YOLO(f"{variant}.yaml")

    def train_model(self, cfg, dataset_yaml: str) -> None:
        """
        Delegate training to Ultralytics trainer.

        Args:
            cfg: OmegaConf config object
            dataset_yaml: path to a YOLO-format dataset.yaml
        """
        self.model.train(
            data=dataset_yaml,
            epochs=cfg.training.epochs,
            imgsz=cfg.data.img_size,
            batch=cfg.data.batch_size,
            optimizer=cfg.training.optimizer.upper(),
            lr0=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            warmup_epochs=cfg.training.warmup_epochs,
            amp=cfg.training.amp,
            device=cfg.training.device,
            project=cfg.project.output_dir,
            name=cfg.logging.run_name,
            exist_ok=True,
            verbose=True,
        )

    @torch.no_grad()
    def predict(
        self,
        images,                    # path | np.ndarray | torch.Tensor | list
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        device: str = "cuda",
    ) -> List[Dict]:
        """
        Run inference.

        Returns list of dicts (one per image):
            {
                "boxes":  torch.Tensor [N, 4]  xyxy absolute pixels,
                "scores": torch.Tensor [N],
                "labels": torch.Tensor [N] int,
            }
        """
        results = self.model.predict(
            images, conf=conf, iou=iou, max_det=max_det,
            device=device, verbose=False
        )
        parsed = []
        for r in results:
            boxes = r.boxes
            parsed.append({
                "boxes":  boxes.xyxy.cpu(),
                "scores": boxes.conf.cpu(),
                "labels": boxes.cls.cpu().long(),
            })
        return parsed

    def forward(self, images, **kwargs):
        return self.predict(images, **kwargs)

    def export(self, format: str = "onnx", **kwargs):
        """Export model to ONNX / TensorRT / CoreML etc."""
        return self.model.export(format=format, **kwargs)

    def load_checkpoint(self, path: str):
        self.model = YOLO(path)

    def __repr__(self):
        return f"YOLODetector(variant={self.variant}, num_classes={self.num_classes})"
