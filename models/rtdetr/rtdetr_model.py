"""
models/rtdetr/rtdetr_model.py

RT-DETR wrapper using the Ultralytics implementation,
with an alternative HuggingFace path for research flexibility.

RT-DETR key differences vs YOLO:
  - Transformer decoder with cross-attention (not CNN anchor head)
  - NMS-free by design (bipartite matching at training, top-k at inference)
  - Visual attention can be extracted for thesis analysis
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class RTDETRDetector(nn.Module):
    """
    Wrapper around RT-DETR with two backend options:
        backend="ultralytics"  → uses ultralytics RT-DETR (recommended)
        backend="huggingface"  → uses transformers RT-DETR (easier attention extraction)
    """

    ULTRALYTICS_VARIANTS = {
        "rtdetr-r18": "rtdetr-r18.pt",
        "rtdetr-r34": "rtdetr-r34.pt",
        "rtdetr-r50": "rtdetr-r50.pt",
        "rtdetr-l":   "rtdetr-l.pt",
        "rtdetr-x":   "rtdetr-x.pt",
    }

    HF_VARIANTS = {
        "rtdetr-r50":  "PekingU/rtdetr_r50vd",
        "rtdetr-r101": "PekingU/rtdetr_r101vd",
        "rtdetr-l":    "PekingU/rtdetr_r50vd_coco_objects365",
    }

    def __init__(
        self,
        variant: str = "rtdetr-l",
        num_classes: int = 6,
        pretrained: bool = True,
        backend: str = "ultralytics",   # "ultralytics" | "huggingface"
        checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.backend = backend

        if backend == "ultralytics":
            self._init_ultralytics(pretrained, checkpoint)
        elif backend == "huggingface":
            self._init_huggingface(pretrained, checkpoint)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ── Init ─────────────────────────────────────────────────────────────

    def _init_ultralytics(self, pretrained: bool, checkpoint: Optional[str]):
        from ultralytics import RTDETR
        if checkpoint:
            self.model = RTDETR(checkpoint)
        elif pretrained:
            weights = self.ULTRALYTICS_VARIANTS.get(self.variant, f"{self.variant}.pt")
            self.model = RTDETR(weights)
        else:
            self.model = RTDETR(f"{self.variant}.yaml")

    def _init_huggingface(self, pretrained: bool, checkpoint: Optional[str]):
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        model_id = checkpoint or self.HF_VARIANTS.get(self.variant, self.HF_VARIANTS["rtdetr-l"])
        self.processor = RTDetrImageProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(
            model_id,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )

    # ── Training ─────────────────────────────────────────────────────────

    def train_model(self, cfg, dataset_yaml: str) -> None:
        """Ultralytics backend only."""
        if self.backend != "ultralytics":
            raise NotImplementedError("Use train_hf() for HuggingFace backend.")
        self.model.train(
            data=dataset_yaml,
            epochs=cfg.training.epochs,
            imgsz=cfg.data.img_size,
            batch=cfg.data.batch_size,
            optimizer=cfg.training.optimizer.upper(),
            lr0=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            amp=cfg.training.amp,
            device=cfg.training.device,
            project=cfg.project.output_dir,
            name=cfg.logging.run_name,
            exist_ok=True,
        )

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        images,
        conf: float = 0.25,
        max_det: int = 300,
        device: str = "cuda",
    ) -> List[Dict]:
        """
        Returns list of dicts (one per image):
            {
                "boxes":  torch.Tensor [N, 4]  xyxy absolute pixels,
                "scores": torch.Tensor [N],
                "labels": torch.Tensor [N] int,
            }
        """
        if self.backend == "ultralytics":
            return self._predict_ultralytics(images, conf, max_det, device)
        else:
            return self._predict_hf(images, conf, max_det, device)

    def _predict_ultralytics(self, images, conf, max_det, device):
        results = self.model.predict(
            images, conf=conf, max_det=max_det, device=device, verbose=False
        )
        return [
            {
                "boxes":  r.boxes.xyxy.cpu(),
                "scores": r.boxes.conf.cpu(),
                "labels": r.boxes.cls.cpu().long(),
            }
            for r in results
        ]

    def _predict_hf(self, images, conf, max_det, device):
        """HuggingFace path — also returns attention weights for analysis."""
        self.model.to(device)
        self.model.eval()
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        outputs = self.model(**inputs, output_attentions=True)

        target_sizes = torch.tensor(
            [[img.shape[-2], img.shape[-1]] for img in images]
        )
        preds = self.processor.post_process_object_detection(
            outputs, threshold=conf, target_sizes=target_sizes
        )
        parsed = []
        for p in preds:
            parsed.append({
                "boxes":  p["boxes"].cpu(),
                "scores": p["scores"].cpu(),
                "labels": p["labels"].cpu().long(),
                # cross-attention from last decoder layer — useful for thesis viz
                "decoder_attentions": outputs.decoder_attentions[-1].cpu()
                    if hasattr(outputs, "decoder_attentions") else None,
            })
        return parsed

    def forward(self, images, **kwargs):
        return self.predict(images, **kwargs)

    def load_checkpoint(self, path: str):
        if self.backend == "ultralytics":
            from ultralytics import RTDETR
            self.model = RTDETR(path)
        else:
            from transformers import RTDetrForObjectDetection
            self.model = RTDetrForObjectDetection.from_pretrained(path)

    def __repr__(self):
        return (
            f"RTDETRDetector(variant={self.variant}, "
            f"backend={self.backend}, num_classes={self.num_classes})"
        )
