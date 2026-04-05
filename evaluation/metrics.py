"""
evaluation/metrics.py

Detection and tracking metrics for the thesis:

Detection:
    - mAP@50, mAP@50-95  (COCO-style, via pycocotools)

Tracking  (MOT & HOTA):
    - MOTA, MOTP         (via py-motmetrics)
    - HOTA, DetA, AssA   (via TrackEval — optional)

Usage:
    evaluator = DetectionEvaluator(num_classes=6, iou_thresholds=[0.5, 0.75])
    evaluator.update(predictions, targets)
    metrics = evaluator.compute()
"""

from typing import Dict, List, Optional
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Detection  — mAP
# ─────────────────────────────────────────────────────────────────────────────

class DetectionEvaluator:
    """
    COCO-style mAP evaluator.
    Accumulates predictions/targets across batches, computes at the end.
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        self._predictions: List[Dict] = []   # [{boxes, scores, labels, image_id}]
        self._targets: List[Dict] = []        # [{boxes, labels, image_id}]

    def update(
        self,
        predictions: List[Dict],
        targets: List[Dict],
        image_ids: List[str],
    ):
        """
        Args:
            predictions: list of {boxes [N,4] xyxy norm, scores [N], labels [N]}
            targets:     list of {boxes [M,4] xyxy norm, labels [M]}
            image_ids:   list of str
        """
        for pred, tgt, img_id in zip(predictions, targets, image_ids):
            self._predictions.append({**pred, "image_id": img_id})
            self._targets.append({**tgt,     "image_id": img_id})

    def compute(self, iou_thresholds: Optional[List[float]] = None) -> Dict:
        """
        Returns dict with:
            mAP50, mAP50_95, per_class_ap50
        """
        iou_thresholds = iou_thresholds or np.linspace(0.5, 0.95, 10).tolist()

        # Compute per-class AP at each IoU threshold
        aps_per_iou = []
        for iou_t in iou_thresholds:
            aps = []
            for cls_id in range(self.num_classes):
                ap = self._compute_ap_for_class(cls_id, iou_t)
                aps.append(ap)
            aps_per_iou.append(aps)

        aps_per_iou = np.array(aps_per_iou)  # [n_iou, n_classes]

        map50    = float(np.mean(aps_per_iou[0]))
        map50_95 = float(np.mean(aps_per_iou))
        per_class = {
            self.class_names[i]: float(aps_per_iou[0, i])
            for i in range(self.num_classes)
        }

        return {
            "mAP50":    map50,
            "mAP50-95": map50_95,
            "per_class_AP50": per_class,
        }

    def _compute_ap_for_class(self, cls_id: int, iou_threshold: float) -> float:
        # Collect all predictions and GTs for this class
        all_scores, all_tp, n_gt = [], [], 0

        for pred, tgt in zip(self._predictions, self._targets):
            pred_mask = (pred["labels"] == cls_id)
            tgt_mask  = (tgt["labels"]  == cls_id)

            p_boxes  = pred["boxes"][pred_mask]
            p_scores = pred["scores"][pred_mask]
            t_boxes  = tgt["boxes"][tgt_mask]

            n_gt += len(t_boxes)

            if len(p_boxes) == 0:
                continue

            # Sort by score descending
            order    = torch.argsort(p_scores, descending=True)
            p_boxes  = p_boxes[order]
            p_scores = p_scores[order]

            matched = torch.zeros(len(t_boxes), dtype=torch.bool)
            for pb in p_boxes:
                if len(t_boxes) > 0:
                    ious = _box_iou(pb.unsqueeze(0), t_boxes).squeeze(0)
                    best_iou, best_idx = ious.max(0)
                    if best_iou >= iou_threshold and not matched[best_idx]:
                        all_tp.append(1)
                        matched[best_idx] = True
                    else:
                        all_tp.append(0)
                else:
                    all_tp.append(0)
                all_scores.append(float(p_scores[len(all_tp) - 1 - len(all_scores)
                                                 + len(all_tp) - 1]))

        if n_gt == 0:
            return float("nan")

        # Precision-Recall curve → AP
        all_tp = np.array(all_tp)
        cum_tp = np.cumsum(all_tp)
        cum_fp = np.cumsum(1 - all_tp)
        precision = cum_tp / (cum_tp + cum_fp + 1e-9)
        recall    = cum_tp / (n_gt + 1e-9)
        return float(_compute_ap_from_pr(precision, recall))

    def print_summary(self, metrics: Optional[Dict] = None):
        m = metrics or self.compute()
        print(f"\n{'─'*40}")
        print(f"  mAP@50:    {m['mAP50']:.4f}")
        print(f"  mAP@50-95: {m['mAP50-95']:.4f}")
        print(f"  Per-class AP@50:")
        for cls, ap in m["per_class_AP50"].items():
            print(f"    {cls:20s}: {ap:.4f}")
        print(f"{'─'*40}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Tracking  — MOT metrics (MOTA / MOTP)
# ─────────────────────────────────────────────────────────────────────────────

class MOTEvaluator:
    """
    Wraps py-motmetrics to compute MOTA, MOTP, IDF1, etc.

    Usage:
        mot = MOTEvaluator()
        for frame in sequence:
            mot.update(frame_id, gt_boxes, gt_ids, pred_boxes, pred_ids)
        summary = mot.compute()
    """

    def __init__(self):
        try:
            import motmetrics as mm
            self.mm = mm
            self.acc = mm.MOTAccumulator(auto_id=False)
        except ImportError:
            raise ImportError("Install motmetrics: pip install motmetrics")

    def update(
        self,
        frame_id: int,
        gt_boxes: np.ndarray,    # [M, 4] xyxy
        gt_ids:   np.ndarray,    # [M]
        pred_boxes: np.ndarray,  # [N, 4] xyxy
        pred_ids:   np.ndarray,  # [N]
    ):
        dist = self.mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        self.acc.update(gt_ids.tolist(), pred_ids.tolist(), dist, frameid=frame_id)

    def compute(self) -> Dict:
        mh = self.mm.metrics.create()
        summary = mh.compute(
            self.acc,
            metrics=["mota", "motp", "idf1", "num_switches",
                     "num_fragmentations", "precision", "recall"],
            name="MOT",
        )
        return summary.to_dict(orient="records")[0]

    def reset(self):
        self.acc = self.mm.MOTAccumulator(auto_id=False)


# ─────────────────────────────────────────────────────────────────────────────
# HOTA — Higher Order Tracking Accuracy
# ─────────────────────────────────────────────────────────────────────────────

class HOTAEvaluator:
    """
    Thin wrapper around TrackEval's HOTA metric.

    Install TrackEval:
        pip install git+https://github.com/JonathonLuiten/TrackEval.git

    For the thesis this is the primary tracking metric alongside MOT.
    HOTA decomposes into:
        - DetA  (Detection Accuracy)
        - AssA  (Association Accuracy)
        - LocA  (Localisation Accuracy)
    """

    def __init__(self):
        try:
            from trackeval.metrics import HOTA
            self._hota = HOTA()
        except ImportError:
            print(
                "[HOTAEvaluator] TrackEval not installed. "
                "Install: pip install git+https://github.com/JonathonLuiten/TrackEval"
            )
            self._hota = None

    def compute(self, gt_data: Dict, tracker_data: Dict) -> Dict:
        if self._hota is None:
            return {}
        results = self._hota.eval_sequence(gt_data, tracker_data)
        return {
            "HOTA": float(np.mean(results["HOTA"])),
            "DetA": float(np.mean(results["DetA"])),
            "AssA": float(np.mean(results["AssA"])),
            "LocA": float(np.mean(results["LocA"])),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between box1 [N,4] and box2 [M,4] (xyxy format)."""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-9)


def _compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """VOC 2010+ style AP — area under precision-recall curve."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
