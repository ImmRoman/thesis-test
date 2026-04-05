"""
utils/visualization.py

Drawing utilities for detection and tracking results.
Useful for debugging, qualitative analysis, and thesis figures.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TABLEAU_COLORS


# ── Color palette ─────────────────────────────────────────────────────────────

PALETTE = {
    "car":         (0,   114, 189),
    "truck":       (217,  83,  25),
    "bus":         (237, 177,  32),
    "motorcycle":  (126,  47, 142),
    "bicycle":     ( 77, 190,  74),
    "pedestrian":  (162,  20,  47),
    # fallback for unknown classes
    "default":     (128, 128, 128),
}


def get_color(class_name: str) -> Tuple[int, int, int]:
    return PALETTE.get(class_name, PALETTE["default"])


# ── Single-image drawing ──────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,         # HxWx3 uint8 BGR or RGB
    boxes: torch.Tensor,       # [N, 4] xyxy (absolute pixels)
    labels: torch.Tensor,      # [N] int
    scores: Optional[torch.Tensor] = None,   # [N] float
    class_names: Optional[List[str]] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    is_bgr: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes on an image (in-place copy).

    Returns: annotated image (same dtype as input).
    """
    img = image.copy()
    class_names = class_names or [str(i) for i in range(100)]

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_name = class_names[int(label)] if int(label) < len(class_names) else "unknown"
        color = get_color(cls_name)

        # BGR if needed
        draw_color = color if not is_bgr else (color[2], color[1], color[0])

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), draw_color, line_thickness)

        # Label
        score_str = f" {scores[i]:.2f}" if scores is not None else ""
        label_str = f"{cls_name}{score_str}"
        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), draw_color, -1)
        cv2.putText(img, label_str, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return img


def draw_tracks(
    image: np.ndarray,
    boxes: torch.Tensor,        # [N, 4] xyxy absolute
    track_ids: torch.Tensor,    # [N] int
    labels: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw tracking boxes with unique color per track ID."""
    img = image.copy()
    colors = list(TABLEAU_COLORS.values())

    for i, (box, tid) in enumerate(zip(boxes, track_ids)):
        x1, y1, x2, y2 = map(int, box.tolist())
        color_hex = colors[int(tid) % len(colors)].lstrip("#")
        r, g, b = tuple(int(color_hex[j:j+2], 16) for j in (0, 2, 4))
        draw_color = (b, g, r)   # BGR

        cv2.rectangle(img, (x1, y1), (x2, y2), draw_color, line_thickness)
        label = f"ID:{int(tid)}"
        if labels is not None and class_names is not None:
            label += f" {class_names[int(labels[i])]}"
        cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, draw_color, 1, cv2.LINE_AA)
    return img


# ── Multi-image grid ──────────────────────────────────────────────────────────

def make_detection_grid(
    images: List[np.ndarray],
    predictions: List[Dict],
    targets: Optional[List[Dict]] = None,
    class_names: Optional[List[str]] = None,
    cols: int = 2,
    img_size: int = 480,
    title: str = "",
) -> plt.Figure:
    """
    Create a matplotlib grid comparing GT (green) vs Pred (colored).

    Args:
        images:      list of HxWx3 RGB uint8
        predictions: list of {boxes, scores, labels}
        targets:     list of {boxes, labels}  (optional, shown in green)
        class_names: list of class name strings
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = np.array(axes).flatten()

    for i, (img, pred) in enumerate(zip(images, predictions)):
        ax = axes[i]
        h, w = img.shape[:2]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        # Ground truth — dashed green
        if targets is not None:
            tgt = targets[i]
            for box, lbl in zip(tgt["boxes"], tgt["labels"]):
                x1, y1, x2, y2 = _denorm_box(box, w, h)
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor="lime",
                    facecolor="none", linestyle="--"
                )
                ax.add_patch(rect)

        # Predictions — solid colored
        for j, (box, lbl) in enumerate(zip(pred["boxes"], pred["labels"])):
            x1, y1, x2, y2 = _denorm_box(box, w, h)
            cls_name = class_names[int(lbl)] if class_names else str(int(lbl))
            color = np.array(get_color(cls_name)) / 255.0
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            score = pred["scores"][j] if "scores" in pred else None
            score_str = f" {float(score):.2f}" if score is not None else ""
            ax.text(x1, y1 - 3, f"{cls_name}{score_str}",
                    color=color, fontsize=7, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Attention visualization (RT-DETR) ─────────────────────────────────────────

def visualize_decoder_attention(
    image: np.ndarray,            # HxWx3 RGB
    attention: torch.Tensor,      # [heads, num_queries, seq_len]
    query_idx: int = 0,
    head_idx: Optional[int] = None,   # None → average over heads
    alpha: float = 0.5,
    cmap: str = "hot",
) -> np.ndarray:
    """
    Overlay decoder cross-attention map onto image.

    Useful for thesis figures comparing CNN saliency vs Transformer attention.

    Args:
        attention: last decoder layer attention [heads, queries, HW]
        query_idx: which detection query to visualize
        head_idx:  specific head or None for average
    """
    h_img, w_img = image.shape[:2]

    # [heads, HW] → [HW]
    if head_idx is not None:
        attn_map = attention[head_idx, query_idx]
    else:
        attn_map = attention[:, query_idx].mean(0)

    attn_map = attn_map.float().cpu().numpy()

    # Infer spatial size
    seq_len = attn_map.shape[-1]
    side = int(seq_len ** 0.5)
    attn_map = attn_map.reshape(side, side)

    # Resize to image
    attn_resized = cv2.resize(attn_map, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

    # Colormap
    colormap = plt.get_cmap(cmap)
    heatmap = (colormap(attn_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ── Metric plots ──────────────────────────────────────────────────────────────

def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: float,
    class_name: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot Precision-Recall curve for a single class."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    ax.plot(recall, precision, lw=2, label=f"AP={ap:.3f}")
    ax.fill_between(recall, precision, alpha=0.15)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {class_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    return fig


def plot_model_comparison(
    results: Dict[str, Dict],    # {"YOLOv8l": {mAP50: ..., mAP50-95: ...}, ...}
    metric: str = "mAP50-95",
    title: str = "Model Comparison",
) -> plt.Figure:
    """
    Bar chart comparing multiple models on a given metric.
    Designed for thesis figures.
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]
    colors = ["#0072B2", "#E69F00", "#56B4E9", "#009E73", "#F0E442"][:len(models)]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
    bars = ax.bar(models, values, color=colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylabel(metric)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim([0, max(values) * 1.15])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _denorm_box(box, img_w: int, img_h: int):
    """Normalized xyxy → absolute pixel coords."""
    x1, y1, x2, y2 = box.tolist()
    # If already absolute (>1), return as-is
    if x2 > 1.5:
        return int(x1), int(y1), int(x2), int(y2)
    return int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h)


def save_figure(fig: plt.Figure, path: str, dpi: int = 150):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
