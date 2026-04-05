"""
models/model_factory.py
Instantiate the right model from a config object.
"""

from models.yolo.yolo_model import YOLODetector
from models.rtdetr.rtdetr_model import RTDETRDetector


def build_model(cfg):
    """
    Args:
        cfg: OmegaConf config (must have cfg.model.name, cfg.model.variant, etc.)
    Returns:
        A model instance with a unified .predict() interface.
    """
    name = cfg.model.name.lower()
    num_classes = len(cfg.data.classes)
    pretrained = cfg.model.get("pretrained", True)
    checkpoint = cfg.model.get("checkpoint", None)

    if name == "yolo":
        return YOLODetector(
            variant=cfg.model.variant,
            num_classes=num_classes,
            pretrained=pretrained,
            checkpoint=checkpoint,
        )
    elif name == "rtdetr":
        backend = cfg.model.get("backend", "ultralytics")
        return RTDETRDetector(
            variant=cfg.model.variant,
            num_classes=num_classes,
            pretrained=pretrained,
            backend=backend,
            checkpoint=checkpoint,
        )
    else:
        raise ValueError(f"Unknown model name: {name}. Choose 'yolo' or 'rtdetr'.")
