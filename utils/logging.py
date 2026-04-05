"""
utils/logging.py + utils/seed.py — combined utilities
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch


# ── Logger ───────────────────────────────────────────────────────────────────

def setup_logger(name: str = "vehicle_detection") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def init_wandb(cfg):
    try:
        import wandb
        wandb.init(
            project=cfg.project.name,
            name=cfg.logging.run_name,
            config=dict(cfg),
        )
    except ImportError:
        print("[wandb] Not installed — skipping. pip install wandb")


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
