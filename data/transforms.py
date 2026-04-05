"""
data/transforms.py
Albumentations-based augmentation pipeline for train / val.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(split: str, cfg):
    img_size = cfg.data.img_size

    if split == "train":
        return A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(img_size, img_size, border_mode=0),
                # Color / photometric
                A.ColorJitter(brightness=0.4, contrast=0.4,
                              saturation=0.4, hue=0.1, p=0.8),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.GaussNoise(p=0.1),
                # Geometric
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                                   rotate_limit=10, p=0.5),
                A.RandomResizedCrop(
                    height=img_size, width=img_size,
                    scale=(0.5, 1.0), p=0.3
                ),
                # Weather simulation (useful for dashcam data)
                A.RandomRain(p=0.05),
                A.RandomFog(p=0.05),
                A.RandomSunFlare(p=0.03),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ],
            bbox_params=A.BboxParams(
                format="albumentations",   # normalized xyxy
                label_fields=["labels"],
                min_visibility=0.3,
            ),
        )
    else:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(img_size, img_size, border_mode=0),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ],
            bbox_params=A.BboxParams(
                format="albumentations",
                label_fields=["labels"],
                min_visibility=0.0,
            ),
        )
