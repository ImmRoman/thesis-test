# Vehicle Detection & Tracking — Thesis Codebase

Comparative study of **CNN-based** (YOLOv8/v9/v10) vs **Transformer-based** (RT-DETR) architectures
for vehicle detection, with multi-object tracking (StrongSORT / alternatives).

## Project Structure

```
vehicle_detection/
├── configs/                  # YAML experiment configs
│   ├── base.yaml             # shared defaults
│   ├── yolo/                 # YOLO-specific overrides
│   └── rtdetr/               # RT-DETR-specific overrides
├── data/                     # Dataset loaders
│   ├── kitti/
│   ├── ua_detrac/
│   └── bdd100k/
├── models/
│   ├── yolo/                 # YOLOv8/v9/v10 wrappers
│   └── rtdetr/               # RT-DETR wrappers
├── training/                 # Trainer, losses, schedulers
├── evaluation/               # mAP, MOT, HOTA metrics
├── utils/                    # Visualization, logging, misc
├── scripts/                  # train.py, eval.py, infer.py
└── notebooks/                # Exploratory analysis
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Train YOLO on BDD100K
python scripts/train.py --config configs/yolo/yolov8_bdd100k.yaml

# Train RT-DETR on KITTI
python scripts/train.py --config configs/rtdetr/rtdetr_kitti.yaml

# Evaluate
python scripts/eval.py --config configs/yolo/yolov8_bdd100k.yaml --checkpoint runs/exp1/best.pt
```

## Datasets

| Dataset    | Size       | Source         | Primary use         |
|------------|------------|----------------|---------------------|
| BDD100K    | 100k imgs  | Dashcam        | Training (main)     |
| KITTI      | ~15k imgs  | Stereo camera  | Test / fine-tune    |
| UA-DETRAC  | ~140k imgs | Traffic CCTV   | Tracking benchmark  |

## Models

| Model      | Backbone      | Head type      | NMS-free |
|------------|---------------|----------------|----------|
| YOLOv8     | CSPDarknet    | Decoupled       | No       |
| YOLOv10    | CSPDarknet    | NMS-free head   | Yes      |
| RT-DETR    | ResNet/HGNet  | Transformer     | Yes      |

## Metrics

- **Detection**: mAP@50, mAP@50-95, Precision, Recall
- **Tracking**: MOTA, MOTP (MOT metrics), HOTA, DetA, AssA
