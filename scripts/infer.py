"""
scripts/infer.py

Run inference on images, folders, or videos.
Saves annotated output to disk.

Usage:
    # Single image
    python scripts/infer.py \
        --config configs/yolo/yolov8_bdd100k.yaml \
        --checkpoint runs/yolov8l_bdd100k/weights/best.pt \
        --source path/to/image.jpg

    # Folder of images
    python scripts/infer.py --source path/to/images/ ...

    # Video file
    python scripts/infer.py --source path/to/video.mp4 ...

    # Webcam
    python scripts/infer.py --source 0 ...
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from models.model_factory import build_model
from utils.visualization import draw_detections
from utils.logging import setup_logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle detection inference")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source",     required=True,
                        help="Image path | folder | video | webcam index")
    parser.add_argument("--output",     default="runs/infer",
                        help="Output directory")
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--iou",        type=float, default=0.45)
    parser.add_argument("--save_txt",   action="store_true",
                        help="Also save YOLO-format .txt labels")
    parser.add_argument("--show",       action="store_true",
                        help="Display results in window (requires display)")
    parser.add_argument("overrides",    nargs="*")
    return parser.parse_args()


def load_config_and_model(args):
    base = OmegaConf.load("configs/base.yaml")
    cfg = OmegaConf.merge(base, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg.model.checkpoint = args.checkpoint
    model = build_model(cfg)
    return cfg, model


def infer_image(img_bgr, model, cfg, args):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    preds = model.predict(
        [img_rgb],
        conf=args.conf,
        iou=args.iou,
        device=cfg.training.device,
    )[0]
    return preds


def annotate(img_bgr, preds, class_names):
    # Convert absolute or normalized boxes to absolute pixels
    h, w = img_bgr.shape[:2]
    boxes = preds["boxes"]
    if len(boxes) > 0 and boxes.max() <= 1.5:
        # Normalized → absolute
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h

    return draw_detections(
        img_bgr,
        boxes,
        preds["labels"],
        preds.get("scores"),
        class_names=class_names,
        is_bgr=True,
    )


def save_labels(txt_path, preds, img_w, img_h):
    """Save predictions as YOLO-format .txt."""
    with open(txt_path, "w") as f:
        for box, label, score in zip(
            preds["boxes"], preds["labels"],
            preds.get("scores", [None] * len(preds["labels"]))
        ):
            x1, y1, x2, y2 = box.tolist()
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            sc = f" {float(score):.4f}" if score is not None else ""
            f.write(f"{int(label)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}{sc}\n")


def run_on_images(sources, model, cfg, args, out_dir, class_names, logger):
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_txt:
        (out_dir / "labels").mkdir(exist_ok=True)

    for src in tqdm(sources, desc="Inferring"):
        img = cv2.imread(str(src))
        if img is None:
            logger.warning(f"Could not read {src}")
            continue
        preds = infer_image(img, model, cfg, args)
        annotated = annotate(img, preds, class_names)
        out_path = out_dir / src.name
        cv2.imwrite(str(out_path), annotated)

        if args.save_txt:
            txt_path = out_dir / "labels" / (src.stem + ".txt")
            save_labels(txt_path, preds, img.shape[1], img.shape[0])

        if args.show:
            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()
    logger.info(f"Results saved to {out_dir}")


def run_on_video(source, model, cfg, args, out_dir, class_names, logger):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Support webcam
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    is_video_file = not str(source).isdigit()
    if is_video_file:
        out_name = Path(source).stem + "_det.mp4"
    else:
        out_name = "webcam_det.mp4"
    out_path = out_dir / out_name

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    frame_count = 0
    logger.info(f"Processing video: {source}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        preds = infer_image(frame, model, cfg, args)
        annotated = annotate(frame, preds, class_names)
        writer.write(annotated)
        frame_count += 1

        if args.show:
            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    logger.info(f"Video saved to {out_path} ({frame_count} frames)")


def main():
    args = parse_args()
    logger = setup_logger("infer")

    cfg, model = load_config_and_model(args)
    class_names = list(cfg.data.classes)
    out_dir = Path(args.output)
    source = args.source

    # Determine source type
    src_path = Path(source)

    if src_path.is_dir():
        image_files = sorted(
            f for f in src_path.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )
        logger.info(f"Found {len(image_files)} images in {src_path}")
        run_on_images(image_files, model, cfg, args, out_dir, class_names, logger)

    elif src_path.is_file() and src_path.suffix.lower() in IMAGE_EXTS:
        run_on_images([src_path], model, cfg, args, out_dir, class_names, logger)

    elif src_path.is_file() and src_path.suffix.lower() in VIDEO_EXTS:
        run_on_video(source, model, cfg, args, out_dir, class_names, logger)

    elif source.isdigit():
        run_on_video(source, model, cfg, args, out_dir, class_names, logger)

    else:
        logger.error(f"Unrecognized source: {source}")
        sys.exit(1)


if __name__ == "__main__":
    main()
