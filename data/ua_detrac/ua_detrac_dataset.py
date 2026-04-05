"""
data/ua_detrac/ua_detrac_dataset.py

UA-DETRAC dataset loader — primarily used for tracking benchmarks.

Expected directory layout:
    data_root/
        Insight-MVT_Annotation_Train/
            MVI_20011/
                img00001.jpg
                ...
        DETRAC-Train-Annotations-XML/
            MVI_20011.xml
            ...
        Insight-MVT_Annotation_Test/
            MVI_39031/
                img00001.jpg
        DETRAC-Test-Annotations-XML/   (if available)

XML annotation format:
    <sequence name="MVI_20011">
      <frame density="..." num="1">
        <target_list>
          <target id="1">
            <box left="..." top="..." width="..." height="..."/>
            <attribute vehicle_type="Car" .../>
          </target>
        </target_list>
      </frame>
    </sequence>
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.base_dataset import VehicleDetectionDataset


class UADETRACDataset(VehicleDetectionDataset):

    CLASSES = ["car", "bus", "van", "others"]

    CLASS_MAP = {
        "Car": "car",
        "Bus": "bus",
        "Van": "car",      # merge Van → car for consistency
        "Others": "car",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        transforms=None,
        classes_filter: Optional[List[str]] = None,
        # UA-DETRAC is a video dataset — optionally sample every N frames
        frame_stride: int = 1,
    ):
        self.frame_stride = frame_stride
        super().__init__(root, split, img_size, transforms, classes_filter)

    def _load_image_paths(self) -> None:
        if self.split in ("train", "val"):
            base = self.root / "Insight-MVT_Annotation_Train"
        else:
            base = self.root / "Insight-MVT_Annotation_Test"

        sequences = sorted(base.iterdir()) if base.exists() else []

        # Optional: split sequences into train/val
        if self.split == "val":
            sequences = sequences[:max(1, len(sequences) // 10)]
        elif self.split == "train":
            sequences = sequences[max(1, len(sequences) // 10):]

        self.image_paths = []
        self.seq_of_path: Dict[Path, str] = {}

        for seq_dir in sequences:
            frames = sorted(seq_dir.glob("img*.jpg"))[::self.frame_stride]
            for f in frames:
                self.image_paths.append(f)
                self.seq_of_path[f] = seq_dir.name

    def _load_annotations(self) -> None:
        if self.split in ("train", "val"):
            ann_dir = self.root / "DETRAC-Train-Annotations-XML"
        else:
            ann_dir = self.root / "DETRAC-Test-Annotations-XML"

        self.annotations = {}
        self._seq_cache: Dict[str, Dict[int, Dict]] = {}

        if not ann_dir.exists():
            print(f"[UA-DETRAC] Warning: annotation dir not found at {ann_dir}")
            return

        for xml_file in ann_dir.glob("*.xml"):
            seq_name = xml_file.stem
            seq_data = self._parse_xml(xml_file)
            self._seq_cache[seq_name] = seq_data

    def _parse_xml(self, xml_path: Path) -> Dict[int, Dict]:
        """Parse one sequence XML → {frame_num: {boxes, labels, track_ids}}"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        frames = {}

        for frame_el in root.findall("frame"):
            frame_num = int(frame_el.get("num"))
            boxes, labels, track_ids = [], [], []

            for target in frame_el.findall(".//target"):
                tid = int(target.get("id"))
                box_el = target.find("box")
                attr_el = target.find("attribute")
                if box_el is None:
                    continue

                left = float(box_el.get("left"))
                top = float(box_el.get("top"))
                w = float(box_el.get("width"))
                h = float(box_el.get("height"))
                vtype = attr_el.get("vehicle_type", "Car") if attr_el is not None else "Car"

                mapped = self.CLASS_MAP.get(vtype, "car")
                if mapped not in self.class_to_id:
                    continue

                boxes.append([left, top, left + w, top + h])
                labels.append(self.class_to_id[mapped])
                track_ids.append(tid)

            frames[frame_num] = {
                "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
                "labels": np.array(labels, dtype=np.int64),
                "track_ids": np.array(track_ids, dtype=np.int64),
            }
        return frames

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        seq_name = self.seq_of_path[img_path]

        # Parse frame number from filename: img00001.jpg → 1
        frame_num = int(img_path.stem.replace("img", ""))

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        seq_data = self._seq_cache.get(seq_name, {})
        ann = seq_data.get(frame_num, {
            "boxes": np.zeros((0, 4), np.float32),
            "labels": np.zeros(0, np.int64),
            "track_ids": np.zeros(0, np.int64),
        })

        boxes = ann["boxes"].copy()
        labels = ann["labels"].copy()
        track_ids = ann["track_ids"].copy()

        if len(boxes) > 0:
            boxes = self._xyxy_to_normalized(boxes, orig_w, orig_h)

        if self.transforms:
            transformed = self.transforms(
                image=img, bboxes=boxes.tolist(), labels=labels.tolist()
            )
            img = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.array(transformed["labels"], dtype=np.int64)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return {
            "image": img_tensor,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "track_ids": torch.from_numpy(track_ids),   # extra for tracking eval
            "image_id": f"{seq_name}_{frame_num:05d}",
            "seq_name": seq_name,
            "frame_num": frame_num,
            "orig_size": (orig_h, orig_w),
        }
