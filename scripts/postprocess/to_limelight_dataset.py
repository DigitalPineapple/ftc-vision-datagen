"""
Package a YOLO-format dataset (produced by coco_to_yolo.py) into a zip ready
for upload to Limelight's Neural Network Trainer, which accepts Roboflow-style
YOLOv5/v8 export zips (train/valid/test split + data.yaml).

Usage:
    python scripts/postprocess/to_limelight_dataset.py --yolo_dir output/2026_biobuzz/yolo \
        --out_zip output/2026_biobuzz/limelight_dataset.zip --val_split 0.15
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import zipfile


def split_and_zip(yolo_dir: str, out_zip: str, val_split: float, seed: int = 0):
    images_dir = os.path.join(yolo_dir, "images")
    labels_dir = os.path.join(yolo_dir, "labels")
    stems = [
        os.path.splitext(f)[0] for f in os.listdir(images_dir) if not f.startswith(".")
    ]
    random.Random(seed).shuffle(stems)
    n_val = max(1, int(len(stems) * val_split))
    val_stems = set(stems[:n_val])

    staging = out_zip + "_staging"
    if os.path.isdir(staging):
        shutil.rmtree(staging)

    for split in ("train", "valid"):
        os.makedirs(os.path.join(staging, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(staging, split, "labels"), exist_ok=True)

    for stem in stems:
        split = "valid" if stem in val_stems else "train"
        for f in os.listdir(images_dir):
            if os.path.splitext(f)[0] == stem:
                shutil.copy2(os.path.join(images_dir, f), os.path.join(staging, split, "images", f))
                break
        label_src = os.path.join(labels_dir, f"{stem}.txt")
        if os.path.isfile(label_src):
            shutil.copy2(label_src, os.path.join(staging, split, "labels", f"{stem}.txt"))

    shutil.copy2(os.path.join(yolo_dir, "data.yaml"), os.path.join(staging, "data.yaml"))

    if os.path.isfile(out_zip):
        os.remove(out_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(staging):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, staging)
                zf.write(full, rel)

    shutil.rmtree(staging)
    print(f"Wrote Limelight-ready dataset zip: {out_zip} ({len(stems) - n_val} train / {n_val} valid)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_dir", required=True)
    parser.add_argument("--out_zip", required=True)
    parser.add_argument("--val_split", type=float, default=0.15)
    args = parser.parse_args()
    split_and_zip(args.yolo_dir, args.out_zip, args.val_split)
