"""
QA helper: scan a rendered dataset's coco_annotations.json and flag any
game-piece instance whose rendered pixels are suspiciously dark (e.g. the
"fully black ball" rendering artifact under investigation) - a legitimate
instance should always be bright yellow/red/blue, never near-black.

Usage:
    python scripts/postprocess/find_dark_instances.py --dir output/<run>
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dataset dir containing coco_annotations.json + images/")
    ap.add_argument("--mean_threshold", type=float, default=60.0,
                     help="flag instances whose mean bbox-region pixel intensity is below this (0-255)")
    args = ap.parse_args()

    coco_path = os.path.join(args.dir, "coco_annotations.json")
    with open(coco_path) as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images_by_id = {img["id"]: img for img in coco["images"]}

    flagged = []
    for ann in coco["annotations"]:
        img_info = images_by_id[ann["image_id"]]
        img_path = os.path.join(args.dir, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        # Use the tight bbox region directly (no pycocotools/mask decoding
        # dependency needed) - a real piece's bbox is dominated by its own
        # pixels, so a near-black mean here is still a strong, simple signal.
        x, y, w, h = [int(round(v)) for v in ann["bbox"]]
        crop = img[max(y, 0):y + h, max(x, 0):x + w]
        if crop.size == 0:
            continue
        mean_intensity = float(crop.mean())
        if mean_intensity < args.mean_threshold:
            flagged.append((img_info["file_name"], cats[ann["category_id"]], mean_intensity, ann["bbox"]))

    if not flagged:
        print(f"No suspiciously dark instances found (threshold={args.mean_threshold}).")
    else:
        print(f"Found {len(flagged)} suspiciously dark instance(s):")
        for file_name, cat, mean_intensity, bbox in flagged:
            print(f"  {file_name}: class={cat} mean_intensity={mean_intensity:.1f} bbox={bbox}")


if __name__ == "__main__":
    main()
