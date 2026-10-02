"""
Convert BlenderProc's COCO-format annotations directly into the TFRecord
dataset layout expected by the Limelight Neural Network Trainer
(https://tools.limelightvision.io/neural-network-trainer), which mirrors
Roboflow's "TFRecord" export:

    dataset.zip
      train.tfrecord
      valid.tfrecord
      label_map.pbtxt

Season-agnostic: class list/ids come from the COCO categories, so it adapts
automatically to whatever classes were rendered.

Uses tfrecord_writer.py (hand-rolled, dependency-free) instead of the
`tensorflow` package, which would be a very heavy install for just writing a
training dataset export.

Usage:
    python scripts/postprocess/coco_to_tfrecord.py \
        --coco output/2026_biobuzz/coco_annotations.json \
        --images_dir output/2026_biobuzz/images \
        --out_dir output/2026_biobuzz/tfrecord --val_split 0.15
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from tfrecord_writer import (
    bytes_list_feature,
    build_example,
    float_list_feature,
    int64_list_feature,
    write_tfrecord,
)


def _build_example(image: dict, anns: list, label_ids_by_cat: dict) -> bytes:
    w, h = image["width"], image["height"]
    filename = os.path.basename(image["file_name"])
    with open(image["_abs_path"], "rb") as f:
        encoded_jpg = f.read()

    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []
    for ann in anns:
        x, y, bw, bh = ann["bbox"]  # COCO: top-left x, y, width, height (pixels)
        xmins.append(max(0.0, x / w))
        xmaxs.append(min(1.0, (x + bw) / w))
        ymins.append(max(0.0, y / h))
        ymaxs.append(min(1.0, (y + bh) / h))
        label_id, name = label_ids_by_cat[ann["category_id"]]
        classes_text.append(name)
        classes.append(label_id)

    feature = {
        "image/height": int64_list_feature([h]),
        "image/width": int64_list_feature([w]),
        "image/filename": bytes_list_feature([filename]),
        "image/source_id": bytes_list_feature([filename]),
        "image/encoded": bytes_list_feature([encoded_jpg]),
        "image/format": bytes_list_feature(["jpeg"]),
        "image/object/bbox/xmin": float_list_feature(xmins),
        "image/object/bbox/xmax": float_list_feature(xmaxs),
        "image/object/bbox/ymin": float_list_feature(ymins),
        "image/object/bbox/ymax": float_list_feature(ymaxs),
        "image/object/class/text": bytes_list_feature(classes_text),
        "image/object/class/label": int64_list_feature(classes),
    }
    return build_example(feature)


def convert(coco_path: str, images_dir: str, out_dir: str, val_split: float = 0.15, seed: int = 0):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    # TF Object Detection API label ids are 1-indexed (0 is reserved).
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    label_ids_by_cat = {cat["id"]: (idx + 1, cat["name"]) for idx, cat in enumerate(categories)}

    images_by_id = {img["id"]: img for img in coco["images"]}
    for img in images_by_id.values():
        img["_abs_path"] = os.path.join(images_dir, os.path.basename(img["file_name"]))

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    image_ids = [
        img_id for img_id, img in images_by_id.items() if os.path.isfile(img["_abs_path"])
    ]
    random.Random(seed).shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * val_split)) if len(image_ids) > 1 else 0
    val_ids = set(image_ids[:n_val])

    os.makedirs(out_dir, exist_ok=True)
    counts = {"train": 0, "valid": 0}
    writers = {
        "train": open(os.path.join(out_dir, "train.tfrecord"), "wb"),
        "valid": open(os.path.join(out_dir, "valid.tfrecord"), "wb"),
    }
    try:
        for img_id in image_ids:
            split = "valid" if img_id in val_ids else "train"
            example_bytes = _build_example(images_by_id[img_id], anns_by_image.get(img_id, []), label_ids_by_cat)
            write_tfrecord(writers[split], example_bytes)
            counts[split] += 1
    finally:
        for w in writers.values():
            w.close()

    with open(os.path.join(out_dir, "label_map.pbtxt"), "w") as f:
        for cat in categories:
            label_id, name = label_ids_by_cat[cat["id"]]
            f.write(f"item {{\n  id: {label_id}\n  name: '{name}'\n}}\n")

    print(f"Wrote TFRecord dataset to {out_dir}: {counts['train']} train / {counts['valid']} valid")
    print(f"Classes: {[cat['name'] for cat in categories]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_split", type=float, default=0.15)
    args = parser.parse_args()
    convert(args.coco, args.images_dir, args.out_dir, args.val_split)
