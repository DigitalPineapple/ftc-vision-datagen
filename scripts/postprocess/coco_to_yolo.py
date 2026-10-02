"""
Convert BlenderProc's COCO-format annotations into YOLO txt labels.
Season-agnostic: class list is read from the COCO categories, so it adapts
automatically to whatever classes were rendered (pollen/nectar this year,
whatever comes next year).

Usage:
    python scripts/postprocess/coco_to_yolo.py --coco output/2026_biobuzz/coco_annotations.json \
        --images_dir output/2026_biobuzz/images --out_dir output/2026_biobuzz/yolo
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def convert(coco_path: str, images_dir: str, out_dir: str):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    cat_id_to_yolo_idx = {
        cat["id"]: idx for idx, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))
    }
    class_names = [
        cat["name"]
        for cat in sorted(coco["categories"], key=lambda c: cat_id_to_yolo_idx[c["id"]])
    ]

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    for image_id, image in images_by_id.items():
        w, h = image["width"], image["height"]
        stem = os.path.splitext(os.path.basename(image["file_name"]))[0]
        label_path = os.path.join(out_dir, "labels", f"{stem}.txt")

        lines = []
        for ann in anns_by_image.get(image_id, []):
            x, y, bw, bh = ann["bbox"]  # COCO: top-left x,y, width, height
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            yolo_cls = cat_id_to_yolo_idx[ann["category_id"]]
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, "w") as lf:
            lf.write("\n".join(lines))

        src_img = os.path.join(images_dir, os.path.basename(image["file_name"]))
        dst_img = os.path.join(out_dir, "images", os.path.basename(image["file_name"]))
        if os.path.isfile(src_img) and not os.path.isfile(dst_img):
            os.link(src_img, dst_img) if hasattr(os, "link") else _copy(src_img, dst_img)

    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print(f"Converted {len(images_by_id)} images -> YOLO format at {out_dir}")
    print(f"Classes: {class_names}")


def _copy(src, dst):
    import shutil

    shutil.copy2(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    convert(args.coco, args.images_dir, args.out_dir)
