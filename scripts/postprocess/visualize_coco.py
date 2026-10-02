"""
Draw COCO bounding boxes + class labels onto rendered images, for visually
verifying that annotations line up with the actual game pieces (sanity check
after any randomizer/camera/CAD change).

Usage:
    python scripts/postprocess/visualize_coco.py \
        --coco output/2026_biobuzz/coco_annotations.json \
        --images_dir output/2026_biobuzz/images \
        --out_dir output/2026_biobuzz/bbox_preview --limit 10
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import cv2

# Distinct, high-contrast colors per category so overlapping pieces of
# different classes are easy to tell apart at a glance. Cycled if there are
# more categories than colors (BGR, since cv2 draws in BGR).
_COLORS = [
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (0, 255, 0),    # green
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]


def visualize(coco_path: str, images_dir: str, out_dir: str, limit: int = 0):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    color_by_cat = {
        cat_id: _COLORS[i % len(_COLORS)] for i, cat_id in enumerate(sorted(categories))
    }

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    images = coco["images"]
    if limit > 0:
        images = images[:limit]

    for image in images:
        src_path = os.path.join(images_dir, os.path.basename(image["file_name"]))
        if not os.path.isfile(src_path):
            print(f"Skipping missing image: {src_path}")
            continue
        img = cv2.imread(src_path)

        for ann in anns_by_image.get(image["id"], []):
            x, y, w, h = [int(round(v)) for v in ann["bbox"]]  # COCO: top-left x,y, width, height
            cat_id = ann["category_id"]
            name = categories.get(cat_id, str(cat_id))
            color = color_by_cat.get(cat_id, (255, 255, 255))

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f"{name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y - 6 if y - 6 > th else y + h + th + 4
            cv2.rectangle(img, (x, label_y - th - 4), (x + tw + 4, label_y + 2), color, -1)
            cv2.putText(img, label, (x + 2, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        out_path = os.path.join(out_dir, os.path.basename(image["file_name"]))
        cv2.imwrite(out_path, img)

    print(f"Wrote {len(images)} annotated preview images to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    args = parser.parse_args()
    visualize(args.coco, args.images_dir, args.out_dir, args.limit)
