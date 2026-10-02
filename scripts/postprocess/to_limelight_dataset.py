"""
Package a BlenderProc COCO-annotated render directly into a zip ready for
upload to Limelight's Neural Network Trainer
(https://tools.limelightvision.io/neural-network-trainer), which expects a
zipped TFRecord dataset (train.tfrecord + valid.tfrecord + label_map.pbtxt) -
the same layout Roboflow produces via its "TFRecord" export format.

Usage:
    python scripts/postprocess/to_limelight_dataset.py \
        --coco output/2026_biobuzz/coco_annotations.json \
        --images_dir output/2026_biobuzz/images \
        --out_zip output/2026_biobuzz/limelight_dataset.zip --val_split 0.15
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile

from coco_to_tfrecord import convert


def package(coco_path: str, images_dir: str, out_zip: str, val_split: float):
    staging = out_zip + "_staging"
    if os.path.isdir(staging):
        shutil.rmtree(staging)

    convert(coco_path, images_dir, staging, val_split)

    if os.path.isfile(out_zip):
        os.remove(out_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("train.tfrecord", "valid.tfrecord", "label_map.pbtxt"):
            zf.write(os.path.join(staging, name), name)

    shutil.rmtree(staging)
    print(f"Wrote Limelight-ready TFRecord dataset zip: {out_zip}")
    print("Upload to Google Drive (shared publicly / 'anyone with the link'),")
    print("then paste the share link into https://tools.limelightvision.io/neural-network-trainer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_zip", required=True)
    parser.add_argument("--val_split", type=float, default=0.15)
    args = parser.parse_args()
    package(args.coco, args.images_dir, args.out_zip, args.val_split)
