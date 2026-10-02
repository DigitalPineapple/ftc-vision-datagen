"""Generate AprilTag 36h11 marker PNGs for use as field-structure decal
textures (see scripts/blenderproc/scene_builder.py's _make_apriltag_material).

Re-run this whenever a new season needs different tag IDs baked onto its CAD
(e.g. a new goal structure with its own fixed AprilTag family/IDs per the
game manual) - this is the one place `opencv-python` (cv2.aruco) is used in
the whole pipeline.

Usage:
    python scripts/assets/generate_apriltag_textures.py --season 2026_biobuzz \
        --ids 30 34 38 42 --family 36h11 --size_px 512
"""
import argparse
import os

import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ARUCO_DICTS = {
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, help="e.g. 2026_biobuzz")
    parser.add_argument("--ids", type=int, nargs="+", required=True)
    parser.add_argument("--family", default="36h11", choices=list(ARUCO_DICTS))
    parser.add_argument("--size_px", type=int, default=512)
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "assets", "seasons", args.season, "textures")
    os.makedirs(out_dir, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.family])
    for tag_id in args.ids:
        out_path = os.path.join(out_dir, f"apriltag_{args.family}_id{tag_id}.png")
        if os.path.isfile(out_path):
            print(f"SKIP (already exists, won't overwrite): {out_path}")
            continue
        marker = cv2.aruco.generateImageMarker(aruco_dict, tag_id, args.size_px)
        cv2.imwrite(out_path, marker)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
