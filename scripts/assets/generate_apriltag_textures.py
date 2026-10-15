"""Generate AprilTag 36h11 marker PNGs for use as field-structure decal
textures (see scripts/blenderproc/scene_builder.py's _make_apriltag_material).

Re-run this whenever a new season needs different tag IDs baked onto its CAD
(e.g. a new goal structure with its own fixed AprilTag family/IDs per the
game manual) - this is the one place `opencv-python` (cv2.aruco) is used in
the whole pipeline.

The Hive's CAD exports each "April Tag Audience/Scoring" panel as a SINGLE
flat decal face per goal side (confirmed by inspecting hive_structure.obj's
geometry - one continuous ~0.43m-wide panel, not 4 separate faces), but the
real structure actually shows 4 distinct AprilTags in a row on each panel
(per the official game manual's per-goal ID ranges: Red Audience=30-33, Red
Scoring=34-37, Blue Audience=38-41, Blue Scoring=42-45). So after generating
the 16 individual tag images, also composite each goal's 4 tags side-by-side
into one row image matching that single decal face.

Usage:
    python scripts/assets/generate_apriltag_textures.py --season 2026_biobuzz \
        --ids 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 --family 36h11 --size_px 512
"""
import argparse
import os

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ARUCO_DICTS = {
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

# Real Hive AprilTag ID ranges per goal/side (see scene_builder.py's
# load_field_elements() for the matching game-manual citation) - each row
# image is composited left-to-right in ascending ID order.
ROW_GROUPS = {
    "red1": list(range(30, 34)),   # Red Goal, Audience side
    "red2": list(range(34, 38)),   # Red Goal, Scoring side
    "blue1": list(range(38, 42)),  # Blue Goal, Audience side
    "blue2": list(range(42, 46)),  # Blue Goal, Scoring side
}


def _generate_tile(aruco_dict, tag_id: int, size_px: int, quiet_frac: float = 0.12):
    """Render one tag with a white quiet-zone margin baked in (on top of the
    tag's own built-in 1-cell black border) - real printed AprilTags always
    carry this margin, and it also keeps every tile visually consistent
    regardless of id (earlier tiles 30/34/38/42 happened to have one from an
    older ad hoc generation; freshly generated ids didn't, which is why one
    tag per row previously looked different from its 3 neighbors)."""
    core_px = max(1, int(round(size_px * (1 - 2 * quiet_frac))))
    core = cv2.aruco.generateImageMarker(aruco_dict, tag_id, core_px)
    pad_lo = (size_px - core_px) // 2
    pad_hi = size_px - core_px - pad_lo
    return cv2.copyMakeBorder(
        core, pad_lo, pad_hi, pad_lo, pad_hi, cv2.BORDER_CONSTANT, value=255
    )


def compose_rows(out_dir: str, family: str, size_px: int, quiet_px: int):
    """Horizontally stitch each group's 4 tag images into one row image,
    with a white quiet-zone gap between tags (matching AprilTag's own white
    border convention, so the composite doesn't read as one merged blob)."""
    for slot, ids in ROW_GROUPS.items():
        tiles = []
        for tag_id in ids:
            tile_path = os.path.join(out_dir, f"apriltag_{family}_id{tag_id}.png")
            tile = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
            if tile is None:
                raise FileNotFoundError(
                    f"Missing tag tile {tile_path} - generate individual tags first."
                )
            if tile.shape != (size_px, size_px):
                tile = cv2.resize(tile, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
            tiles.append(tile)
        gap = np.full((size_px, quiet_px), 255, dtype=np.uint8)
        row_parts = []
        for i, tile in enumerate(tiles):
            if i > 0:
                row_parts.append(gap)
            row_parts.append(tile)
        row_img = np.hstack(row_parts)
        out_path = os.path.join(out_dir, f"apriltag_row_{slot}.png")
        cv2.imwrite(out_path, row_img)
        print(f"Wrote {out_path} (ids {ids})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, help="e.g. 2026_biobuzz")
    parser.add_argument("--ids", type=int, nargs="+", required=True)
    parser.add_argument("--family", default="36h11", choices=list(ARUCO_DICTS))
    parser.add_argument("--size_px", type=int, default=512)
    parser.add_argument("--quiet_px", type=int, default=40,
                         help="white gap (px) between tags in the composited row images")
    parser.add_argument("--skip_rows", action="store_true",
                         help="only generate individual tag PNGs, skip row compositing")
    parser.add_argument("--force", action="store_true",
                         help="regenerate individual tag PNGs even if already present")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "assets", "seasons", args.season, "textures")
    os.makedirs(out_dir, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.family])
    for tag_id in args.ids:
        out_path = os.path.join(out_dir, f"apriltag_{args.family}_id{tag_id}.png")
        if os.path.isfile(out_path) and not args.force:
            print(f"SKIP (already exists, won't overwrite): {out_path}")
            continue
        marker = _generate_tile(aruco_dict, tag_id, args.size_px)
        cv2.imwrite(out_path, marker)
        print(f"Wrote {out_path}")

    if not args.skip_rows:
        compose_rows(out_dir, args.family, args.size_px, args.quiet_px)


if __name__ == "__main__":
    main()

