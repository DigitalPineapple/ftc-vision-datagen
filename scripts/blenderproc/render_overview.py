# blenderproc requires its import to be the literal first line of the entry
# script - see generate_dataset.py's note on this.
import blenderproc as bproc

"""
One-off visualization helper: render a few high, wide-angle "overview" shots
showing the whole field + Hive/Flower structures (plus some scattered game
pieces for context), for visually sanity-checking the scene layout/materials
at a glance. NOT a training-data generator - no sensor degradation, no COCO
annotations, fixed non-robot-mounted camera poses.

Usage:
    blenderproc run scripts/blenderproc/render_overview.py \
        --season 2026_biobuzz --output output/overview
"""

import argparse
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_builder import build_scene
from randomizers import randomize_lighting, spawn_pieces

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_yaml(rel_path: str):
    with open(os.path.join(PROJECT_ROOT, rel_path), "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="2026_biobuzz")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    common_cfg = load_yaml("configs/common.yaml")
    season_cfg = load_yaml(f"configs/seasons/{args.season}.yaml")
    output_dir = args.output or os.path.join(PROJECT_ROOT, "output", "overview")
    os.makedirs(output_dir, exist_ok=True)

    bproc.init()
    bproc.renderer.set_max_amount_of_samples(common_cfg["render"]["samples"])
    bproc.renderer.set_denoiser("INTEL" if common_cfg["render"].get("denoise") else None)
    bproc.renderer.set_output_format(view_transform=common_cfg["render"].get("view_transform", "AgX"))

    scene = build_scene(common_cfg, season_cfg, PROJECT_ROOT)
    field_size_m = common_cfg["field"]["size_m"]
    field_w, field_h = field_size_m

    # Wide-ish still-photo FOV (not the Limelight's narrow FOV) so a high
    # overview shot can actually fit the whole field in frame.
    width, height = 960, 720
    hfov_deg, vfov_deg = 70.0, 55.0
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(K, width, height)

    randomize_lighting(common_cfg, PROJECT_ROOT)
    instances = spawn_pieces(scene, season_cfg, field_size_m)

    # A handful of fixed, high, angled vantage points around the field so the
    # Hive (field center) and Flower structures (wall midpoints) are both
    # visible in context, not just a single top-down shot.
    half_diag = math.hypot(field_w, field_h) / 2.0
    overview_poses = [
        # (x, y, z, look_at)
        (0.0, -half_diag * 1.1, half_diag * 1.3, (0.0, 0.0, 0.1)),
        (half_diag * 1.1, half_diag * 1.1, half_diag * 1.4, (0.0, 0.0, 0.1)),
        (-half_diag * 1.1, 0.0, half_diag * 1.6, (0.0, 0.0, 0.1)),
        (0.0, 0.0, half_diag * 2.2, (0.0, 0.0, 0.0)),  # near-top-down
        # Low, robot-camera-height vantage points looking UP at the Hive
        # (center, ~1.68m tall) and a Flower post (wall midpoint, ~0.59m
        # tall) - the kind of steep upward angle an intake robot driving
        # right up to the goal would actually see, vs. the bird's-eye shots
        # above.
        (1.3, 0.0, 0.2, (0.0, 0.0, 1.2)),             # close, looking up at Hive (red side)
        (-0.9, -0.9, 0.2, (0.0, 0.0, 1.3)),           # looking up at Hive (blue side, diagonal)
        (0.4, 0.0, 0.12, (0.0, 0.0, 1.5)),            # very close/steep look-up at Hive
        (-0.5976, 1.25, 0.2, (-0.5976, 1.7469, 0.4)),  # looking up at a Flower post
    ]
    for x, y, z, look_at in overview_poses:
        location = np.array([x, y, z])
        forward_vec = np.array(look_at) - location
        rotation = bproc.camera.rotation_from_forward_vec(forward_vec)
        cam2world = bproc.math.build_transformation_mat(location, rotation)
        bproc.camera.add_camera_pose(cam2world)

    data = bproc.renderer.render()
    bproc.writer.write_hdf5(output_dir, data, append_to_existing_output=False)

    # Also dump plain JPEGs for quick viewing (hdf5 isn't directly viewable).
    import cv2
    for i, img in enumerate(data["colors"]):
        cv2.imwrite(os.path.join(output_dir, f"overview_{i:02d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    for inst in instances:
        inst.delete()

    print(f"Wrote {len(overview_poses)} overview renders to {output_dir}")


if __name__ == "__main__":
    main()
