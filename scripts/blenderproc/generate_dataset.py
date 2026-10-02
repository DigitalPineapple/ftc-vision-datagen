# blenderproc requires its import to be the literal first line of the entry
# script (checked textually, before any docstring/other import) - see
# blenderproc.python.utility.SetupUtility.check_if_setup_utilities_are_at_the_top
import blenderproc as bproc

"""
Main entry point for synthetic FTC vision dataset generation.

Season-agnostic: pass --season to pick which game-piece config to load.
Everything about field/robot/camera/lighting is shared and defined in
configs/common.yaml + configs/camera/<camera>.yaml.

Usage (run with blenderproc, not plain python):

    blenderproc run scripts/blenderproc/generate_dataset.py \
        --season 2026_biobuzz --camera limelight3a --num_images 200 \
        --output output/2026_biobuzz

Adding a NEW season next year:
    1. Create configs/seasons/<year>_<gamename>.yaml (copy 2026_biobuzz.yaml
       as a template, define classes/spawn rules).
    2. Create assets/seasons/<year>_<gamename>/cad/*.obj (or leave empty to
       use auto-generated placeholder primitives).
    3. Run this script with --season <year>_<gamename>. No code changes needed.
"""

import argparse
import os
import sys

import yaml

from blenderproc.python.utility.LabelIdMapping import LabelIdMapping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_builder import build_scene
from randomizers import randomize_lighting, spawn_pieces, randomize_camera
from camera_utils import set_intrinsics_from_camera_cfg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_yaml(rel_path: str):
    with open(os.path.join(PROJECT_ROOT, rel_path), "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, help="e.g. 2026_biobuzz")
    parser.add_argument("--camera", default="limelight3a")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--output", default=None, help="defaults to output/<season>")
    return parser.parse_args()


def main():
    args = parse_args()

    common_cfg = load_yaml("configs/common.yaml")
    cam_cfg = load_yaml(f"configs/camera/{args.camera}.yaml")
    season_cfg = load_yaml(f"configs/seasons/{args.season}.yaml")

    output_dir = args.output or os.path.join(
        PROJECT_ROOT, common_cfg["output"]["base_dir"], args.season
    )
    os.makedirs(output_dir, exist_ok=True)

    bproc.init()
    bproc.renderer.set_max_amount_of_samples(common_cfg["render"]["samples"])
    bproc.renderer.set_denoiser("INTEL" if common_cfg["render"].get("denoise") else None)
    # "Standard" (BlenderProc/Blender's non-photographic default) clips highlights
    # and looks flat/CG under HDRI lighting. AgX (Blender 4.x's built-in filmic-like
    # tonemapper) rolls off highlights/saturation the way a real camera sensor
    # does, which is the single biggest lever for photorealism here.
    bproc.renderer.set_output_format(
        view_transform=common_cfg["render"].get("view_transform", "AgX")
    )

    scene = build_scene(common_cfg, season_cfg, PROJECT_ROOT)
    field_size_m = common_cfg["field"]["size_m"]

    set_intrinsics_from_camera_cfg(cam_cfg)
    # Templates carry an int category_id (see scene_builder.build_label_mapping);
    # give everything else (field, distractors) a default "background" id of 0.
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id", "instance"], default_values={"category_id": 0}
    )
    label_mapping = LabelIdMapping.from_dict(scene["label_mapping"])

    for i in range(args.num_images):
        # Clear previous frame's spawned instances/camera pose, then re-randomize.
        bproc.utility.reset_keyframes()
        randomize_lighting(common_cfg, PROJECT_ROOT)
        instances = spawn_pieces(scene, season_cfg, field_size_m)
        randomize_camera(cam_cfg, field_size_m, instances)

        data = bproc.renderer.render()

        bproc.writer.write_coco_annotations(
            output_dir,
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            color_file_format="JPEG",
            append_to_existing_output=True,
            label_mapping=label_mapping,
        )

        # Clean up this frame's spawned pieces so the next iteration starts fresh.
        for inst in instances:
            inst.delete()

    print(f"Wrote {args.num_images} rendered images + COCO annotations to {output_dir}")


if __name__ == "__main__":
    main()
