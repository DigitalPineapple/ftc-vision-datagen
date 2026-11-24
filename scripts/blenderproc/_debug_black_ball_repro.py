import blenderproc as bproc

"""
TEMP debug script: fast repro loop for the "solid black ball" rendering bug.

Bypasses the slow real field/Hive/Flower CAD loading entirely (that's what
makes the normal generate_dataset.py pipeline take ~20-30 min per iteration)
and instead builds a minimal scene - just the real piece CAD templates (same
code path as production: build_piece_templates/load_or_create_piece_template)
on a flat floor plane - so each frame renders in seconds, letting us hammer
through many frames to catch the rare (~0.7%) black-ball artifact quickly.

Additionally enables the diffuse-color (albedo) AOV alongside the normal
combined/beauty render for each frame: albedo is the material's base color
BEFORE any lighting is applied, so if a flagged instance's albedo pixels are
also black, the bug is in material/color assignment; if albedo looks normal
(correct class color) but the beauty render is black, the bug is in
lighting/shading (e.g. flipped normals, self-shadowing) instead.

Usage: blenderproc run scripts/blenderproc/_debug_black_ball_repro.py [--frames N] [--pieces N]
"""

import argparse
import os
import sys

import bpy
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_builder import build_label_mapping, build_piece_templates
from randomizers import randomize_lighting, spawn_pieces, randomize_camera
from camera_utils import set_intrinsics_from_camera_cfg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_yaml(rel_path):
    with open(os.path.join(PROJECT_ROOT, rel_path), "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--pieces", type=int, default=25)
    ap.add_argument("--out", default="output/_black_ball_fastrepro")
    ap.add_argument("--dark_threshold", type=float, default=35.0)
    args = ap.parse_args()

    common_cfg = load_yaml("configs/common.yaml")
    cam_cfg = load_yaml("configs/camera/limelight3a.yaml")
    season_cfg = load_yaml("configs/seasons/2026_biobuzz.yaml")

    out_dir = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    bproc.init()
    bpy.context.scene.render.use_persistent_data = False
    bproc.renderer.set_max_amount_of_samples(common_cfg["render"]["samples"])
    bproc.renderer.set_denoiser("INTEL" if common_cfg["render"].get("denoise") else None)
    bproc.renderer.set_output_format(view_transform=common_cfg["render"].get("view_transform", "AgX"))
    bproc.renderer.enable_diffuse_color_output(output_key="diffuse")

    set_intrinsics_from_camera_cfg(cam_cfg)
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id", "instance"], default_values={"category_id": 0}
    )

    label_mapping = build_label_mapping(season_cfg)
    templates = build_piece_templates(season_cfg, PROJECT_ROOT, label_mapping)

    # Minimal flat floor instead of the real (slow-to-load) field CAD - the
    # artifact is per-instance material/shading, not floor-dependent.
    field_size_m = common_cfg["field"]["size_m"]
    w, h = field_size_m
    floor = bproc.object.create_primitive("PLANE")
    floor.set_scale([w / 2.0, h / 2.0, 1.0])
    floor_mat = bproc.material.create("floor_fallback")
    floor_mat.set_principled_shader_value("Base Color", [0.6, 0.6, 0.6, 1.0])
    floor.replace_materials(floor_mat)

    scene = {"templates": templates, "field_elements": []}

    flagged = 0
    for i in range(args.frames):
        bproc.utility.reset_keyframes()
        randomize_lighting(common_cfg, PROJECT_ROOT)
        # Force a denser-than-default spawn to maximize ball-instances/frame.
        season_cfg["spawn"]["total_pieces_range"] = [args.pieces, args.pieces]
        instances = spawn_pieces(scene, season_cfg, field_size_m)
        randomize_camera(cam_cfg, field_size_m, instances)

        data = bproc.renderer.render()
        colors = data["colors"][0]
        diffuse = data["diffuse"][0]
        segmap = data["instance_segmaps"][0]

        inst_by_passidx = {inst.blender_obj.pass_index: inst for inst in instances}
        for pid in np.unique(segmap):
            if pid == 0 or pid not in inst_by_passidx:
                continue
            mask = segmap == pid
            if mask.sum() < 20:
                continue
            mean_rgb = colors[..., :3][mask].mean()
            if mean_rgb < args.dark_threshold:
                inst = inst_by_passidx[pid]
                flagged += 1
                tag = f"frame{i:03d}_inst{pid}"
                print(f"[FLAGGED] {tag}: mean_rgb={mean_rgb:.1f} name={inst.get_name()} "
                      f"loc={inst.get_location()} rot_euler={inst.blender_obj.rotation_euler[:]}")
                diffuse_mean = diffuse[..., :3][mask].mean()
                print(f"          diffuse(albedo)_mean={diffuse_mean:.1f} "
                      f"(low here too => material/color bug; normal here => lighting/shading bug)")

                import cv2
                bgr = colors[..., :3][..., ::-1].copy()
                cv2.imwrite(os.path.join(out_dir, f"{tag}_beauty.png"), bgr)
                diffuse_bgr = diffuse[..., :3][..., ::-1].copy()
                cv2.imwrite(os.path.join(out_dir, f"{tag}_diffuse.png"), diffuse_bgr)
                # Also dump a visualization of just this instance's mask region.
                mask_vis = (mask.astype(np.uint8) * 255)
                cv2.imwrite(os.path.join(out_dir, f"{tag}_mask.png"), mask_vis)

        for inst in instances:
            inst.delete()

        print(f"frame {i+1}/{args.frames} done, {flagged} flagged so far")

    print(f"DONE. {flagged} total flagged instances across {args.frames} frames. See {out_dir}")


if __name__ == "__main__":
    main()
