"""
Domain randomization: lighting, game-piece spawning/placement, camera pose.
These operate purely off the loaded config dicts, so nothing here is
season-specific except the class list/spawn rules already captured in
configs/seasons/<season>.yaml.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List

import numpy as np
import blenderproc as bproc

from camera_utils import sample_camera_pose

# bproc.renderer.enable_segmentation_output() (called once, before any pieces
# are spawned) assigns each mesh object present at that time a unique
# `pass_index`, which Blender's Object Index render pass - and therefore
# BlenderProc's "instance" segmap - uses as the per-pixel instance id.
# template.duplicate() copies the source template object's pass_index as-is,
# so every spawned piece of the same class inherited the SAME id as its
# (hidden) template and therefore the same instance id as every other piece
# duplicated from it - collapsing all of them into a single COCO annotation/
# bounding box per class per frame instead of one per piece. Handing out a
# fresh pass_index to each spawned instance fixes this. NOTE: Object.pass_index
# is clamped by Blender to [0, 32767], so a huge never-reused counter would
# just wrap/clamp back down and collide again - instead start from a small
# base (still well above the handful of static scene objects) and reset it
# every frame, which is safe since all spawned instances are deleted at the
# end of each frame before the next frame's are created.
_INSTANCE_PASS_INDEX_BASE = 1000


def randomize_lighting(common_cfg: Dict[str, Any], project_root: str):
    hdri_dir = os.path.join(project_root, common_cfg["hdri"]["dir"])
    hdri_files = []
    if os.path.isdir(hdri_dir):
        hdri_files = [f for f in os.listdir(hdri_dir) if f.lower().endswith((".hdr", ".exr"))]

    if hdri_files:
        chosen = os.path.join(hdri_dir, random.choice(hdri_files))
        bproc.world.set_world_background_hdr_img(chosen)
    else:
        # Fallback: a few randomized point/area lights approximating gym lighting.
        # (Point-light Watts fall off with distance, but these sit close to the
        # field - 300-1200W here was blowing out the matte field materials to a
        # washed-out, glossy-looking white. Toned down to a more realistic
        # gym-ceiling-light range.)
        for _ in range(random.randint(2, 4)):
            light = bproc.types.Light()
            light.set_type("POINT")
            light.set_location(
                [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(2, 4)]
            )
            light.set_energy(random.uniform(80, 300))


def _world_min_z(obj) -> float:
    """True minimum world-space Z over the object's actual mesh vertices.

    NOT the same as obj.get_bound_box()[:, 2].min(): that helper transforms
    the *local axis-aligned bbox corners* (a box circumscribing the mesh)
    through the world matrix. For round/irregular shapes like our perforated
    wiffle-ball game pieces, the local bbox is a cube that the sphere is
    inscribed in - after rotation, the (empty) cube corners can land well
    below the sphere's actual lowest point, making pieces appear to float
    above the tile floor even though the "bbox" touches Z=0.
    """
    mesh = obj.get_mesh()
    mat = np.array(obj.get_local2world_mat())
    verts = np.array([v.co for v in mesh.vertices])
    verts_h = np.hstack([verts, np.ones((len(verts), 1))])
    world_verts = verts_h @ mat.T
    return float(np.min(world_verts[:, 2]))


def _class_weights(season_cfg: Dict[str, Any]) -> Dict[str, float]:
    classes = season_cfg["classes"]
    mode = season_cfg["spawn"].get("per_class_ratio", "even")
    if mode == "proportional_to_count_in_full_set":
        total = sum(c.get("count_in_full_set", 1) for c in classes)
        return {c["name"]: c.get("count_in_full_set", 1) / total for c in classes}
    # default: even split
    return {c["name"]: 1.0 / len(classes) for c in classes}


def spawn_pieces(scene: Dict[str, Any], season_cfg: Dict[str, Any], field_size_m):
    """Duplicate hidden per-class templates and place them directly at their
    known resting height on the tile floor (Z=0 - see
    assets/common/field/README.md), scattered non-overlappingly across the
    field. No physics/rigidbody simulation is used anywhere in this pipeline -
    we only care about the final static appearance (material/lighting/pose
    realism), not physically-accurate settling dynamics, and skipping physics
    entirely is dramatically faster with the real field CAD's complex mesh.
    """
    templates = scene["templates"]
    weights = _class_weights(season_cfg)
    names, probs = zip(*weights.items())
    class_by_name = {c["name"]: c for c in season_cfg["classes"]}

    lo, hi = season_cfg["spawn"]["total_pieces_range"]
    n_pieces = random.randint(lo, hi)

    field_w, field_h = field_size_m
    margin = 0.9
    max_x, max_y = field_w / 2 * margin, field_h / 2 * margin

    # Real FTC matches tend to have pieces scattered in loose piles rather than
    # perfectly uniform across the whole field - occasionally bias toward a
    # few cluster centers for a more realistic distribution.
    cluster_prob = season_cfg["spawn"].get("cluster_probability", 0.0)
    cluster_centers = []
    if random.random() < cluster_prob:
        cluster_centers = [
            (random.uniform(-max_x, max_x), random.uniform(-max_y, max_y))
            for _ in range(random.randint(1, 3))
        ]
    cluster_spread_m = 0.35

    # Static field structures (Hive base frame, Flower posts, ...) occupy
    # real floor footprint too - without this, pieces were being spawned at
    # random (x, y) with no awareness of them, landing inside/clipping
    # through the Hive's frame legs. get_bound_box() returns world-space
    # AABB corners (see _world_min_z's note above), so take each element's
    # XY extent as a no-spawn rectangle, padded a bit so pieces don't spawn
    # flush against the frame either.
    exclusion_rects = []  # (xmin, xmax, ymin, ymax), already padded
    pad = 0.05
    for el in scene.get("field_elements", []):
        bbox = np.array(el.get_bound_box())
        exclusion_rects.append((
            bbox[:, 0].min() - pad, bbox[:, 0].max() + pad,
            bbox[:, 1].min() - pad, bbox[:, 1].max() + pad,
        ))

    def _blocked_by_structure(x, y, radius):
        for xmin, xmax, ymin, ymax in exclusion_rects:
            if xmin - radius <= x <= xmax + radius and ymin - radius <= y <= ymax + radius:
                return True
        return False

    placed = []  # (x, y, radius) of already-placed pieces, for overlap checks
    instances = []
    next_pass_index = _INSTANCE_PASS_INDEX_BASE
    for _ in range(n_pieces):
        class_name = random.choices(names, weights=probs, k=1)[0]
        template = templates[class_name]
        radius = class_by_name[class_name].get("diameter_m", 0.07) / 2.0

        x, y = 0.0, 0.0
        found_spot = False
        for _attempt in range(30):
            if cluster_centers:
                cx, cy = random.choice(cluster_centers)
                x = float(np.clip(cx + random.uniform(-cluster_spread_m, cluster_spread_m), -max_x, max_x))
                y = float(np.clip(cy + random.uniform(-cluster_spread_m, cluster_spread_m), -max_y, max_y))
            else:
                x = random.uniform(-max_x, max_x)
                y = random.uniform(-max_y, max_y)
            if _blocked_by_structure(x, y, radius):
                continue
            if all(np.hypot(x - px, y - py) >= radius + pr + 0.003 for px, py, pr in placed):
                found_spot = True
                break  # found a non-overlapping spot
        if not found_spot:
            # Field's gotten too crowded (or this spot is pinned against a
            # structure) - skip this piece rather than force it into an
            # overlapping/clipping position. It's fine for pieces to end up
            # occluded *behind* one another from the camera's view; actual
            # interpenetrating geometry is what we're avoiding here.
            continue

        inst = template.duplicate()
        inst.hide(False)
        # Give this instance its own unique instance-segmentation id (see
        # module docstring note above) - otherwise every piece duplicated
        # from the same template shares the template's pass_index and all
        # pieces of that class in a frame collapse into a single annotation.
        inst.blender_obj.pass_index = next_pass_index
        next_pass_index += 1
        inst.set_location([x, y, 0.0])
        inst.set_rotation_euler([random.uniform(0, np.pi) for _ in range(3)])
        # Real Onshape-exported CAD isn't guaranteed to be a perfectly
        # centered sphere (e.g. nectar balls are subtly asymmetric), so rest
        # each instance on the floor using its *actual* post-rotation
        # geometry rather than trusting diameter_m/2 as a fixed offset.
        # NOTE: get_bound_box() just transforms the *local axis-aligned bbox
        # corners* through the rotation - for a sphere the local bbox is a
        # cube circumscribing it, so those corners land well below the true
        # rotated surface and made pieces float above the tiles. We need the
        # true minimum Z over actual mesh vertices instead.
        floor_z = _world_min_z(inst)
        inst.set_location([x, y, -floor_z])
        placed.append((x, y, radius))
        instances.append(inst)

    return instances


def randomize_camera(cam_cfg: Dict[str, Any], field_size_m, instances=None) -> np.ndarray:
    """Aim the camera at a locally-relevant cluster of spawned pieces (with
    jitter) most of the time, matching how a robot's Limelight would actually
    be pointed at the piece(s) it's driving up to intake. Occasionally (10%)
    look at a random field point with no pieces nearby, to include
    background-only negatives.

    Pieces are scattered independently across the whole field, so averaging
    *all* of their positions would often aim at an empty point between
    far-apart pieces. Instead, pick one random "anchor" piece and look at the
    centroid of it plus any neighbors within a small radius - this keeps the
    target (and therefore the pieces in frame) local and camera-reachable.
    """
    look_at_point = None
    if instances and random.random() > 0.1:
        positions = np.array([inst.get_location() for inst in instances])
        anchor = positions[random.randrange(len(positions))]
        nearby_radius_m = 0.5
        dists = np.linalg.norm(positions[:, :2] - anchor[:2], axis=1)
        cluster = positions[dists <= nearby_radius_m]
        centroid = cluster.mean(axis=0)
        jitter = np.array(
            [random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15), random.uniform(-0.05, 0.05)]
        )
        look_at_point = centroid + jitter

    cam2world = sample_camera_pose(cam_cfg, field_size_m, look_at_point=look_at_point)
    bproc.camera.add_camera_pose(cam2world)
    return cam2world
