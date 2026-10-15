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


_randomized_lights: List[Any] = []
_WORLD_NODE_TYPES = (
    "ShaderNodeTexEnvironment", "ShaderNodeMapping", "ShaderNodeTexCoord",
    "ShaderNodeHueSaturation", "ShaderNodeMixRGB",
)


def _hsv_to_rgb(h: float, s: float, v: float):
    import colorsys
    return list(colorsys.hsv_to_rgb(h, s, v))


_hdri_stats_cache: Dict[str, Any] = {}


def _hdri_stats(image):
    """(per-channel white-balance gain, upper-hemisphere irradiance proxy) for
    an equirectangular HDRI. The irradiance proxy is the cosine-weighted mean
    of the upper hemisphere, i.e. roughly how brightly it lights a flat floor."""
    if image.name not in _hdri_stats_cache:
        w, h = image.size
        px = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(px)
        # Blender stores rows bottom-first; flip so row 0 is the zenith.
        rgb = px.reshape(h, w, 4)[::-1, ::8, :3][::4]
        theta = (np.arange(rgb.shape[0]) + 0.5) / rgb.shape[0] * np.pi
        # Cosine-weighted, extra-steep toward the zenith: the perimeter walls block
        # low-elevation light, so a floor is mostly lit from high overhead.
        weight = np.where(theta < np.pi / 2, np.cos(theta) ** 3 * np.sin(theta), 0.0)[:, None, None]
        # Color cast as seen by a floor: cosine-weighted upper hemisphere per channel.
        floor_rgb = (rgb * weight).sum(axis=(0, 1)) / (weight.sum() * rgb.shape[1]) + 1e-6
        irradiance = float(floor_rgb.mean())
        _hdri_stats_cache[image.name] = (floor_rgb.mean() / floor_rgb, irradiance + 1e-6)
    return _hdri_stats_cache[image.name]


def _lighting_gain(image, cfg: Dict[str, Any]) -> List[float]:
    """Per-channel gain: cancels `neutralize` of the HDRI's color cast and
    moves its floor-lighting brightness `normalize_amount` of the way (in log
    space) toward `normalize_target`, so very bright venues don't blow out."""
    wb, irradiance = _hdri_stats(image)
    amount = cfg.get("neutralize", 0.0)
    gain = np.asarray(wb) ** amount if amount > 0 else np.ones(3)
    target = cfg.get("normalize_target")
    if target:
        exposure = (target / irradiance) ** cfg.get("normalize_amount", 1.0)
        gain = gain * float(np.clip(exposure, 0.2, 4.0))
    return [float(g) for g in gain]


def _set_randomized_hdri(path: str, hdri_cfg: Dict[str, Any], bg_paths: List[str]):
    """Build the world shader for this frame: HDRI -> hue/saturation ->
    color tint (multiply) -> Background, with random yaw rotation and
    brightness. BlenderProc's own set_world_background_hdr_img() adds fresh
    nodes on every call without removing the previous frame's, so the world
    node tree is rebuilt from scratch here instead of letting nodes pile up."""
    import bpy

    world = bpy.context.scene.world
    nodes, links = world.node_tree.nodes, world.node_tree.links
    for node in [n for n in nodes if n.bl_idname in _WORLD_NODE_TYPES]:
        nodes.remove(node)
    background = next(n for n in nodes if n.bl_idname == "ShaderNodeBackground")
    background.inputs["Strength"].default_value = 1.0

    def env_chain(hdri_path: str, cfg: Dict[str, Any]):
        """HDRI -> hue/saturation -> tint -> brightness; returns the color socket."""
        tex_coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri_path, check_existing=True)
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        if cfg.get("random_rotation", True):
            mapping.inputs["Rotation"].default_value = [0.0, 0.0, random.uniform(0, 2 * np.pi)]

        hue_sat = nodes.new("ShaderNodeHueSaturation")
        # Hue 0.5 / Saturation 1.0 are the identity values.
        hue_sat.inputs["Hue"].default_value = (
            random.random() if random.random() < cfg.get("hue_shift_prob", 0.0) else 0.5
        )
        hue_sat.inputs["Saturation"].default_value = random.uniform(*cfg.get("saturation_range", [1.0, 1.0]))
        links.new(env.outputs["Color"], hue_sat.inputs["Color"])

        tint = nodes.new("ShaderNodeMixRGB")
        tint.blend_type = "MULTIPLY"
        use_tint = random.random() < cfg.get("tint_prob", 0.0)
        tint.inputs["Fac"].default_value = random.uniform(0.3, 0.9) if use_tint else 0.0
        tint.inputs["Color2"].default_value = _hsv_to_rgb(random.random(), random.uniform(0.4, 1.0), 1.0) + [1.0]
        links.new(hue_sat.outputs["Color"], tint.inputs["Color1"])

        lo, hi = cfg.get("strength_range", [1.0, 1.0])
        # Log-uniform so dim and bright venues are sampled evenly.
        gain = float(np.exp(random.uniform(np.log(lo), np.log(hi))))
        bright = nodes.new("ShaderNodeMixRGB")
        bright.blend_type = "MULTIPLY"
        bright.inputs["Fac"].default_value = 1.0
        bright.inputs["Color2"].default_value = [gain, gain, gain, 1.0]
        links.new(tint.outputs["Color"], bright.inputs["Color1"])

        if cfg.get("neutralize", 0.0) <= 0.0 and not cfg.get("normalize_target"):
            return bright.outputs["Color"]
        # White balance + exposure normalization (see _lighting_gain).
        wb = nodes.new("ShaderNodeMixRGB")
        wb.blend_type = "MULTIPLY"
        wb.inputs["Fac"].default_value = 1.0
        wb.inputs["Color2"].default_value = _lighting_gain(env.image, cfg) + [1.0]
        links.new(bright.outputs["Color"], wb.inputs["Color1"])
        return wb.outputs["Color"]

    lighting_color = env_chain(path, hdri_cfg)

    bg_cfg = hdri_cfg.get("background", {})
    if bg_cfg.get("separate", False) and bg_paths:
        # Camera rays see their own, independently randomized HDRI, while
        # lighting/reflections (diffuse, glossy, shadow rays) keep using the
        # subtle lighting HDRI above - so the visible scenery can vary a lot
        # without changing the floor/tile color or the scene's ambient light.
        bg_color = env_chain(random.choice(bg_paths), bg_cfg)
        light_path = nodes.new("ShaderNodeLightPath")
        select = nodes.new("ShaderNodeMixRGB")
        links.new(light_path.outputs["Is Camera Ray"], select.inputs["Fac"])
        links.new(lighting_color, select.inputs["Color1"])
        links.new(bg_color, select.inputs["Color2"])
        links.new(select.outputs["Color"], background.inputs["Color"])
    else:
        links.new(lighting_color, background.inputs["Color"])


def _add_colored_lights(field_size_m, lights_cfg: Dict[str, Any]):
    """Extra saturated point lights (stage/LED-style venue lighting) around the
    field. Deleted and re-rolled every frame."""
    for light in _randomized_lights:
        light.delete()
    _randomized_lights.clear()

    if random.random() >= lights_cfg.get("prob", 0.6):
        return
    field_w, field_h = field_size_m
    for _ in range(random.randint(*lights_cfg.get("count_range", [1, 3]))):
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location([
            random.uniform(-field_w * 0.8, field_w * 0.8),
            random.uniform(-field_h * 0.8, field_h * 0.8),
            random.uniform(*lights_cfg.get("z_range", [1.0, 3.0])),
        ])
        light.set_color(_hsv_to_rgb(random.random(), random.uniform(0.5, 1.0), 1.0))
        light.set_energy(random.uniform(*lights_cfg.get("energy_range", [60, 300])))
        _randomized_lights.append(light)


def randomize_lighting(common_cfg: Dict[str, Any], project_root: str):
    hdri_cfg = common_cfg["hdri"]
    hdri_dir = os.path.join(project_root, hdri_cfg["dir"])
    hdri_files = []
    if os.path.isdir(hdri_dir):
        hdri_files = [f for f in os.listdir(hdri_dir) if f.lower().endswith((".hdr", ".exr"))]

    if hdri_files:
        full_paths = [os.path.join(hdri_dir, f) for f in hdri_files]
        # Strongly colored-light HDRIs (colored studios) are scenery-only:
        # using them for lighting would tint the floor/tiles.
        excluded = hdri_cfg.get("lighting_exclude", [])
        lighting_paths = [p for p in full_paths if not any(x in os.path.basename(p) for x in excluded)]
        _set_randomized_hdri(random.choice(lighting_paths or full_paths), hdri_cfg, full_paths)
        _add_colored_lights(common_cfg["field"]["size_m"], hdri_cfg.get("colored_lights", {}))
    else:
        # Fallback: a few randomized point/area lights approximating gym lighting.
        # (Point-light Watts fall off with distance, but these sit close to the
        # field - 300-1200W here was blowing out the matte field materials to a
        # washed-out, glossy-looking white. Toned down to a more realistic
        # gym-ceiling-light range.)
        for light in _randomized_lights:
            light.delete()
        _randomized_lights.clear()
        for _ in range(random.randint(2, 4)):
            light = bproc.types.Light()
            light.set_type("POINT")
            light.set_location(
                [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(2, 4)]
            )
            light.set_energy(random.uniform(80, 300))
            _randomized_lights.append(light)


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
    verts = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3)
    # Only the Z row of the affine transform is needed.
    world_z = verts @ mat[2, :3] + mat[2, 3]
    return float(world_z.min())


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
        # Stash the radius as a custom property so randomize_camera() can do
        # a cheap point-in-sphere check against this instance later, without
        # re-deriving it from season_cfg again.
        inst.blender_obj["piece_radius"] = radius
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


def _camera_pose_blocked(location, instances=None, structures=None, pad=0.05, piece_pad=0.12) -> bool:
    """True if `location` (a candidate camera position) ends up physically
    inside a spawned game piece or a static field structure (Hive frame/
    cells/legs, Flower posts, distractor robots, ...). sample_camera_pose()
    only constrains (x, y) to the field footprint and z to the configured
    mount-height range - it has no awareness of what's actually standing at
    that (x, y, z), so without this check the random pose occasionally lands
    the viewpoint *inside* a ball (solid-color/out-of-focus frame filling the
    whole image) or clipped through a structure's geometry.
    """
    loc = np.asarray(location)
    for inst in instances or []:
        radius = inst.blender_obj.get("piece_radius", 0.035)
        if np.linalg.norm(loc - np.array(inst.get_location())) <= radius + piece_pad:
            return True
    for el in structures or []:
        # get_bound_box() gives world-space AABB corners (see spawn_pieces'
        # exclusion_rects note) - a coarse but safe-by-over-exclusion test
        # for irregularly-shaped structures like the Hive frame.
        bbox = np.array(el.get_bound_box())
        xmin, xmax = bbox[:, 0].min() - pad, bbox[:, 0].max() + pad
        ymin, ymax = bbox[:, 1].min() - pad, bbox[:, 1].max() + pad
        zmin, zmax = bbox[:, 2].min() - pad, bbox[:, 2].max() + pad
        if xmin <= loc[0] <= xmax and ymin <= loc[1] <= ymax and zmin <= loc[2] <= zmax:
            return True
    return False


def randomize_camera(
    cam_cfg: Dict[str, Any], field_size_m, instances=None, structures=None
) -> np.ndarray:
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

    cam2world = None
    for _attempt in range(100):
        cam2world = sample_camera_pose(cam_cfg, field_size_m, look_at_point=look_at_point)
        if not _camera_pose_blocked(cam2world[:3, 3], instances, structures):
            break
    else:
        # Still blocked after many tries; retry once without a look-at target
        # (random field position) before accepting a blocked pose.
        for _attempt in range(100):
            cam2world = sample_camera_pose(cam_cfg, field_size_m, look_at_point=None)
            if not _camera_pose_blocked(cam2world[:3, 3], instances, structures):
                break
        else:
            print("[randomizers] Warning: no unblocked camera pose found; using last sample")

    bproc.camera.add_camera_pose(cam2world)
    return cam2world
