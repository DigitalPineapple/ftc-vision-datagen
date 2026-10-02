"""
Builds the season-agnostic scene: field + robot distractors (common, shared
every year) and game-piece templates (season-specific, from configs/seasons).
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List

import blenderproc as bproc

from placeholder_assets import load_or_create_piece_template


def load_field(common_cfg: Dict[str, Any], project_root: str):
    """Load the common perimeter+tile field geometry (same every season).
    Falls back to a flat plane if no mesh has been exported yet."""
    mesh_rel_paths = common_cfg["field"].get("meshes")
    if mesh_rel_paths is None and common_cfg["field"].get("mesh"):
        mesh_rel_paths = [common_cfg["field"]["mesh"]]  # back-compat single-mesh key
    mesh_rel_paths = mesh_rel_paths or []

    loaded = []
    for rel_path in mesh_rel_paths:
        mesh_path = os.path.join(project_root, rel_path)
        if os.path.isfile(mesh_path):
            # Blender's OBJ importer defaults to a traditional Y-up OBJ convention
            # and rotates on import to match Blender's Z-up world. These OBJs are
            # exported directly from Onshape (Z-up, already in meters - see
            # assets/common/field/README.md), so that auto-rotation must be
            # disabled here or the field ends up tipped onto its side.
            objs = bproc.loader.load_obj(mesh_path, forward_axis="NEGATIVE_Y", up_axis="Z")
            loaded.extend(objs)
        else:
            print(f"[scene_builder] Field mesh not found, skipping: {mesh_path}")

    if loaded:
        # Assign materials while names are still the original per-part,
        # group-name-baked ones (glass detection below keys off "glass"
        # appearing in that name - see assets/common/field/README.md) -
        # *before* any renaming/merging, which would destroy that name.
        _style_field_materials(loaded)

        # The Onshape OBJ export has no instancing, so repeated small hardware
        # (rivets, pins, clips - thousands of them) becomes thousands of
        # separate Blender objects. Merge them down to one object per material
        # look (opaque field vs. transparent glass) to keep the scene's total
        # object count (and therefore render/BVH overhead) sane. No physics/
        # rigidbody is used anywhere in this pipeline - pieces are placed at
        # their known resting height directly (see randomizers.spawn_pieces) -
        # so no collision shape is needed on the field at all.
        glass_objs = [o for o in loaded if "glass" in o.get_name().lower()]
        solid_objs = [o for o in loaded if o not in glass_objs]

        field = solid_objs[0]
        if len(solid_objs) > 1:
            field.join_with_other_objects(solid_objs[1:])
        field.set_name("field")

        if glass_objs:
            glass = glass_objs[0]
            if len(glass_objs) > 1:
                glass.join_with_other_objects(glass_objs[1:])
            glass.set_name("field_glass")
    else:
        w, h = common_cfg["field"]["size_m"]
        field = bproc.object.create_primitive("PLANE")
        field.set_scale([w / 2.0, h / 2.0, 1.0])
        mat = bproc.material.create("field_fallback")
        mat.set_principled_shader_value(
            "Base Color", list(common_cfg["field"]["fallback_plane_color_rgb"]) + [1.0]
        )
        field.replace_materials(mat)
        field.set_name("field")
    return field


def _mattify_materials(objs: List) -> None:
    """Real FTC field/structure materials (raw plastic, metal, fabric) are
    matte, but Onshape's OBJ export only carries a flat diffuse Kd color with
    no roughness data, so Blender's default glossy Principled BSDF import
    makes everything look unrealistically shiny/plasticky. Turn down
    specular and add roughness in place, without touching the real color."""
    tuned_materials = set()
    for obj in objs:
        for mat in obj.get_materials():
            if mat is None or mat.get_name() in tuned_materials:
                continue
            tuned_materials.add(mat.get_name())
            mat.set_principled_shader_value("Roughness", 0.85)
            mat.set_principled_shader_value("Specular IOR Level", 0.15)


def _style_field_materials(objs: List):
    """Tune the field's Onshape-exported per-part materials for a realistic
    render instead of the raw exported look:
    - Real FTC perimeter panels include actual clear/translucent polycarbonate
      sections (named "FTC Field Side Glass ..." in the Onshape assembly,
      preserved on these objects' names - see assets/common/field/README.md
      for how that name survives Blender's OBJ import). Give those genuine
      transparency instead of a flat opaque color.
    - Everything else (metal rivets/pins, white plastic panels, gray tiles)
      is matte in real life - see _mattify_materials().
    """
    glass_mat = bproc.material.create("field_glass")
    glass_mat.set_principled_shader_value("Base Color", [0.9, 0.95, 0.98, 1.0])
    glass_mat.set_principled_shader_value("Roughness", 0.02)
    glass_mat.set_principled_shader_value("Transmission Weight", 0.85)
    glass_mat.set_principled_shader_value("IOR", 1.45)
    # Physically, ANY dielectric (regardless of IOR) approaches 100% Fresnel
    # reflectance at true grazing angles - which is exactly the typical view
    # angle from a ball-level camera looking across the field toward the far
    # wall, so pure physically-based transmission alone reads as a near-mirror
    # in most frames, not "see-through". Alpha (unlike Transmission) blends in
    # a flat, angle-independent pass-through of whatever is behind the
    # surface, guaranteeing the panel still reads as visibly semi-transparent
    # at any viewing angle - a deliberate realism-vs-legibility trade.
    glass_mat.set_principled_shader_value("Alpha", 0.35)

    glass_objs = [o for o in objs if "glass" in o.get_name().lower()]
    for obj in glass_objs:
        obj.replace_materials(glass_mat)
    _mattify_materials([o for o in objs if o not in glass_objs])


def _make_cell_wall_material():
    """Hive basket ("cell") walls are thin clear polycarbonate/acrylic panels
    in real life (so refs/cameras can see pieces inside), but Onshape only
    exported them as a flat opaque white Kd color. Give them the same
    glass-style transparency treatment as the field perimeter panels."""
    mat = bproc.material.create("hive_cell_wall")
    mat.set_principled_shader_value("Base Color", [0.95, 0.97, 0.98, 1.0])
    mat.set_principled_shader_value("Roughness", 0.05)
    mat.set_principled_shader_value("Transmission Weight", 0.8)
    mat.set_principled_shader_value("IOR", 1.45)
    mat.set_principled_shader_value("Alpha", 0.4)
    return mat


def _make_apriltag_material(image_path: str):
    """The Onshape export has no image textures, so the April Tag decal
    parts come through as a flat solid Kd=white - indistinguishable from the
    skin/panel they're mounted on. Use a real, math-generated AprilTag 36h11
    marker image (see assets/seasons/2026_biobuzz/textures/, generated via
    cv2.aruco) instead of a flat color or generic checker, so these actually
    look like the genuine fiducial markers used on the real structure."""
    import bpy

    mat = bproc.material.create(f"hive_apriltag_{os.path.basename(image_path)}")
    image = bpy.data.images.load(image_path, check_existing=True)
    tex_node = mat.new_node("ShaderNodeTexImage")
    tex_node.image = image
    tex_node.interpolation = "Closest"  # keep the tag's sharp black/white edges crisp
    tex_coord = mat.new_node("ShaderNodeTexCoord")
    mat.link(tex_coord.outputs["Object"], tex_node.inputs["Vector"])
    mat.set_principled_shader_value("Base Color", tex_node.outputs["Color"])
    mat.set_principled_shader_value("Roughness", 0.4)
    return mat


def _make_panel_sticker_material():
    """'Panel Sticker' is the yellow/honeycomb "FIRST TECH CHALLENGE BIOBUZZ"
    sponsor banner (see reference photo) - but it's a flat Kd=white decal in
    the Onshape export (no real artwork pixels available). We can't recover
    the actual logo, but giving it the banner's real gold/yellow color (vs.
    plain white) is a big legibility win over "invisible blank panel"."""
    mat = bproc.material.create("hive_panel_sticker")
    mat.set_principled_shader_value("Base Color", [0.85, 0.72, 0.28, 1.0])
    mat.set_principled_shader_value("Roughness", 0.35)
    return mat


def _cache_path_for(mesh_path: str) -> str:
    """Processed-template cache lives next to the source CAD file. Keeping it
    file-local (rather than a shared cache dir) means deleting/moving the
    source OBJ naturally invalidates its own cache too."""
    return mesh_path + ".processed_cache.blend"


def _load_or_build_template(mesh_path: str, build_fn):
    """Loading+rejoining the Hive's ~400-fragment, ~1.2M-vertex Onshape OBJ
    export (and re-applying all per-part materials) takes 5-10+ minutes, and
    was being repeated on every debug render even though the processed result
    never changes unless the source CAD/materials do. Cache the single
    merged, fully-materialed template object as a tiny .blend data library
    next to the source OBJ, and reuse it on later runs instead of re-parsing
    the OBJ from scratch. The cache is invalidated automatically if the
    source .obj is touched/re-exported after the cache was written."""
    import bpy

    cache_path = _cache_path_for(mesh_path)
    if os.path.isfile(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(mesh_path):
        try:
            loaded = bproc.loader.load_blend(cache_path, obj_types=["mesh"], data_blocks="objects")
            if loaded:
                obj = loaded[0]
                if len(loaded) > 1:
                    obj.join_with_other_objects(loaded[1:])
                print(f"[scene_builder] Loaded cached processed template: {cache_path}")
                return obj
            print(f"[scene_builder] Cache at {cache_path} contained no mesh objects, rebuilding")
        except Exception as exc:  # pragma: no cover - cache is best-effort
            print(f"[scene_builder] Failed to load cache {cache_path} ({exc}), rebuilding")

    obj = build_fn()
    try:
        bpy.data.libraries.write(cache_path, {obj.blender_obj}, fake_user=True)
        print(f"[scene_builder] Wrote processed template cache: {cache_path}")
    except Exception as exc:  # pragma: no cover - cache is best-effort
        print(f"[scene_builder] Failed to write cache {cache_path} ({exc}), continuing without it")
    return obj


def load_field_elements(season_cfg: Dict[str, Any], project_root: str):
    """Load season-specific static field structures (scoring elements like this
    year's "Hive" goal/basket structure - these change every season, unlike
    the common perimeter/tiles). Each entry needs a `cad_path`; entries whose
    CAD file doesn't exist yet are skipped (no primitive fallback, since we
    don't know these structures' exact geometry the way we do for the simple
    spherical game pieces)."""
    elements = []
    cell_wall_mat = _make_cell_wall_material()
    # 4 real April Tag decals on the hive (blue/red x audience/scoring) - use
    # a distinct generated tag ID per decal for variety, matching how the
    # real field uses different fixed IDs per goal/location.
    # Real Hive tag IDs from the game manual: Red side Audience=30-33,
    # Scoring=34-37; Blue side Audience=38-41, Scoring=42-45 (ranges exist
    # because each physical field at an event gets a different ID to avoid
    # cross-field interference) - picking one representative valid ID per
    # decal slot.
    tex_dir = os.path.join(project_root, "assets", "seasons", "2026_biobuzz", "textures")
    apriltag_mats = {
        key: _make_apriltag_material(os.path.join(tex_dir, f"apriltag_36h11_id{tag_id}.png"))
        for key, tag_id in [("red1", 30), ("red2", 34), ("blue1", 38), ("blue2", 42)]
    }
    default_apriltag_mat = next(iter(apriltag_mats.values()))
    sticker_mat = _make_panel_sticker_material()
    for elem_cfg in season_cfg.get("field_elements", []):
        mesh_path = os.path.join(project_root, elem_cfg["cad_path"])
        if not os.path.isfile(mesh_path):
            print(f"[scene_builder] Skipping field element '{elem_cfg.get('name', '?')}': "
                  f"no CAD found at {mesh_path}")
            continue
        placements = elem_cfg.get("placements", [{"position": [0, 0, 0], "rotation_euler": [0, 0, 0]}])

        def _build_template(mesh_path=mesh_path):
            # Same CAD-sourced axis convention as load_field() - disable the
            # importer's default Y-up-to-Z-up auto-rotation.
            objs = bproc.loader.load_obj(mesh_path, forward_axis="NEGATIVE_Y", up_axis="Z")
            # Re-key per-fragment overrides off each fragment's baked part
            # name (see onshape_download/bake_group_names.py) before any
            # merging, which would destroy that name - same pattern as
            # load_field()'s "glass" name-based detection.
            # Names come from bake_group_names.py's sanitize(), which
            # replaces runs of non-alphanumeric chars (spaces, parens, etc.)
            # with a single "_" - so match on underscore-joined substrings.
            tag_objs = [o for o in objs if "april_tag" in o.get_name().lower()]
            sticker_objs = [o for o in objs if "panel_sticker" in o.get_name().lower()]
            skin_objs = [
                o for o in objs
                if "skin" in o.get_name().lower() and o not in tag_objs and o not in sticker_objs
            ]
            special = set(id(o) for o in tag_objs + sticker_objs + skin_objs)
            plain_objs = [o for o in objs if id(o) not in special]

            for o in tag_objs:
                name_lower = o.get_name().lower()
                tag_key = next((k for k in apriltag_mats if k in name_lower), None)
                o.replace_materials(apriltag_mats.get(tag_key, default_apriltag_mat))
            for o in sticker_objs:
                o.replace_materials(sticker_mat)
            for o in skin_objs:
                o.replace_materials(cell_wall_mat)
            _mattify_materials(plain_objs)

            # Same no-instancing quirk as the rest of the Onshape OBJ exports:
            # a single structure like the hive goal splits into hundreds of
            # separate mesh fragments (hardware, skins, ribs, etc). Materials
            # are tuned above while names/materials were still per-fragment;
            # merge into one object now for render/BVH performance.
            merged = objs[0]
            if len(objs) > 1:
                merged.join_with_other_objects(objs[1:])
            return merged

        # The expensive load+merge+material-tuning above only needs to run
        # once per source CAD file (cached across runs - see
        # _load_or_build_template()); every placement of the same element
        # just duplicates that single processed template and repositions it.
        template = _load_or_build_template(mesh_path, _build_template)
        for i, placement in enumerate(placements):
            obj = template if i == 0 else template.duplicate()
            obj.set_location(placement.get("position", [0, 0, 0]))
            obj.set_rotation_euler(placement.get("rotation_euler", [0, 0, 0]))
            obj.set_name(elem_cfg.get("name", "field_element"))
            elements.append(obj)
    return elements


def load_distractors(common_cfg: Dict[str, Any], season_cfg: Dict[str, Any], project_root: str):
    """Optional generic robot-part meshes to occlude game pieces for realism."""
    if not season_cfg.get("distractors", {}).get("use_robot_meshes", False):
        return []
    dir_path = os.path.join(project_root, common_cfg["robot_distractors"]["dir"])
    if not os.path.isdir(dir_path):
        return []
    mesh_files = [f for f in os.listdir(dir_path) if f.lower().endswith((".obj", ".stp", ".step"))]
    if not mesh_files:
        return []
    max_count = season_cfg["distractors"].get("max_count", 1)
    chosen = random.sample(mesh_files, k=min(max_count, len(mesh_files)))
    distractors = []
    for fname in chosen:
        objs = bproc.loader.load_obj(os.path.join(dir_path, fname))
        distractors.extend(objs)
    return distractors


def build_label_mapping(season_cfg: Dict[str, Any]) -> Dict[str, int]:
    """1-indexed {class_name: category_id}; 0 is reserved for background
    (field/distractors) via enable_segmentation_output's default_values."""
    return {c["name"]: idx for idx, c in enumerate(season_cfg["classes"], start=1)}


def build_piece_templates(season_cfg: Dict[str, Any], project_root: str, label_mapping: Dict[str, int]) -> Dict[str, Any]:
    """Return {class_name: hidden_template_object} for every class in the season config."""
    templates = {}
    for class_cfg in season_cfg["classes"]:
        templates[class_cfg["name"]] = load_or_create_piece_template(
            class_cfg, project_root, label_mapping[class_cfg["name"]]
        )
    return templates


def build_scene(common_cfg: Dict[str, Any], season_cfg: Dict[str, Any], project_root: str):
    field = load_field(common_cfg, project_root)
    field_elements = load_field_elements(season_cfg, project_root)
    distractors = load_distractors(common_cfg, season_cfg, project_root)
    label_mapping = build_label_mapping(season_cfg)
    templates = build_piece_templates(season_cfg, project_root, label_mapping)
    return {
        "field": field,
        "field_elements": field_elements,
        "distractors": distractors,
        "templates": templates,
        "label_mapping": label_mapping,
    }
