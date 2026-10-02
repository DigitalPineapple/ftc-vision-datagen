"""
Placeholder CAD generator.

Every season's game pieces are defined in configs/seasons/<season>.yaml with a
`cad_path`. If the real Onshape-exported mesh isn't there yet, we synthesize a
correctly sized/colored primitive (currently: sphere) so the rest of the
pipeline (rendering, domain randomization, label export) can be built and
tested before CAD is available. Swapping in real CAD later requires no code
changes - scene_builder.py just prefers the file on disk when it exists.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import blenderproc as bproc


def make_material(color_rgb, roughness_jitter: float = 0.0):
    mat = bproc.material.create("piece_material")
    mat.set_principled_shader_value("Base Color", list(color_rgb) + [1.0])
    mat.set_principled_shader_value("Roughness", 0.55 + roughness_jitter)
    # A perfectly smooth Principled BSDF sphere reads as an obviously-CG,
    # mirror-plastic ball. Real foam/rubber game pieces have subtle surface
    # irregularity (mold seams, texture) - a cheap procedural bump (no extra
    # geometry/render cost worth mentioning) breaks up the flat specular
    # highlight and is one of the highest-impact realism fixes for a sphere.
    nodes = mat.nodes
    links = mat.links
    bsdf = mat.get_the_one_node_with_type("BsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 40.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.6
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def create_placeholder(class_cfg: Dict[str, Any], category_id: int):
    """Create a UV-sphere game-piece stand-in from a season class config entry.

    class_cfg fields used: name, color_rgb, diameter_m, shape (only 'sphere'
    supported for now; extend here if a future season's pieces aren't balls).
    """
    shape = class_cfg.get("shape", "sphere")
    if shape != "sphere":
        raise NotImplementedError(
            f"Placeholder generator only supports 'sphere' shapes so far, "
            f"got '{shape}' for class '{class_cfg['name']}'. Provide a real "
            f"cad_path for this class instead."
        )

    radius = class_cfg["diameter_m"] / 2.0
    obj = bproc.object.create_primitive("SPHERE", radius=radius)
    obj.set_name(f"{class_cfg['name']}_template")
    # COCO writer requires an int category_id; keep the human-readable name too.
    obj.set_cp("category_id", category_id)
    obj.set_cp("category_name", class_cfg["name"])
    mat = make_material(class_cfg["color_rgb"])
    obj.replace_materials(mat)
    obj.hide(True)  # template stays hidden; instances are duplicated from it
    return obj


def load_or_create_piece_template(class_cfg: Dict[str, Any], project_root: str, category_id: int):
    """Load real CAD for a game-piece class if present, else make a placeholder."""
    cad_path = class_cfg.get("cad_path")
    if cad_path:
        full_path = os.path.join(project_root, cad_path)
        if os.path.isfile(full_path):
            # Same Z-up-native Onshape export convention as the field CAD -
            # disable the importer's default Y-up-to-Z-up auto-rotation.
            objs = bproc.loader.load_obj(full_path, forward_axis="NEGATIVE_Y", up_axis="Z")
            obj = objs[0]
            if len(objs) > 1:
                # Like the field CAD, Onshape's OBJ export has no instancing
                # and splits a single part's surface (e.g. the perforated
                # pollen ball) into many separate mesh fragments - join them
                # back into one object so materials/transform/bbox are all
                # correct for the whole piece.
                obj.join_with_other_objects(objs[1:])
            obj.set_name(f"{class_cfg['name']}_template")
            obj.set_cp("category_id", category_id)
            obj.set_cp("category_name", class_cfg["name"])
            mat = make_material(class_cfg["color_rgb"])
            obj.replace_materials(mat)
            obj.hide(True)
            return obj
    return create_placeholder(class_cfg, category_id)
