"""Procedural surface wear (scuffs, dirt, roughness/bump variation) for the
tile floor and the game pieces. Everything multiplies/mixes around the
material's existing base color with a small, mostly-darkening amplitude, so
the class color and the floor color stay essentially unchanged.
"""
from __future__ import annotations

from typing import Any


def _ramp(nodes, lo: float, hi: float, lo_val, hi_val):
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = lo
    ramp.color_ramp.elements[0].color = lo_val
    ramp.color_ramp.elements[1].position = hi
    ramp.color_ramp.elements[1].color = hi_val
    return ramp


def _noise(nodes, links, coord, scale: float, detail: float = 4.0, stretch=(1.0, 1.0, 1.0)):
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = list(stretch)
    links.new(coord, mapping.inputs["Vector"])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    return noise.outputs["Fac"]


def _grey(v: float):
    return [v, v, v, 1.0]


def _math(nodes, links, op: str, a, b=None, clamp: bool = False):
    node = nodes.new("ShaderNodeMath")
    node.operation = op
    node.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            node.inputs[i].default_value = float(v)
        else:
            links.new(v, node.inputs[i])
    return node.outputs["Value"]


def add_floor_wear(mat: Any) -> None:
    """Foam-tile look: mottled shading, directional scuffs, fine grain bump
    and varying roughness. Uses object-space coordinates."""
    nodes, links = mat.nodes, mat.links
    bsdf = mat.get_the_one_node_with_type("BsdfPrincipled")
    base = list(bsdf.inputs["Base Color"].default_value)
    coord = nodes.new("ShaderNodeTexCoord").outputs["Object"]

    # Large mottled variation (0.90..1.0 darkening only).
    mottle = _ramp(nodes, 0.35, 0.65, _grey(0.90), _grey(1.0))
    links.new(_noise(nodes, links, coord, 2.5, 3.0), mottle.inputs["Fac"])

    # Scuffs: thin stretched streaks, darker where present.
    scuff_mask = _ramp(nodes, 0.62, 0.72, _grey(1.0), _grey(0.78))
    links.new(_noise(nodes, links, coord, 14.0, 6.0, stretch=(1.0, 6.0, 1.0)), scuff_mask.inputs["Fac"])

    shade = nodes.new("ShaderNodeMixRGB")
    shade.blend_type = "MULTIPLY"
    shade.inputs["Fac"].default_value = 1.0
    links.new(mottle.outputs["Color"], shade.inputs["Color1"])
    links.new(scuff_mask.outputs["Color"], shade.inputs["Color2"])

    color = nodes.new("ShaderNodeMixRGB")
    color.blend_type = "MULTIPLY"
    color.inputs["Fac"].default_value = 1.0
    color.inputs["Color1"].default_value = base
    links.new(shade.outputs["Color"], color.inputs["Color2"])
    links.new(color.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness 0.7..0.95, rougher on scuffs.
    rough = _ramp(nodes, 0.3, 0.7, _grey(0.7), _grey(0.95))
    links.new(_noise(nodes, links, coord, 6.0), rough.inputs["Fac"])
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])

    # Fine foam grain.
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.002
    links.new(_noise(nodes, links, coord, 400.0, 2.0), bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def add_piece_wear(mat: Any) -> None:
    """Per-instance wear for game pieces (driven by Object Info > Random, so one
    shared material still gives every ball its own dirt level/roughness):
    patchy dirt darkening, lighter scuff specks and roughness variation."""
    nodes, links = mat.nodes, mat.links
    bsdf = mat.get_the_one_node_with_type("BsdfPrincipled")
    base = list(bsdf.inputs["Base Color"].default_value)
    base_rough = float(bsdf.inputs["Roughness"].default_value)

    coord = nodes.new("ShaderNodeTexCoord").outputs["Object"]
    rand = nodes.new("ShaderNodeObjectInfo").outputs["Random"]
    # Pseudo-independent second/third values from the single random output.
    rand2 = _math(nodes, links, "FRACT", _math(nodes, links, "MULTIPLY", rand, 7.31))
    rand3 = _math(nodes, links, "FRACT", _math(nodes, links, "MULTIPLY", rand, 13.17))
    # Offset the pattern per instance so dirt isn't identical on every ball.
    offset = nodes.new("ShaderNodeVectorMath")
    offset.operation = "ADD"
    links.new(coord, offset.inputs[0])
    combine = nodes.new("ShaderNodeCombineXYZ")
    links.new(_math(nodes, links, "MULTIPLY", rand, 50.0), combine.inputs["X"])
    links.new(_math(nodes, links, "MULTIPLY", rand2, 50.0), combine.inputs["Y"])
    links.new(_math(nodes, links, "MULTIPLY", rand3, 50.0), combine.inputs["Z"])
    links.new(combine.outputs["Vector"], offset.inputs[1])
    pcoord = offset.outputs["Vector"]

    # Dirt: patchy darkening up to ~18%, overall amount varies per ball.
    dirt_amount = _math(nodes, links, "MULTIPLY", rand2, 1.0)
    dirt_noise = _ramp(nodes, 0.45, 0.75, _grey(0.0), _grey(1.0))
    links.new(_noise(nodes, links, pcoord, 5.0, 5.0), dirt_noise.inputs["Fac"])
    dirt_fac = _math(nodes, links, "MULTIPLY", dirt_noise.outputs["Color"], _math(nodes, links, "MULTIPLY", dirt_amount, 0.18))
    dirty = nodes.new("ShaderNodeMixRGB")
    dirty.inputs["Fac"].default_value = 1.0
    dirty.inputs["Color1"].default_value = base
    dirty.inputs["Color2"].default_value = [base[0] * 0.55, base[1] * 0.55, base[2] * 0.55, 1.0]
    dirty.blend_type = "MIX"
    links.new(dirt_fac, dirty.inputs["Fac"])

    # Scuffs: small lighter specks (worn plastic), stronger on some balls.
    scuff_noise = _ramp(nodes, 0.70, 0.78, _grey(0.0), _grey(1.0))
    links.new(_noise(nodes, links, pcoord, 60.0, 3.0), scuff_noise.inputs["Fac"])
    scuff_fac = _math(nodes, links, "MULTIPLY", scuff_noise.outputs["Color"], _math(nodes, links, "MULTIPLY", rand3, 0.35))
    scuffed = nodes.new("ShaderNodeMixRGB")
    scuffed.inputs["Color2"].default_value = [0.85, 0.85, 0.85, 1.0]
    links.new(dirty.outputs["Color"], scuffed.inputs["Color1"])
    links.new(scuff_fac, scuffed.inputs["Fac"])
    links.new(scuffed.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness: per-ball offset +-0.15 plus patchy variation; scuffs and dirt are rougher.
    ball_off = _math(nodes, links, "MULTIPLY", _math(nodes, links, "SUBTRACT", rand, 0.5), 0.3)
    patch = _math(nodes, links, "MULTIPLY", _noise(nodes, links, pcoord, 8.0), 0.2)
    wear = _math(nodes, links, "MULTIPLY", _math(nodes, links, "ADD", scuff_fac, dirt_fac), 0.5)
    rough = _math(nodes, links, "ADD", _math(nodes, links, "ADD", ball_off, patch), _math(nodes, links, "ADD", wear, base_rough - 0.1))
    links.new(_math(nodes, links, "MAXIMUM", _math(nodes, links, "MINIMUM", rough, 0.95), 0.25), bsdf.inputs["Roughness"])
