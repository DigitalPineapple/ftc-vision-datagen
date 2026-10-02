# Onshape CAD Processing Scripts

One-off/reusable utility scripts used while turning raw Onshape STEP/OBJ
exports into the final assets under `assets/common/field/` and
`assets/seasons/<season>/cad/`. Not part of the render pipeline itself
(`scripts/blenderproc/` never imports from here) - these are maintainer
tools you reach for when re-exporting or fixing up CAD.

- `bake_group_names.py` - Blender's OBJ importer names objects from the `o`
  line only, discarding Onshape's `g <PartName>` line. Rewrites each `o`
  line to embed the most recent `g` name so imported object names retain
  real part identity (needed for `scene_builder.py`'s name-based material
  special-casing: AprilTag decals, panel stickers, cell walls, glass).
- `strip_small_parts.py` - Drops small/negligible hardware (rivets, pins,
  fasteners) and other non-visually-significant groups out of a large OBJ
  export by keyword match, properly renumbering vertex/normal indices
  (OBJ indices are global across the whole file, so naive line deletion
  would corrupt face references for every part after a dropped one).
- `fix_mtllib.py` - Patches the `mtllib` reference line in an OBJ's header
  (e.g. after renaming/moving its companion `.mtl` file).
- `convert_field_parts_colored.py` - Re-converts the perimeter/tiles STEP
  files to OBJ preserving per-face color (STEP `STYLED_ITEM`/`COLOUR_RGB`)
  as OBJ material groups, instead of collapsing everything into one
  uncolored compound. Must be run with FreeCAD's bundled Python (has
  `FreeCAD`/`Part`/`MeshPart`/`ImportGui` on its path) - see the script's
  docstring for invocation.

See `assets/common/field/README.md` and
`assets/seasons/2026_biobuzz/README.md` for the full re-export recipe/
provenance notes for each asset.
