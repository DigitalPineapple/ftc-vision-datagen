# 2026 BIOBUZZ Season Assets

This folder holds season-specific CAD: the two game pieces ("Pollen" and
"Nectar") plus the two static scoring structures ("Hive" goal, "Flower"
goals). Everything else (field, robot distractors, HDRIs, camera model)
lives under `assets/common/` and `configs/camera/` and does **not** need to
change here.

## CAD status: real Onshape exports, all wired in

| File | Source | Notes |
|---|---|---|
| `cad/pollen.obj` | AndyMark am-5851 Pollen, exported per-part from Onshape | 2.8" ball |
| `cad/nectar.obj` | AndyMark am-5852 Nectar, exported per-part from Onshape | 3.6" ball; shared geometry, recolored per alliance in config |
| `cad/hive_structure.obj` | AndyMark am-5850 BIOBUZZ Hive, full assembly export | Single combined structure (both alliance sides + both AprilTag decal types) at field center |
| `cad/flower_structure.obj` | AndyMark am-5855 Flower Assembly, full assembly export | One template, duplicated + placed at the 4 wall-mounted locations in `configs/seasons/2026_biobuzz.yaml` |

`hive_structure.obj`/`flower_structure.obj` are large (65-212MB) and are
**not committed to git** - see the repo root README's "Large CAD assets are
published as GitHub Release assets" section and run
`scripts/assets/fetch_cad_assets.py` after cloning to restore them.
`pollen.obj`/`nectar.obj` are small (~1MB) and are committed normally.

The `*.processed_cache.blend` files next to `hive_structure.obj`/
`flower_structure.obj` are a derived, gitignored build cache written by
`scene_builder.py`'s `_load_or_build_template()` (merged mesh + tuned
materials) - delete them any time to force a clean rebuild from the source
OBJ (e.g. after re-exporting the CAD or changing material logic).

`scene_builder.py` automatically prefers real CAD over a placeholder
primitive sphere for `pollen`/`nectar` if it finds a file at the `cad_path`
listed in `configs/seasons/2026_biobuzz.yaml`; `hive_structure`/
`flower_structure` have no placeholder fallback (field structures aren't
simple primitives) - if their CAD is missing they're just skipped.

## If you need to re-export any of these from Onshape

The official BIOBUZZ field Onshape document is:
`https://cad.onshape.com/documents/a355e772e3d24813de7852ee/w/f106353168f1f92100b81259/e/95d1e1e442b4138cccaf2d73`
(direct STEP download: `https://ftc-resources.firstinspires.org/ftc/field/field-cad-step`).

Export the part/assembly you need individually as OBJ (Onshape's own
per-part export avoids assembly-context issues that a whole-field STEP
import runs into). If the export's per-part/group names get collapsed by
Blender's OBJ importer (named only `meshN`), run it through
`onshape_download/bake_group_names.py` first to bake the original Onshape
part names into each `o` line - several special-cased materials in
`scene_builder.py` (AprilTag decals, panel stickers, cell walls, glass) key
off those names surviving import.

## Verify against the official manual

Double check color, diameter, and weight in `configs/seasons/2026_biobuzz.yaml`
against the official BIOBUZZ Game Manual Part 2 once released - the values
here are from early supplier listings and may be refined.
