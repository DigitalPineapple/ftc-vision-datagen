# Common Field Asset

Real official FTC field CAD, exported directly from Onshape as two OBJ
meshes (units = meters, Z-up, per-part names/colors preserved via the
accompanying `.mtl`):

- `tiles.obj` - the foam field tiles.
- `perimeter.obj` - the perimeter wall/pin assembly, including the clear
  polycarbonate "glass" side panels (named `FTC Field Side Glass ...` in the
  Onshape assembly - `scene_builder.py` detects that name substring to give
  those panels real transparency instead of a flat opaque color).

Both are referenced from `configs/common.yaml -> field.meshes`. The FTC
field perimeter/tile layout is stable year to year, so these typically only
need re-exporting if official field graphics change.

These files are large (~30-115MB) and are **not committed to git** - see the
repo root README's "Large CAD assets are published as GitHub Release assets"
section and run `scripts/assets/fetch_cad_assets.py` after cloning to
restore them.

If either file is absent, `scene_builder.py` automatically falls back to a
flat plane sized to `configs/common.yaml -> field.size_m` with
`fallback_plane_color_rgb`.
