# FTC Synthetic Vision Training Data Framework

Generates synthetic, labeled training images for FTC game-element detection
(Limelight 3A neural detector), using BlenderProc for physically-based
rendering + domain randomization.

## Design principle: common vs. season-specific

| Layer | Changes yearly? | Location |
|---|---|---|
| Field, generic robot distractors, lighting HDRIs | No | `assets/common/`, `configs/common.yaml` |
| Camera model (Limelight 3A intrinsics/mount envelope) | No | `configs/camera/` |
| Game piece classes, colors, sizes, spawn rules, CAD | **Yes** | `configs/seasons/<season>.yaml`, `assets/seasons/<season>/` |
| Rendering/randomization/export code | No | `scripts/` |

Adding a new season = adding one YAML config + optional CAD files. No script
changes required.

## Current season: 2026 BIOBUZZ

Classes: `pollen` (yellow, 2.8" ball), `nectar_red` / `nectar_blue`
(alliance-colored, 3.6" ball). Defined in
`configs/seasons/2026_biobuzz.yaml`. Real Onshape-exported CAD is wired in
for all of pollen, nectar, the Hive goal structure, and the 4 Flower goal
structures (see `assets/seasons/2026_biobuzz/README.md` for provenance/
processing notes). If any `cad_path` file is ever missing, game pieces
(not field structures) automatically fall back to correctly-sized/colored
placeholder spheres so the pipeline keeps running.

## Setup

```powershell
.\setup.ps1      # Windows
```
```bash
./setup.sh       # macOS/Linux
```

Each script creates `.venv`, installs `requirements.txt`, and (if the `gh`
CLI is installed and authenticated - `gh auth login`) fetches the large CAD
meshes from the GitHub Release. Safe to re-run. Equivalent manual steps:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\assets\fetch_cad_assets.py   # downloads large CAD meshes from the GitHub Release (see below)
```

BlenderProc downloads/manages its own Blender install on first run
(needs unrestricted internet access to `download.blender.org`). If that host
is blocked in your network (returns HTTP 403 - seen in some sandboxed/CI
environments), download Blender 4.2 LTS from an official mirror instead
(e.g. `https://mirrors.ocf.berkeley.edu/blender/release/Blender4.2/`), unzip
it anywhere, and pass its folder to every `blenderproc run` via
`--custom-blender-path <path-to-unzipped-folder>`.

## Usage

### 1. Render synthetic images + COCO annotations

```powershell
blenderproc run scripts\blenderproc\generate_dataset.py `
    --season 2026_biobuzz --camera limelight3a --num_images 200 `
    --custom-blender-path path\to\unzipped\blender-4.2.16-windows-x64   # omit if auto-download works for you
```

Output: `output/2026_biobuzz/` (images in `images/` + `coco_annotations.json`
at the top level - BlenderProc's native COCO writer layout).

### 2. Convert to YOLO format

```powershell
python scripts\postprocess\coco_to_yolo.py `
    --coco output\2026_biobuzz\coco_annotations.json `
    --images_dir output\2026_biobuzz\images `
    --out_dir output\2026_biobuzz\yolo
```

### 3. Package for Limelight's Neural Network Trainer

```powershell
python scripts\postprocess\to_limelight_dataset.py `
    --yolo_dir output\2026_biobuzz\yolo `
    --out_zip output\2026_biobuzz\limelight_dataset.zip
```

Upload `limelight_dataset.zip` to the Limelight NN Trainer
(https://docs.limelightvision.io/neural-network-training/training-a-custom-detector).

## Recommended: mix with real data

Synthetic-only models tend to have a sim-to-real gap. Blend renders with a
smaller set of real Limelight captures (e.g. 70/30 synthetic/real) before
final training, and iterate domain randomization ranges based on real-field
failure cases.

## Project layout

```
configs/
  common.yaml              # field size, robot distractor dir, HDRI dir, render settings
  camera/limelight3a.yaml  # camera intrinsics + mounting envelope
  seasons/2026_biobuzz.yaml
assets/
  common/{field,robot,hdri}/   # shared every year
  seasons/2026_biobuzz/cad/    # this year's Onshape-exported game pieces + field structures
scripts/
  blenderproc/     # scene_builder, randomizers, camera_utils, placeholder_assets, generate_dataset
  postprocess/     # coco_to_yolo, to_limelight_dataset
  assets/          # package_release_assets.py, fetch_cad_assets.py (large CAD via GitHub Release),
                   # generate_apriltag_textures.py (regenerate AprilTag decal PNGs for new seasons/IDs)
output/            # gitignored render output
```

## Large CAD assets are published as GitHub Release assets

The real field/Hive/Flower CAD exports (`assets/common/field/{perimeter,tiles}.obj`,
`assets/seasons/*/cad/{hive_structure,flower_structure}.obj`) are 29-212MB
each - over GitHub's 100MB hard file-size limit for regular git objects - so
they are **not committed to git**. Instead they're gzip-compressed
(~20% of original size) and attached to a GitHub Release; everything needed
to fetch and verify them is checked into git:

- `configs/assets_release.yaml` - which repo/release tag hosts the assets.
- `assets/large_assets_manifest.json` - maps each release asset back to its
  destination path, with a sha256 checksum for integrity verification.
- `scripts/assets/fetch_cad_assets.py` - downloads + decompresses + verifies
  all large assets via the `gh` CLI (`gh auth login` once beforehand). Safe
  to re-run; skips files that already match their expected checksum. Pass
  `--force` to re-download anyway.
- `scripts/assets/package_release_assets.py` - (maintainers only) re-run
  this after updating any large CAD source file, then publish a new release
  with the regenerated `dist/release_assets/*.gz`, e.g.:
  ```powershell
  gh release create <tag> dist/release_assets/*.gz --repo <owner/repo> `
      --title "CAD assets" --notes "Large CAD meshes, see assets/large_assets_manifest.json"
  ```
  and bump the `tag` in `configs/assets_release.yaml` if it changed.

Each CAD asset's directory has a `README.md` with provenance/re-export notes
in case you need to regenerate one from the official Onshape source instead.
