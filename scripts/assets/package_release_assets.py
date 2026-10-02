"""Stage gzip-compressed copies of the large CAD mesh assets for upload as
GitHub Release assets, and write/update the manifest that
fetch_cad_assets.py uses to download + verify them after a fresh clone.

Usage:
    python scripts/assets/package_release_assets.py

Outputs:
    dist/release_assets/<flat_name>.obj.gz   (one per large asset)
    assets/large_assets_manifest.json        (checked into git)
"""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = PROJECT_ROOT / "dist" / "release_assets"
MANIFEST_PATH = PROJECT_ROOT / "assets" / "large_assets_manifest.json"

# Large CAD meshes that are impractical to keep as regular git blobs
# (hundreds of MB combined). Everything else under assets/ (materials,
# small game-piece OBJs, HDRIs) stays as normal committed files.
LARGE_ASSETS = [
    "assets/common/field/perimeter.obj",
    "assets/common/field/tiles.obj",
    "assets/seasons/2026_biobuzz/cad/hive_structure.obj",
    "assets/seasons/2026_biobuzz/cad/flower_structure.obj",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def flat_asset_name(rel_path: str) -> str:
    # GitHub release assets are a flat namespace, so encode the relative
    # path into the filename, e.g. "common_field_perimeter.obj.gz".
    return rel_path.replace("assets/", "").replace("/", "_") + ".gz"


def main():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"entries": []}

    for rel_path in LARGE_ASSETS:
        src = PROJECT_ROOT / rel_path
        if not src.exists():
            print(f"SKIP (missing): {rel_path}")
            continue

        asset_name = flat_asset_name(rel_path)
        dest = STAGING_DIR / asset_name

        print(f"Compressing {rel_path} -> dist/release_assets/{asset_name}")
        with open(src, "rb") as f_in, gzip.open(dest, "wb", compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)

        size_orig = src.stat().st_size
        size_gz = dest.stat().st_size
        digest = sha256_of(src)

        manifest["entries"].append({
            "dest_path": rel_path,
            "release_asset_name": asset_name,
            "sha256": digest,
            "size_original": size_orig,
            "size_compressed": size_gz,
        })
        print(f"  {size_orig/1e6:.1f} MB -> {size_gz/1e6:.1f} MB "
              f"({100*size_gz/size_orig:.0f}%)  sha256={digest[:12]}...")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote manifest: {MANIFEST_PATH}")
    print(f"Staged {len(manifest['entries'])} compressed assets in {STAGING_DIR}")
    print("\nNext: create/upload the release, e.g.")
    print('  gh release create <tag> dist/release_assets/*.gz --repo <owner/repo> '
          '--title "CAD assets" --notes "Large CAD meshes, see assets/large_assets_manifest.json"')


if __name__ == "__main__":
    main()
