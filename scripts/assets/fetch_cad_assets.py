"""Fetch the large CAD mesh assets from the project's GitHub Release and
decompress them into place. Run this once after cloning the repo (the
large OBJs are intentionally NOT committed to git - see
configs/assets_release.yaml and assets/large_assets_manifest.json).

Requires the GitHub CLI (`gh`), already authenticated (`gh auth login`).

Usage:
    python scripts/assets/fetch_cad_assets.py [--force]
"""
import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "assets" / "large_assets_manifest.json"
RELEASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "assets_release.yaml"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Re-download even if a valid local file already exists")
    args = parser.parse_args()

    release_cfg = yaml.safe_load(RELEASE_CONFIG_PATH.read_text())
    repo = release_cfg.get("repo")
    tag = release_cfg.get("tag")
    if not repo:
        sys.exit(f"configs/assets_release.yaml has no 'repo' set - fill it in first "
                  f"(see {RELEASE_CONFIG_PATH}).")

    manifest = json.loads(MANIFEST_PATH.read_text())

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for entry in manifest["entries"]:
            dest = PROJECT_ROOT / entry["dest_path"]

            if dest.exists() and not args.force:
                if sha256_of(dest) == entry["sha256"]:
                    print(f"OK (already present): {entry['dest_path']}")
                    continue
                print(f"MISMATCH, re-downloading: {entry['dest_path']}")

            asset_name = entry["release_asset_name"]
            gz_path = tmp_dir / asset_name
            print(f"Downloading {asset_name} from {repo}@{tag} ...")
            subprocess.run(
                ["gh", "release", "download", tag, "-R", repo,
                 "-p", asset_name, "-O", str(gz_path), "--clobber"],
                check=True,
            )

            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Decompressing -> {entry['dest_path']}")
            with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            digest = sha256_of(dest)
            if digest != entry["sha256"]:
                sys.exit(f"sha256 mismatch for {entry['dest_path']}: "
                          f"expected {entry['sha256']}, got {digest}")
            print(f"  Verified sha256 OK ({dest.stat().st_size/1e6:.1f} MB)")

    print("\nAll large CAD assets fetched and verified.")


if __name__ == "__main__":
    main()
