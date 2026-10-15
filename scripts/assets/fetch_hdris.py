"""Download a curated set of CC0 indoor HDRIs from Poly Haven (polyhaven.com)
into assets/common/hdri/ for lighting/background randomization.

These are large (~6 MB each at 2k) and are NOT committed to git (see
.gitignore); run this once after cloning. Already-present files are skipped.

Usage:
    python scripts/assets/fetch_hdris.py [--res 2k] [--force]
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HDRI_DIR = PROJECT_ROOT / "assets" / "common" / "hdri"

# Large, event-like indoor venues (gyms, halls, arenas, workshops, lobbies)
# plus a few strongly colored-light studios for color variety.
HDRI_IDS = [
    "climbing_gym",
    "crossfit_gym",
    "events_hall_interior",
    "circus_arena",
    "cinema_lobby",
    "dancing_hall",
    "basement_boxing_ring",
    "empty_play_room",
    "empty_workshop",
    "entrance_hall",
    "ballroom",
    "billiard_hall",
    "aircraft_workshop_01",
    "abandoned_factory_canteen_01",
    "cinema_hall",
    "dresden_station_night",
    "aerodynamics_workshop",
    "colorful_studio",
    "ferndale_studio_02",
    "ferndale_studio_06",
]

API = "https://api.polyhaven.com/files/"
HEADERS = {"User-Agent": "ftc-vision-datagen/1.0"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", default="2k", help="Poly Haven resolution key (1k, 2k, 4k, ...)")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    HDRI_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for hdri_id in HDRI_IDS:
        dest = HDRI_DIR / f"{hdri_id}_{args.res}.hdr"
        if dest.exists() and not args.force:
            print(f"OK (already present): {dest.name}")
            continue
        try:
            files = json.loads(_get(API + hdri_id))
            url = files["hdri"][args.res]["hdr"]["url"]
            print(f"Downloading {dest.name} ...")
            dest.write_bytes(_get(url))
        except Exception as exc:
            failures += 1
            print(f"  FAILED {hdri_id}: {exc}")
    if failures:
        sys.exit(f"{failures} HDRI download(s) failed")
    print(f"\nAll HDRIs present in {HDRI_DIR}")


if __name__ == "__main__":
    main()
