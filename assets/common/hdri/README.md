# HDRI Lighting Environments

Drop `.hdr`/`.exr` environment maps here (e.g. gym/warehouse interiors) to
randomize scene lighting realistically. Free sources: polyhaven.com (CC0).

`python scripts/assets/fetch_hdris.py` downloads a curated set of ~20 indoor
venues (gyms, halls, arenas, workshops, colored studios); these are not
committed to git. Per-frame rotation, brightness, hue/saturation, tint, and
extra colored lights are configured under `hdri:` in `configs/common.yaml`.

If empty, `randomizers.py` falls back to randomized point lights.
