"""
Per-instance visibility scoring: visible_px / full_unoccluded_px.

visible_px comes from the instance segmentation map (what the COCO writer
also stores as `area`). full_px is the pixel area of the instance's projected
mesh convex hull, NOT clipped to the image - exact enough for near-spherical
game pieces, and needs no extra render pass. A piece cut off by the frame edge
is therefore scored as partly invisible, same as one occluded by other objects.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _world_vertices(obj) -> np.ndarray:
    mesh = obj.get_mesh()
    verts = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3)
    mat = np.array(obj.get_local2world_mat())
    return verts @ mat[:3, :3].T + mat[:3, 3]


def projected_hull(world_pts: np.ndarray, cam2world: np.ndarray, K: np.ndarray
                   ) -> Tuple[Optional[int], Optional[List[int]]]:
    """(pixel area, [x, y, w, h] bbox) of the convex hull of the projected
    points, NOT clipped to the image: the part of a piece outside the frame
    or behind other objects is included, i.e. the full ("amodal") extent.
    (None, None) if any point is at/behind the camera plane (projection invalid)."""
    pts_cam = (np.linalg.inv(cam2world) @ np.c_[world_pts, np.ones(len(world_pts))].T).T[:, :3]
    depth = -pts_cam[:, 2]  # Blender cameras look down -Z
    if np.any(depth <= 1e-4):
        return None, None
    u = K[0, 0] * pts_cam[:, 0] / depth + K[0, 2]
    v = K[1, 2] - K[1, 1] * pts_cam[:, 1] / depth
    hull = np.round(cv2.convexHull(np.stack([u, v], axis=1).astype(np.float32))).astype(np.int32).reshape(-1, 2)
    origin = hull.min(axis=0)
    size = hull.max(axis=0) - origin + 1
    # Same inclusive-pixel convention as the segmap-derived visible bbox.
    bbox = [int(origin[0]), int(origin[1]), int(size[0]), int(size[1])]
    if size.max() > 8000:
        return int(cv2.contourArea(hull.astype(np.float32))), bbox
    # Rasterize on a canvas covering the hull's own extent (rather than the
    # image) so the count matches how the segmap's visible pixels are counted.
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull - origin, 1)
    return int(mask.sum()), bbox


def compute_frame_visibility(segmap: np.ndarray, instances: List, cam2world: np.ndarray,
                             K: np.ndarray) -> List[Dict[str, Any]]:
    """One record per spawned instance that is visible in this frame. Keyed
    for later matching to COCO annotations by (bbox, area)."""
    records = []
    for inst in instances:
        pass_index = inst.blender_obj.pass_index
        mask = segmap == pass_index
        visible_px = int(mask.sum())
        if visible_px < 1:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        bbox = [int(cols[0]), int(rows[0]),
                int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)]
        full_px, amodal_bbox = projected_hull(_world_vertices(inst), cam2world, K)
        visibility = None
        if full_px:
            visibility = round(min(1.0, visible_px / full_px), 4)
        records.append({"bbox": bbox, "area": visible_px, "visible_px": visible_px,
                        "full_px": full_px, "amodal_bbox": amodal_bbox, "visibility": visibility})
    return records


def attach_visibility_to_coco(output_dir: str, records: List[Dict[str, Any]]) -> int:
    """Merge this frame's visibility records into the just-appended COCO
    annotations (the last image in coco_annotations.json). Returns the number
    of annotations matched."""
    path = os.path.join(output_dir, "coco_annotations.json")
    with open(path, "r") as f:
        coco = json.load(f)
    last_image_id = max(img["id"] for img in coco["images"])
    by_key = {(tuple(r["bbox"]), r["area"]): r for r in records}
    matched = 0
    for ann in coco["annotations"]:
        if ann["image_id"] != last_image_id:
            continue
        rec = by_key.get((tuple(ann["bbox"]), ann["area"]))
        if rec is None:
            continue
        ann["visible_px"] = rec["visible_px"]
        ann["full_px"] = rec["full_px"]
        ann["amodal_bbox"] = rec["amodal_bbox"]
        ann["visibility"] = rec["visibility"]
        matched += 1
    with open(path, "w") as f:
        json.dump(coco, f)
    return matched
