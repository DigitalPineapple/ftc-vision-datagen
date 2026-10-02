"""
Camera model helpers: convert a camera YAML spec (see configs/camera/) into
BlenderProc intrinsics, and sample randomized-but-realistic mounting poses.

Keeping this separate from the render loop means adding a future camera
(e.g. a different Limelight model) is just a new YAML file - no code changes.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict

import numpy as np
import blenderproc as bproc


def set_intrinsics_from_camera_cfg(cam_cfg: Dict[str, Any]):
    width, height = cam_cfg["resolution"]
    hfov_rad = math.radians(cam_cfg["hfov_deg"])
    # Pinhole model: fx = (width / 2) / tan(hfov / 2)
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    vfov_rad = math.radians(cam_cfg["vfov_deg"])
    fy = (height / 2.0) / math.tan(vfov_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(K, width, height)


def sample_camera_pose(cam_cfg: Dict[str, Any], field_size_m, look_at_point=None):
    """Sample a random-but-plausible robot-mounted camera pose above the field.

    Position: random (x, y) within the field footprint, height per mount config.
    Orientation: aimed at `look_at_point` (e.g. the spawned game pieces'
    centroid) with jitter so the pieces are usually - but not always -
    in frame, matching real Limelight footage where the target isn't
    always dead-center. Falls back to the field-center-with-random-yaw
    behavior if no look_at_point is given (useful for background-only /
    negative training samples).
    """
    field_w, field_h = field_size_m
    # Keep the camera comfortably inside the perimeter walls (their footprint
    # matches field_size_m almost exactly) rather than clipping through them.
    margin = 0.85
    max_x, max_y = field_w / 2 * margin, field_h / 2 * margin

    if look_at_point is not None and "standoff_range_m" in cam_cfg:
        # Place the robot/camera at a realistic distance from the piece(s) it's
        # targeting, instead of anywhere on the field - otherwise the look-at
        # target and camera position are uncorrelated and pieces end up as
        # tiny, distant specks in most frames.
        standoff = random.uniform(*cam_cfg["standoff_range_m"])
        approach_angle = random.uniform(0, 2 * math.pi)
        x = look_at_point[0] + standoff * math.cos(approach_angle)
        y = look_at_point[1] + standoff * math.sin(approach_angle)
        x = float(np.clip(x, -max_x, max_x))
        y = float(np.clip(y, -max_y, max_y))
    else:
        x = random.uniform(-max_x, max_x)
        y = random.uniform(-max_y, max_y)
    z = random.uniform(*cam_cfg["mount_height_range_m"])
    location = np.array([x, y, z])

    roll_deg = random.uniform(-cam_cfg["mount_roll_jitter_deg"], cam_cfg["mount_roll_jitter_deg"])
    roll = math.radians(roll_deg)

    if look_at_point is not None:
        forward_vec = np.array(look_at_point) - location
        rotation = bproc.camera.rotation_from_forward_vec(forward_vec, inplane_rot=roll)
    else:
        tilt_deg = random.uniform(*cam_cfg["mount_tilt_deg_range"])
        yaw_deg = random.uniform(-cam_cfg["mount_yaw_jitter_deg"], cam_cfg["mount_yaw_jitter_deg"])
        # Blender camera looks down -Z by default; pitch by (90 + tilt) to
        # point it forward/down at the field instead of straight down.
        pitch = math.radians(90 + tilt_deg)
        yaw = math.radians(yaw_deg)
        rotation = bproc.math.build_transformation_mat(
            [0, 0, 0], [pitch, roll, yaw]
        )[:3, :3]

    cam2world = bproc.math.build_transformation_mat(location, rotation)
    return cam2world
