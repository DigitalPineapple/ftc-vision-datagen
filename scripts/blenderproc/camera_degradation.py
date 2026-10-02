"""
Simulate the Limelight 3A's actual sensor/image-pipeline quality instead of
rendering "clean" CG frames: sensor noise (worse in dim gym lighting),
exposure/white-balance drift from auto-exposure, slight blur (defocus +
robot-vibration motion blur), vignetting, and JPEG compression artifacts
(the Limelight streams/processes a compressed feed, not raw sensor data).

Applied as a post-render, pre-write step on BlenderProc's rendered RGB
frames (uint8 HxWx3 numpy arrays), so none of this touches the "clean"
ground-truth geometry/placement - only final pixel quality. Config-driven
(configs/camera/<camera>.yaml -> sensor_degradation) so the realism can be
tuned per camera or dialed back to 0 by setting `enabled: false`.
"""
from __future__ import annotations

import random
from typing import Any, Dict

import cv2
import numpy as np


def _jitter_exposure_and_contrast(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    exposure = random.uniform(*cfg.get("exposure_jitter", [1.0, 1.0]))
    contrast = random.uniform(*cfg.get("contrast_jitter", [1.0, 1.0]))
    out = img.astype(np.float32)
    out = (out - 127.5) * contrast + 127.5  # contrast around mid-gray
    out *= exposure
    return out


def _jitter_white_balance(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    lo, hi = cfg.get("white_balance_jitter", [1.0, 1.0])
    if lo == hi == 1.0:
        return img
    # Opposing R/B gain shift - a cheap but effective stand-in for a sensor's
    # auto white-balance drifting under mixed/variable gym lighting.
    r_gain = random.uniform(lo, hi)
    b_gain = 1.0 / r_gain
    out = img.copy()
    out[..., 0] *= r_gain
    out[..., 2] *= b_gain
    return out


def _maybe_blur(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    if random.random() < cfg.get("motion_blur_prob", 0.0):
        k = random.randrange(*cfg.get("motion_blur_kernel_range", [3, 7]), 2) or 3
        angle = random.uniform(0, 180)
        kernel = _motion_blur_kernel(k, angle)
        return cv2.filter2D(img, -1, kernel)
    if random.random() < cfg.get("gaussian_blur_prob", 0.0):
        sigma = random.uniform(*cfg.get("gaussian_blur_sigma_range", [0.0, 0.0]))
        if sigma > 0:
            return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return img


def _motion_blur_kernel(size: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    rot_mat = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (size, size))
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def _add_sensor_noise(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    if random.random() >= cfg.get("noise_prob", 0.0):
        return img
    out = img.astype(np.float32)

    # Shot noise: variance scales with signal (brighter pixels -> more
    # absolute noise), matching real CMOS sensor behavior, strongest in the
    # low/mid exposure ranges typical of indoor gym lighting.
    shot_scale = random.uniform(*cfg.get("shot_noise_scale_range", [0.0, 0.0]))
    if shot_scale > 0:
        shot = np.random.randn(*out.shape).astype(np.float32) * np.sqrt(np.clip(out, 0, 255)) * shot_scale
        out += shot

    # Fixed read noise: flat gaussian noise independent of signal level.
    read_std = random.uniform(*cfg.get("gaussian_noise_std_range", [0.0, 0.0]))
    if read_std > 0:
        out += np.random.randn(*out.shape).astype(np.float32) * read_std

    return out


def _apply_vignette(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    strength = random.uniform(*cfg.get("vignette_strength_range", [0.0, 0.0]))
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    max_r = np.hypot(cx, cy)
    r = np.hypot(xx - cx, yy - cy) / max_r
    mask = 1.0 - strength * (r ** 2)
    return img * mask[..., None]


def _jpeg_roundtrip(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    lo, hi = cfg.get("jpeg_quality_range", [100, 100])
    if lo >= 100 and hi >= 100:
        return img
    quality = random.randint(lo, hi)
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR if img.shape[2] == 3 else cv2.IMREAD_UNCHANGED)


def apply_sensor_degradation(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Run the full degradation chain on one rendered uint8 RGB frame."""
    if not cfg or not cfg.get("enabled", False):
        return img

    out = _jitter_exposure_and_contrast(img, cfg)
    out = _jitter_white_balance(out, cfg)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = _maybe_blur(out, cfg)
    out = _add_sensor_noise(out, cfg)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = _apply_vignette(out, cfg)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = _jpeg_roundtrip(out, cfg)
    return out
