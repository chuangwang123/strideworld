"""CoTracker/PnP measurement primitives.

This module is intentionally a measurement layer.  It produces a complete
per-frame pose/world-anchor artifact, but does not decide a Task-1 score.  The
final evaluator in :mod:`task1.evaluate_task1_final` applies the frozen score
contract to these arrays.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .constants import FROZEN_METRIC_SETTINGS, FROZEN_METRIC_SETTINGS_SHA256


TRACK_VISIBILITY_THRESHOLD = float(FROZEN_METRIC_SETTINGS["tracker_visibility_threshold"])
TRACK_CONFIDENCE_THRESHOLD = float(FROZEN_METRIC_SETTINGS["min_track_confidence"])
POSE_RAY_THRESHOLD_RAD = math.radians(float(FROZEN_METRIC_SETTINGS["pose_ransac_threshold_deg"]))
MIN_POSE_ANCHORS = int(FROZEN_METRIC_SETTINGS["min_pose_points"])
MIN_PNP_INLIERS = 6
# Task-1 measurement-quality gate.  This is intentionally a parameter of
# recover_pose_trajectory rather than a hidden post-processing constant so
# every pose artifact records the exact profile used to compute M_t/C_pose.
MIN_POSE_INLIER_RATIO = float(FROZEN_METRIC_SETTINGS["min_pose_inlier_ratio"])
MIN_POSE_ANCHOR_COVERAGE = float(FROZEN_METRIC_SETTINGS["min_pose_anchor_coverage"])


def _scalar(value: Any, default: Any = None) -> Any:
    """Convert numpy scalar/0-d arrays to ordinary JSON-friendly values."""
    if value is None:
        return default
    value = np.asarray(value)
    if value.ndim == 0:
        return value.item()
    return value


def world_to_camera_from_c2w(c2w: np.ndarray) -> np.ndarray:
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.shape != (4, 4):
        raise ValueError("camera-to-world transform must be [4,4]")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = c2w[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ c2w[:3, 3]
    return result


def camera_to_world_from_rt(rotation_w2c: np.ndarray, translation_w2c: np.ndarray) -> np.ndarray:
    rotation_w2c = np.asarray(rotation_w2c, dtype=np.float64)
    translation_w2c = np.asarray(translation_w2c, dtype=np.float64).reshape(3)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_w2c.T
    result[:3, 3] = -rotation_w2c.T @ translation_w2c
    return result


def project_world(xyz_world: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz_world = np.asarray(xyz_world, dtype=np.float64)
    w2c = world_to_camera_from_c2w(c2w)
    xyz_cam = xyz_world @ w2c[:3, :3].T + w2c[:3, 3]
    depth = xyz_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        homogeneous = xyz_cam @ np.asarray(K, dtype=np.float64).T
        uv = homogeneous[:, :2] / homogeneous[:, 2, None]
    return uv, depth


def uv_to_unit_rays(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    homogeneous = np.c_[uv, np.ones(len(uv), dtype=np.float64)]
    rays = homogeneous @ np.linalg.inv(np.asarray(K, dtype=np.float64)).T
    norm = np.linalg.norm(rays, axis=1, keepdims=True)
    return rays / np.maximum(norm, 1e-12)


def ray_angular_errors(observed_uv: np.ndarray, predicted_uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    observed = uv_to_unit_rays(observed_uv, K)
    predicted = uv_to_unit_rays(predicted_uv, K)
    cosine = np.sum(observed * predicted, axis=1)
    sine = np.linalg.norm(np.cross(observed, predicted), axis=1)
    return np.arctan2(sine, np.clip(cosine, -1.0, 1.0))


def _bearing_residual(parameters: np.ndarray, xyz_world: np.ndarray, observed_rays: np.ndarray) -> np.ndarray:
    rvec = np.asarray(parameters[:3], dtype=np.float64)
    tvec = np.asarray(parameters[3:], dtype=np.float64)
    rotation, _ = cv2.Rodrigues(rvec)
    xyz_cam = xyz_world @ rotation.T + tvec
    norm = np.linalg.norm(xyz_cam, axis=1, keepdims=True)
    return (xyz_cam / np.maximum(norm, 1e-12) - observed_rays).reshape(-1)


def refine_camera_in_bearing_space(
    initial_c2w: np.ndarray,
    xyz_world: np.ndarray,
    uv: np.ndarray,
    K: np.ndarray,
    angular_threshold_rad: float = POSE_RAY_THRESHOLD_RAD,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Refine PnP in unit-bearing space and apply the angular inlier gate."""
    from scipy.optimize import least_squares

    initial_uv, initial_depth = project_world(xyz_world, initial_c2w, K)
    initial_angles = ray_angular_errors(uv, initial_uv, K)
    inliers = np.flatnonzero(
        (initial_depth > 0) & np.isfinite(initial_angles) & (initial_angles <= angular_threshold_rad)
    )
    if len(inliers) < MIN_PNP_INLIERS:
        return None, inliers
    w2c = world_to_camera_from_c2w(initial_c2w)
    rvec, _ = cv2.Rodrigues(w2c[:3, :3])
    result = least_squares(
        _bearing_residual,
        np.r_[rvec.reshape(3), w2c[:3, 3]],
        args=(xyz_world[inliers], uv_to_unit_rays(uv[inliers], K)),
        method="trf",
        loss="linear",
        max_nfev=300,
    )
    if not result.success or not np.isfinite(result.x).all():
        return None, inliers
    rotation, _ = cv2.Rodrigues(result.x[:3])
    refined = camera_to_world_from_rt(rotation, result.x[3:])
    final_uv, final_depth = project_world(xyz_world, refined, K)
    final_angles = ray_angular_errors(uv, final_uv, K)
    final_inliers = np.flatnonzero(
        (final_depth > 0) & np.isfinite(final_angles) & (final_angles <= angular_threshold_rad)
    )
    return (refined if len(final_inliers) >= MIN_PNP_INLIERS else None), final_inliers


def estimate_camera_from_correspondences(
    xyz_world: np.ndarray,
    uv: np.ndarray,
    K: np.ndarray,
    reprojection_threshold_px: float | None = None,
    angular_threshold_rad: float = POSE_RAY_THRESHOLD_RAD,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """EPNP + LM followed by the specification's bearing-space refinement."""
    xyz_world = np.asarray(xyz_world, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if len(xyz_world) < MIN_POSE_ANCHORS:
        return None, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if reprojection_threshold_px is None:
        focal = math.sqrt(abs(float(K[0, 0])) * abs(float(K[1, 1])))
        reprojection_threshold_px = focal * math.tan(angular_threshold_rad)
    success, rvec, tvec, ransac_inliers = cv2.solvePnPRansac(
        xyz_world, uv, K, None, iterationsCount=300,
        reprojectionError=float(reprojection_threshold_px), confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or rvec is None or tvec is None:
        return None, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    # The frozen pipeline explicitly performs an LM refinement before the
    # bearing-space pass.  Keeping this separate also makes the intermediate
    # EPNP/LM result straightforward to inspect while debugging.
    lm_ids = ransac_inliers.reshape(-1) if ransac_inliers is not None else np.arange(len(xyz_world))
    if len(lm_ids) < MIN_PNP_INLIERS:
        return None, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    rvec, tvec = cv2.solvePnPRefineLM(xyz_world[lm_ids], uv[lm_ids], K, None, rvec, tvec)
    rotation, _ = cv2.Rodrigues(rvec)
    c2w = camera_to_world_from_rt(rotation, tvec)
    refined, inliers = refine_camera_in_bearing_space(c2w, xyz_world, uv, K, angular_threshold_rad)
    if refined is None:
        return None, inliers, np.full(len(xyz_world), np.nan, dtype=np.float64)
    predicted, _ = project_world(xyz_world, refined, K)
    return refined, inliers, np.linalg.norm(predicted - uv, axis=1)


def load_anchor_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the anchor fields used by the measurement contract."""
    with np.load(path, allow_pickle=False) as data:
        fields = {key: np.asarray(data[key]) for key in data.files}
    required = {"anchor_id", "xyz_world_cm", "is_pose_anchor"}
    missing = required - fields.keys()
    if missing:
        raise ValueError(f"anchor file {path} misses fields: {sorted(missing)}")
    fields["anchor_id"] = fields["anchor_id"].astype(np.int64, copy=False)
    fields["xyz_world_cm"] = fields["xyz_world_cm"].astype(np.float64, copy=False)
    fields["is_pose_anchor"] = fields["is_pose_anchor"].astype(bool, copy=False)
    return fields


def load_track_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load CoTracker output and derive the frozen validity mask."""
    with np.load(path, allow_pickle=False) as data:
        fields = {key: np.asarray(data[key]) for key in data.files}
    required = {"anchor_id", "uv"}
    missing = required - fields.keys()
    if missing:
        raise ValueError(f"track file {path} misses fields: {sorted(missing)}")
    uv = fields["uv"].astype(np.float64, copy=False)
    if uv.ndim != 3 or uv.shape[-1] != 2:
        raise ValueError("track uv must have shape [T,N,2]")
    valid = fields.get("valid", np.ones(uv.shape[:2], dtype=bool)).astype(bool, copy=False)
    visibility = fields.get("visibility_probability", fields.get("visibility", np.ones(uv.shape[:2]))).astype(float)
    confidence = fields.get("confidence", np.ones(uv.shape[:2])).astype(float)
    fields.update({
        "anchor_id": fields["anchor_id"].astype(np.int64, copy=False),
        "uv": uv,
        "valid": valid,
        "visibility_probability": visibility,
        "confidence": confidence,
    })
    image_size = fields.get("image_size")
    if image_size is None:
        image_size = np.array([0, 0], dtype=np.int64)
    fields["image_size"] = np.asarray(image_size, dtype=np.int64)
    return fields


def _in_frame(uv: np.ndarray, image_size: np.ndarray) -> np.ndarray:
    # CoTracker artifacts freeze image_size as [height, width].
    height, width = map(float, image_size.reshape(2))
    if width <= 0 or height <= 0:
        return np.ones(uv.shape[:-1], dtype=bool)
    return (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)


def track_validity_mask(tracks: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return the frozen per-frame, per-anchor CoTracker validity mask."""
    uv = np.asarray(tracks["uv"], dtype=np.float64)
    if uv.ndim != 3 or uv.shape[-1] != 2:
        raise ValueError("track uv must have shape [T,N,2]")
    shape = uv.shape[:2]
    valid = np.asarray(tracks.get("valid", np.ones(shape)), dtype=bool)
    visibility = np.asarray(tracks.get("visibility_probability", np.ones(shape)), dtype=float)
    confidence = np.asarray(tracks.get("confidence", np.ones(shape)), dtype=float)
    if valid.shape != shape or visibility.shape != shape or confidence.shape != shape:
        raise ValueError("track validity fields must all have shape [T,N]")
    return (
        valid
        & (visibility >= TRACK_VISIBILITY_THRESHOLD)
        & (confidence >= TRACK_CONFIDENCE_THRESHOLD)
        & np.isfinite(uv).all(axis=2)
        & _in_frame(uv, np.asarray(tracks.get("image_size", [0, 0])))
    )


def world_measurement_mask(
    pose: Mapping[str, np.ndarray],
    tracks: Mapping[str, np.ndarray],
    anchors: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Align artifacts and return the valid held-out World-anchor samples."""
    pose_ids = np.asarray(pose["anchor_id"], dtype=np.int64)
    track_ids = np.asarray(tracks["anchor_id"], dtype=np.int64)
    anchor_ids = np.asarray(anchors["anchor_id"], dtype=np.int64)
    track_positions = {int(value): index for index, value in enumerate(track_ids)}
    anchor_positions = {int(value): index for index, value in enumerate(anchor_ids)}
    try:
        track_order = np.asarray([track_positions[int(value)] for value in pose_ids], dtype=np.int64)
        anchor_order = np.asarray([anchor_positions[int(value)] for value in pose_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError("pose, track and anchor artifacts contain different anchor_id sets") from error
    mask = track_validity_mask(tracks)[:, track_order]
    mask &= (~np.asarray(anchors["is_pose_anchor"], dtype=bool)[anchor_order])[None]
    depth = np.asarray(pose.get("predicted_depth_cm", np.empty(0)), dtype=np.float64)
    if depth.shape == mask.shape:
        mask &= np.isfinite(depth) & (depth > 0)
    return mask


def is_rigid_transform(matrix: np.ndarray, *, atol: float = 1e-5) -> bool:
    """Return whether a value is a finite, proper 4x4 rigid transform."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        return False
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        return False
    rotation = value[:3, :3]
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=atol)
    )


def recover_pose_trajectory(
    anchors: Mapping[str, np.ndarray],
    tracks: Mapping[str, np.ndarray],
    K: np.ndarray,
    expected_pose_count: int | None = None,
    expected_pose_visible: np.ndarray | None = None,
    min_pose_inlier_ratio: float = MIN_POSE_INLIER_RATIO,
    min_pose_anchor_coverage: float = MIN_POSE_ANCHOR_COVERAGE,
) -> dict[str, Any]:
    """Recover all frames and return a self-contained, debug-friendly artifact."""
    if not 0.0 <= float(min_pose_inlier_ratio) <= 1.0:
        raise ValueError("min_pose_inlier_ratio must be in [0, 1]")
    if not 0.0 <= float(min_pose_anchor_coverage) <= 1.0:
        raise ValueError("min_pose_anchor_coverage must be in [0, 1]")
    anchor_ids = np.asarray(anchors["anchor_id"], dtype=np.int64)
    track_ids = np.asarray(tracks["anchor_id"], dtype=np.int64)
    if not np.array_equal(anchor_ids, track_ids):
        position = {int(value): index for index, value in enumerate(track_ids)}
        try:
            order = np.asarray([position[int(value)] for value in anchor_ids], dtype=np.int64)
        except KeyError as error:
            raise ValueError("anchor and track anchor_id sets differ") from error
        tracks = {key: (value[:, order] if np.asarray(value).ndim >= 2 and np.asarray(value).shape[1] == len(track_ids) else value) for key, value in tracks.items()}
    xyz = np.asarray(anchors["xyz_world_cm"], dtype=np.float64)
    pose_mask = np.asarray(anchors["is_pose_anchor"], dtype=bool)
    world_mask = ~pose_mask
    uv = np.asarray(tracks["uv"], dtype=np.float64)
    T, N = uv.shape[:2]
    eligible = track_validity_mask(tracks)
    if expected_pose_visible is not None:
        expected_visible = np.asarray(expected_pose_visible, dtype=bool)
        if expected_visible.shape != (T, N):
            raise ValueError(f"expected_pose_visible must have shape {(T, N)}")
        expected_visible &= pose_mask[None]
    else:
        expected_visible = np.broadcast_to(pose_mask[None], (T, N)).copy()
    fixed_pose_expected = int(expected_pose_count) if expected_pose_count is not None else None
    pose_coverage = np.zeros(T, dtype=np.float64)
    camera_valid = np.zeros(T, dtype=bool)
    pose_ratio = np.zeros(T, dtype=np.float64)
    cameras = np.full((T, 4, 4), np.nan, dtype=np.float64)
    cameras[:, 3, 3] = 1.0
    predicted_uv = np.full((T, N, 2), np.nan, dtype=np.float64)
    predicted_depth = np.full((T, N), np.nan, dtype=np.float64)
    ray_error = np.full((T, N), np.nan, dtype=np.float64)
    pixel_error = np.full((T, N), np.nan, dtype=np.float64)
    world_scored = np.zeros(T, dtype=np.int32)
    world_scored_mask = np.zeros((T, N), dtype=bool)
    for frame in range(T):
        pose_expected = fixed_pose_expected if fixed_pose_expected is not None else int(expected_visible[frame].sum())
        pose_ids = np.flatnonzero(eligible[frame] & expected_visible[frame])
        pose_coverage[frame] = float(len(pose_ids) / pose_expected) if pose_expected else 0.0
        if len(pose_ids) < MIN_POSE_ANCHORS:
            continue
        c2w, _, _ = estimate_camera_from_correspondences(xyz[pose_ids], uv[frame, pose_ids], K)
        if c2w is None:
            continue
        camera_valid[frame] = True
        cameras[frame] = c2w
        all_uv, all_depth = project_world(xyz, c2w, K)
        predicted_uv[frame], predicted_depth[frame] = all_uv, all_depth
        angles = ray_angular_errors(uv[frame], all_uv, K)
        final_pose_inliers = np.sum((angles <= POSE_RAY_THRESHOLD_RAD) & (all_depth > 0) & eligible[frame] & expected_visible[frame])
        pose_ratio[frame] = float(final_pose_inliers / pose_expected) if pose_expected else 0.0
        world_keep = eligible[frame] & world_mask & (all_depth > 0) & np.isfinite(angles)
        world_scored_mask[frame] = world_keep
        ray_error[frame, world_keep] = angles[world_keep]
        pixel_error[frame, world_keep] = np.linalg.norm(uv[frame, world_keep] - all_uv[world_keep], axis=1)
        world_scored[frame] = int(world_keep.sum())
    measurable = camera_valid & (pose_coverage >= float(min_pose_anchor_coverage)) & (pose_ratio >= float(min_pose_inlier_ratio))
    return {
        "camera_to_world_cv_rh": cameras,
        "camera_valid": camera_valid,
        "pose_inlier_ratio": pose_ratio,
        "pose_anchor_coverage": pose_coverage,
        "measurable": measurable,
        "world_anchor_coverage": world_scored.astype(np.float64) / max(int(world_mask.sum()), 1),
        "world_scored": world_scored,
        "world_scored_mask": world_scored_mask,
        "predicted_uv": predicted_uv.astype(np.float32),
        "predicted_depth_cm": predicted_depth.astype(np.float32),
        "world_ray_error_rad": ray_error.astype(np.float32),
        "world_pixel_error_px": pixel_error.astype(np.float32),
        "frame_ids": np.asarray(tracks.get("frame_ids", np.arange(T)), dtype=str),
        "track_image_size": np.asarray(tracks.get("image_size", [0, 0]), dtype=np.int64),
        "anchor_id": anchor_ids,
        "camera_matrix": np.asarray(K, dtype=np.float64),
        "expected_pose_visible": expected_visible,
        "expected_pose_count": expected_visible.sum(axis=1).astype(np.int32),
        "constants": {
            "visibility_threshold": TRACK_VISIBILITY_THRESHOLD,
            "confidence_threshold": TRACK_CONFIDENCE_THRESHOLD,
            "pose_ray_threshold_deg": 1.0,
            "min_pose_anchors": MIN_POSE_ANCHORS,
            "min_pnp_inliers": MIN_PNP_INLIERS,
            "min_pose_inlier_ratio": float(min_pose_inlier_ratio),
            "min_pose_anchor_coverage": float(min_pose_anchor_coverage),
            "frozen_settings_sha256": FROZEN_METRIC_SETTINGS_SHA256,
        },
        **{
            key: np.asarray(tracks[key])
            for key in ("scoring_prefix_sha256", "first_frame_sha256")
            if key in tracks
        },
    }


def save_pose_npz(path: str | Path, result: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> None:
    """Persist numerical arrays plus a small JSON metadata sidecar field."""
    payload = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    if result.get("camera_matrix") is not None:
        payload["fixed_camera_matrix"] = np.asarray(result["camera_matrix"], dtype=np.float64)
    payload["measurement_constants_json"] = np.asarray(json.dumps(result.get("constants", {}), sort_keys=True))
    if metadata:
        payload["measurement_metadata_json"] = np.asarray(json.dumps(dict(metadata), sort_keys=True, default=str))
        for key in ("model", "scene_id", "case_id", "seed", "route_id", "state_id"):
            if key in metadata and metadata[key] is not None and key not in payload:
                payload[key] = np.asarray(metadata[key])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def run_cotracker(
    video: str | Path,
    anchors: str | Path,
    condition_image: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    *,
    script: str | Path | None = None,
    python: str | Path = sys.executable,
    frame_limit: int | None = None,
    gpu: int = 0,
    cotracker_root: str | Path | None = None,
    source_transform: str = "center_crop",
    first_frame_max_l1: float | None = None,
) -> Path:
    """Call the existing offline CoTracker runner and return its NPZ path."""
    if script is None:
        script = Path(__file__).resolve().parents[2] / "scripts" / "run_cotracker3_task1.py"
    command = [str(python), str(script), "--video", str(video), "--anchors", str(anchors), "--condition-image", str(condition_image), "--checkpoint", str(checkpoint), "--output", str(output), "--gpu", str(gpu), "--source-transform", str(source_transform)]
    if frame_limit is not None:
        command.extend(["--frame-limit", str(frame_limit)])
    if cotracker_root is not None:
        command.extend(["--cotracker-root", str(cotracker_root)])
    if first_frame_max_l1 is not None:
        command.extend(["--first-frame-max-l1", str(first_frame_max_l1)])
    subprocess.run(command, check=True)
    return Path(output)


__all__ = [
    "MIN_POSE_ANCHORS", "MIN_PNP_INLIERS", "POSE_RAY_THRESHOLD_RAD",
    "MIN_POSE_INLIER_RATIO", "MIN_POSE_ANCHOR_COVERAGE",
    "TRACK_CONFIDENCE_THRESHOLD", "TRACK_VISIBILITY_THRESHOLD",
    "camera_to_world_from_rt", "estimate_camera_from_correspondences",
    "load_anchor_npz", "load_track_npz", "project_world", "ray_angular_errors",
    "recover_pose_trajectory", "refine_camera_in_bearing_space", "run_cotracker",
    "save_pose_npz", "track_validity_mask", "uv_to_unit_rays",
    "world_measurement_mask", "world_to_camera_from_c2w", "is_rigid_transform",
]
