"""Reference-image GT and anchor generation for the StrideWorld handoff.

The handoff package describes an SIFT/grid sampler, while the historical
``tools/data_prep`` extractor uses a dense parity grid.  This module implements
the former explicitly and records the sampler version in every artifact.  It
is a *candidate* generator by default: a depth sidecar marked
``CANDIDATE_REQUIRES_UE_EXPORT_REVIEW`` never silently becomes formal GT.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SAMPLER_VERSION = "strideworld-sift-grid-v1"
GRID_COLUMNS = 8
GRID_ROWS = 6
MAX_POINTS_PER_CELL = 12
SIFT_NFEATURES = 5000
DEPTH_NEIGHBOR_RELATIVE_RANGE = 0.05


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_camera(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    intrinsics = payload.get("K", payload.get("intrinsics"))
    if intrinsics is None:
        raise ValueError(f"camera file has no K/intrinsics: {path}")
    if isinstance(intrinsics, list):
        K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    else:
        K = np.array([
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    image_size = payload.get("image_size", {"width": payload.get("width"), "height": payload.get("height")})
    width = int(image_size["width"] if isinstance(image_size, dict) else image_size[1])
    height = int(image_size["height"] if isinstance(image_size, dict) else image_size[0])
    return {"payload": payload, "K": K, "image_size": np.array([height, width], dtype=np.int64)}


def backproject_camera_z(depth: np.ndarray, K: np.ndarray, raw_to_cm: float = 100.0) -> np.ndarray:
    """Back-project camera-z depth to OpenCV camera coordinates in centimetres."""
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    vv, uu = np.mgrid[0:height, 0:width]
    z = depth * float(raw_to_cm)
    x = (uu - K[0, 2]) / K[0, 0] * z
    y = (vv - K[1, 2]) / K[1, 1] * z
    xyz = np.stack([x, y, z], axis=-1)
    xyz[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return xyz


def _stable_depth(depth: np.ndarray, u: int, v: int, max_depth_m: float) -> bool:
    height, width = depth.shape
    if u < 1 or v < 1 or u >= width - 1 or v >= height - 1:
        return False
    patch = depth[v - 1:v + 2, u - 1:u + 2].astype(np.float64)
    if not np.isfinite(patch).all() or (patch <= 0).any():
        return False
    median = float(np.median(patch))
    center = float(depth[v, u])
    return (
        median > 0
        and median < max_depth_m
        and center < max_depth_m
        and float((patch.max() - patch.min()) / median) <= DEPTH_NEIGHBOR_RELATIVE_RANGE
    )


def _sparse_depth_corners(
    gray: np.ndarray,
    depth: np.ndarray,
    max_depth: float,
) -> list[tuple[int, int, float]]:
    """Select trackable corners whose exact source pixel has metric depth."""
    valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth)
    height, width = depth.shape
    selected: list[tuple[int, int, float]] = []
    for cell_y in range(GRID_ROWS):
        y0, y1 = cell_y * height // GRID_ROWS, (cell_y + 1) * height // GRID_ROWS
        for cell_x in range(GRID_COLUMNS):
            x0, x1 = cell_x * width // GRID_COLUMNS, (cell_x + 1) * width // GRID_COLUMNS
            points = cv2.goodFeaturesToTrack(
                gray[y0:y1, x0:x1], maxCorners=MAX_POINTS_PER_CELL,
                qualityLevel=0.001, minDistance=8,
                mask=(valid[y0:y1, x0:x1].astype(np.uint8) * 255),
                useHarrisDetector=False,
            )
            if points is None:
                continue
            for point in points.reshape(-1, 2):
                u, v = int(round(float(point[0]))) + x0, int(round(float(point[1]))) + y0
                if valid[v, u]:
                    selected.append((u, v, 0.0))
    return selected


def sample_sift_anchors(
    rgb: np.ndarray,
    depth: np.ndarray,
    xyz_camera_cm: np.ndarray,
    source_sha256: str,
    selection_seed: int = 20260913,
    max_depth_m: float = 200.0,
) -> dict[str, Any]:
    """Select deterministic SIFT anchors and split them into Pose/World roles."""
    if rgb.ndim != 3 or rgb.shape[:2] != depth.shape:
        raise ValueError("RGB/depth shape mismatch")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.shape[2] == 3 else rgb
    height, width = depth.shape
    selected: list[tuple[int, int, float]] = []
    valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth_m)
    stable = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    selector = "sift_stable_depth"
    if stable.sum() >= max(24, int(valid.sum() * 0.01)):
        sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
        keypoints, _ = sift.detectAndCompute(gray, None)
        cells: dict[tuple[int, int], list[Any]] = {(x, y): [] for y in range(GRID_ROWS) for x in range(GRID_COLUMNS)}
        for keypoint in keypoints:
            u, v = map(int, np.rint(keypoint.pt))
            if u < 2 or v < 2 or u >= width - 2 or v >= height - 2 or not _stable_depth(depth, u, v, max_depth_m):
                continue
            cell = (min(GRID_COLUMNS - 1, int(u * GRID_COLUMNS / width)), min(GRID_ROWS - 1, int(v * GRID_ROWS / height)))
            cells[cell].append(keypoint)
        for cell in sorted(cells):
            candidates = sorted(cells[cell], key=lambda kp: (-float(kp.response), float(kp.pt[1]), float(kp.pt[0])))
            kept: list[tuple[int, int, float]] = []
            for keypoint in candidates:
                u, v = map(int, np.rint(keypoint.pt))
                if all((u - ku) ** 2 + (v - kv) ** 2 >= 4 for ku, kv, _ in kept):
                    kept.append((u, v, float(keypoint.response)))
                if len(kept) == MAX_POINTS_PER_CELL:
                    break
            selected.extend(kept)
    if len(selected) < 24:
        selected = _sparse_depth_corners(gray, depth, max_depth_m)
        selector = "gftt_sparse_depth"
    selected.sort(key=lambda item: hashlib.sha256(f"{source_sha256}:{selection_seed}:{item[0]}:{item[1]}".encode()).hexdigest())
    if not selected:
        raise ValueError("SIFT/depth filtering selected zero anchors")
    uv = np.asarray([[u, v] for u, v, _ in selected], dtype=np.float32)
    xyz = xyz_camera_cm[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    anchor_id = np.arange(len(uv), dtype=np.int32)
    role = (anchor_id % 2 == 0)
    canonical = [{"anchor_id": int(index), "u": int(u), "v": int(v), "role": "pose" if role[index] else "world"} for index, (u, v, _) in enumerate(selected)]
    return {
        "anchor_id": anchor_id,
        "source_uv": uv,
        "xyz_world_cm": xyz.astype(np.float64),
        "source_depth_cm": xyz[:, 2].astype(np.float32),
        "is_pose_anchor": role,
        "anchor_split_sha256": hashlib.sha256(json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()).hexdigest(),
        "assignment": canonical,
        "config": {
            "sampler_version": SAMPLER_VERSION,
            "feature_selector": selector,
            "selection_seed": selection_seed,
            "grid_columns": GRID_COLUMNS,
            "grid_rows": GRID_ROWS,
            "max_points_per_cell": MAX_POINTS_PER_CELL,
            "sift_nfeatures": SIFT_NFEATURES,
            "depth_neighbor_relative_range": DEPTH_NEIGHBOR_RELATIVE_RANGE,
            "max_depth_m": max_depth_m,
            "role_rule": "hash_sorted_anchor_order_even_pose_odd_world",
        },
    }


def build_candidate_gt(
    condition_image: str | Path,
    depth_npy: str | Path,
    camera_json: str | Path,
    output_dir: str | Path,
    *,
    raw_to_cm: float = 100.0,
    depth_mode: str = "camera_z",
    depth_encoding: str | Path | None = None,
    allow_candidate: bool = True,
    selection_seed: int = 20260913,
    max_valid_depth_cm: float = 20000.0,
) -> dict[str, Any]:
    """Create a reproducible anchor candidate from a reference image/depth."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    condition_image = Path(condition_image)
    depth_npy = Path(depth_npy)
    camera_json = Path(camera_json)
    rgb_bgr = cv2.imread(str(condition_image), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise ValueError(f"cannot read condition image: {condition_image}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth = np.asarray(np.load(depth_npy, allow_pickle=False), dtype=np.float64)
    camera = load_camera(camera_json)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"depth/image shape mismatch: {depth.shape} vs {rgb.shape[:2]}")
    contract = json.loads(Path(depth_encoding).read_text()) if depth_encoding else {}
    status = str(contract.get("status", "UNBOUND"))
    cm_interface = (
        contract.get("quantity") == "camera_z"
        and contract.get("unit") == "cm"
        and contract.get("filled") is False
    )
    formal_eligible = status in {"UE_EXPORT_CONFIRMED", "GEOMETRICALLY_VALIDATED"} or cm_interface
    if status == "UNBOUND" and cm_interface:
        status = "CM_INTERFACE_CONFIRMED"
    if not allow_candidate and not formal_eligible:
        raise ValueError(f"depth contract is not formal: {status}")
    max_depth_m = float(max_valid_depth_cm) / float(raw_to_cm)
    xyz = backproject_camera_z(depth, camera["K"], raw_to_cm)
    xyz[(depth <= 0) | ~np.isfinite(depth) | (depth >= max_depth_m)] = np.nan
    sampled = sample_sift_anchors(rgb, depth, xyz, sha256_file(condition_image), selection_seed, max_depth_m=max_depth_m)
    payload = {
        "anchor_id": sampled["anchor_id"],
        "source_frame": np.asarray("condition"),
        "source_uv": sampled["source_uv"],
        "xyz_world_cm": sampled["xyz_world_cm"],
        "xyz_world_m": sampled["xyz_world_cm"] / 100.0,
        "xyz_world_rh_cm": sampled["xyz_world_cm"],
        "xyz_world_rh_m": sampled["xyz_world_cm"] / 100.0,
        "source_depth_cm": sampled["source_depth_cm"],
        "is_pose_anchor": sampled["is_pose_anchor"],
        "anchor_split_sha256": np.asarray(sampled["anchor_split_sha256"]),
        "anchor_generation_config_json": np.asarray(json.dumps(sampled["config"], sort_keys=True)),
        "anchor_generation_config_sha256": np.asarray(hashlib.sha256(json.dumps(sampled["config"], sort_keys=True).encode()).hexdigest()),
        "condition_image_sha256": np.asarray(sha256_file(condition_image)),
        "source_depth_exr_sha256": np.asarray(sha256_file(depth_npy)),
        "raw_to_cm": np.asarray(raw_to_cm),
        "anchor_sampling_max_depth_cm": np.asarray(max_valid_depth_cm),
        "metric_scale_confirmed": np.asarray(bool(formal_eligible)),
        "depth_contract_confirmed": np.asarray(bool(formal_eligible)),
        "depth_mode": np.asarray("z" if depth_mode == "camera_z" else depth_mode),
        "mode_status": np.asarray(status),
        "scene_id": np.asarray(str(contract.get("scene_id", ""))),
        "depth_encoding_contract_sha256": np.asarray(sha256_file(depth_encoding) if depth_encoding else ""),
        "coordinate_frame": np.asarray("input_camera_cv_rh"),
        "formal_eligible": np.asarray(bool(formal_eligible)),
    }
    np.savez_compressed(output / ("anchors.npz" if formal_eligible else "anchors_candidate.npz"), **payload)
    manifest = {
        "schema_version": "strideworld-gt-candidate-v1",
        "sampler": sampled["config"],
        "anchor_count": int(len(sampled["anchor_id"])),
        "pose_anchor_count": int(sampled["is_pose_anchor"].sum()),
        "world_anchor_count": int((~sampled["is_pose_anchor"]).sum()),
        "condition_image": str(condition_image), "depth_npy": str(depth_npy), "camera_json": str(camera_json),
        "condition_image_sha256": sha256_file(condition_image), "depth_sha256": sha256_file(depth_npy), "camera_sha256": sha256_file(camera_json),
        "depth_contract_status": status, "formal_eligible": bool(formal_eligible),
        "coordinate_frame": "input_camera_cv_rh", "depth_mode": depth_mode, "raw_to_cm": raw_to_cm,
        "anchor_sampling_max_depth_cm": max_valid_depth_cm,
        "anchor_split_sha256": sampled["anchor_split_sha256"], "assignment": sampled["assignment"],
    }
    (output / "gt_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


__all__ = ["SAMPLER_VERSION", "backproject_camera_z", "build_candidate_gt", "load_camera", "sample_sift_anchors"]
