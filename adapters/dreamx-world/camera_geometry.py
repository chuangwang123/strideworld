"""Auditable DreamX camera bookkeeping.

DreamX native AR uses a fixed normalized PRoPE camera vector.  It does not
consume benchmark K, so source K, resized GT K, and native K are recorded as
separate quantities.  Hash-bound v9 camera/RGB bindings are loaded fail-closed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

MODEL_WIDTH = 1280
MODEL_HEIGHT = 704
ACTUAL_MODEL_CONDITIONING_K = [969.6969696969696 / 1920.0, 969.6969696969696 / 1080.0, 0.5, 0.5]


def _matrix(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    if any(not isinstance(row, list) or len(row) != 3 for row in value):
        return None
    try:
        return [[float(x) for x in row] for row in value]
    except (TypeError, ValueError):
        return None


def _k_matrix(value: Any) -> list[list[float]] | None:
    matrix = _matrix(value)
    if matrix is not None:
        return matrix
    if isinstance(value, dict):
        try:
            fx, fy = float(value["fx"]), float(value["fy"])
            cx, cy = float(value["cx"]), float(value["cy"])
        except (KeyError, TypeError, ValueError):
            return None
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_camera(input_row: dict[str, Any]) -> dict[str, Any]:
    for key in ("camera_geometry", "camera", "camera_metadata"):
        value = input_row.get(key)
        if isinstance(value, dict):
            return value
    return input_row


def _resolve_asset_ref(data_root: Path, uri: str) -> Path:
    relative = uri.removeprefix("data://")
    candidate = Path(relative)
    resolved = candidate.resolve() if candidate.is_absolute() else (data_root / candidate).resolve()
    root = data_root.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("asset URI escapes data root")
    return resolved


def _bound_camera(input_row: dict[str, Any], *, data_root: Path | None,
                  state_id: str | None) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    """Read a camera and RGB pair from SW_ASSET_BINDINGS_PATH, fail closed."""
    overlay = os.environ.get("SW_ASSET_BINDINGS_PATH", "").strip()
    if not overlay:
        return _source_camera(input_row), None, None
    if data_root is None or not state_id:
        return {}, None, "asset binding requires data_root and state_id"
    try:
        binding_path = Path(overlay).resolve()
        if not binding_path.is_file():
            raise ValueError(f"asset bindings path missing: {binding_path}")
        binding_sha = _sha256(binding_path)
        expected = os.environ.get("SW_ASSET_BINDINGS_SHA256", "").strip()
        if expected and expected != binding_sha:
            raise ValueError("asset bindings SHA256 mismatch")
        payload = json.loads(binding_path.read_text(encoding="utf-8"))
        cases = payload.get("cases")
        if not isinstance(cases, list):
            raise ValueError("asset bindings cases missing")
        matches = [case for case in cases if isinstance(case, dict) and case.get("state_id") == state_id]
        if len(matches) != 1:
            raise ValueError(f"state_id is not uniquely bound: {state_id}")
        case = matches[0]
        camera_ref, rgb_ref = case.get("camera"), case.get("first_frame_rgb")
        if not isinstance(camera_ref, dict) or not isinstance(rgb_ref, dict):
            raise ValueError("camera/first_frame_rgb binding missing")
        camera_uri, rgb_uri = camera_ref.get("uri"), rgb_ref.get("uri")
        if not isinstance(camera_uri, str) or not isinstance(rgb_uri, str):
            raise ValueError("camera/RGB URI missing")
        camera_path, rgb_path = _resolve_asset_ref(data_root, camera_uri), _resolve_asset_ref(data_root, rgb_uri)
        if not camera_path.is_file() or not rgb_path.is_file():
            raise ValueError("bound camera or RGB file missing")
        camera_sha, rgb_sha = _sha256(camera_path), _sha256(rgb_path)
        if camera_sha != camera_ref.get("sha256"):
            raise ValueError("bound camera SHA256 mismatch")
        if rgb_sha != rgb_ref.get("sha256"):
            raise ValueError("bound RGB SHA256 mismatch")
        camera_json = json.loads(camera_path.read_text(encoding="utf-8"))
        if not isinstance(camera_json, dict):
            raise ValueError("camera JSON must be an object")
        binding = {
            "schema_version": "strideworld-v9-camera-binding-v1",
            "binding_path": str(binding_path), "binding_sha256": binding_sha,
            "state_id": state_id,
            "camera": {"uri": camera_uri, "sha256": camera_sha, "path": str(camera_path)},
            "first_frame_rgb": {"uri": rgb_uri, "sha256": rgb_sha, "path": str(rgb_path),
                                "width": rgb_ref.get("width"), "height": rgb_ref.get("height")},
        }
        return camera_json, binding, None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {}, None, str(exc)


def _size(source: dict[str, Any], binding: dict[str, Any] | None) -> list[int] | None:
    value = source.get("image_size") or source.get("source_rgb_size")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            pass
    width, height = source.get("width"), source.get("height")
    if isinstance(width, (int, float)) and isinstance(height, (int, float)):
        return [int(width), int(height)]
    if binding:
        rgb = binding.get("first_frame_rgb", {})
        if isinstance(rgb.get("width"), int) and isinstance(rgb.get("height"), int):
            return [rgb["width"], rgb["height"]]
    return None


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def camera_geometry(input_row: dict[str, Any], *, output_pixel_frames: int,
                    output_latent_frames: int, checkpoint_pixels: list[int] | None = None,
                    data_root: Path | None = None, state_id: str | None = None) -> dict[str, Any]:
    """Return auditable geometry and binding information, never a guessed K."""
    source, binding, binding_error = _bound_camera(input_row, data_root=data_root, state_id=state_id)
    k = _k_matrix(source.get("K") or source.get("gt_K") or source.get("source_gt_K"))
    size = _size(source, binding)
    status = "READY" if k is not None and size is not None and binding_error is None else "SOURCE_METADATA_MISSING"
    base: dict[str, Any] = {
        "schema_version": "strideworld-dreamx-camera-geometry-v2", "status": status,
        "binding_error": binding_error, "asset_binding": binding,
        "source_rgb_size_wh": size, "source_gt_K": k,
        "source_gt_K_semantics": "real benchmark camera K in source RGB pixel coordinates" if k is not None else None,
        "model_conditioning_size_wh": [MODEL_WIDTH, MODEL_HEIGHT],
        "actual_model_conditioning_K": list(ACTUAL_MODEL_CONDITIONING_K),
        "actual_model_conditioning_K_parameterization": "[fx/W, fy/H, cx/W, cy/H], fixed DreamX native PRoPE value",
        "native_dreamx_camera_control": "official_AR_entrypoint_fixed_normalized_PRoPE_K_and_action_conditioning",
        "geometry_claim": "source_gt_K, gt_K_in_model_rgb_grid, and fixed actual_model_conditioning_K are separate quantities",
    }
    if status == "READY":
        sw, sh = float(size[0]), float(size[1])
        sx, sy = MODEL_WIDTH / sw, MODEL_HEIGHT / sh
        tx, ty = (sx - 1.0) / 2.0, (sy - 1.0) / 2.0
        resize = [[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]]
        inverse = [[1.0 / sx, 0.0, -tx / sx], [0.0, 1.0 / sy, -ty / sy], [0.0, 0.0, 1.0]]
        base.update({
            "resize_source_rgb_to_model_matrix": resize,
            "resize_model_to_source_rgb_inverse": inverse,
            "resize_policy": "torchvision.Resize_pixel_half_pixel_no_crop_no_letterbox",
            "gt_K_in_model_rgb_grid": _matmul(resize, k),
            "gt_K_in_model_rgb_grid_formula": "A @ source_gt_K, A includes torchvision pixel half_pixel translation",
        })
    pixel_boundaries = list(checkpoint_pixels) if checkpoint_pixels is not None else [4 * i for i in range(1, output_pixel_frames // 4 + 1) if 4 * i < output_pixel_frames]
    native_latent = [0] + [1 + 4 * i for i in range(output_latent_frames - 1)]
    base["boundary_sampling"] = {
        "pixel_frame_index_convention": "zero_based_native_output_pixel_frame",
        "pixel_frame_indices_all": list(range(output_pixel_frames)),
        "action_boundary_pixel_frame_indices": pixel_boundaries,
        "native_latent_camera_sample_indices": native_latent,
        "native_latent_camera_sample_index_convention": "DreamX AR native camera samples: 0 then 1+4*i; not pixel_index//4",
        "scoring_boundary_pixel_frame_indices": pixel_boundaries,
        "scoring_boundaries_are_native_latent_samples": False,
        "scoring_endpoint_relation": "pixel scoring endpoints remain pixel indices; no latent p//4 equivalence is claimed",
    }
    return base


def missing_camera_geometry(*, output_pixel_frames: int, output_latent_frames: int,
                            checkpoint_pixels: list[int] | None = None) -> dict[str, Any]:
    return camera_geometry({}, output_pixel_frames=output_pixel_frames,
                           output_latent_frames=output_latent_frames,
                           checkpoint_pixels=checkpoint_pixels)
