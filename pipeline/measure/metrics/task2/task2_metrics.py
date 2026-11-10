"""StrideWorld Task 2 metric implementation.

This module implements the final specification's held-action Task 2 contract
on top of the pose artifacts produced by :mod:`metrics.tools.pose`.  A cell is
one atomic video (model, state, seed, action) containing sixteen repeated
units.  The module deliberately keeps the cell calculations separate from
the aggregation code so the same logic is used by the single-process CLI and
the persistent worker runner.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from metrics.tools.constants import FROZEN_METRIC_SETTINGS, FROZEN_METRIC_SETTINGS_SHA256
from metrics.tools.pose import load_anchor_npz, load_track_npz, world_measurement_mask
from metrics.tools.se2 import (
    action_name,
    action_scalar,
    distance_se2,
    exp_se2,
    log_se2,
    project_se3_to_se2,
    se2_power,
)


SCHEMA_VERSION = "strideworld-task2-final-v2"
ACTIONS = ("W", "S", "A", "D", "YL", "YR")
TRANSLATION_ACTIONS = frozenset(("W", "S", "A", "D"))
ROTATION_ACTIONS = frozenset(("YL", "YR"))
DOSES = tuple(int(value) for value in FROZEN_METRIC_SETTINGS["checkpoint_doses"])
UNIT_DOSES = tuple(k for k in DOSES if k <= int(FROZEN_METRIC_SETTINGS["unit_estimation_dose_max"]))
TASK3_OPERATOR_DOSES = tuple(k for k in DOSES if k <= int(FROZEN_METRIC_SETTINGS["task3_operator_dose_max"]))
POWER_LAW_DOSES = tuple(k for k in DOSES if k <= int(FROZEN_METRIC_SETTINGS["power_law_dose_max"]))
ROBUST_SCALE = float(FROZEN_METRIC_SETTINGS["robust_scale"])
EPSILON = float(FROZEN_METRIC_SETTINGS["epsilon"])
TRANSLATION_FLOOR_CM = float(FROZEN_METRIC_SETTINGS["translation_floor_cm"])
ROTATION_FLOOR_DEG = float(FROZEN_METRIC_SETTINGS["rotation_floor_deg"])
MIN_WORLD_POINTS = int(FROZEN_METRIC_SETTINGS["min_world_points"])
DEFAULT_MIN_POSE_INLIER_RATIO = float(FROZEN_METRIC_SETTINGS["min_pose_inlier_ratio"])
DEFAULT_MIN_POSE_ANCHOR_COVERAGE = float(FROZEN_METRIC_SETTINGS["min_pose_anchor_coverage"])


def scalar(value: Any, default: Any = None) -> Any:
    """Convert numpy scalar values to ordinary Python values."""
    if value is None:
        return default
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def state_id_for_key(key: Mapping[str, Any]) -> str:
    return f"{key.get('scene_id')}_{key.get('case_id')}_state0000"


def action_from_route(route_id: str) -> str | None:
    token = str(route_id).strip()
    if token.lower().startswith("atomic_"):
        token = token.split("_", 1)[1]
    try:
        action = action_name(token)
    except KeyError:
        return None
    return action if action in ACTIONS else None


def ready_records(
    data_root: str | Path,
    *,
    model: str | None = None,
    seed: int | None = None,
    task: str = "task2",
) -> list[dict[str, Any]]:
    """Load the fixed-denominator atomic action records."""
    index = Path(data_root) / "common" / "index" / "by_host" / "eval.ready.jsonl"
    if not index.exists():
        raise FileNotFoundError(index)
    rows: list[dict[str, Any]] = []
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("primary_key", {})
        if key.get("task") != task:
            continue
        if model is not None and key.get("model") != model:
            continue
        if seed is not None and int(key.get("seed")) != int(seed):
            continue
        if action_from_route(str(key.get("route_id", ""))) is None:
            continue
        rows.append(row)
    rows.sort(key=lambda row: tuple(str(row["primary_key"].get(name)) for name in (
        "model", "scene_id", "case_id", "seed", "route_id"
    )))
    if not rows:
        raise ValueError("ready index filter selected no Task 2 atomic records")
    return rows


def record_path(data_root: str | Path, row: Mapping[str, Any]) -> Path:
    absolute = row.get("record_path") or row.get("source_record") or row.get("retained_source_record")
    if absolute is not None:
        return Path(str(absolute))
    value = row.get("record_relpath")
    if value is None:
        raise KeyError("ready record has no record_relpath")
    return Path(data_root) / "common" / str(value)


def pose_metadata(pose_path: str | Path) -> dict[str, Any]:
    with np.load(pose_path, allow_pickle=False) as payload:
        raw = payload.get("measurement_metadata_json")
        if raw is not None:
            try:
                value = json.loads(str(scalar(raw)))
                if isinstance(value, dict):
                    return value
            except (TypeError, json.JSONDecodeError):
                pass
    manifest = Path(pose_path).parent / "run_manifest.json"
    if manifest.exists():
        value = read_json(manifest).get("record", {}).get("primary_key", {})
        return value if isinstance(value, dict) else {}
    return {}


def pose_key(pose_path: str | Path) -> tuple[str, str, str, int, str] | None:
    metadata = pose_metadata(pose_path)
    model = metadata.get("model")
    scene = metadata.get("scene_id")
    case = metadata.get("case_id")
    seed = metadata.get("seed")
    route = metadata.get("route_id")
    action = action_from_route(str(route)) if route is not None else None
    if None in (model, scene, case, seed) or action is None:
        return None
    return str(model), str(scene), str(case), int(seed), action


def index_pose_files(pose_root: str | Path) -> dict[tuple[str, str, str, int, str], Path]:
    """Index precomputed Task 1 pose artifacts by their atomic cell identity."""
    indexed: dict[tuple[str, str, str, int, str], Path] = {}
    for path in sorted(Path(pose_root).rglob("pose.npz")):
        key = pose_key(path)
        if key is not None and key not in indexed:
            indexed[key] = path
    return indexed


def _finite_matrix(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"expected finite array {shape}, got {array.shape}")
    return array


def _unit_boundaries(record: Mapping[str, Any]) -> list[int]:
    timing = record.get("timing", {})
    values = timing.get("unit_boundary_frames")
    if values is None:
        values = record.get("route", {}).get("unit_boundary_frames")
    if not isinstance(values, list) or len(values) != 17:
        raise ValueError("Task 2 requires exactly 17 unit boundary frames")
    result = [int(value) for value in values]
    if result[0] < 0 or any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError(f"invalid d16 unit boundary frames: {result}")
    return result


def _validate_atomic_record(record: Mapping[str, Any], action: str) -> None:
    route = record.get("route", {})
    if route.get("total_units") not in (None, 16):
        raise ValueError("Task 2 atomic record must contain dose_max=16")
    expanded = route.get("expanded_unit_actions")
    if expanded is not None:
        normalized = [action_name(str(value)) for value in expanded]
        if normalized != [action] * 16:
            raise ValueError("atomic route is not sixteen repetitions of its declared action")
    timing = record.get("timing", {})
    if timing.get("unscored_tail_frames") is not None and int(timing["unscored_tail_frames"]) < 0:
        raise ValueError("unscored tail frame count must be non-negative")


def _checkpoint_template(
    frame_index: int | None,
    measurable: bool | None,
    transform: np.ndarray | None = None,
    leakage: Mapping[str, float] | None = None,
    twist: np.ndarray | None = None,
    u_signed: float | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "frame_index": frame_index,
        "measurable": measurable,
        "t_x": None,
        "t_z": None,
        "theta_y_deg": None,
        "t_y": None,
        "pitch_deg": None,
        "roll_deg": None,
        "u_signed": u_signed,
    }
    if transform is not None and twist is not None and leakage is not None:
        output.update({
            "t_x": float(transform[0, 3]),
            "t_z": float(transform[2, 3]),
            "theta_y_deg": float(np.degrees(twist[2])),
            "t_y": float(leakage["ty_cm"]),
            "pitch_deg": float(leakage["pitch_deg"]),
            "roll_deg": float(leakage["roll_deg"]),
        })
    return output


def _projected_action_scalar(transform: np.ndarray, twist: np.ndarray, action: str) -> float:
    """Return the §0.7 scalar from the projected relative transform."""
    name = action_name(action)
    if name == "W":
        return float(transform[2, 3])
    if name == "S":
        return float(-transform[2, 3])
    if name == "A":
        return float(-transform[0, 3])
    if name == "D":
        return float(transform[0, 3])
    if name == "YL":
        return float(-np.degrees(twist[2]))
    if name == "YR":
        return float(np.degrees(twist[2]))
    raise KeyError(f"unknown action: {action}")


def _fit_power_law(values: list[float], doses: list[int]) -> dict[str, Any]:
    usable = [
        (float(dose), float(value))
        for dose, value in zip(doses, values)
        if np.isfinite(value) and abs(float(value)) > EPSILON
    ]
    if len(usable) < 2:
        return {"gamma": None, "r2": None, "n_points": len(usable), "intercept": None}
    x = np.log(np.asarray([item[0] for item in usable], dtype=np.float64))
    y = np.log(np.abs(np.asarray([item[1] for item in usable], dtype=np.float64)))
    design = np.c_[x, np.ones(len(x))]
    gamma, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ np.array([gamma, intercept])
    residual = y - predicted
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if total <= EPSILON and np.all(np.abs(residual) <= 1e-10) else (
        float(1.0 - np.sum(residual**2) / total) if total > EPSILON else 0.0
    )
    return {
        "gamma": float(gamma),
        "r2": float(r2),
        "n_points": len(usable),
        "intercept": float(intercept),
    }


def _mad(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def _percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, q)) if len(array) else None


def _empty_power(reason: str | None = None) -> dict[str, Any]:
    return {
        "status": "FIT_UNAVAILABLE", "dose_horizon": max(POWER_LAW_DOSES),
        "gamma": None, "r2": None, "n_points": 0,
        "residual_by_k": {}, "residual_max_k": None,
        "failure_reasons": [reason] if reason else [],
    }


def _empty_unit(action: str, doses: tuple[int, ...], reason: str | None = None) -> dict[str, Any]:
    return {
        "status": "UNIT_NOT_IDENTIFIABLE", "dose_horizon": max(doses),
        "U_a_se2": None, "xi_bar": None, "u_a": None,
        "u_a_unit": "deg_per_dose" if action in ROTATION_ACTIONS else "cm_per_dose",
        "non_degenerate": False, "non_degenerate_reason": reason,
        "failure_reasons": [reason] if reason else [],
    }


def _estimate_unit(
    twists: Mapping[int, np.ndarray],
    signed_values: Mapping[int, float],
    ambiguous_doses: set[int],
    action: str,
    doses: tuple[int, ...],
) -> dict[str, Any]:
    missing = [dose for dose in doses if dose not in twists]
    ambiguous = sorted(ambiguous_doses.intersection(doses))
    if missing:
        return _empty_unit(action, doses, "unmeasurable_checkpoints:" + ",".join(map(str, missing)))
    if ambiguous:
        return _empty_unit(action, doses, "rotation_log_branch_ambiguous:" + ",".join(map(str, ambiguous)))
    xi_bar = np.mean(np.asarray([twists[dose] / dose for dose in doses]), axis=0)
    unit_scalar = action_scalar(xi_bar, action)
    if action in ROTATION_ACTIONS:
        unit_scalar = math.degrees(unit_scalar)
    unit_one_scalar = float(signed_values[1])
    floor = ROTATION_FLOOR_DEG if action in ROTATION_ACTIONS else TRANSLATION_FLOOR_CM
    non_degenerate = abs(unit_one_scalar) >= floor
    payload = _empty_unit(action, doses)
    payload.update({
        "status": "UNIT_IDENTIFIED",
        "U_a_se2": exp_se2(xi_bar).tolist(),
        "xi_bar": {"rho_x": float(xi_bar[0]), "rho_z": float(xi_bar[1]), "theta_rad": float(xi_bar[2])},
        "u_a": float(unit_scalar), "u_a_at_k1": unit_one_scalar,
        "non_degenerate": non_degenerate,
        "non_degenerate_reason": None if non_degenerate else "below_resolution_floor",
    })
    return payload


def _empty_leakage() -> dict[str, Any]:
    return {
        "acc_sign_correct": None, "purity": None,
        "yaw_crosstalk_deg": None, "turn_translation_cm": None,
        "t_y_cm": None, "pitch_roll_deg": None,
    }


def _find_track_and_anchor(case_dir: Path) -> tuple[Path | None, Path | None]:
    track = next((case_dir / name for name in ("tracks.npz", "track.npz") if (case_dir / name).exists()), None)
    anchor = next((case_dir / name for name in (
        "gt/anchors.npz", "gt/anchors_candidate.npz", "anchors.npz", "anchors_candidate.npz"
    ) if (case_dir / name).exists()), None)
    return track, anchor


def rigidity_from_pose(
    pose: Mapping[str, np.ndarray],
    case_dir: Path,
    measurable_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute Task 1 rigidity using the per-point valid mask when available."""
    ray = np.asarray(pose.get("world_ray_error_rad"), dtype=np.float64)
    measurable = np.asarray(
        pose.get("measurable") if measurable_mask is None else measurable_mask,
        dtype=bool,
    )
    if ray.ndim != 2 or len(measurable) != len(ray):
        return {"mean_deg": None, "std_deg": None, "n": 0, "source": "unavailable"}
    stored_world_mask = np.asarray(pose.get("world_scored_mask", np.empty(0)), dtype=bool)
    valid_points: np.ndarray | None = stored_world_mask if stored_world_mask.shape == ray.shape else None
    track_path, anchor_path = _find_track_and_anchor(case_dir)
    if valid_points is None and track_path is not None and anchor_path is not None:
        try:
            tracks = load_track_npz(track_path)
            anchors = load_anchor_npz(anchor_path)
            valid_points = world_measurement_mask(pose, tracks, anchors)
        except (KeyError, ValueError, IndexError):
            valid_points = None
    if valid_points is None or valid_points.shape != ray.shape:
        return {"mean_deg": None, "std_deg": None, "n": 0, "source": "world_validity_unavailable"}
    values: list[float] = []
    for frame in range(1, len(ray)):
        if not measurable[frame]:
            continue
        mask = np.isfinite(ray[frame])
        mask &= valid_points[frame]
        count = int(mask.sum())
        if count >= MIN_WORLD_POINTS:
            values.extend(np.degrees(ray[frame, mask]).tolist())
    if not values:
        return {"mean_deg": None, "std_deg": None, "n": 0, "source": "pose_world_validity"}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_deg": float(np.mean(array)),
        "std_deg": float(np.std(array)),
        "n": int(len(array)),
        "source": "pose_world_validity",
    }


def missing_cell(
    row: Mapping[str, Any], action: str, reason: str,
) -> dict[str, Any]:
    key = row["primary_key"]
    state_id = state_id_for_key(key)
    return {
        "model": str(key.get("model")), "state_id": state_id,
        "scene_id": str(key.get("scene_id")), "case_id": str(key.get("case_id")),
        "first_frame_sha256": None, "seed": int(key.get("seed")),
        "action": action, "route_id": str(key.get("route_id")),
        "status": "UNIT_NOT_IDENTIFIABLE", "failure_reasons": [reason],
        "checkpoints": {
            str(k): _checkpoint_template(None, None) for k in DOSES
        },
        "unit": _empty_unit(action, UNIT_DOSES, reason),
        "task3_unit": _empty_unit(action, TASK3_OPERATOR_DOSES, reason),
        "power_law": _empty_power(reason), "leakage": _empty_leakage(),
        "rigidity": {
            "e_rig_deg": None, "e_rig_over_u": None,
            "e_rig_over_u_unit": "dimensionless" if action in ROTATION_ACTIONS else "deg_per_cm",
        },
        "pose_path": None,
    }


def evaluate_cell(
    pose_path: str | Path,
    record_path: str | Path | Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    min_pose_inlier_ratio: float = DEFAULT_MIN_POSE_INLIER_RATIO,
    min_pose_anchor_coverage: float = DEFAULT_MIN_POSE_ANCHOR_COVERAGE,
) -> dict[str, Any]:
    """Evaluate one atomic sixteen-unit Task 2 video."""
    key = row["primary_key"]
    action = action_from_route(str(key.get("route_id", "")))
    if action is None:
        return missing_cell(row, "UNKNOWN", "unsupported_action_route")
    record = dict(record_path) if isinstance(record_path, Mapping) else read_json(record_path)
    _validate_atomic_record(record, action)
    boundaries = _unit_boundaries(record)
    with np.load(pose_path, allow_pickle=False) as payload:
        pose = {name: np.asarray(payload[name]) for name in payload.files}
    matrices = _finite_matrix(pose["camera_to_world_cv_rh"], (len(pose["camera_to_world_cv_rh"]), 4, 4))
    stored_measurable = np.asarray(pose.get("measurable", np.zeros(len(matrices))), dtype=bool)
    camera_valid = np.asarray(pose.get("camera_valid", []), dtype=bool)
    coverage = np.asarray(pose.get("pose_anchor_coverage", []), dtype=np.float64)
    ratio = np.asarray(pose.get("pose_inlier_ratio", []), dtype=np.float64)
    if all(name in pose for name in ("camera_valid", "pose_anchor_coverage", "pose_inlier_ratio")) and len(camera_valid) == len(matrices) and len(coverage) == len(matrices) and len(ratio) == len(matrices):
        measurable = camera_valid & (coverage >= float(min_pose_anchor_coverage)) & (ratio >= float(min_pose_inlier_ratio))
    else:
        measurable = stored_measurable
    if len(measurable) != len(matrices):
        raise ValueError("pose measurable length differs from trajectory")
    if boundaries[-1] >= len(matrices):
        raise ValueError(f"unit boundary {boundaries[-1]} exceeds pose length {len(matrices)}")
    endpoint_finite = np.isfinite(matrices[boundaries]).all(axis=(1, 2))
    initial_usable = bool(measurable[boundaries[0]]) and bool(endpoint_finite[0])
    start = matrices[boundaries[0]]
    checkpoint_payload: dict[str, Any] = {}
    transforms: dict[int, np.ndarray] = {}
    twists: dict[int, np.ndarray] = {}
    leakages: dict[int, dict[str, float]] = {}
    signed_values: dict[int, float] = {}
    ambiguous_doses: set[int] = set()
    for dose, index in enumerate(boundaries[1:], 1):
        checkpoint_measurable = bool(measurable[index]) and bool(endpoint_finite[dose])
        if not initial_usable or not checkpoint_measurable:
            checkpoint_payload[str(dose)] = _checkpoint_template(index, checkpoint_measurable)
            continue
        relative = np.linalg.inv(start) @ matrices[index]
        planar, leakage = project_se3_to_se2(relative)
        twist = log_se2(planar)
        if abs(float(np.degrees(twist[2]))) >= 170.0:
            ambiguous_doses.add(dose)
        signed = _projected_action_scalar(relative, twist, action)
        transforms[dose] = planar
        twists[dose] = twist
        leakages[dose] = leakage
        signed_values[dose] = float(signed)
        checkpoint_payload[str(dose)] = _checkpoint_template(
            index, True, relative, leakage, twist, float(signed)
        )
    if initial_usable:
        unit_payload = _estimate_unit(twists, signed_values, ambiguous_doses, action, UNIT_DOSES)
        task3_unit = _estimate_unit(twists, signed_values, ambiguous_doses, action, TASK3_OPERATOR_DOSES)
    else:
        unit_payload = _empty_unit(action, UNIT_DOSES, "initial_pose_unmeasurable")
        task3_unit = _empty_unit(action, TASK3_OPERATOR_DOSES, "initial_pose_unmeasurable")
    identifiable = unit_payload["status"] == "UNIT_IDENTIFIED"
    reasons = list(unit_payload["failure_reasons"])
    if identifiable and not unit_payload["non_degenerate"]:
        reasons.append("below_resolution_floor")
    power = _empty_power()
    leakage_payload = _empty_leakage()
    if identifiable:
        first_transform = transforms[1]
        first_twist = twists[1]
        first_leakage = leakages[1]
        norm_translation = float(np.linalg.norm(first_transform[:2, 2]))
        purity = None
        if action in TRANSLATION_ACTIONS and norm_translation > EPSILON:
            purity = min(1.0, max(0.0, abs(float(signed_values[1])) / norm_translation))
        leakage_payload = {
            "acc_sign_correct": bool(signed_values[1] > 0.0),
            "purity": purity,
            "yaw_crosstalk_deg": abs(float(np.degrees(first_twist[2]))) if action in TRANSLATION_ACTIONS else None,
            "turn_translation_cm": norm_translation if action in ROTATION_ACTIONS else None,
            "t_y_cm": abs(float(first_leakage["ty_cm"])),
            "pitch_roll_deg": float(math.hypot(first_leakage["pitch_deg"], first_leakage["roll_deg"])),
        }
    power_missing = [dose for dose in POWER_LAW_DOSES if dose not in transforms]
    power_ambiguous = sorted(ambiguous_doses.intersection(POWER_LAW_DOSES))
    if not initial_usable:
        power = _empty_power("initial_pose_unmeasurable")
    elif power_missing:
        power = _empty_power("unmeasurable_checkpoints:" + ",".join(map(str, power_missing)))
    elif power_ambiguous:
        power = _empty_power("rotation_log_branch_ambiguous:" + ",".join(map(str, power_ambiguous)))
    else:
        fit = _fit_power_law([signed_values[dose] for dose in POWER_LAW_DOSES], list(POWER_LAW_DOSES))
        unit_operator = np.asarray(unit_payload["U_a_se2"], dtype=float)
        residuals = {}
        for dose in POWER_LAW_DOSES:
            distances = distance_se2(se2_power(unit_operator, dose), transforms[dose])
            numerator = distances["yaw_deg"] if action in ROTATION_ACTIONS else distances["translation_cm"]
            residuals[str(dose)] = float(numerator / max(abs(signed_values[dose]), EPSILON))
        power = {
            "status": "FIT_AVAILABLE", "dose_horizon": max(POWER_LAW_DOSES),
            "gamma": fit["gamma"], "r2": fit["r2"], "n_points": fit["n_points"],
            "residual_by_k": residuals, "residual_max_k": residuals[str(max(POWER_LAW_DOSES))],
            "failure_reasons": [],
        }
    rigidity = rigidity_from_pose(pose, Path(pose_path).parent, measurable)
    e_rig = rigidity["mean_deg"]
    unit_scalar = unit_payload["u_a"]
    rigidity_payload = {
        "e_rig_deg": e_rig,
        "e_rig_over_u": (float(e_rig / abs(float(unit_scalar))) if e_rig is not None and unit_scalar is not None and abs(float(unit_scalar)) > EPSILON else None),
        "e_rig_over_u_unit": "dimensionless" if action in ROTATION_ACTIONS else "deg_per_cm",
        "n": rigidity["n"], "source": rigidity["source"],
    }
    first_frame_sha = None
    track_path, _ = _find_track_and_anchor(Path(pose_path).parent)
    if track_path is not None:
        try:
            with np.load(track_path, allow_pickle=False) as tracks:
                if "condition_image_sha256" in tracks:
                    first_frame_sha = str(scalar(tracks["condition_image_sha256"]))
        except (OSError, ValueError):
            pass
    metadata = pose_metadata(pose_path)
    return {
        "model": str(key.get("model", metadata.get("model", "unknown"))),
        "state_id": state_id_for_key(key),
        "scene_id": str(key.get("scene_id")), "case_id": str(key.get("case_id")),
        "first_frame_sha256": first_frame_sha,
        "seed": int(key.get("seed")), "action": action,
        "route_id": str(key.get("route_id")),
        "status": "UNIT_IDENTIFIED" if identifiable else "UNIT_NOT_IDENTIFIABLE",
        "failure_reasons": reasons,
        "checkpoints": checkpoint_payload,
        "unit": unit_payload, "task3_unit": task3_unit, "power_law": power,
        "leakage": leakage_payload, "rigidity": rigidity_payload,
        "pose_path": str(pose_path), "record_path": str(record_path) if not isinstance(record_path, Mapping) else "embedded_eval_index_record",
    }


def _nested_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for token in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(token)
    return current


def _usable_cell_values(cells: Iterable[Mapping[str, Any]], field: str, *, non_degenerate: bool = True) -> list[float]:
    values: list[float] = []
    for cell in cells:
        if cell.get("status") != "UNIT_IDENTIFIED":
            continue
        if non_degenerate and not bool(cell.get("unit", {}).get("non_degenerate")):
            continue
        value = _nested_value(cell, field)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return values


def _cell_positive(cell: Mapping[str, Any]) -> bool | None:
    value = cell.get("checkpoints", {}).get("1", {}).get("u_signed")
    return bool(float(value) > 0) if value is not None and np.isfinite(float(value)) else None


def aggregate_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate cell results without averaging SE(2) matrices across cells."""
    by_model_action: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_state_action: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_model_action[(str(cell["model"]), str(cell["action"]))].append(cell)
        by_state_action[(str(cell["model"]), str(cell["state_id"]), str(cell["action"]))].append(cell)
    model_action: dict[str, dict[str, Any]] = defaultdict(dict)
    for (model, action), group in sorted(by_model_action.items()):
        direction_flags = [_cell_positive(cell) for cell in group]
        positive = sum(flag is True for flag in direction_flags)
        negative = sum(flag is False for flag in direction_flags)
        direction_supported = positive + negative
        identified_flags = [
            flag for cell, flag in zip(group, direction_flags)
            if cell.get("status") == "UNIT_IDENTIFIED" and flag is not None
        ]
        usable = [cell for cell in group if cell.get("status") == "UNIT_IDENTIFIED"]
        task3_usable = [
            cell for cell in group
            if cell.get("task3_unit", {}).get("status") == "UNIT_IDENTIFIED"
            and cell.get("task3_unit", {}).get("non_degenerate")
        ]
        power_usable = [
            cell for cell in group
            if cell.get("power_law", {}).get("status") == "FIT_AVAILABLE"
            and cell.get("unit", {}).get("non_degenerate")
        ]
        values = _usable_cell_values(group, "unit.u_a")
        gamma = _usable_cell_values(group, "power_law.gamma")
        r2 = _usable_cell_values(group, "power_law.r2")
        r16 = _usable_cell_values(group, "power_law.residual_max_k")
        e_over = _usable_cell_values(group, "rigidity.e_rig_over_u")
        purity = _usable_cell_values(group, "leakage.purity")
        direction_leakage = [1.0 - value for value in purity]
        yaw_leak = _usable_cell_values(group, "leakage.yaw_crosstalk_deg", non_degenerate=False)
        ty_leak = _usable_cell_values(group, "leakage.t_y_cm", non_degenerate=False)
        pr_leak = _usable_cell_values(group, "leakage.pitch_roll_deg", non_degenerate=False)
        turn_translation = _usable_cell_values(group, "leakage.turn_translation_cm", non_degenerate=False)
        checkpoint_coverage = {
            str(k): float(sum(bool(cell.get("checkpoints", {}).get(str(k), {}).get("measurable")) for cell in group) / max(len(group), 1))
            for k in DOSES
        }
        model_action[model][action] = {
            "acc": float(positive / max(len(group), 1)),
            # ``acc`` deliberately follows the specification's fixed denominator
            # (all cells, including cells whose pose is unavailable).  Keep the
            # decomposition beside it so a low value is not mistaken for a
            # conditional direction-reversal rate.
            "direction_positive_count": int(positive),
            "direction_negative_count": int(negative),
            "direction_unknown_count": int(len(group) - direction_supported),
            "direction_supported_count": int(direction_supported),
            "direction_accuracy_conditional": (
                float(positive / direction_supported) if direction_supported else None
            ),
            "direction_accuracy_identified": (
                float(sum(flag is True for flag in identified_flags) / len(identified_flags))
                if identified_flags else None
            ),
            "purity_median": float(np.median(purity)) if purity else None,
            "purity_p90": _percentile(purity, 90.0) if purity else None,
            "direction_leakage_median": float(np.median(direction_leakage)) if direction_leakage else None,
            "direction_leakage_p90": _percentile(direction_leakage, 90.0) if direction_leakage else None,
            "turn_translation_cm_median": float(np.median(turn_translation)) if action in ROTATION_ACTIONS and turn_translation else None,
            "turn_translation_cm_p90": _percentile(turn_translation, 90.0) if action in ROTATION_ACTIONS and turn_translation else None,
            "yaw_crosstalk_deg_median": float(np.median(yaw_leak)) if action in TRANSLATION_ACTIONS and yaw_leak else None,
            "yaw_crosstalk_deg_p90": _percentile(yaw_leak, 90.0) if action in TRANSLATION_ACTIONS and yaw_leak else None,
            "t_y_cm_median": float(np.median(ty_leak)) if ty_leak else None,
            "t_y_cm_p90": _percentile(ty_leak, 90.0) if ty_leak else None,
            "pitch_roll_deg_median": float(np.median(pr_leak)) if pr_leak else None,
            "pitch_roll_deg_p90": _percentile(pr_leak, 90.0) if pr_leak else None,
            "u_median": float(np.median(np.abs(values))) if values else None,
            "u_p90": _percentile(np.abs(values), 90.0) if values else None,
            "gamma_median": float(np.median(gamma)) if gamma else None,
            "gamma_p90": _percentile(gamma, 90.0) if gamma else None,
            "r2_median": float(np.median(r2)) if r2 else None,
            "r2_p90": _percentile(r2, 90.0) if r2 else None,
            "r16_median": float(np.median(r16)) if r16 else None,
            "r16_p90": _percentile(r16, 90.0) if r16 else None,
            "e_rig_over_u_median": float(np.median(e_over)) if e_over else None,
            "e_rig_over_u_p90": _percentile(e_over, 90.0) if e_over else None,
            "cell_coverage": float(len(usable) / max(len(group), 1)),
            "task3_operator_coverage": float(len(task3_usable) / max(len(group), 1)),
            "power_law_coverage": float(len(power_usable) / max(len(group), 1)),
            "checkpoint_coverage": checkpoint_coverage,
            "n_cells": len(group), "n_identified": len(usable),
            "n_task3_operator": len(task3_usable), "n_power_law": len(power_usable),
            "n_non_degenerate": len(values),
        }
    state_action: dict[str, Any] = {}
    for (model, state_id, action), group in sorted(by_state_action.items()):
        measurable = [cell for cell in group if cell.get("status") == "UNIT_IDENTIFIED" and cell.get("unit", {}).get("non_degenerate")]
        values = [float(cell["unit"]["u_a"]) for cell in measurable if cell["unit"].get("u_a") is not None]
        signs = [_cell_positive(cell) for cell in measurable]
        if len(values) >= 3:
            median = float(np.median(values)); mad = _mad(values)
            sigma = float(ROBUST_SCALE * mad) if mad is not None else None
            cv = float(sigma / (abs(median) + EPSILON)) if sigma is not None else None
            repeat = float(math.exp(-cv)) if cv is not None else None
            state_payload = {"u_median": median, "sigma_mad": sigma, "cv_r": cv, "s_repeat": repeat,
                             "sign_consistency": bool(all(sign is True for sign in signs) or all(sign is False for sign in signs))}
        else:
            state_payload = {"u_median": None, "sigma_mad": None, "cv_r": None, "s_repeat": None, "sign_consistency": None}
        if len(values) < 2:
            state_payload["sign_consistency"] = None
        state_payload.update({"n_seeds": len(values), "n_measurable": len(measurable)})
        state_action[f"{model}|{state_id}|{action}"] = state_payload
    cross_state: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (model, action), group in sorted(by_model_action.items()):
        by_seed: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for cell in group:
            if cell.get("status") == "UNIT_IDENTIFIED" and cell.get("unit", {}).get("non_degenerate"):
                by_seed[int(cell["seed"])].append((str(cell["state_id"]), float(cell["unit"]["u_a"])))
        for seed, values in sorted(by_seed.items()):
            if len(values) < 3:
                cross_state[model][f"{seed}|{action}"] = {"cv_x": None, "e_scale": {}, "n_states": len(values), "calibration_state_id": None}
                continue
            values = sorted(values, key=lambda item: item[0])
            state_values = [value for _, value in values]
            median = float(np.median(state_values)); mad = _mad(state_values)
            cv = float(ROBUST_SCALE * mad / (abs(median) + EPSILON)) if mad is not None else None
            cross_state[model][f"{seed}|{action}"] = {
                "cv_x": cv, "e_scale": {},
                "n_states": len(values), "calibration_state_id": None,
            }
    return {
        "model_action": dict(model_action),
        "state_action": state_action,
        "ablation": {"cross_state": {model: dict(values) for model, values in cross_state.items()}},
        "noop": {},
    }


def build_document(
    cells: list[dict[str, Any]],
    *,
    pose_root: str | Path,
    data_root: str | Path,
    model: str | None,
    seed: int | None,
    task: str,
    worker_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    aggregate = aggregate_cells(cells)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "constants": {
            "actions": list(ACTIONS), "dose_max": 16, "checkpoint_doses": list(DOSES),
            "unit_estimation_doses": list(UNIT_DOSES),
            "task3_operator_doses": list(TASK3_OPERATOR_DOSES),
            "power_law_doses": list(POWER_LAW_DOSES),
            "robust_scale": ROBUST_SCALE, "epsilon": EPSILON,
            "translation_floor_cm": TRANSLATION_FLOOR_CM,
            "rotation_floor_deg": ROTATION_FLOOR_DEG,
            "min_world_points": MIN_WORLD_POINTS,
            "frozen_settings_sha256": FROZEN_METRIC_SETTINGS_SHA256,
            "min_pose_inlier_ratio": float((worker_profile or {}).get("min_pose_inlier_ratio", DEFAULT_MIN_POSE_INLIER_RATIO)),
            "min_pose_anchor_coverage": float((worker_profile or {}).get("min_pose_anchor_coverage", DEFAULT_MIN_POSE_ANCHOR_COVERAGE)),
            "tail_included_in_unit_pose": False,
        },
        "pose_root": str(pose_root), "data_root": str(data_root),
        "model": model, "seed": seed, "task": task,
        "n_cells": len(cells), "cells": cells,
        "state_action": aggregate["state_action"],
        "model_action": aggregate["model_action"],
        "ablation": aggregate["ablation"],
        "noop": aggregate["noop"],
        "diagnostic_warnings": [
            "no_noop_cells_supplied; task2.noop side table is unavailable for this input slice",
            "formal_action_record_provenance_audit_not_run_by_this_pose_artifact_consumer",
        ],
    }
    if worker_profile is not None:
        document["run_profile"] = dict(worker_profile)
    return document


def flatten_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    unit = cell.get("unit", {})
    task3_unit = cell.get("task3_unit", {})
    power = cell.get("power_law", {})
    leakage = cell.get("leakage", {})
    rigidity = cell.get("rigidity", {})
    return {
        "model": cell.get("model"), "state_id": cell.get("state_id"),
        "scene_id": cell.get("scene_id"), "case_id": cell.get("case_id"),
        "seed": cell.get("seed"), "action": cell.get("action"),
        "status": cell.get("status"),
        "failure_reasons": ";".join(cell.get("failure_reasons", [])),
        "u_a": unit.get("u_a"), "u_a_unit": unit.get("u_a_unit"),
        "non_degenerate": unit.get("non_degenerate"),
        "task3_unit_status": task3_unit.get("status"),
        "task3_u_a": task3_unit.get("u_a"),
        "task3_non_degenerate": task3_unit.get("non_degenerate"),
        "power_law_status": power.get("status"),
        "gamma": power.get("gamma"), "r2": power.get("r2"),
        "r16": power.get("residual_max_k"),
        "acc_sign_correct": leakage.get("acc_sign_correct"),
        "purity": leakage.get("purity"),
        "yaw_crosstalk_deg": leakage.get("yaw_crosstalk_deg"),
        "turn_translation_cm": leakage.get("turn_translation_cm"),
        "e_rig_deg": rigidity.get("e_rig_deg"),
        "e_rig_over_u": rigidity.get("e_rig_over_u"),
        "pose_path": cell.get("pose_path"),
    }


def _markdown_number(value: Any, digits: int = 4, unit: str = "") -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"{numeric:.{digits}f}{suffix}"


def render_markdown_report(
    document: Mapping[str, Any],
    *,
    invocation: str | None = None,
) -> str:
    """Render the complete Task 2 summary as a Markdown report."""
    profile = document.get("run_profile", {})
    cells = list(document.get("cells", []))
    n_cells = int(document.get("n_cells", len(cells)))
    n_identified = sum(cell.get("status") == "UNIT_IDENTIFIED" for cell in cells)
    n_task3_operator = sum(
        cell.get("task3_unit", {}).get("status") == "UNIT_IDENTIFIED"
        and cell.get("task3_unit", {}).get("non_degenerate") for cell in cells
    )
    n_power_law = sum(
        cell.get("power_law", {}).get("status") == "FIT_AVAILABLE"
        and cell.get("unit", {}).get("non_degenerate") for cell in cells
    )
    pose_artifacts = int(profile.get("pose_artifacts", sum(cell.get("pose_path") is not None for cell in cells)))
    worker_errors = profile.get("worker_errors")
    lines = [
        "# StrideWorld Task 2 Evaluation Report",
        "",
        "> This report is generated automatically by the Task 2 evaluation entry point. Primary metrics follow the final specification; conditional Accuracy and the leakage breakdown are diagnostic fields.",
        "",
        "## Run and Data",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| model | `{document.get('model')}` |",
        f"| seed | `{document.get('seed')}` |",
        f"| task | `{document.get('task')}` |",
        f"| data root | `{document.get('data_root')}` |",
        f"| pose root | `{document.get('pose_root')}` |",
        f"| Fixed-denominator cells | {n_cells} |",
        f"| pose artifacts | {pose_artifacts} |",
        f"| Task 2 unit-1 usable (k=1..12) | {n_identified} |",
        f"| UNIT_NOT_IDENTIFIABLE | {n_cells - n_identified} |",
        f"| Task 3 short-range operator usable (k=1..4) | {n_task3_operator} |",
        f"| Power-law fit usable (k=1..16) | {n_power_law} |",
        f"| pose inlier ratio threshold | {_markdown_number(document.get('constants', {}).get('min_pose_inlier_ratio'))} |",
        f"| pose anchor coverage threshold | {_markdown_number(document.get('constants', {}).get('min_pose_anchor_coverage'))} |",
    ]
    if worker_errors is not None:
        lines.append(f"| worker errors | {int(worker_errors)} |")
    if profile.get("gpus") is not None:
        lines.extend([
            f"| GPU allocation | `{profile.get('gpus')}` |",
            f"| Workers per GPU | {profile.get('workers_per_gpu')} |",
            f"| Total workers | {profile.get('total_workers')} |",
        ])
    if invocation:
        lines.extend(["", "To reproduce this invocation:", "", "```bash", invocation, "```"])
    lines.extend([
        "",
        "## Metric Definitions",
        "",
        "- `Acc (fixed denominator)`: number of cells with the correct direction at checkpoint 1, divided by all cells; unmeasurable cells stay in the denominator.",
        "- `Acc (measurable-conditional)`: direction accuracy computed only on cells with a finite pose and a signed displacement at checkpoint 1.",
        "- `Acc (complete cell)`: computed only on `UNIT_IDENTIFIED` cells where all of k=1..12 are measurable, as a diagnostic of the primary unit quantity.",
        "- Unit-1 is estimated from k=1..12; the Task 3 atomic operator is estimated separately from k=1..4; the power-law fit still strictly requires k=1..16.",
        "- Direction leakage of translation actions is `1 - Purity`; yaw crosstalk is reported separately. Direction leakage of rotation actions is expressed as the planar translation while turning.",
        "- Out-of-plane leakage is reported uniformly as `|t_y|` and `sqrt(pitch^2 + roll^2)`. All leakage quantities are taken at checkpoint 1.",
        "",
        "## Full Results",
    ])
    all_actions: list[Mapping[str, Any]] = []
    for model, action_map in document.get("model_action", {}).items():
        lines.extend(["", f"### Model `{model}`", "", "#### Primary Metrics", ""])
        lines.extend([
            "| action | cells | unit@12 | Task3 op@4 | power@16 | Acc fixed denominator | unit coverage | \\|u\\| median | \\|u\\| P90 | gamma median | R2 median | r16 median | E_rig/\\|u\\| median |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        ordered = [(action, action_map[action]) for action in ACTIONS if action in action_map]
        ordered.extend((action, values) for action, values in action_map.items() if action not in ACTIONS)
        for action, values in ordered:
            all_actions.append(values)
            unit = "deg/dose" if action in ROTATION_ACTIONS else "cm/dose"
            rigidity_unit = "deg/deg" if action in ROTATION_ACTIONS else "deg/cm"
            lines.append(
                f"| {action} | {values.get('n_cells', 0)} | {values.get('n_identified', 0)} | "
                f"{values.get('n_task3_operator', 0)} | {values.get('n_power_law', 0)} | "
                f"{_markdown_number(values.get('acc'))} | {_markdown_number(values.get('cell_coverage'))} | "
                f"{_markdown_number(values.get('u_median'), unit=unit)} | {_markdown_number(values.get('u_p90'), unit=unit)} | "
                f"{_markdown_number(values.get('gamma_median'))} | {_markdown_number(values.get('r2_median'))} | "
                f"{_markdown_number(values.get('r16_median'))} | "
                f"{_markdown_number(values.get('e_rig_over_u_median'), unit=rigidity_unit)} |"
            )
        lines.extend(["", "#### Accuracy Breakdown", ""])
        lines.extend([
            "| action | Acc fixed denominator (primary) | Acc measurable-conditional (auxiliary) | Acc complete cell (diagnostic) | positive | negative | unknown | checkpoint 1 coverage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for action, values in ordered:
            checkpoint_one = values.get("checkpoint_coverage", {}).get("1")
            lines.append(
                f"| {action} | {_markdown_number(values.get('acc'))} | "
                f"{_markdown_number(values.get('direction_accuracy_conditional'))} | "
                f"{_markdown_number(values.get('direction_accuracy_identified'))} | "
                f"{values.get('direction_positive_count', 0)} | {values.get('direction_negative_count', 0)} | "
                f"{values.get('direction_unknown_count', 0)} | {_markdown_number(checkpoint_one)} |"
            )
        lines.extend(["", "#### Direction and Out-of-Plane Leakage", ""])
        lines.extend([
            "| action | Purity median | direction leakage median | direction leakage P90 | yaw crosstalk median/P90 | turn translation median/P90 | \\|t_y\\| median/P90 | pitch-roll median/P90 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for action, values in ordered:
            lines.append(
                f"| {action} | {_markdown_number(values.get('purity_median'))} | "
                f"{_markdown_number(values.get('direction_leakage_median'))} | "
                f"{_markdown_number(values.get('direction_leakage_p90'))} | "
                f"{_markdown_number(values.get('yaw_crosstalk_deg_median'), unit='deg')} / "
                f"{_markdown_number(values.get('yaw_crosstalk_deg_p90'), unit='deg')} | "
                f"{_markdown_number(values.get('turn_translation_cm_median'), unit='cm')} / "
                f"{_markdown_number(values.get('turn_translation_cm_p90'), unit='cm')} | "
                f"{_markdown_number(values.get('t_y_cm_median'), unit='cm')} / "
                f"{_markdown_number(values.get('t_y_cm_p90'), unit='cm')} | "
                f"{_markdown_number(values.get('pitch_roll_deg_median'), unit='deg')} / "
                f"{_markdown_number(values.get('pitch_roll_deg_p90'), unit='deg')} |"
            )
        lines.extend(["", "#### Checkpoint Coverage", ""])
        lines.append("| checkpoint | " + " | ".join(action for action, _ in ordered) + " |")
        lines.append("|---:|" + "---:|" * len(ordered))
        for dose in DOSES:
            values = [
                _markdown_number(summary.get("checkpoint_coverage", {}).get(str(dose)))
                for _, summary in ordered
            ]
            lines.append(f"| {dose} | " + " | ".join(values) + " |")
    lines.extend(["", "## Automatic Diagnostics", ""])
    if n_cells:
        lines.append(f"- pose artifact coverage is `{pose_artifacts}/{n_cells} = {pose_artifacts / n_cells:.4f}`; missing artifacts lower the fixed-denominator Acc.")
        lines.append(f"- Task 2 unit@12 coverage is `{n_identified}/{n_cells} = {n_identified / n_cells:.4f}`; Task 3 op@4 is `{n_task3_operator}/{n_cells}`; strict power@16 is `{n_power_law}/{n_cells}`.")
    conditional = [float(item["direction_accuracy_conditional"]) for item in all_actions if item.get("direction_accuracy_conditional") is not None]
    purity_values = [float(item["purity_median"]) for item in all_actions if item.get("purity_median") is not None]
    if conditional:
        lines.append(f"- Measurable-conditional Accuracy across actions ranges over `{min(conditional):.4f}–{max(conditional):.4f}`.")
    if purity_values:
        lines.append(f"- Median Purity of translation actions ranges over `{min(purity_values):.4f}–{max(purity_values):.4f}`.")
    warnings = list(document.get("diagnostic_warnings", []))
    if warnings:
        lines.extend(["", "## Diagnostic Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in warnings)
    return "\n".join(lines) + "\n"


def write_outputs(
    document: Mapping[str, Any],
    output_json: str | Path,
    output_csv: str | Path,
    model_action_csv: str | Path | None = None,
    report_markdown: str | Path | None = None,
    *,
    invocation: str | None = None,
) -> None:
    output_json = Path(output_json); output_csv = Path(output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False))
    rows = [flatten_cell(cell) for cell in document.get("cells", [])]
    columns = sorted({key for row in rows for key in row}) if rows else ["model", "state_id", "action", "status"]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    if model_action_csv is not None:
        path = Path(model_action_csv); path.parent.mkdir(parents=True, exist_ok=True)
        summary_rows: list[dict[str, Any]] = []
        for model, actions in document.get("model_action", {}).items():
            for action, values in actions.items():
                row = {"model": model, "action": action}
                row.update({key: value for key, value in values.items() if key != "checkpoint_coverage"})
                for dose, coverage in values.get("checkpoint_coverage", {}).items():
                    row[f"checkpoint_coverage_{dose}"] = coverage
                summary_rows.append(row)
        columns = sorted({key for row in summary_rows for key in row}) if summary_rows else ["model", "action"]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(summary_rows)
    if report_markdown is not None:
        path = Path(report_markdown); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown_report(document, invocation=invocation))


__all__ = [
    "ACTIONS", "DOSES", "SCHEMA_VERSION", "DEFAULT_MIN_POSE_INLIER_RATIO",
    "DEFAULT_MIN_POSE_ANCHOR_COVERAGE", "aggregate_cells", "action_from_route",
    "build_document", "evaluate_cell", "index_pose_files",
    "missing_cell", "pose_key", "ready_records", "record_path", "render_markdown_report",
    "state_id_for_key", "write_outputs",
]
