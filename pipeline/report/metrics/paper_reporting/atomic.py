"""Read-only paper metrics from saved atomic poses and frozen K=4 or K=8 units.

Nothing here estimates an operator, runs tracking, or changes the original
measurement gate.  Distances are computed in the initial camera frame.  Each
requested checkpoint stays in the denominator, including when its pose or
the selected frozen operator is unavailable.  Direction uses the requested
action axis, independently of the learned operator.

The default zero motion floors are numerical guards only, not a calibrated
claim about physical measurement resolution.  Translation floors are cm and
rotation floors are degrees.  Pose quality thresholds are a separate policy.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metrics.task2.task2_metrics import (
    ACTIONS,
    _projected_action_scalar,
    _unit_boundaries,
    action_from_route,
)
from metrics.tools.se2 import (
    log_se2,
    project_se3_to_se2,
    se2_power,
    wrap_angle_rad,
)


EVALUATION_DOSES = (8, 12, 16)
UNIT_HORIZON = 4
TRANSLATION_ACTIONS = frozenset(("W", "S", "A", "D"))
DEFAULT_POSE_INLIER_RATIO = 0.1
DEFAULT_POSE_ANCHOR_COVERAGE = 0.1


def _unique(reasons: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(reason) for reason in reasons))


def _validate_floor(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _unit_matrix(unit: Any, horizon: int) -> tuple[np.ndarray | None, list[str], Any]:
    """Accept one selected-horizon export payload, or an explicit SE(2) matrix."""
    reasons: list[str] = []
    raw = unit
    if isinstance(unit, Mapping):
        reasons.extend(str(reason) for reason in unit.get("failure_reasons", []))
        if "dose_horizon" in unit and unit["dose_horizon"] != horizon:
            return None, _unique(reasons + [f"unit_horizon_must_equal_{horizon}"]), raw
        if unit.get("status", "UNIT_IDENTIFIED") != "UNIT_IDENTIFIED":
            return None, _unique(reasons + [f"unit_status:{unit.get('status')}"]), raw
        unit = unit.get("operator_se2", unit.get("U_a_se2"))
    if unit is None:
        return None, _unique(reasons + [f"missing_k{horizon}_unit"]), raw
    try:
        matrix = np.asarray(unit, dtype=np.float64)
        # The official logarithm validates the homogeneous row and rotation.
        log_se2(matrix)
    except (TypeError, ValueError, np.linalg.LinAlgError) as error:
        return None, _unique(reasons + [f"invalid_k{horizon}_unit:{type(error).__name__}:{error}"]), raw
    return matrix, _unique(reasons), raw


def _empty_point(dose: int, frame: int | None = None) -> dict[str, Any]:
    return {
        "dose": dose,
        "frame_index": frame,
        "measurable": False,
        "score": 0.0,
        "r": None,
        "direction_correct": False,
        "signed_action_scalar": None,
        "observed_se2": None,
        "predicted_se2": None,
        "observed_translation_cm": None,
        "predicted_translation_cm": None,
        "observed_yaw_deg": None,
        "predicted_yaw_deg": None,
        "signed_yaw_error_deg": None,
        "translation_error_cm": None,
        "normalization_scale": None,
        "failure_reasons": [],
    }


def _empty_leakage(reason: str) -> dict[str, Any]:
    return {
        "dose": 12,
        "ty_cm": None,
        "pitch_deg": None,
        "roll_deg": None,
        "relative_vertical": None,
        "vertical_scale_cm": None,
        "vertical_failure_reasons": [reason],
        "pitch_leak": None,
        "pitch_scale_deg": None,
        "pitch_failure_reasons": [reason],
    }


def evaluate_atomic_pose(
    poses: np.ndarray,
    measurable: np.ndarray,
    boundaries: Sequence[int],
    action: str,
    unit: Any,
    *,
    calibration_horizon: int = UNIT_HORIZON,
    evaluation_doses: Sequence[int] = EVALUATION_DOSES,
    translation_floor: float = 0.0,
    rotation_floor: float = 0.0,
) -> dict[str, Any]:
    """Evaluate extrapolation after an explicitly selected calibration horizon.

    ``unit`` is the report's matching ``unit_horizons`` payload or a 3x3 matrix.
    The default is K4 evaluated at 8,12,16.  For a fair K4/K8 comparison, pass
    ``evaluation_doses=(12,16)`` to both.  Every evaluation dose must be strictly
    after calibration; calibration observations cannot be scored as held out.
    Translation error uses both planar components, divided by
    ``k * norm(t(U))``; yaw error is wrapped and divided by
    ``k * abs(yaw(U))``.  ``score=100/(1+r)`` is applied to each point before
    averaging.  Direction is the official signed requested-axis scalar strictly
    greater than the chosen floor, and never uses the operator's direction.

    Invalid data is retained with score zero and explicit reasons.  Unsupported
    actions/floors are caller errors.  The official SE(3) projection requires
    SciPy; dependency errors propagate rather than changing metric definitions.
    """
    if action not in ACTIONS:
        raise ValueError(f"unsupported atomic action: {action}")
    translation_floor = _validate_floor(translation_floor, "translation_floor")
    rotation_floor = _validate_floor(rotation_floor, "rotation_floor")
    floor = translation_floor if action in TRANSLATION_ACTIONS else rotation_floor
    if (isinstance(calibration_horizon, bool)
            or not isinstance(calibration_horizon, (int, np.integer))
            or calibration_horizon not in (4, 8)):
        raise ValueError("calibration_horizon must be 4 or 8")
    doses = tuple(evaluation_doses)
    if (not doses or any(isinstance(dose, bool) or not isinstance(dose, (int, np.integer)) for dose in doses)
            or len(set(doses)) != len(doses) or tuple(sorted(doses)) != doses
            or any(dose <= calibration_horizon or dose > 16 for dose in doses)):
        raise ValueError("evaluation_doses must be unique increasing integers strictly after calibration_horizon and <=16")
    operator, unit_reasons, raw_unit = _unit_matrix(unit, calibration_horizon)
    unit_translation: float | None = None
    unit_yaw: float | None = None
    if operator is not None:
        unit_translation = float(np.linalg.norm(operator[:2, 2]))
        unit_yaw = abs(math.degrees(float(log_se2(operator)[2])))

    failures: list[str] = list(unit_reasons)
    points = {str(dose): _empty_point(dose) for dose in doses}
    leakage = _empty_leakage("k12_pose_unavailable" if 12 in doses else "k12_not_evaluated")
    data_error: str | None = None
    try:
        poses = np.asarray(poses, dtype=np.float64)
        measurable = np.asarray(measurable, dtype=bool)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("camera_to_world_cv_rh must have shape [T,4,4]")
        if measurable.shape != (len(poses),):
            raise ValueError("measurable mask length differs from trajectory")
        frames = list(boundaries)
        if len(frames) != 17 or any(isinstance(frame, bool) or int(frame) != frame for frame in frames):
            raise ValueError("atomic boundaries must contain 17 integer frames")
        frames = [int(frame) for frame in frames]
        if frames[0] < 0 or any(right <= left for left, right in zip(frames, frames[1:])):
            raise ValueError("atomic boundaries must be non-negative and strictly increasing")
        # Individual out-of-range evaluation frames are missing points, rather
        # than a reason to discard still-observable earlier checkpoints.
        for dose in doses:
            points[str(dose)]["frame_index"] = frames[dose]
        start = frames[0]
        if start >= len(poses) or not measurable[start] or not np.isfinite(poses[start]).all():
            data_error = "initial_pose_unmeasurable"
        else:
            initial_inverse = np.linalg.inv(poses[start])
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as error:
        data_error = f"invalid_pose_input:{type(error).__name__}:{error}"

    if data_error is not None:
        failures.append(data_error)
        for point in points.values():
            point["failure_reasons"] = _unique([data_error] + unit_reasons)
    else:
        for dose in doses:
            point = points[str(dose)]
            frame = frames[dose]
            point["failure_reasons"].extend(unit_reasons)
            if frame >= len(poses):
                point["failure_reasons"].append("checkpoint_frame_out_of_range")
                continue
            if not measurable[frame] or not np.isfinite(poses[frame]).all():
                point["failure_reasons"].append("checkpoint_pose_unmeasurable")
                continue
            relative = initial_inverse @ poses[frame]
            try:
                planar, observed_leakage = project_se3_to_se2(relative)
                twist = log_se2(planar)
                signed = float(_projected_action_scalar(relative, twist, action))
            except (ValueError, np.linalg.LinAlgError) as error:
                point["failure_reasons"].append(f"invalid_checkpoint_pose:{type(error).__name__}:{error}")
                continue
            point.update({
                "measurable": True,
                "observed_se2": planar.tolist(),
                "observed_translation_cm": planar[:2, 2].tolist(),
                "observed_yaw_deg": float(math.degrees(twist[2])),
                "signed_action_scalar": signed if math.isfinite(signed) else None,
                "direction_correct": bool(math.isfinite(signed) and signed > floor),
            })
            if not math.isfinite(signed):
                point["failure_reasons"].append("nonfinite_signed_action_scalar")
            if dose == 12:
                leakage = {
                    "dose": 12, **observed_leakage,
                    "relative_vertical": None,
                    "vertical_scale_cm": None if unit_translation is None else 12.0 * unit_translation,
                    "vertical_failure_reasons": [],
                    "pitch_leak": None,
                    "pitch_scale_deg": None if unit_yaw is None else 12.0 * unit_yaw,
                    "pitch_failure_reasons": [],
                }
                if unit_translation is not None and unit_translation > translation_floor:
                    leakage["relative_vertical"] = abs(float(observed_leakage["ty_cm"])) / (12.0 * unit_translation)
                else:
                    leakage["vertical_failure_reasons"] = [f"k{calibration_horizon}_translation_scale_unavailable_or_at_floor"]
                if unit_yaw is not None and unit_yaw > rotation_floor:
                    leakage["pitch_leak"] = abs(float(observed_leakage["pitch_deg"])) / (12.0 * unit_yaw)
                else:
                    leakage["pitch_failure_reasons"] = [f"k{calibration_horizon}_yaw_scale_unavailable_or_at_floor"]
            if operator is None:
                continue
            predicted = se2_power(operator, dose)
            predicted_yaw = float(log_se2(predicted)[2])
            yaw_error = math.degrees(wrap_angle_rad(float(twist[2]) - predicted_yaw))
            translation_error = float(np.linalg.norm(planar[:2, 2] - predicted[:2, 2]))
            unit_scale = unit_translation if action in TRANSLATION_ACTIONS else unit_yaw
            point.update({
                "predicted_se2": predicted.tolist(),
                "predicted_translation_cm": predicted[:2, 2].tolist(),
                "predicted_yaw_deg": float(math.degrees(predicted_yaw)),
                "signed_yaw_error_deg": float(yaw_error),
                "translation_error_cm": translation_error,
                "normalization_scale": dose * unit_scale,
            })
            if not math.isfinite(unit_scale) or unit_scale <= floor:
                point["failure_reasons"].append(f"k{calibration_horizon}_action_scale_unavailable_or_at_floor")
                continue
            error = translation_error if action in TRANSLATION_ACTIONS else abs(yaw_error)
            ratio = error / (dose * unit_scale)
            if not math.isfinite(ratio):
                point["failure_reasons"].append("nonfinite_normalized_error")
                continue
            point["r"] = float(ratio)
            point["score"] = 100.0 / (1.0 + ratio)

    for dose, point in points.items():
        point["failure_reasons"] = _unique(point["failure_reasons"])
        failures.extend(f"k{dose}:{reason}" for reason in point["failure_reasons"])
    return {
        "action": action,
        "unit_horizon": calibration_horizon,
        "calibration_horizon": calibration_horizon,
        "evaluation_doses": list(doses),
        "points": points,
        "atomic_score": float(sum(point["score"] for point in points.values()) / len(doses)),
        "direction_rate": float(sum(point["direction_correct"] for point in points.values()) / len(doses)),
        "n_points": len(doses),
        "n_scored": sum(point["r"] is not None for point in points.values()),
        "n_measurable": sum(point["measurable"] for point in points.values()),
        "failure_reasons": _unique(failures),
        "k12_relative_vertical": leakage["relative_vertical"],
        "k12_pitch_leak": leakage["pitch_leak"],
        "k12_diagnostics": leakage,
        "unit_source": raw_unit.tolist() if isinstance(raw_unit, np.ndarray) else raw_unit,
        "motion_floor_policy": {
            "translation_floor_cm": translation_floor,
            "rotation_floor_deg": rotation_floor,
            "zero_floor_is_numerical_guard_only": True,
            "physical_resolution_calibrated": False,
        },
    }


def load_unit_report(path: str | Path) -> dict[str, Any]:
    """Load the existing ``strideworld-task2-unit-horizons-v1`` JSON report."""
    report = json.loads(Path(path).read_text())
    if not isinstance(report, dict):
        raise ValueError("unit report must be a JSON object")
    return report


def _check_quality_policy(
    report: Mapping[str, Any], ratio: float, coverage: float,
) -> list[str]:
    definition = report.get("definition", {})
    reasons: list[str] = []
    for key, expected in (("min_pose_inlier_ratio", ratio), ("min_pose_anchor_coverage", coverage)):
        try:
            actual = float(definition[key])
            same = math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        except (KeyError, TypeError, ValueError):
            same = False
        if not same:
            reasons.append(f"unit_report_pose_policy_mismatch:{key}:expected={expected}:actual={definition.get(key)}")
    return reasons


def _quality_mask(pose: Mapping[str, np.ndarray], ratio: float, coverage: float) -> np.ndarray:
    count = len(pose["camera_to_world_cv_rh"])
    camera = np.asarray(pose.get("camera_valid", []), dtype=bool)
    inlier = np.asarray(pose.get("pose_inlier_ratio", []), dtype=float)
    anchor = np.asarray(pose.get("pose_anchor_coverage", []), dtype=float)
    if not camera.shape == inlier.shape == anchor.shape == (count,):
        raise ValueError("missing_or_malformed_pose_quality_arrays; stored_measurable_is_not_a_policy_substitute")
    return camera & np.isfinite(inlier) & np.isfinite(anchor) & (inlier >= ratio) & (anchor >= coverage)


def _sha256(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) else None


def evaluate_atomic_row(
    row: Mapping[str, Any],
    results_root: str | Path,
    unit_report: Mapping[str, Any] | str | Path,
    *,
    calibration_horizon: int = UNIT_HORIZON,
    evaluation_doses: Sequence[int] = EVALUATION_DOSES,
    translation_floor: float = 0.0,
    rotation_floor: float = 0.0,
    min_pose_inlier_ratio: float = DEFAULT_POSE_INLIER_RATIO,
    min_pose_anchor_coverage: float = DEFAULT_POSE_ANCHOR_COVERAGE,
) -> dict[str, Any]:
    """Read one normalized-index row and its official saved pose artifact.

    Uses ``results_root/artifacts/<record_id>/pose.npz`` and matches the selected
    entry by scene/case/seed/action.  Pose gates must match the unit report's
    definition, and are recomputed from saved camera/inlier/coverage arrays.
    The historic stored ``measurable`` mask is deliberately not substituted.
    All artifact and generation failures remain explicit in the returned row.
    """
    ratio, coverage = float(min_pose_inlier_ratio), float(min_pose_anchor_coverage)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (ratio, coverage)):
        raise ValueError("pose quality thresholds must be finite and in [0,1]")
    if not isinstance(unit_report, Mapping):
        unit_report = load_unit_report(unit_report)
    key = row.get("primary_key", {})
    route = str(key.get("route_id", row.get("route_id", "")))
    action = action_from_route(route)
    if action not in ACTIONS:
        raise ValueError(f"unsupported atomic action: {route}")
    record_id = str(row.get("record_id", ""))
    pose_path = Path(results_root) / "artifacts" / record_id / "pose.npz"
    source_failures = [str(reason) for reason in row.get("failure_reasons", [])]
    errors = _check_quality_policy(unit_report, ratio, coverage)
    row_hash = _sha256(row.get("video_sha256"))
    if row_hash is None:
        errors.append("normalized_row_video_sha256_missing_or_invalid")
    unit_hash: str | None = None
    pose_hash: str | None = None
    pose_metadata: dict[str, Any] = {}
    pose_hash_verified = False
    raw_unit: Any = None
    try:
        entry = unit_report["scenes"][str(key["scene_id"])]["cases"][str(key["case_id"])]["seeds"][str(key["seed"])]["actions"][action]
        raw_unit = entry["unit_horizons"][str(calibration_horizon)]
        locator = entry.get("video_locator", {})
        source_record = locator.get("record_id")
        if not record_id or source_record is None or str(source_record) != record_id:
            errors.append(f"unit_report_record_id_mismatch:{source_record}:{record_id}")
        unit_hash = _sha256(locator.get("video_sha256"))
        if unit_hash is None:
            errors.append("unit_report_video_sha256_missing_or_invalid")
        elif row_hash is not None and unit_hash != row_hash:
            errors.append(f"unit_report_video_sha256_mismatch:{unit_hash}:{row_hash}")
        source_model = unit_report.get("model")
        row_model = key.get("model", row.get("model"))
        if source_model is None or row_model is None or str(source_model) != str(row_model):
            errors.append(f"unit_report_model_mismatch:{source_model}")
    except (KeyError, TypeError) as error:
        errors.append(f"missing_k{calibration_horizon}_unit_report_cell:{type(error).__name__}:{error}")
    unit = raw_unit if not errors else {
        "status": "UNIT_NOT_IDENTIFIABLE", "dose_horizon": calibration_horizon,
        "failure_reasons": errors, "operator_se2": None,
    }
    pose: dict[str, np.ndarray] = {}
    mask = np.zeros(0, dtype=bool)
    boundaries: Sequence[int] = []
    load_errors: list[str] = []
    if row_hash is None:
        load_errors.append("normalized_row_video_sha256_missing_or_invalid")
    if row.get("status") != "GENERATION_READY":
        load_errors.append(f"generation_status:{row.get('status')}")
    elif not pose_path.is_file():
        load_errors.append("missing_pose_artifact")
    else:
        try:
            with np.load(pose_path, allow_pickle=False) as archive:
                pose = {name: np.asarray(archive[name]) for name in (
                    "camera_to_world_cv_rh", "camera_valid", "pose_anchor_coverage", "pose_inlier_ratio",
                ) if name in archive}
                if "measurement_metadata_json" in archive:
                    # This exact field is written by save_pose_npz; full
                    # evaluator metadata binds it to the source video hash.
                    pose_metadata = json.loads(str(np.asarray(archive["measurement_metadata_json"]).item()))
                    if not isinstance(pose_metadata, dict):
                        raise ValueError("measurement_metadata_json must be a JSON object")
                    if "video_sha256" in pose_metadata:
                        pose_hash = _sha256(pose_metadata["video_sha256"])
                        if pose_hash is None:
                            load_errors.append("pose_video_sha256_missing_or_invalid")
                        elif row_hash is None or pose_hash != row_hash:
                            load_errors.append(f"pose_video_sha256_mismatch:{pose_hash}:{row_hash}")
                        else:
                            pose_hash_verified = True
                    for field, expected in (("record_id", record_id), ("model", key.get("model", row.get("model")))):
                        if field in pose_metadata and str(pose_metadata[field]) != str(expected):
                            load_errors.append(f"pose_{field}_mismatch:{pose_metadata[field]}:{expected}")
            mask = _quality_mask(pose, ratio, coverage)
            record = row.get("_record")
            if not isinstance(record, Mapping):
                record = {"timing": {"unit_boundary_frames": row.get("unit_boundary_frames")}}
            boundaries = _unit_boundaries(record)
        except (OSError, KeyError, TypeError, ValueError) as error:
            load_errors.append(f"pose_artifact_error:{type(error).__name__}:{error}")
    if load_errors:
        pose, mask, boundaries = {}, np.zeros(0, dtype=bool), []
    result = evaluate_atomic_pose(
        pose.get("camera_to_world_cv_rh", np.zeros((0, 4, 4))), mask, boundaries,
        action, unit, calibration_horizon=calibration_horizon, evaluation_doses=evaluation_doses,
        translation_floor=translation_floor, rotation_floor=rotation_floor,
    )
    result.update({
        "record_id": record_id,
        "primary_key": dict(key),
        "generation_status": row.get("status"),
        "pose_path": str(pose_path),
        "source_failure_reasons": source_failures,
        "unit_source": raw_unit,
        "provenance": {
            "normalized_row_video_sha256": row_hash,
            "unit_report_video_sha256": unit_hash,
            "pose_video_sha256": pose_hash,
            "pose_video_sha256_verified": pose_hash_verified,
            "pose_video_sha256_present": "video_sha256" in pose_metadata,
            "pose_metadata_source": "measurement_metadata_json",
        },
        "pose_quality_policy": {
            "min_pose_inlier_ratio": ratio,
            "min_pose_anchor_coverage": coverage,
            "source": "saved_camera_valid_pose_inlier_ratio_pose_anchor_coverage",
            "unit_report_policy_matches": not _check_quality_policy(unit_report, ratio, coverage),
        },
    })
    result["failure_reasons"] = _unique(source_failures + errors + load_errors + result["failure_reasons"])
    return result


__all__ = [
    "EVALUATION_DOSES", "UNIT_HORIZON", "evaluate_atomic_pose", "evaluate_atomic_row",
    "load_unit_report",
]
