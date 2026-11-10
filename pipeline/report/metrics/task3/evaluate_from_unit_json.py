#!/usr/bin/env python3
"""Evaluate Task 3 routes with per-case Task 2 4-dose unit operators."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.task2.task2_metrics import ACTIONS, ROTATION_ACTIONS  # noqa: E402
from metrics.task3.task3_metrics import classify_relation, pair_rows  # noqa: E402
from metrics.tools.pose import is_rigid_transform  # noqa: E402
from metrics.tools.se2 import (  # noqa: E402
    distance_se2,
    inverse_se2,
    log_se2,
    project_se3_to_se2,
)


SCHEMA_VERSION = "strideworld-task3-case-unit4-v1"


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def _finite_nonzero(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and abs(float(value)) > 0.0
    except (TypeError, ValueError):
        return False


def _load_task3_rows(results_root: Path, model: str) -> list[dict[str, Any]]:
    index_path = results_root / "details" / "normalized_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    rows = []
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("primary_key", {})
        if key.get("task") == "task3" and str(key.get("model")) == model:
            rows.append(row)
    if not rows:
        raise ValueError(f"no Task 3 rows for model {model!r} in {index_path}")
    return rows


def _unit_index(document: Mapping[str, Any]) -> dict[tuple[str, str, int], dict[str, dict[str, Any]]]:
    output: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for scene_id, scene in document.get("scenes", {}).items():
        for case_id, case in scene.get("cases", {}).items():
            for seed, seed_value in case.get("seeds", {}).items():
                actions: dict[str, dict[str, Any]] = {}
                for action in ACTIONS:
                    value = seed_value.get("actions", {}).get(action, {})
                    horizon = value.get("unit_horizons", {}).get("4", {})
                    actions[action] = {
                        "status": horizon.get("status"),
                        "unit_one": horizon.get("unit_one"),
                        "unit": horizon.get("unit"),
                        "operator_se2": horizon.get("operator_se2"),
                        "record_id": value.get("video_locator", {}).get("record_id"),
                    }
                output[(str(scene_id), str(case_id), int(seed))] = actions
    return output


def _usable_units(values: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, np.ndarray], list[str]]:
    scalars: dict[str, float] = {}
    operators: dict[str, np.ndarray] = {}
    reasons: list[str] = []
    for action in ACTIONS:
        value = values.get(action, {})
        operator = np.asarray(value.get("operator_se2"), dtype=float)
        if value.get("status") != "UNIT_IDENTIFIED" or not _finite_nonzero(value.get("unit_one")):
            reasons.append(f"unit4_unavailable:{action}")
        elif operator.shape != (3, 3) or not np.isfinite(operator).all():
            reasons.append(f"unit4_operator_invalid:{action}")
        else:
            scalars[action] = abs(float(value["unit_one"]))
            operators[action] = operator
    return scalars, operators, reasons


def _route_data(row: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    record = row.get("_record", {})
    route = record.get("route", {})
    actions = [str(action).upper() for action in route.get("expanded_unit_actions", [])]
    frames = [int(frame) for frame in record.get("timing", {}).get("unit_boundary_frames", [])]
    return actions, frames


def _first_frame_gate(track_path: Path, expected_sha: Any) -> tuple[dict[str, Any], list[str]]:
    if not track_path.is_file():
        return {"pass": False, "track_path": str(track_path)}, ["missing_tracks_artifact"]
    try:
        with np.load(track_path, allow_pickle=False) as payload:
            match_pass = bool(_scalar(payload["first_frame_match_pass"])) if "first_frame_match_pass" in payload else False
            first_sha = str(_scalar(payload["first_frame_sha256"])) if "first_frame_sha256" in payload else None
            condition_sha = str(_scalar(payload["condition_image_sha256"])) if "condition_image_sha256" in payload else None
            first_l1 = float(_scalar(payload["first_frame_l1"])) if "first_frame_l1" in payload else None
    except (OSError, ValueError, KeyError) as exc:
        return {"pass": False, "track_path": str(track_path)}, [f"tracks_load_error:{type(exc).__name__}"]
    expected = str(expected_sha) if expected_sha is not None else None
    hash_match = bool(first_sha and condition_sha and first_sha == condition_sha)
    if expected:
        hash_match = hash_match and condition_sha == expected
    reasons = []
    if not match_pass:
        reasons.append("first_frame_match_failed")
    if not hash_match:
        reasons.append("first_frame_binding_mismatch")
    return {
        "pass": not reasons,
        "track_path": str(track_path),
        "first_frame_l1": first_l1,
        "first_frame_match_pass": match_pass,
        "first_frame_sha256": first_sha,
        "condition_image_sha256": condition_sha,
        "expected_condition_sha256": expected,
        "hash_binding_pass": hash_match,
    }, reasons


def _error_metric(observed: np.ndarray, predicted: np.ndarray,
                  units: Mapping[str, float]) -> dict[str, Any]:
    raw = distance_se2(predicted, observed)
    residual = inverse_se2(predicted) @ observed
    twist = log_se2(residual)
    rho_x, rho_z, theta = map(float, twist)
    if abs(rho_x) >= abs(rho_z):
        translation_action = "D" if rho_x >= 0.0 else "A"
    else:
        translation_action = "W" if rho_z >= 0.0 else "S"
    rotation_action = "YR" if theta >= 0.0 else "YL"
    return {
        "translation_error_cm": raw["translation_cm"],
        "translation_error_ratio": float(raw["translation_cm"] / units[translation_action]),
        "translation_normalizer_action": translation_action,
        "translation_unit_one_cm": units[translation_action],
        "angular_error_deg": raw["yaw_deg"],
        "angular_error_ratio": float(raw["yaw_deg"] / units[rotation_action]),
        "angular_normalizer_action": rotation_action,
        "angular_unit_one_deg": units[rotation_action],
    }


def _evaluate_video(row: Mapping[str, Any], results_root: Path,
                    unit_values: Mapping[str, Mapping[str, Any]], *,
                    track_ratio_threshold: float,
                    motion_ratio_threshold: float) -> tuple[dict[str, Any], dict[str, list[np.ndarray]]]:
    key = row["primary_key"]
    record_id = str(row["record_id"])
    artifact_root = results_root / "artifacts" / record_id
    pose_path, track_path = artifact_root / "pose.npz", artifact_root / "tracks.npz"
    actions, frames = _route_data(row)
    units, operators, reasons = _usable_units(unit_values)
    first_frame, first_reasons = _first_frame_gate(track_path, row.get("condition_sha256"))
    reasons.extend(first_reasons)
    output: dict[str, Any] = {
        "record_id": record_id,
        "route_id": str(key.get("route_id")),
        "video_path": row.get("video_path"),
        "video_sha256": row.get("video_sha256"),
        "pose_path": str(pose_path),
        "actions": actions,
        "frame_boundaries": frames,
        "unit_one_by_action": {
            action: {
                "value": unit_values.get(action, {}).get("unit_one"),
                "unit": unit_values.get(action, {}).get("unit"),
                "task2_record_id": unit_values.get(action, {}).get("record_id"),
            }
            for action in ACTIONS
        },
        "first_frame_gate": first_frame,
        "boundary_quality": [],
        "motion_gate": [],
        "per_action_errors": [],
        "quality_pass_without_motion_gate": False,
    }
    internal: dict[str, list[np.ndarray]] = {"observed": [], "predicted": []}
    if not pose_path.is_file():
        reasons.append("missing_pose_artifact")
    if not actions or len(frames) != len(actions) + 1:
        reasons.append("unit_boundary_action_count_mismatch")
    if any(action not in ACTIONS for action in actions):
        reasons.append("unsupported_action")
    if reasons:
        output.update({"quality_pass": False, "failure_reasons": sorted(set(reasons))})
        return output, internal

    try:
        with np.load(pose_path, allow_pickle=False) as payload:
            matrices = np.asarray(payload["camera_to_world_cv_rh"], dtype=float)
            camera_valid = np.asarray(payload["camera_valid"], dtype=bool)
            coverage = np.asarray(payload["pose_anchor_coverage"], dtype=float)
            inlier = np.asarray(payload["pose_inlier_ratio"], dtype=float)
    except (OSError, ValueError, KeyError) as exc:
        output.update({"quality_pass": False, "failure_reasons": [f"pose_load_error:{type(exc).__name__}"]})
        return output, internal

    n = len(matrices)
    if any(len(values) != n for values in (camera_valid, coverage, inlier)):
        reasons.append("pose_quality_length_mismatch")
    if min(frames) < 0 or max(frames) >= n:
        reasons.append("unit_boundary_out_of_range")
    if reasons:
        output.update({"quality_pass": False, "failure_reasons": sorted(set(reasons))})
        return output, internal

    for frame in frames:
        finite = bool(np.isfinite(matrices[frame]).all())
        rigid = bool(finite and is_rigid_transform(matrices[frame]))
        track_ratio = float(min(coverage[frame], inlier[frame]))
        passed = bool(camera_valid[frame] and rigid and track_ratio >= track_ratio_threshold)
        output["boundary_quality"].append({
            "frame_index": frame,
            "pass": passed,
            "camera_valid": bool(camera_valid[frame]),
            "pose_anchor_coverage": float(coverage[frame]),
            "pose_inlier_ratio": float(inlier[frame]),
            "trackable_pose_point_ratio": track_ratio,
            "finite_rigid_transform": rigid,
        })
        if not passed:
            reasons.append(f"boundary_quality_failed:{frame}")

    if not all(item["finite_rigid_transform"] for item in output["boundary_quality"]):
        output.update({"quality_pass": False, "failure_reasons": sorted(set(reasons))})
        return output, internal

    start_inv = np.linalg.inv(matrices[frames[0]])
    previous = np.eye(3, dtype=float)
    predicted = np.eye(3, dtype=float)
    output["quality_pass_without_motion_gate"] = not reasons
    for step, (action, frame) in enumerate(zip(actions, frames[1:]), 1):
        observed, _ = project_se3_to_se2(start_inv @ matrices[frame])
        movement = distance_se2(previous, observed)
        value = movement["yaw_deg"] if action in ROTATION_ACTIONS else movement["translation_cm"]
        threshold = units[action] * motion_ratio_threshold
        passed = bool(value >= threshold)
        output["motion_gate"].append({
            "step": step,
            "action": action,
            "frame_index": frame,
            "movement_type": "rotation" if action in ROTATION_ACTIONS else "translation",
            "movement_value": float(value),
            "movement_unit": "deg" if action in ROTATION_ACTIONS else "cm",
            "unit_one": units[action],
            "required_ratio": motion_ratio_threshold,
            "threshold": float(threshold),
            "pass": passed,
        })
        if not passed:
            reasons.append(f"motion_below_unit_fraction:{step}:{action}")
        predicted = predicted @ operators[action]
        internal["observed"].append(observed)
        internal["predicted"].append(predicted.copy())
        previous = observed

    quality_pass = not reasons
    if quality_pass:
        output["per_action_errors"] = [
            {
                "step": step,
                "action": action,
                "frame_index": frame,
                **_error_metric(observed, predicted, units),
            }
            for step, (action, frame, observed, predicted) in enumerate(
                zip(actions, frames[1:], internal["observed"], internal["predicted"]), 1
            )
        ]
    output.update({"quality_pass": quality_pass, "failure_reasons": sorted(set(reasons))})
    return output, internal


def _case_macro_mean(entries: list[tuple[tuple[str, str], float]]) -> tuple[float | None, int]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for case, value in entries:
        grouped[case].append(value)
    if not grouped:
        return None, 0
    return float(np.mean([np.mean(values) for values in grouped.values()])), len(grouped)


def _summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in ("inverse", "order"):
        selected = [pair for pair in pairs if pair["relation_family"] == family]
        videos = [pair[member] for pair in selected for member in ("alpha", "beta")]
        valid_videos = [video for video in videos if video["quality_pass"]]
        valid_videos_without_motion = [video for video in videos if video["quality_pass_without_motion_gate"]]
        valid_pairs = [pair for pair in selected if pair["quality_pass"]]
        max_step = max((len(video["actions"]) for video in videos), default=0)
        per_step = []
        for step in range(1, max_step + 1):
            video_points = []
            for pair in selected:
                case = (pair["scene_id"], pair["case_id"])
                for member in ("alpha", "beta"):
                    errors = pair[member]["per_action_errors"]
                    if len(errors) >= step:
                        video_points.append((case, errors[step - 1]))
            pair_points = []
            for pair in valid_pairs:
                if len(pair["relation_errors_by_step"]) >= step:
                    pair_points.append(((pair["scene_id"], pair["case_id"]), pair["relation_errors_by_step"][step - 1]))
            video_t, video_cases = _case_macro_mean([(case, point["translation_error_ratio"]) for case, point in video_points])
            video_r, _ = _case_macro_mean([(case, point["angular_error_ratio"]) for case, point in video_points])
            pair_t, pair_cases = _case_macro_mean([(case, point["translation_error_ratio"]) for case, point in pair_points])
            pair_r, _ = _case_macro_mean([(case, point["angular_error_ratio"]) for case, point in pair_points])
            per_step.append({
                "step": step,
                "n_video_errors": len(video_points),
                "n_video_cases": video_cases,
                "mean_video_translation_error_ratio": video_t,
                "mean_video_angular_error_ratio": video_r,
                "n_pair_errors": len(pair_points),
                "n_pair_cases": pair_cases,
                "mean_pair_translation_error_ratio": pair_t,
                "mean_pair_angular_error_ratio": pair_r,
            })
        result[family] = {
            "total_videos": len(videos),
            "quality_passed_videos": len(valid_videos),
            "quality_passed_videos_without_motion_gate": len(valid_videos_without_motion),
            "total_pairs": len(selected),
            "quality_passed_pairs": len(valid_pairs),
            "video_failure_reasons": dict(Counter(reason for video in videos for reason in video["failure_reasons"])),
            "per_step": per_step,
        }
    result["all"] = {
        "total_videos": sum(value["total_videos"] for value in result.values()),
        "quality_passed_videos": sum(value["quality_passed_videos"] for value in result.values()),
        "quality_passed_videos_without_motion_gate": sum(value["quality_passed_videos_without_motion_gate"] for value in result.values()),
        "total_pairs": sum(value["total_pairs"] for value in result.values()),
        "quality_passed_pairs": sum(value["quality_passed_pairs"] for value in result.values()),
    }
    return result


def _write_csv(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "relation_family", "total_videos", "quality_passed_videos", "quality_passed_videos_without_motion_gate",
        "total_pairs", "quality_passed_pairs",
        "step", "n_video_errors", "n_video_cases", "mean_video_translation_error_ratio",
        "mean_video_angular_error_ratio", "n_pair_errors", "n_pair_cases",
        "mean_pair_translation_error_ratio", "mean_pair_angular_error_ratio",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family in ("inverse", "order"):
            value = summary[family]
            for step in value["per_step"]:
                writer.writerow({"relation_family": family, **{key: value[key] for key in fields[1:6]}, **step})


def _self_check() -> None:
    units = {"W": 2.0, "S": 4.0, "A": 5.0, "D": 10.0, "YL": 20.0, "YR": 40.0}
    observed = np.eye(3)
    observed[0, 2] = -5.0
    angle = math.radians(-10.0)
    observed[:2, :2] = [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]]
    value = _error_metric(observed, np.eye(3), units)
    assert value["translation_normalizer_action"] == "A"
    assert value["angular_normalizer_action"] == "YL"
    assert math.isclose(value["translation_error_ratio"], 1.0)
    assert math.isclose(value["angular_error_ratio"], 0.5)
    print("self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task2-json", type=Path)
    parser.add_argument("--results-root", type=Path, help="Defaults to source.results_root in the Task 2 JSON")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--track-ratio-threshold", type=float, default=0.1)
    parser.add_argument("--motion-ratio-threshold", type=float, default=0.1)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if args.task2_json is None or args.output_json is None:
        parser.error("--task2-json and --output-json are required")
    if not 0.0 <= args.track_ratio_threshold <= 1.0 or not 0.0 <= args.motion_ratio_threshold <= 1.0:
        parser.error("quality thresholds must be in [0, 1]")

    task2 = json.loads(args.task2_json.read_text())
    model = str(task2["model"])
    results_root = (args.results_root or Path(task2["source"]["results_root"])).resolve()
    unit_index = _unit_index(task2)
    pairs = []
    for row_a, row_b in pair_rows(_load_task3_rows(results_root, model)):
        key = row_a["primary_key"]
        identity = (str(key["scene_id"]), str(key["case_id"]), int(key["seed"]))
        unit_values = unit_index.get(identity, {})
        alpha, alpha_internal = _evaluate_video(
            row_a, results_root, unit_values,
            track_ratio_threshold=args.track_ratio_threshold,
            motion_ratio_threshold=args.motion_ratio_threshold,
        )
        beta, beta_internal = _evaluate_video(
            row_b, results_root, unit_values,
            track_ratio_threshold=args.track_ratio_threshold,
            motion_ratio_threshold=args.motion_ratio_threshold,
        )
        units, _, _ = _usable_units(unit_values)
        pair_pass = bool(alpha["quality_pass"] and beta["quality_pass"])
        relation_errors = []
        if pair_pass:
            for step, (obs_a, pred_a, obs_b, pred_b) in enumerate(zip(
                alpha_internal["observed"], alpha_internal["predicted"],
                beta_internal["observed"], beta_internal["predicted"],
            ), 1):
                observed_relation = inverse_se2(obs_a) @ obs_b
                predicted_relation = inverse_se2(pred_a) @ pred_b
                relation_errors.append({
                    "step": step,
                    "alpha_action": alpha["actions"][step - 1],
                    "beta_action": beta["actions"][step - 1],
                    **_error_metric(observed_relation, predicted_relation, units),
                })
        route_a = row_a.get("_record", {}).get("route", {})
        pairs.append({
            "model": model,
            "scene_id": identity[0],
            "case_id": identity[1],
            "seed": identity[2],
            "pair_id": str(row_a.get("pair_id") or route_a.get("pair_id")),
            "route_family": str(route_a.get("family")),
            "relation_family": classify_relation(route_a, row_b.get("_record", {}).get("route", {})),
            "quality_pass": pair_pass,
            "failure_reasons": ([] if pair_pass else [
                *(f"alpha:{reason}" for reason in alpha["failure_reasons"]),
                *(f"beta:{reason}" for reason in beta["failure_reasons"]),
            ]),
            "alpha": alpha,
            "beta": beta,
            "relation_errors_by_step": relation_errors,
        })

    summary = _summary(pairs)
    output_csv = args.output_csv or args.output_json.with_suffix(".summary.csv")
    document = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "source": {
            "task2_unit_json": str(args.task2_json.resolve()),
            "results_root": str(results_root),
            "normalized_index": str(results_root / "details" / "normalized_index.jsonl"),
        },
        "definition": {
            "unit_horizon": 4,
            "unit_binding": "same model/scene_id/case_id/seed/action",
            "operator_composition": "right_multiply_in_execution_order",
            "quality_gate": {
                "first_frame": "cached first_frame_match_pass and condition hash binding",
                "boundary_policy": "operator start and every unit-action boundary",
                "trackable_pose_point_ratio": "min(pose_anchor_coverage, pose_inlier_ratio)",
                "trackable_pose_point_ratio_min": args.track_ratio_threshold,
                "per_action_motion_min": args.motion_ratio_threshold,
                "video_policy": "any failed gate invalidates the whole video",
            },
            "error_normalization": {
                "translation": "divide by the same-case unit_one of the dominant signed W/S/A/D residual direction",
                "rotation": "divide by the same-case unit_one of the signed YL/YR residual direction",
                "aggregation": "mean within each scene/case, then mean across cases",
            },
        },
        "summary": summary,
        "pairs": pairs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False))
    _write_csv(summary, output_csv)
    print("| relation | videos | passed without motion gate | quality passed | pairs | quality passed pairs |")
    print("|---|---:|---:|---:|---:|---:|")
    for family in ("inverse", "order"):
        value = summary[family]
        print(f"| {family} | {value['total_videos']} | {value['quality_passed_videos_without_motion_gate']} | {value['quality_passed_videos']} | {value['total_pairs']} | {value['quality_passed_pairs']} |")
    print(json.dumps({"json": str(args.output_json), "csv": str(output_csv)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
