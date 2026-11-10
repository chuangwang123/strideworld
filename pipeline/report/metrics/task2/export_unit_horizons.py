#!/usr/bin/env python3
"""Export per-video Task-2 unit operators at 4, 8, and 16 doses."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.task2.task2_metrics import (  # noqa: E402
    ACTIONS,
    _estimate_unit,
    _projected_action_scalar,
    _unit_boundaries,
    action_from_route,
)
from metrics.tools.constants import (  # noqa: E402
    FROZEN_METRIC_SETTINGS,
    FROZEN_METRIC_SETTINGS_SHA256,
)
from metrics.tools.se2 import log_se2, project_se3_to_se2  # noqa: E402


HORIZONS = (4, 8, 16)
READY = "GENERATION_READY"


def _unavailable(action: str, horizon: int, reason: str) -> dict[str, Any]:
    return {
        "status": "UNIT_NOT_IDENTIFIABLE",
        "unit_one": None,
        "unit": "deg_per_dose" if action in {"YL", "YR"} else "cm_per_dose",
        "operator_se2": None,
        "xi_bar": None,
        "u_at_dose_1": None,
        "non_degenerate": False,
        "failure_reasons": [reason],
        "dose_horizon": horizon,
    }


def _format_unit(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": payload["status"],
        "unit_one": payload["u_a"],
        "unit": payload["u_a_unit"],
        "operator_se2": payload["U_a_se2"],
        "xi_bar": payload["xi_bar"],
        "u_at_dose_1": payload.get("u_a_at_k1"),
        "non_degenerate": payload["non_degenerate"],
        "failure_reasons": payload["failure_reasons"],
        "dose_horizon": payload["dose_horizon"],
    }


def unit_horizons(
    matrices: np.ndarray,
    measurable: np.ndarray,
    boundaries: list[int],
    action: str,
) -> dict[str, dict[str, Any]]:
    """Apply the existing Task-2 log-average definition at three horizons."""
    matrices = np.asarray(matrices, dtype=np.float64)
    measurable = np.asarray(measurable, dtype=bool)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("camera_to_world_cv_rh must have shape [T,4,4]")
    if measurable.shape != (len(matrices),):
        raise ValueError("measurable mask length differs from trajectory")
    if len(boundaries) != 17 or boundaries[-1] >= len(matrices):
        raise ValueError("Task-2 boundaries must contain 17 in-range frames")

    start_index = boundaries[0]
    start_usable = bool(measurable[start_index]) and bool(np.isfinite(matrices[start_index]).all())
    if not start_usable:
        return {str(h): _unavailable(action, h, "initial_pose_unmeasurable") for h in HORIZONS}

    start_inverse = np.linalg.inv(matrices[start_index])
    twists: dict[int, np.ndarray] = {}
    signed: dict[int, float] = {}
    ambiguous: set[int] = set()
    for dose, frame in enumerate(boundaries[1:], 1):
        if not measurable[frame] or not np.isfinite(matrices[frame]).all():
            continue
        relative = start_inverse @ matrices[frame]
        planar, _ = project_se3_to_se2(relative)
        twist = log_se2(planar)
        if abs(float(np.degrees(twist[2]))) >= 170.0:
            ambiguous.add(dose)
        twists[dose] = twist
        signed[dose] = _projected_action_scalar(relative, twist, action)

    return {
        str(horizon): _format_unit(
            _estimate_unit(
                twists,
                signed,
                ambiguous,
                action,
                tuple(range(1, horizon + 1)),
            )
        )
        for horizon in HORIZONS
    }


def _measurement_mask(
    pose: Mapping[str, np.ndarray], ratio_threshold: float, coverage_threshold: float,
) -> np.ndarray:
    n = len(pose["camera_to_world_cv_rh"])
    camera = np.asarray(pose.get("camera_valid", []), dtype=bool)
    coverage = np.asarray(pose.get("pose_anchor_coverage", []), dtype=float)
    ratio = np.asarray(pose.get("pose_inlier_ratio", []), dtype=float)
    if camera.shape == coverage.shape == ratio.shape == (n,):
        return camera & (coverage >= coverage_threshold) & (ratio >= ratio_threshold)
    stored = np.asarray(pose.get("measurable", np.zeros(n)), dtype=bool)
    return stored if stored.shape == (n,) else np.zeros(n, dtype=bool)


def _source_thresholds(results_root: Path) -> tuple[float, float, str]:
    task2_json = results_root / "task2" / "task2.json"
    if task2_json.is_file():
        constants = json.loads(task2_json.read_text()).get("constants", {})
        if "min_pose_inlier_ratio" in constants and "min_pose_anchor_coverage" in constants:
            return (
                float(constants["min_pose_inlier_ratio"]),
                float(constants["min_pose_anchor_coverage"]),
                str(task2_json),
            )
    return (
        float(FROZEN_METRIC_SETTINGS["min_pose_inlier_ratio"]),
        float(FROZEN_METRIC_SETTINGS["min_pose_anchor_coverage"]),
        "current_frozen_defaults",
    )


def _read_rows(index_path: Path, model: str | None) -> list[dict[str, Any]]:
    rows = []
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("primary_key", {})
        if key.get("task") != "task2" and not str(row.get("route_id", "")).startswith("atomic_"):
            continue
        if model is None or str(key.get("model", row.get("model"))) == model:
            rows.append(row)
    if not rows:
        raise ValueError("no Task-2 atomic rows selected")
    models = {str(row.get("primary_key", {}).get("model", row.get("model"))) for row in rows}
    if len(models) != 1:
        raise ValueError(f"expected one model, found {sorted(models)}; pass --model")
    return sorted(rows, key=lambda row: (
        str(row["primary_key"].get("scene_id")),
        str(row["primary_key"].get("case_id")),
        int(row["primary_key"].get("seed")),
        str(row["primary_key"].get("route_id")),
    ))


def build_report(
    results_root: Path,
    *,
    model: str | None = None,
    min_pose_inlier_ratio: float | None = None,
    min_pose_anchor_coverage: float | None = None,
) -> dict[str, Any]:
    index_path = results_root / "details" / "normalized_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    rows = _read_rows(index_path, model)
    inferred_ratio, inferred_coverage, threshold_source = _source_thresholds(results_root)
    ratio = inferred_ratio if min_pose_inlier_ratio is None else min_pose_inlier_ratio
    coverage = inferred_coverage if min_pose_anchor_coverage is None else min_pose_anchor_coverage
    if not 0.0 <= ratio <= 1.0 or not 0.0 <= coverage <= 1.0:
        raise ValueError("pose thresholds must be in [0,1]")
    if min_pose_inlier_ratio is not None or min_pose_anchor_coverage is not None:
        threshold_source = "command_line_override"

    model_name = str(rows[0]["primary_key"].get("model", rows[0].get("model")))
    scenes: dict[str, Any] = {}
    statuses: Counter[str] = Counter()
    available = {str(horizon): 0 for horizon in HORIZONS}
    for row in rows:
        key = row["primary_key"]
        scene_id, case_id, seed = str(key["scene_id"]), str(key["case_id"]), int(key["seed"])
        action = action_from_route(str(key.get("route_id", row.get("route_id", ""))))
        if action not in ACTIONS:
            raise ValueError(f"unsupported atomic action: {key.get('route_id')}")
        record_id = str(row["record_id"])
        pose_path = results_root / "artifacts" / record_id / "pose.npz"
        reason = None
        if row.get("status") != READY:
            reason = f"generation_status:{row.get('status')}"
        elif not pose_path.is_file():
            reason = "missing_pose_artifact"
        try:
            if reason is not None:
                units = {str(h): _unavailable(action, h, reason) for h in HORIZONS}
            else:
                with np.load(pose_path, allow_pickle=False) as payload:
                    pose = {
                        name: np.asarray(payload[name])
                        for name in (
                            "camera_to_world_cv_rh", "camera_valid", "pose_anchor_coverage",
                            "pose_inlier_ratio", "measurable",
                        )
                        if name in payload
                    }
                units = unit_horizons(
                    pose["camera_to_world_cv_rh"],
                    _measurement_mask(pose, ratio, coverage),
                    _unit_boundaries(row["_record"]),
                    action,
                )
        except Exception as error:
            reason = f"evaluation_error:{type(error).__name__}:{error}"
            units = {str(h): _unavailable(action, h, reason) for h in HORIZONS}

        status = "ok" if all(unit["status"] == "UNIT_IDENTIFIED" for unit in units.values()) else "partial_or_unavailable"
        statuses[status] += 1
        for horizon, unit in units.items():
            available[horizon] += int(unit["status"] == "UNIT_IDENTIFIED")
        video = {
            "record_id": record_id,
            "route_id": str(key.get("route_id")),
            "generation_status": row.get("status"),
            "video_path": row.get("video_path"),
            "video_sha256": row.get("video_sha256"),
            "pose_path": str(pose_path),
            "pose_realpath": str(pose_path.resolve()) if pose_path.exists() else None,
            "condition_sha256": row.get("condition_sha256"),
            "unit_boundary_frames": row.get("unit_boundary_frames", row.get("_record", {}).get("timing", {}).get("unit_boundary_frames")),
        }
        case = scenes.setdefault(scene_id, {"cases": {}})["cases"].setdefault(
            case_id, {"state_id": row.get("state_id"), "seed_ids": [], "seed_count": 0, "seeds": {}}
        )
        seed_entry = case["seeds"].setdefault(str(seed), {"actions": {}})
        if action in seed_entry["actions"]:
            raise ValueError(f"duplicate cell: {scene_id}/{case_id}/seed={seed}/{action}")
        seed_entry["actions"][action] = {"status": status, "video_locator": video, "unit_horizons": units}

    case_count = 0
    seed_ids: set[int] = set()
    for scene in scenes.values():
        for case in scene["cases"].values():
            ids = sorted(int(seed) for seed in case["seeds"])
            case["seed_ids"], case["seed_count"] = ids, len(ids)
            seed_ids.update(ids)
            for seed in case["seeds"].values():
                seed["missing_actions"] = [action for action in ACTIONS if action not in seed["actions"]]
                seed["actions"] = {
                    action: seed["actions"][action] for action in ACTIONS if action in seed["actions"]
                }
            case_count += 1
        scene["case_count"] = len(scene["cases"])

    return {
        "schema_version": "strideworld-task2-unit-horizons-v1",
        "model": model_name,
        "source": {"results_root": str(results_root), "normalized_index": str(index_path)},
        "definition": {
            "formula": "U_K=exp(mean(log(pi(inv(T_c0)@T_ck))/k, k=1..K))",
            "dose_horizons": list(HORIZONS),
            "min_pose_inlier_ratio": ratio,
            "min_pose_anchor_coverage": coverage,
            "threshold_source": threshold_source,
            "translation_floor_cm": FROZEN_METRIC_SETTINGS["translation_floor_cm"],
            "rotation_floor_deg": FROZEN_METRIC_SETTINGS["rotation_floor_deg"],
            "frozen_settings_sha256": FROZEN_METRIC_SETTINGS_SHA256,
        },
        "summary": {
            "scene_count": len(scenes),
            "case_count": case_count,
            "seed_ids": sorted(seed_ids),
            "seed_count": len(seed_ids),
            "video_count": len(rows),
            "status_counts": dict(sorted(statuses.items())),
            "available_by_horizon": available,
        },
        "scenes": scenes,
    }


def self_check() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 17, axis=0)
    poses[:, 2, 3] = np.arange(17) * 2.0
    result = unit_horizons(poses, np.ones(17, dtype=bool), list(range(17)), "W")
    assert all(abs(result[str(h)]["unit_one"] - 2.0) < 1e-9 for h in HORIZONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--min-pose-inlier-ratio", type=float)
    parser.add_argument("--min-pose-anchor-coverage", type=float)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check and (args.results_root is None or args.output is None):
        parser.error("--results-root and --output are required unless --self-check is used")
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    report = build_report(
        args.results_root.resolve(),
        model=args.model,
        min_pose_inlier_ratio=args.min_pose_inlier_ratio,
        min_pose_anchor_coverage=args.min_pose_anchor_coverage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({**report["summary"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
