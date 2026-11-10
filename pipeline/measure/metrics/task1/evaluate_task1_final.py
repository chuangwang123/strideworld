#!/usr/bin/env python3
"""Final-spec Task-1 evaluator.

The evaluator consumes pose NPZ artifacts produced by ``tools.pose`` (and is
backwards-compatible with the earlier ``task1_metrics.poses.npz`` field
names).  It applies the specification's fixed ``t>=1`` denominator, reports
conditional ray/pixel rigidity statistics and writes both a flat CSV and a
debug-rich JSON document.  It does not calculate any of the removed V0-PC,
S_MC or M_stable headline metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.tools.constants import FROZEN_METRIC_SETTINGS, FROZEN_METRIC_SETTINGS_SHA256  # noqa: E402
from metrics.tools.pose import (  # noqa: E402
    load_anchor_npz,
    load_track_npz,
    recover_pose_trajectory,
    save_pose_npz,
    world_measurement_mask,
)


SCHEMA_VERSION = "strideworld-task1-final-v1"
WORLD_MIN_POINTS = int(FROZEN_METRIC_SETTINGS["min_world_points"])
DEFAULT_MIN_POSE_INLIER_RATIO = float(FROZEN_METRIC_SETTINGS["min_pose_inlier_ratio"])
DEFAULT_MIN_POSE_ANCHOR_COVERAGE = float(FROZEN_METRIC_SETTINGS["min_pose_anchor_coverage"])
STRATA = ("near", "mid", "far")


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def _stats(values: Iterable[float], *, unit: str | None = None) -> dict[str, Any] | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    result: dict[str, Any] = {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "n": int(len(array)),
    }
    if unit:
        result["unit"] = unit
    return result


def _find_anchor_file(anchors_root: Path | None, state_id: str | None) -> Path | None:
    if anchors_root is None or not anchors_root.exists():
        return None
    candidates = list(anchors_root.rglob("anchors_candidate.npz")) + list(anchors_root.rglob("anchors.npz"))
    if state_id:
        matching = [path for path in candidates if state_id in str(path)]
        if matching:
            return sorted(matching)[0]
    return sorted(candidates)[0] if len(candidates) == 1 else None


def _find_track_file(case_dir: Path) -> Path | None:
    for name in ("tracks.npz", "track.npz", "cotracker.npz"):
        path = case_dir / name
        if path.exists():
            return path
    return None


def _find_pose_files(root: Path) -> list[Path]:
    names = {"task1_metrics.poses.npz", "pose.npz", "poses.npz", "task1_pose.npz"}
    return sorted(path for path in root.rglob("*.npz") if path.name in names)


def _find_track_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("tracks.npz"))


def _json_for_pose(path: Path) -> dict[str, Any]:
    candidates = [path.with_name("task1_metrics.json"), path.with_suffix(".json")]
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def _assignment_hash(anchor_id: np.ndarray, depth: np.ndarray, world_mask: np.ndarray) -> tuple[str, dict[str, str]]:
    eligible = np.flatnonzero(world_mask & np.isfinite(depth))
    order = sorted(eligible.tolist(), key=lambda index: (float(depth[index]), int(anchor_id[index])))
    chunks = np.array_split(np.asarray(order, dtype=np.int64), 3)
    mapping: dict[str, str] = {}
    for label, chunk in zip(STRATA, chunks):
        for index in chunk.tolist():
            mapping[str(int(anchor_id[index]))] = label
    canonical = [{"anchor_id": int(key), "stratum": mapping[key]} for key in sorted(mapping, key=lambda token: int(token))]
    digest = hashlib.sha256(json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    return digest, mapping


def _match_anchor_arrays(pose: dict[str, np.ndarray], anchors: dict[str, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(pose.get("anchor_id"), dtype=np.int64)
    n = len(ids)
    if anchors is None:
        return np.arange(n), np.zeros(n, dtype=bool), np.full(n, np.nan), np.zeros(n, dtype=bool)
    anchor_ids = np.asarray(anchors["anchor_id"], dtype=np.int64)
    positions = {int(value): index for index, value in enumerate(anchor_ids)}
    try:
        order = np.asarray([positions[int(value)] for value in ids], dtype=np.int64)
    except KeyError:
        raise ValueError("pose and anchor files do not contain the same anchor_id set")
    world_mask = ~np.asarray(anchors["is_pose_anchor"], dtype=bool)[order]
    depth = np.asarray(anchors.get("source_depth_cm", np.full(len(anchor_ids), np.nan)), dtype=np.float64)[order]
    return order, world_mask, depth, np.isfinite(depth)


def _load_pose(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _pose_metadata(pose: dict[str, np.ndarray], pose_path: Path) -> dict[str, Any]:
    raw = pose.get("measurement_metadata_json")
    if raw is not None:
        try:
            return json.loads(str(_scalar(raw)))
        except (TypeError, json.JSONDecodeError):
            pass
    manifest = pose_path.parent / "run_manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text())
            key = payload.get("record", {}).get("primary_key", {})
            return {"model": key.get("model"), "seed": key.get("seed"), "state_id": key.get("state_id"), "case_id": key.get("case_id")}
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _case_result(
    pose_path: Path,
    anchors_root: Path | None,
    min_pose_inlier_ratio: float,
    min_pose_anchor_coverage: float = DEFAULT_MIN_POSE_ANCHOR_COVERAGE,
) -> dict[str, Any]:
    pose = _load_pose(pose_path)
    metadata = _pose_metadata(pose, pose_path)
    sidecar = _json_for_pose(pose_path)
    state_id = str(_scalar(pose.get("state_id"), metadata.get("state_id", "unknown")))
    case_id = str(_scalar(pose.get("case_id"), metadata.get("case_id", pose_path.parent.name)))
    local_anchors = [
        pose_path.parent / "anchors.npz", pose_path.parent / "anchors_candidate.npz",
        pose_path.parent / "gt" / "anchors.npz", pose_path.parent / "gt" / "anchors_candidate.npz",
    ]
    anchors_path = next((candidate for candidate in local_anchors if candidate.exists()), None)
    if anchors_path is None:
        anchors_path = _find_anchor_file(anchors_root, state_id)
    anchors = load_anchor_npz(anchors_path) if anchors_path else None
    _, world_mask, depth, depth_available = _match_anchor_arrays(pose, anchors)
    track_path = _find_track_file(pose_path.parent)
    tracks = None
    if track_path is not None:
        try:
            tracks = load_track_npz(track_path)
        except (OSError, ValueError, KeyError):
            tracks = None
    T = int(len(pose["camera_valid"]))
    camera_valid = np.resize(np.asarray(pose.get("camera_valid", np.zeros(T)), dtype=bool), T)
    coverage = np.resize(np.asarray(pose.get("pose_anchor_coverage", np.zeros(T)), dtype=np.float64), T)
    ratio = np.resize(np.asarray(pose.get("pose_inlier_ratio", np.zeros(T)), dtype=np.float64), T)
    if not np.any(coverage):
        frame_metrics = sidecar.get("frame_metrics", [])
        coverage = np.resize(np.asarray([frame.get("pose_anchor_coverage", 0.0) for frame in frame_metrics], dtype=np.float64), T)
        ratio = np.resize(np.asarray([frame.get("camera_pose_inlier_ratio", 0.0) for frame in frame_metrics], dtype=np.float64), T)
    # Recompute M_t from the selected evaluation profile instead of trusting
    # a stale measurable array produced with another threshold.
    measurable = camera_valid & (coverage >= float(min_pose_anchor_coverage)) & (ratio >= float(min_pose_inlier_ratio))
    pose_ratio = ratio
    pose_coverage = coverage

    ray = np.asarray(pose.get("world_ray_error_rad", np.full((T, len(world_mask)), np.nan)), dtype=np.float64)
    px = np.asarray(pose.get("world_pixel_error_px", np.full_like(ray, np.nan)), dtype=np.float64)
    if np.isnan(px).all() and "predicted_uv" in pose and tracks is not None:
        # Old artifacts did not save pixel error, but do save predicted_uv.  If
        # a local tracks.npz is present, calculate it without changing the
        # historical pose artifact.
        track_positions = {
            int(value): index
            for index, value in enumerate(np.asarray(tracks["anchor_id"], dtype=np.int64))
        }
        try:
            order = np.asarray(
                [track_positions[int(value)] for value in np.asarray(pose["anchor_id"])],
                dtype=np.int64,
            )
            observed = np.asarray(tracks["uv"], dtype=np.float64)[:, order]
            predicted = np.asarray(pose["predicted_uv"], dtype=np.float64)
            px = np.linalg.norm(observed - predicted, axis=2)
        except (KeyError, ValueError, IndexError):
            pass
    if ray.ndim != 2:
        ray = np.full((T, len(world_mask)), np.nan)
    if px.ndim != 2:
        px = np.full_like(ray, np.nan)
    world_mask = np.resize(world_mask.astype(bool), ray.shape[1])
    stored_world_mask = np.asarray(pose.get("world_scored_mask", np.empty(0)), dtype=bool)
    if stored_world_mask.shape == ray.shape:
        scored_world = stored_world_mask
    elif tracks is not None and anchors is not None:
        try:
            candidate = world_measurement_mask(pose, tracks, anchors)
            scored_world = candidate if candidate.shape == ray.shape else np.isfinite(ray)
        except (KeyError, ValueError, IndexError):
            scored_world = np.isfinite(ray)
    else:
        scored_world = np.isfinite(ray)
    frame_records: list[dict[str, Any]] = []
    ray_values: list[float] = []
    px_values: list[float] = []
    px_norm_values: list[float] = []
    strata_values: dict[str, list[float]] = {label: [] for label in STRATA}
    if anchors is not None and depth_available.any():
        assignment_hash, assignment = _assignment_hash(np.asarray(pose["anchor_id"]), depth, world_mask)
    else:
        assignment_hash, assignment = None, {}
    image_size = np.asarray(pose.get("track_image_size", [0, 0]), dtype=np.float64).reshape(-1)
    if (len(image_size) < 2 or np.any(image_size[:2] <= 0)):
        if tracks is not None:
            image_size = np.asarray(tracks["image_size"], dtype=np.float64).reshape(-1)
    diagonal = float(np.linalg.norm(image_size[:2])) if len(image_size) >= 2 and np.all(image_size[:2] > 0) else None
    for frame in range(T):
        finite_world = world_mask & scored_world[frame] & np.isfinite(ray[frame])
        if frame == 0 or not measurable[frame]:
            finite_world[:] = False
        n_scored = int(finite_world.sum())
        if n_scored >= WORLD_MIN_POINTS:
            ray_deg = np.degrees(ray[frame, finite_world])
            ray_values.extend(ray_deg.tolist())
            if px.shape[1] == len(world_mask):
                px_frame = px[frame, finite_world]
                px_frame = px_frame[np.isfinite(px_frame)]
                px_values.extend(px_frame.tolist())
                if diagonal:
                    px_norm_values.extend((px_frame / diagonal).tolist())
            if assignment:
                for label in STRATA:
                    ids = np.asarray([index for index, value in enumerate(pose["anchor_id"]) if assignment.get(str(int(value))) == label], dtype=np.int64)
                    ids = ids[np.isin(ids, np.flatnonzero(finite_world))]
                    strata_values[label].extend(np.degrees(ray[frame, ids]).tolist())
        frame_records.append({
            "frame_index": frame,
            "frame_id": str(np.asarray(pose.get("frame_ids", np.arange(T)))[frame]),
            "measurable": bool(measurable[frame]),
            "pose_inlier_ratio": float(pose_ratio[frame]),
            "pose_anchor_coverage": float(pose_coverage[frame]),
            "n_world_scored": n_scored,
            "e_ray_deg_median": float(np.median(np.degrees(ray[frame, finite_world]))) if n_scored else None,
            "status": "world_points_insufficient" if bool(measurable[frame]) and n_scored < WORLD_MIN_POINTS else ("ok" if n_scored else "unmeasurable"),
        })
    strata: dict[str, Any] = {}
    for label in STRATA:
        ids = np.asarray([index for index, value in enumerate(pose["anchor_id"]) if assignment.get(str(int(value))) == label], dtype=np.int64)
        stats = _stats(strata_values[label], unit="deg")
        entry: dict[str, Any] = {"depth_min_cm": None, "depth_max_cm": None, "n_anchors": int(len(ids)), "e_rig_ray_deg": stats}
        if len(ids) and np.isfinite(depth[ids]).any():
            entry["depth_min_cm"] = float(np.nanmin(depth[ids]))
            entry["depth_max_cm"] = float(np.nanmax(depth[ids]))
        strata[label] = entry
    c_pose = float(np.mean(measurable[1:])) if T > 1 else 0.0
    task = {
        "c_pose": c_pose,
        "min_pose_inlier_ratio": float(min_pose_inlier_ratio),
        "min_pose_anchor_coverage": float(min_pose_anchor_coverage),
        "instrument_failure_rate": 1.0 - c_pose,
        "e_rig_ray_deg": _stats(ray_values, unit="deg"),
        "e_rig_px": _stats(px_values, unit="px"),
        "e_rig_px_normalized": _stats(px_norm_values, unit="fraction_of_image_diagonal"),
        "depth_strata": strata,
        "depth_strata_assignment_sha256": assignment_hash,
        "frames": frame_records,
    }
    row: dict[str, Any] = {
        "model": str(sidecar.get("case_identity", {}).get("model", _scalar(pose.get("model"), metadata.get("model", "unknown")))),
        "state_id": state_id,
        "case_id": case_id,
        "seed": str(sidecar.get("case_identity", {}).get("seed", _scalar(pose.get("seed"), metadata.get("seed", "unknown")))),
        "c_pose": c_pose,
        "min_pose_inlier_ratio": float(min_pose_inlier_ratio),
        "min_pose_anchor_coverage": float(min_pose_anchor_coverage),
        "instrument_failure_rate": 1.0 - c_pose,
        "e_rig_ray_deg_mean": task["e_rig_ray_deg"]["mean"] if task["e_rig_ray_deg"] else None,
        "e_rig_ray_deg_std": task["e_rig_ray_deg"]["std"] if task["e_rig_ray_deg"] else None,
        "e_rig_ray_deg_median": task["e_rig_ray_deg"]["median"] if task["e_rig_ray_deg"] else None,
        "e_rig_ray_deg_p90": task["e_rig_ray_deg"]["p90"] if task["e_rig_ray_deg"] else None,
        "e_rig_ray_deg_n": task["e_rig_ray_deg"]["n"] if task["e_rig_ray_deg"] else 0,
        "e_rig_px_mean": task["e_rig_px"]["mean"] if task["e_rig_px"] else None,
        "e_rig_px_std": task["e_rig_px"]["std"] if task["e_rig_px"] else None,
        "e_rig_px_normalized_mean": task["e_rig_px_normalized"]["mean"] if task["e_rig_px_normalized"] else None,
        "depth_strata_assignment_sha256": assignment_hash,
        "n_frames": T,
        "n_measurable_frames": int(np.sum(measurable[1:])),
        "input_pose_npz": str(pose_path),
        "input_anchors_npz": str(anchors_path) if anchors_path else None,
    }
    for label in STRATA:
        stratum_stats = strata[label]["e_rig_ray_deg"]
        row[f"{label}_depth_min_cm"] = strata[label]["depth_min_cm"]
        row[f"{label}_depth_max_cm"] = strata[label]["depth_max_cm"]
        row[f"{label}_n_anchors"] = strata[label]["n_anchors"]
        row[f"{label}_e_rig_ray_deg_mean"] = stratum_stats["mean"] if stratum_stats else None
        row[f"{label}_e_rig_ray_deg_std"] = stratum_stats["std"] if stratum_stats else None
        row[f"{label}_e_rig_ray_deg_median"] = stratum_stats["median"] if stratum_stats else None
        row[f"{label}_e_rig_ray_deg_p90"] = stratum_stats["p90"] if stratum_stats else None
        row[f"{label}_e_rig_ray_deg_n"] = stratum_stats["n"] if stratum_stats else 0
    return {"row": row, "task1": task, "frames": frame_records, "status": "ok", "pose_path": str(pose_path), "anchors_path": str(anchors_path) if anchors_path else None}


def evaluate(
    input_root: str | Path,
    output_csv: str | Path,
    output_json: str | Path | None = None,
    anchors_root: str | Path | None = None,
    camera_matrix: np.ndarray | None = None,
    min_pose_inlier_ratio: float = DEFAULT_MIN_POSE_INLIER_RATIO,
    min_pose_anchor_coverage: float = DEFAULT_MIN_POSE_ANCHOR_COVERAGE,
) -> dict[str, Any]:
    if not 0.0 <= float(min_pose_inlier_ratio) <= 1.0:
        raise ValueError("min_pose_inlier_ratio must be in [0, 1]")
    if not 0.0 <= float(min_pose_anchor_coverage) <= 1.0:
        raise ValueError("min_pose_anchor_coverage must be in [0, 1]")
    input_root = Path(input_root)
    output_csv = Path(output_csv)
    output_json = Path(output_json) if output_json else output_csv.with_suffix(".json")
    anchors_root = Path(anchors_root) if anchors_root else None
    pose_paths = _find_pose_files(input_root) if input_root.exists() else []
    recovery_records: list[dict[str, Any]] = []
    if not pose_paths and camera_matrix is not None and input_root.exists():
        # A raw-result folder may only contain CoTracker tracks.  Recover the
        # pose artifact once, persist it beside the requested reports, then
        # run exactly the same case evaluator as for a precomputed NPZ.
        recovery_root = output_csv.parent / f"{output_csv.stem}_artifacts" / "recovered_pose"
        for track_path in _find_track_files(input_root):
            state_guess = track_path.parent.name
            state_match = re.search(r"(wr_state_[0-9]+)", str(track_path))
            if state_match:
                state_guess = state_match.group(1)
            anchor_path = track_path.parent / "anchors.npz"
            if not anchor_path.exists():
                anchor_path = _find_anchor_file(anchors_root, state_guess)
            if anchor_path is None or not anchor_path.exists():
                recovery_records.append({"tracks": str(track_path), "status": "missing_anchors"})
                continue
            try:
                anchor_data = load_anchor_npz(anchor_path)
                track_data = load_track_npz(track_path)
                result = recover_pose_trajectory(
                    anchor_data,
                    track_data,
                    camera_matrix,
                    min_pose_inlier_ratio=min_pose_inlier_ratio,
                    min_pose_anchor_coverage=min_pose_anchor_coverage,
                )
                recovered = recovery_root / f"{track_path.parent.name}.pose.npz"
                save_pose_npz(recovered, result, {"recovered_from_tracks": str(track_path), "state_id": state_guess})
                pose_paths.append(recovered)
                recovery_records.append({"tracks": str(track_path), "pose": str(recovered), "status": "ok"})
            except Exception as error:
                recovery_records.append({"tracks": str(track_path), "status": "error", "error": repr(error)})
    cases: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in pose_paths:
        try:
            cases.append(_case_result(path, anchors_root, min_pose_inlier_ratio, min_pose_anchor_coverage))
        except Exception as error:  # preserve a debuggable record and continue other cases
            errors.append({"pose_path": str(path), "error": repr(error)})
    rows = [case["row"] for case in cases]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_csv.parent / f"{output_csv.stem}_artifacts" / "cases"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        case_name = str(case["row"].get("case_id") or f"case_{index:04d}")
        safe_name = "".join(character if character.isalnum() or character in "-_." else "_" for character in case_name)
        (artifact_dir / f"{safe_name}.json").write_text(json.dumps(case, indent=2, ensure_ascii=False, allow_nan=False, default=str))
    columns = sorted({key for row in rows for key in row}) if rows else ["model", "state_id", "case_id", "status"]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    model_summary: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(case["row"].get("model", "unknown")), []).append(case)
    for model, model_cases in grouped.items():
        c_values = np.asarray([case["task1"]["c_pose"] for case in model_cases], dtype=np.float64)
        ray_means = [case["task1"]["e_rig_ray_deg"]["mean"] for case in model_cases if case["task1"]["e_rig_ray_deg"]]
        ray_count = sum(
            int(case["task1"]["e_rig_ray_deg"]["n"])
            for case in model_cases if case["task1"]["e_rig_ray_deg"]
        )
        ray_sum = sum(
            float(case["task1"]["e_rig_ray_deg"]["n"]) * float(case["task1"]["e_rig_ray_deg"]["mean"])
            for case in model_cases if case["task1"]["e_rig_ray_deg"]
        )
        ray_sq_sum = sum(
            float(case["task1"]["e_rig_ray_deg"]["n"]) * (
                float(case["task1"]["e_rig_ray_deg"]["std"]) ** 2
                + float(case["task1"]["e_rig_ray_deg"]["mean"]) ** 2
            )
            for case in model_cases if case["task1"]["e_rig_ray_deg"]
        )
        pooled_ray_mean = ray_sum / ray_count if ray_count else None
        pooled_ray_std = (
            max(ray_sq_sum / ray_count - pooled_ray_mean**2, 0.0) ** 0.5
            if ray_count else None
        )
        frame_count = sum(int(case["row"].get("n_frames") or 0) for case in model_cases)
        measurable_count = sum(int(case["row"].get("n_measurable_frames") or 0) for case in model_cases)
        model_summary[model] = {
            "n_cases": len(model_cases),
            "c_pose_mean": float(np.mean(c_values)) if len(c_values) else None,
            "c_pose_std": float(np.std(c_values)) if len(c_values) else None,
            "instrument_failure_rate_mean": float(1.0 - np.mean(c_values)) if len(c_values) else None,
            "n_frames_total": frame_count,
            "n_measurable_frames_total": measurable_count,
            "weighted_c_pose": float(measurable_count / max(frame_count - len(model_cases), 1)) if frame_count else None,
            "e_rig_ray_deg_pooled_n": ray_count,
            "e_rig_ray_deg_pooled_mean": pooled_ray_mean,
            "e_rig_ray_deg_pooled_std": pooled_ray_std,
            "e_rig_ray_deg_case_mean": float(np.mean(ray_means)) if ray_means else None,
            "e_rig_ray_deg_case_std": float(np.std(ray_means)) if ray_means else None,
            "cases_with_no_rig_observation": int(sum(case["task1"]["e_rig_ray_deg"] is None for case in model_cases)),
        }
    document = {
        "schema_version": SCHEMA_VERSION,
        "constants": {"world_min_points": WORLD_MIN_POINTS, "strata": list(STRATA), "exclude_frame_zero": True, "min_pose_inlier_ratio": float(min_pose_inlier_ratio), "min_pose_anchor_coverage": float(min_pose_anchor_coverage), "frozen_settings_sha256": FROZEN_METRIC_SETTINGS_SHA256},
        "input_root": str(input_root),
        "anchors_root": str(anchors_root) if anchors_root else None,
        "n_pose_files": len(pose_paths),
        "n_cases": len(cases),
        "model_summary": model_summary,
        "errors": errors,
        "recovery": recovery_records,
        "cases": cases,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False, default=str))
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--anchors-root")
    parser.add_argument("--camera-matrix", nargs=9, type=float, metavar="K", help="K row-major values; enables tracks.npz -> pose recovery")
    parser.add_argument("--min-pose-inlier-ratio", type=float, default=DEFAULT_MIN_POSE_INLIER_RATIO,
                        help="Task-1 M_t pose inlier-ratio gate (evaluation hyperparameter)")
    parser.add_argument("--min-pose-anchor-coverage", type=float, default=DEFAULT_MIN_POSE_ANCHOR_COVERAGE)
    args = parser.parse_args()
    K = np.asarray(args.camera_matrix, dtype=np.float64).reshape(3, 3) if args.camera_matrix else None
    document = evaluate(args.input_root, args.output_csv, args.output_json, args.anchors_root, K, args.min_pose_inlier_ratio, args.min_pose_anchor_coverage)
    print(json.dumps({"n_pose_files": document["n_pose_files"], "n_cases": document["n_cases"], "errors": document["errors"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
