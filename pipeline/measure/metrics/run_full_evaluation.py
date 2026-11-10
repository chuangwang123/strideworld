#!/usr/bin/env python3
"""Run the complete StrideWorld Task 1/2/3 evaluation from one eval index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import re
import shlex
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.task1.evaluate_task1_final import _case_result  # noqa: E402
from metrics.task1.run_strideworld_task1 import source_to_target_affine  # noqa: E402
from metrics.task1.run_strideworld_task1_pool import PersistentCoTracker  # noqa: E402
from metrics.task2.task2_metrics import (  # noqa: E402
    ACTIONS,
    DOSES,
    ROTATION_ACTIONS,
    aggregate_cells,
    build_document as build_task2_document,
    evaluate_cell,
    missing_cell,
    write_outputs as write_task2_outputs,
)
from metrics.task3.task3_metrics import (  # noqa: E402
    build_document as build_task3_document,
    evaluate_pair,
    pair_rows,
    write_outputs as write_task3_outputs,
)
from metrics.tools.constants import FROZEN_METRIC_SETTINGS, FROZEN_METRIC_SETTINGS_SHA256  # noqa: E402
from metrics.tools.gt import build_candidate_gt, load_camera, sha256_file  # noqa: E402
from metrics.tools.pose import load_anchor_npz, load_track_npz, recover_pose_trajectory, save_pose_npz  # noqa: E402


READY = "GENERATION_READY"
TRANSLATION_ACTIONS = frozenset(("W", "S", "A", "D"))


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _segments(actions: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for action in actions:
        action = str(action).upper()
        if result and result[-1]["action"] == action:
            result[-1]["units"] += 1
        else:
            result.append({"action": action, "units": 1})
    return result


def _source_transform(row: Mapping[str, Any]) -> str:
    generation = Path(str(row.get("generation_record", "")))
    if generation.is_file():
        payload = json.loads(generation.read_text())
        executed = generation.parent / str(payload.get("executed_action_record_path", ""))
        if executed.is_file():
            value = json.loads(executed.read_text()).get("source_transform")
            mode = value.get("mode") if isinstance(value, dict) else value
            if mode in {"center_crop", "stretch", "center_crop_floor"}:
                return str(mode)
    raise ValueError(f"missing verified source_transform: {row.get('record_id')}")


def _normalise_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    case_uid = str(value.get("case_uid", ""))
    if "/" not in case_uid:
        raise ValueError(f"invalid case_uid: {case_uid!r}")
    scene, case = case_uid.split("/", 1)
    route_id = str(value.get("route_id", ""))
    actions = [str(action).upper() for action in value.get("actions", [])]
    if not actions or any(action not in ACTIONS for action in actions):
        raise ValueError(f"invalid actions for {value.get('record_id')}")
    boundaries = [int(frame) for frame in value.get("unit_boundary_frames", [])]
    if len(boundaries) != len(actions) + 1 or boundaries[0] < 0 or any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError(f"invalid unit boundaries for {value.get('record_id')}")
    is_atomic = route_id.startswith("atomic_")
    member = route_id.rsplit("_", 1)[-1] if not is_atomic else None
    pair_id = route_id.rsplit("_", 1)[0] if member in {"A", "B"} else None
    segments = _segments(actions)
    cumulative = np.cumsum([segment["units"] for segment in segments]).tolist()
    record = {
        "route": {
            "route_id": route_id, "kind": "atomic" if is_atomic else "combination",
            "family": None if is_atomic else route_id.split("_", 1)[0],
            "member": member, "pair_id": pair_id, "segments": segments,
            "expanded_unit_actions": actions, "total_units": len(actions),
            "segment_boundary_units": cumulative,
        },
        "timing": {
            "unit_boundary_frames": boundaries,
            "segment_boundary_frames": [boundaries[index] for index in cumulative],
            "scored_frame_start": int(value.get("scored_frame_start", boundaries[0])),
            "scored_frame_end": int(value.get("scored_frame_end", boundaries[-1])),
            "frame_count": int(value.get("decoded_metadata", {}).get("nb_read_frames", boundaries[-1] + 1)),
        },
    }
    value.update({
        "primary_key": {"model": str(value.get("model")), "scene_id": scene, "case_id": case,
                        "seed": int(value.get("seed")), "route_id": route_id,
                        "task": "task2" if is_atomic else "task3"},
        "state_id": f"{scene}_{case}_state0000", "pair_id": pair_id, "member": member,
        "_record": record,
    })
    if value.get("status") == READY:
        video = Path(str(value.get("video_path", "")))
        case_root = Path(str(value.get("eval_case_path", "")))
        required = [video, case_root / "rgb.png", case_root / "depth_cm.npy",
                    case_root / "valid_mask.npy", case_root / "camera.json", case_root / "depth_encoding.json"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{value.get('record_id')} missing: {missing}")
        value["_source_transform"] = _source_transform(value)
    return value


def load_eval_index(data_root: Path, model: str) -> list[dict[str, Any]]:
    path = data_root if data_root.is_file() else data_root / "eval_index" / f"{model}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [_normalise_row(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(row["primary_key"]["model"] != model for row in rows):
        raise ValueError(f"index does not exclusively describe model {model!r}")
    ids = [str(row.get("record_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record_id in eval index")
    for case_uid, group in _groups(rows, lambda row: row["case_uid"]).items():
        hashes = {row.get("condition_sha256") for row in group}
        paths = {row.get("eval_case_path") for row in group}
        if len(hashes) != 1 or len(paths) != 1:
            raise ValueError(f"state binding is inconsistent: {case_uid}")
        if sha256_file(Path(str(group[0]["eval_case_path"])) / "rgb.png") != next(iter(hashes)):
            raise ValueError(f"condition hash mismatch: {case_uid}")
    return rows


def _groups(values: Iterable[Any], key: Callable[[Any], Any]) -> dict[Any, list[Any]]:
    result: dict[Any, list[Any]] = defaultdict(list)
    for value in values:
        result[key(value)].append(value)
    return dict(result)


def _build_state_anchor(case_uid: str, case_root: str, output: str, max_depth_cm: float) -> dict[str, Any]:
    case_path, output_path = Path(case_root), Path(output)
    anchor_path = output_path / "anchors.npz"
    if anchor_path.is_file():
        anchors = load_anchor_npz(anchor_path)
        pose_count = int(np.asarray(anchors["is_pose_anchor"], dtype=bool).sum())
        world_count = int((~np.asarray(anchors["is_pose_anchor"], dtype=bool)).sum())
        if pose_count < int(FROZEN_METRIC_SETTINGS["min_pose_points"]) or world_count < int(FROZEN_METRIC_SETTINGS["min_world_points"]):
            raise RuntimeError(f"{case_uid}: cached anchors are insufficient pose={pose_count}, world={world_count}")
        return {"case_uid": case_uid, "anchor_path": str(anchor_path), "count": len(anchors["anchor_id"]), "resumed": True}
    if output_path.exists():
        rejected = output_path.with_name(f"{output_path.name}.incomplete.{os.getpid()}")
        output_path.rename(rejected)
    manifest = build_candidate_gt(
        case_path / "rgb.png", case_path / "depth_cm.npy", case_path / "camera.json", output_path,
        raw_to_cm=1.0, depth_mode="camera_z", depth_encoding=case_path / "depth_encoding.json",
        allow_candidate=False, max_valid_depth_cm=max_depth_cm,
    )
    anchors = load_anchor_npz(anchor_path)
    pose_count = int(np.asarray(anchors["is_pose_anchor"], dtype=bool).sum())
    world_count = int((~np.asarray(anchors["is_pose_anchor"], dtype=bool)).sum())
    minimum = int(FROZEN_METRIC_SETTINGS["min_pose_points"])
    if pose_count < minimum or world_count < int(FROZEN_METRIC_SETTINGS["min_world_points"]):
        raise RuntimeError(f"{case_uid}: insufficient anchors pose={pose_count}, world={world_count}")
    return {"case_uid": case_uid, "anchor_path": str(anchor_path), "count": len(anchors["anchor_id"]),
            "pose": pose_count, "world": world_count, "manifest": manifest, "resumed": False}


def prepare_anchors(rows: list[dict[str, Any]], output_root: Path, workers: int, max_depth_cm: float) -> dict[str, Path]:
    states = {row["case_uid"]: row for row in rows if row.get("status") == READY}
    jobs = [(uid, str(row["eval_case_path"]), str(output_root / "state_anchors" / _safe(uid)), max_depth_cm)
            for uid, row in sorted(states.items())]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_build_state_anchor, *job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 20 == 0 or index == len(futures):
                print(json.dumps({"stage": "anchors", "completed": index, "total": len(futures)}), flush=True)
    _write_json(output_root / "details" / "anchor_results.json", results)
    return {result["case_uid"]: Path(result["anchor_path"]) for result in results}


def _pose_output(output_root: Path, row: Mapping[str, Any]) -> Path:
    return output_root / "artifacts" / _safe(str(row["record_id"]))


def _pose_worker(gpu: int, config: dict[str, Any], tasks: Any, results: Any) -> None:
    try:
        tracker = PersistentCoTracker(config, gpu)
    except Exception as error:
        results.put({"status": "worker_init_error", "gpu": gpu, "error": repr(error), "traceback": traceback.format_exc()})
        return
    while True:
        row = tasks.get()
        if row is None:
            return
        started = time.perf_counter()
        try:
            output = _pose_output(Path(config["output_root"]), row)
            output.mkdir(parents=True, exist_ok=True)
            pose_path, tracks_path = output / "pose.npz", output / "tracks.npz"
            if pose_path.is_file():
                with np.load(pose_path, allow_pickle=False) as pose:
                    count = int(np.asarray(pose["camera_valid"], dtype=bool).sum())
                    frames = int(len(pose["camera_valid"]))
                results.put({"status": "ok", "record_id": row["record_id"], "pose": str(pose_path),
                             "gpu": gpu, "resumed": True, "camera_valid_frames": count,
                             "n_frames": frames, "elapsed_sec": time.perf_counter() - started})
                continue
            anchor_path = Path(str(row["_anchor_path"]))
            local_gt = output / "gt"; local_gt.mkdir(exist_ok=True)
            local_anchor = local_gt / "anchors.npz"
            if not local_anchor.exists():
                local_anchor.symlink_to(anchor_path)
            frame_limit = int(row["_record"]["timing"]["scored_frame_end"]) + 1
            tracker.set_frame_limit(frame_limit)
            tracker_summary: dict[str, Any] | None = None
            if tracks_path.is_file():
                try:
                    tracks = load_track_npz(tracks_path)
                except Exception:
                    tracks_path.unlink()
                    tracks = None
            else:
                tracks = None
            if tracks is None:
                tracker_summary = tracker.track(
                    Path(row["video_path"]), anchor_path, Path(row["eval_case_path"]) / "rgb.png",
                    tracks_path, source_transform=str(row["_source_transform"]),
                )
                tracks = load_track_npz(tracks_path)
            anchors = load_anchor_npz(anchor_path)
            camera = load_camera(Path(row["eval_case_path"]) / "camera.json")
            target_h, target_w = map(int, tracks["image_size"])
            affine = source_to_target_affine(tuple(camera["image_size"]), (target_h, target_w), str(row["_source_transform"]))
            K_target = affine @ camera["K"]
            pose = recover_pose_trajectory(
                anchors, tracks, K_target,
                min_pose_inlier_ratio=float(config["min_pose_inlier_ratio"]),
                min_pose_anchor_coverage=float(config["min_pose_anchor_coverage"]),
            )
            key = row["primary_key"]
            metadata = {
                **key, "state_id": row["state_id"], "record_id": row["record_id"],
                "video": row["video_path"], "video_sha256": row.get("video_sha256"),
                "anchor_path": str(anchor_path), "source_to_track_affine": affine.tolist(),
                "source_transform": row["_source_transform"],
                "expected_visibility_source": "all_pose_anchors_fallback",
                "frozen_settings_sha256": FROZEN_METRIC_SETTINGS_SHA256,
            }
            save_pose_npz(pose_path, pose, metadata)
            _write_json(output / "run_manifest.json", {
                "status": "complete", "record_id": row["record_id"], "primary_key": key,
                "video": row["video_path"], "video_sha256": row.get("video_sha256"),
                "anchor_path": str(anchor_path), "tracks": str(tracks_path), "pose": str(pose_path),
                "frame_limit": frame_limit, "source_transform": row["_source_transform"],
                "source_to_track_affine": affine.tolist(), "K_target": K_target.tolist(),
                "expected_visibility_source": "all_pose_anchors_fallback",
                "tracker": tracker_summary, "frozen_settings_sha256": FROZEN_METRIC_SETTINGS_SHA256,
            })
            results.put({"status": "ok", "record_id": row["record_id"], "pose": str(pose_path), "gpu": gpu,
                         "resumed": False, "camera_valid_frames": int(np.asarray(pose["camera_valid"], bool).sum()),
                         "measurable_frames": int(np.asarray(pose["measurable"], bool).sum()),
                         "n_frames": len(pose["camera_valid"]), "elapsed_sec": time.perf_counter() - started})
        except Exception as error:
            results.put({"status": "error", "record_id": row.get("record_id"), "gpu": gpu,
                         "error": repr(error), "traceback": traceback.format_exc(),
                         "elapsed_sec": time.perf_counter() - started})


def run_pose_stage(rows: list[dict[str, Any]], output_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("status") == READY]
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus or args.workers_per_gpu < 1:
        raise ValueError("--gpus must be non-empty and --workers-per-gpu >= 1")
    config = {
        "output_root": str(output_root), "checkpoint": str(args.checkpoint),
        "cotracker_root": str(args.cotracker_root), "visibility_threshold": 0.9,
        "source_transform": "center_crop", "first_frame_max_l1": args.first_frame_max_l1,
        "min_pose_inlier_ratio": args.min_pose_inlier_ratio,
        "min_pose_anchor_coverage": args.min_pose_anchor_coverage,
    }
    context = mp.get_context("spawn")
    tasks = context.Queue()
    results = context.Queue()
    workers = [context.Process(target=_pose_worker, args=(gpu, config, tasks, results))
               for gpu in gpus for _ in range(args.workers_per_gpu)]
    for worker in workers:
        worker.start()
    for row in selected:
        tasks.put(row)
    for _ in workers:
        tasks.put(None)
    output: list[dict[str, Any]] = []
    while len(output) < len(selected):
        try:
            item = results.get(timeout=30)
        except queue.Empty:
            if not any(worker.is_alive() for worker in workers):
                raise RuntimeError(f"all pose workers exited after {len(output)}/{len(selected)} results")
            continue
        if item.get("status") == "worker_init_error":
            for worker in workers:
                worker.terminate()
            raise RuntimeError(json.dumps(item, ensure_ascii=False))
        output.append(item)
        if len(output) % 25 == 0 or len(output) == len(selected):
            failures = sum(result["status"] != "ok" for result in output)
            print(json.dumps({"stage": "poses", "completed": len(output), "total": len(selected), "failures": failures}), flush=True)
    for worker in workers:
        worker.join()
    output.sort(key=lambda result: str(result.get("record_id")))
    _write_jsonl(output_root / "details" / "pose_worker_results.jsonl", output)
    return output


def _path_value(value: Mapping[str, Any], path: str) -> float | None:
    current: Any = value
    for token in path.split("."):
        current = current.get(token) if isinstance(current, Mapping) else None
    try:
        number = float(current)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _hierarchical(
    items: Iterable[Mapping[str, Any]], path: str, *, reducer: str = "median",
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    states = _groups(items, lambda item: (str(item.get("scene_id")), str(item.get("state_id", item.get("case_id")))))
    state_values: list[tuple[str, float]] = []
    reduce_fn = np.mean if reducer == "mean" else np.median
    for (scene, _), group in states.items():
        values = [value for item in group if (value := _path_value(item, path)) is not None]
        if values:
            state_values.append((scene, float(reduce_fn(values))))
    values = np.asarray([value for _, value in state_values], dtype=float)
    if not len(values):
        return {"value": None, "p90": None, "std": None, "ci95": [None, None], "n_states": 0, "n_scenes": 0}
    central = float(reduce_fn(values))
    scenes = sorted({scene for scene, _ in state_values})
    by_scene = {scene: [value for owner, value in state_values if owner == scene] for scene in scenes}
    rng = np.random.default_rng(20260917)
    sampled: list[float] = []
    for _ in range(max(0, bootstrap_samples)):
        picked = rng.choice(scenes, len(scenes), replace=True)
        sample = [value for scene in picked for value in by_scene[str(scene)]]
        sampled.append(float(reduce_fn(sample)))
    ci = [float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))] if sampled else [None, None]
    return {"value": central, "p90": float(np.percentile(values, 90)), "std": float(np.std(values)), "ci95": ci,
            "n_states": len(values), "n_scenes": len(scenes), "reducer": reducer}


def _task2_hierarchy(cells: list[dict[str, Any]], bootstrap_samples: int) -> dict[str, Any]:
    fields = {
        "u_median": "u_abs", "gamma_median": "gamma", "r2_median": "r2",
        "r16_median": "r16", "e_rig_over_u_median": "e_rig_over_u",
        "purity_median": "purity", "yaw_crosstalk_deg_median": "yaw_crosstalk_deg",
        "turn_translation_cm_median": "turn_translation_cm", "t_y_cm_median": "t_y_cm",
        "pitch_roll_deg_median": "pitch_roll_deg",
    }
    for cell in cells:
        identified = cell.get("status") == "UNIT_IDENTIFIED"
        nondegenerate = identified and bool(cell.get("unit", {}).get("non_degenerate"))
        task3_unit = cell.get("task3_unit", {})
        task3_operator = task3_unit.get("status") == "UNIT_IDENTIFIED" and bool(task3_unit.get("non_degenerate"))
        power_law = cell.get("power_law", {})
        power_available = power_law.get("status") == "FIT_AVAILABLE" and nondegenerate
        signed = _path_value(cell, "checkpoints.1.u_signed")
        report = cell["_hierarchy"] = {
            "acc": float(signed is not None and signed > 0),
            "cell_coverage": float(identified),
            "task3_operator_coverage": float(task3_operator),
            "power_law_coverage": float(power_available),
            "checkpoint_coverage": {
                str(dose): float(cell.get("checkpoints", {}).get(str(dose), {}).get("measurable") is True)
                for dose in DOSES
            },
        }
        if nondegenerate:
            value = _path_value(cell, "unit.u_a")
            report["u_abs"] = abs(value) if value is not None else None
            for target, source in (("gamma", "power_law.gamma"), ("r2", "power_law.r2"),
                                   ("r16", "power_law.residual_max_k"),
                                   ("e_rig_over_u", "rigidity.e_rig_over_u"),
                                   ("purity", "leakage.purity")):
                report[target] = _path_value(cell, source)
        if identified:
            for target in ("yaw_crosstalk_deg", "turn_translation_cm", "t_y_cm", "pitch_roll_deg"):
                report[target] = _path_value(cell, f"leakage.{target}")
    output: dict[str, Any] = {}
    for action in ACTIONS:
        group = [cell for cell in cells if cell.get("action") == action]
        summary = {
            "acc": _hierarchical(group, "_hierarchy.acc", reducer="mean", bootstrap_samples=bootstrap_samples),
            "cell_coverage": _hierarchical(group, "_hierarchy.cell_coverage", reducer="mean", bootstrap_samples=bootstrap_samples),
            "task3_operator_coverage": _hierarchical(group, "_hierarchy.task3_operator_coverage", reducer="mean", bootstrap_samples=bootstrap_samples),
            "power_law_coverage": _hierarchical(group, "_hierarchy.power_law_coverage", reducer="mean", bootstrap_samples=bootstrap_samples),
        }
        summary.update({name: _hierarchical(group, f"_hierarchy.{field}", bootstrap_samples=bootstrap_samples)
                        for name, field in fields.items()})
        summary["checkpoint_coverage"] = {
            str(dose): _hierarchical(group, f"_hierarchy.checkpoint_coverage.{dose}", reducer="mean",
                                     bootstrap_samples=bootstrap_samples)
            for dose in DOSES
        }
        output[action] = summary
    return output


def _task3_hierarchy(routes: list[dict[str, Any]], bootstrap_samples: int) -> dict[str, Any]:
    numeric = {
        "relation_translation_cm": "relation.error.translation_cm",
        "relation_yaw_deg": "relation.error.yaw_deg",
        "relation_d_n": "relation.error.d_n", "relation_score": "relation.error.score",
        "self_translation_cm": "self_consistency.mean.error.translation_cm",
        "self_yaw_deg": "self_consistency.mean.error.yaw_deg",
        "self_d_n": "self_consistency.mean.error.d_n", "self_score": "self_consistency.mean.error.score",
    }
    for route in routes:
        status = route.get("status")
        d_n = _path_value(route, "relation.error.d_n")
        report = route["_hierarchy"] = {
            "raw_measured": float(status == "MEASURED"),
            "measurability": float(status in {"MEASURED", "STATIC_SHORTCUT"}),
            "normalized": float(status == "MEASURED" and d_n is not None),
            "e2e": float(status == "MEASURED" and route.get("relation", {}).get("error", {}).get("pass") is True),
            "conditional_pass": (float(route.get("relation", {}).get("error", {}).get("pass") is True)
                                 if status == "MEASURED" and d_n is not None else None),
        }
        if status == "MEASURED":
            report.update({name: _path_value(route, path) for name, path in numeric.items()})
    output: dict[str, Any] = {}
    for family in ("inverse", "order"):
        group = [route for route in routes if route.get("relation_family") == family]
        summary = {name: _hierarchical(group, f"_hierarchy.{name}", reducer="mean", bootstrap_samples=bootstrap_samples)
                   for name in ("raw_measured", "measurability", "normalized", "e2e", "conditional_pass")}
        summary.update({name: _hierarchical(group, f"_hierarchy.{name}", bootstrap_samples=bootstrap_samples)
                        for name in numeric})
        output[family] = summary
    return output


def _task3_seed_results(routes: list[dict[str, Any]], bootstrap_samples: int) -> tuple[dict[str, Any], dict[str, Any]]:
    by_seed = {str(seed): _task3_hierarchy([route for route in routes if int(route.get("seed")) == seed], bootstrap_samples)
               for seed in sorted({int(route.get("seed")) for route in routes})}
    aggregate: dict[str, Any] = {}
    for family in ("inverse", "order"):
        aggregate[family] = {}
        for metric in ("relation_translation_cm", "relation_yaw_deg", "self_translation_cm", "self_yaw_deg",
                       "relation_d_n", "relation_score", "measurability", "normalized", "e2e"):
            values = [seed[family][metric]["value"] for seed in by_seed.values()
                      if seed[family][metric].get("value") is not None]
            aggregate[family][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) >= 2 else None,
                "n_seeds": len(values),
            }
    return by_seed, aggregate


def _evaluate_task1(rows: list[dict[str, Any]], poses: Mapping[str, Path], output_root: Path,
                    args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        pose = poses.get(str(row["record_id"]))
        if row.get("status") != READY:
            cases.append({"status": "GENERATION_FAILED", "record_id": row["record_id"], "scene_id": row["primary_key"]["scene_id"],
                          "case_id": row["primary_key"]["case_id"], "state_id": row["state_id"], "seed": row["seed"]})
        elif pose is None:
            cases.append({"status": "POSE_PIPELINE_FAILED", "record_id": row["record_id"], "scene_id": row["primary_key"]["scene_id"],
                          "case_id": row["primary_key"]["case_id"], "state_id": row["state_id"], "seed": row["seed"]})
        else:
            try:
                result = _case_result(pose, None, args.min_pose_inlier_ratio, args.min_pose_anchor_coverage)
                result["record_id"] = row["record_id"]
                result["row"]["record_id"] = row["record_id"]
                result["row"]["route_id"] = row["route_id"]
                result["row"]["scene_id"] = row["primary_key"]["scene_id"]
                cases.append(result)
            except Exception as error:
                cases.append({"status": "METRIC_ERROR", "record_id": row["record_id"], "scene_id": row["primary_key"]["scene_id"],
                              "case_id": row["primary_key"]["case_id"], "state_id": row["state_id"], "seed": row["seed"], "error": repr(error)})
    flat = [case["row"] for case in cases if case.get("status") == "ok"]
    metrics = {
        "c_pose": _hierarchical(flat, "c_pose", reducer="mean", bootstrap_samples=args.bootstrap_samples),
        "instrument_failure_rate": _hierarchical(flat, "instrument_failure_rate", reducer="mean", bootstrap_samples=args.bootstrap_samples),
    }
    for field in ("e_rig_ray_deg_mean", "e_rig_ray_deg_std", "e_rig_ray_deg_median", "e_rig_ray_deg_p90",
                  "e_rig_px_mean", "e_rig_px_std", "e_rig_px_normalized_mean"):
        metrics[field] = _hierarchical(flat, field, bootstrap_samples=args.bootstrap_samples)
    for label in ("near", "mid", "far"):
        metrics[f"{label}_e_rig_ray_deg_mean"] = _hierarchical(flat, f"{label}_e_rig_ray_deg_mean", bootstrap_samples=args.bootstrap_samples)
    summary = {"expected": len(rows), "successful": len(flat), "failed": len(rows) - len(flat),
               "rigidity_observed": sum(int(row.get("e_rig_ray_deg_n", 0)) > 0 for row in flat), "metrics": metrics}
    _write_json(output_root / "task1" / "task1.json", {"summary": summary, "cases": cases})
    _write_jsonl(output_root / "details" / "task1.items.jsonl", cases)
    _write_csv(output_root / "details" / "task1.items.csv", flat)
    return summary, cases


def _evaluate_task2(rows: list[dict[str, Any]], poses: Mapping[str, Path], output_root: Path,
                    args: argparse.Namespace, invocation: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cells: list[dict[str, Any]] = []
    for row in rows:
        action = str(row["route_id"]).split("_", 1)[-1]
        pose = poses.get(str(row["record_id"]))
        if row.get("status") != READY:
            cells.append(missing_cell(row, action, f"generation_status:{row.get('status')}"))
        elif pose is None:
            cells.append(missing_cell(row, action, "missing_pose_artifact"))
        else:
            try:
                cells.append(evaluate_cell(pose, row["_record"], row,
                                           min_pose_inlier_ratio=args.min_pose_inlier_ratio,
                                           min_pose_anchor_coverage=args.min_pose_anchor_coverage))
            except Exception as error:
                cells.append(missing_cell(row, action, f"evaluation_error:{type(error).__name__}:{error}"))
    profile = {"runner": "unified_full_evaluation_v1", "gpus": args.gpus,
               "workers_per_gpu": args.workers_per_gpu, "pose_artifacts": sum(cell.get("pose_path") is not None for cell in cells),
               "total_workers": len([gpu for gpu in args.gpus.split(",") if gpu.strip()]) * args.workers_per_gpu,
               "python": sys.executable, "checkpoint": str(args.checkpoint),
               "first_frame_max_l1": args.first_frame_max_l1,
               "expected_visibility_source": "all_pose_anchors_fallback",
               "bootstrap_samples": args.bootstrap_samples,
               "min_pose_inlier_ratio": args.min_pose_inlier_ratio,
               "min_pose_anchor_coverage": args.min_pose_anchor_coverage}
    hierarchy = _task2_hierarchy(cells, args.bootstrap_samples)
    document = build_task2_document(cells, pose_root=output_root / "artifacts", data_root=args.data_root,
                                    model=args.model, seed=None, task="task2", worker_profile=profile)
    document["hierarchical_model_action"] = {args.model: hierarchy}
    directory = output_root / "task2"; directory.mkdir(parents=True, exist_ok=True)
    write_task2_outputs(document, directory / "task2.json", directory / "task2.cells.csv",
                        directory / "task2.model_action.csv", directory / "TASK2_REPORT.md", invocation=invocation)
    _write_jsonl(output_root / "details" / "task2.items.jsonl", cells)
    return document, cells


def _generation_failed_pair(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> dict[str, Any]:
    key = row_a["primary_key"]
    return {"model": key["model"], "scene_id": key["scene_id"], "case_id": key["case_id"],
            "state_id": row_a["state_id"], "seed": key["seed"], "pair_id": row_a.get("pair_id"),
            "relation_family": "unknown", "route_family": str(row_a["route_id"]).split("_", 1)[0],
            "route_id_alpha": row_a["route_id"], "route_id_beta": row_b["route_id"],
            "status": "GENERATION_FAILED", "failure_reasons": [f"generation_status:{row_a.get('status')},{row_b.get('status')}"]}


def _evaluate_task3(complex_rows: list[dict[str, Any]], atomic_rows: list[dict[str, Any]], poses: Mapping[str, Path],
                    output_root: Path, args: argparse.Namespace, invocation: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    route_index = {tuple(row["primary_key"][key] for key in ("model", "scene_id", "case_id", "seed", "route_id")): poses[row["record_id"]]
                   for row in complex_rows if row["record_id"] in poses}
    atomic_index: dict[tuple[str, str, str, int, str], Path] = {}
    atomic_by_key: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for row in atomic_rows:
        key = row["primary_key"]; action = str(row["route_id"]).split("_", 1)[-1]
        identity = (key["model"], key["scene_id"], key["case_id"], key["seed"], action)
        atomic_by_key[identity] = row
        if row["record_id"] in poses:
            atomic_index[identity] = poses[row["record_id"]]
    cache: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    routes: list[dict[str, Any]] = []
    for row_a, row_b in pair_rows(complex_rows):
        if row_a.get("status") != READY or row_b.get("status") != READY:
            routes.append(_generation_failed_pair(row_a, row_b)); continue
        try:
            routes.append(evaluate_pair(
                row_a, row_b, data_root=args.data_root, route_pose_index=route_index,
                atomic_pose_index=atomic_index, atomic_rows=atomic_by_key, atomic_cache=cache,
                min_pose_inlier_ratio=args.min_pose_inlier_ratio,
                min_pose_anchor_coverage=args.min_pose_anchor_coverage,
                normalized_tau=args.normalized_pass_tau,
            ))
        except Exception as error:
            failed = _generation_failed_pair(row_a, row_b)
            failed.update({"status": "EVALUATION_ERROR", "failure_reasons": [f"{type(error).__name__}:{error}"]})
            routes.append(failed)
    profile = {"runner": "unified_full_evaluation_v1", "model": args.model, "gpus": args.gpus,
               "min_pose_inlier_ratio": args.min_pose_inlier_ratio,
               "min_pose_anchor_coverage": args.min_pose_anchor_coverage,
               "task3_operator_dose_max": int(FROZEN_METRIC_SETTINGS["task3_operator_dose_max"]),
               "normalized_pass_tau": args.normalized_pass_tau,
               "boundary_policy": "all_operator_boundaries_require_M_t"}
    hierarchy = _task3_hierarchy(routes, args.bootstrap_samples)
    seed_results, seed_aggregate = _task3_seed_results(routes, args.bootstrap_samples)
    document = build_task3_document(routes, data_root=args.data_root, task2_pose_root=output_root / "artifacts",
                                    task3_pose_root=output_root / "artifacts", worker_profile=profile)
    document["hierarchical_relation"] = hierarchy
    document["seed_results"] = seed_results
    document["seed_aggregate"] = seed_aggregate
    directory = output_root / "task3"
    write_task3_outputs(document, directory, invocation=invocation)
    _write_jsonl(output_root / "details" / "task3.items.jsonl", routes)
    return document, routes


def _fmt(value: Any, digits: int = 3, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number * 100:.{digits}f}%" if percent else f"{number:.{digits}f}"


def _fmt_ci(metric: Mapping[str, Any] | None, *, percent: bool = False) -> str:
    values = (metric or {}).get("ci95", [None, None])
    return f"[{_fmt(values[0], percent=percent)}, {_fmt(values[1], percent=percent)}]"


def _gpu_environment(gpus: str) -> tuple[str, str]:
    try:
        import torch
        devices = []
        for token in gpus.split(","):
            index = int(token.strip()); props = torch.cuda.get_device_properties(index)
            devices.append(f"{index}: {props.name} ({props.total_memory / 2**30:.0f} GiB)")
        return "; ".join(devices), f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}"
    except Exception:
        return gpus, "N/A"


def _metric_value(summary: Mapping[str, Any], name: str) -> Any:
    return summary.get("metrics", {}).get(name, {}).get("value")


def _threshold_scan(routes: list[dict[str, Any]], bootstrap_samples: int) -> dict[float, dict[str, Any]]:
    supported = [route for route in routes if route.get("status") != "INTERFACE_UNSUPPORTED"]
    output: dict[float, dict[str, Any]] = {}
    for tau in (0.25, 0.5, 1.0, 2.0):
        key = "tau_" + str(tau).replace(".", "_")
        for route in supported:
            value = _path_value(route, "relation.error.d_n")
            route.setdefault("_hierarchy", {})[key] = float(value is not None and value <= tau)
        output[tau] = _hierarchical(supported, f"_hierarchy.{key}", reducer="mean", bootstrap_samples=bootstrap_samples)
    return output


def _create_plots(output_root: Path, task1: Mapping[str, Any], cells: list[Mapping[str, Any]], routes: list[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root = output_root / "plots"; plot_root.mkdir(exist_ok=True)
    labels = ["near", "mid", "far"]
    values = [_metric_value(task1, f"{label}_e_rig_ray_deg_mean") for label in labels]
    fig, axis = plt.subplots(figsize=(6, 4)); axis.bar(labels, [np.nan if value is None else value for value in values])
    axis.set_ylabel("E_rig (deg)"); axis.set_title("Task 1 depth strata"); fig.tight_layout()
    fig.savefig(plot_root / "task1_depth_strata.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, action in zip(axes.flat, ACTIONS):
        group = [cell for cell in cells if cell.get("action") == action]
        medians=[]; coverage=[]
        for dose in DOSES:
            samples=[_path_value(cell, f"checkpoints.{dose}.u_signed") for cell in group]
            samples=[abs(value) for value in samples if value is not None and abs(value) > 0]
            medians.append(float(np.median(samples)) if samples else np.nan)
            coverage.append(sum(cell.get("checkpoints", {}).get(str(dose), {}).get("measurable") is True for cell in group) / max(len(group), 1))
        axis.loglog(DOSES, medians, marker="o", label="median |u(k)|")
        axis.loglog(DOSES, np.asarray(DOSES) * medians[0] if np.isfinite(medians[0]) else np.full(len(DOSES), np.nan), linestyle="--", label="slope=1")
        second=axis.twinx(); second.plot(DOSES, coverage, color="tab:orange", alpha=.55); second.set_ylim(0,1)
        axis.set_title(action); axis.set_xlabel("dose k"); axis.set_ylabel("cm/deg"); second.set_ylabel("coverage")
    axes.flat[0].legend(fontsize=7); fig.savefig(plot_root / "task2_power_coverage.png", dpi=150); plt.close(fig)

    series: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for route in routes:
        if route.get("status") != "MEASURED": continue
        for side in ("alpha", "beta"):
            for point in route.get("ablation", {}).get(side, []):
                if point.get("translation_cm") is not None and point.get("yaw_deg") is not None:
                    series[(side, int(point["units"]))].append((float(point["translation_cm"]), float(point["yaw_deg"])))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for index, (unit, ylabel) in enumerate(((0, "cumulative translation error (cm)"), (1, "cumulative yaw error (deg)"))):
        for side, color in (("alpha", "tab:red"), ("beta", "tab:blue")):
            xs=sorted(x for (owner,x) in series if owner==side); med=[]; low=[]; high=[]
            for x in xs:
                vals=[value[unit] for value in series[(side,x)]]; med.append(np.median(vals)); low.append(np.percentile(vals,25)); high.append(np.percentile(vals,75))
            if xs:
                axes[index].plot(xs,med,color=color,label=side); axes[index].fill_between(xs,low,high,color=color,alpha=.2)
        axes[index].set_xlabel("unit boundary n"); axes[index].set_ylabel(ylabel)
        if axes[index].lines:
            axes[index].legend()
    fig.savefig(plot_root / "task3_cumulative_error.png", dpi=150); plt.close(fig)


def render_report(model: str, data_root: Path, output_root: Path, gpus: str, pose_results: list[Mapping[str, Any]] | None = None,
                  task1: Mapping[str, Any] | None = None, task2: Mapping[str, Any] | None = None,
                  cells: list[Mapping[str, Any]] | None = None, task3: Mapping[str, Any] | None = None,
                  routes: list[Mapping[str, Any]] | None = None, invocation: str = "") -> str:
    if pose_results is None:
        return "\n".join(["# StrideWorld Three-Task Summary of Results", "", "> Status: pending computation. When the run finishes, this file will be replaced in place with numerical results.", "",
                          "|Model|Task 1|Task 2|Task 3|", "|---|---:|---:|---:|", f"|{model}|pending|pending|pending|", "",
                          "**Metric definitions for this table**", "", "- `Model`: the world model to be evaluated; `Task 1/2/3`: current computation status of the three tasks.", ""]) + "\n"
    cells = cells or []; routes = routes or []; task1 = task1 or {}; task2 = task2 or {}; task3 = task3 or {}
    pose_ok = sum(result.get("status") == "ok" for result in pose_results); pose_total = len(pose_results)
    t2_ok = sum(cell.get("status") == "UNIT_IDENTIFIED" for cell in cells)
    t2_task3_operator = sum(
        cell.get("task3_unit", {}).get("status") == "UNIT_IDENTIFIED"
        and cell.get("task3_unit", {}).get("non_degenerate") for cell in cells
    )
    t2_power = sum(
        cell.get("power_law", {}).get("status") == "FIT_AVAILABLE"
        and cell.get("unit", {}).get("non_degenerate") for cell in cells
    )
    route_counts = Counter(str(route.get("status")) for route in routes)
    t3_raw = route_counts["MEASURED"]
    t3_measurable = t3_raw + route_counts["STATIC_SHORTCUT"]
    t3_normalized = sum(_path_value(route, "relation.error.d_n") is not None for route in routes if route.get("status") == "MEASURED")
    t3_e2e = sum(route.get("status") == "MEASURED" and route.get("relation", {}).get("error", {}).get("pass") is True for route in routes)
    profile = task2.get("run_profile", {})
    gpu_environment, torch_environment = _gpu_environment(gpus)
    lines = ["# StrideWorld Three-Task Summary of Results", "", "## Evaluation Overview", "",
             "|Item|Result|", "|---|---|", f"|Model|`{model}`|", f"|Input|`{data_root}`|", f"|Output|`{output_root}`|",
             f"|GPU|`{gpu_environment}`|", f"|PyTorch / CUDA|`{torch_environment}`|", f"|Frozen config SHA256|`{FROZEN_METRIC_SETTINGS_SHA256}`|",
             f"|Pose computation succeeded/failed|{pose_ok} / {pose_total-pose_ok}|", f"|Pose computation success rate|{_fmt(pose_ok/max(pose_total,1), percent=True)}|",
             f"|Python|`{profile.get('python', sys.executable)}`|",
             f"|CoTracker checkpoint|`{profile.get('checkpoint', 'N/A')}`|",
             f"|Concurrency|per GPU: {profile.get('workers_per_gpu', 'N/A')} persistent worker(s)|",
             f"|Geometry gate|inlier≥{profile.get('min_pose_inlier_ratio', FROZEN_METRIC_SETTINGS['min_pose_inlier_ratio'])}, anchor coverage≥{profile.get('min_pose_anchor_coverage', FROZEN_METRIC_SETTINGS['min_pose_anchor_coverage'])}|",
             f"|expected visibility|`{profile.get('expected_visibility_source', 'all_pose_anchors_fallback')}` (fallback permitted by the spec)|",
             f"|First-frame L1 diagnostic gate|≤{profile.get('first_frame_max_l1', 'N/A')}; condition SHA and source_transform are the authoritative binding|",
             "", "**Metric definitions for this table**", "",
             "- `Model/Input/Output`: the evaluated model, the evaluation data entry point, and the results directory of this run.",
             "- `GPU/PyTorch/CUDA/Python/checkpoint/Concurrency`: the software, hardware, weights, and worker configuration actually used in this run.",
             "- `Frozen config SHA256`: digest that pins the metric configuration version; only identical digests indicate the same set of frozen parameters.",
             "- `Pose computation succeeded/failed, success rate`: number of videos with successfully generated pose artifacts, number of failures, and the success fraction over all inputs.",
             "- `Geometry gate`: minimum pose inlier ratio and anchor coverage required for a frame to count as measurable.",
             "- `expected visibility`: source of the expected-visible anchors; the `First-frame L1 diagnostic gate` is used to detect input misalignment.",
             "", "## Task 1: World Rigidity and Measurability", "",
             "|Metric|Unit|Result|Assessment|", "|---|---:|---:|---|",
             f"|C_pose (hierarchical macro-average)|%|{_fmt(_metric_value(task1,'c_pose'),percent=True)}|{'Reliably interpretable' if (_metric_value(task1,'c_pose') or 0)>=.5 else 'Below 50%; interpret geometry results with caution'}|",
             f"|Instrument failure rate|%|{_fmt(_metric_value(task1,'instrument_failure_rate'),percent=True)}|Lower is better|",
             f"|E_rig ray mean|deg|{_fmt(_metric_value(task1,'e_rig_ray_deg_mean'))}|Conditional rigidity error; lower is better|",
             f"|E_rig ray std|deg|{_fmt(_metric_value(task1,'e_rig_ray_deg_std'))}|Error dispersion|",
             f"|E_rig ray median|deg|{_fmt(_metric_value(task1,'e_rig_ray_deg_median'))}|Robust center|",
             f"|E_rig ray P90|deg|{_fmt(_metric_value(task1,'e_rig_ray_deg_p90'))}|Tail error|",
             f"|E_rig pixel mean|px|{_fmt(_metric_value(task1,'e_rig_px_mean'))}|Pixel-domain supplement|",
             f"|E_rig / image diagonal|%|{_fmt(_metric_value(task1,'e_rig_px_normalized_mean'),percent=True)}|Cross-resolution supplement|",
             f"|near / mid / far E_rig|deg|{_fmt(_metric_value(task1,'near_e_rig_ray_deg_mean'))} / {_fmt(_metric_value(task1,'mid_e_rig_ray_deg_mean'))} / {_fmt(_metric_value(task1,'far_e_rig_ray_deg_mean'))}|Depth strata|",
             f"|Videos computed successfully/failed|count|{task1.get('successful',0)} / {task1.get('failed',0)}|Fixed Task 1 roster|",
             f"|Videos with rigidity observations|count|{task1.get('rigidity_observed',0)}|At least one frame has ≥12 World points|", "",
             "**Metric definitions for this table**", "",
             "- `C_pose`: fraction of frames, excluding the first frame, that have a valid camera and meet both the pose inlier ratio and anchor coverage thresholds; higher is better.",
             "- `Instrument failure rate`: fraction of frames that fail the pose measurability gate above; lower is better.",
             "- `E_rig ray mean/std/median/P90`: mean, standard deviation, median, and 90th percentile of the ray-angle rigidity error of independent World anchors, in degrees; lower is better.",
             "- `E_rig pixel mean`: mean of the same rigidity error in the pixel domain; `E_rig / image diagonal` is the resolution-normalized ratio.",
             "- `near/mid/far E_rig`: mean ray-angle error within the near, mid, and far depth strata.",
             "- `Videos computed successfully/failed`: number of videos for which the Task 1 metric program succeeded or failed; `Videos with rigidity observations` requires at least one frame with no fewer than 12 valid World points.", "",
             "### Task 1 Scene-Clustered Bootstrap", "", "|Metric|95% CI|state / scene|", "|---|---:|---:|",
             f"|C_pose|{_fmt_ci(task1.get('metrics',{}).get('c_pose'),percent=True)}|{task1.get('metrics',{}).get('c_pose',{}).get('n_states',0)} / {task1.get('metrics',{}).get('c_pose',{}).get('n_scenes',0)}|",
             f"|E_rig ray mean (deg)|{_fmt_ci(task1.get('metrics',{}).get('e_rig_ray_deg_mean'))}|{task1.get('metrics',{}).get('e_rig_ray_deg_mean',{}).get('n_states',0)} / {task1.get('metrics',{}).get('e_rig_ray_deg_mean',{}).get('n_scenes',0)}|", "",
             "**Metric definitions for this table**", "",
             "- `95% CI`: 95% confidence interval of the hierarchical macro-average from a bootstrap clustered by scene, expressing cross-scene uncertainty.",
             "- `C_pose`: fraction of frames passing the pose measurability gate; `E_rig ray mean`: mean ray-angle rigidity error of independent World anchors.",
             "- `state / scene`: number of valid states and scenes that actually contribute to the interval estimate of this metric.", "",
             "![Task 1 depth strata](plots/task1_depth_strata.png)", "", "## Task 2: Unit-1 of the Six Basic Actions", "",
             "|Action|cells|unit@12|op@4|power@16|Acc|Purity (1) / turn translation (cm)|Unit-1|Unit|gamma|R²|r16|E_rig/\\|u\\| (deg/cm or deg/deg)|unit coverage|Assessment|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    action_map = task2.get("model_action", {}).get(model, {})
    hierarchy_map = task2.get("hierarchical_model_action", {}).get(model, {})
    direction_issues: list[str] = []
    coverage_extreme: list[str] = []
    coverage_low: list[str] = []
    for action in ACTIONS:
        value = action_map.get(action, {}); hierarchy = hierarchy_map.get(action, {}); rotation = action in ROTATION_ACTIONS
        metric = lambda name: hierarchy.get(name, {}).get("value", value.get(name))
        primary = metric("turn_translation_cm_median" if rotation else "purity_median")
        acc, coverage = metric("acc"), metric("cell_coverage")
        if acc is not None and acc < .9:
            direction_issues.append(action)
        if coverage is not None and coverage < .1:
            coverage_extreme.append(action)
        elif coverage is not None and coverage < .5:
            coverage_low.append(action)
        judgement = "Direction anomaly" if acc is not None and acc < .9 else ("Extremely low full coverage" if coverage is not None and coverage < .1 else ("Low full coverage" if coverage is not None and coverage < .5 else "Direction and coverage usable"))
        lines.append(f"|{action}|{value.get('n_cells',0)}|{value.get('n_identified',0)}|{value.get('n_task3_operator',0)}|{value.get('n_power_law',0)}|{_fmt(acc,percent=True)}|{_fmt(primary)}|{_fmt(metric('u_median'))}|{'deg/dose' if rotation else 'cm/dose'}|{_fmt(metric('gamma_median'))}|{_fmt(metric('r2_median'))}|{_fmt(metric('r16_median'),percent=True)}|{_fmt(metric('e_rig_over_u_median'))}|{_fmt(coverage,percent=True)}|{judgement}|")
    lines += ["", "**Metric definitions for this table**", "",
              "- `Action`: W/S/A/D are forward/backward/left/right translation, YL/YR are left/right turns; `cells` is the fixed denominator; `unit@12`, `op@4`, and `power@16` are the usable counts of the primary unit quantity, the Task 3 short-range operator, and the strict power-law fit, respectively.",
              "- `Acc`: direction accuracy of the signed displacement at dose 1, with unmeasurable cells counted in the denominator; it is not a classification accuracy.",
              "- `Purity (1)`: fraction of the total planar displacement at dose 1 that lies along the target direction; for rotation actions this column instead reports the translation distance produced while turning.",
              "- `Unit-1`: per-dose motion magnitude from the SE(2) log-average over k=1..12; units are cm/dose for translation and deg/dose for rotation.",
              "- `gamma/R²/r16`: dose power-law slope, coefficient of determination of the fit, and relative residual at dose 16, computed only when all of k=1..16 are measurable; gamma closer to 1, R² closer to 1, and lower r16 are better.",
              "- `E_rig/|u|`: rigidity angle error divided by the unit-action magnitude; deg/cm for translation and deg/deg for rotation; lower is better.",
              "- `coverage`: fraction of the fixed total of cells whose unit operator was successfully identified; `Assessment` gives a direct verdict based on Acc and coverage.", "",
              f"Task 2 unit@12 succeeded/failed: **{t2_ok}/{len(cells)-t2_ok}**, success rate **{_fmt(t2_ok/max(len(cells),1),percent=True)}**; Task 3 op@4 usable **{t2_task3_operator}/{len(cells)}**; strict power@16 usable **{t2_power}/{len(cells)}**. When Acc is below 90%, check direction consistency; unmeasurable cells are kept in the Acc and coverage denominators.", "",
              "### Task 2 Scene-Clustered Bootstrap of Hierarchical Statistics", "", "|Action|Acc 95% CI|Unit-1 95% CI|coverage 95% CI|Unit-1 valid state / scene|", "|---|---:|---:|---:|---:|"]
    for action in ACTIONS:
        hierarchy = hierarchy_map.get(action, {})
        unit = hierarchy.get("u_median", {})
        lines.append(f"|{action}|{_fmt_ci(hierarchy.get('acc'),percent=True)}|{_fmt_ci(unit)}|{_fmt_ci(hierarchy.get('cell_coverage'),percent=True)}|{unit.get('n_states',0)} / {unit.get('n_scenes',0)}|")
    lines += ["", "**Metric definitions for this table**", "",
              "- `Acc 95% CI`: scene-clustered bootstrap 95% confidence interval of the dose-1 direction accuracy.",
              "- `Unit-1 95% CI`: 95% confidence interval of the hierarchical median of the per-dose motion magnitude; cm/dose for translation, deg/dose for rotation.",
              "- `coverage 95% CI`: 95% confidence interval of the fraction of cells whose primary unit operator is fully identified over k=1..12; unmeasurable cells stay in the denominator.",
              "- `Unit-1 valid state / scene`: number of states and scenes with enough valid cells that actually contribute to the Unit-1 statistic.", "",
              "### Task 2 Out-of-Plane Leakage and Checkpoint Coverage", "", "|Action|\\|t_y\\| median (cm)|pitch-roll median (deg)|yaw crosstalk (deg)|turn translation (cm)|k=1 / k=8 / k=16 coverage|",
              "|---|---:|---:|---:|---:|---:|"]
    for action in ACTIONS:
        value=action_map.get(action,{}); hierarchy=hierarchy_map.get(action,{})
        metric=lambda name: hierarchy.get(name,{}).get("value",value.get(name))
        cp=hierarchy.get("checkpoint_coverage",{})
        lines.append(f"|{action}|{_fmt(metric('t_y_cm_median'))}|{_fmt(metric('pitch_roll_deg_median'))}|{_fmt(metric('yaw_crosstalk_deg_median'))}|{_fmt(metric('turn_translation_cm_median'))}|{_fmt(cp.get('1',{}).get('value'),percent=True)} / {_fmt(cp.get('8',{}).get('value'),percent=True)} / {_fmt(cp.get('16',{}).get('value'),percent=True)}|")
    lines += ["", "**Metric definitions for this table**", "",
              "- `|t_y| median`: median displacement leakage perpendicular to the evaluation plane at dose 1; `pitch-roll median` is the median combined pitch-and-roll angular leakage.",
              "- `yaw crosstalk`: extra yaw produced by translation actions; `turn translation` is the extra translation distance produced by rotation actions; lower is better for both.",
              "- `k=1/8/16 coverage`: fraction of cells whose checkpoint at that cumulative dose passes the pose measurability gate, showing dropout over long horizons.", "",
              "### Task 2 Across Seeds / Across States", "", "|Action|σ_MAD|Unit|CV_r|S_repeat|Direction-consistent state fraction|CV_x median|", "|---|---:|---|---:|---:|---:|---:|"]
    state_action=task2.get("state_action",{}); cross=task2.get("ablation",{}).get("cross_state",{}).get(model,{})
    for action in ACTIONS:
        state_values=[value for key,value in state_action.items() if key.startswith(model+"|") and key.endswith("|"+action)]
        def median(field):
            data=[float(value[field]) for value in state_values if value.get(field) is not None]; return float(np.median(data)) if data else None
        signs=[value.get("sign_consistency") for value in state_values if value.get("sign_consistency") is not None]
        cvs=[float(value["cv_x"]) for key,value in cross.items() if key.endswith("|"+action) and value.get("cv_x") is not None]
        lines.append(f"|{action}|{_fmt(median('sigma_mad'))}|{'deg' if action in ROTATION_ACTIONS else 'cm'}|{_fmt(median('cv_r'))}|{_fmt(median('s_repeat'))}|{_fmt(sum(signs)/len(signs) if signs else None,percent=True)}|{_fmt(np.median(cvs) if cvs else None)}|")
    lines += ["", "**Metric definitions for this table**", "",
              "- `σ_MAD`: robust standard deviation of the unit action across seeds for the same state/action, i.e. `1.4826 × MAD`.",
              "- `CV_r`: `σ_MAD / |median unit action across seeds|`, the relative repeat variation; lower is more stable.",
              "- `S_repeat`: `exp(-CV_r)`, range 0–1; higher means better repeatability across seeds.",
              "- `Direction-consistent state fraction`: fraction of states in which the action sign is identical across all seeds.",
              "- `CV_x median`: median robust relative scale variation across states for a fixed seed/action; lower means the unit ruler depends less on the first frame or scene.", "",
              "NOOP drift: **N/A** (the preregistered index of this run has no NOOP cell; not filled with 0). `e_scale`: **N/A** (calibration state x0 is not frozen; only CV_x is reported).", "",
              "![Task 2 power law and coverage](plots/task2_power_coverage.png)", "", "## Task 3: Self-Consistency of Complex Routes", "",
              "|Relation family|Pairs|Raw measurable|Normalized measurable|D_rel_t median/P90 (cm)|D_rel_r median/P90 (deg)|D_self_t median (cm)|D_self_r median (deg)|d_n median (unit steps)|S median|Conditional pass rate|measurability|E2E|Assessment|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    summaries=task3.get("model_relation_summary",{})
    hierarchy3=task3.get("hierarchical_relation",{})
    for family in ("inverse","order"):
        value=summaries.get(f"{model}|{family}",{}); h=hierarchy3.get(family,{})
        normalized = int(value.get("n_normalized", 0))
        lines.append(f"|{family}|{value.get('n_pairs',0)}|{value.get('n_measured',0)}|{normalized}|{_fmt(h.get('relation_translation_cm',{}).get('value'))} / {_fmt(h.get('relation_translation_cm',{}).get('p90'))}|{_fmt(h.get('relation_yaw_deg',{}).get('value'))} / {_fmt(h.get('relation_yaw_deg',{}).get('p90'))}|{_fmt(h.get('self_translation_cm',{}).get('value'))}|{_fmt(h.get('self_yaw_deg',{}).get('value'))}|{_fmt(h.get('relation_d_n',{}).get('value'))}|{_fmt(h.get('relation_score',{}).get('value'))}|{_fmt(h.get('conditional_pass',{}).get('value'),percent=True)}|{_fmt(h.get('measurability',{}).get('value'),percent=True)}|{_fmt(h.get('e2e',{}).get('value'),percent=True)}|{'Normalization unavailable' if normalized == 0 else 'Judge by d_n/E2E'}|")
    lines += ["", "**Metric definitions for this table**", "",
              "- `Relation family`: inverse compares mutually inverse routes, order compares routes with reordered actions; `Pairs` is the total number of route pairs in the fixed denominator.",
              "- `Raw measurable/Normalized measurable`: respectively, the number of route pairs whose complex-route boundaries are all measurable and whose required k=1..4 atomic operators are available, and the number of route pairs that can also obtain the complete six-action k=1..4 scale.",
              "- `D_rel_t/D_rel_r`: translation/yaw error between the observed relative relation of the two routes and the relation predicted from atomic operators, as median and P90; lower is better.",
              "- `D_self_t/D_self_r`: median of the mean translation/yaw error between the observed endpoint of each route and the endpoint predicted by composing its own atomic operators; lower is better.",
              "- `d_n`: combined error after normalizing by the translation/rotation unit-1 estimated from k=1..4 for the same model and state/seed, measured in unit steps; `S=exp(-d_n)`, closer to 1 is better.",
              "- `Conditional pass rate`: fraction with `d_n≤0.5` among normalized routes only; `measurability` is the geometry-and-atomic measurable fraction; `E2E` uses all supported routes as the denominator and counts unmeasurable routes as failures.", "",
              "### Task 3 Scene-Clustered Bootstrap of Hierarchical Statistics", "", "|Relation family|D_rel_t 95% CI (cm)|D_rel_r 95% CI (deg)|measurability 95% CI|Normalized coverage 95% CI|E2E 95% CI|", "|---|---:|---:|---:|---:|---:|"]
    for family in ("inverse", "order"):
        h=hierarchy3.get(family,{})
        lines.append(f"|{family}|{_fmt_ci(h.get('relation_translation_cm'))}|{_fmt_ci(h.get('relation_yaw_deg'))}|{_fmt_ci(h.get('measurability'),percent=True)}|{_fmt_ci(h.get('normalized'),percent=True)}|{_fmt_ci(h.get('e2e'),percent=True)}|")
    lines += ["", "**Metric definitions for this table**", "",
              "- `D_rel_t/D_rel_r 95% CI`: scene-clustered bootstrap 95% confidence interval of the hierarchical median translation/yaw error between the observed relative relation and the relation predicted from atomic operators.",
              "- `measurability 95% CI`: interval of the fraction of routes for which both geometry and atomic operators are measurable.",
              "- `Normalized coverage 95% CI`: interval of the fraction of routes that also obtain the complete six-action k=1..4 ruler and can compute `d_n/S`.",
              "- `E2E 95% CI`: interval of the end-to-end success rate, with all supported routes as the denominator, for routes that are measurable, non-static-shortcut and with `d_n≤0.5`.", "",
              "### Task 3 Results by Seed", "", "|seed|Relation family|Pairs|Raw measurable|Normalized measurable|D_rel_t (cm)|D_rel_r (deg)|measurability|E2E|", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, families in task3.get("seed_results",{}).items():
        for family in ("inverse", "order"):
            h=families.get(family,{}); group=[route for route in routes if str(route.get('seed'))==str(seed) and route.get('relation_family')==family]
            raw=sum(route.get('status')=='MEASURED' for route in group); normalized=sum(_path_value(route,'relation.error.d_n') is not None for route in group if route.get('status')=='MEASURED')
            lines.append(f"|{seed}|{family}|{len(group)}|{raw}|{normalized}|{_fmt(h.get('relation_translation_cm',{}).get('value'))}|{_fmt(h.get('relation_yaw_deg',{}).get('value'))}|{_fmt(h.get('measurability',{}).get('value'),percent=True)}|{_fmt(h.get('e2e',{}).get('value'),percent=True)}|")
    lines += ["", "**Metric definitions for this table**", "",
              "- `seed`: generation random seed; `Relation family` distinguishes inverse (mutually inverse routes) from order (routes with reordered actions); `Pairs` is the fixed route-pair denominator for that seed.",
              "- `Raw measurable/Normalized measurable`: number of route pairs in that seed for which the raw cm/deg error can be computed, and for which the complete six-action k=1..4 ruler is further obtained.",
              "- `D_rel_t/D_rel_r`: hierarchical median translation/yaw error between the observed relative relation and the relation predicted from atomic operators, for that seed.",
              "- `measurability`: fraction of routes in that seed for which both geometry and atomics are measurable; `E2E` is the fraction, under the fixed denominator, of routes that are measurable, non-static and with `d_n≤0.5`."]
    seed_count=len(task3.get("seed_results",{}))
    lines += ["", "### Task 3 Cross-Seed Summary (mean ± std)", "", "|Relation family|D_rel_t (cm)|D_rel_r (deg)|D_self_t (cm)|D_self_r (deg)|d_n (unit steps)|S|measurability|E2E|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    seed_aggregate=task3.get("seed_aggregate",{})
    def mean_std(value, percent=False):
        return f"{_fmt(value.get('mean'),percent=percent)} ± {_fmt(value.get('std'),percent=percent)}"
    for family in ("inverse", "order"):
        value=seed_aggregate.get(family,{})
        lines.append(f"|{family}|{mean_std(value.get('relation_translation_cm',{}))}|{mean_std(value.get('relation_yaw_deg',{}))}|{mean_std(value.get('self_translation_cm',{}))}|{mean_std(value.get('self_yaw_deg',{}))}|{mean_std(value.get('relation_d_n',{}))}|{mean_std(value.get('relation_score',{}))}|{mean_std(value.get('measurability',{}),percent=True)}|{mean_std(value.get('e2e',{}),percent=True)}|")
    lines += ["", "**Metric definitions for this table**", "",
              "- Each cell is the `mean ± sample std` of the per-seed hierarchical statistics of the same relation family; with only one valid seed the std is N/A.",
              "- `D_rel_t/D_rel_r`: translation/yaw error between the observed and predicted relative relations; `D_self_t/D_self_r`: translation/yaw error between the observed endpoint of each side and its own predicted endpoint.",
              "- `d_n`: combined error in unit steps, normalized by the unit-1 of this model; `S=exp(-d_n)`, higher is better.",
              "- `measurability`: fraction of routes for which both geometry and atomics are measurable; `E2E`: end-to-end `d_n≤0.5` success rate with all supported routes as the denominator.", "",
              f"Task 3 seed coverage is **{seed_count}/3**; currently only seed 0, so the std is **N/A** (not replaced by 0).", ""]
    scan=_threshold_scan(routes, int(profile.get("bootstrap_samples", 2000)))
    task2_issues = []
    if direction_issues:
        task2_issues.append(f"{'/'.join(direction_issues)} (low direction accuracy)")
    if coverage_extreme:
        task2_issues.append(f"{'/'.join(coverage_extreme)} (extremely low full coverage)")
    if coverage_low:
        task2_issues.append(f"{'/'.join(coverage_low)} (low full coverage)")
    task2_conclusion = ", and ".join(task2_issues) if task2_issues else "none (direction and coverage are both usable)"
    task3_conclusion = (
        f"Task 3 has {t3_raw}/{len(routes)} raw errors and {t3_normalized}/{len(routes)} normalized errors usable for formal statistics"
        if t3_normalized else
        "Task 3 does not yet have the complete six-action k=1..4 scale, so only raw cm/deg errors can be interpreted"
    )
    lines += ["Task 3 status: " + "; ".join(f"`{key}`={value}" for key,value in sorted(route_counts.items())) + ".", "",
              "Threshold scan (fixed interface-supported denominator, state macro-average): " + ", ".join(f"τ={tau:g}: {_fmt(value.get('value'),percent=True)}" for tau,value in scan.items()) + ".", "",
              "![Task 3 cumulative error](plots/task3_cumulative_error.png)", "", "## Current Computation Success Rates", "", "|Stage|Succeeded/Total|Success rate|Failure meaning|", "|---|---:|---:|---|",
              f"|Pose pipeline|{pose_ok}/{pose_total}|{_fmt(pose_ok/max(pose_total,1),percent=True)}|Code, input, or pose recovery error|",
              f"|Task 1 video aggregation|{task1.get('successful',0)}/{task1.get('expected',0)}|{_fmt(task1.get('successful',0)/max(task1.get('expected',0),1),percent=True)}|Task 1 metric computation failed|",
              f"|Task 1 with rigidity observations|{task1.get('rigidity_observed',0)}/{task1.get('expected',0)}|{_fmt(task1.get('rigidity_observed',0)/max(task1.get('expected',0),1),percent=True)}|No frame passes the World-anchor gate|",
              f"|Task 2 primary unit quantity (k=12)|{t2_ok}/{len(cells)}|{_fmt(t2_ok/max(len(cells),1),percent=True)}|At least one of the first 12 checkpoints unmeasurable, or invalid rotation branch|",
              f"|Task 2 strict power law (k=16)|{t2_power}/{len(cells)}|{_fmt(t2_power/max(len(cells),1),percent=True)}|At least one of the 16 checkpoints unmeasurable, or invalid rotation branch|",
              f"|Task 3 short-range atomic operator (k=4)|{t2_task3_operator}/{len(cells)}|{_fmt(t2_task3_operator/max(len(cells),1),percent=True)}|First 4 checkpoints incomplete or motion below the resolution floor|",
              f"|Task 3 geometry+atomic measurable|{t3_measurable}/{len(routes)}|{_fmt(t3_measurable/max(len(routes),1),percent=True)}|Route boundaries or atomics required by the route unmeasurable|",
              f"|Task 3 raw metrics valid|{t3_raw}/{len(routes)}|{_fmt(t3_raw/max(len(routes),1),percent=True)}|Static shortcuts additionally excluded|",
              f"|Task 3 fully normalized|{t3_normalized}/{len(routes)}|{_fmt(t3_normalized/max(len(routes),1),percent=True)}|Missing complete four-translation + two-rotation unit-1 scale|",
              f"|Task 3 E2E pass|{t3_e2e}/{len(routes)}|{_fmt(t3_e2e/max(len(routes),1),percent=True)}|Success requires full normalization and d_n≤0.5|",
              "", "**Metric definitions for this table**", "",
              "- `Succeeded/Total`: number of items meeting the success condition of the stage, and the fixed evaluation denominator; `Success rate` is their ratio, and failed or unmeasurable items are never removed from the denominator.",
              "- `Pose pipeline` checks pose artifact generation; `Task 1 video aggregation/with rigidity observations` checks metric execution and independent World-anchor observations.",
              "- `Task 2 primary unit quantity/strict power law/Task 3 short-range atomic operator` require the first 12, 16, and 4 checkpoints to be measurable, respectively; the three are not interchangeable.",
              "- `Task 3 geometry+atomic measurable/raw metrics valid/fully normalized/E2E pass` successively add the conditions: all complex-route boundaries measurable, short-range operators required by the route, non-static-shortcut, complete six-action k=1..4 ruler, and `d_n≤0.5`.",
              "- `Failure meaning`: the main reason a stage did not succeed; it does not necessarily mean the program crashed.",
              "", f"Assessment: Task 1 pose coverage and rigidity measurement are reliable; the main Task 2 issue is {task2_conclusion}; {task3_conclusion}.",
              "", "## Per-Item Results", "",
              "- Pose: `details/pose_worker_results.jsonl`", "- Task 1: `details/task1.items.jsonl`, `details/task1.items.csv`",
              "- Task 2: `details/task2.items.jsonl`, `task2/task2.cells.csv`", "- Task 3: `details/task3.items.jsonl`, `task3/task3.routes.csv`",
              "", "## Reproduction Command", "", "```bash", invocation, "```", ""]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root or one model eval-index JSONL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--anchor-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cotracker3/scaled_offline.pth"))
    parser.add_argument("--cotracker-root", type=Path, default=Path("third_party/co-tracker"))
    parser.add_argument("--first-frame-max-l1", type=float, default=32.0,
                        help="Alignment sanity gate; hashes and source_transform remain authoritative")
    parser.add_argument("--max-valid-depth-cm", type=float, default=20000.0)
    parser.add_argument("--min-pose-inlier-ratio", type=float, default=float(FROZEN_METRIC_SETTINGS["min_pose_inlier_ratio"]))
    parser.add_argument("--min-pose-anchor-coverage", type=float, default=float(FROZEN_METRIC_SETTINGS["min_pose_anchor_coverage"]))
    parser.add_argument("--normalized-pass-tau", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--limit", type=int, help="Development smoke limit; omit for the fixed full denominator")
    parser.add_argument("--template-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers_per_gpu < 1 or args.anchor_workers < 1 or args.bootstrap_samples < 0:
        raise ValueError("worker counts must be positive and bootstrap samples non-negative")
    args.output_root.mkdir(parents=True, exist_ok=True)
    invocation = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    report_path = args.output_root / "StrideWorld_three_task_summary.md"
    if not report_path.exists() or args.template_only:
        report_path.write_text(render_report(args.model, args.data_root, args.output_root, args.gpus), encoding="utf-8")
    if args.template_only:
        print(report_path); return
    rows = load_eval_index(args.data_root, args.model)
    if args.limit:
        rows = rows[:args.limit]
    _write_jsonl(args.output_root / "details" / "normalized_index.jsonl", rows)
    anchors = prepare_anchors(rows, args.output_root, args.anchor_workers, args.max_valid_depth_cm)
    for row in rows:
        if row.get("status") == READY:
            row["_anchor_path"] = str(anchors[row["case_uid"]])
    pose_results = run_pose_stage(rows, args.output_root, args)
    poses = {str(result["record_id"]): Path(str(result["pose"])) for result in pose_results if result.get("status") == "ok"}
    atomic_rows = [row for row in rows if row["route_id"].startswith("atomic_")]
    complex_rows = [row for row in rows if not row["route_id"].startswith("atomic_")]
    task1_summary, task1_cases = _evaluate_task1(complex_rows, poses, args.output_root, args)
    task2_document, cells = _evaluate_task2(atomic_rows, poses, args.output_root, args, invocation)
    task3_document, routes = _evaluate_task3(complex_rows, atomic_rows, poses, args.output_root, args, invocation) if complex_rows else ({"model_relation_summary": {}}, [])
    _create_plots(args.output_root, task1_summary, cells, routes)
    report = render_report(args.model, args.data_root, args.output_root, args.gpus, pose_results, task1_summary,
                           task2_document, cells, task3_document, routes, invocation)
    report_path.write_text(report, encoding="utf-8")
    summary = {"model": args.model, "data_root": str(args.data_root), "output_root": str(args.output_root),
               "pose": {"expected": len(pose_results), "successful": sum(result.get("status") == "ok" for result in pose_results)},
               "task1": task1_summary, "task2": {
                   "expected": len(cells),
                   "identified": sum(cell.get("status") == "UNIT_IDENTIFIED" for cell in cells),
                   "unit_k12": sum(cell.get("status") == "UNIT_IDENTIFIED" for cell in cells),
                   "task3_operator_k4": sum(cell.get("task3_unit", {}).get("status") == "UNIT_IDENTIFIED" and cell.get("task3_unit", {}).get("non_degenerate") for cell in cells),
                   "power_law_k16": sum(cell.get("power_law", {}).get("status") == "FIT_AVAILABLE" and cell.get("unit", {}).get("non_degenerate") for cell in cells),
               },
               "task3": {"expected": len(routes), "statuses": dict(Counter(str(route.get("status")) for route in routes)),
                         "normalized": sum(_path_value(route, "relation.error.d_n") is not None
                                           for route in routes if route.get("status") == "MEASURED")},
               "report": str(report_path), "invocation": invocation}
    _write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
