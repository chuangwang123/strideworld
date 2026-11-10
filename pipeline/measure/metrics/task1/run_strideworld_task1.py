#!/usr/bin/env python3
"""Run the complete candidate GT -> CoTracker -> PnP -> Task-1 chain.

The command consumes one ready record (or a filtered set of records),
creates the handoff document's frozen reference anchors, tracks every anchor,
recovers the camera trajectory, and writes all intermediate artifacts.  A
candidate depth contract is preserved as candidate-only metadata; use
``--allow-candidate-depth`` only for diagnostics, never as evidence that the
UE export depth has been formally certified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.tools.gt import build_candidate_gt, load_camera  # noqa: E402
from metrics.tools.pose import (  # noqa: E402
    MIN_POSE_INLIER_RATIO,
    load_anchor_npz,
    load_track_npz,
    recover_pose_trajectory,
    run_cotracker,
    save_pose_npz,
)


def ready_records(data_root: Path) -> list[dict[str, Any]]:
    index = data_root / "common" / "index" / "by_host" / "eval.ready.jsonl"
    if not index.exists():
        raise FileNotFoundError(index)
    return [json.loads(line) for line in index.read_text().splitlines() if line.strip()]


def select_record(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    selected = []
    for row in rows:
        key = row["primary_key"]
        if args.model and key.get("model") != args.model:
            continue
        if args.scene and key.get("scene_id") != args.scene:
            continue
        if args.case and key.get("case_id") != args.case:
            continue
        if args.seed is not None and int(key.get("seed")) != args.seed:
            continue
        if args.route and key.get("route_id") != args.route:
            continue
        if args.task and key.get("task") != args.task:
            continue
        selected.append(row)
    if args.limit:
        selected = selected[:args.limit]
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= shard-index < shard-count")
    selected = selected[args.shard_index::args.shard_count]
    if not selected:
        raise ValueError("ready index filter selected no records")
    return selected[0] if args.single else {"records": selected}


def source_paths(data_root: Path, row: dict[str, Any]) -> dict[str, Path]:
    key = row["primary_key"]
    scene, case = key["scene_id"], key["case_id"]
    sidecar = data_root / "assets" / "eval_sidecars" / scene / case
    condition = data_root / "assets" / "images" / scene / case / "condition.png"
    record = data_root / "common" / row["record_relpath"]
    video = data_root / "common" / row["video_relpath"]
    return {
        "condition": condition,
        "depth": sidecar / "depth_m.candidate.npy",
        "depth_encoding": sidecar / "depth_encoding.json",
        "camera": sidecar / "camera.json",
        "record": record,
        "video": video,
    }


def source_to_target_affine(source_size: tuple[int, int], target_size: tuple[int, int], mode: str = "center_crop") -> np.ndarray:
    source_h, source_w = source_size
    target_h, target_w = target_size
    if mode == "stretch":
        return np.array([[target_w / source_w, 0.0, 0.0], [0.0, target_h / source_h, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    scale = max(target_w / source_w, target_h / source_h)
    resized_w, resized_h = int(round(source_w * scale)), int(round(source_h * scale))
    if mode == "center_crop_floor":
        return np.array([[resized_w / source_w, 0.0, -float((resized_w - target_w) // 2)], [0.0, resized_h / source_h, -float((resized_h - target_h) // 2)], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.array([[resized_w / source_w, 0.0, (target_w - resized_w) / 2.0], [0.0, resized_h / source_h, (target_h - resized_h) / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def run_one(data_root: Path, output_root: Path, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    paths = source_paths(data_root, row)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing source artifacts: " + ", ".join(missing))
    key = row["primary_key"]
    name = "__".join([str(key.get("model")), str(key.get("scene_id")), str(key.get("case_id")), f"seed_{key.get('seed')}", str(key.get("route_id"))])
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    output = output_root / name
    output.mkdir(parents=True, exist_ok=False)
    gt_dir = output / "gt"
    contract = json.loads(paths["depth_encoding"].read_text())
    if str(contract.get("status", "")).startswith("CANDIDATE") and not args.allow_candidate_depth:
        raise RuntimeError("depth contract is candidate-only; pass --allow-candidate-depth for diagnostic artifacts")
    gt_manifest = build_candidate_gt(paths["condition"], paths["depth"], paths["camera"], gt_dir, depth_encoding=paths["depth_encoding"], allow_candidate=args.allow_candidate_depth, max_valid_depth_cm=args.max_valid_depth_cm)
    anchor_name = "anchors.npz" if (gt_dir / "anchors.npz").exists() else "anchors_candidate.npz"
    record = json.loads(paths["record"].read_text())
    frame_limit = int(record["timing"].get("scored_frame_end", record["timing"].get("frame_count", 0) - 1)) + 1
    tracks_path = output / "tracks.npz"
    # The lower-level runner keeps this gate strict by default.  v3
    # videos are lossy H.264, so expose the threshold explicitly while
    # retaining the measured value in the run manifest.
    run_cotracker(paths["video"], gt_dir / anchor_name, paths["condition"], args.checkpoint, tracks_path, script=args.cotracker_script, python=args.python, frame_limit=frame_limit, gpu=args.gpu, cotracker_root=args.cotracker_root, source_transform=args.source_transform, first_frame_max_l1=args.first_frame_max_l1)
    tracks = load_track_npz(tracks_path)
    anchor_data = load_anchor_npz(gt_dir / anchor_name)
    camera = load_camera(paths["camera"])
    target_h, target_w = tracks["image_size"]
    affine = source_to_target_affine(tuple(camera["image_size"]), (int(target_h), int(target_w)), args.source_transform)
    K_target = affine @ camera["K"]
    pose = recover_pose_trajectory(
        anchor_data,
        tracks,
        K_target,
        min_pose_inlier_ratio=args.min_pose_inlier_ratio,
    )
    pose_path = output / "pose.npz"
    pose_meta = {"model": key.get("model"), "scene_id": key.get("scene_id"), "case_id": key.get("case_id"), "seed": key.get("seed"), "route_id": key.get("route_id"), "state_id": f"{key.get('scene_id')}_{key.get('case_id')}_state0000", "record": str(paths["record"]), "video": str(paths["video"]), "gt_manifest": str(gt_dir / "gt_manifest.json"), "source_to_track_affine": affine.tolist(), "depth_contract_status": contract.get("status"), "formal_gt_eligible": gt_manifest["formal_eligible"], "min_pose_inlier_ratio": args.min_pose_inlier_ratio}
    save_pose_npz(pose_path, pose, pose_meta)
    (output / "run_manifest.json").write_text(json.dumps({"record": row, "paths": {key: str(value) for key, value in paths.items()}, "gt": gt_manifest, "frame_limit": frame_limit, "K_source": camera["K"].tolist(), "K_target": K_target.tolist(), "source_to_track_affine": affine.tolist(), "source_transform": args.source_transform, "first_frame_max_l1": args.first_frame_max_l1, "min_pose_inlier_ratio": args.min_pose_inlier_ratio, "pose": str(pose_path)}, indent=2, ensure_ascii=False))
    return {"output": str(output), "pose": str(pose_path), "tracks": str(tracks_path), "gt": gt_manifest, "frame_limit": frame_limit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cotracker-root", type=Path)
    parser.add_argument("--cotracker-script", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    # The v3 records currently decode as a 1920x1080 -> 1280x704
    # stretch (verified from the condition/video first frame).  Keep the
    # option explicit so another model with center-crop preprocessing cannot
    # silently reuse this transform.
    parser.add_argument("--source-transform", choices=["stretch", "center_crop"], default="stretch")
    parser.add_argument("--model")
    parser.add_argument("--scene")
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--route")
    parser.add_argument("--task")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Deterministic record shard index for parallel runs")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Number of deterministic record shards")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--allow-candidate-depth", action="store_true")
    parser.add_argument("--first-frame-max-l1", type=float, default=12.0)
    parser.add_argument("--max-valid-depth-cm", type=float, default=20000.0)
    parser.add_argument("--min-pose-inlier-ratio", type=float, default=MIN_POSE_INLIER_RATIO,
                        help="Task-1 M_t pose inlier-ratio gate (evaluation hyperparameter)")
    args = parser.parse_args()
    if args.shard_count < 1:
        raise ValueError("shard-count must be >= 1")
    selected = select_record(ready_records(args.data_root), args)
    rows = [selected] if args.single else selected["records"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        try:
            results.append(run_one(args.data_root, args.output_root, row, args))
        except Exception as error:
            results.append({"status": "error", "primary_key": row.get("primary_key"), "error": repr(error)})
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
