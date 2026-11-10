#!/usr/bin/env python3
"""Run the StrideWorld Task-1 chain with persistent CoTracker workers.

The historical runner starts a Python process (and reloads the checkpoint) for
each video.  This runner starts a fixed number of long-lived processes instead:
each process loads CoTracker once, consumes many videos, and only exits after
the complete queue is drained.  CPU-side GT generation, PnP, and artifact
serialization run in the same worker, so they are parallelised along with the
GPU inference without creating a second process per video.

The final JSON contains the aggregate Task-1 metrics and the complete per-video
case list produced by ``evaluate_task1_final.py``.  ``worker_results`` records
successful and failed pipeline jobs, including paths to all intermediate
artifacts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.task1.evaluate_task1_final import evaluate  # noqa: E402
from metrics.task1.run_strideworld_task1 import (  # noqa: E402
    ready_records,
    source_paths,
    source_to_target_affine,
)
from metrics.tools.gt import build_candidate_gt, load_camera  # noqa: E402
from metrics.tools.pose import (  # noqa: E402
    MIN_POSE_INLIER_RATIO,
    load_anchor_npz,
    load_track_npz,
    recover_pose_trajectory,
    save_pose_npz,
)


def _safe_name(row: dict[str, Any]) -> str:
    key = row["primary_key"]
    raw = "__".join(
        [
            str(key.get("model")),
            str(key.get("scene_id")),
            str(key.get("case_id")),
            f"seed_{key.get('seed')}",
            str(key.get("route_id")),
        ]
    )
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in raw)


def _select_records(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
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
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("ready index filter selected no records")
    return selected


class PersistentCoTracker:
    """One checkpoint-loaded CoTracker instance owned by one worker process."""

    def __init__(self, config: dict[str, Any], gpu: int):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Importing the existing script keeps its frozen video decoding,
        # affine mapping, and probability handling byte-for-byte compatible.
        self.runner = importlib.import_module("scripts.run_cotracker3_task1")
        self.torch = importlib.import_module("torch")
        self.gpu = int(gpu)
        self.torch.cuda.set_device(self.gpu)
        self.device = self.torch.device(f"cuda:{self.gpu}")
        self.torch.empty(1, device=self.device)
        self.torch.cuda.reset_peak_memory_stats(self.gpu)
        root = Path(config["cotracker_root"]).resolve()
        checkpoint = Path(config["checkpoint"]).resolve()
        predictor_cls, grid_fn, root, checkpoint = self.runner.load_cotracker(root, checkpoint)
        self.checkpoint = checkpoint
        self.cotracker_root = root
        self.grid_fn = grid_fn
        self.predictor = predictor_cls(
            checkpoint=str(checkpoint), offline=True, window_len=60
        ).to(self.device).eval()
        self.visibility_threshold = float(config["visibility_threshold"])
        self.source_transform = str(config["source_transform"])
        self.first_frame_max_l1 = float(config["first_frame_max_l1"])

    def track(
        self, video: Path, anchors: Path, condition: Path, output: Path,
        source_transform: str | None = None,
    ) -> dict[str, Any]:
        args = SimpleNamespace(
            video=video,
            anchors=anchors,
            condition_image=condition,
            output=output,
            source_transform=source_transform or self.source_transform,
            visibility_threshold=self.visibility_threshold,
            first_frame_max_l1=self.first_frame_max_l1,
            frame_limit=int(self.frame_limit),
        )
        started = time.perf_counter()
        inputs = self.runner.load_tracking_inputs(args)
        first = self.runner.first_frame_metrics(args, inputs)
        self.runner.validate_anchor_condition(inputs, str(first["condition_sha256"]))
        frames = inputs["frames"]
        query_uv = inputs["query_uv"]
        inside_source_mapping = (
            (query_uv[:, 0] >= 0)
            & (query_uv[:, 0] <= inputs["target_w"] - 1)
            & (query_uv[:, 1] >= 0)
            & (query_uv[:, 1] <= inputs["target_h"] - 1)
        )
        torch = self.torch
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(self.device)
        query_tensor = torch.from_numpy(query_uv.astype(np.float32))[None].to(self.device)
        with torch.inference_mode():
            tracks, visibility, confidence = self.runner.predict_with_probabilities(
                self.predictor, video_tensor, query_tensor, self.grid_fn
            )
        uv = tracks[0].float().cpu().numpy()
        vis_prob = visibility[0].float().cpu().numpy()
        conf = confidence[0].float().cpu().numpy()
        in_frame = (
            (uv[..., 0] >= 0) & (uv[..., 0] <= inputs["target_w"] - 1)
            & (uv[..., 1] >= 0) & (uv[..., 1] <= inputs["target_h"] - 1)
        )
        prediction = {
            "uv": uv,
            "visibility_probability": vis_prob,
            "confidence": conf,
            "valid": (vis_prob >= self.visibility_threshold) & in_frame
            & inside_source_mapping[None] & np.isfinite(uv).all(axis=-1),
            "inside_source_mapping": inside_source_mapping,
        }
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(self.gpu) / (1024**2)
        self.runner.save_tracks(
            args, inputs, prediction, first, self.checkpoint, self.cotracker_root,
            elapsed, peak,
        )
        del video_tensor, query_tensor, tracks, visibility, confidence
        # Release per-video tensors while keeping the predictor/checkpoint
        # resident in this worker.  This avoids accumulation across 80–100
        # videos assigned to one worker.
        torch.cuda.empty_cache()
        return self.runner.output_summary(args, inputs, prediction, first, elapsed, peak)

    def set_frame_limit(self, frame_limit: int) -> None:
        self.frame_limit = int(frame_limit)


def _run_one(task: dict[str, Any], tracker: PersistentCoTracker, config: dict[str, Any]) -> dict[str, Any]:
    data_root = Path(config["data_root"])
    output_root = Path(config["output_root"])
    row = task["row"]
    paths = source_paths(data_root, row)
    # Most UE5 sidecars are named ``depth_m.candidate.npy``; the replica
    # apartment subset already has the confirmed-style ``depth_m.npy`` name.
    # Treat both as the same input field and preserve the contract status from
    # depth_encoding.json.
    if not paths["depth"].exists():
        fallback_depth = paths["depth"].with_name("depth_m.npy")
        if fallback_depth.exists():
            paths["depth"] = fallback_depth
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing source artifacts: " + ", ".join(missing))
    output = output_root / _safe_name(row)
    output.mkdir(parents=True, exist_ok=False)
    gt_dir = output / "gt"
    contract = json.loads(paths["depth_encoding"].read_text())
    if str(contract.get("status", "")).startswith("CANDIDATE") and not config["allow_candidate_depth"]:
        raise RuntimeError("depth contract is candidate-only; pass --allow-candidate-depth")
    gt_manifest = build_candidate_gt(
        paths["condition"], paths["depth"], paths["camera"], gt_dir,
        depth_encoding=paths["depth_encoding"], allow_candidate=config["allow_candidate_depth"],
        max_valid_depth_cm=float(config["max_valid_depth_cm"]),
    )
    anchor_name = "anchors.npz" if (gt_dir / "anchors.npz").exists() else "anchors_candidate.npz"
    record = json.loads(paths["record"].read_text())
    frame_limit = int(record["timing"].get("scored_frame_end", record["timing"].get("frame_count", 0) - 1)) + 1
    tracker.set_frame_limit(frame_limit)
    tracks_path = output / "tracks.npz"
    tracker_summary = tracker.track(paths["video"], gt_dir / anchor_name, paths["condition"], tracks_path)
    tracks = load_track_npz(tracks_path)
    anchors = load_anchor_npz(gt_dir / anchor_name)
    camera = load_camera(paths["camera"])
    target_h, target_w = tracks["image_size"]
    affine = source_to_target_affine(tuple(camera["image_size"]), (int(target_h), int(target_w)), config["source_transform"])
    K_target = affine @ camera["K"]
    pose = recover_pose_trajectory(
        anchors, tracks, K_target,
        min_pose_inlier_ratio=float(config["min_pose_inlier_ratio"]),
    )
    pose_path = output / "pose.npz"
    key = row["primary_key"]
    metadata = {
        "model": key.get("model"), "scene_id": key.get("scene_id"),
        "case_id": key.get("case_id"), "seed": key.get("seed"),
        "route_id": key.get("route_id"),
        "state_id": f"{key.get('scene_id')}_{key.get('case_id')}_state0000",
        "record": str(paths["record"]), "video": str(paths["video"]),
        "gt_manifest": str(gt_dir / "gt_manifest.json"),
        "source_to_track_affine": affine.tolist(),
        "depth_contract_status": contract.get("status"),
        "formal_gt_eligible": gt_manifest["formal_eligible"],
        "min_pose_inlier_ratio": float(config["min_pose_inlier_ratio"]),
    }
    save_pose_npz(pose_path, pose, metadata)
    manifest = {
        "record": row,
        "paths": {key: str(value) for key, value in paths.items()},
        "gt": gt_manifest,
        "tracker": tracker_summary,
        "frame_limit": frame_limit,
        "K_source": camera["K"].tolist(), "K_target": K_target.tolist(),
        "source_to_track_affine": affine.tolist(),
        "source_transform": config["source_transform"],
        "first_frame_max_l1": config["first_frame_max_l1"],
        "min_pose_inlier_ratio": config["min_pose_inlier_ratio"],
        "pose": str(pose_path),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return {
        "status": "ok", "model": key.get("model"), "scene_id": key.get("scene_id"),
        "case_id": key.get("case_id"), "seed": key.get("seed"), "route_id": key.get("route_id"),
        "output": str(output), "tracks": str(tracks_path), "pose": str(pose_path),
        "gt": str(gt_dir / "gt_manifest.json"), "tracker": tracker_summary,
        "camera_valid_frames": int(np.asarray(pose["camera_valid"], dtype=bool).sum()),
        "measurable_frames_excluding_t0": int(np.asarray(pose["measurable"], dtype=bool)[1:].sum()),
        "n_frames": int(len(pose["camera_valid"])),
    }


def _worker_main(gpu: int, config: dict[str, Any], tasks: Any, results: Any) -> None:
    try:
        tracker = PersistentCoTracker(config, gpu)
    except Exception as error:
        results.put({"status": "worker_init_error", "gpu": gpu, "error": repr(error), "traceback": traceback.format_exc()})
        return
    while True:
        task = tasks.get()
        if task is None:
            return
        started = time.perf_counter()
        try:
            result = _run_one(task, tracker, config)
            result["gpu"] = gpu
            result["worker_elapsed_sec"] = time.perf_counter() - started
        except Exception as error:
            key = task["row"].get("primary_key", {})
            result = {
                "status": "error", "gpu": gpu,
                "model": key.get("model"), "scene_id": key.get("scene_id"),
                "case_id": key.get("case_id"), "seed": key.get("seed"),
                "route_id": key.get("route_id"), "error": repr(error),
                "traceback": traceback.format_exc(),
                "worker_elapsed_sec": time.perf_counter() - started,
            }
        results.put(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cotracker-root", required=True, type=Path)
    parser.add_argument("--gpus", default="6,7", help="Comma-separated GPU ids")
    parser.add_argument("--workers-per-gpu", type=int, default=5)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--task", default="task2")
    parser.add_argument("--scene")
    parser.add_argument("--case")
    parser.add_argument("--route")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-candidate-depth", action="store_true")
    parser.add_argument("--source-transform", choices=["stretch", "center_crop"], default="stretch")
    parser.add_argument("--first-frame-max-l1", type=float, default=12.0)
    parser.add_argument("--max-valid-depth-cm", type=float, default=20000.0)
    parser.add_argument("--min-pose-inlier-ratio", type=float, default=MIN_POSE_INLIER_RATIO)
    parser.add_argument("--visibility-threshold", type=float, default=0.90)
    args = parser.parse_args()
    gpus = [int(token.strip()) for token in args.gpus.split(",") if token.strip()]
    if not gpus or args.workers_per_gpu < 1:
        raise ValueError("gpus must be non-empty and workers-per-gpu must be >= 1")
    rows = _select_records(ready_records(args.data_root), args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "data_root": str(args.data_root), "output_root": str(args.output_root),
        "checkpoint": str(args.checkpoint), "cotracker_root": str(args.cotracker_root),
        "allow_candidate_depth": bool(args.allow_candidate_depth),
        "source_transform": args.source_transform,
        "first_frame_max_l1": float(args.first_frame_max_l1),
        "max_valid_depth_cm": float(args.max_valid_depth_cm),
        "min_pose_inlier_ratio": float(args.min_pose_inlier_ratio),
        "visibility_threshold": float(args.visibility_threshold),
    }
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=max(2 * len(gpus) * args.workers_per_gpu, 32))
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(target=_worker_main, args=(gpu, config, task_queue, result_queue), daemon=False)
        for gpu in gpus for _ in range(args.workers_per_gpu)
    ]
    for worker in workers:
        worker.start()
    for index, row in enumerate(rows):
        task_queue.put({"index": index, "row": row})
    for _ in workers:
        task_queue.put(None)
    results: list[dict[str, Any]] = []
    while len(results) < len(rows):
        result = result_queue.get()
        results.append(result)
        if len(results) % 10 == 0 or len(results) == len(rows):
            print(json.dumps({"completed": len(results), "total": len(rows)}, ensure_ascii=False), flush=True)
    for worker in workers:
        worker.join()
    results.sort(key=lambda item: (str(item.get("model")), str(item.get("scene_id")), str(item.get("case_id")), str(item.get("route_id"))))
    output_csv = args.output_root.with_suffix(".task1.csv")
    output_json = args.output_root.with_suffix(".task1.json")
    document = evaluate(
        args.output_root, output_csv, output_json,
        min_pose_inlier_ratio=float(args.min_pose_inlier_ratio),
    )
    document["run_profile"] = {
        "runner": "persistent_cotracker_workers_v1",
        "model": args.model, "seed": args.seed, "task": args.task,
        "requested_videos": len(rows), "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "total_workers": len(workers),
        "min_pose_inlier_ratio": float(args.min_pose_inlier_ratio),
        "allow_candidate_depth": bool(args.allow_candidate_depth),
    }
    document["worker_results"] = results
    document["worker_summary"] = {
        "ok": sum(item.get("status") == "ok" for item in results),
        "errors": sum(item.get("status") != "ok" for item in results),
    }
    output_json.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False, default=str))
    summary_path = args.output_root / "worker_results.json"
    summary_path.write_text(json.dumps({"run_profile": document["run_profile"], "worker_results": results, "worker_summary": document["worker_summary"]}, indent=2, ensure_ascii=False, allow_nan=False, default=str))
    print(json.dumps({"output_csv": str(output_csv), "output_json": str(output_json), "worker_results": str(summary_path), "worker_summary": document["worker_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
