#!/usr/bin/env python3
"""Track a selected Task 1 anchor artifact in an already-generated MP4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_COTRACKER_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "CoTracker"


def load_visible0_contract():
    path = Path(__file__).resolve().parents[1] / "metrics" / "task1" / "visible0_contract.py"
    spec = importlib.util.spec_from_file_location("strideworld_tracker_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load scoring-prefix contract: {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


VISIBLE0 = load_visible0_contract()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_video(path: Path, frame_limit: int) -> tuple[np.ndarray, float, int]:
    if frame_limit < 2:
        raise ValueError("--frame-limit must be at least 2")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames, decoded_count, shape = [], 0, None
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if shape is not None and rgb.shape != shape:
            raise RuntimeError("Video changes resolution between decoded frames")
        shape, decoded_count = rgb.shape, decoded_count + 1
        if len(frames) < frame_limit:
            frames.append(rgb)
    capture.release()
    if decoded_count < frame_limit:
        raise RuntimeError(
            f"Video has {decoded_count} frames, below frozen prefix {frame_limit}: {path}"
        )
    return np.stack(frames), fps, decoded_count


def source_to_target_affine(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    mode: str,
) -> np.ndarray:
    source_h, source_w = source_size
    target_h, target_w = target_size
    if mode == "stretch":
        return np.array(
            [[target_w / source_w, 0.0, 0.0], [0.0, target_h / source_h, 0.0]],
            dtype=np.float64,
        )
    if mode not in {"center_crop", "center_crop_floor"}:
        raise ValueError(f"Unsupported source transform: {mode}")
    scale = max(target_w / source_w, target_h / source_h)
    resized_w = int(round(source_w * scale))
    resized_h = int(round(source_h * scale))
    sx, sy = resized_w / source_w, resized_h / source_h
    if mode == "center_crop_floor":
        # Matrix-Game-3.5 cover resize: integer floor crop offsets (adapters/build_model_inputs.py matrix_cover_resize_pixels).
        offset_x = -float((resized_w - target_w) // 2)
        offset_y = -float((resized_h - target_h) // 2)
    else:
        offset_x = (target_w - resized_w) / 2.0
        offset_y = (target_h - resized_h) / 2.0
    return np.array([[sx, 0.0, offset_x], [0.0, sy, offset_y]], dtype=np.float64)


def apply_affine(uv: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return np.c_[uv.astype(np.float64), np.ones(len(uv))] @ affine.T


@torch.inference_mode()
def predict_with_probabilities(
    predictor: Any,
    video: torch.Tensor,
    query_uv: torch.Tensor,
    get_points_on_a_grid: Callable[..., torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror the official sparse predictor while retaining both probabilities."""
    batch, frames, channels, height, width = video.shape
    if batch != 1 or channels != 3:
        raise ValueError(f"Expected video [1,T,3,H,W], got {tuple(video.shape)}")
    interp_h, interp_w = predictor.interp_shape
    resized = F.interpolate(
        video.reshape(batch * frames, channels, height, width),
        (interp_h, interp_w),
        mode="bilinear",
        align_corners=True,
    ).reshape(batch, frames, channels, interp_h, interp_w)

    queries = torch.cat(
        [torch.zeros((batch, query_uv.shape[1], 1), device=video.device), query_uv],
        dim=-1,
    )
    queries_model = queries.clone()
    queries_model[:, :, 1:] *= queries_model.new_tensor(
        [(interp_w - 1) / (width - 1), (interp_h - 1) / (height - 1)]
    )
    support = get_points_on_a_grid(
        predictor.support_grid_size,
        predictor.interp_shape,
        device=video.device,
    )
    support = torch.cat([torch.zeros_like(support[:, :, :1]), support], dim=-1)
    all_queries = torch.cat([queries_model, support], dim=1)
    tracks, visibility_probability, tracking_confidence, _ = predictor.model.forward(
        video=resized,
        queries=all_queries,
        iters=6,
    )
    count = query_uv.shape[1]
    tracks = tracks[:, :, :count]
    visibility_probability = visibility_probability[:, :, :count]
    tracking_confidence = tracking_confidence[:, :, :count]
    tracks[:, 0] = queries_model[:, :, 1:]
    visibility_probability[:, 0] = 1.0
    tracking_confidence[:, 0] = 1.0
    tracks *= tracks.new_tensor(
        [(width - 1) / (interp_w - 1), (height - 1) / (interp_h - 1)]
    )
    return tracks, visibility_probability, tracking_confidence


def load_cotracker(cotracker_root: Path, checkpoint: Path):
    root = cotracker_root.resolve()
    source_marker = root / "cotracker" / "__init__.py"
    if not source_marker.is_file():
        raise FileNotFoundError(
            f"CoTracker source is missing {source_marker}; pass --cotracker-root"
        )
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CoTracker checkpoint does not exist: {checkpoint}")
    sys.path.insert(0, str(root))
    try:
        from cotracker.models.core.model_utils import get_points_on_a_grid
        from cotracker.predictor import CoTrackerPredictor
    except ImportError as exc:
        raise RuntimeError(f"Cannot import CoTracker from {root}") from exc
    return CoTrackerPredictor, get_points_on_a_grid, root, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--condition-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cotracker-root", type=Path, default=DEFAULT_COTRACKER_ROOT)
    parser.add_argument(
        "--source-transform",
        choices=["stretch", "center_crop", "center_crop_floor"],
        default="center_crop",
    )
    parser.add_argument("--visibility-threshold", type=float, default=0.9)
    parser.add_argument(
        "--first-frame-max-l1", type=float, default=5.0,
        help="Hard mean absolute RGB difference allowed after MP4 decode.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--frame-limit", type=int, required=True,
        help="Frozen evaluation-prefix length; tail frames are decoded for audit only.",
    )
    return parser.parse_args()


def configure_cuda(gpu: str) -> tuple[int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CoTracker3 Task 1 extraction requires CUDA")
    device_index = int(gpu)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(device_index)
    return device_index, device


def load_tracking_inputs(args: argparse.Namespace) -> dict[str, Any]:
    frames, fps, decoded_count = read_video(
        args.video.resolve(), args.frame_limit,
    )
    target_h, target_w = frames.shape[1:3]
    source = cv2.imread(str(args.condition_image.resolve()), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Cannot read condition image: {args.condition_image}")
    source_h, source_w = source.shape[:2]
    anchors = dict(np.load(args.anchors.resolve(), allow_pickle=False))
    anchor_id = np.asarray(anchors["anchor_id"], dtype=np.int64)
    source_uv = np.asarray(anchors["source_uv"], dtype=np.float64)
    affine = source_to_target_affine(
        (source_h, source_w), (target_h, target_w), args.source_transform
    )
    query_uv = apply_affine(source_uv, affine)
    return {
        "frames": frames, "fps": fps, "target_h": target_h, "target_w": target_w,
        "source": source, "source_h": source_h, "source_w": source_w,
        "anchors": anchors, "anchor_id": anchor_id, "affine": affine,
        "query_uv": query_uv, "transform_name": args.source_transform,
        "full_decoded_frame_count": decoded_count,
        "scoring_prefix_sha256": VISIBLE0.decoded_scoring_prefix_sha256(frames),
    }


def first_frame_metrics(
    args: argparse.Namespace, inputs: dict[str, Any],
) -> dict[str, float | str]:
    source_rgb = cv2.cvtColor(inputs["source"], cv2.COLOR_BGR2RGB)
    expected_first = cv2.warpAffine(
        source_rgb,
        inputs["affine"].astype(np.float32),
        (inputs["target_w"], inputs["target_h"]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )
    first_diff = inputs["frames"][0].astype(np.float32) - expected_first.astype(np.float32)
    first_mse = float(np.mean(first_diff * first_diff))
    first_psnr = 99.0 if first_mse == 0 else float(10.0 * np.log10(255.0**2 / first_mse))
    first_l1 = float(np.mean(np.abs(first_diff)))
    first_max_abs = float(np.max(np.abs(first_diff), initial=0.0))
    if first_l1 > args.first_frame_max_l1:
        raise RuntimeError(
            f"Condition/video first-frame match failed: mean_l1={first_l1:.4f} "
            f"> {args.first_frame_max_l1:.4f}"
        )
    condition_sha256 = file_sha256(args.condition_image.resolve())
    return {
        "first_psnr": first_psnr, "first_l1": first_l1,
        "first_max_abs": first_max_abs, "condition_sha256": condition_sha256,
    }


def validate_anchor_condition(inputs: dict[str, Any], condition_sha256: str) -> None:
    anchors = inputs["anchors"]
    if "condition_image_sha256" not in anchors:
        raise RuntimeError(
            "Selected anchor artifact is missing condition_image_sha256; "
            "rerun the GT preparer"
        )
    declared = str(np.asarray(anchors["condition_image_sha256"]).item()).lower()
    if declared != condition_sha256:
        raise RuntimeError("Selected anchor artifact condition_image_sha256 mismatch")


def run_model(
    args: argparse.Namespace, inputs: dict[str, Any], device: torch.device,
    checkpoint: Path, CoTrackerPredictor, get_points_on_a_grid,
) -> dict[str, np.ndarray]:
    frames, query_uv = inputs["frames"], inputs["query_uv"]
    inside_source_mapping = (
        (query_uv[:, 0] >= 0)
        & (query_uv[:, 0] <= inputs["target_w"] - 1)
        & (query_uv[:, 1] >= 0)
        & (query_uv[:, 1] <= inputs["target_h"] - 1)
    )
    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    query_tensor = torch.from_numpy(query_uv.astype(np.float32))[None].to(device)
    predictor = CoTrackerPredictor(
        checkpoint=str(checkpoint),
        offline=True,
        # The scaled offline checkpoint stores a 60-frame time embedding; the
        # model interpolates it internally when T differs from 60.
        window_len=60,
    ).to(device).eval()
    tracks, visibility_probability, tracking_confidence = predict_with_probabilities(
        predictor, video_tensor, query_tensor, get_points_on_a_grid
    )
    uv = tracks[0].float().cpu().numpy()
    vis_prob = visibility_probability[0].float().cpu().numpy()
    confidence = tracking_confidence[0].float().cpu().numpy()
    in_frame = (
        (uv[..., 0] >= 0)
        & (uv[..., 0] <= inputs["target_w"] - 1)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] <= inputs["target_h"] - 1)
    )
    valid = (
        (vis_prob >= args.visibility_threshold)
        & in_frame
        & inside_source_mapping[None]
        & np.isfinite(uv).all(axis=-1)
    )
    return {
        "uv": uv, "visibility_probability": vis_prob,
        "confidence": confidence, "valid": valid,
        "inside_source_mapping": inside_source_mapping,
    }


def save_tracks(
    args: argparse.Namespace, inputs: dict[str, Any], prediction: dict[str, np.ndarray],
    first: dict[str, float | str], checkpoint: Path, cotracker_root: Path,
    elapsed_sec: float, gpu_peak_memory_mib: float,
) -> None:
    frames = inputs["frames"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        anchor_id=inputs["anchor_id"], uv=prediction["uv"].astype(np.float32),
        valid=prediction["valid"], confidence=prediction["confidence"].astype(np.float32),
        visibility_probability=prediction["visibility_probability"].astype(np.float32),
        frame_ids=np.arange(len(frames)).astype(str),
        image_size=np.array([inputs["target_h"], inputs["target_w"]], dtype=np.int32),
        source_image_size=np.array([inputs["source_h"], inputs["source_w"]], dtype=np.int32),
        source_uv_track=inputs["query_uv"].astype(np.float32),
        source_to_track_affine=inputs["affine"],
        tracker=np.array("CoTracker3 scaled_offline"),
        anchors_sha256=np.array(file_sha256(args.anchors.resolve())),
        tracker_checkpoint_sha256=np.array(file_sha256(checkpoint)),
        cotracker_source_root=np.array(str(cotracker_root)),
        video_sha256=np.array(file_sha256(args.video.resolve())),
        generated_video_sha256=np.array(file_sha256(args.video.resolve())),
        generated_video_decoded_frame_count=np.array(
            inputs["full_decoded_frame_count"], dtype=np.int64
        ),
        generated_video_fps=np.array(inputs["fps"], dtype=np.float32),
        generated_video_image_size=np.array([inputs["target_h"], inputs["target_w"]], dtype=np.int32),
        tracked_prefix_frame_count=np.array(len(frames), dtype=np.int64),
        scoring_prefix_sha256=np.array(inputs["scoring_prefix_sha256"]),
        scoring_prefix_hash_version=np.array(VISIBLE0.SCORING_PREFIX_HASH_VERSION),
        first_frame_sha256=np.array(first["condition_sha256"]),
        condition_image_sha256=np.array(first["condition_sha256"]),
        source_transform=np.array(inputs["transform_name"]),
        fps=np.array(inputs["fps"], dtype=np.float32),
        visibility_threshold=np.array(args.visibility_threshold, dtype=np.float32),
        first_frame_psnr=np.array(first["first_psnr"], dtype=np.float32),
        first_frame_l1=np.array(first["first_l1"], dtype=np.float32),
        first_frame_max_abs=np.array(first["first_max_abs"], dtype=np.float32),
        first_frame_match_pass=np.array(True),
        runtime_sec=np.array(elapsed_sec, dtype=np.float32),
        gpu_peak_memory_mib=np.array(gpu_peak_memory_mib, dtype=np.float32),
    )


def output_summary(
    args: argparse.Namespace, inputs: dict[str, Any], prediction: dict[str, np.ndarray],
    first: dict[str, float | str], elapsed_sec: float, gpu_peak_memory_mib: float,
) -> dict[str, Any]:
    valid, confidence = prediction["valid"], prediction["confidence"]
    return {
        "output": str(args.output),
        "tracked_prefix_frames": len(inputs["frames"]),
        "full_decoded_frames": inputs["full_decoded_frame_count"],
        "scoring_prefix_sha256": inputs["scoring_prefix_sha256"],
        "anchors": len(inputs["anchor_id"]),
        "anchors_sha256": file_sha256(args.anchors.resolve()),
        "image_size": [inputs["target_h"], inputs["target_w"]],
        "source_transform": inputs["transform_name"],
        "source_to_track_affine": inputs["affine"].tolist(),
        "source_points_inside": int(prediction["inside_source_mapping"].sum()),
        "mean_valid_fraction": float(valid.mean()),
        "last_frame_valid_fraction": float(valid[-1].mean()),
        "mean_tracking_confidence": float(confidence.mean()),
        "first_frame_psnr": first["first_psnr"],
        "first_frame_l1": first["first_l1"],
        "first_frame_max_abs": first["first_max_abs"],
        "runtime_sec": elapsed_sec,
        "gpu_peak_memory_mib": gpu_peak_memory_mib,
    }


def main() -> None:
    args = parse_args()
    CoTrackerPredictor, get_points_on_a_grid, cotracker_root, checkpoint = (
        load_cotracker(args.cotracker_root, args.checkpoint)
    )
    device_index, device = configure_cuda(args.gpu)
    started_at = time.perf_counter()
    inputs = load_tracking_inputs(args)
    first = first_frame_metrics(args, inputs)
    validate_anchor_condition(inputs, str(first["condition_sha256"]))
    prediction = run_model(
        args, inputs, device, checkpoint, CoTrackerPredictor, get_points_on_a_grid
    )
    elapsed_sec = time.perf_counter() - started_at
    gpu_peak_memory_mib = torch.cuda.max_memory_allocated(device_index) / (1024**2)
    save_tracks(
        args, inputs, prediction, first, checkpoint, cotracker_root,
        elapsed_sec, gpu_peak_memory_mib,
    )
    summary = output_summary(
        args, inputs, prediction, first, elapsed_sec, gpu_peak_memory_mib
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
