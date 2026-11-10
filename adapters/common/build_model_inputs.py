#!/usr/bin/env python3
"""Build immutable, model-native inputs for the frozen StrideWorld campaign.

This adapter stage consumes only the campaign record, its frozen first image,
and its frozen prompt.  It does not read videos or future/ground-truth frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any, Iterable

import numpy as np

from strideworld_actions import canonical_record_sha, expand_record_actions, pixel_intrinsics, sample_c2w


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "manifests/source/hy15_native_campaign.json"
DEFAULT_PROFILES = ROOT / "adapters/common/model_profiles.json"
DEFAULT_DATA_ROOT = Path("<path>/data")
DEFAULT_OUTPUT = Path("<path>/adapter_inputs")
SOURCE_ROOTS = {
    "matrix_game_35_distilled_fp": ROOT / "sources/Matrix-Game-3.5",
    "solarwm_h3_stage2_fix": ROOT / "sources/SolarWM",
    "evoke_turbo_i2v": ROOT / "sources/Evoke",
}
MODEL_ORDER = tuple(SOURCE_ROOTS)
MATRIX_LITERAL_ANCHOR_SCHEMA = "matrix-literal-center-crop-v1"
MATRIX_COVER_ANCHOR_SCHEMA = "matrix-cover-resize-center-crop-v1"
MATRIX_ANCHOR_MODES = ("literal-center-crop", "cover-resize-center-crop")
_MATRIX_COVER_CACHE: dict[tuple[str, str, int, int], tuple[Path, str, dict[str, Any], np.ndarray]] = {}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == payload:
        return
    part = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        part.write_bytes(payload)
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_bytes(path, b"".join(canonical_bytes(row) + b"\n" for row in rows))


def save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with part.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        if path.is_file() and file_sha(path) == file_sha(part):
            part.unlink()
        else:
            part.replace(path)
    finally:
        part.unlink(missing_ok=True)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with part.open("wb") as handle:
            np.savez(handle, **arrays)
        if path.is_file() and file_sha(path) == file_sha(part):
            part.unlink()
        else:
            part.replace(path)
    finally:
        part.unlink(missing_ok=True)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    part = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(source, part)
        except OSError:
            shutil.copyfile(source, part)
        try:
            part.replace(target)
        except FileExistsError:
            pass
    finally:
        part.unlink(missing_ok=True)


def parse_image_uri(uri: str) -> str:
    prefix = "data://"
    if not isinstance(uri, str) or not uri.startswith(prefix):
        raise ValueError(f"unsupported condition image URI: {uri!r}")
    relative = uri[len(prefix):]
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"unsafe condition image URI: {uri!r}")
    return relative


def select_records(campaign: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    records = campaign["records"]
    if args.smoke:
        groups = []
        for domain in ("eth3d", "ue5"):
            match = next(
                group for group in campaign["groups"]
                if group["seed"] == 0 and group["scene_id"].startswith(domain)
            )
            groups.append(match["group_id"])
        wanted = set(groups)
        records = [record for record in records if record["group_id"] in wanted]
    if args.group:
        wanted = set(args.group)
        records = [record for record in records if record["group_id"] in wanted]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError("record selection is empty")
    return records


def common(record: dict[str, Any], output: Path, data_root: Path, *, verify_source: bool) -> dict[str, Any]:
    condition = record["condition"]
    source_rel = parse_image_uri(condition["image_uri"])
    source = data_root / source_rel
    if verify_source:
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != int(condition["image_byte_size"]):
            raise ValueError(f"source image size drift: {source}")
        if file_sha(source) != condition["image_sha256"]:
            raise ValueError(f"source image SHA drift: {source}")
        image_target = output / "shared/images" / f"{condition['image_sha256']}{source.suffix.lower()}"
        link_or_copy(source, image_target)
        if file_sha(image_target) != condition["image_sha256"]:
            raise ValueError(f"staged image SHA drift: {image_target}")
        image_path = str(image_target.resolve())
    else:
        image_path = str(source.resolve())
    prompt_path = output / "shared/prompts" / f"{condition['prompt_sha256']}.txt"
    prompt = condition["prompt"].strip()
    if hashlib.sha256(prompt.encode()).hexdigest() != condition["prompt_sha256"]:
        raise ValueError(f"prompt SHA drift: {record['record_id']}")
    write_bytes(prompt_path, prompt.encode())
    actions = expand_record_actions(record)
    return {
        "record_id": record["record_id"],
        "record_sha256": canonical_record_sha(record),
        "group_id": record["group_id"],
        "scene_id": record["benchmark_scene_id"],
        "case_id": record["case_id"],
        "kind": record["kind"],
        "seed": int(record["seed"]),
        "route_length": int(record["route_length"]),
        "actions": actions,
        "condition_image": image_path,
        "condition_image_sha256": condition["image_sha256"],
        "condition_width": int(condition["width"]),
        "condition_height": int(condition["height"]),
        "prompt": prompt,
        "prompt_path": str(prompt_path.resolve()),
        "prompt_sha256": condition["prompt_sha256"],
        "future_rgb_used": False,
    }


def trajectory(actions: list[str], count: int, profile: dict[str, Any], *, dtype: str) -> np.ndarray:
    times = np.linspace(0.0, len(actions), count)
    return sample_c2w(
        actions,
        times,
        translation_per_unit=float(profile["translation_per_unit"]),
        yaw_degrees_per_unit=float(profile["yaw_degrees_per_unit"]),
        dtype=dtype,
    )


def matrix_cropped_anchor(item: dict[str, Any], root: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Materialize the exact center crop consumed by the public Matrix dataset.

    Campaign anchors are all at least 1280 x 704.  Keeping this operation a
    literal crop avoids a second resize and gives the camera file intrinsics
    in the same pixel coordinates as the actual anchor passed to infer.py.
    """
    import cv2

    width, height = int(item["condition_width"]), int(item["condition_height"])
    target_width, target_height = 1280, 704
    if width < target_width or height < target_height:
        raise ValueError(f"Matrix crop contract requires >=1280x704: {width}x{height}")
    source = Path(item["condition_image"])
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (height, width):
        raise ValueError(f"Matrix anchor decode/geometry mismatch: {source}")
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    cropped = image[top : top + target_height, left : left + target_width]
    ok, encoded = cv2.imencode(".png", cropped, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError(f"Matrix anchor PNG encoding failed: {source}")
    payload = encoded.tobytes()
    derived_sha = hashlib.sha256(payload).hexdigest()
    derived = root.parent / "shared/matrix_cropped_images" / f"{derived_sha}.png"
    write_bytes(derived, payload)
    if file_sha(derived) != derived_sha:
        raise ValueError(f"Matrix cropped image SHA drift: {derived}")
    source_k = pixel_intrinsics(width, height, 84.0)
    # The official loader normalizes K by cx/cy to the target frame center.
    # Supply K already in target coordinates so that normalization preserves
    # the crop-consistent focal length rather than shrinking it as for resize.
    k = source_k.copy()
    k[0, 2] = target_width / 2.0
    k[1, 2] = target_height / 2.0
    corrected = {
        **item,
        "source_condition_image": item["condition_image"],
        "source_condition_image_sha256": item["condition_image_sha256"],
        "source_condition_width": width,
        "source_condition_height": height,
        "condition_image": str(derived.resolve()),
        "condition_image_sha256": derived_sha,
        "condition_width": target_width,
        "condition_height": target_height,
        "matrix_anchor_preprocess": {
            "schema_version": MATRIX_LITERAL_ANCHOR_SCHEMA,
            "crop_xywh": [left, top, target_width, target_height],
            "original_principal_point_after_crop": [float(source_k[0, 2] - left), float(source_k[1, 2] - top)],
            "runtime_principal_point": [target_width / 2.0, target_height / 2.0],
            "focal_px": [float(k[0, 0]), float(k[1, 1])],
            "interpolation": "none",
        },
    }
    return corrected, k


def matrix_cover_resize_pixels(
    image: np.ndarray, target_width: int = 1280, target_height: int = 704,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Mirror the released training video frame resize, then center crop.

    VideoLatentDataset.center_crop_resize runs F.interpolate on float frames
    divided by 255 with bilinear/align_corners=False/antialias=True.  The PNG
    boundary quantizes the same way its verbose uint8 frames do.  The public
    inference dataset then receives an already-target-size anchor and takes
    no further spatial resize.
    """
    import torch
    import torch.nn.functional as F

    height, width = image.shape[:2]
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Matrix cover-resize input must be HxWx3 uint8")
    scale = max(target_height / height, target_width / width)
    resized_height = int(round(height * scale))
    resized_width = int(round(width * scale))
    if resized_height < target_height or resized_width < target_width:
        raise ValueError("Matrix cover resize did not cover target")
    left = (resized_width - target_width) // 2
    top = (resized_height - target_height) // 2
    frames = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
    frames = frames.float() / 255.0
    if (resized_height, resized_width) != (height, width):
        frames = F.interpolate(
            frames, size=(resized_height, resized_width), mode="bilinear",
            align_corners=False, antialias=True,
        )
    cropped = frames[:, :, top : top + target_height, left : left + target_width]
    pixels = (cropped * 255.0).clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8)
    return pixels[0].contiguous().numpy(), (resized_width, resized_height, left, top)


def matrix_cover_resized_anchor(item: dict[str, Any], root: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Create a versioned Matrix anchor in the training cover-resize geometry."""
    import cv2

    width, height = int(item["condition_width"]), int(item["condition_height"])
    source = Path(item["condition_image"])
    key = (str(root.parent.resolve()), item["condition_image_sha256"], width, height)
    cached = _MATRIX_COVER_CACHE.get(key)
    if cached is None or not cached[0].is_file() or file_sha(cached[0]) != cached[1]:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (height, width):
            raise ValueError(f"Matrix anchor decode/geometry mismatch: {source}")
        target_width, target_height = 1280, 704
        cropped, (resized_width, resized_height, left, top) = matrix_cover_resize_pixels(
            image, target_width, target_height,
        )
        ok, encoded = cv2.imencode(".png", cropped, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise ValueError(f"Matrix cover-resized anchor PNG encoding failed: {source}")
        payload = encoded.tobytes()
        derived_sha = hashlib.sha256(payload).hexdigest()
        derived = root.parent / "shared/matrix_cover_resized_images" / f"{derived_sha}.png"
        write_bytes(derived, payload)
        if file_sha(derived) != derived_sha:
            raise ValueError(f"Matrix cover-resized image SHA drift: {derived}")
        scale_x, scale_y = resized_width / width, resized_height / height
        source_k = pixel_intrinsics(width, height, 84.0)
        k = source_k.copy()
        k[0, 0] *= scale_x
        k[1, 1] *= scale_y
        # The released loader recenters K to target cx/cy. Set it there already
        # so its normalization does not perturb the crop-consistent focal values
        # by up to half a pixel for odd source/resized dimensions.
        k[0, 2] = target_width / 2.0
        k[1, 2] = target_height / 2.0
        metadata = {
            "schema_version": MATRIX_COVER_ANCHOR_SCHEMA,
            "resize_wh": [resized_width, resized_height],
            "crop_xywh": [left, top, target_width, target_height],
            "source_to_resized_scale_xy": [scale_x, scale_y],
            "original_principal_point_after_crop": [
                float(source_k[0, 2] * scale_x - left),
                float(source_k[1, 2] * scale_y - top),
            ],
            "runtime_principal_point": [target_width / 2.0, target_height / 2.0],
            "focal_px": [float(k[0, 0]), float(k[1, 1])],
            "interpolation": "torch-bilinear-align_corners_false-antialias_true",
            "png_quantization": "float_0_1_times_255_clamp_uint8_truncate",
        }
        cached = (derived, derived_sha, metadata, k)
        _MATRIX_COVER_CACHE[key] = cached
    derived, derived_sha, metadata, k = cached
    corrected = {
        **item,
        "source_condition_image": item["condition_image"],
        "source_condition_image_sha256": item["condition_image_sha256"],
        "source_condition_width": width,
        "source_condition_height": height,
        "condition_image": str(derived.resolve()),
        "condition_image_sha256": derived_sha,
        "condition_width": 1280,
        "condition_height": 704,
        "matrix_anchor_preprocess": metadata,
    }
    return corrected, k


def matrix_case(
    item: dict[str, Any], profile: dict[str, Any], root: Path, *,
    preprocess_anchor: bool = False, anchor_mode: str = "literal-center-crop",
) -> dict[str, Any]:
    if preprocess_anchor:
        if anchor_mode == "literal-center-crop":
            item, K = matrix_cropped_anchor(item, root)
        elif anchor_mode == "cover-resize-center-crop":
            item, K = matrix_cover_resized_anchor(item, root)
        else:
            raise ValueError(f"unsupported Matrix anchor mode: {anchor_mode}")
    else:
        K = pixel_intrinsics(item["condition_width"], item["condition_height"], 84.0)
    length = item["route_length"]
    blocks = math.ceil(length * 8 / 80)
    poses = 1 + 84 * blocks
    camera = root / "cases" / item["record_id"] / "camera.npz"
    # Formal evaluation uses eight frames/action. Extra native alignment poses
    # hold the endpoint rather than stretching the route into unscored padding.
    c2w = sample_c2w(
        item["actions"], np.minimum(np.arange(poses, dtype=np.float64) / 8.0, length),
        translation_per_unit=float(profile["translation_per_unit"]),
        yaw_degrees_per_unit=float(profile["yaw_degrees_per_unit"]), dtype="float32",
    )
    # The public CLI expects pixel intrinsics in the anchor image's resolution.
    save_npz(camera, extrinsics_c2w=c2w, intrinsics=np.repeat(K[None], poses, axis=0))
    config = root / "configs" / f"infer_distilled_{blocks}blocks_seed{item['seed']}.yaml"
    source_config = SOURCE_ROOTS["matrix_game_35_distilled_fp"] / "configs/infer_distilled.yaml"
    config_text = source_config.read_text()
    config_text = config_text.replace("num_blocks: 1", f"num_blocks: {blocks}", 1)
    config_text = config_text.replace("seed: 3407", f"seed: {item['seed']}", 1)
    write_bytes(config, config_text.encode())
    return {
        **item,
        "num_blocks": blocks,
        "raw_frame_count": 1 + 84 * blocks,
        "formal_frame_count": 8 * length,
        "metric_frame_slice_half_open": [0, 8 * length],
        "unscored_tail_frames": 1 + 84 * blocks - 8 * length,
        "camera_path": str(camera.resolve()),
        "camera_sha256": file_sha(camera),
        "config_sha256": file_sha(config),
        "run_argv": [
            "python", "infer_distilled.py", "--config", str(config.resolve()),
            "--checkpoint", "${MATRIX_DISTILLED_CHECKPOINT}", "--image", item["condition_image"],
            "--camera", str(camera.resolve()), "--prompt-file", item["prompt_path"],
            "--wan-dir", "${WAN22_TI2V_5B}",
            "--tokenizer-dir", "${WAN22_TI2V_5B}/google/umt5-xxl",
            "--da3-dir", "${DA3_NESTED_GIANT_LARGE}",
            "--output", f"${{OUTPUT_ROOT}}/{item['record_id']}.mp4",
        ],
    }


def solar_case(item: dict[str, Any], profile: dict[str, Any], root: Path) -> dict[str, Any]:
    source_frames = 1 + 8 * item["route_length"]
    n = (source_frames - 5 + 16) // 17
    decoded = 17 * n + 5
    case = root / "dataset/cases" / item["record_id"]
    camera = case / "camera.npy"
    prompt = case / "prompt.txt"
    image = case / "image.png"
    save_npy(camera, trajectory(item["actions"], source_frames, profile, dtype="float64"))
    write_bytes(prompt, item["prompt"].encode())
    if Path(item["condition_image"]).is_file():
        link_or_copy(Path(item["condition_image"]), image)
    else:
        image = Path(item["condition_image"])
    row = {
        "sample_id": item["record_id"], "key": item["record_id"], "dataset": "strideworld",
        "input_kind": "demo", "num_frames": source_frames, "fps": 24.0,
        "height": item["condition_height"], "width": item["condition_width"], "caption": item["prompt"],
        "image": image.relative_to(root / "dataset").as_posix() if image.is_relative_to(root / "dataset") else str(image),
        "image_sha256": item["condition_image_sha256"],
        "camera": camera.relative_to(root / "dataset").as_posix(), "camera_sha256": file_sha(camera),
        "prompt": prompt.relative_to(root / "dataset").as_posix(), "prompt_sha256": file_sha(prompt),
    }
    return {
        **item, "source_frame_count": source_frames, "raw_frame_count": decoded,
        "formal_frame_count": source_frames, "metric_frame_slice_half_open": [0, source_frames],
        "unscored_tail_frames": decoded - source_frames, "camera_path": str(camera.resolve()),
        "camera_sha256": file_sha(camera), "plan_row": row,
    }


def evoke_case(item: dict[str, Any], profile: dict[str, Any], root: Path) -> dict[str, Any]:
    formal = 8 * item["route_length"]
    chunks = math.ceil((formal + 3) / 36)
    raw = 36 * chunks - 3
    camera = root / "cases" / item["record_id"] / "pose.npz"
    c2w = trajectory(item["actions"], raw, profile, dtype="float32")
    K = pixel_intrinsics(640, 384, 84.0)
    save_npz(camera, cam_c2w=c2w, intrinsics=np.repeat(K[None], raw, axis=0))
    row = {
        "name": item["record_id"], "image_path": item["condition_image"],
        "pose_path": str(camera.resolve()), "prompt_path": item["prompt_path"],
        "pose_fps": 24, "pose_source_resolution": [384, 640], "pose_type": "vipe",
        "seed": item["seed"],
    }
    return {
        **item, "num_chunks": chunks, "raw_frame_count": raw, "formal_frame_count": formal,
        "metric_frame_slice_half_open": [0, formal], "unscored_tail_frames": raw - formal,
        "camera_path": str(camera.resolve()), "camera_sha256": file_sha(camera), "jsonl_row": row,
    }


BUILDERS = {
    "matrix_game_35_distilled_fp": matrix_case,
    "solarwm_h3_stage2_fix": solar_case,
    "evoke_turbo_i2v": evoke_case,
}


def finalize_model(model: str, rows: list[dict[str, Any]], root: Path) -> None:
    for row in rows:
        receipt = dict(row)
        receipt["schema_version"] = "strideworld-model-native-input-v1"
        receipt["model_id"] = model
        receipt["adapter_input_sha256"] = canonical_sha(receipt)
        write_json(root / "receipts" / f"{row['record_id']}.json", receipt)
    write_jsonl(root / "cases.jsonl", rows)
    if model == "solarwm_h3_stage2_fix":
        write_json(root / "plan.json", [row["plan_row"] for row in rows])
    if model == "evoke_turbo_i2v":
        write_jsonl(root / "evoke_i2v.jsonl", [row["jsonl_row"] for row in rows])
        chunks = sorted({row["num_chunks"] for row in rows})
        write_json(root / "chunk_partitions.json", {
            str(chunk): [row["record_id"] for row in rows if row["num_chunks"] == chunk]
            for chunk in chunks
        })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--smoke", action="store_true", help="select first seed-0 ETH3D and UE5 groups")
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-source-validation", action="store_true")
    parser.add_argument("--matrix-preprocess-anchor", action="store_true",
                        help="center-crop Matrix anchors to 1280x704 and bind crop-consistent K")
    parser.add_argument("--matrix-anchor-mode", choices=MATRIX_ANCHOR_MODES,
                        help="experimental Matrix anchor preprocessing; requires a new output identity")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign = json.loads(args.campaign.read_text())
    profiles_doc = json.loads(args.profiles.read_text())
    records = select_records(campaign, args)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    unknown = sorted(set(models) - set(BUILDERS))
    if unknown:
        raise ValueError(f"unsupported models: {unknown}")
    if args.matrix_preprocess_anchor and args.matrix_anchor_mode:
        raise ValueError("use only one Matrix anchor preprocessing option")
    matrix_anchor_mode = args.matrix_anchor_mode or (
        "literal-center-crop" if args.matrix_preprocess_anchor else None
    )
    if matrix_anchor_mode and "matrix_game_35_distilled_fp" not in models:
        raise ValueError("Matrix anchor preprocessing requires the Matrix model")
    if matrix_anchor_mode and (args.output_root / "MANIFEST.json").exists():
        raise FileExistsError("corrected Matrix inputs require a new output identity")
    args.output_root.mkdir(parents=True, exist_ok=True)
    common_rows = [
        common(record, args.output_root, args.data_root, verify_source=not args.skip_source_validation)
        for record in records
    ]
    counts: dict[str, int] = {}
    for model in models:
        root = args.output_root / model
        profile = profiles_doc["models"][model]
        if model == "matrix_game_35_distilled_fp":
            rows = [matrix_case(item, profile, root, preprocess_anchor=matrix_anchor_mode is not None,
                                anchor_mode=matrix_anchor_mode or "literal-center-crop")
                    for item in common_rows]
        else:
            rows = [BUILDERS[model](item, profile, root) for item in common_rows]
        finalize_model(model, rows, root)
        counts[model] = len(rows)
    manifest = {
        "schema_version": "strideworld-model-input-build-v1",
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": campaign["source_plan_sha256"],
        "campaign_document_sha256": canonical_sha(campaign),
        "profile_sha256": file_sha(args.profiles),
        "builder_sha256": file_sha(Path(__file__)),
        "record_count": len(records), "group_count": len({row["group_id"] for row in records}),
        "models": counts, "source_validation": not args.skip_source_validation,
        "future_rgb_used": False,
    }
    if matrix_anchor_mode:
        manifest["matrix_anchor_preprocess"] = (
            MATRIX_LITERAL_ANCHOR_SCHEMA if matrix_anchor_mode == "literal-center-crop"
            else MATRIX_COVER_ANCHOR_SCHEMA
        )
    manifest["build_sha256"] = canonical_sha(manifest)
    write_json(args.output_root / "MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
