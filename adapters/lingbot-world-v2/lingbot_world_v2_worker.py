#!/usr/bin/env python3
"""Run one frozen final shard with official LingBot-World 2.0 causal-fast.

The worker is the execution boundary between ``campaignctl.py`` and the
official LingBot v2 causal-fast pipeline on HOST.  It accepts only the frozen
9-scene/44-case/528-record seed-0 campaign.  A shard constructs the Fast
pipeline once, then generates every pending record as one independent native
rollout.  Complex routes are camera-pose trajectories; generated RGB is never
fed back as a condition and videos are never stitched.

LingBot's ``poses.npy`` interface also requires an intrinsic vector.  This is
the fixed 832x480 *model-control* profile from the official v2 example and is
read from the hash-recorded adapter.  It is not
per-case evaluator GT and makes no claim about UE camera/depth availability.
The evaluator can bind its own GT later without changing the generated RGB.

``--validate-only`` validates the complete CPU contract (including all 528
trajectory mappings and all 44 condition-image bindings) without importing
torch or any other GPU package.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import struct
import sys
import tempfile
from typing import Any, Mapping, Sequence


MODEL_ID = "lingbot_world_v2"
MODEL_VARIANT = "LingBot-World-v2-14b-causal-fast"
MODEL_VARIANT_STATUS = "OFFICIAL_V2_CAUSAL_FAST"
CAMPAIGN_ID = "strideworld_refresh_20260915"
CAMPAIGN_SHA256 = (
    "b80d4a252d6add1ab04246997fde81748939e487e57df1795c6761d3b798eff3"
)

SHARD_SCHEMA = "strideworld-worker-shard-plan-v1"
INPUT_SCHEMA = "strideworld-worker-record-input-v1"
RESULT_SCHEMA = "strideworld-generation-result-v1"
NATIVE_RECORD_SCHEMA = "strideworld-lingbot-world-v2-final-native-generation-v1"
EXECUTED_RECORD_SCHEMA = "strideworld-executed-action-record-v1"
CONTROL_SCHEMA = "strideworld-lingbot-world-v2-final-control-v1"

SEEDS = (0, 1, 2)
FPS = 16
NATIVE_WIDTH = 832
NATIVE_HEIGHT = 480
# These are the dimensions of the official v2 model-control profile.  The
# decoded RGB size is *not* fixed: WanI2VCausal.generate derives it from the
# condition-image aspect ratio and ``max_area``, then floors each side to the
# VAE(8) * patch(2) = 16-pixel grid.
DECODED_WIDTH = 832
DECODED_HEIGHT = 464
MAX_AREA = 480 * 832
MODEL_SPATIAL_ALIGNMENT = 8 * 2
NATIVE_CHUNK_SIZE = 3
LOCAL_ATTENTION_SIZE = 18
SINK_SIZE = 6
DECODED_INTERVALS_PER_UNIT = 4
DECODED_TRANSLATION_PER_INTERVAL = 0.025
# Frozen StrideWorld control profile: 3/4 of the original 1-degree interval
# yaw, applied symmetrically to left and right.  This is trajectory
# conditioning, not a sampler/CFG or inference-speed override.
DECODED_YAW_DEGREES_PER_INTERVAL = 0.75
MODEL_CONTROL_INTRINSICS = (415.5298, 415.6922, 415.77786, 239.77779)
CHECKPOINT_DOSES = tuple(float(value) for value in range(1, 17))
CHECKPOINT_FRAME_INDICES = tuple(4 * value for value in range(1, 17))
COMPLEX_ROUTE_LENGTHS = (8, 11, 14)
CORE_ACTIONS = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "YAW_LEFT",
    "YAW_RIGHT",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMPAIGN = REPO_ROOT / (
    "ops/scene_queue/configs/campaigns/"
    "worldbench_16ue5_6eth3d_v9.json"
)
DEFAULT_ADAPTER = Path(__file__).with_name("lingbot_world_v2_adapter.json")
DEFAULT_RUNTIME_CONTRACT = (
    REPO_ROOT / "scripts/run_lingbot_v2_strideworld_compiled.py"
)
RUNTIME_CONTRACT_SHA256 = (
    "d970e307ca7f1a3861da65d748b1c5daa3d32356be27bb3b73830c1b9f0fbf0d"
)
DEFAULT_DATA_ROOT = Path("<path>/data")
DEFAULT_NATIVE_SOURCE = Path(
    "<path>/third_party"
    "strideworld/lingbot-world-v2"
)
DEFAULT_PYTHON = Path(
    "<path>/envs"
    "strideworld/lingbot-world-v2/bin/python3.12"
)
DEFAULT_CHECKPOINT = Path(
    "<path>/weights"
    "LingBot-World-v2-14b-causal-fast"
)
DEFAULT_CHECKPOINT_MANIFEST = DEFAULT_CHECKPOINT / "strideworld_download_manifest.sha256"
DEFAULT_CHECKPOINT_INFO = DEFAULT_CHECKPOINT / "strideworld_download_info.txt"
DEFAULT_CHECKPOINT_MARKER = DEFAULT_CHECKPOINT / ".strideworld_full_verified.v1.json"
CHECKPOINT_MANIFEST_SHA256 = (
    "6c344e794beb3c2e3eaf6b12222aeb223e9e11124818b78a9c4959a9f68d9a7a"
)
CHECKPOINT_INFO_SHA256 = (
    "9956e19a8b6c63f8e609aba698b5c1c0b189df24290703d2f53186e56315cb66"
)
CHECKPOINT_REPOSITORY = "robbyant/lingbot-world-v2-14b-causal-fast"
CHECKPOINT_REVISION = "5c33dd40b213598c418fd25bff30fdbd23fd38a7"


class WorkerError(RuntimeError):
    """A fail-closed final or native-runtime contract violation."""


def predict_decoded_video_size(source_width: int, source_height: int) -> tuple[int, int]:
    """Predict WanI2VCausal's output grid from the official source formula.

    ``generate`` preserves the condition image aspect ratio while fitting its
    requested area under ``MAX_AREA``.  The result is floored to the VAE
    spatial stride (8) times the patch size (2).  This is deliberately kept
    independent of any output video: it is used before inference and is
    recorded as evidence alongside the decoded dimensions.
    """

    if (
        not isinstance(source_width, int)
        or not isinstance(source_height, int)
        or source_width <= 0
        or source_height <= 0
    ):
        raise WorkerError("source image dimensions must be positive integers")
    aspect = float(source_width) / float(source_height)
    width = int(math.sqrt(float(MAX_AREA) * aspect))
    height = int(math.sqrt(float(MAX_AREA) / aspect))
    width = (width // MODEL_SPATIAL_ALIGNMENT) * MODEL_SPATIAL_ALIGNMENT
    height = (height // MODEL_SPATIAL_ALIGNMENT) * MODEL_SPATIAL_ALIGNMENT
    if width <= 0 or height <= 0:
        raise WorkerError(
            f"source image aspect produces an invalid LingBot grid: "
            f"{source_width}x{source_height} -> {width}x{height}"
        )
    return width, height


def effective_decoded_intrinsics(
    decoded_size: Sequence[int],
) -> tuple[float, float, float, float]:
    """Scale the frozen model-control K to the actual decoded RGB grid."""

    if len(decoded_size) != 2:
        raise WorkerError("decoded video size must contain width and height")
    width, height = (int(decoded_size[0]), int(decoded_size[1]))
    if width <= 0 or height <= 0:
        raise WorkerError("decoded video dimensions must be positive")
    sx = float(width) / float(NATIVE_WIDTH)
    sy = float(height) / float(NATIVE_HEIGHT)
    fx, fy, cx, cy = MODEL_CONTROL_INTRINSICS
    return (fx * sx, fy * sy, cx * sx, cy * sy)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise WorkerError(f"cannot read JSONL {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkerError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise WorkerError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent),
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkerError(f"{label} must be a non-empty string")
    return value


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_component(value: str, label: str) -> str:
    if not value or not all(character.isalnum() or character in "._-" for character in value):
        raise WorkerError(f"unsafe {label}: {value!r}")
    return value


def safe_relative_path(value: str, label: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise WorkerError(f"{label} is not a safe relative POSIX path: {value!r}")
    return Path(*pure.parts)


def validate_adapter(path: Path) -> dict[str, Any]:
    adapter = read_json(path.resolve())
    if (
        adapter.get("schema_version") != "strideworld-v10-native-adapter-v1"
        or adapter.get("adapter_id") != "lingbot_world_v2_causal_fast_v10_d16_v1"
        or adapter.get("model_id") != MODEL_ID
        or adapter.get("model_variant") != MODEL_VARIANT
        or adapter.get("variant_status") != MODEL_VARIANT_STATUS
        or adapter.get("interface_track") != "pose_conditioned"
        or adapter.get("representation") != "camera_trajectory"
        or adapter.get("action_reference_frame") != "current_camera_cv_rh_body"
    ):
        raise WorkerError("LingBot final adapter identity/representation mismatch")
    timing = adapter.get("timing")
    if not isinstance(timing, dict) or any((
        timing.get("fps") != FPS,
        timing.get("conditioning_frames") != 1,
        timing.get("decoded_intervals_per_unit") != DECODED_INTERVALS_PER_UNIT,
        timing.get("atomic_held_dose") != 16,
        timing.get("atomic_action_intervals") != 64,
        timing.get("atomic_output_frames") != 65,
        timing.get("atomic_native_output_frames") != 69,
        timing.get("checkpoint_doses") != list(range(1, 17)),
        timing.get("checkpoint_frame_indices") != list(CHECKPOINT_FRAME_INDICES),
        timing.get("complex_exact_lengths") != list(COMPLEX_ROUTE_LENGTHS),
        timing.get("terminal_noop_frames") != 0,
        timing.get("unscored_padding_frames") != {"atomic_D16": 4, "combination": 0},
    )):
        raise WorkerError("LingBot final adapter timing mismatch")
    expected_frames = {str(length): 1 + 4 * length for length in COMPLEX_ROUTE_LENGTHS}
    if timing.get("complex_output_frames") != expected_frames:
        raise WorkerError("LingBot final adapter complex frame map mismatch")
    native = adapter.get("native_control")
    if not isinstance(native, dict) or any((
        native.get("infer_mode") != "causal_fast",
        native.get("chunk_size_latents") != NATIVE_CHUNK_SIZE,
        native.get("local_attention_size_latents") != LOCAL_ATTENTION_SIZE,
        native.get("sink_size_latents") != SINK_SIZE,
        native.get("decoded_translation_per_interval")
        != DECODED_TRANSLATION_PER_INTERVAL,
        native.get("decoded_yaw_degrees_per_interval")
        != DECODED_YAW_DEGREES_PER_INTERVAL,
        native.get("translation_normalization")
        != "official_compute_relative_poses_max_norm_per_video",
        native.get("single_native_generation_call") is not True,
        native.get("external_rgb_chaining") is not False,
    )):
        raise WorkerError("LingBot final adapter native-control mismatch")
    actions = adapter.get("actions")
    expected_actions = {
        "MOVE_FORWARD": {"translation_body_xyz_sign": [0, 0, 1]},
        "MOVE_BACKWARD": {"translation_body_xyz_sign": [0, 0, -1]},
        "MOVE_LEFT": {"translation_body_xyz_sign": [-1, 0, 0]},
        "MOVE_RIGHT": {"translation_body_xyz_sign": [1, 0, 0]},
        "YAW_LEFT": {"yaw_sign": -1},
        "YAW_RIGHT": {"yaw_sign": 1},
    }
    if actions != expected_actions:
        raise WorkerError("LingBot final adapter action mapping mismatch")
    intrinsics = adapter.get("intrinsics")
    if (
        not isinstance(intrinsics, dict)
        or intrinsics.get("policy") != "frozen_official_v2_model_control_profile"
        or intrinsics.get("fx_fy_cx_cy") != list(MODEL_CONTROL_INTRINSICS)
        or intrinsics.get("source_evidence")
        != "official_lingbot_world_v2_commit_2648877_examples_03_intrinsics_first_row"
        or intrinsics.get("native_control_size") != [NATIVE_WIDTH, NATIVE_HEIGHT]
        or intrinsics.get("decoded_video_size") != [DECODED_WIDTH, DECODED_HEIGHT]
        or intrinsics.get("decoded_video_size_semantics")
        != "nominal_16_9_reference_only"
        or intrinsics.get("decoded_video_size_policy")
        != "official_max_area_aspect_vae8_patch2_floor16"
        or intrinsics.get("camera_intrinsics_required_for_generation") is not True
        or intrinsics.get("absolute_c2w_required_for_generation") is not False
        or intrinsics.get("depth_required_for_generation") is not False
        or intrinsics.get("evaluation_gt_claim") is not False
    ):
        raise WorkerError("LingBot fixed model-control intrinsics contract mismatch")
    if adapter.get("seed_policy") != {
        "seeds": [0, 1, 2], "reset_for_every_independent_video": True,
    }:
        raise WorkerError("LingBot v10 seed policy mismatch")
    return adapter


def measurement_checkpoints() -> list[dict[str, Any]]:
    return [
        {
            "dose_requested": dose,
            "decoded_action_interval": frame,
            "frame_index": frame,
            "prefix_frame_slice_half_open": [0, frame + 1],
        }
        for dose, frame in zip(CHECKPOINT_DOSES, CHECKPOINT_FRAME_INDICES)
    ]


def _validate_step(
    record_id: str, step: Any, step_index: int, expected_dose: float,
) -> str:
    if not isinstance(step, dict):
        raise WorkerError(f"record {record_id} step {step_index} is not an object")
    actions = step.get("canonical_actions")
    if (
        step.get("step_index") != step_index
        or not isinstance(actions, list)
        or len(actions) != 1
        or actions[0] not in CORE_ACTIONS
        or step.get("dose") != expected_dose
    ):
        raise WorkerError(
            f"record {record_id} step {step_index} is not one canonical "
            f"dose-{expected_dose:g} action"
        )
    return str(actions[0])


def compile_record_control(record: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one immutable campaign record to decoded camera intervals."""

    record_id = safe_component(
        require_string(record.get("record_id"), "record_id"), "record_id",
    )
    if record.get("seed") not in SEEDS:
        raise WorkerError(f"record {record_id} does not use v10 seed 0, 1, or 2")
    if record.get("formal_metric_scoring_allowed") is not False:
        raise WorkerError(f"record {record_id} improperly enables formal scoring")
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise WorkerError(f"record {record_id} has no steps")
    kind = record.get("kind")

    if kind == "TASK2_ATOMIC":
        action = record.get("benchmark_action_label")
        if (
            action not in CORE_ACTIONS
            or len(steps) != 1
            or record.get("checkpoint_doses") != list(CHECKPOINT_DOSES)
            or record.get("source_video_dose") != 16.0
            or record.get("one_continuous_video") is not True
            or record.get("intermediate_noop_forbidden") is not True
        ):
            raise WorkerError(f"record {record_id} is not a frozen held-d16 atom")
        if _validate_step(record_id, steps[0], 1, 16.0) != action:
            raise WorkerError(f"record {record_id} atomic label/action mismatch")
        unit_actions = [str(action)] * 16
        logical_steps = [{
            "step_index": 1,
            "canonical_action": str(action),
            "dose": 16.0,
            "unit_start_inclusive": 0,
            "unit_end_exclusive": 16,
            "decoded_interval_start_inclusive": 0,
            "decoded_interval_end_exclusive": 64,
            "boundary_frame_index": 64,
        }]
        checkpoints = measurement_checkpoints()
    elif kind == "TASK1_TASK3_SHARED_COMPLEX":
        length = len(steps)
        if (
            length not in COMPLEX_ROUTE_LENGTHS
            or record.get("route_length") != length
            or record.get("checkpoint_doses") != []
            or record.get("benchmark_action_label") not in ("", None)
            or record.get("metric_consumers")
            != ["TASK1_VISIBLE_WORLD_STABILITY", "TASK3_OPEN_COMPOSITION"]
        ):
            raise WorkerError(f"record {record_id} is not a frozen final shared route")
        unit_actions = []
        logical_steps = []
        for index, step in enumerate(steps, 1):
            action = _validate_step(record_id, step, index, 1.0)
            unit_actions.append(action)
            start = (index - 1) * DECODED_INTERVALS_PER_UNIT
            logical_steps.append({
                "step_index": index,
                "canonical_action": action,
                "dose": 1.0,
                "unit_start_inclusive": index - 1,
                "unit_end_exclusive": index,
                "decoded_interval_start_inclusive": start,
                "decoded_interval_end_exclusive": start + 4,
                "boundary_frame_index": start + 4,
            })
        checkpoints = []
    else:
        raise WorkerError(f"record {record_id} has unsupported kind: {kind!r}")

    decoded_intervals = len(unit_actions) * DECODED_INTERVALS_PER_UNIT
    formal_output_frames = decoded_intervals + 1
    latent_frames = decoded_intervals // 4 + 1
    padding_latents = (-latent_frames) % NATIVE_CHUNK_SIZE
    padding_intervals = padding_latents * 4
    output_frames = formal_output_frames + padding_intervals
    latent_frames += padding_latents
    if decoded_intervals % 4 or latent_frames % NATIVE_CHUNK_SIZE:
        raise WorkerError(
            f"record {record_id} does not fit LingBot Fast's 3-latent chunk grid"
        )
    return {
        "schema_version": CONTROL_SCHEMA,
        "kind": kind,
        "record_id": record_id,
        "unit_actions": unit_actions,
        "unit_count": len(unit_actions),
        "decoded_intervals_per_unit": DECODED_INTERVALS_PER_UNIT,
        "decoded_action_intervals": decoded_intervals,
        "formal_output_frame_count": formal_output_frames,
        "native_output_frame_count": output_frames,
        "native_latent_frame_count": latent_frames,
        "native_chunk_size_latents": NATIVE_CHUNK_SIZE,
        "terminal_noop_frames": 0,
        "unscored_padding_frames": padding_intervals,
        "one_continuous_native_rollout": True,
        "external_rgb_chaining": False,
        "integration_reference_frame": "current_camera_cv_rh_body",
        "steps": logical_steps,
        "measurement_checkpoints": checkpoints,
    }


def rotation_y(angle_radians: float) -> list[list[float]]:
    cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
    return [
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ]


def matmul3(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]],
) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3)]
        for row in range(3)
    ]


def integrate_unit_actions(unit_actions: Sequence[str]) -> list[list[list[float]]]:
    """Right-compose each action in the current camera/body frame."""

    rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    position = [0.0, 0.0, 0.0]
    poses: list[list[list[float]]] = []

    def pose() -> list[list[float]]:
        return [rotation[row][:] + [position[row]] for row in range(3)] + [
            [0.0, 0.0, 0.0, 1.0]
        ]

    poses.append(pose())
    for action in unit_actions:
        if action not in CORE_ACTIONS:
            raise WorkerError(f"cannot integrate unsupported action: {action!r}")
        for _ in range(DECODED_INTERVALS_PER_UNIT):
            if action in {
                "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT",
            }:
                local = {
                    "MOVE_FORWARD": (0.0, 0.0, DECODED_TRANSLATION_PER_INTERVAL),
                    "MOVE_BACKWARD": (0.0, 0.0, -DECODED_TRANSLATION_PER_INTERVAL),
                    "MOVE_LEFT": (-DECODED_TRANSLATION_PER_INTERVAL, 0.0, 0.0),
                    "MOVE_RIGHT": (DECODED_TRANSLATION_PER_INTERVAL, 0.0, 0.0),
                }[action]
                world = [
                    sum(rotation[row][column] * local[column] for column in range(3))
                    for row in range(3)
                ]
                position = [position[index] + world[index] for index in range(3)]
            else:
                sign = -1.0 if action == "YAW_LEFT" else 1.0
                rotation = matmul3(
                    rotation,
                    rotation_y(math.radians(sign * DECODED_YAW_DEGREES_PER_INTERVAL)),
                )
            poses.append(pose())
    expected = 1 + DECODED_INTERVALS_PER_UNIT * len(unit_actions)
    if len(poses) != expected:
        raise WorkerError("internal LingBot pose integration length mismatch")
    return poses


def poses_for_control(control: Mapping[str, Any]) -> list[list[list[float]]]:
    actions = control.get("unit_actions")
    if not isinstance(actions, list) or not all(isinstance(value, str) for value in actions):
        raise WorkerError("compiled control has invalid unit_actions")
    poses = integrate_unit_actions(actions)
    padding = int(control.get("unscored_padding_frames", 0))
    if padding:
        poses.extend([[row[:] for row in poses[-1]] for _ in range(padding)])
    if len(poses) != control.get("native_output_frame_count"):
        raise WorkerError("integrated poses disagree with compiled frame count")
    return poses


def validate_frozen_campaign(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise WorkerError(f"LingBot v10 native campaign is missing: {resolved}")
    campaign = read_json(resolved)
    if (
        campaign.get("schema_version") != "strideworld-v10-native-campaign-v1"
        or campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("source_plan_sha256") != CAMPAIGN_SHA256
        or campaign.get("campaign_status")
        != "FROZEN_V10_COMMON_PLAN_DERIVATIVE"
    ):
        raise WorkerError("frozen final campaign identity/status mismatch")
    groups = campaign.get("groups")
    records = campaign.get("records")
    counts = campaign.get("counts")
    if not isinstance(groups, list) or not isinstance(records, list) or not isinstance(counts, dict):
        raise WorkerError("frozen final campaign arrays/counts are invalid")
    expected_counts = {
        "scene_count": 30,
        "case_group_count": 420,
        "records_per_model": 3640,
        "task2_atomic_records": 2520,
        "shared_task1_task3_records": 1120,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise WorkerError("frozen final campaign count contract mismatch")
    if len(groups) != 420 or len(records) != 3640:
        raise WorkerError("frozen v10 campaign is not 420 seed-groups / 3640 records")
    # v9 binds its source manifests through campaignctl; it does not duplicate
    # group/record canonical hashes at top level. The full campaign SHA is
    # checked before this worker is launched.
    if campaign.get("execution_contract", {}).get("formal_metric_scoring_allowed") is not False:
        raise WorkerError("v9 campaign unexpectedly enables formal metric scoring")
    record_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("record_index") != index:
            raise WorkerError(f"frozen record index mismatch at position {index}")
        record_id = safe_component(
            require_string(record.get("record_id"), "campaign record_id"),
            "campaign record_id",
        )
        if record_id in record_ids:
            raise WorkerError(f"duplicate frozen record id: {record_id}")
        record_ids.add(record_id)
        compile_record_control(record)
    for index, group in enumerate(groups):
        if (
            not isinstance(group, dict)
            or group.get("group_index") != index
            or not isinstance(group.get("record_ids"), list)
            or len(group["record_ids"]) != (14 if group.get("seed") == 0 else 6)
        ):
            raise WorkerError(f"frozen group contract mismatch at position {index}")
    campaign["_verified_path"] = str(resolved)
    campaign["_verified_sha256"] = sha256_file(resolved)
    return campaign


def campaign_condition_expectations(
    campaign: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for record in campaign["records"]:
        state_id = require_string(record.get("state_id"), "state_id")
        condition = record.get("condition")
        if not isinstance(condition, dict):
            raise WorkerError(f"record {record.get('record_id')} has no condition")
        binding = {
            "state_id": state_id,
            "case_id": record.get("case_id"),
            "condition_image_sha256": condition.get("image_sha256"),
            "condition_image_uri": condition.get("image_uri"),
            "condition_image_byte_size": condition.get("image_byte_size"),
            "width": condition.get("width"),
            "height": condition.get("height"),
        }
        previous = expected.setdefault(state_id, binding)
        if previous != binding:
            raise WorkerError(f"state {state_id} has inconsistent campaign conditions")
    if len(expected) != 140:
        raise WorkerError("frozen v10 campaign does not contain exactly 140 condition states")
    return expected


def resolve_data_uri(uri: str, data_root: Path) -> Path:
    if not uri.startswith("data://"):
        raise WorkerError(f"not a data URI: {uri!r}")
    root = data_root.resolve()
    relative = safe_relative_path(uri.removeprefix("data://"), "data URI")
    path = (root / relative).resolve()
    if not inside(root, path) or not path.is_file() or path.stat().st_size <= 0:
        raise WorkerError(f"data URI is missing or escapes the data root: {uri}")
    return path


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise WorkerError(f"condition image is not a valid PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise WorkerError(f"condition PNG dimensions are invalid: {path}")
    return width, height


def load_condition_assets(
    campaign: Mapping[str, Any], data_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Verify all 44 frozen condition PNGs without consulting UE metadata."""

    expected = campaign_condition_expectations(campaign)
    assets: dict[str, dict[str, Any]] = {}
    for state_id, binding in sorted(expected.items()):
        uri = require_string(binding.get("condition_image_uri"), "condition image URI")
        image = resolve_data_uri(uri, data_root)
        if (
            image.stat().st_size != binding["condition_image_byte_size"]
            or sha256_file(image) != binding["condition_image_sha256"]
            or png_size(image) != (binding["width"], binding["height"])
        ):
            raise WorkerError(f"frozen condition PNG bytes/dimensions mismatch for {state_id}")
        assets[state_id] = {
            **binding,
            "condition_image": image,
        }
    return assets, {
        "data_root": str(data_root.resolve()),
        "verified_condition_count": len(assets),
        "verification": "campaign_uri_size_sha256_png_dimensions",
        "camera_or_depth_required_for_generation": False,
    }


def validate_record_output_slot(
    output_root: Path, row: Mapping[str, Any],
) -> Path:
    """Accept campaignctl's immutable input directory, but no output artifact.

    ``campaignctl`` creates ``<record_dir>/input.json`` before the worker is
    launched.  The worker must preserve that file and publish beneath the same
    directory; treating the pre-existing record directory as an output
    collision would make every real run fail before inference.
    """

    root = output_root.resolve()
    relative_dir = safe_relative_path(
        require_string(row.get("record_output_dir"), "record_output_dir"),
        "record_output_dir",
    )
    record_dir = (root / relative_dir).resolve()
    if not inside(root, record_dir) or not record_dir.is_dir():
        raise WorkerError("campaignctl record directory is missing or unsafe")
    input_path = record_dir / "input.json"
    if not input_path.is_file() or canonical_bytes(read_json(input_path)) != canonical_bytes(row):
        raise WorkerError("campaignctl immutable input.json differs from pending row")
    if (record_dir / "generation_record.json").exists() or (record_dir / "artifact").exists():
        raise WorkerError(
            "pending record already contains generation output; reconcile or quarantine first"
        )
    return record_dir


def _validate_prompt_and_condition(record: Mapping[str, Any]) -> None:
    record_id = str(record.get("record_id"))
    condition = record.get("condition")
    if not isinstance(condition, dict):
        raise WorkerError(f"record {record_id} has no condition object")
    prompt = condition.get("prompt")
    if (
        not isinstance(prompt, str)
        or not prompt
        or hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        != condition.get("prompt_sha256")
        or not is_sha256(condition.get("image_sha256"))
        or not isinstance(condition.get("image_uri"), str)
    ):
        raise WorkerError(f"record {record_id} condition binding mismatch")


def validate_work(
    campaign_path: Path, shard_plan_path: Path, shard_manifest_path: Path,
    output_root: Path, adapter_path: Path, data_root: Path,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, Any],
    dict[str, dict[str, Any]], dict[str, Any],
]:
    campaign = validate_frozen_campaign(campaign_path)
    adapter = validate_adapter(adapter_path)
    assets, asset_evidence = load_condition_assets(campaign, data_root)
    # Resolve the official dynamic output grid for every bound state during
    # CPU preflight.  This makes a bad dimensions policy fail before a GPU
    # worker is scheduled, instead of discovering it after the first decode.
    decoded_predictions: dict[str, list[int]] = {}
    for state_id, asset in sorted(assets.items()):
        source_size = (int(asset["width"]), int(asset["height"]))
        decoded_predictions[state_id] = list(predict_decoded_video_size(*source_size))
    asset_evidence = {
        **asset_evidence,
        "decoded_video_size_policy": (
            "official_max_area_aspect_vae8_patch2_floor16"
        ),
        "decoded_size_predictions_by_state": decoded_predictions,
    }
    plan = read_json(shard_plan_path.resolve())
    output_root = output_root.resolve()
    if (
        plan.get("schema_version") != SHARD_SCHEMA
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("campaign_sha256") != CAMPAIGN_SHA256
        or plan.get("model_id") != MODEL_ID
    ):
        raise WorkerError("shard plan identity/campaign/model mismatch")
    if Path(require_string(plan.get("output_root"), "plan output_root")).resolve() != output_root:
        raise WorkerError("shard plan output_root differs from --output-root")
    if (
        not isinstance(plan.get("shard_index"), int)
        or not isinstance(plan.get("shard_count"), int)
        or not 0 <= plan["shard_index"] < plan["shard_count"]
        or not isinstance(plan.get("gpu_group"), list)
        or len(plan["gpu_group"]) != 1
        or not isinstance(plan["gpu_group"][0], int)
    ):
        raise WorkerError("LingBot shard plan requires one valid GPU group")
    environment = {
        "SW_MODEL_ID": MODEL_ID,
        "SW_CAMPAIGN_SHA256": CAMPAIGN_SHA256,
        "SW_PLAN_SHA256": plan.get("plan_sha256"),
        "SW_OUTPUT_ROOT": str(output_root),
        "SW_SHARD_INDEX": str(plan["shard_index"]),
        "SW_GPU_GROUP": str(plan["gpu_group"][0]),
    }
    for name, expected in environment.items():
        observed = os.environ.get(name)
        if observed is not None and observed != str(expected):
            raise WorkerError(f"worker environment mismatch for {name}")
    rows = read_jsonl(shard_manifest_path.resolve())
    if not rows:
        raise WorkerError("pending shard manifest is empty")
    allowed_groups = set(plan.get("case_group_ids", []))
    frozen_by_id = {record["record_id"]: record for record in campaign["records"]}
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        if (
            row.get("schema_version") != INPUT_SCHEMA
            or row.get("campaign_id") != CAMPAIGN_ID
            or row.get("campaign_sha256") != CAMPAIGN_SHA256
            or row.get("plan_sha256") != plan.get("plan_sha256")
            or row.get("model_id") != MODEL_ID
            or row.get("required_result_schema") != RESULT_SCHEMA
        ):
            raise WorkerError(f"pending row {row_index} identity/schema mismatch")
        record = row.get("record")
        if not isinstance(record, dict):
            raise WorkerError(f"pending row {row_index} has no record object")
        record_id = require_string(record.get("record_id"), "record_id")
        if record_id in seen or record_id not in frozen_by_id:
            raise WorkerError(f"pending row has duplicate/unknown record: {record_id}")
        seen.add(record_id)
        if canonical_bytes(record) != canonical_bytes(frozen_by_id[record_id]):
            raise WorkerError(f"pending record differs from frozen campaign: {record_id}")
        if record.get("group_id") not in allowed_groups:
            raise WorkerError(f"pending record is outside shard groups: {record_id}")
        compile_record_control(record)
        _validate_prompt_and_condition(record)
        if record.get("state_id") not in assets:
            raise WorkerError(f"pending record has no bound condition image: {record_id}")
        relative_dir = safe_relative_path(
            require_string(row.get("record_output_dir"), "record_output_dir"),
            "record_output_dir",
        )
        expected_result = (relative_dir / "generation_record.json").as_posix()
        if row.get("required_generation_record") != expected_result:
            raise WorkerError(f"pending record result path mismatch: {record_id}")
        validate_record_output_slot(output_root, row)
    if plan.get("runnable_record_count") is not None and len(rows) > int(plan["runnable_record_count"]):
        raise WorkerError("pending shard contains more rows than its runnable plan")
    return plan, rows, adapter, assets, asset_evidence


def load_runtime_contract(path: Path) -> Any:
    resolved = path.resolve()
    if not resolved.is_file():
        raise WorkerError(f"LingBot v2 runtime contract is missing: {resolved}")
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != RUNTIME_CONTRACT_SHA256:
        raise WorkerError(
            "LingBot v2 runtime contract SHA256 mismatch: "
            f"expected={RUNTIME_CONTRACT_SHA256} actual={actual_sha256}"
        )
    name = "strideworld_lingbot_world_v2_contract_for_final"
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise WorkerError(f"cannot import LingBot v2 runtime contract: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        if (
            module.NATIVE_COMMIT != "2648877f763a06cc743bcd919936da4d25f12e7b"
            or module.MAX_AREA != MAX_AREA
            or module.NATIVE_WIDTH != NATIVE_WIDTH
            or module.NATIVE_HEIGHT != NATIVE_HEIGHT
        ):
            raise WorkerError("LingBot v2 runtime constants drifted")
    except Exception as exc:
        raise WorkerError(f"LingBot v2 runtime contract failed: {exc}") from exc
    return module


def validate_checkpoint_download(
    root: Path, manifest_path: Path, info_path: Path, marker_path: Path,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    info_path = info_path.resolve()
    marker_path = marker_path.resolve()
    if not root.is_dir():
        raise WorkerError(f"LingBot v2 checkpoint root is missing: {root}")
    if sha256_file(manifest_path) != CHECKPOINT_MANIFEST_SHA256:
        raise WorkerError("LingBot v2 checkpoint manifest SHA256 mismatch")
    if sha256_file(info_path) != CHECKPOINT_INFO_SHA256:
        raise WorkerError("LingBot v2 checkpoint info SHA256 mismatch")
    info: dict[str, str] = {}
    for line in info_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            info[key] = value
    if (
        info.get("repository") != CHECKPOINT_REPOSITORY
        or info.get("revision") != CHECKPOINT_REVISION
    ):
        raise WorkerError("LingBot v2 checkpoint repository/revision mismatch")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1,
    ):
        digest, separator, relative = line.partition("  ")
        if not separator or not is_sha256(digest) or relative in seen:
            raise WorkerError(
                f"invalid LingBot v2 checkpoint manifest row {line_number}"
            )
        relative_path = safe_relative_path(relative, "checkpoint manifest path")
        file_path = (root / relative_path).resolve()
        if not inside(root, file_path) or not file_path.is_file():
            raise WorkerError(f"LingBot v2 checkpoint file is missing: {relative}")
        byte_size = file_path.stat().st_size
        if byte_size <= 0:
            raise WorkerError(f"LingBot v2 checkpoint file is empty: {relative}")
        seen.add(relative)
        rows.append({
            "relative_path": relative,
            "byte_size": byte_size,
            "sha256": digest,
        })
    required = {
        "config.json", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
        "transformers/diffusion_pytorch_model.safetensors.index.json",
        "google/umt5-xxl/special_tokens_map.json", "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/tokenizer.json", "google/umt5-xxl/tokenizer_config.json",
        *(f"transformers/model-{index:05d}-of-00008.safetensors" for index in range(1, 9)),
    }
    if not required.issubset(seen):
        raise WorkerError(
            f"LingBot v2 checkpoint manifest misses: {sorted(required - seen)}"
        )
    marker = read_json(marker_path)
    if (
        marker.get("schema_version") != "strideworld-full-checkpoint-verification-v1"
        or marker.get("repository") != CHECKPOINT_REPOSITORY
        or marker.get("revision") != CHECKPOINT_REVISION
        or marker.get("manifest_sha256") != CHECKPOINT_MANIFEST_SHA256
        or marker.get("info_sha256") != CHECKPOINT_INFO_SHA256
        or marker.get("verified_entry_count") != len(rows)
        or marker.get("verification_result") != "ALL_ENTRIES_SHA256_OK"
    ):
        raise WorkerError("LingBot v2 full-checksum verification marker mismatch")
    return {
        "schema_version": "strideworld-checkpoint-manifest-binding-v1",
        "repository": CHECKPOINT_REPOSITORY,
        "revision": CHECKPOINT_REVISION,
        "file_count": len(rows),
        "files": rows,
        "aggregate_sha256": CHECKPOINT_MANIFEST_SHA256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "info_path": str(info_path),
        "info_sha256": CHECKPOINT_INFO_SHA256,
        "verified_marker_path": str(marker_path),
        "verified_marker_sha256": sha256_file(marker_path),
        "verification": "pinned_manifest_plus_prior_full_sha256_check",
    }


def validate_native_runtime(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    expected_python = args.expected_python.resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise WorkerError(
            f"LingBot worker must run under {expected_python}; current={Path(sys.executable).resolve()}"
        )
    runtime_contract = load_runtime_contract(args.runtime_contract)
    try:
        source_repository = runtime_contract.validate_native_repository(
            args.native_source_root.resolve()
        )
        source_files = runtime_contract.validate_native_sources(
            args.native_source_root.resolve()
        )
        inventory = validate_checkpoint_download(
            args.checkpoint, args.checkpoint_manifest,
            args.checkpoint_info, args.checkpoint_verified_marker,
        )
    except Exception as exc:
        raise WorkerError(f"LingBot v2 source/checkpoint validation failed: {exc}") from exc
    return runtime_contract, {
        "runtime_contract_path": str(args.runtime_contract.resolve()),
        "runtime_contract_sha256": sha256_file(args.runtime_contract.resolve()),
        "expected_python": str(expected_python),
        "native_repository": source_repository,
        "native_source_files": source_files,
        "checkpoint_root": str(args.checkpoint.resolve()),
        "checkpoint_inventory": inventory,
    }


def write_native_controls(
    directory: Path, poses: Sequence[Sequence[Sequence[float]]],
    native_intrinsics: Sequence[float], np: Any,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=False)
    pose_values = np.asarray(poses, dtype=np.float32)
    intrinsic_values = np.broadcast_to(
        np.asarray(tuple(native_intrinsics), dtype=np.float32),
        (len(poses), 4),
    ).copy()
    if pose_values.shape != (len(poses), 4, 4) or intrinsic_values.shape != (len(poses), 4):
        raise WorkerError("LingBot native control array shape mismatch")
    poses_path = directory / "poses.npy"
    intrinsics_path = directory / "intrinsics.npy"
    np.save(poses_path, pose_values)
    np.save(intrinsics_path, intrinsic_values)
    return poses_path, intrinsics_path


def decoded_video_info(
    path: Path,
    imageio: Any,
    expected_frames: int,
    expected_size: Sequence[int],
) -> dict[str, Any]:
    try:
        reader = imageio.get_reader(path)
        frame_count = 0
        width = height = None
        for frame in reader:
            if not hasattr(frame, "shape") or len(frame.shape) < 2:
                raise WorkerError("decoded LingBot frame has no image shape")
            frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])
            expected_width, expected_height = (
                int(expected_size[0]), int(expected_size[1])
            )
            if (frame_width, frame_height) != (expected_width, expected_height):
                raise WorkerError(
                    "LingBot decoded frame dimension mismatch: "
                    f"frame={frame_width}x{frame_height} "
                    f"expected={expected_width}x{expected_height}"
                )
            frame_count += 1
            height, width = frame_height, frame_width
        metadata = reader.get_meta_data()
        reader.close()
        result = {
            "decoded_frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": float(metadata.get("fps", 0.0)),
        }
    except Exception as exc:
        raise WorkerError(f"cannot decode generated LingBot video {path}: {exc}") from exc
    expected = {
        "decoded_frame_count": expected_frames,
        "width": int(expected_size[0]),
        "height": int(expected_size[1]),
        "fps": float(FPS),
    }
    if result != expected:
        raise WorkerError(f"LingBot decoded video contract mismatch: {result}")
    return result


def build_native_record(
    row: Mapping[str, Any], control: Mapping[str, Any],
    image: Path, video: Path, poses_path: Path, intrinsics_path: Path,
    decoded: Mapping[str, Any], adapter_path: Path, runtime: Mapping[str, Any],
    model_control_intrinsics: Sequence[float],
    source_size: Sequence[int], decoded_size: Sequence[int],
    pipeline_load_count: int,
) -> dict[str, Any]:
    record = row["record"]
    poses_sha = sha256_file(poses_path)
    intrinsics_sha = sha256_file(intrinsics_path)
    video_sha = sha256_file(video)
    return {
        "schema_version": NATIVE_RECORD_SCHEMA,
        "status": "COMPLETED_MACHINE_VALIDATED_MANUAL_REVIEW_PENDING",
        "formal_metric_scoring_allowed": False,
        "record_id": record["record_id"],
        "model_id": MODEL_ID,
        "model_variant": MODEL_VARIANT,
        "model_variant_status": MODEL_VARIANT_STATUS,
        "campaign_sha256": row["campaign_sha256"],
        "plan_sha256": row["plan_sha256"],
        "seed": int(record["seed"]),
        "independent_seed_reset_for_record": True,
        "condition": {
            "image_uri": record["condition"]["image_uri"],
            "image_path": str(image),
            "image_sha256": record["condition"]["image_sha256"],
            "prompt_sha256": record["condition"]["prompt_sha256"],
        },
        "model_control_intrinsics": {
            "fx_fy_cx_cy": list(model_control_intrinsics),
            "native_control_size": [NATIVE_WIDTH, NATIVE_HEIGHT],
            "decoded_video_size": [int(decoded_size[0]), int(decoded_size[1])],
            "decoded_video_size_policy": (
                "official_max_area_aspect_vae8_patch2_floor16"
            ),
            "source_condition_size": [int(source_size[0]), int(source_size[1])],
            "effective_decoded_fx_fy_cx_cy": list(
                effective_decoded_intrinsics(decoded_size)
            ),
            "source": "official_lingbot_world_v2_commit_2648877_examples_03",
            "shared_across_generation_cases": True,
            "camera_or_depth_required_for_generation": False,
            "evaluation_gt_claim": False,
            "note": (
                "This K conditions the LingBot pose-control interface only; "
                "it is not per-case UE/evaluator ground truth."
            ),
        },
        "control": {
            **dict(control),
            "poses_path": "native_controls/poses.npy",
            "poses_sha256": poses_sha,
            "poses_content_sha256": canonical_sha256(poses_for_control(control)),
            "intrinsics_path": "native_controls/intrinsics.npy",
            "intrinsics_sha256": intrinsics_sha,
            "translation_per_decoded_interval": DECODED_TRANSLATION_PER_INTERVAL,
            "yaw_degrees_per_decoded_interval": DECODED_YAW_DEGREES_PER_INTERVAL,
            "runtime_translation_normalization": (
                "official_compute_relative_poses_max_norm_per_video"
            ),
        },
        "native_inference": {
            "pipeline_class": "wan.WanI2VCausal",
            "infer_mode": "causal_fast",
            "pipeline_load_count_for_shard": pipeline_load_count,
            "native_generation_calls_for_record": 1,
            "frame_num": control["native_output_frame_count"],
            "chunk_size": NATIVE_CHUNK_SIZE,
            "local_attention_size_latents": LOCAL_ATTENTION_SIZE,
            "sink_size_latents": SINK_SIZE,
            "max_area": MAX_AREA,
            "fps": FPS,
            "shift_source": "WAN_CONFIGS[i2v-A14B].sample_shift",
            "timesteps_index_source": "WanI2VCausal causal-fast default [0,179,358,679]",
            "offload_model": False,
            "external_rgb_chaining": False,
            "terminal_noop_frames": 0,
            "unscored_padding_frames": int(control["unscored_padding_frames"]),
        },
        "execution_output": {
            **dict(decoded),
            "decoded_video_size_policy": (
                "official_max_area_aspect_vae8_patch2_floor16"
            ),
            "source_condition_size": [int(source_size[0]), int(source_size[1])],
            "effective_decoded_fx_fy_cx_cy": list(
                effective_decoded_intrinsics(decoded_size)
            ),
            "video_path": "raw_generated.mp4",
            "video_sha256": video_sha,
            "evaluation_window": {
                "start_frame": 0,
                "end_frame_exclusive": control["formal_output_frame_count"],
            },
            "metric_frame_slice_half_open": [0, control["formal_output_frame_count"]],
            "measurement_checkpoints": control["measurement_checkpoints"],
        },
        "adapter": {
            "path": str(adapter_path.resolve()),
            "sha256": sha256_file(adapter_path.resolve()),
        },
        "runtime": dict(runtime),
        "completed_at": utc_now(),
    }


def build_executed_record(
    row: Mapping[str, Any], control: Mapping[str, Any], image: Path, video: Path,
    native_record_path: Path, checkpoint_fingerprint: str,
    decoded: Mapping[str, Any],
) -> dict[str, Any]:
    record = dict(row["record"])
    return {
        **record,
        "execution_record_schema_version": EXECUTED_RECORD_SCHEMA,
        "source_campaign_record_sha256": canonical_sha256(record),
        "condition_image_sha256": sha256_file(image),
        "generated_video_sha256": sha256_file(video),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "checkpoint_fingerprint_source": (
            "pinned_expected_manifests_plus_live_path_size_sha256_all_required_files"
        ),
        "native_generation_record_path": "native_generation_record.json",
        "native_generation_record_sha256": sha256_file(native_record_path),
        "required_output_frames": control["formal_output_frame_count"],
        "execution_output": {
            **dict(decoded),
            "evaluation_window": {
                "start_frame": 0,
                "end_frame_exclusive": control["formal_output_frame_count"],
            },
            "metric_frame_slice_half_open": [0, control["formal_output_frame_count"]],
            "measurement_checkpoints": control["measurement_checkpoints"],
            "terminal_noop_frames": 0,
            "unscored_padding_frames": int(control["unscored_padding_frames"]),
        },
        "executed_control": dict(control),
    }


def publish_controller_result(
    record_dir: Path, row: Mapping[str, Any], plan: Mapping[str, Any],
    artifact_dir: Path,
) -> None:
    video = artifact_dir / "raw_generated.mp4"
    native_record_path = artifact_dir / "native_generation_record.json"
    executed_record_path = artifact_dir / "executed_action_record.json"
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "SUCCEEDED",
        "record_id": row["record"]["record_id"],
        "model_id": MODEL_ID,
        "campaign_sha256": plan["campaign_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "video_path": "artifact/raw_generated.mp4",
        "video_sha256": sha256_file(video),
        "native_generation_record_path": "artifact/native_generation_record.json",
        "native_generation_record_sha256": sha256_file(native_record_path),
        "executed_action_record_path": "artifact/executed_action_record.json",
        "executed_action_record_sha256": sha256_file(executed_record_path),
        "evidence_status": "RGB_GENERATED_MACHINE_VALIDATED_MANUAL_REVIEW_PENDING",
        "formal_metric_scoring_allowed": False,
        "published_at": utc_now(),
    }
    result_path = record_dir / "generation_record.json"
    if result_path.exists():
        raise WorkerError(f"refusing existing controller result: {result_path}")
    atomic_write_json(result_path, result)


def generate_shard(
    args: argparse.Namespace, plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    adapter: Mapping[str, Any], assets: Mapping[str, Mapping[str, Any]],
    asset_evidence: Mapping[str, Any], runtime_contract: Any,
    runtime: Mapping[str, Any],
) -> None:
    try:
        wan, configs, save_video, Image, helpers = runtime_contract.runtime_imports(
            args.native_source_root.resolve()
        )
        np, imageio = helpers
    except Exception as exc:
        raise WorkerError(f"cannot import LingBot v2 GPU runtime: {exc}") from exc
    config = configs["i2v-A14B"]
    try:
        pipe = wan.WanI2VCausal(
            config=config,
            checkpoint_dir=str(args.checkpoint.resolve()),
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
            convert_model_dtype=False,
            local_attn_size=LOCAL_ATTENTION_SIZE,
            sink_size=SINK_SIZE,
            infer_mode="causal_fast",
        )
    except Exception as exc:
        raise WorkerError(f"cannot construct LingBot v2 causal-fast pipeline: {exc}") from exc
    pipeline_load_count = 1
    attempt_id = os.environ.get("SW_ATTEMPT_ID") or (
        f"manual_shard_{plan['shard_index']:03d}_{os.getpid()}"
    )
    safe_component(attempt_id, "attempt id")
    attempt_root = args.shard_manifest.resolve().parent / f"lingbot_world_v2_native_{attempt_id}"
    if attempt_root.exists():
        raise WorkerError(f"refusing existing LingBot attempt root: {attempt_root}")
    attempt_root.mkdir(parents=True)
    checkpoint_fingerprint = str(runtime["checkpoint_inventory"]["aggregate_sha256"])
    if not is_sha256(checkpoint_fingerprint):
        raise WorkerError("LingBot v2 checkpoint inventory has no aggregate SHA256")
    model_control_intrinsics = tuple(
        float(value) for value in adapter["intrinsics"]["fx_fy_cx_cy"]
    )

    for ordinal, row in enumerate(rows, 1):
        record = row["record"]
        record_id = str(record["record_id"])
        control = compile_record_control(record)
        poses = poses_for_control(control)
        asset = assets[str(record["state_id"])]
        image = Path(asset["condition_image"])
        source_size = png_size(image)
        decoded_size = predict_decoded_video_size(*source_size)
        stage = attempt_root / record_id
        stage.mkdir()
        controls_dir = stage / "native_controls"
        poses_path, intrinsics_path = write_native_controls(
            controls_dir, poses, model_control_intrinsics, np,
        )
        try:
            with Image.open(image) as source_image:
                condition_image = source_image.convert("RGB")
            video_tensor = pipe.generate(
                str(record["condition"]["prompt"]),
                condition_image,
                action_path=str(controls_dir),
                chunk_size=NATIVE_CHUNK_SIZE,
                max_area=MAX_AREA,
                frame_num=int(control["native_output_frame_count"]),
                shift=config.sample_shift,
                seed=int(record["seed"]),
                offload_model=False,
                max_attention_size=None,
            )
        except Exception as exc:
            raise WorkerError(f"LingBot generation failed for {record_id}: {exc}") from exc
        expected_frames = int(control["native_output_frame_count"])
        if video_tensor is None or int(video_tensor.shape[1]) != expected_frames:
            raise WorkerError(
                f"LingBot native output shape mismatch for {record_id}: "
                f"{getattr(video_tensor, 'shape', None)}"
            )
        if (
            len(video_tensor.shape) < 4
            or tuple(int(value) for value in video_tensor.shape[2:4])
            != (int(decoded_size[1]), int(decoded_size[0]))
        ):
            raise WorkerError(
                f"LingBot native spatial shape mismatch for {record_id}: "
                f"shape={getattr(video_tensor, 'shape', None)} "
                f"expected=(C,{expected_frames},{decoded_size[1]},{decoded_size[0]})"
            )
        video = stage / "raw_generated.mp4"
        save_video(tensor=video_tensor[None], save_file=str(video), fps=FPS, nrow=1)
        decoded = decoded_video_info(video, imageio, expected_frames, decoded_size)
        native_record = build_native_record(
            row, control, image, video, poses_path, intrinsics_path,
            decoded, args.adapter, {
                **dict(runtime),
                "condition_assets": dict(asset_evidence),
                "native_size_prediction": {
                    "source_condition_size": list(source_size),
                    "predicted_decoded_size": list(decoded_size),
                    "formula": "floor16(sqrt(MAX_AREA * aspect)), floor16(sqrt(MAX_AREA / aspect))",
                    "max_area": MAX_AREA,
                    "spatial_alignment": MODEL_SPATIAL_ALIGNMENT,
                },
            }, model_control_intrinsics, source_size, decoded_size,
            pipeline_load_count,
        )
        native_record_path = stage / "native_generation_record.json"
        atomic_write_json(native_record_path, native_record)
        executed = build_executed_record(
            row, control, image, video, native_record_path,
            checkpoint_fingerprint, decoded,
        )
        executed_path = stage / "executed_action_record.json"
        atomic_write_json(executed_path, executed)
        final_relative = safe_relative_path(
            str(row["record_output_dir"]), "record_output_dir",
        )
        record_dir = (args.output_root.resolve() / final_relative).resolve()
        if not inside(args.output_root.resolve(), record_dir) or not record_dir.is_dir():
            raise WorkerError(f"unsafe/missing campaignctl record directory: {record_dir}")
        artifact_dir = record_dir / "artifact"
        if artifact_dir.exists() or (record_dir / "generation_record.json").exists():
            raise WorkerError(f"refusing existing LingBot record output: {record_dir}")
        os.replace(stage, artifact_dir)
        publish_controller_result(record_dir, row, plan, artifact_dir)
        del video_tensor
        print(
            f"[{ordinal}/{len(rows)}] SAVED {artifact_dir / 'raw_generated.mp4'}",
            flush=True,
        )

    with contextlib.suppress(OSError):
        attempt_root.rmdir()
    print(json.dumps({
        "status": "SUCCEEDED",
        "model_id": MODEL_ID,
        "shard_index": plan["shard_index"],
        "record_count": len(rows),
        "pipeline_load_count": pipeline_load_count,
        "independent_seed_resets": len(rows),
    }, sort_keys=True))


def run(args: argparse.Namespace) -> None:
    if args.shard_index is not None:
        raw_plan = read_json(args.shard_plan.resolve())
        if raw_plan.get("shard_index") != args.shard_index:
            raise WorkerError("--shard-index differs from shard plan")
    plan, rows, adapter, assets, asset_evidence = validate_work(
        args.campaign, args.shard_plan, args.shard_manifest,
        args.output_root, args.adapter, args.data_root,
    )
    controls = [compile_record_control(row["record"]) for row in rows]
    if args.validate_only:
        print(json.dumps({
            "status": "VALIDATION_OK_CPU_ONLY",
            "model_id": MODEL_ID,
            "campaign_id": CAMPAIGN_ID,
            "campaign_records": 3640,
            "campaign_case_groups": 420,
            "campaign_scenes": 30,
            "pending_record_count": len(rows),
            "pending_atomic_count": sum(
                control["kind"] == "TASK2_ATOMIC" for control in controls
            ),
            "pending_complex_count": sum(
                control["kind"] == "TASK1_TASK3_SHARED_COMPLEX" for control in controls
            ),
            "native_output_frame_counts": sorted({
                control["native_output_frame_count"] for control in controls
            }),
            "verified_condition_images": len(assets),
            "model_control_intrinsics": list(MODEL_CONTROL_INTRINSICS),
            "model_control_intrinsics_evaluation_gt_claim": False,
            "gpu_imported": False,
            "gpu_started": False,
            "pipeline_load_count": 0,
        }, sort_keys=True))
        return
    expected_cuda = str(plan["gpu_group"][0])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_cuda:
        raise WorkerError(
            "LingBot GPU run requires campaignctl CUDA_VISIBLE_DEVICES="
            f"{expected_cuda}; actual={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    runtime_contract, runtime = validate_native_runtime(args)
    generate_shard(
        args, plan, rows, adapter, assets, asset_evidence,
        runtime_contract, runtime,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--shard-plan", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--source-root", "--native-source-root", dest="native_source_root",
        type=Path, default=DEFAULT_NATIVE_SOURCE,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST,
    )
    parser.add_argument(
        "--checkpoint-info", type=Path, default=DEFAULT_CHECKPOINT_INFO,
    )
    parser.add_argument(
        "--checkpoint-verified-marker", type=Path,
        default=DEFAULT_CHECKPOINT_MARKER,
    )
    parser.add_argument(
        "--runtime-contract", type=Path, default=DEFAULT_RUNTIME_CONTRACT,
    )
    parser.add_argument("--expected-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    try:
        run(parse_args())
    except (WorkerError, OSError, ValueError) as exc:
        print(f"ERROR [{MODEL_ID}]: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
