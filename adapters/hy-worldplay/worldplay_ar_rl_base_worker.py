#!/usr/bin/env python3
"""Run one official HY-WorldPlay AR-RL shard on HOST.

The worker consumes the immutable ``campaignctl`` shard plan together with the
attempt-local pending manifest.  It accepts only the frozen campaign contract
and the exact official HunyuanVideo AR-RL WorldPlay variant.  One
official pipeline is loaded per shard and every record is an independent
seed-0 call from its frozen first frame.

Atomic records are one held-eight rollout.  Complex records are one ordered
comma-separated native pose and one pipeline call; RGB outputs are never
chained.  The official AR implementation requires the latent count to be a
multiple of four, so the minimum positive terminal ``none`` segment needed by
the native interface is retained only as unscored alignment padding.  No NOOP
frame enters a benchmark measurement window; the exact action-only prefix is
recorded for downstream scoring.

``--validate-only`` performs the complete CPU contract, source, all-SHA
receipt, weight-size, input-image, and native-request preflight.  It deliberately
does not import torch or the official CUDA entrypoint.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


MODEL_ID = "worldplay_ar_rl_official"
MODEL_VARIANT = "HY-World1.5-AR-RL-480P-I2V-official"
CAMPAIGN_ID = "strideworld_refresh_20260915"
CAMPAIGN_SHA256 = "b80d4a252d6add1ab04246997fde81748939e487e57df1795c6761d3b798eff3"

SHARD_PLAN_SCHEMA = "strideworld-worker-shard-plan-v1"
WORKER_INPUT_SCHEMA = "strideworld-worker-record-input-v1"
RESULT_SCHEMA = "strideworld-generation-result-v1"
EXECUTED_SCHEMA = "strideworld-executed-action-record-v1"
NATIVE_RECORD_SCHEMA = "strideworld-hy-world15-hunyuan-exact-final-native-v1"

TASK2_KIND = "TASK2_ATOMIC"
COMPLEX_KIND = "TASK1_TASK3_SHARED_COMPLEX"
CORE_ACTIONS = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "YAW_LEFT",
    "YAW_RIGHT",
)
ACTION_TOKEN = {
    "MOVE_FORWARD": "w",
    "MOVE_BACKWARD": "s",
    "MOVE_LEFT": "a",
    "MOVE_RIGHT": "d",
    "YAW_LEFT": "left",
    "YAW_RIGHT": "right",
}
FINAL_LENGTHS = (8, 11, 14)
FINAL_LENGTH_DISTRIBUTION = {8: 280, 11: 280, 14: 560}
CHECKPOINT_DOSES = tuple(range(1, 17))
CHECKPOINT_FRAMES = tuple(4 * value for value in CHECKPOINT_DOSES)
# NOOP is generation-only alignment padding.  It is never part of a benchmark
# measurement window, including the first terminal NOOP tick.
SCORED_TERMINAL_NONE_TICKS = 0
FRAMES_PER_LATENT_TICK = 4
FPS = 24.0
WIDTH = 832
HEIGHT = 480

# length: (total terminal none ticks, raw native frames, formal prefix frames)
LENGTH_ALIGNMENT = {
    8: (3, 45, 33),
    11: (4, 61, 45),
    14: (5, 77, 57),
    16: (3, 77, 65),
}

SOURCE_COMMIT = "1588e1336e842b03b0a7860c654ebd7c46bb065e"
SOURCE_ARCHIVE_SHA256 = "bb9dd3b4db53271aef1beb104dd99126f43f8858616dae8753ec9c970afd2591"
SOURCE_SHA256 = {
    "hyvideo/generate.py":
        "0ee7b50e9d4e9f93d627af409b1049c885f94c41c17004d5603d7e7164c819e3",
    "hyvideo/generate_custom_trajectory.py":
        "6fdb8c892d00a8cfe3d5fd23639be089f8f3d075c1e814c5cd980a8e52b4febd",
    "hyvideo/pipelines/worldplay_video_pipeline.py":
        "18e3237b261557e785c42fdfc01b66d2c3b5e420ad34c45dc846f89732cf6586",
    "run.sh":
        "938ddc02d60fc55da875f133a402b07f867a1a0ac441a5b043f54edd0de2bc13",
    "download_models.py":
        "4b59d65a262c8cc18abf704fb3b328f2e5814b83e4fe2dcaef5eb99a216b0fa1",
}
WEIGHT_MANIFEST_SHA256 = "f5d9fa2657d59f04ac0bfd8b6251c637606a29d5bebb6380f8ad82ad3861d1f0"
ALL_SHA_RECEIPT_SHA256 = "f914ba9a8d8d3357af844f6b566cc7aa3d6b6067d8122f7b51588716785d8634"
EXPECTED_WEIGHT_FILE_COUNT = 33
EXPECTED_WEIGHT_BYTES = 92_109_604_555
AR_RL_CHECKPOINT_SHA256 = "3c2af6d59b464dd39408ae47e5038635b830891fde9682aad7e0d89f619e146c"
AR_RL_CHECKPOINT_BYTES = 34_231_948_208

HOST_ROOT = Path("<path>/anon")
DEFAULT_DATA_ROOT = HOST_ROOT / "data"
DEFAULT_SOURCE_ROOT = HOST_ROOT / "lgt/third_party/strideworld/HY-WorldPlay-strict-1588e133"
DEFAULT_WEIGHT_ROOT = HOST_ROOT / "models/strideworld/weights/HY-World-1.5-exact"
DEFAULT_WEIGHT_MANIFEST = (
    HOST_ROOT
    / "lgt/logs/strideworld/model-download-exact/"
      "hy_world15_exact_frozen_33files_manifest_v1.json"
)
DEFAULT_ALL_SHA_RECEIPT = (
    HOST_ROOT
    / "models/strideworld_exact/audit/"
      "hy_world15_exact_scene01_case02_move_forward_d8_seed0_allsha_preflight_v1.json"
)
DEFAULT_ALL_SHA_MARKER = DEFAULT_ALL_SHA_RECEIPT.with_suffix(".complete")
DEFAULT_MODEL_ROOT = (
    DEFAULT_WEIGHT_ROOT
    / "HunyuanVideo-1.5_9b49404b3f5df2a8f0b31df27a0c7ab872e7b038"
)
DEFAULT_ACTION_CHECKPOINT = (
    HOST_ROOT
    / "work/benchmark_refresh_20261019/models/HY-WorldPlay/"
      "ar_rl_model/diffusion_pytorch_model.safetensors"
)
DEFAULT_ADAPTER = Path(__file__).with_name("worldplay_ar_rl_adapter_source.json")

SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NEGATIVE_PROMPT = "static, blurry, low quality, geometry distortion, abrupt cut"


class WorkerError(RuntimeError):
    """Raised before publication when any frozen contract disagrees."""


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise WorkerError(f"{label} must be a lowercase SHA256")
    return value


def require_safe_component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SAFE_COMPONENT.fullmatch(value):
        raise WorkerError(f"{label} is not a safe path component: {value!r}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WorkerError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise WorkerError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise WorkerError(f"cannot read JSONL {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise WorkerError(f"invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise WorkerError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def require_inside(root: Path, candidate: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise WorkerError(f"{label} escapes {resolved_root}: {resolved}") from error
    return resolved


def alignment_for_length(length: int) -> dict[str, int]:
    if length not in LENGTH_ALIGNMENT:
        raise WorkerError(f"unsupported final route length: {length}")
    none_ticks, raw_frames, formal_frames = LENGTH_ALIGNMENT[length]
    latent_count = (raw_frames - 1) // FRAMES_PER_LATENT_TICK + 1
    if latent_count % 4 != 0:
        raise WorkerError(f"official latent count is not divisible by four for L{length}")
    if latent_count != 1 + length + none_ticks:
        raise WorkerError(f"pose/native latent alignment drift for L{length}")
    expected_formal = length * FRAMES_PER_LATENT_TICK + 1
    if formal_frames != expected_formal or raw_frames < formal_frames:
        raise WorkerError(f"formal/native frame horizon drift for L{length}")
    return {
        "route_length": length,
        "terminal_none_ticks": none_ticks,
        "scored_terminal_none_ticks": SCORED_TERMINAL_NONE_TICKS,
        "unscored_alignment_none_ticks": none_ticks - SCORED_TERMINAL_NONE_TICKS,
        "latent_count": latent_count,
        "raw_frame_count": raw_frames,
        "formal_frame_count": formal_frames,
        "unscored_alignment_frame_count": raw_frames - formal_frames,
    }


def measurement_checkpoints() -> list[dict[str, Any]]:
    return [
        {
            "dose_requested": float(dose),
            "native_action_tick": dose,
            "frame_index": frame,
            "prefix_frame_slice_half_open": [0, frame + 1],
        }
        for dose, frame in zip(CHECKPOINT_DOSES, CHECKPOINT_FRAMES)
    ]


def _validate_condition(record: Mapping[str, Any]) -> None:
    condition = record.get("condition")
    if not isinstance(condition, dict):
        raise WorkerError(f"record {record.get('record_id')} has no condition object")
    uri = condition.get("image_uri")
    if not isinstance(uri, str) or not uri.startswith("data://"):
        raise WorkerError(f"record {record.get('record_id')} has an invalid data URI")
    relative = Path(uri.removeprefix("data://"))
    if relative.is_absolute() or ".." in relative.parts:
        raise WorkerError(f"record {record.get('record_id')} image URI escapes data root")
    require_sha256(condition.get("image_sha256"), "condition.image_sha256")
    prompt = condition.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise WorkerError(f"record {record.get('record_id')} has an empty prompt")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != condition.get("prompt_sha256"):
        raise WorkerError(f"record {record.get('record_id')} prompt SHA256 mismatch")
    width = condition.get("width")
    height = condition.get("height")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise WorkerError(f"record {record.get('record_id')} frozen image geometry is invalid")


def validate_record(record: Mapping[str, Any]) -> None:
    record_id = require_safe_component(record.get("record_id"), "record_id")
    require_safe_component(record.get("group_id"), "group_id")
    require_safe_component(record.get("benchmark_scene_id"), "benchmark_scene_id")
    require_safe_component(record.get("case_id"), "case_id")
    require_safe_component(record.get("state_id"), "state_id")
    if record.get("seed") not in (0, 1, 2):
        raise WorkerError(f"record {record_id} must use v10 seed 0, 1, or 2")
    if record.get("formal_metric_scoring_allowed") is not False:
        raise WorkerError(f"record {record_id} cannot claim formal scoring before GT binding")
    record_index = record.get("record_index")
    global_ordinal = record.get("portable_global_ordinal", record.get("global_ordinal"))
    case_ordinal = record.get("portable_ordinal_in_case", record.get("ordinal_in_case"))
    if (
        isinstance(record_index, bool)
        or not isinstance(record_index, int)
        or record_index < 0
        or global_ordinal != record_index + 1
        or isinstance(case_ordinal, bool)
        or not isinstance(case_ordinal, int)
        or not 1 <= case_ordinal <= 14
    ):
        raise WorkerError(f"record {record_id} portable ordinal/index drift")
    require_sha256(
        record.get("portable_record_content_sha256"),
        f"record {record_id} portable_record_content_sha256",
    )
    portable_output = record.get("portable_output_relpath")
    if not isinstance(portable_output, str):
        raise WorkerError(f"record {record_id} has no portable output path")
    portable_path = Path(portable_output)
    if (
        portable_path.is_absolute()
        or ".." in portable_path.parts
        or portable_path.name != record_id
    ):
        raise WorkerError(f"record {record_id} portable output path drift")
    _validate_condition(record)
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise WorkerError(f"record {record_id} has no steps")
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict) or step.get("step_index") != index:
            raise WorkerError(f"record {record_id} has a non-contiguous step index")
        actions = step.get("canonical_actions")
        if not isinstance(actions, list) or len(actions) != 1:
            raise WorkerError(f"record {record_id} must have exactly one action per step")
        if actions[0] not in ACTION_TOKEN:
            raise WorkerError(f"record {record_id} contains unsupported action {actions[0]!r}")

    kind = record.get("kind")
    if kind == TASK2_KIND:
        if record.get("portable_kind") != TASK2_KIND or not 1 <= case_ordinal <= 6:
            raise WorkerError(f"atomic record {record_id} portable kind/ordinal drift")
        if len(steps) != 1 or steps[0].get("dose") != 16.0:
            raise WorkerError(f"atomic record {record_id} must be one held dose-16 step")
        action = steps[0]["canonical_actions"][0]
        if record.get("benchmark_action_label") != action:
            raise WorkerError(f"atomic record {record_id} action label drift")
        if record.get("source_video_dose") != 16.0:
            raise WorkerError(f"atomic record {record_id} source dose drift")
        if record.get("checkpoint_doses") != [float(value) for value in CHECKPOINT_DOSES]:
            raise WorkerError(f"atomic record {record_id} checkpoint doses drift")
        if record.get("route_id") != "" or record.get("output_namespace") != "task2_atomic":
            raise WorkerError(f"atomic record {record_id} route/namespace drift")
        if record.get("metric_consumers") != ["TASK2_ATOMIC_RESPONSE"]:
            raise WorkerError(f"atomic record {record_id} metric consumer drift")
        if record.get("task2_dependency_ids") != []:
            raise WorkerError(f"atomic record {record_id} cannot depend on Task2 records")
        if record.get("one_continuous_video") is not True:
            raise WorkerError(f"atomic record {record_id} must be one continuous video")
        if record.get("intermediate_noop_forbidden") is not True:
            raise WorkerError(f"atomic record {record_id} intermediate NOOP gate drift")
    elif kind == COMPLEX_KIND:
        if record.get("portable_kind") != "SHARED_TASK1_TASK3" or not 7 <= case_ordinal <= 14:  # seed-0-only guard lifted
            raise WorkerError(f"complex record {record_id} portable kind/ordinal drift")
        length = record.get("route_length")
        if length not in FINAL_LENGTHS or length != len(steps):
            raise WorkerError(f"complex record {record_id} route length drift")
        if any(step.get("dose") != 1.0 for step in steps):
            raise WorkerError(f"complex record {record_id} must use dose=1 per step")
        if not record.get("route_id") or record.get("output_namespace") != "shared_task1_task3":
            raise WorkerError(f"complex record {record_id} route/namespace drift")
        if record.get("checkpoint_doses") != [] or record.get("benchmark_action_label") != "":
            raise WorkerError(f"complex record {record_id} cannot declare atomic checkpoints")
        if record.get("metric_consumers") != [
            "TASK1_VISIBLE_WORLD_STABILITY", "TASK3_OPEN_COMPOSITION",
        ]:
            raise WorkerError(f"complex record {record_id} metric consumer drift")
        dependencies = record.get("task2_dependency_ids")
        if not isinstance(dependencies, list) or not 1 <= len(dependencies) <= 6:
            raise WorkerError(f"complex record {record_id} must bind required action atomic dependencies")
        if len(set(dependencies)) != len(dependencies) or not all(isinstance(item, str) for item in dependencies):
            raise WorkerError(f"complex record {record_id} has invalid Task2 dependencies")
        require_sha256(
            record.get("source_route_content_sha256"),
            f"complex record {record_id} source_route_content_sha256",
        )
    else:
        raise WorkerError(f"record {record_id} has unsupported kind {kind!r}")


def build_native_request(record: Mapping[str, Any]) -> dict[str, Any]:
    validate_record(record)
    steps = record["steps"]
    kind = record["kind"]
    if kind == TASK2_KIND:
        route_length = 16
        action = steps[0]["canonical_actions"][0]
        action_tokens = [f"{ACTION_TOKEN[action]}-16"]
        boundaries = [{
            "step_index": 1,
            "action": action,
            "dose": 16.0,
            "native_token": ACTION_TOKEN[action],
            "boundary_frame_index": 64,
        }]
        checkpoints = measurement_checkpoints()
    else:
        route_length = len(steps)
        action_tokens = []
        boundaries = []
        for index, step in enumerate(steps, 1):
            action = step["canonical_actions"][0]
            token = ACTION_TOKEN[action]
            # Keep each segment explicit even when two adjacent actions match;
            # this preserves every benchmark action boundary in provenance.
            action_tokens.append(f"{token}-1")
            boundaries.append({
                "step_index": index,
                "action": action,
                "dose": 1.0,
                "native_token": token,
                "boundary_frame_index": FRAMES_PER_LATENT_TICK * index,
            })
        checkpoints = []
    alignment = alignment_for_length(route_length)
    pose = ",".join(
        action_tokens + [f"none-{alignment['terminal_none_ticks']}"]
    )
    request = {
        "record_id": record["record_id"],
        "kind": kind,
        "seed": int(record["seed"]),
        "pose": pose,
        "route_length": route_length,
        "video_length": alignment["raw_frame_count"],
        "latent_count": alignment["latent_count"],
        "formal_evaluation_frame_count": alignment["formal_frame_count"],
        "metric_frame_slice_half_open": [0, alignment["formal_frame_count"]],
        "terminal_none_ticks": alignment["terminal_none_ticks"],
        "scored_terminal_none_ticks": SCORED_TERMINAL_NONE_TICKS,
        "unscored_alignment_none_ticks": alignment[
            "unscored_alignment_none_ticks"
        ],
        "unscored_alignment_frame_count": alignment[
            "unscored_alignment_frame_count"
        ],
        "action_boundary_frames": boundaries,
        "checkpoint_doses": (
            [float(value) for value in CHECKPOINT_DOSES]
            if kind == TASK2_KIND else []
        ),
        "checkpoint_frame_indices": (
            list(CHECKPOINT_FRAMES) if kind == TASK2_KIND else []
        ),
        "measurement_checkpoints": checkpoints,
        "single_pipeline_call": True,
        "single_pose_to_input_call": True,
        "external_rgb_chaining": False,
        "current_camera_local_composition": True,
    }
    request["native_request_sha256"] = canonical_sha256(request)
    return request


def validate_complete_record_set(
    records: Sequence[Mapping[str, Any]], *, require_complete_campaign: bool,
) -> dict[str, Any]:
    if not records:
        raise WorkerError("final record set is empty")
    ids = [record.get("record_id") for record in records]
    if len(ids) != len(set(ids)):
        raise WorkerError("final record set contains duplicate record IDs")
    for record in records:
        validate_record(record)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["group_id"]), []).append(record)
    for group_id, group in grouped.items():
        group_seed = int(group[0]["seed"])
        expected_count = 14 if group_seed == 0 else 6
        if len(group) != expected_count:
            raise WorkerError(
                f"v10 group {group_id} seed{group_seed} must contain {expected_count} records"
            )
        atomic = [record for record in group if record["kind"] == TASK2_KIND]
        complex_rows = [record for record in group if record["kind"] == COMPLEX_KIND]
        expected_complex = 8 if group_seed == 0 else 0
        if len(atomic) != 6 or len(complex_rows) != expected_complex:
            raise WorkerError(
                f"v10 group {group_id} must contain 6 atomic + {expected_complex} complex"
            )
        if {record.get("portable_ordinal_in_case", record.get("ordinal_in_case")) for record in group} != set(range(1, expected_count + 1)):
            raise WorkerError(f"v10 group {group_id} portable ordinal coverage drift")
        action_to_id = {
            record["steps"][0]["canonical_actions"][0]: record["record_id"]
            for record in atomic
        }
        if set(action_to_id) != set(CORE_ACTIONS):
            raise WorkerError(f"final group {group_id} atomic action coverage drift")
        atomic_ids = set(action_to_id.values())
        identity_fields = (
            "benchmark_scene_id", "case_id", "scene_id", "state_id", "seed",
        )
        reference = group[0]
        for record in group[1:]:
            if any(record.get(key) != reference.get(key) for key in identity_fields):
                raise WorkerError(f"final group {group_id} identity drift")
            if record.get("condition") != reference.get("condition"):
                raise WorkerError(f"final group {group_id} condition drift")
        for record in complex_rows:
            required_ids = {action_to_id[a] for step in record["steps"] for a in step["canonical_actions"]}
            if set(record["task2_dependency_ids"]) != required_ids:
                raise WorkerError(
                    f"final complex record {record['record_id']} dependency coverage drift"
                )
    complex_lengths = [
        int(record["route_length"])
        for record in records if record["kind"] == COMPLEX_KIND
    ]
    summary = {
        "record_count": len(records),
        "case_group_count": len(grouped),
        "atomic_record_count": sum(record["kind"] == TASK2_KIND for record in records),
        "complex_record_count": len(complex_lengths),
        "complex_length_distribution": {
            length: complex_lengths.count(length) for length in FINAL_LENGTHS
        },
    }
    if require_complete_campaign:
        if {record["record_index"] for record in records} != set(range(3640)):
            raise WorkerError("complete v10 campaign record-index coverage drift")
        if {record.get("portable_global_ordinal", record.get("global_ordinal")) for record in records} != set(range(1, 3641)):
            raise WorkerError("complete v10 campaign global ordinal coverage drift")
        if summary != {
            "record_count": 3640,
            "case_group_count": 420,
            "atomic_record_count": 2520,
            "complex_record_count": 1120,
            "complex_length_distribution": FINAL_LENGTH_DISTRIBUTION,
        }:
            raise WorkerError(f"complete final campaign distribution drift: {summary}")
    return summary


def _expected_record_output_dir(record: Mapping[str, Any]) -> str:
    components = (
        "records",
        record["benchmark_scene_id"],
        record["case_id"],
        record["output_namespace"],
        record["record_id"],
    )
    for component in components[1:]:
        require_safe_component(component, "record output path component")
    return Path(*components).as_posix()


def validate_worker_input(
    row: Mapping[str, Any], plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    if row.get("schema_version") != WORKER_INPUT_SCHEMA:
        raise WorkerError("worker input schema mismatch")
    expected = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_sha256": CAMPAIGN_SHA256,
        "plan_sha256": plan["plan_sha256"],
        "model_id": MODEL_ID,
        "required_result_schema": RESULT_SCHEMA,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise WorkerError(f"worker input {key} mismatch")
    record = row.get("record")
    if not isinstance(record, dict):
        raise WorkerError("worker input has no record object")
    validate_record(record)
    expected_dir = _expected_record_output_dir(record)
    if row.get("record_output_dir") != expected_dir:
        raise WorkerError(f"record {record['record_id']} output directory drift")
    if row.get("required_generation_record") != f"{expected_dir}/generation_record.json":
        raise WorkerError(f"record {record['record_id']} generation result path drift")
    return record


def validate_shard_plan(
    plan: Mapping[str, Any], output_root: Path,
) -> None:
    expected = {
        "schema_version": SHARD_PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "campaign_sha256": CAMPAIGN_SHA256,
        "model_id": MODEL_ID,
        "all_records": "all_records.jsonl",
        "runnable_records": "runnable_records.jsonl",
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise WorkerError(f"shard plan {key} mismatch")
    if Path(str(plan.get("output_root"))).resolve() != output_root.resolve():
        raise WorkerError("shard plan/output-root mismatch")
    require_sha256(plan.get("plan_sha256"), "shard plan plan_sha256")
    if not isinstance(plan.get("shard_index"), int) or not isinstance(
        plan.get("shard_count"), int
    ):
        raise WorkerError("shard plan index/count is invalid")
    if not 0 <= plan["shard_index"] < plan["shard_count"]:
        raise WorkerError("shard plan index is outside its shard count")
    gpu_group = plan.get("gpu_group")
    if (
        not isinstance(gpu_group, list)
        or len(gpu_group) != 1
        or isinstance(gpu_group[0], bool)
        or not isinstance(gpu_group[0], int)
        or gpu_group[0] < 0
    ):
        raise WorkerError("HY exact worker requires one physical GPU per shard")
    for count_key in ("all_record_count", "runnable_record_count"):
        value = plan.get(count_key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WorkerError(f"shard plan {count_key} is invalid")
    groups = plan.get("case_group_ids")
    if not isinstance(groups, list) or len(groups) != len(set(groups)):
        raise WorkerError("shard plan case-group list is invalid")
    for group in groups:
        require_safe_component(group, "shard plan group_id")


def validate_environment(plan: Mapping[str, Any], *, required: bool) -> None:
    expected = {
        "SW_CAMPAIGN_ID": CAMPAIGN_ID,
        "SW_CAMPAIGN_SHA256": CAMPAIGN_SHA256,
        "SW_PLAN_SHA256": str(plan["plan_sha256"]),
        "SW_MODEL_ID": MODEL_ID,
        "SW_SHARD_INDEX": str(plan["shard_index"]),
        "SW_GPU_GROUP": str(plan["gpu_group"][0]),
        "CUDA_VISIBLE_DEVICES": str(plan["gpu_group"][0]),
    }
    for key, value in expected.items():
        actual = os.environ.get(key)
        if actual is None and not required:
            continue
        if actual != value:
            raise WorkerError(
                f"worker environment mismatch for {key}: expected={value!r} actual={actual!r}"
            )


def validate_work(
    shard_plan_path: Path, shard_manifest_path: Path, output_root: Path,
    *, require_environment: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    plan_path = require_inside(output_root, shard_plan_path, "shard plan")
    pending_path = require_inside(output_root, shard_manifest_path, "pending manifest")
    plan = read_json(plan_path)
    validate_shard_plan(plan, output_root)
    validate_environment(plan, required=require_environment)

    immutable_path = require_inside(
        plan_path.parent, plan_path.parent / str(plan["all_records"]),
        "immutable all-record manifest",
    )
    all_rows = read_jsonl(immutable_path)
    if len(all_rows) != plan["all_record_count"]:
        raise WorkerError("immutable shard record count disagrees with shard plan")
    records = [validate_worker_input(row, plan) for row in all_rows]
    shard_summary = validate_complete_record_set(records, require_complete_campaign=False)
    if set(plan["case_group_ids"]) != {str(record["group_id"]) for record in records}:
        raise WorkerError("immutable shard records disagree with assigned case groups")

    pending_rows = read_jsonl(pending_path)
    if not pending_rows or len(pending_rows) > plan["runnable_record_count"]:
        raise WorkerError("pending manifest count is empty or exceeds shard plan")
    all_by_id = {str(row["record"]["record_id"]): row for row in all_rows}
    pending_ids: list[str] = []
    for row in pending_rows:
        record = validate_worker_input(row, plan)
        record_id = str(record["record_id"])
        if record_id in pending_ids:
            raise WorkerError("pending manifest contains duplicate record IDs")
        pending_ids.append(record_id)
        if record_id not in all_by_id or canonical_bytes(row) != canonical_bytes(all_by_id[record_id]):
            raise WorkerError(f"pending record {record_id} differs from immutable shard input")
    return plan, all_rows, pending_rows, shard_summary


def resolve_condition_image(
    record: Mapping[str, Any], data_root: Path, sha_cache: dict[Path, str],
) -> Path:
    uri = str(record["condition"]["image_uri"])
    image = require_inside(
        data_root, data_root / uri.removeprefix("data://"), "condition image",
    )
    if not image.is_file() or image.is_symlink():
        raise WorkerError(f"condition image is missing or unsafe: {image}")
    actual = sha_cache.get(image)
    if actual is None:
        actual = sha_cache[image] = sha256_file(image)
    if actual != record["condition"]["image_sha256"]:
        raise WorkerError(f"condition image SHA256 mismatch for {record['record_id']}")
    return image


def _safe_manifest_target(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise WorkerError(f"unsafe weight manifest path: {relative!r}")
    return require_inside(root, root / candidate, "weight manifest target")


def validate_adapter(path: Path) -> dict[str, Any]:
    adapter_path = path.expanduser()
    if not adapter_path.is_file() or adapter_path.is_symlink():
        raise WorkerError(f"WorldPlay final adapter is missing or unsafe: {adapter_path}")
    adapter_path = adapter_path.resolve()
    adapter = read_json(adapter_path)
    exact_top_level = {
        "schema_version": "strideworld-v10-native-adapter-v1",
        "adapter_id": "worldplay_ar_rl_official_v1",
        "model_id": MODEL_ID,
        "model_variant": MODEL_VARIANT,
        "interface_track": "official_pose_string",
        "representation": "ordered_discrete_pose_tokens",
        "action_reference_frame": "current_camera_body",
    }
    for key, expected in exact_top_level.items():
        if adapter.get(key) != expected:
            raise WorkerError(f"WorldPlay final adapter {key} drift")
    if adapter.get("seed_policy") != {
        "seeds": [0, 1, 2], "reset_for_every_independent_video": True,
    }:
        raise WorkerError("WorldPlay final adapter seed policy drift")

    timing = adapter.get("timing")
    if not isinstance(timing, dict):
        raise WorkerError("WorldPlay final adapter has no timing contract")
    timing_expected = {
        "fps": 24,
        "conditioning_frames": 1,
        "decoded_intervals_per_native_tick": FRAMES_PER_LATENT_TICK,
        "atomic_held_dose": 16,
        "atomic_action_ticks": 16,
        "checkpoint_doses": list(CHECKPOINT_DOSES),
        "checkpoint_frame_indices": list(CHECKPOINT_FRAMES),
        "complex_exact_lengths": list(FINAL_LENGTHS),
        "terminal_scored_noop_ticks": SCORED_TERMINAL_NONE_TICKS,
    }
    for key, expected in timing_expected.items():
        if timing.get(key) != expected:
            raise WorkerError(f"WorldPlay final adapter timing.{key} drift")
    length_contracts = timing.get("length_contracts")
    expected_length_contracts = {
        str(length): {
            "raw_frames": raw,
            "formal_frames": formal,
            "total_noop_ticks": none,
        }
        for length, (none, raw, formal) in LENGTH_ALIGNMENT.items()
    }
    if length_contracts != expected_length_contracts:
        raise WorkerError("WorldPlay final adapter length contract drift")

    actions = adapter.get("actions")
    expected_actions = {
        **{action: {"token": token} for action, token in ACTION_TOKEN.items()},
        "NO_OP": {"token": "none"},
    }
    if actions != expected_actions:
        raise WorkerError("WorldPlay final adapter action-token mapping drift")
    native_control = adapter.get("native_control")
    if not isinstance(native_control, dict) or not (
        native_control.get("single_native_pipeline_call") is True
        and native_control.get("external_rgb_chaining") is False
        and native_control.get("official_alignment_rule")
        == "latent_pose_count_divisible_by_4"
        and native_control.get("translation_per_native_tick") == 0.08
        and native_control.get("yaw_degrees_per_native_tick") == 3.0
    ):
        raise WorkerError("WorldPlay final adapter native-control contract drift")
    capability = adapter.get("capability")
    if not isinstance(capability, dict) or not (
        capability.get("task2_atomic") == "READY"
        and capability.get("supported_complex_exact_lengths") == list(FINAL_LENGTHS)
        and capability.get("simultaneous_actions") is False
    ):
        raise WorkerError("WorldPlay final adapter capability contract drift")
    return {
        "adapter_id": adapter["adapter_id"],
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_path),
    }


def validate_exact_runtime(args: argparse.Namespace) -> dict[str, Any]:
    adapter_binding = validate_adapter(args.adapter)
    source_root = args.source_root.resolve()
    source_metadata = {
        ".source_revision": SOURCE_COMMIT,
        ".source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        ".source_repository": "https://github.com/Tencent-Hunyuan/HY-WorldPlay.git",
        ".source_transport": "codeload archive pinned by full commit SHA",
    }
    for relative, expected in source_metadata.items():
        path = source_root / relative
        if not path.is_file() or path.read_text(encoding="utf-8").strip() != expected:
            raise WorkerError(f"pinned source metadata mismatch: {relative}")
    source_files = []
    for relative, expected in SOURCE_SHA256.items():
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise WorkerError(f"pinned source file is missing/unsafe: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise WorkerError(f"pinned source SHA256 mismatch: {relative}")
        source_files.append({"relative_path": relative, "sha256": actual})
    generate_text = (source_root / "hyvideo/generate.py").read_text(encoding="utf-8")
    required_literals = (
        "forward_speed = 0.08",
        "yaw_speed = np.deg2rad(3)",
        'commands = [cmd.strip() for cmd in pose_string.split(",")]',
        "pose_string_to_json(pose_data)",
        "generate_camera_trajectory_local(motions)",
        'imageio.mimwrite(path, vid, fps=24)',
    )
    missing = [literal for literal in required_literals if literal not in generate_text]
    if missing:
        raise WorkerError(f"official WorldPlay source interface drift: {missing}")

    receipt_path = args.all_sha_receipt.resolve()
    receipt_sha = sha256_file(receipt_path) if receipt_path.is_file() else ""
    if receipt_sha != ALL_SHA_RECEIPT_SHA256:
        raise WorkerError(
            f"HY exact all-SHA receipt drift: expected={ALL_SHA_RECEIPT_SHA256} "
            f"actual={receipt_sha or 'missing'}"
        )
    receipt = read_json(receipt_path)
    embedded_sha = receipt.get("plan_sha256")
    unhashed_receipt = dict(receipt)
    unhashed_receipt.pop("plan_sha256", None)
    if embedded_sha != canonical_sha256(unhashed_receipt):
        raise WorkerError("HY exact all-SHA receipt embedded plan SHA256 mismatch")
    weights_receipt = receipt.get("weights")
    if not isinstance(weights_receipt, dict) or not (
        receipt.get("status") == "READY_FOR_GPU_RUNNER"
        # The frozen receipt binds the shared HunyuanVideo-1.5 base tree.  The
        # AR-RL transformer is an additional official checkpoint validated
        # independently below, so it must not inherit the distilled runner's
        # variant label.
        and receipt.get("model_variant") == "HY-World1.5-AR-480P-I2V-distill"
        and weights_receipt.get("ready") is True
        and weights_receipt.get("sha_policy") == "all_files"
        and weights_receipt.get("complete_file_count") == EXPECTED_WEIGHT_FILE_COUNT
        and weights_receipt.get("complete_bytes") == EXPECTED_WEIGHT_BYTES
        and not weights_receipt.get("missing")
        and not weights_receipt.get("bad_size")
        and not weights_receipt.get("bad_sha")
    ):
        raise WorkerError("HY exact all-SHA receipt is not complete and generation-ready")
    receipt_weight_root = Path(str(weights_receipt.get("root"))).resolve()
    receipt_source = receipt.get("source")
    if not isinstance(receipt_source, dict) or (
        receipt_source.get("commit") != SOURCE_COMMIT
    ):
        raise WorkerError("all-SHA receipt source binding drift")

    marker_path = args.all_sha_marker.resolve()
    if not marker_path.is_file():
        raise WorkerError(f"HY exact all-SHA completion marker is missing: {marker_path}")
    marker_tokens = marker_path.read_text(encoding="utf-8").split()
    if not marker_tokens or marker_tokens[0] != receipt_sha:
        raise WorkerError("HY exact all-SHA completion marker disagrees with receipt")

    manifest_path = args.weight_manifest.resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != WEIGHT_MANIFEST_SHA256:
        raise WorkerError("HY exact frozen weight manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    manifest_rows = manifest.get("records")
    if not isinstance(manifest_rows, list) or not (
        manifest.get("expected_file_count") == EXPECTED_WEIGHT_FILE_COUNT
        and len(manifest_rows) == EXPECTED_WEIGHT_FILE_COUNT
        and manifest.get("expected_total_bytes") == EXPECTED_WEIGHT_BYTES
    ):
        raise WorkerError("HY exact frozen weight manifest cardinality drift")
    if weights_receipt.get("manifest_sha256") != WEIGHT_MANIFEST_SHA256:
        raise WorkerError("all-SHA receipt and frozen weight manifest disagree")

    weight_root = args.weight_root.resolve()
    weight_root_matches_receipt = weight_root == receipt_weight_root
    manifest_by_path: dict[Path, Mapping[str, Any]] = {}
    seen_relative: set[str] = set()
    for row in manifest_rows:
        if not isinstance(row, dict):
            raise WorkerError("weight manifest row is not an object")
        relative = row.get("target_relative_path")
        if not isinstance(relative, str) or not relative or relative in seen_relative:
            raise WorkerError("weight manifest target path is invalid or duplicated")
        seen_relative.add(relative)
        target = _safe_manifest_target(weight_root, relative)
        expected_size = row.get("size_bytes")
        if not target.is_file() or target.stat().st_size != expected_size:
            raise WorkerError(f"all-SHA-bound weight is missing or size-drifted: {target}")
        expected_sha = require_sha256(
            row.get("sha256"), f"weight manifest {relative} sha256",
        )
        # The frozen receipt proves all-file SHA only at its bound root.  A
        # CLI-overridden copied root remains supported, but must earn the same
        # guarantee by rehashing every file instead of inheriting provenance
        # from a different path.
        if not weight_root_matches_receipt and sha256_file(target) != expected_sha:
            raise WorkerError(f"overridden all-SHA weight content drift: {target}")
        manifest_by_path[target] = row

    model_root = args.model_root.resolve()
    checkpoint = args.action_checkpoint.resolve()
    if not model_root.is_dir() or not checkpoint.is_file():
        raise WorkerError("official Hunyuan base or AR-RL checkpoint is missing")
    require_inside(weight_root, model_root, "model root")
    if checkpoint.stat().st_size != AR_RL_CHECKPOINT_BYTES:
        raise WorkerError("official AR-RL checkpoint size mismatch")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != AR_RL_CHECKPOINT_SHA256:
        raise WorkerError("official AR-RL checkpoint SHA256 mismatch")
    model_rows = [path for path in manifest_by_path if model_root in path.parents]
    if not model_rows:
        raise WorkerError("base model root is not covered by the all-SHA manifest")
    return {
        **adapter_binding,
        "model_variant": MODEL_VARIANT,
        "source_root": str(source_root),
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_files": source_files,
        "weight_root": str(weight_root),
        "weight_manifest": str(manifest_path),
        "weight_manifest_sha256": WEIGHT_MANIFEST_SHA256,
        "all_sha_receipt": str(receipt_path),
        "all_sha_receipt_sha256": receipt_sha,
        "all_sha_marker": str(marker_path),
        "weight_file_count": EXPECTED_WEIGHT_FILE_COUNT,
        "weight_bytes": EXPECTED_WEIGHT_BYTES,
        "weight_validation_policy": (
            "frozen_all_sha_receipt_plus_current_size"
            if weight_root_matches_receipt
            else "all_files_sha256_reverified_for_overridden_root"
        ),
        "action_checkpoint": str(checkpoint),
        "action_checkpoint_sha256": checkpoint_sha256,
        "model_root": str(model_root),
    }


def pipeline_scalar_kwargs(
    request: Mapping[str, Any], image: Path, prompt: str,
) -> dict[str, Any]:
    return {
        "enable_sr": False,
        "prompt": prompt,
        "aspect_ratio": "16:9",
        "num_inference_steps": 40,
        "sr_num_inference_steps": None,
        "video_length": request["video_length"],
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": int(request["seed"]),
        "output_type": "pt",
        "prompt_rewrite": False,
        "return_pre_sr_video": False,
        "reference_image": str(image.resolve()),
        "few_step": False,
        "chunk_latent_frames": 4,
        "model_type": "ar",
        "user_height": HEIGHT,
        "user_width": WIDTH,
        "transformer_resident_ar_rollout": True,
    }


def load_official_pipeline(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    if os.environ.get("WORLD_SIZE") != "1" or os.environ.get("LOCAL_RANK") not in {
        None, "0",
    }:
        raise WorkerError("HY exact worker must use one-process torchrun")
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))
    # Deliberately delayed until every CPU validation and --validate-only gate
    # has completed.
    from hyvideo import generate as native  # type: ignore

    torch = native.torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise WorkerError("HY exact worker requires exactly one visible CUDA GPU")
    if "GPU" not in torch.cuda.get_device_name(0).upper():
        raise WorkerError("HY exact accepted runtime is frozen to one GPU GPU")
    native.initialize_infer_state(SimpleNamespace(
        sage_blocks_range="0-53",
        enable_torch_compile=False,
        use_fp8_gemm=False,
        quant_type="fp8-per-block",
        include_patterns="double_blocks",
        use_vae_parallel=False,
        use_sageattn=False,
    ))
    pipe = native.HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=str(args.model_root.resolve()),
        transformer_version="480p_i2v",
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=torch.bfloat16,
        enable_offloading=True,
        enable_group_offloading=None,
        action_ckpt=str(args.action_checkpoint.resolve()),
    )
    return native, pipe, torch


def probe_video(path: Path, expected_frames: int) -> dict[str, Any]:
    try:
        import av
    except ImportError as error:
        raise WorkerError("PyAV is required for decoded-frame validation") from error
    try:
        container = av.open(str(path), mode="r")
    except Exception as error:
        raise WorkerError(f"PyAV could not open generated video {path}: {error}") from error
    with container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            raise WorkerError("generated MP4 does not have exactly one video stream")
        stream = streams[0]
        rate = stream.average_rate or stream.base_rate or stream.guessed_rate
        if rate is None or abs(float(rate) - FPS) > 1e-6:
            raise WorkerError(f"generated video FPS drift: {rate}")
        frame_count = 0
        for frame in container.decode(stream):
            if (frame.width, frame.height) != (WIDTH, HEIGHT):
                raise WorkerError(
                    f"generated frame dimensions drift: {frame.width}x{frame.height}"
                )
            frame_count += 1
    if frame_count != expected_frames:
        raise WorkerError(
            f"generated frame count drift: expected={expected_frames} actual={frame_count}"
        )
    return {
        "decoded_frame_count": frame_count,
        "fps": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "decoder": "PyAV_full_decode",
    }


def _post_required_padding(request: Mapping[str, Any]) -> dict[str, Any] | None:
    formal = int(request["formal_evaluation_frame_count"])
    raw = int(request["video_length"])
    if raw == formal:
        return None
    return {
        "start_frame": formal,
        "end_frame_exclusive": raw,
        "role": "unscored_post_required_padding",
        "control_semantics": "explicit_terminal_none",
        "excluded_from_scoring": True,
    }


def _controller_result(
    row: Mapping[str, Any], plan: Mapping[str, Any], artifact_dir: Path,
) -> dict[str, Any]:
    video = artifact_dir / "raw_generated.mp4"
    native_record = artifact_dir / "native_generation_record.json"
    executed = artifact_dir / "executed_action_record.json"
    for path in (video, native_record, executed):
        if not path.is_file() or path.stat().st_size <= 0:
            raise WorkerError(f"published artifact is missing or empty: {path}")
    native_record_payload = read_json(native_record)
    for key in ("adapter_id", "adapter_path", "adapter_sha256"):
        if not native_record_payload.get(key):
            raise WorkerError(f"native generation record omitted {key}")
    runtime_provenance = native_record_payload.get("runtime_provenance")
    runtime_provenance_sha256 = native_record_payload.get(
        "runtime_provenance_sha256"
    )
    if not isinstance(runtime_provenance, dict) or (
        runtime_provenance_sha256 != canonical_sha256(runtime_provenance)
    ):
        raise WorkerError("native generation record runtime provenance drift")
    required_runtime_fields = (
        "source_commit",
        "weight_manifest_sha256",
        "all_sha_receipt_sha256",
        "action_checkpoint_sha256",
    )
    for key in required_runtime_fields:
        if not runtime_provenance.get(key):
            raise WorkerError(f"native generation record runtime provenance omitted {key}")
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "SUCCEEDED",
        "record_id": row["record"]["record_id"],
        "model_id": MODEL_ID,
        "campaign_sha256": CAMPAIGN_SHA256,
        "plan_sha256": plan["plan_sha256"],
        "video_path": "artifact/raw_generated.mp4",
        "video_sha256": sha256_file(video),
        "native_generation_record_path": "artifact/native_generation_record.json",
        "native_generation_record_sha256": sha256_file(native_record),
        "executed_action_record_path": "artifact/executed_action_record.json",
        "executed_action_record_sha256": sha256_file(executed),
        "adapter_id": native_record_payload["adapter_id"],
        "adapter_path": native_record_payload["adapter_path"],
        "adapter_sha256": native_record_payload["adapter_sha256"],
        "runtime_provenance_sha256": runtime_provenance_sha256,
        "source_commit": runtime_provenance["source_commit"],
        "weight_manifest_sha256": runtime_provenance["weight_manifest_sha256"],
        "all_sha_receipt_sha256": runtime_provenance["all_sha_receipt_sha256"],
        "action_checkpoint_sha256": runtime_provenance[
            "action_checkpoint_sha256"
        ],
        "evidence_status": "RGB_GENERATED_FORMAL_GT_SCORING_PENDING",
        "formal_metric_scoring_allowed": False,
        "published_at": utc_now(),
    }


def _validate_existing_artifact(
    row: Mapping[str, Any], plan: Mapping[str, Any], request: Mapping[str, Any],
    artifact_dir: Path, runtime: Mapping[str, Any],
) -> None:
    expected_result = _controller_result(row, plan, artifact_dir)
    native_record = read_json(artifact_dir / "native_generation_record.json")
    executed = read_json(artifact_dir / "executed_action_record.json")
    video_sha = expected_result["video_sha256"]
    runtime_sha = canonical_sha256(runtime)
    source_record_sha = canonical_sha256(row["record"])
    condition_sha = row["record"]["condition"]["image_sha256"]
    prompt_sha = row["record"]["condition"]["prompt_sha256"]
    if (
        native_record.get("schema_version") != NATIVE_RECORD_SCHEMA
        or native_record.get("status") != "COMPLETED"
        or executed.get("execution_record_schema_version") != EXECUTED_SCHEMA
    ):
        raise WorkerError("existing artifact schema/status mismatch")
    for payload, label in ((native_record, "native"), (executed, "executed")):
        if payload.get("record_id") != row["record"]["record_id"]:
            raise WorkerError(f"existing {label} artifact record identity mismatch")
        if payload.get("model_id") != MODEL_ID:
            raise WorkerError(f"existing {label} artifact model identity mismatch")
        if payload.get("campaign_sha256") != CAMPAIGN_SHA256:
            raise WorkerError(f"existing {label} artifact campaign identity mismatch")
        if payload.get("plan_sha256") != plan["plan_sha256"]:
            raise WorkerError(f"existing {label} artifact plan identity mismatch")
        if payload.get("generated_video_sha256") != video_sha:
            raise WorkerError(f"existing {label} artifact video SHA256 mismatch")
        if payload.get("condition_image_sha256") != condition_sha:
            raise WorkerError(f"existing {label} artifact condition SHA256 mismatch")
        if payload.get("prompt_sha256") != prompt_sha:
            raise WorkerError(f"existing {label} artifact prompt SHA256 mismatch")
        if payload.get("source_campaign_record_sha256") != source_record_sha:
            raise WorkerError(
                f"existing {label} artifact source campaign record SHA256 mismatch"
            )
        for key in ("adapter_id", "adapter_path", "adapter_sha256"):
            if payload.get(key) != runtime[key]:
                raise WorkerError(f"existing {label} artifact {key} mismatch")
        provenance = payload.get("runtime_provenance")
        if not isinstance(provenance, dict) or canonical_bytes(provenance) != canonical_bytes(runtime):
            raise WorkerError(f"existing {label} artifact runtime provenance mismatch")
        if payload.get("runtime_provenance_sha256") != runtime_sha:
            raise WorkerError(
                f"existing {label} artifact runtime provenance SHA256 mismatch"
            )
        if payload.get("native_request_sha256") != request["native_request_sha256"]:
            raise WorkerError(f"existing {label} artifact request SHA256 mismatch")
    decoded = probe_video(
        artifact_dir / "raw_generated.mp4", int(request["video_length"]),
    )
    native_output = native_record.get("execution_output")
    executed_output = executed.get("execution_output")
    if not isinstance(native_output, dict) or not isinstance(executed_output, dict):
        raise WorkerError("existing artifact omitted execution output metadata")
    for key, expected in decoded.items():
        if native_output.get(key) != expected or executed_output.get(key) != expected:
            raise WorkerError(f"existing artifact decoded {key} metadata mismatch")


def _publish_controller_result(
    row: Mapping[str, Any], plan: Mapping[str, Any], artifact_dir: Path,
    output_root: Path,
) -> None:
    result = _controller_result(row, plan, artifact_dir)
    result_path = require_inside(
        output_root,
        output_root / str(row["required_generation_record"]),
        "controller generation result",
    )
    if result_path.exists():
        existing = read_json(result_path)
        # ``published_at`` is intentionally stable once committed.
        comparable = dict(result)
        comparable["published_at"] = existing.get("published_at")
        if canonical_bytes(existing) != canonical_bytes(comparable):
            raise WorkerError(f"existing generation result differs: {result_path}")
        return
    atomic_write_json(result_path, result)


def generate_one(
    native: Any,
    pipe: Any,
    torch: Any,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    image: Path,
    runtime: Mapping[str, Any],
    output_root: Path,
) -> str:
    record = row["record"]
    record_id = str(record["record_id"])
    record_dir = require_inside(
        output_root, output_root / str(row["record_output_dir"]), "record output directory",
    )
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = record_dir / "artifact"
    if artifact_dir.exists():
        _validate_existing_artifact(row, plan, request, artifact_dir, runtime)
        _publish_controller_result(row, plan, artifact_dir, output_root)
        return "RESUMED"

    stage = record_dir / f".artifact.{os.getpid()}.tmp"
    if stage.exists():
        raise WorkerError(f"stale artifact staging directory: {stage}")
    stage.mkdir()
    try:
        viewmats, intrinsics, action_label = native.pose_to_input(
            request["pose"], request["latent_count"],
        )
        kwargs = pipeline_scalar_kwargs(request, image, record["condition"]["prompt"])
        # One ordered pose, one official pipeline call, no RGB chain.
        output = pipe(
            **kwargs,
            viewmats=viewmats.unsqueeze(0),
            Ks=intrinsics.unsqueeze(0),
            action=action_label.unsqueeze(0),
        )
        if not hasattr(output, "videos") or output.videos is None:
            raise WorkerError(f"official pipeline returned no video for {record_id}")
        video = stage / "raw_generated.mp4"
        native.save_video(output.videos, str(video))
        decoded = probe_video(video, int(request["video_length"]))
        video_sha = sha256_file(video)
        pipeline_kwargs_record = dict(kwargs)
        pipeline_kwargs_record["reference_image"] = str(image)
        native_record = {
            "schema_version": NATIVE_RECORD_SCHEMA,
            "status": "COMPLETED",
            "record_id": record_id,
            "model_id": MODEL_ID,
            "model_variant": MODEL_VARIANT,
            "adapter_id": runtime["adapter_id"],
            "adapter_path": runtime["adapter_path"],
            "adapter_sha256": runtime["adapter_sha256"],
            "campaign_sha256": CAMPAIGN_SHA256,
            "plan_sha256": plan["plan_sha256"],
            "shard_index": plan["shard_index"],
            "seed": int(record["seed"]),
            "condition_image_path": str(image),
            "condition_image_sha256": record["condition"]["image_sha256"],
            "prompt_sha256": record["condition"]["prompt_sha256"],
            "source_campaign_record_sha256": canonical_sha256(record),
            "native_request": dict(request),
            "native_request_sha256": request["native_request_sha256"],
            "pipeline_scalar_kwargs": pipeline_kwargs_record,
            "single_pipeline_call": True,
            "external_rgb_chaining": False,
            "pipeline_load_count_for_shard": 1,
            "runtime_provenance": dict(runtime),
            "runtime_provenance_sha256": canonical_sha256(runtime),
            "generated_video_path": "raw_generated.mp4",
            "generated_video_sha256": video_sha,
            "execution_output": {
                **decoded,
                "formal_evaluation_frame_count": request[
                    "formal_evaluation_frame_count"
                ],
                "metric_frame_slice_half_open": request[
                    "metric_frame_slice_half_open"
                ],
                "action_boundary_frames": request["action_boundary_frames"],
                "measurement_checkpoints": request["measurement_checkpoints"],
                "post_required_padding": _post_required_padding(request),
            },
        }
        executed = {
            **record,
            "execution_record_schema_version": EXECUTED_SCHEMA,
            "record_id": record_id,
            "model_id": MODEL_ID,
            "adapter_id": runtime["adapter_id"],
            "adapter_path": runtime["adapter_path"],
            "adapter_sha256": runtime["adapter_sha256"],
            "campaign_sha256": CAMPAIGN_SHA256,
            "plan_sha256": plan["plan_sha256"],
            "source_campaign_record_sha256": canonical_sha256(record),
            "condition_image_sha256": record["condition"]["image_sha256"],
            "prompt_sha256": record["condition"]["prompt_sha256"],
            "generated_video_sha256": video_sha,
            "checkpoint_fingerprint": runtime["action_checkpoint_sha256"],
            "checkpoint_fingerprint_source": "all_sha_weight_manifest",
            "native_request_sha256": request["native_request_sha256"],
            "runtime_provenance": dict(runtime),
            "runtime_provenance_sha256": canonical_sha256(runtime),
            "execution_output": native_record["execution_output"],
        }
        atomic_write_json(stage / "native_generation_record.json", native_record)
        atomic_write_json(stage / "executed_action_record.json", executed)
        os.replace(stage, artifact_dir)
        _publish_controller_result(row, plan, artifact_dir, output_root)
        del output, viewmats, intrinsics, action_label
        gc.collect()
        torch.cuda.empty_cache()
        return "GENERATED"
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def prepare_pending(
    rows: Sequence[Mapping[str, Any]], data_root: Path,
) -> list[dict[str, Any]]:
    sha_cache: dict[Path, str] = {}
    prepared = []
    for row in rows:
        record = row["record"]
        image = resolve_condition_image(record, data_root, sha_cache)
        request = build_native_request(record)
        prepared.append({"row": row, "record": record, "image": image, "request": request})
    return prepared


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-plan", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--weight-root", type=Path, default=DEFAULT_WEIGHT_ROOT)
    parser.add_argument("--weight-manifest", type=Path, default=DEFAULT_WEIGHT_MANIFEST)
    parser.add_argument("--all-sha-receipt", type=Path, default=DEFAULT_ALL_SHA_RECEIPT)
    parser.add_argument("--all-sha-marker", type=Path, default=DEFAULT_ALL_SHA_MARKER)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--action-checkpoint", type=Path, default=DEFAULT_ACTION_CHECKPOINT)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run all CPU gates without importing torch or starting inference",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    plan, all_rows, pending_rows, shard_summary = validate_work(
        args.shard_plan,
        args.shard_manifest,
        output_root,
        require_environment=not args.validate_only,
    )
    runtime = validate_exact_runtime(args)
    # Validate the immutable shard assets as well as the current pending subset.
    prepared_all = prepare_pending(all_rows, args.data_root.resolve())
    prepared_by_id = {
        str(item["record"]["record_id"]): item for item in prepared_all
    }
    prepared = [
        prepared_by_id[str(row["record"]["record_id"])] for row in pending_rows
    ]
    if args.validate_only:
        print(json.dumps({
            "status": "VALIDATED_CPU_ONLY",
            "model_id": MODEL_ID,
            "campaign_sha256": CAMPAIGN_SHA256,
            "shard_index": plan["shard_index"],
            "shard_summary": shard_summary,
            "pending_record_count": len(prepared),
            "pipeline_imported": False,
            "model_loaded": False,
            "native_request_lengths": sorted({
                item["request"]["route_length"] for item in prepared
            }),
            "all_sha_receipt_sha256": runtime["all_sha_receipt_sha256"],
        }, sort_keys=True))
        return 0

    resumable = []
    to_generate = []
    for item in prepared:
        row = item["row"]
        result_path = require_inside(
            output_root,
            output_root / str(row["required_generation_record"]),
            "controller generation result",
        )
        artifact_dir = result_path.parent / "artifact"
        if result_path.exists() or artifact_dir.exists():
            resumable.append(item)
        else:
            to_generate.append(item)
    for item in resumable:
        artifact_dir = (
            output_root / str(item["row"]["record_output_dir"]) / "artifact"
        ).resolve()
        if not artifact_dir.exists():
            raise WorkerError(
                f"generation result exists without artifact: {item['record']['record_id']}"
            )
        _validate_existing_artifact(
            item["row"], plan, item["request"], artifact_dir, runtime,
        )
        _publish_controller_result(item["row"], plan, artifact_dir, output_root)

    if not to_generate:
        print(json.dumps({
            "status": "SUCCEEDED_RESUME_ONLY",
            "model_id": MODEL_ID,
            "shard_index": plan["shard_index"],
            "record_count": len(prepared),
            "generated_record_count": 0,
            "resumed_record_count": len(resumable),
            "pipeline_load_count": 0,
        }, sort_keys=True))
        return 0

    native = pipe = torch = None
    generated = 0
    try:
        native, pipe, torch = load_official_pipeline(args)
        for item in to_generate:
            status = generate_one(
                native,
                pipe,
                torch,
                item["row"],
                plan,
                item["request"],
                item["image"],
                runtime,
                output_root,
            )
            generated += status == "GENERATED"
    finally:
        if torch is not None and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        del pipe
    print(json.dumps({
        "status": "SUCCEEDED",
        "model_id": MODEL_ID,
        "shard_index": plan["shard_index"],
        "record_count": len(prepared),
        "generated_record_count": generated,
        "resumed_record_count": len(resumable),
        "pipeline_load_count": 1,
        "seeds": sorted({int(item["record"]["seed"]) for item in prepared}),
    }, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (WorkerError, OSError) as error:
        print(f"ERROR [{MODEL_ID}]: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
