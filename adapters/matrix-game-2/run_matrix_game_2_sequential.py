#!/usr/bin/env python3
"""Run a hash-bound sequential Task 3 smoke with Matrix-Game 2.

The upstream Matrix-Game 2 batch entrypoint samples a random condition and does
not inspect ``args.action`` or ``args.action_sequence``.  This runner therefore
installs a case-local replacement at the exact upstream lookup site
(``inference.Bench_actions_universal``).  Every native payload is first encoded
by upstream ``utils.conditions.combine_data`` and then arranged according to
the frozen compiled timeline.  The replacement is restored after every case.

Matrix-Game 2 emits ``(latent_frames - 1) * 4 + 1`` pixel frames in one causal
inference call.  Legacy dose 1/2/3 cases retain the frozen 3-latent/9-frame
request.  A consolidated Task 2 held-action dose 16 uses 6 latent/21 raw frames,
with measurements frozen at dose 1/2/3/5/8/16 inside one rollout.  The scored
timeline is followed only by explicit NO_OP padding.  This experimental runner
adds ordered, dose-1 control switching inside the same causal inference call;
it never chains RGB frames and does not alter any native action payload or
motion gain.  It is smoke-only until its generated video is manually accepted.

The separate ``matrix_game_2_final_atomic_v3`` adapter extends only the
stdlib validation and native-request planning contract to the final portable
5/8/11/14-step routes.  That adapter is deliberately labelled CPU-plan
only; accepting it here is not evidence that an L23 GPU rollout has run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from strideworld_compiled_subset_contract import (  # noqa: E402
    CompiledSubsetContractError, is_queue_subset_path,
    validate_compound_container_authorization, validate_or_raise,
)


MODEL_ID = "matrix_game_2"
ADAPTER_PATH = (
    ROOT / "action_interface" / "adapters_task3_smoke" / "matrix_game_2.json"
)
ADAPTER_ID = "matrix_game_2_lite_v1"
HELD_ADAPTER_ID = "matrix_game_2_held8_v2"
ATOMIC_V3_ADAPTER_ID = "matrix_game_2_atomic_v3"
FINAL_ADAPTER_ID = "matrix_game_2_final_atomic_v3"
FINAL_ROUTE_LENGTHS = (5, 8, 11, 14)
FINAL_COMPILE_MODE = "FINAL_PORTABLE_RECORD_CPU_ONLY"
FINAL_EVIDENCE_STATUS = "CPU_COMPILED_NATIVE_REQUEST_PLAN_ONLY"
FINAL_GPU_STATUS = "NOT_RUN"


def _safe_mouse_icon_path() -> str:
    """Return a real image path for upstream cv2.imread, never an ndarray."""
    path = Path(os.getenv("SW_MOUSE_ICON", "/tmp/strideworld-mouse-icon.png"))
    if not path.is_file() or path.stat().st_size < 100:
        path.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAE0lEQVR4nGNgGAWjYBSMAgYwAAAEEAABsax5zAAAAABJRU5ErkJggg=="
        ))
    return str(path)
COMPOUND_PROTOCOL_PATH = ROOT / "action_interface" / "protocol.compound.v1.json"
COMPOUND_PROTOCOL_VERSION = "strideworld-action-protocol-compound-v1"
SUPPORTED_ADAPTER_IDS = {
    ADAPTER_ID, HELD_ADAPTER_ID, ATOMIC_V3_ADAPTER_ID, FINAL_ADAPTER_ID,
}
COMPILED_SCHEMA = "strideworld-compiled-actions-lite-v1"
RECORD_SCHEMA = "strideworld-matrix-game-2-generation-record-v1"
DEFAULT_NATIVE_ROOT = Path(
    "<path>/Matrix-Game-2"
)
DEFAULT_CHECKPOINT_ROOT = Path(
    "<path>/Matrix-Game-2.0"
)
NATIVE_CONFIG_RELATIVE = "configs/inference_yaml/inference_universal.yaml"
NATIVE_CHECKPOINT_RELATIVE = "base_distilled_model/base_distill.safetensors"
NATIVE_BASE_COMMIT = "71c3cd7f741311f8100f6cf9cde942b6c1378d11"
CHECKPOINT_REPO = "Skywork/Matrix-Game-2.0"
CHECKPOINT_REVISION = "f1729d99a80e0f07993a77d7dad4a3190e23c2c8"
CHECKPOINT_REQUIRED_FILES = (
    NATIVE_CHECKPOINT_RELATIVE,
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "xlm-roberta-large/sentencepiece.bpe.model",
    "xlm-roberta-large/special_tokens_map.json",
    "xlm-roberta-large/tokenizer.json",
    "xlm-roberta-large/tokenizer_config.json",
)
NATIVE_SOURCE_SHA256 = {
    "inference.py":
        "ac4e40ab22d265d949d649470d9d227471bd8f519b25678d01be1548d4b04d92",
    "utils/conditions.py":
        "4c567442abbcf42e87e0c01d1001ca7046b18b0bd9608512c35a68e8afb309a9",
    "pipeline/causal_inference.py":
        "d3b200417253810aa2b1097aa02041a562cb3fae92170052ae232204aa63cf12",
    "wan/modules/action_module.py":
        "3b59c210c224fcc77bb0ff1bf79ea551b201a6c3625f4d172117f51a626cbc4b",
    "utils/misc.py":
        "610bead16d50d99fabdccb2e9ce3c20fa1fa1cd3bdf735d7ff7ffda978bfe195",
    "utils/visualize.py":
        "64fcd0d4c81ee9874a42998c712025722ca6ac77ad49b7e3654d024d205518a2",
    NATIVE_CONFIG_RELATIVE:
        "316cae08502ffe32c619d7fedd18dbfe1141a221af578de5f1f3939c3d6a0554",
    "configs/distilled_model/universal/config.json":
        "9624b794bc2ddfa10c6cd94f9e0b3eba2f8d03b9e4ef458ee2eb01ff17332dd5",
}
NATIVE_SOURCE_TREE_SHA256 = (
    "ddb586743c451c477901b2b98941e78c3c1a9121b1c3a490814eb62f41c55234"
)
NATIVE_SOURCE_TREE_COUNT = 60
NATIVE_SOURCE_DIRS = ("configs", "demo_utils", "pipeline", "utils", "wan")
NATIVE_SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml"}

NATIVE_OUTPUT_FRAMES = 9
NATIVE_OUTPUT_LATENT_FRAMES = 3
NATIVE_TRANSITION_ROWS = 8
NATIVE_TERMINAL_ROWS = 1
LEGACY_DOSES = {1.0, 2.0, 3.0}
HELD_DOSE = 16.0
HELD_CHECKPOINT_DOSES = (1.0, 2.0, 3.0, 5.0, 8.0, 16.0)
MEASUREMENT_CONTRACT_VERSION = "strideworld-task2-checkpoints-v1"
NATIVE_WIDTH = 640
NATIVE_HEIGHT = 352
UPSTREAM_REFERENCE_WRITER_FPS = 12
BENCHMARK_BENCHMARK_FPS = 16
SUPPORTED_ACTIONS = {
    "NO_OP", "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT",
    "YAW_LEFT", "YAW_RIGHT",
}
CANONICAL_ACTION_ORDER = (
    "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT",
    "YAW_LEFT", "YAW_RIGHT",
)
OPPOSITE_ACTION_PAIRS = (
    frozenset(("MOVE_FORWARD", "MOVE_BACKWARD")),
    frozenset(("MOVE_LEFT", "MOVE_RIGHT")),
    frozenset(("YAW_LEFT", "YAW_RIGHT")),
)
TASK_DIR = {"TASK2": "task2_cases", "TASK3": "task3_cases"}
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
RUN_NAMESPACE_FILE = "strideworld_run_namespace.json"


class RunnerContractError(ValueError):
    """Raised before or during inference when execution evidence disagrees."""


@dataclass(frozen=True)
class SelectedRecord:
    compiled_path: Path
    compiled_sha256: str
    record: dict[str, Any]


@dataclass(frozen=True)
class ExecutionPlan:
    selected: SelectedRecord
    input_image: Path
    input_sha256: str
    case_dir: Path
    native_request: dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerContractError(f"Cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RunnerContractError(f"JSON root must be an object: {path}")
    return value


def require_hash(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise RunnerContractError(f"{label} must be a lowercase SHA256")
    return text


def load_frozen_adapter(path: Path = ADAPTER_PATH) -> dict[str, Any]:
    adapter = read_object(path.resolve())
    adapter_id = adapter.get("adapter_id")
    if adapter_id not in SUPPORTED_ADAPTER_IDS:
        raise RunnerContractError("Matrix-Game 2 adapter_id mismatch")
    expected_timing = {
        "fps": BENCHMARK_BENCHMARK_FPS, "frames": 9, "conditioning_frames": 1,
        "native_tick_frames": 1, "base_control_ticks": 4,
    }
    if adapter.get("timing") != expected_timing:
        raise RunnerContractError("Matrix-Game 2 frozen timing mismatch")
    if adapter.get("representation") != "multi_hot":
        raise RunnerContractError("Matrix-Game 2 representation mismatch")
    sequential = adapter.get("sequential")
    expected_smoke_sequential = {
        "supported": True,
        "verified_runner": "scripts/run_matrix_game_2_sequential.py",
        "verification_source": "preacceptance_task3_sequential_smoke_candidate_v1",
        "max_steps": 8,
        "switch_granularity": "native_tick_boundary",
        "generation_extension_supported": True,
        "generation_extension_verification_source": (
            "scripts/run_matrix_game_2_strideworld_compiled.py:"
            "dose8_native_single_inference_v1"
        ),
        "per_step_reference_frame": "current_camera_cv_rh_body",
        "initial_state_pose_policy": "required_if_conditioning_frames_zero",
    }
    expected_final_sequential = {
        "supported": True,
        "verified_runner": "scripts/run_matrix_game_2_sequential.py",
        "verification_source": (
            "cpu_compiled_native_request_plan_final_"
            "v9_l5_l8_l11_l14_v1_no_gpu_execution"
        ),
        "max_steps": 23,
        "switch_granularity": "native_tick_boundary",
        "generation_extension_supported": True,
        "generation_extension_verification_source": (
            "scripts/run_matrix_game_2_sequential.py:"
            "native_output_plan_dynamic_horizon_to_l23_cpu_only_v1"
        ),
        "per_step_reference_frame": "current_camera_cv_rh_body",
        "initial_state_pose_policy": "required_if_conditioning_frames_zero",
    }
    expected_final_contract = {
        "schema_version": "strideworld-matrix-final-portable-contract-v1",
        "case_manifest_schema": "strideworld-portable-case-v1",
        "record_kind": "SHARED_TASK1_TASK3",
        "route_lengths": [5, 8, 11, 14],
        "actions_per_step": 1,
        "dose_per_step": 1.0,
        "execution_evidence_status": "CPU_COMPILED_NATIVE_REQUEST_PLAN_ONLY",
        "gpu_execution_validation_status": "NOT_RUN",
    }
    expected_sequential = {
        ATOMIC_V3_ADAPTER_ID: expected_smoke_sequential,
        FINAL_ADAPTER_ID: expected_final_sequential,
    }.get(adapter_id)
    if expected_sequential is None or sequential != expected_sequential:
        raise RunnerContractError(
            "Matrix-Game 2 sequential capability/evidence mismatch"
        )
    final_contract = adapter.get("final_portable_contract")
    if adapter_id == FINAL_ADAPTER_ID:
        if final_contract != expected_final_contract:
            raise RunnerContractError("Matrix-Game 2 final contract mismatch")
    elif final_contract is not None:
        raise RunnerContractError("Matrix-Game 2 smoke adapter contains final metadata")
    for action in SUPPORTED_ACTIONS:
        payload = adapter.get("actions", {}).get(action)
        if not isinstance(payload, dict):
            raise RunnerContractError(f"Matrix-Game 2 missing action {action}")
        if sorted(payload) != ["keyboard", "mouse"]:
            raise RunnerContractError(f"Malformed Matrix-Game 2 action {action}")
        if len(payload["keyboard"]) != 4 or len(payload["mouse"]) != 2:
            raise RunnerContractError(f"Malformed Matrix-Game 2 action dimensions: {action}")
    return adapter


def source_tree_manifest(root: Path) -> list[dict[str, str]]:
    paths: set[Path] = set()
    for path in root.iterdir():
        if path.is_file() and path.suffix in NATIVE_SOURCE_SUFFIXES:
            paths.add(path)
    for dirname in NATIVE_SOURCE_DIRS:
        directory = root / dirname
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in NATIVE_SOURCE_SUFFIXES:
                paths.add(path)
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}
        for path in sorted(paths)
    ]


def validate_native_sources(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    key_files = []
    for relative, expected in NATIVE_SOURCE_SHA256.items():
        path = (root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != expected:
            raise RunnerContractError(f"Native source SHA256 mismatch: {relative}")
        key_files.append({"path": str(path), "sha256": actual})
    manifest = source_tree_manifest(root)
    fingerprint = canonical_sha256(manifest)
    if len(manifest) != NATIVE_SOURCE_TREE_COUNT or fingerprint != NATIVE_SOURCE_TREE_SHA256:
        raise RunnerContractError(
            "Matrix-Game 2 source tree is not the pinned official revision"
        )
    return {
        "repository": "SkyworkAI/Matrix-Game",
        "base_commit": NATIVE_BASE_COMMIT,
        "validation": "exact_code_config_tree_manifest",
        "source_tree_file_count": len(manifest),
        "source_tree_sha256": fingerprint,
        "key_source_files": key_files,
    }


def validate_record_identity(record: dict[str, Any]) -> None:
    if record.get("case_identity_status") != "FORMAL_BOUND":
        raise RunnerContractError("Runner requires a formally bound compiled record")
    fields = ("state_id", "first_frame_sha256", "probe_plan_sha256", "probe_case_id")
    if any(not record.get(field) for field in fields):
        raise RunnerContractError("Compiled record has incomplete formal identity")
    require_hash(record["first_frame_sha256"], "first_frame_sha256")
    require_hash(record["probe_plan_sha256"], "probe_plan_sha256")
    contract = {"case_id": record.get("case_id")}
    contract.update({field: record[field] for field in fields})
    if record.get("case_identity_contract") != contract:
        raise RunnerContractError("case_identity_contract content mismatch")
    if record.get("case_identity_contract_sha256") != canonical_sha256(contract):
        raise RunnerContractError("case_identity_contract SHA256 mismatch")


def round_half_up(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise RunnerContractError("native tick request must be finite and non-negative")
    return int(math.floor(value + 0.5))


def native_output_plan(required_frames: int) -> dict[str, int]:
    """Return the smallest official one-call latent horizon covering the window."""

    if isinstance(required_frames, bool) or not isinstance(required_frames, int):
        raise RunnerContractError("required output frames must be an integer")
    latent_frames = max(
        NATIVE_OUTPUT_LATENT_FRAMES,
        int(math.ceil((required_frames - 1) / 4.0)) + 1,
    )
    latent_frames += (-latent_frames) % 3
    output_frames = (latent_frames - 1) * 4 + 1
    return {
        "num_output_latent_frames": latent_frames,
        "output_frame_count": output_frames,
        "native_transition_rows": output_frames - 1,
        "unused_terminal_condition_rows": NATIVE_TERMINAL_ROWS,
    }


def validate_measurement_checkpoints(
    record: dict[str, Any], adapter: dict[str, Any],
) -> tuple[list[float], list[dict[str, Any]]]:
    """Validate the frozen held-action measurement boundary contract.

    Old Task 2 rows did not materialize checkpoint fields and remain readable as
    a single checkpoint at their sole step dose.  New dose-16 rows must carry the
    same explicit contract at request, top-level, and hashed schedule locations.
    """

    if record.get("task") != "TASK2":
        fields = (
            "checkpoint_doses", "measurement_checkpoints",
        )
        request = record.get("request_contract", {})
        schedule = record.get("schedule_contract", {})
        if (any(field in record for field in fields)
                or "measurement_contract_version" in record
                or "checkpoint_doses" in request
                or "measurement_checkpoints" in schedule):
            raise RunnerContractError("measurement checkpoints are Task 2 only")
        return [], []

    step_dose = float(record["steps"][0]["dose_requested"])
    request = record["request_contract"]
    schedule = record["schedule_contract"]
    top_doses = record.get("checkpoint_doses")
    request_doses = request.get("checkpoint_doses")
    top_points = record.get("measurement_checkpoints")
    schedule_points = schedule.get("measurement_checkpoints")
    explicit = any(value is not None for value in (
        top_doses, request_doses, top_points, schedule_points,
        record.get("measurement_contract_version"),
    ))
    if not explicit:
        if step_dose == HELD_DOSE:
            raise RunnerContractError("dose 16 requires explicit checkpoint_doses")
        return [step_dose], [{
            "dose_requested": step_dose,
            "native_tick": round_half_up(
                float(adapter["timing"]["base_control_ticks"]) * step_dose
            ),
            "frame_index": (
                int(adapter["timing"]["conditioning_frames"])
                + round_half_up(
                    float(adapter["timing"]["base_control_ticks"]) * step_dose
                ) * int(adapter["timing"]["native_tick_frames"])
                - 1
            ),
        }]
    if not isinstance(top_doses, list) or request_doses != top_doses:
        raise RunnerContractError("top/request checkpoint_doses mismatch")
    if record.get("measurement_contract_version") != MEASUREMENT_CONTRACT_VERSION:
        raise RunnerContractError("Task 2 measurement contract version mismatch")
    if not isinstance(top_points, list) or schedule_points != top_points:
        raise RunnerContractError("top/schedule measurement_checkpoints mismatch")
    doses: list[float] = []
    for raw in top_doses:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RunnerContractError("checkpoint doses must be numeric")
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise RunnerContractError("checkpoint doses must be finite and positive")
        doses.append(value)
    if not doses or any(right <= left for left, right in zip(doses, doses[1:])):
        raise RunnerContractError("checkpoint doses must be strictly increasing")
    if doses[-1] != step_dose or any(value > step_dose for value in doses):
        raise RunnerContractError("checkpoint doses must end at the held step dose")
    if step_dose == HELD_DOSE:
        if tuple(doses) != HELD_CHECKPOINT_DOSES:
            raise RunnerContractError("dose 16 requires checkpoints 1/2/3/5/8/16")
    elif doses != [step_dose]:
        raise RunnerContractError("legacy Task 2 rows allow only their endpoint checkpoint")
    base_ticks = float(adapter["timing"]["base_control_ticks"])
    tick_frames = int(adapter["timing"]["native_tick_frames"])
    conditioning = int(adapter["timing"]["conditioning_frames"])
    expected = []
    for dose in doses:
        tick = round_half_up(base_ticks * dose)
        expected.append({
            "dose_requested": dose,
            "native_tick": tick,
            "frame_index": conditioning + tick * tick_frames - 1,
        })
    if top_points != expected:
        raise RunnerContractError("measurement checkpoint tick/frame mapping mismatch")
    return doses, expected


def merged_action_payload(
    adapter: dict[str, Any], actions: list[str],
) -> dict[str, Any]:
    if not 1 <= len(actions) <= 3 or len(actions) != len(set(actions)):
        raise RunnerContractError("step actions must contain 1-3 unique actions")
    if any(action not in SUPPORTED_ACTIONS for action in actions):
        raise RunnerContractError(f"Unsupported formal Matrix-Game 2 actions: {actions}")
    if "NO_OP" in actions and len(actions) != 1:
        raise RunnerContractError("NO_OP cannot be combined with another action")
    selected = set(actions)
    if any(pair <= selected for pair in OPPOSITE_ACTION_PAIRS):
        raise RunnerContractError("simultaneous step combines opposite actions")
    canonical = [action for action in CANONICAL_ACTION_ORDER if action in selected]
    if actions != canonical and actions != ["NO_OP"]:
        raise RunnerContractError("simultaneous actions must use canonical order")
    if len(actions) == 1:
        return adapter["actions"][actions[0]]
    rule = adapter.get("simultaneous", {})
    if rule.get("supported") is not True or rule.get("merge_fields") != ["keyboard", "mouse"]:
        raise RunnerContractError("Matrix-Game 2 adapter lacks native simultaneous merge")
    merged: dict[str, list[float]] = {}
    for field in ("keyboard", "mouse"):
        vectors = [adapter["actions"][action][field] for action in actions]
        if len({len(vector) for vector in vectors}) != 1:
            raise RunnerContractError(f"cannot merge Matrix-Game 2 {field} dimensions")
        merged[field] = [sum(values) for values in zip(*vectors)]
    return merged


def expected_step(
    adapter: dict[str, Any], step: dict[str, Any], step_index: int,
    start_tick: int,
) -> dict[str, Any]:
    actions = step.get("canonical_actions")
    if not isinstance(actions, list):
        raise RunnerContractError("Task 3 step actions must be a list")
    payload = merged_action_payload(adapter, actions)
    dose = step.get("dose_requested")
    if step.get("step_index") != step_index or dose not in LEGACY_DOSES | {HELD_DOSE}:
        raise RunnerContractError("Task 3 step index/dose mismatch")
    base_ticks = int(adapter["timing"]["base_control_ticks"])
    requested_ticks = base_ticks * float(dose)
    ticks = int(requested_ticks)
    conditioning = int(adapter["timing"]["conditioning_frames"])
    return {
        "mode": "simultaneous" if len(actions) > 1 else "single",
        "dose_requested": dose,
        "native_ticks_requested_float": requested_ticks,
        "start_tick": start_tick, "end_tick_exclusive": start_tick + ticks,
        "native_ticks": ticks, "dose_realized": ticks / base_ticks,
        "start_frame": conditioning + start_tick,
        "end_frame_exclusive": conditioning + start_tick + ticks,
        "operator_window": {
            "start_frame": conditioning + start_tick - 1,
            "end_frame": conditioning + start_tick + ticks - 1,
            "end_is_inclusive": True,
            "definition": "frame_at_tick_minus_one_state_boundaries",
            "tail_included_in_unit_pose": False,
        },
        "relative_start_frame": 0,
        "relative_end_frame_exclusive": ticks,
        "per_step_reference_frame": "current_camera_cv_rh_body",
        "duration_seconds_realized": ticks / float(adapter["timing"]["fps"]),
        "native_payload": payload,
        "native_payload_sha256": canonical_sha256(payload),
    }


def validate_steps(record: dict[str, Any], adapter: dict[str, Any]) -> int:
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps or not all(
        isinstance(step, dict) for step in steps
    ):
        raise RunnerContractError("Matrix-Game 2 steps must be non-empty objects")
    task = record.get("task")
    if task == "TASK3":
        capability = adapter.get("sequential", {})
        if capability.get("supported") is not True:
            raise RunnerContractError("Task 3 requires a sequential adapter")
        if not 2 <= len(steps) <= int(capability.get("max_steps", 0)):
            raise RunnerContractError("Task 3 exceeds the adapter step bound")
        if any(
            step.get("dose_requested") != 1.0
            or len(step.get("canonical_actions", [])) != 1
            for step in steps
        ):
            raise RunnerContractError("Task 3 requires ordered dose-1 atoms")
    elif task == "TASK2":
        if adapter.get("adapter_id") != FINAL_ADAPTER_ID:
            raise RunnerContractError(
                "Task 2 mixed execution requires the final adapter"
            )
        if (
            len(steps) != 1
            or steps[0].get("dose_requested") != HELD_DOSE
            or len(steps[0].get("canonical_actions", [])) != 1
        ):
            raise RunnerContractError(
                "Task 2 mixed execution requires one held dose-16 atom"
            )
    else:
        raise RunnerContractError("sequential runner accepts only Task 2/Task 3")
    next_tick = 0
    for index, step in enumerate(steps):
        expected = expected_step(adapter, step, index, next_tick)
        if expected["dose_requested"] == HELD_DOSE and task != "TASK2":
            raise RunnerContractError("held dose 16 is valid only for Task 2")
        if any(step.get(key) != value for key, value in expected.items()):
            raise RunnerContractError("step timeline or native payload mismatch")
        next_tick = int(expected["end_tick_exclusive"])
    return next_tick


def validate_request(record: dict[str, Any]) -> None:
    request = record.get("request_contract")
    if not isinstance(request, dict) or record.get("request_contract_sha256") != canonical_sha256(request):
        raise RunnerContractError("request_contract SHA256 mismatch")
    expected = {
        "case_id": record["case_id"], "task": record["task"],
        "seed": record["seed"],
        "steps": [{
            "step_index": index,
            "canonical_actions": list(step["canonical_actions"]),
            "dose_requested": step["dose_requested"],
        } for index, step in enumerate(record["steps"])],
        "state_id": record["state_id"],
        "first_frame_sha256": record["first_frame_sha256"],
        "probe_plan_sha256": record["probe_plan_sha256"],
        "probe_case_id": record["probe_case_id"],
        "benchmark_action_label": record["benchmark_action_label"],
    }
    if "checkpoint_doses" in request:
        expected["checkpoint_doses"] = record.get("checkpoint_doses")
    if request != expected:
        raise RunnerContractError("request_contract content mismatch")


def validate_schedule(record: dict[str, Any], adapter: dict[str, Any], command_ticks: int) -> None:
    schedule = record.get("schedule_contract")
    if not isinstance(schedule, dict) or record.get("schedule_contract_sha256") != canonical_sha256(schedule):
        raise RunnerContractError("schedule_contract SHA256 mismatch")
    command = record.get("command_window")
    observation = record.get("observation_window")
    if schedule.get("command_window") != command or schedule.get("observation_window") != observation:
        raise RunnerContractError("top-level and hashed schedule windows differ")
    if schedule.get("terminal_tail_applied_once") is not True:
        raise RunnerContractError("terminal observation tail must be applied once")
    if not isinstance(observation, dict):
        raise RunnerContractError("observation_window must be an object")
    tail_ticks = int(observation.get("tail_ticks", -1))
    conditioning = int(adapter["timing"]["conditioning_frames"])
    expected_tail = int(adapter["timing"]["base_control_ticks"])
    required = int(record.get("required_output_frames", -1))
    expected_required = conditioning + command_ticks + tail_ticks
    if tail_ticks != expected_tail or required != expected_required:
        raise RunnerContractError("required output or terminal tail duration mismatch")
    expected_command = {
        "start_tick": 0, "end_tick_exclusive": command_ticks,
        "start_frame": conditioning,
        "end_frame_exclusive": conditioning + command_ticks,
    }
    if command != expected_command or observation.get("start_frame") != 0:
        raise RunnerContractError("command/observation window boundary mismatch")
    if observation.get("tail_dose_realized") != 1.0:
        raise RunnerContractError("terminal observation tail dose mismatch")
    if observation.get("end_frame_exclusive") != required:
        raise RunnerContractError("observation window does not end at required output")
    if schedule.get("evaluation_window") != {"start_frame": 0, "end_frame_exclusive": required}:
        raise RunnerContractError("evaluation window mismatch")
    keys = (
        "step_index", "canonical_actions", "start_tick", "end_tick_exclusive",
        "start_frame", "end_frame_exclusive", "operator_window",
    )
    boundaries = [
        {key: step[key] for key in keys} for step in record["steps"]
    ]
    if schedule.get("ordered_step_boundaries") != boundaries:
        raise RunnerContractError("hashed ordered step boundary mismatch")


def validate_native_timeline(record: dict[str, Any], adapter: dict[str, Any]) -> None:
    expected = [
        payload
        for step in record["steps"]
        for payload in [step["native_payload"]] * int(step["native_ticks"])
    ]
    expected.extend(
        [adapter["actions"]["NO_OP"]] * int(record["observation_window"]["tail_ticks"])
    )
    if record.get("native_timeline") != expected:
        raise RunnerContractError("native timeline/step/terminal tail mismatch")


def validate_record(record: dict[str, Any], adapter: dict[str, Any]) -> None:
    task = record.get("task")
    if task not in TASK_DIR:
        raise RunnerContractError(
            "Matrix-Game 2 sequential runner accepts only Task 2/Task 3"
        )
    if task == "TASK2" and adapter.get("adapter_id") != FINAL_ADAPTER_ID:
        raise RunnerContractError(
            "Matrix-Game 2 smoke adapter accepts only Task 3; "
            "Task 2 mixed execution requires the final adapter"
        )
    if (record.get("status") != "SUPPORTED"
            or record.get("adapter_id") != adapter.get("adapter_id")):
        raise RunnerContractError("selected record is not SUPPORTED by Matrix-Game 2")
    if record.get("adapter_fingerprint") != canonical_sha256(adapter):
        raise RunnerContractError("compiled adapter fingerprint mismatch")
    capability = adapter["sequential"]
    if record.get("sequential_capability") != capability:
        raise RunnerContractError("compiled sequential capability mismatch")
    if record.get("sequential_capability_sha256") != canonical_sha256(capability):
        raise RunnerContractError("compiled sequential capability SHA256 mismatch")
    if record.get("default_output_frames") != NATIVE_OUTPUT_FRAMES:
        raise RunnerContractError("compiled default output frame count mismatch")
    required = int(record.get("required_output_frames", -1))
    extension_required = required > NATIVE_OUTPUT_FRAMES
    if record.get("generation_extension_required") is not extension_required:
        raise RunnerContractError("compiled generation-extension flag mismatch")
    if (record.get("native_tick_frames") != 1
            or record.get("fps") != BENCHMARK_BENCHMARK_FPS):
        raise RunnerContractError("compiled native timing fields mismatch")
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise RunnerContractError("compiled seed must be a fixed uint32")
    validate_record_identity(record)
    compound_steps = [
        step for step in record.get("steps", [])
        if len(step.get("canonical_actions", [])) > 1
    ]
    if compound_steps:
        if adapter.get("adapter_id") != ATOMIC_V3_ADAPTER_ID:
            raise RunnerContractError(
                "compound records require the frozen atomic-v3 adapter"
            )
        if record.get("compound_protocol_version") != COMPOUND_PROTOCOL_VERSION:
            raise RunnerContractError("compound record lacks the frozen protocol version")
        protocol = read_object(COMPOUND_PROTOCOL_PATH)
        if record.get("compound_protocol_fingerprint") != canonical_sha256(protocol):
            raise RunnerContractError("compound protocol fingerprint mismatch")
    command_ticks = validate_steps(record, adapter)
    dose = float(record["steps"][0]["dose_requested"])
    expected_adapter_ids = (
        {HELD_ADAPTER_ID, ATOMIC_V3_ADAPTER_ID, FINAL_ADAPTER_ID}
        if dose == HELD_DOSE else {ADAPTER_ID}
    )
    if (record.get("task") == "TASK2"
            and adapter.get("adapter_id") not in expected_adapter_ids):
        raise RunnerContractError("legacy/held dose is bound to the wrong adapter_id")
    validate_request(record)
    validate_schedule(record, adapter, command_ticks)
    validate_measurement_checkpoints(record, adapter)
    validate_native_timeline(record, adapter)


def native_condition_rows(record: dict[str, Any], adapter: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = list(record["native_timeline"])
    required = int(record["required_output_frames"])
    if len(timeline) != required - 1:
        raise RunnerContractError("compiled timeline must control every scored transition")
    plan = native_output_plan(required)
    transition_rows = plan["native_transition_rows"]
    padding = transition_rows - len(timeline)
    if padding < 0:
        raise RunnerContractError("compiled timeline exceeds Matrix-Game 2 capacity")
    rows = timeline + [adapter["actions"]["NO_OP"]] * padding
    rows += [adapter["actions"]["NO_OP"]] * NATIVE_TERMINAL_ROWS
    if len(rows) != plan["output_frame_count"]:
        raise RunnerContractError("internal Matrix-Game 2 condition row count mismatch")
    return rows


def native_request(record: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    rows = native_condition_rows(record, adapter)
    required = int(record["required_output_frames"])
    output = native_output_plan(required)
    checkpoint_doses, checkpoints = validate_measurement_checkpoints(record, adapter)
    compiled_ticks = len(record["native_timeline"])
    padding = output["native_transition_rows"] - compiled_ticks
    return {
        "mode": "universal",
        "native_rollout_mode": "single_causal_inference_call",
        "external_rgb_last_frame_chaining": False,
        "num_output_latent_frames": output["num_output_latent_frames"],
        "output_frame_count": output["output_frame_count"],
        "condition_builder_lookup": "inference.Bench_actions_universal",
        "condition_encoder": "utils.conditions.combine_data",
        "condition_encoding_policy": "one_exact_native_payload_per_official_encode",
        "compiled_timeline_ticks": compiled_ticks,
        "command_ticks": int(record["command_window"]["end_tick_exclusive"]),
        "terminal_noop_tail_ticks": int(record["observation_window"]["tail_ticks"]),
        "native_transition_rows": output["native_transition_rows"],
        "unscored_noop_padding_rows": padding,
        "unused_terminal_condition_rows": NATIVE_TERMINAL_ROWS,
        "condition_rows": rows,
        "condition_rows_sha256": canonical_sha256(rows),
        "required_output_frames": required,
        "formal_evaluation_frame_count": required,
        "checkpoint_doses": checkpoint_doses,
        "measurement_contract_version": record.get("measurement_contract_version"),
        "measurement_checkpoints": checkpoints,
        "checkpoint_frame_indices": [item["frame_index"] for item in checkpoints],
        "evaluation_window": {"start_frame": 0, "end_frame_exclusive": required},
        "metric_frame_slice_half_open": [0, required],
        "post_required_padding": {
            "start_frame": required,
            "end_frame_exclusive": output["output_frame_count"],
            "role": "unscored_post_required_padding",
            "control_semantics": "explicit_noop",
            "excluded_from_scoring": True,
        } if required < output["output_frame_count"] else None,
    }


def validate_final_compiled_container(
    source: Path, payload: dict[str, Any], adapter: dict[str, Any],
) -> None:
    """Authorize one worker-produced portable final record for real inference."""

    if adapter.get("adapter_id") != FINAL_ADAPTER_ID:
        raise RunnerContractError("final compiled input requires the final adapter")
    expected_top = {
        "strideworld_compilation_mode": FINAL_COMPILE_MODE,
        "strideworld_execution_evidence_status": FINAL_EVIDENCE_STATUS,
        "strideworld_gpu_execution_validation_status": FINAL_GPU_STATUS,
        "record_count": 1,
        "case_count": 1,
        "adapter_count": 1,
    }
    for key, expected in expected_top.items():
        if payload.get(key) != expected:
            raise RunnerContractError(f"final compiled container {key} mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise RunnerContractError("final compiled container must contain one record")
    record = records[0]
    binding = payload.get("strideworld_portable_binding")
    if not isinstance(binding, dict):
        raise RunnerContractError("final compiled container lacks portable binding")
    binding_sha256 = canonical_sha256(binding)
    if payload.get("strideworld_portable_binding_sha256") != binding_sha256:
        raise RunnerContractError("final top-level portable binding SHA256 mismatch")
    if (
        record.get("strideworld_portable_binding") != binding
        or record.get("strideworld_portable_binding_sha256") != binding_sha256
    ):
        raise RunnerContractError("final record/top portable binding mismatch")
    if (
        record.get("strideworld_execution_evidence_status") != FINAL_EVIDENCE_STATUS
        or record.get("strideworld_gpu_execution_validation_status") != FINAL_GPU_STATUS
    ):
        raise RunnerContractError("final compiled record evidence status mismatch")
    route_length = binding.get("route_length")
    if (
        route_length not in FINAL_ROUTE_LENGTHS
        or route_length != len(record.get("steps", ()))
        or binding.get("record_id") != record.get("case_id")
        or binding.get("record_id") != record.get("probe_case_id")
    ):
        raise RunnerContractError("final portable route identity/length mismatch")
    require_hash(binding.get("case_manifest_sha256"), "case_manifest_sha256")
    require_hash(binding.get("case_content_sha256"), "case_content_sha256")
    require_hash(binding.get("record_content_sha256"), "record_content_sha256")
    manifest_value = binding.get("case_manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise RunnerContractError("final portable binding lacks case manifest path")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RunnerContractError("final bound case manifest is missing/unsafe")
    if file_sha256(manifest_path) != binding["case_manifest_sha256"]:
        raise RunnerContractError("final bound case manifest SHA256 mismatch")


def records_for_task(
    path: Path, task: str, adapter: dict[str, Any],
    *, allow_compound_g3_smoke: bool = False,
) -> list[SelectedRecord]:
    source = path.resolve()
    allowed_suffixes = (".all.json", f".{task.lower()}.json")
    if not source.name.endswith(allowed_suffixes):
        raise RunnerContractError(
            f"compiled input must end in .all.json or .{task.lower()}.json: {source}"
        )
    payload = read_object(source)
    if payload.get("schema_version") != COMPILED_SCHEMA or not isinstance(payload.get("records"), list):
        raise RunnerContractError(f"Invalid compiled container: {source}")
    is_final = adapter.get("adapter_id") == FINAL_ADAPTER_ID
    if is_final:
        if task != "TASK3":
            raise RunnerContractError("final compiled containers are Task 3 only")
        validate_final_compiled_container(source, payload, adapter)
        is_compound = False
    else:
        try:
            is_compound = validate_compound_container_authorization(
                source, payload, MODEL_ID, task, Path(__file__),
                allow_g3_smoke=allow_compound_g3_smoke,
            )
        except CompiledSubsetContractError as error:
            raise RunnerContractError(str(error)) from error
        if not is_compound:
            validate_or_raise(source, payload, MODEL_ID, task, RunnerContractError)
    digest = file_sha256(source)
    candidates = [
        item for item in payload["records"] if isinstance(item, dict)
        and item.get("adapter_id") == adapter.get("adapter_id")
        and item.get("task") == task
    ]
    selected = [item for item in candidates if item.get("status") == "SUPPORTED"]
    if is_compound and not (
        len(candidates) == len(selected) == len(payload["records"])
    ):
        raise RunnerContractError(
            "compound authorization forbids adapter/task/status filtering"
        )
    if not selected:
        structural_na = sum(item.get("status") == "STRUCTURAL_NA" for item in candidates)
        raise RunnerContractError(
            f"No supported Matrix-Game 2 {task} records in {source}; "
            f"STRUCTURAL_NA={structural_na}"
        )
    return [SelectedRecord(source, digest, item) for item in selected]


def load_selected(args: argparse.Namespace, adapter: dict[str, Any]) -> list[SelectedRecord]:
    selected: list[SelectedRecord] = []
    for path in args.task1_compiled:
        selected.extend(records_for_task(
            path, "TASK1", adapter,
            allow_compound_g3_smoke=getattr(args, "compound_g3_smoke", False),
        ))
    for path in args.task2_compiled:
        selected.extend(records_for_task(
            path, "TASK2", adapter,
            allow_compound_g3_smoke=getattr(args, "compound_g3_smoke", False),
        ))
    for path in args.task3_compiled:
        selected.extend(records_for_task(path, "TASK3", adapter))
    if not selected:
        raise RunnerContractError("No compiled records were selected")
    queue_subset = any(is_queue_subset_path(item.compiled_path) for item in selected)
    frozen_keys = {
        (item.record.get("task"), item.record.get("case_id")) for item in selected
    }
    identifiers = list(getattr(args, "case_id", []))
    compound_batch = any(
        (read_object(item.compiled_path).get("authorization_contract") or {}).get("mode")
        == "AUTHORIZED_TASK2_COMPOUND_BATCH"
        for item in selected
    )
    if identifiers and compound_batch:
        raise RunnerContractError(
            "--case-id is forbidden for the exact frozen compound batch"
        )
    if identifiers:
        filtered = []
        for identifier in identifiers:
            matches = [item for item in selected if identifier in {
                str(item.record.get("case_id")), str(item.record.get("probe_case_id")),
            }]
            if len(matches) != 1:
                raise RunnerContractError(
                    f"case identifier {identifier!r} matched {len(matches)} records"
                )
            filtered.append(matches[0])
        selected = filtered
    if queue_subset and {
        (item.record.get("task"), item.record.get("case_id")) for item in selected
    } != frozen_keys:
        raise RunnerContractError("--case-id must cover the complete frozen queue subset")
    keys = [(item.record.get("task"), item.record.get("case_id")) for item in selected]
    if len(keys) != len(set(keys)):
        raise RunnerContractError("Selected compiled records contain duplicates")
    for item in selected:
        validate_record(item.record, adapter)
    return selected


def parse_state_inputs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        state, separator, raw_path = value.partition("=")
        if not separator or not state or state in result:
            raise RunnerContractError("--state-input values must be unique STATE_ID=PATH pairs")
        result[state] = Path(raw_path).resolve()
    return result


def bind_input(
    record: dict[str, Any], state_inputs: dict[str, Path], fallback: Path | None,
) -> tuple[Path, str]:
    state = str(record["state_id"])
    image = state_inputs.get(state)
    if image is None:
        if fallback is None:
            raise RunnerContractError(f"No condition image supplied for state {state}")
        image = fallback.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    digest = file_sha256(image)
    if digest != record["first_frame_sha256"]:
        raise RunnerContractError(f"condition image SHA256 mismatch for state {state}")
    return image, digest


def png_width_height(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RunnerContractError("formal Matrix-Game 2 condition image must be PNG")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise RunnerContractError("condition PNG has invalid dimensions")
    return int(width), int(height)


def source_transform_contract(source_size: tuple[int, int]) -> dict[str, Any]:
    source_width, source_height = source_size
    target_ratio = NATIVE_HEIGHT / NATIVE_WIDTH
    if source_height / source_width > target_ratio:
        crop_width = source_width
        crop_height = int(source_width * target_ratio)
    else:
        crop_height = source_height
        crop_width = int(source_height / target_ratio)
    crop_x = (source_width - crop_width) / 2.0
    crop_y = (source_height - crop_height) / 2.0
    return {
        "mode": "center_crop",
        "resize_interpolation": "bilinear",
        "resize_antialias": True,
        "align_corners": False,
        "pixel_center_convention": "half_pixel",
        "source_width_height": [source_width, source_height],
        "crop_xywh": [crop_x, crop_y, crop_width, crop_height],
        "native_contract_width_height": [NATIVE_WIDTH, NATIVE_HEIGHT],
        "decoded_output_width_height": [NATIVE_WIDTH, NATIVE_HEIGHT],
        "evidence": (
            "Matrix-Game-2 inference.InteractiveGameInference._resizecrop then "
            "torchvision.transforms.v2.Resize(size=(352,640), antialias=True)"
        ),
    }


def output_case_dir(root: Path, record: dict[str, Any]) -> Path:
    probe_id = str(record["probe_case_id"])
    if not SAFE_ID.fullmatch(probe_id):
        raise RunnerContractError(f"unsafe probe_case_id: {probe_id!r}")
    return root.resolve() / TASK_DIR[str(record["task"])] / probe_id


def build_plans(
    selected: list[SelectedRecord], adapter: dict[str, Any], args: argparse.Namespace,
) -> list[ExecutionPlan]:
    state_inputs = parse_state_inputs(list(getattr(args, "state_input", [])))
    fallback = getattr(args, "input_image", None)
    states = {str(item.record["state_id"]) for item in selected}
    if fallback is not None and len(states) != 1:
        raise RunnerContractError("--input-image is only valid for one selected state")
    plans = []
    for item in selected:
        image, digest = bind_input(item.record, state_inputs, fallback)
        case_dir = output_case_dir(args.output, item.record)
        if case_dir.exists():
            raise FileExistsError(f"Refusing to overwrite case directory: {case_dir}")
        request = native_request(item.record, adapter)
        request["source_transform"] = source_transform_contract(png_width_height(image))
        plans.append(ExecutionPlan(item, image, digest, case_dir, request))
    return plans


def metadata_path(checkpoint_root: Path, relative: str) -> Path:
    return checkpoint_root / ".cache" / "huggingface" / "download" / f"{relative}.metadata"


def checkpoint_provenance(checkpoint_root: Path) -> dict[str, Any]:
    files = []
    for relative in CHECKPOINT_REQUIRED_FILES:
        path = (checkpoint_root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata = metadata_path(checkpoint_root, relative).resolve()
        if not metadata.is_file():
            raise FileNotFoundError(metadata)
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or lines[0] != CHECKPOINT_REVISION or not lines[1]:
            raise RunnerContractError(f"checkpoint metadata revision mismatch: {relative}")
        files.append({
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "huggingface_revision": lines[0],
            "huggingface_etag": lines[1],
            "metadata_sha256": file_sha256(metadata),
        })
    contract = {
        "repo": CHECKPOINT_REPO,
        "revision": CHECKPOINT_REVISION,
        "variant": "base_distilled",
        "required_runtime_files": files,
    }
    return {
        "checkpoint_contract": contract,
        "checkpoint_fingerprint": canonical_sha256(contract),
        "checkpoint_fingerprint_source": "required_runtime_files_manifest_sha256",
    }


def bind_output_namespace(
    output_root: Path, adapter: dict[str, Any], checkpoint: dict[str, Any],
) -> dict[str, Any]:
    root = output_root.resolve()
    marker = root / RUN_NAMESPACE_FILE
    contract = {
        "schema_version": "strideworld-model-output-namespace-v1",
        "adapter_id": adapter["adapter_id"],
        "adapter_fingerprint": canonical_sha256(adapter),
        "model_id": adapter["model_id"],
        "model_variant": "base_distilled",
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
    }
    if marker.exists():
        if read_object(marker) != contract:
            raise RunnerContractError(
                f"output root is bound to a different model/adapter/checkpoint: {root}"
            )
    else:
        if root.exists() and any(root.iterdir()):
            raise RunnerContractError(f"non-empty output root lacks {RUN_NAMESPACE_FILE}: {root}")
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return {
        "output_namespace_contract": contract,
        "output_namespace_contract_sha256": canonical_sha256(contract),
        "output_namespace_path": str(marker),
    }


def runtime_imports(native_root: Path) -> dict[str, Any]:
    os.chdir(native_root)
    sys.path.insert(0, str(native_root))
    import cv2  # type: ignore
    import torch  # type: ignore
    import inference as native_inference  # type: ignore
    import utils.conditions as native_conditions  # type: ignore
    from utils.misc import set_seed  # type: ignore
    expected = {
        "inference": (native_root / "inference.py").resolve(),
        "conditions": (native_root / "utils" / "conditions.py").resolve(),
    }
    actual = {
        "inference": Path(native_inference.__file__).resolve(),
        "conditions": Path(native_conditions.__file__).resolve(),
    }
    if actual != expected:
        raise RunnerContractError(f"Matrix-Game 2 module origin mismatch: {actual}")
    return {
        "cv2": cv2, "torch": torch, "inference": native_inference,
        "conditions": native_conditions, "set_seed": set_seed,
    }


def make_condition_builder(
    rows: list[dict[str, Any]], official_combine_data: Callable[..., Any], torch: Any,
) -> tuple[Callable[..., dict[str, Any]], dict[str, Any]]:
    native_output_frames = len(rows)
    expected_hash = canonical_sha256(rows)
    evidence: dict[str, Any] = {
        "call_count": 0,
        "expected_condition_rows_sha256": expected_hash,
        "observed_condition_rows_sha256": None,
        "official_encode_call_count": 0,
    }

    def builder(num_frames: int, num_samples_per_action: int = 4) -> dict[str, Any]:
        if num_frames != native_output_frames or num_samples_per_action != 4:
            raise RunnerContractError("unexpected Matrix-Game 2 condition-builder request")
        evidence["call_count"] += 1
        if evidence["call_count"] != 1:
            raise RunnerContractError("official condition builder was invoked more than once")

        cache: dict[str, dict[str, Any]] = {}
        for payload in rows:
            key = canonical_sha256(payload)
            if key in cache:
                continue
            keyboard = payload.get("keyboard")
            mouse = payload.get("mouse")
            if not isinstance(keyboard, list) or len(keyboard) != 4:
                raise RunnerContractError("invalid keyboard payload at runtime")
            if not isinstance(mouse, list) or len(mouse) != 2:
                raise RunnerContractError("invalid mouse payload at runtime")
            datum = [{
                "keyboard_condition": torch.tensor([keyboard] * 4),
                "mouse_condition": torch.tensor([mouse] * 4),
            }]
            encoded = official_combine_data(
                datum, num_frames=native_output_frames, keyboard_dim=4, mouse=True,
            )
            evidence["official_encode_call_count"] += 1
            if not isinstance(encoded, dict) or sorted(encoded) != [
                "keyboard_condition", "mouse_condition",
            ]:
                raise RunnerContractError("official combine_data returned malformed fields")
            expected_keyboard = torch.tensor(
                [keyboard] * native_output_frames, dtype=torch.float32,
            )
            expected_mouse = torch.tensor(
                [mouse] * native_output_frames, dtype=torch.float32,
            )
            if encoded["keyboard_condition"].shape != (native_output_frames, 4):
                raise RunnerContractError("official keyboard condition shape mismatch")
            if encoded["mouse_condition"].shape != (native_output_frames, 2):
                raise RunnerContractError("official mouse condition shape mismatch")
            if not torch.equal(encoded["keyboard_condition"].cpu(), expected_keyboard):
                raise RunnerContractError("official keyboard encoding changed the payload")
            if not torch.equal(encoded["mouse_condition"].cpu(), expected_mouse):
                raise RunnerContractError("official mouse encoding changed the payload")
            cache[key] = encoded

        keyboard_rows = torch.stack([
            cache[canonical_sha256(payload)]["keyboard_condition"][index]
            for index, payload in enumerate(rows)
        ])
        mouse_rows = torch.stack([
            cache[canonical_sha256(payload)]["mouse_condition"][index]
            for index, payload in enumerate(rows)
        ])
        expected_keyboard = torch.tensor(
            [payload["keyboard"] for payload in rows], dtype=torch.float32,
        )
        expected_mouse = torch.tensor(
            [payload["mouse"] for payload in rows], dtype=torch.float32,
        )
        if not torch.equal(keyboard_rows.cpu(), expected_keyboard):
            raise RunnerContractError("assembled keyboard timeline mismatch")
        if not torch.equal(mouse_rows.cpu(), expected_mouse):
            raise RunnerContractError("assembled mouse timeline mismatch")
        evidence["observed_condition_rows_sha256"] = expected_hash
        return {
            "keyboard_condition": keyboard_rows,
            "mouse_condition": mouse_rows,
        }

    return builder, evidence


@contextmanager
def installed_condition_builder(
    native_inference: Any, builder: Callable[..., dict[str, Any]],
) -> Iterator[None]:
    if not hasattr(native_inference, "Bench_actions_universal"):
        raise RunnerContractError("official inference has no Bench_actions_universal lookup")
    original = native_inference.Bench_actions_universal
    native_inference.Bench_actions_universal = builder
    try:
        yield
    finally:
        native_inference.Bench_actions_universal = original


@contextmanager
def installed_benchmark_writer(native_inference: Any) -> Iterator[dict[str, Any]]:
    """Write the unchanged upstream frame list at the benchmark's 16 fps."""

    process_video = getattr(native_inference, "process_video", None)
    globals_dict = getattr(process_video, "__globals__", None)
    if not isinstance(globals_dict, dict):
        raise RunnerContractError("official process_video globals are unavailable")
    original = globals_dict.get("export_to_video")
    original_process = getattr(native_inference, "process_video", None)
    if callable(original_process):
        def safe_process(*args, **kwargs):
            values = list(args)
            if len(values) >= 4 and (values[3] is None or (isinstance(values[3], str) and not __import__('pathlib').Path(values[3]).is_file())):
                values[3] = _safe_mouse_icon_path()
            return original_process(*values, **kwargs)
        native_inference.process_video = safe_process
    if not callable(original):
        raise RunnerContractError("official process_video export_to_video is unavailable")
    evidence: dict[str, Any] = {
        "call_count": 0,
        "expected_call_count": 2,
        "upstream_reference_writer_fps": UPSTREAM_REFERENCE_WRITER_FPS,
        "benchmark_benchmark_writer_fps": BENCHMARK_BENCHMARK_FPS,
        "decoded_frame_resampling": False,
        "frame_list_preserved": True,
        "upstream_requested_fps": [],
    }

    def export_at_benchmark_fps(*export_args: Any, **export_kwargs: Any) -> Any:
        evidence["call_count"] += 1
        if evidence["call_count"] > evidence["expected_call_count"]:
            raise RunnerContractError("official video exporter was invoked too many times")
        requested = export_kwargs.get("fps")
        if requested is None and len(export_args) >= 3:
            requested = export_args[2]
            export_args = (
                *export_args[:2], BENCHMARK_BENCHMARK_FPS, *export_args[3:],
            )
        else:
            export_kwargs["fps"] = BENCHMARK_BENCHMARK_FPS
        if float(requested) != float(UPSTREAM_REFERENCE_WRITER_FPS):
            raise RunnerContractError(
                f"upstream Matrix-Game 2 writer fps changed to {requested}"
            )
        evidence["upstream_requested_fps"].append(float(requested))
        return original(*export_args, **export_kwargs)

    globals_dict["export_to_video"] = export_at_benchmark_fps
    try:
        yield evidence
    finally:
        globals_dict["export_to_video"] = original
        if callable(original_process):
            native_inference.process_video = original_process


def inspect_video(
    path: Path, cv2: Any, expected_frames: int = NATIVE_OUTPUT_FRAMES,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
    height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    if decoded != expected_frames:
        raise RunnerContractError(f"native video decoded {decoded} frames")
    if (width, height) != (NATIVE_WIDTH, NATIVE_HEIGHT):
        raise RunnerContractError(f"native video size is {width}x{height}")
    if (not math.isfinite(fps)
            or abs(fps - BENCHMARK_BENCHMARK_FPS) > 0.01):
        raise RunnerContractError(f"native video FPS is {fps}")
    return {
        "decoded_frame_count": decoded, "fps": fps,
        "decoded_width_height": [width, height],
    }


def inference_payload(plan: ExecutionPlan, args: argparse.Namespace) -> dict[str, Any]:
    native = plan.native_request
    return {
        "config_path": str((args.native_source_root / NATIVE_CONFIG_RELATIVE).resolve()),
        "checkpoint_path": str(
            (args.checkpoint_root / NATIVE_CHECKPOINT_RELATIVE).resolve()
        ),
        "pretrained_model_path": str(args.checkpoint_root.resolve()),
        "num_output_latent_frames": native["num_output_latent_frames"],
        "num_output_pixel_frames": native["output_frame_count"],
        "seed": int(plan.selected.record["seed"]),
        "seed_source": "compiled_record",
        "seed_override_allowed": False,
        "mode": "universal",
        "timebase_profile": "benchmark_16fps_preserve_native_frames",
        "upstream_reference_writer_fps": UPSTREAM_REFERENCE_WRITER_FPS,
        "benchmark_benchmark_writer_fps": BENCHMARK_BENCHMARK_FPS,
        "decoded_frame_resampling": False,
    }


def generation_record(
    plan: ExecutionPlan, video: Path, args: argparse.Namespace,
    provenance: dict[str, Any], builder_evidence: dict[str, Any],
    writer_evidence: dict[str, Any], video_evidence: dict[str, Any],
) -> dict[str, Any]:
    record = plan.selected.record
    request = inference_payload(plan, args)
    native_control = {
        **plan.native_request,
        "runtime_builder_evidence": dict(builder_evidence),
    }
    return {
        "schema_version": RECORD_SCHEMA, "status": "COMPLETED",
        "task": record["task"], "case_id": record["case_id"],
        "probe_case_id": record["probe_case_id"], "state_id": record["state_id"],
        "seed": record["seed"], "adapter_id": record["adapter_id"],
        "adapter_fingerprint": record["adapter_fingerprint"],
        "request_contract_sha256": record["request_contract_sha256"],
        "schedule_contract_sha256": record["schedule_contract_sha256"],
        "compiled_action_record_path": str(plan.selected.compiled_path),
        "compiled_action_record_sha256": plan.selected.compiled_sha256,
        "compiled_selected_record_sha256": canonical_sha256(record),
        "condition_image_path": str(plan.input_image),
        "condition_image_sha256": plan.input_sha256,
        "source_transform": "center_crop",
        "source_transform_contract": plan.native_request["source_transform"],
        "generated_video_path": str(video),
        "generated_video_sha256": file_sha256(video),
        "native_inference_request": request,
        "native_inference_request_sha256": canonical_sha256(request),
        "native_control_evidence": native_control,
        "native_control_evidence_sha256": canonical_sha256(native_control),
        "writer_timebase_evidence": dict(writer_evidence),
        "formal_evaluation_frame_count": plan.native_request[
            "formal_evaluation_frame_count"
        ],
        "checkpoint_doses": plan.native_request["checkpoint_doses"],
        "measurement_checkpoints": plan.native_request["measurement_checkpoints"],
        "checkpoint_frame_indices": plan.native_request["checkpoint_frame_indices"],
        "measurement_contract_version": plan.native_request[
            "measurement_contract_version"
        ],
        "execution_output": {
            **video_evidence,
            "raw_native_output_frame_count": plan.native_request["output_frame_count"],
            "formal_evaluation_frame_count": plan.native_request[
                "formal_evaluation_frame_count"
            ],
            "evaluation_window": plan.native_request["evaluation_window"],
            "metric_frame_slice_half_open": plan.native_request[
                "metric_frame_slice_half_open"
            ],
            "measurement_checkpoints": plan.native_request[
                "measurement_checkpoints"
            ],
            "post_required_padding": plan.native_request["post_required_padding"],
        },
        **provenance,
    }


def stamp_executed(
    plan: ExecutionPlan, video: Path, checkpoint_sha256: str,
    generation_record_path: Path, native_control: dict[str, Any],
) -> dict[str, Any]:
    stamp_root = ROOT / "metrics" / "task2"
    sys.path.insert(0, str(stamp_root))
    from stamp_executed_action_record import stamp_record  # type: ignore
    executed = stamp_record(
        plan.selected.compiled_path, plan.input_image, video,
        str(plan.selected.record["case_id"]), checkpoint_sha256,
        "provided_checkpoint_sha256",
    )
    executed.update({
        "source_transform": "center_crop",
        "generation_record_path": str(generation_record_path.resolve()),
        "generation_record_sha256": file_sha256(generation_record_path),
        "generation_record_schema_version": RECORD_SCHEMA,
        "native_control_evidence": native_control,
        "native_control_evidence_sha256": canonical_sha256(native_control),
        "execution_output": {
            **executed["execution_output"],
            "raw_native_output_frame_count": native_control["output_frame_count"],
            "formal_evaluation_frame_count": native_control[
                "formal_evaluation_frame_count"
            ],
            "evaluation_window": native_control["evaluation_window"],
            "metric_frame_slice_half_open": native_control[
                "metric_frame_slice_half_open"
            ],
            "measurement_checkpoints": native_control["measurement_checkpoints"],
            "post_required_padding": native_control["post_required_padding"],
        },
    })
    return executed


def run_batch(
    plans: list[ExecutionPlan], args: argparse.Namespace, provenance: dict[str, Any],
) -> None:
    runtime = runtime_imports(args.native_source_root.resolve())
    from types import SimpleNamespace

    native_args = SimpleNamespace(
        config_path=str((args.native_source_root / NATIVE_CONFIG_RELATIVE).resolve()),
        checkpoint_path=str(
            (args.checkpoint_root / NATIVE_CHECKPOINT_RELATIVE).resolve()
        ),
        pretrained_model_path=str(args.checkpoint_root.resolve()),
        img_path=str(plans[0].input_image),
        output_folder=str(plans[0].case_dir / "native_tmp"),
        num_output_frames=plans[0].native_request["num_output_latent_frames"],
        seed=int(plans[0].selected.record["seed"]),
    )
    pipe = runtime["inference"].InteractiveGameInference(native_args)
    initial_mode = pipe.config.get("mode")
    if initial_mode != "universal":
        raise RunnerContractError(f"pinned Matrix-Game 2 config mode is {initial_mode!r}")

    for index, plan in enumerate(plans, start=1):
        plan.case_dir.mkdir(parents=True, exist_ok=False)
        native_dir = plan.case_dir / "native_tmp"
        native_dir.mkdir()
        native_args.img_path = str(plan.input_image)
        native_args.output_folder = str(native_dir)
        native_args.seed = int(plan.selected.record["seed"])
        native_args.num_output_frames = plan.native_request["num_output_latent_frames"]
        pipe.config["mode"] = "universal"
        builder, builder_evidence = make_condition_builder(
            plan.native_request["condition_rows"],
            runtime["conditions"].combine_data,
            runtime["torch"],
        )
        runtime["set_seed"](native_args.seed)
        with installed_condition_builder(runtime["inference"], builder):
            with installed_benchmark_writer(runtime["inference"]) as writer_evidence:
                pipe.generate_videos()
        if builder_evidence["call_count"] != 1:
            raise RunnerContractError("official inference did not invoke strict condition builder")
        if (
            builder_evidence["observed_condition_rows_sha256"]
            != plan.native_request["condition_rows_sha256"]
        ):
            raise RunnerContractError("runtime condition evidence does not match compiled rows")
        if writer_evidence["call_count"] != writer_evidence["expected_call_count"]:
            raise RunnerContractError(
                "official inference did not invoke both bound video writers"
            )

        native_video = native_dir / "demo.mp4"
        if not native_video.is_file():
            raise FileNotFoundError(native_video)
        video_evidence = inspect_video(
            native_video, runtime["cv2"], plan.native_request["output_frame_count"],
        )
        video = plan.case_dir / "raw_generated.mp4"
        os.replace(native_video, video)
        icon_video = native_dir / "demo_icon.mp4"
        if icon_video.exists():
            icon_video.unlink()
        shutil.rmtree(native_dir)

        metadata = generation_record(
            plan, video, args, provenance, builder_evidence, writer_evidence,
            video_evidence,
        )
        generation_path = plan.case_dir / "generation_record.json"
        generation_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        native_control = metadata["native_control_evidence"]
        executed = stamp_executed(
            plan, video, provenance["checkpoint_fingerprint"], generation_path,
            native_control,
        )
        (plan.case_dir / "executed_action_record.json").write_text(
            json.dumps(executed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print(f"[{index}/{len(plans)}] SAVED {video}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task1-compiled", type=Path, action="append", default=[])
    parser.add_argument("--task2-compiled", type=Path, action="append", default=[])
    parser.add_argument("--task3-compiled", type=Path, action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--compound-g3-smoke", action="store_true",
        help="explicitly acknowledge one hash-bound pre-acceptance compound smoke",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-image", type=Path)
    inputs.add_argument("--state-input", action="append", default=[], metavar="STATE_ID=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--native-source-root", type=Path, default=DEFAULT_NATIVE_ROOT)
    parser.add_argument("--adapter", type=Path, default=ADAPTER_PATH)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.native_source_root = args.native_source_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    adapter = load_frozen_adapter(args.adapter)
    selected = load_selected(args, adapter)
    plans = build_plans(selected, adapter, args)
    summary = {
        "selected": len(plans),
        "by_task": {
            task: sum(plan.selected.record["task"] == task for plan in plans)
            for task in TASK_DIR
        },
    }
    if args.preflight_only:
        print(json.dumps({"status": "PREFLIGHT_OK_CPU_ONLY", **summary}, sort_keys=True))
        return
    if not args.checkpoint_root.is_dir():
        raise FileNotFoundError(args.checkpoint_root)
    checkpoint = checkpoint_provenance(args.checkpoint_root)
    provenance = {
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "adapter_path": str(args.adapter.resolve()),
        "adapter_sha256": file_sha256(args.adapter.resolve()),
        "checkpoint_root": str(args.checkpoint_root),
        **checkpoint,
        "native_source_root": str(args.native_source_root),
        "native_source_revision": validate_native_sources(args.native_source_root),
        "runtime_environment": {"process_count": 1, "device_id": 0},
    }
    provenance.update(bind_output_namespace(args.output, adapter, checkpoint))
    run_batch(plans, args, provenance)


if __name__ == "__main__":
    main()
