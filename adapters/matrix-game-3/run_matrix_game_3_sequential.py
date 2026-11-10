#!/usr/bin/env python3
"""Run a hash-bound sequential Task 3 smoke with Matrix-Game 3.

Contract parsing and preflight are stdlib-only.  Heavy Matrix-Game imports occur
only after every selected record, input image and immutable source has passed
validation.  Legacy dose 1/2/3 keeps one 57-frame native iteration.  A
consolidated held-action dose 16 freezes dose 1/2/3/5/8/16 measurements inside one
rollout.  This experimental runner adds ordered, dose-1 control switching in
the same pipeline call, without RGB chaining or any native motion-gain change.
It remains smoke-only until the resulting video is manually accepted.

The separate ``matrix_game_3_final_atomic_v3`` adapter extends only the
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
import struct
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from strideworld_compiled_subset_contract import (  # noqa: E402
    CompiledSubsetContractError, is_queue_subset_path,
    validate_compound_container_authorization, validate_or_raise,
)


MODEL_ID = "matrix_game_3"
ADAPTER_PATH = (
    ROOT / "action_interface" / "adapters_task3_smoke" / "matrix_game_3.json"
)
ADAPTER_ID = "matrix_game_3_lite_v1"
HELD_ADAPTER_ID = "matrix_game_3_held8_v2"
ATOMIC_V3_ADAPTER_ID = "matrix_game_3_atomic_v3"
FINAL_ADAPTER_ID = "matrix_game_3_final_atomic_v3"
FINAL_ROUTE_LENGTHS = (5, 8, 11, 14)
FINAL_COMPILE_MODE = "FINAL_PORTABLE_RECORD_CPU_ONLY"
FINAL_EVIDENCE_STATUS = "CPU_COMPILED_NATIVE_REQUEST_PLAN_ONLY"
FINAL_GPU_STATUS = "NOT_RUN"
COMPOUND_PROTOCOL_PATH = ROOT / "action_interface" / "protocol.compound.v1.json"
COMPOUND_PROTOCOL_VERSION = "strideworld-action-protocol-compound-v1"
SUPPORTED_ADAPTER_IDS = {
    ADAPTER_ID, HELD_ADAPTER_ID, ATOMIC_V3_ADAPTER_ID, FINAL_ADAPTER_ID,
}
COMPILED_SCHEMA = "strideworld-compiled-actions-lite-v1"
RECORD_SCHEMA = "strideworld-matrix-game-3-generation-record-v1"
DEFAULT_NATIVE_ROOT = Path(
    "<path>/Matrix-Game-3"
)
DEFAULT_CHECKPOINT = Path("<path>/Matrix-Game-3.0")
DEFAULT_PROMPT = (
    "First-person view in a stable realistic environment with detailed geometry "
    "and natural lighting."
)
NATIVE_OUTPUT_FRAMES = 57
NATIVE_FIRST_ITERATION_FRAMES = 57
NATIVE_ADDITIONAL_ITERATION_FRAMES = 40
LEGACY_DOSES = {1.0, 2.0, 3.0}
HELD_DOSE = 16.0
HELD_CHECKPOINT_DOSES = (1.0, 2.0, 3.0, 5.0, 8.0, 16.0)
MEASUREMENT_CONTRACT_VERSION = "strideworld-task2-checkpoints-v1"
NATIVE_SOURCE_SHA256 = {
    "pipeline/inference_pipeline.py":
        "1cfc024e8db249a365276866b50e351ab7b1a98f8d325a0afce02ea2cd083f40",
    "utils/conditions.py":
        "79d09e6e818cf5b346fa682ab11f3873139a4b8a5c170a1beaada0f0eb246345",
    "utils/utils.py":
        "2a5bd17f4f5d0dc150cee194d858bb7724628a0cc5bc919ea460fc7ed068cdaf",
    "wan/configs/__init__.py":
        "126c17c5f11eca16c3903b4496e6781bbf41700ae962f5f29e1a5dd9f7aa6082",
}
CHECKPOINT_REQUIRED_FILES = (
    "base_distilled_model/config.json",
    "base_distilled_model/diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "MG-LightVAE_v2.pth",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)
CHECKPOINT_REPO = "Skywork/Matrix-Game-3.0"
CHECKPOINT_REVISION = "382aa382a03cec057360755aa72c7735025770bb"
NATIVE_BASE_COMMIT = "71c3cd7f741311f8100f6cf9cde942b6c1378d11"
NATIVE_PATCH_PATHS = (
    "pipeline/inference_pipeline.py", "utils/conditions.py", "utils/utils.py",
)
NATIVE_PATCH_SHA256 = "37fe06f227e1633441f320cca752d87541dcb2cfdd1aa32cc2f997c8e8c676f0"
NATIVE_PATCH_FILE = (
    ROOT / "scripts" / "patches" / "matrix_game_3_strideworld_action_sequence.patch"
)
NATIVE_PATCH_MANIFEST = NATIVE_PATCH_FILE.with_suffix(".json")
SUPPORTED_ACTIONS = {
    "NO_OP", "MOVE_FORWARD", "MOVE_LEFT", "MOVE_BACKWARD", "MOVE_RIGHT",
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
    """Raised before inference when immutable execution evidence disagrees."""


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
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


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
        raise RunnerContractError("Matrix-Game 3 adapter_id mismatch")
    expected_timing = {
        "fps": 17, "frames": 57, "conditioning_frames": 1,
        "native_tick_frames": 1,
        "base_control_ticks": 4
        if adapter_id in {ATOMIC_V3_ADAPTER_ID, FINAL_ADAPTER_ID}
        else 14,
    }
    if adapter.get("timing") != expected_timing:
        raise RunnerContractError("Matrix-Game 3 frozen timing mismatch")
    if adapter.get("representation") != "multi_hot":
        raise RunnerContractError("Matrix-Game 3 representation mismatch")
    sequential = adapter.get("sequential", {})
    expected_smoke_sequential = {
        "supported": True,
        "verified_runner": "scripts/run_matrix_game_3_sequential.py",
        "verification_source": "preacceptance_task3_sequential_smoke_candidate_v1",
        "max_steps": 8,
        "switch_granularity": "native_tick_boundary",
        "generation_extension_supported": True,
        "generation_extension_verification_source": (
            "scripts/run_matrix_game_3_strideworld_compiled.py:"
            "dose8_native_internal_iterations_v1"
        ),
        "per_step_reference_frame": "current_camera_cv_rh_body",
        "initial_state_pose_policy": "required_if_conditioning_frames_zero",
    }
    expected_final_sequential = {
        "supported": True,
        "verified_runner": "scripts/run_matrix_game_3_sequential.py",
        "verification_source": (
            "cpu_compiled_native_request_plan_final_"
            "v9_l5_l8_l11_l14_v1_no_gpu_execution"
        ),
        "max_steps": 23,
        "switch_granularity": "native_tick_boundary",
        "generation_extension_supported": True,
        "generation_extension_verification_source": (
            "scripts/run_matrix_game_3_sequential.py:"
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
            "Matrix-Game 3 sequential capability/evidence mismatch"
        )
    final_contract = adapter.get("final_portable_contract")
    if adapter_id == FINAL_ADAPTER_ID:
        if final_contract != expected_final_contract:
            raise RunnerContractError("Matrix-Game 3 final contract mismatch")
    elif final_contract is not None:
        raise RunnerContractError("Matrix-Game 3 smoke adapter contains final metadata")
    for action in SUPPORTED_ACTIONS:
        payload = adapter.get("actions", {}).get(action)
        if not isinstance(payload, dict):
            raise RunnerContractError(f"Matrix-Game 3 missing action {action}")
    return adapter


def validate_native_sources(root: Path) -> list[dict[str, str]]:
    evidence = []
    for relative, expected in NATIVE_SOURCE_SHA256.items():
        path = (root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != expected:
            raise RunnerContractError(f"Native source SHA256 mismatch: {relative}")
        evidence.append({"path": str(path), "sha256": actual})
    return evidence


def validate_checked_in_patch(
    patch_path: Path = NATIVE_PATCH_FILE,
    manifest_path: Path = NATIVE_PATCH_MANIFEST,
) -> dict[str, Any]:
    patch = patch_path.resolve()
    manifest_file = manifest_path.resolve()
    if not patch.is_file():
        raise FileNotFoundError(patch)
    manifest = read_object(manifest_file)
    expected_paths = [f"Matrix-Game-3/{relative}" for relative in NATIVE_PATCH_PATHS]
    expected = {
        "schema_version": "strideworld-third-party-patch-manifest-v1",
        "component": "Matrix-Game-3", "repository": "SkyworkAI/Matrix-Game",
        "base_commit": NATIVE_BASE_COMMIT, "patch_file": patch.name,
        "patch_sha256": NATIVE_PATCH_SHA256, "paths": expected_paths,
        "apply_from": "third_party/Matrix-Game",
        "runner_policy": "verify_only_fail_closed_never_auto_apply",
    }
    if manifest != expected:
        raise RunnerContractError("checked-in Matrix-Game 3 patch manifest mismatch")
    patch_sha256 = file_sha256(patch)
    if patch_sha256 != NATIVE_PATCH_SHA256:
        raise RunnerContractError("checked-in Matrix-Game 3 patch SHA256 mismatch")
    return {
        "checked_in_patch_path": str(patch),
        "checked_in_patch_sha256": patch_sha256,
        "checked_in_patch_manifest_path": str(manifest_file),
        "checked_in_patch_manifest_sha256": file_sha256(manifest_file),
        "patch_apply_from": expected["apply_from"],
        "patch_policy": expected["runner_policy"],
    }


def validate_native_revision(root: Path) -> dict[str, Any]:
    checked_in = validate_checked_in_patch()
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.decode().strip()
        patch = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "--", *NATIVE_PATCH_PATHS],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RunnerContractError(f"Cannot validate Matrix-Game 3 git state: {error}") from error
    patch_sha256 = hashlib.sha256(patch).hexdigest()
    if commit != NATIVE_BASE_COMMIT or patch_sha256 != NATIVE_PATCH_SHA256:
        raise RunnerContractError("Matrix-Game 3 base commit/local patch mismatch")
    return {
        "base_commit": commit,
        "local_patch_paths": list(NATIVE_PATCH_PATHS),
        "local_patch_sha256": patch_sha256,
        **checked_in,
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
    if isinstance(required_frames, bool) or not isinstance(required_frames, int):
        raise RunnerContractError("required output frames must be an integer")
    extra = max(0, required_frames - NATIVE_FIRST_ITERATION_FRAMES)
    iterations = 1 + int(math.ceil(extra / NATIVE_ADDITIONAL_ITERATION_FRAMES))
    output_frames = (
        NATIVE_FIRST_ITERATION_FRAMES
        + (iterations - 1) * NATIVE_ADDITIONAL_ITERATION_FRAMES
    )
    return {"num_iterations": iterations, "output_frame_count": output_frames}


def validate_measurement_checkpoints(
    record: dict[str, Any], adapter: dict[str, Any],
) -> tuple[list[float], list[dict[str, Any]]]:
    request = record.get("request_contract", {})
    schedule = record.get("schedule_contract", {})
    if record.get("task") != "TASK2":
        if ("checkpoint_doses" in record or "measurement_checkpoints" in record
                or "measurement_contract_version" in record
                or "checkpoint_doses" in request
                or "measurement_checkpoints" in schedule):
            raise RunnerContractError("measurement checkpoints are Task 2 only")
        return [], []
    step_dose = float(record["steps"][0]["dose_requested"])
    top_doses = record.get("checkpoint_doses")
    request_doses = request.get("checkpoint_doses")
    top_points = record.get("measurement_checkpoints")
    schedule_points = schedule.get("measurement_checkpoints")
    explicit = any(value is not None for value in (
        top_doses, request_doses, top_points, schedule_points,
        record.get("measurement_contract_version"),
    ))
    base_ticks = float(adapter["timing"]["base_control_ticks"])
    tick_frames = int(adapter["timing"]["native_tick_frames"])
    conditioning = int(adapter["timing"]["conditioning_frames"])
    if not explicit:
        if step_dose == HELD_DOSE:
            raise RunnerContractError("dose 16 requires explicit checkpoint_doses")
        tick = round_half_up(base_ticks * step_dose)
        return [step_dose], [{
            "dose_requested": step_dose, "native_tick": tick,
            "frame_index": conditioning + tick * tick_frames - 1,
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
    expected = []
    for dose in doses:
        tick = round_half_up(base_ticks * dose)
        expected.append({
            "dose_requested": dose, "native_tick": tick,
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
        raise RunnerContractError(f"Unsupported formal Matrix-Game 3 actions: {actions}")
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
        raise RunnerContractError("Matrix-Game 3 adapter lacks native simultaneous merge")
    merged: dict[str, list[float]] = {}
    for field in ("keyboard", "mouse"):
        vectors = [adapter["actions"][action][field] for action in actions]
        if len({len(vector) for vector in vectors}) != 1:
            raise RunnerContractError(f"cannot merge Matrix-Game 3 {field} dimensions")
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
        raise RunnerContractError("Matrix-Game 3 steps must be non-empty objects")
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
    expected_tail = int(adapter["timing"]["base_control_ticks"])
    conditioning = int(adapter["timing"]["conditioning_frames"])
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
            "Matrix-Game 3 sequential runner accepts only Task 2/Task 3"
        )
    if task == "TASK2" and adapter.get("adapter_id") != FINAL_ADAPTER_ID:
        raise RunnerContractError(
            "Matrix-Game 3 smoke adapter accepts only Task 3; "
            "Task 2 mixed execution requires the final adapter"
        )
    if (record.get("status") != "SUPPORTED"
            or record.get("adapter_id") != adapter.get("adapter_id")):
        raise RunnerContractError("selected record is not supported by Matrix-Game 3")
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
    if record.get("native_tick_frames") != 1 or record.get("fps") != 17:
        raise RunnerContractError("compiled native timing fields mismatch")
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


def payload_token(payload: dict[str, Any], adapter: dict[str, Any]) -> str:
    mapping = {
        "NO_OP": "none", "MOVE_FORWARD": "w", "MOVE_LEFT": "a",
        "MOVE_BACKWARD": "s", "MOVE_RIGHT": "d", "YAW_LEFT": "camera_l",
        "YAW_RIGHT": "camera_r",
    }
    matches = [mapping[action] for action in SUPPORTED_ACTIONS
               if payload == adapter["actions"][action]]
    if len(matches) != 1:
        raise RunnerContractError("native payload does not uniquely map to a supported action")
    token = matches[0]
    # turn-magnitude rerun: upstream camera tokens default to 0.1; pass the adapter magnitude
    # through the checked-in patch's "@scale" suffix when it differs.
    if token.startswith("camera_"):
        scale = max(abs(float(v)) for v in payload["mouse"])
        if abs(scale - 0.1) > 1e-12:
            token = f"{token}@{scale:g}"
    return token


def compact_action_sequence(tokens: list[str]) -> str:
    runs: list[tuple[str, int]] = []
    for token in tokens:
        if runs and runs[-1][0] == token:
            runs[-1] = (token, runs[-1][1] + 1)
        else:
            runs.append((token, 1))
    return ",".join(f"{token}-{count}" for token, count in runs)


def native_request(record: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    timeline = list(record["native_timeline"])
    required = int(record["required_output_frames"])
    if len(timeline) != required - 1:
        raise RunnerContractError("compiled timeline must control every scored transition")
    output = native_output_plan(required)
    native_transition_capacity = output["output_frame_count"] - 1
    padding_transitions = native_transition_capacity - len(timeline)
    if padding_transitions < 0:
        raise RunnerContractError("native output has no capacity for required timeline")
    # Matrix-Game consumes one action row per output frame, but only rows 0..55
    # advance the 57 camera poses. Row 56 is an explicit unused terminator.
    padded = timeline + [adapter["actions"]["NO_OP"]] * padding_transitions
    padded.append(adapter["actions"]["NO_OP"])
    compound = len(record["steps"][0]["canonical_actions"]) > 1
    if compound:
        action_sequence = f"strideworld_compound_payload_rows-{len(padded)}"
        condition_injection = "case_local_utils_utils_Action_sequence_universal_override"
    else:
        tokens = [payload_token(payload, adapter) for payload in padded]
        action_sequence = compact_action_sequence(tokens)
        condition_injection = "checked_in_action_sequence_patch"
    checkpoint_doses, checkpoints = validate_measurement_checkpoints(record, adapter)
    return {
        "output_frame_count": output["output_frame_count"],
        "num_iterations": output["num_iterations"],
        "native_rollout_mode": "single_pipeline_call_with_internal_iterations",
        "external_rgb_last_frame_chaining": False,
        "action_sequence": action_sequence,
        "condition_injection": condition_injection,
        "condition_rows": padded,
        "condition_rows_sha256": canonical_sha256(padded),
        "compiled_timeline_ticks": len(timeline),
        "command_ticks": int(record["command_window"]["end_tick_exclusive"]),
        "terminal_noop_tail_ticks": int(record["observation_window"]["tail_ticks"]),
        "native_action_token_count": len(padded),
        "native_transition_capacity": native_transition_capacity,
        "unscored_padding_transitions": padding_transitions,
        "unused_terminal_action_tokens": 1,
        "padded_native_timeline_sha256": canonical_sha256(padded),
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
    candidates = [item for item in payload["records"] if isinstance(item, dict)
                  and item.get("adapter_id") == adapter.get("adapter_id")
                  and item.get("task") == task]
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
            f"No supported Matrix-Game 3 {task} records in {source}; "
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


def bind_input(record: dict[str, Any], state_inputs: dict[str, Path],
               fallback: Path | None) -> tuple[Path, str]:
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
        raise RunnerContractError("formal Matrix-Game 3 condition image must be PNG")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise RunnerContractError("condition PNG has invalid dimensions")
    return int(width), int(height)


def source_transform_contract(source_size: tuple[int, int]) -> dict[str, Any]:
    source_width, source_height = source_size
    target_width, target_height = 1280, 704
    target_ratio = target_height / target_width
    if source_height / source_width > target_ratio:
        crop_height, crop_width = int(source_width * target_ratio), source_width
    else:
        crop_height, crop_width = source_height, int(source_height / target_ratio)
    crop_y = int(round((source_height - crop_height) / 2.0))
    crop_x = int(round((source_width - crop_width) / 2.0))
    return {
        "mode": "center_crop", "interpolation": "bicubic",
        "align_corners": True, "antialias": True,
        "source_width_height": [source_width, source_height],
        "crop_xywh": [crop_x, crop_y, crop_width, crop_height],
        "native_contract_width_height": [target_width, target_height],
        "decoded_output_width_height": [target_width, target_height],
        "evidence": (
            "Matrix-Game-3 utils.transform.CenterCropResizeVideo: center_crop_th_tw "
            "then bicubic resize"
        ),
    }


def output_case_dir(root: Path, record: dict[str, Any]) -> Path:
    probe_id = str(record["probe_case_id"])
    if not SAFE_ID.fullmatch(probe_id):
        raise RunnerContractError(f"unsafe probe_case_id: {probe_id!r}")
    return root.resolve() / TASK_DIR[str(record["task"])] / probe_id


def build_plans(selected: list[SelectedRecord], adapter: dict[str, Any],
                args: argparse.Namespace) -> list[ExecutionPlan]:
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


def checkpoint_provenance(checkpoint_root: Path) -> dict[str, Any]:
    snapshot_path = checkpoint_root / "snapshot_manifest.json"
    snapshot = read_object(snapshot_path)
    if snapshot.get("repo") != CHECKPOINT_REPO or snapshot.get("revision") != CHECKPOINT_REVISION:
        raise RunnerContractError("Matrix-Game 3 checkpoint repo/revision mismatch")
    if snapshot.get("failures") != []:
        raise RunnerContractError("Matrix-Game 3 snapshot manifest contains failures")
    listed = snapshot.get("files")
    if not isinstance(listed, list) or any(
        relative not in listed for relative in CHECKPOINT_REQUIRED_FILES
    ):
        raise RunnerContractError("snapshot manifest omits a required runtime checkpoint file")
    files = []
    for relative in CHECKPOINT_REQUIRED_FILES:
        path = (checkpoint_root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({
            "relative_path": relative, "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    contract = {
        "repo": CHECKPOINT_REPO, "revision": CHECKPOINT_REVISION,
        "variant": "base_distilled", "required_runtime_files": files,
    }
    return {
        "checkpoint_contract": contract,
        "checkpoint_fingerprint": canonical_sha256(contract),
        "checkpoint_fingerprint_source": "required_runtime_files_manifest_sha256",
        "snapshot_manifest_path": str(snapshot_path.resolve()),
        "snapshot_manifest_sha256": file_sha256(snapshot_path.resolve()),
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
            raise RunnerContractError(
                f"non-empty output root lacks {RUN_NAMESPACE_FILE}: {root}"
            )
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
    icon = native_root / "assets" / "images" / "mouse.png"
    if not icon.is_file() or icon.stat().st_size < 100:
        icon.parent.mkdir(parents=True, exist_ok=True)
        icon.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAE0lEQVR4nGNgGAWjYBSMAgYwAAAEEAABsax5zAAAAABJRU5ErkJggg=="
        ))
    os.chdir(native_root)
    sys.path.insert(0, str(native_root))
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from PIL import Image  # type: ignore
    from pipeline.inference_pipeline import MatrixGame3Pipeline  # type: ignore
    import utils.utils as condition_utils  # type: ignore
    from utils.misc import set_seed  # type: ignore
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # type: ignore
    return {
        "cv2": cv2, "np": np, "torch": torch, "Image": Image,
        "Pipeline": MatrixGame3Pipeline, "set_seed": set_seed,
        "condition_utils": condition_utils,
        "MAX_AREA_CONFIGS": MAX_AREA_CONFIGS, "WAN_CONFIGS": WAN_CONFIGS,
    }


@contextmanager
def installed_compound_condition_rows(
    condition_utils: Any, torch: Any, native_request: dict[str, Any],
):
    """Install one case-local exact multi-hot schedule at the patched lookup."""

    if native_request["condition_injection"] != (
        "case_local_utils_utils_Action_sequence_universal_override"
    ):
        yield
        return
    rows = native_request["condition_rows"]
    expected_sequence = native_request["action_sequence"]
    original = condition_utils.Action_sequence_universal

    def replacement(num_frames: int, action_sequence: str) -> dict[str, Any]:
        if num_frames != len(rows) or action_sequence != expected_sequence:
            raise RunnerContractError("Matrix-Game 3 compound condition lookup mismatch")
        keyboard = [row["keyboard"] for row in rows]
        mouse = [row["mouse"] for row in rows]
        return {
            "keyboard_condition": torch.tensor(keyboard),
            "mouse_condition": torch.tensor(mouse),
        }

    condition_utils.Action_sequence_universal = replacement
    try:
        yield
    finally:
        condition_utils.Action_sequence_universal = original


def inference_payload(plan: ExecutionPlan, args: argparse.Namespace) -> dict[str, Any]:
    prompt = plan.selected.record.get("scene_prompt", args.prompt)
    if not isinstance(prompt, str) or not prompt.strip():
        raise RunnerContractError("Matrix-Game 3 prompt must be a non-empty string")
    return {
        "prompt": prompt, "height": 704, "width": 1280,
        "num_frames": plan.native_request["output_frame_count"],
        "num_iterations": plan.native_request["num_iterations"],
        "num_inference_steps": 3, "guide_scale": 5.0, "shift": 5.0,
        "seed": int(plan.selected.record["seed"]),
        "action_sequence": plan.native_request["action_sequence"],
        "use_base_model": False, "use_int8": True, "compile_vae": True,
        "vae_type": "mg_lightvae", "lightvae_pruning_rate": 0.5,
        "sampling_profile_provenance": "pinned_upstream_reference",
        "benchmark_vae_disclosure": "unknown",
    }


def save_raw_video(
    video: Any, path: Path, cv2: Any, np: Any,
    expected_frames: int = NATIVE_OUTPUT_FRAMES,
) -> int:
    frames = (
        ((video.detach().cpu().permute(1, 2, 3, 0).float() + 1.0) * 127.5)
        .clamp(0, 255).numpy().astype(np.uint8)
    )
    if frames.shape[0] != expected_frames:
        raise RunnerContractError(f"native runner returned {frames.shape[0]} frames")
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 17.0, (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RunnerContractError(f"cannot open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    if decoded != expected_frames:
        raise RunnerContractError(f"encoded video decoded {decoded} frames")
    return decoded


def generation_record(plan: ExecutionPlan, video: Path, args: argparse.Namespace,
                      provenance: dict[str, Any]) -> dict[str, Any]:
    record = plan.selected.record
    request = inference_payload(plan, args)
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
        "native_control_evidence": plan.native_request,
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
            "decoded_frame_count": plan.native_request["output_frame_count"],
            "raw_native_output_frame_count": plan.native_request["output_frame_count"],
            "formal_evaluation_frame_count": plan.native_request[
                "formal_evaluation_frame_count"
            ],
            "fps": 17,
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
    generation_record_path: Path,
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
        "execution_output": {
            **executed["execution_output"],
            "raw_native_output_frame_count": plan.native_request[
                "output_frame_count"
            ],
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
            "post_required_padding": plan.native_request[
                "post_required_padding"
            ],
        },
    })
    return executed


def run_batch(plans: list[ExecutionPlan], args: argparse.Namespace,
              provenance: dict[str, Any]) -> None:
    runtime = runtime_imports(args.native_source_root.resolve())
    from types import SimpleNamespace

    first_dir = plans[0].case_dir / "native"
    config = SimpleNamespace(
        size="704*1280", ckpt_dir=str(args.checkpoint.resolve()),
        output_dir=str(first_dir), save_name="native_overlay", action="random",
        action_sequence=plans[0].native_request["action_sequence"],
        num_iterations=plans[0].native_request["num_iterations"],
        use_async_vae=False, async_vae_warmup_iters=0, compile_vae=True,
        vae_type="mg_lightvae", lightvae_pruning_rate=0.5, use_int8=True,
        verify_quant=False,
    )
    pipe = runtime["Pipeline"](
        config=runtime["WAN_CONFIGS"]["matrix_game3"],
        checkpoint_dir=str(args.checkpoint.resolve()), device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, args=config, fa_version=None, use_base_model=False,
    )
    for index, plan in enumerate(plans, start=1):
        plan.case_dir.mkdir(parents=True, exist_ok=False)
        native_dir = plan.case_dir / "native"
        native_dir.mkdir()
        config.output_dir = str(native_dir)
        config.action_sequence = plan.native_request["action_sequence"]
        config.num_iterations = plan.native_request["num_iterations"]
        pipe.output_dir = str(native_dir)
        seed = int(plan.selected.record["seed"])
        runtime["set_seed"](seed)
        request = inference_payload(plan, args)
        compound_context = installed_compound_condition_rows(
            runtime["condition_utils"], runtime["torch"], plan.native_request,
        ) if plan.native_request["condition_injection"].startswith("case_local") else nullcontext()
        with compound_context:
            video_tensor = pipe.generate(
                request["prompt"], runtime["Image"].open(plan.input_image).convert("RGB"),
                max_area=runtime["MAX_AREA_CONFIGS"][config.size], shift=request["shift"],
                num_inference_steps=request["num_inference_steps"],
                guide_scale=request["guide_scale"], seed=seed,
                use_base_model=False, args=config,
            )
        if video_tensor is None:
            raise RunnerContractError("native runner returned no video tensor")
        video = plan.case_dir / "raw_generated.mp4"
        save_raw_video(
            video_tensor, video, runtime["cv2"], runtime["np"],
            plan.native_request["output_frame_count"],
        )
        metadata = generation_record(plan, video, args, provenance)
        generation_path = plan.case_dir / "generation_record.json"
        generation_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        executed = stamp_executed(
            plan, video, provenance["checkpoint_fingerprint"], generation_path,
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
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--native-source-root", type=Path, default=DEFAULT_NATIVE_ROOT)
    parser.add_argument("--adapter", type=Path, default=ADAPTER_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = load_frozen_adapter(args.adapter)
    selected = load_selected(args, adapter)
    plans = build_plans(selected, adapter, args)
    summary = {"selected": len(plans), "by_task": {
        task: sum(plan.selected.record["task"] == task for plan in plans)
        for task in TASK_DIR
    }}
    if args.preflight_only:
        print(json.dumps({"status": "PREFLIGHT_OK_CPU_ONLY", **summary}, sort_keys=True))
        return
    checkpoint_root = args.checkpoint.resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(checkpoint_root)
    checkpoint = checkpoint_provenance(checkpoint_root)
    provenance = {
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "adapter_path": str(args.adapter.resolve()),
        "adapter_sha256": file_sha256(args.adapter.resolve()),
        "checkpoint_path": str(checkpoint_root),
        **checkpoint,
        "native_source_files": validate_native_sources(args.native_source_root.resolve()),
        "native_source_revision": validate_native_revision(args.native_source_root.resolve()),
        "runtime_environment": {"process_count": 1, "device_id": 0},
    }
    provenance.update(bind_output_namespace(args.output, adapter, checkpoint))
    run_batch(plans, args, provenance)


if __name__ == "__main__":
    main()
