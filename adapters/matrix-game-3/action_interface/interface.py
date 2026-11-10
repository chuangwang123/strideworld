"""Core validation and compilation for the StrideWorld action interface.

The module is intentionally independent of model frameworks.  It produces a
fully explicit native control timeline that a server-side model runner can
consume without changing action semantics after seeing generated videos.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


class InterfaceError(ValueError):
    """Raised when a protocol, adapter, or case violates the interface contract."""


ACTION_REFERENCE_FRAME = "start_camera_cv_rh_right_action"
SEQUENTIAL_REFERENCE_FRAME = "current_camera_cv_rh_body"
FORMAL_CASE_IDENTITY_FIELDS = (
    "state_id", "first_frame_sha256", "probe_plan_sha256", "probe_case_id",
)
CASE_IDENTITY_BOUND = "FORMAL_BOUND"
CASE_IDENTITY_UNBOUND = "DIAGNOSTIC_LEGACY_UNBOUND"
TASK2_MEASUREMENT_CONTRACT_VERSION = "strideworld-task2-checkpoints-v1"
ATOMIC_PROTOCOL_VERSION = "strideworld-action-protocol-atomic-v3"
COMPOUND_PROTOCOL_VERSION = "strideworld-action-protocol-compound-v1"
ATOMIC_UNIT_VERSION = "strideworld-measurable-atomic-unit-v1"
ATOMIC_ALLOWED_DOSES = [1.0, 2.0, 3.0, 5.0, 8.0, 16.0]


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InterfaceError(f"Cannot read JSON {source}: {error}") from error
    if not isinstance(value, dict):
        raise InterfaceError(f"Top-level JSON value must be an object: {source}")
    return value


def discover_adapters(adapter_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load adapter JSON files, keyed by their filename stem."""

    directory = Path(adapter_dir)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise InterfaceError(f"No adapter JSON files found in {directory}")
    return {path.stem: read_json(path) for path in paths}


def require_keys(value: dict[str, Any], keys: Iterable[str], label: str) -> None:
    """Require a small set of object keys and report all missing keys together."""

    missing = sorted(set(keys) - value.keys())
    if missing:
        raise InterfaceError(f"{label} is missing keys: {', '.join(missing)}")


def is_positive_number(value: Any) -> bool:
    """Return true for positive int/float values, excluding booleans."""

    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _validate_protocol_vocabulary(protocol: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """Validate the shared action, dose, and sequencing vocabulary."""

    actions = protocol["actions"]
    if not isinstance(actions, list) or not actions or len(actions) != len(set(actions)):
        raise InterfaceError("protocol.actions must be a non-empty unique list")
    if "NO_OP" not in actions:
        raise InterfaceError("protocol.actions must include NO_OP")
    doses = protocol["allowed_doses"]
    if not isinstance(doses, list) or not doses or not all(is_positive_number(x) for x in doses):
        raise InterfaceError("protocol.allowed_doses must contain positive numbers")
    if protocol["rounding"] != "half_up":
        raise InterfaceError("Only deterministic half_up tick rounding is supported")
    if not is_positive_number(protocol["observation_tail_dose"]):
        raise InterfaceError("protocol.observation_tail_dose must be positive")
    for pair in protocol["opposite_pairs"]:
        if not isinstance(pair, list) or len(pair) != 2 or any(x not in actions for x in pair):
            raise InterfaceError(f"Invalid opposite action pair: {pair!r}")
    if set(protocol["task_rules"]) != {"TASK1", "TASK2", "TASK3"}:
        raise InterfaceError("protocol.task_rules must define TASK1, TASK2, and TASK3")
    expected = {
        "per_step_reference_frame": SEQUENTIAL_REFERENCE_FRAME,
        "operator_application": "right_multiply_in_execution_order",
        "unverified_multi_step_policy": "STRUCTURAL_NA",
    }
    if protocol["sequential_contract"] != expected:
        raise InterfaceError("protocol.sequential_contract differs from the frozen contract")
    evidence_required = protocol.get("generation_extension_evidence_required")
    if evidence_required is not None and not isinstance(evidence_required, bool):
        raise InterfaceError("protocol generation-extension evidence flag must be boolean")
    return actions, doses


def _validate_checkpoint_contract(protocol: dict[str, Any], doses: list[Any]) -> None:
    """Validate the optional held-rollout checkpoint contract."""

    held_contract = protocol.get("task2_checkpoint_contract")
    if held_contract is None:
        return
    expected_held = {
        "explicit_checkpoint_doses_required": True,
        "video_dose": 16.0,
        "checkpoint_doses": [1.0, 2.0, 3.0, 5.0, 8.0, 16.0],
    }
    if held_contract != expected_held:
        raise InterfaceError("protocol Task2 checkpoint contract differs from held-D16 v3")
    if any(value not in doses for value in expected_held["checkpoint_doses"]):
        raise InterfaceError("protocol held-D16 checkpoints must be allowed doses")


def _validate_atomic_protocol_contract(protocol: dict[str, Any], doses: list[Any]) -> None:
    """Validate the frozen measurable-atom extension, when selected."""

    atomic_contract = protocol.get("atomic_unit_contract")
    measurable_unit_protocol = protocol["schema_version"] in {
        ATOMIC_PROTOCOL_VERSION, COMPOUND_PROTOCOL_VERSION,
    }
    if measurable_unit_protocol:
        expected_atomic_contract = {
            "schema_version": ATOMIC_UNIT_VERSION,
            "definition": (
                "smallest frozen integer-native-tick block selected before benchmark "
                "testing that remains evaluator-resolvable while preserving the "
                "visible-anchor budget across the main response curve"
            ),
            "raw_native_ticks_required": True,
            "fractional_formal_dose_forbidden": True,
            "same_unit_for_all_supported_actions_within_adapter": True,
            "left_right_pair_must_share_unit": True,
            "wasd_family_must_share_unit": True,
            "selection_split": "calibration_only_disjoint_from_formal_test_scenes",
            "test_time_retuning_forbidden": True,
            "report_fields": [
                "native_ticks", "native_tick_frames", "duration_seconds_realized",
                "legacy_base_control_ticks", "legacy_dose_equivalent",
            ],
        }
        if atomic_contract != expected_atomic_contract:
            raise InterfaceError("protocol atomic-unit contract differs from frozen v3")
        if doses != ATOMIC_ALLOWED_DOSES or any(float(dose) != int(dose) for dose in doses):
            raise InterfaceError("atomic v3 allows only frozen integer doses 1/2/3/5/8/16")
        if float(protocol["observation_tail_dose"]) != 1.0:
            raise InterfaceError("atomic v3 requires exactly one atomic observation tail")
    elif atomic_contract is not None:
        raise InterfaceError("atomic_unit_contract is reserved for atomic protocol v3")

    compound_contract = protocol.get("compound_unit_contract")
    if protocol["schema_version"] == COMPOUND_PROTOCOL_VERSION:
        expected_compound_contract = {
            "schema_version": "strideworld-compound-control-unit-v1",
            "minimum_actions_per_step": 2,
            "maximum_actions_per_step": 3,
            "one_action_maximum_per_motion_axis": True,
            "task2_exactly_one_simultaneous_step": True,
            "joint_response_is_its_own_measured_primitive": True,
            "atomic_matrix_product_as_joint_ground_truth": "forbidden",
            "atomic_order_controls": "diagnostic_ablation_only",
            "route_steps_require_frozen_compound_calibration": True,
            "alternating_atomic_emulation": "forbidden",
        }
        if compound_contract != expected_compound_contract:
            raise InterfaceError("protocol compound-unit contract differs from frozen v1")
    elif compound_contract is not None:
        raise InterfaceError("compound_unit_contract is reserved for compound protocol v1")


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Validate the canonical vocabulary and global scheduling rules."""

    require_keys(
        protocol,
        [
            "schema_version",
            "actions",
            "interface_tracks",
            "allowed_doses",
            "rounding",
            "observation_tail_dose",
            "opposite_pairs",
            "task_rules",
            "sequential_contract",
        ],
        "protocol",
    )
    _, doses = _validate_protocol_vocabulary(protocol)
    _validate_checkpoint_contract(protocol, doses)
    _validate_atomic_protocol_contract(protocol, doses)


def controllable_ticks(adapter: dict[str, Any]) -> int:
    """Return the number of native control ticks in the adapter's default clip."""

    timing = adapter["timing"]
    remaining_frames = timing["frames"] - timing["conditioning_frames"]
    return remaining_frames // timing["native_tick_frames"]


def validate_timing(adapter: dict[str, Any], label: str) -> None:
    """Validate frame/tick metadata used to construct exact windows."""

    timing = adapter["timing"]
    require_keys(
        timing,
        ["fps", "frames", "conditioning_frames", "native_tick_frames", "base_control_ticks"],
        f"{label}.timing",
    )
    if not is_positive_number(timing["fps"]):
        raise InterfaceError(f"{label}.timing.fps must be positive")
    for key in ("frames", "native_tick_frames", "base_control_ticks"):
        if not isinstance(timing[key], int) or timing[key] <= 0:
            raise InterfaceError(f"{label}.timing.{key} must be a positive integer")
    if not isinstance(timing["conditioning_frames"], int) or timing["conditioning_frames"] < 0:
        raise InterfaceError(f"{label}.timing.conditioning_frames must be a non-negative integer")
    if timing["conditioning_frames"] >= timing["frames"]:
        raise InterfaceError(f"{label} has no controllable output frames")
    if timing["base_control_ticks"] > controllable_ticks(adapter):
        raise InterfaceError(f"{label}.base_control_ticks exceeds its default controllable horizon")


def validate_measurable_atomic_unit(adapter: dict[str, Any], label: str) -> None:
    """Validate the v3 adapter's frozen, integer-native-tick atomic unit."""

    unit = adapter.get("measurable_atomic_unit")
    if not isinstance(unit, dict):
        raise InterfaceError(f"{label}.measurable_atomic_unit must be an object")
    require_keys(
        unit,
        [
            "schema_version", "native_ticks", "legacy_base_control_ticks",
            "legacy_dose_equivalent", "same_for_all_supported_actions",
            "selection_status", "selection_evidence",
        ],
        f"{label}.measurable_atomic_unit",
    )
    if unit["schema_version"] != ATOMIC_UNIT_VERSION:
        raise InterfaceError(f"{label}.measurable_atomic_unit has an unknown schema")
    native_ticks = unit["native_ticks"]
    legacy_ticks = unit["legacy_base_control_ticks"]
    if not isinstance(native_ticks, int) or isinstance(native_ticks, bool) or native_ticks <= 0:
        raise InterfaceError(f"{label}.measurable_atomic_unit.native_ticks must be a positive integer")
    if not isinstance(legacy_ticks, int) or isinstance(legacy_ticks, bool) or legacy_ticks <= 0:
        raise InterfaceError(
            f"{label}.measurable_atomic_unit.legacy_base_control_ticks must be a positive integer"
        )
    if native_ticks != adapter["timing"]["base_control_ticks"]:
        raise InterfaceError(f"{label} atomic native ticks differ from timing.base_control_ticks")
    legacy_equivalent = unit["legacy_dose_equivalent"]
    if not is_positive_number(legacy_equivalent) or not math.isclose(
        float(legacy_equivalent), native_ticks / legacy_ticks, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise InterfaceError(f"{label} has an inconsistent legacy_dose_equivalent")
    if unit["same_for_all_supported_actions"] is not True:
        raise InterfaceError(f"{label} must use one atomic unit for all supported actions")
    for field in ("selection_status", "selection_evidence"):
        if not isinstance(unit[field], str) or not unit[field].strip():
            raise InterfaceError(f"{label}.measurable_atomic_unit.{field} must be non-empty")


def validate_action_map(adapter: dict[str, Any], protocol: dict[str, Any], label: str) -> None:
    """Validate supported payloads and explicit N/A reasons."""

    expected = set(protocol["actions"])
    actual = set(adapter["actions"])
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise InterfaceError(f"{label}.actions mismatch; missing={missing}, extra={extra}")
    reasons = adapter["unsupported_reasons"]
    if not isinstance(reasons, dict):
        raise InterfaceError(f"{label}.unsupported_reasons must be an object")
    for action, payload in adapter["actions"].items():
        if payload is None and not reasons.get(action):
            raise InterfaceError(f"{label}.{action} is N/A but has no reason")
        if payload is not None and not isinstance(payload, dict):
            raise InterfaceError(f"{label}.{action} payload must be an object or null")


def validate_simultaneous_rule(adapter: dict[str, Any], label: str) -> None:
    """Validate how an adapter represents two actions active in one step."""

    rule = adapter["simultaneous"]
    require_keys(rule, ["supported", "merge_fields"], f"{label}.simultaneous")
    if not isinstance(rule["supported"], bool) or not isinstance(rule["merge_fields"], list):
        raise InterfaceError(f"{label}.simultaneous has invalid types")
    if rule["supported"] and not rule["merge_fields"]:
        raise InterfaceError(f"{label} supports simultaneous actions but defines no merge fields")
    if not rule["supported"] and not rule.get("reason"):
        raise InterfaceError(f"{label} must explain why simultaneous actions are unsupported")


def validate_sequential_capability(
    adapter: dict[str, Any], label: str, *, require_extension_source: bool = False
) -> None:
    """Validate frozen evidence for executing ordered multi-step controls."""

    capability = adapter["sequential"]
    required = [
        "supported", "verified_runner", "verification_source", "max_steps",
        "switch_granularity", "generation_extension_supported",
        "per_step_reference_frame", "initial_state_pose_policy",
    ]
    require_keys(capability, required, f"{label}.sequential")
    if capability["per_step_reference_frame"] != SEQUENTIAL_REFERENCE_FRAME:
        raise InterfaceError(f"{label}.sequential has unsupported per-step reference frame")
    if capability["switch_granularity"] != "native_tick_boundary":
        raise InterfaceError(f"{label}.sequential must switch at native tick boundaries")
    if capability["initial_state_pose_policy"] != "required_if_conditioning_frames_zero":
        raise InterfaceError(f"{label}.sequential has invalid initial-state pose policy")
    if not isinstance(capability["generation_extension_supported"], bool):
        raise InterfaceError(f"{label}.sequential generation extension flag must be boolean")
    if require_extension_source and capability["generation_extension_supported"] and not isinstance(
        capability.get("generation_extension_verification_source"), str
    ):
        raise InterfaceError(
            f"{label}.sequential generation extension lacks a verification source"
        )
    if require_extension_source and capability["generation_extension_supported"] and not capability[
        "generation_extension_verification_source"
    ]:
        raise InterfaceError(
            f"{label}.sequential generation extension verification source is empty"
        )
    if capability["supported"] is True:
        if not capability["verified_runner"] or not capability["verification_source"]:
            raise InterfaceError(f"{label}.sequential supported capability lacks runner evidence")
        if not isinstance(capability["max_steps"], int) or capability["max_steps"] < 2:
            raise InterfaceError(f"{label}.sequential max_steps must be at least two")
    elif capability["supported"] is False:
        if capability["verified_runner"] is not None or capability["max_steps"] != 0:
            raise InterfaceError(f"{label}.sequential unverified capability must be inert")
        if not capability.get("reason"):
            raise InterfaceError(f"{label}.sequential unverified capability needs a reason")
    else:
        raise InterfaceError(f"{label}.sequential supported flag must be boolean")


def validate_adapter(adapter: dict[str, Any], protocol: dict[str, Any], label: str) -> None:
    """Validate one frozen model adapter."""

    require_keys(
        adapter,
        [
            "adapter_id",
            "model_id",
            "interface_track",
            "action_reference_frame",
            "representation",
            "timing",
            "default_strength",
            "simultaneous",
            "sequential",
            "actions",
            "unsupported_reasons",
        ],
        label,
    )
    if adapter["interface_track"] not in protocol["interface_tracks"]:
        raise InterfaceError(f"{label} has unknown interface_track")
    if adapter["action_reference_frame"] != ACTION_REFERENCE_FRAME:
        raise InterfaceError(
            f"{label}.action_reference_frame must be {ACTION_REFERENCE_FRAME}"
        )
    if not adapter["adapter_id"] or not adapter["model_id"] or not adapter["representation"]:
        raise InterfaceError(f"{label} identifiers must be non-empty")
    validate_timing(adapter, label)
    if protocol["schema_version"] in {
        ATOMIC_PROTOCOL_VERSION, COMPOUND_PROTOCOL_VERSION,
    }:
        if not adapter["adapter_id"].endswith("_atomic_v3"):
            raise InterfaceError(f"{label}.adapter_id must end in _atomic_v3")
        validate_measurable_atomic_unit(adapter, label)
    validate_action_map(adapter, protocol, label)
    validate_simultaneous_rule(adapter, label)
    validate_sequential_capability(
        adapter,
        label,
        require_extension_source=bool(
            protocol.get("generation_extension_evidence_required", False)
        ),
    )


def opposite_sets(protocol: dict[str, Any]) -> set[frozenset[str]]:
    """Return unordered action pairs that cannot be active simultaneously."""

    return {frozenset(pair) for pair in protocol["opposite_pairs"]}


def validate_step(step: dict[str, Any], protocol: dict[str, Any], label: str) -> None:
    """Validate one sequential step; multiple actions in a step mean simultaneous."""

    require_keys(step, ["actions", "dose"], label)
    actions = step["actions"]
    if not isinstance(actions, list) or not actions or len(actions) != len(set(actions)):
        raise InterfaceError(f"{label}.actions must be a non-empty unique list")
    unknown = sorted(set(actions) - set(protocol["actions"]))
    if unknown:
        raise InterfaceError(f"{label} contains unknown actions: {unknown}")
    if "NO_OP" in actions and len(actions) != 1:
        raise InterfaceError(f"{label}: NO_OP cannot be combined with another action")
    selected = set(actions)
    if any(pair <= selected for pair in opposite_sets(protocol)):
        raise InterfaceError(f"{label} combines opposite actions")
    if step["dose"] not in protocol["allowed_doses"]:
        raise InterfaceError(f"{label}.dose must be one of {protocol['allowed_doses']}")


def primitive_count(case: dict[str, Any]) -> int:
    """Count canonical primitive occurrences in a case."""

    return sum(len(step["actions"]) for step in case["steps"])


def effective_checkpoint_doses(case: dict[str, Any]) -> list[float]:
    """Return the frozen Task 2 checkpoints, with legacy one-dose compatibility."""

    if case.get("task") != "TASK2":
        return []
    raw = case.get("checkpoint_doses")
    if raw is None:
        steps = case.get("steps")
        if not isinstance(steps, list) or len(steps) != 1:
            return []
        return [steps[0].get("dose")]
    return list(raw) if isinstance(raw, list) else []


def validate_task_shape(
    case: dict[str, Any], protocol: dict[str, Any], label: str
) -> None:
    """Prevent accidental use of a composition schedule as a calibration case."""

    task = case["task"]
    is_compound = protocol["schema_version"] == COMPOUND_PROTOCOL_VERSION
    if is_compound:
        if any(not 2 <= len(step["actions"]) <= 3 for step in case["steps"]):
            raise InterfaceError(
                f"{label}: every compound task step must contain 2-3 simultaneous actions"
            )
        if task == "TASK2" and len(case["steps"]) != 1:
            raise InterfaceError(
                f"{label}: compound TASK2 must contain exactly one simultaneous step"
            )
        if task == "TASK3" and len(case["steps"]) < 2:
            raise InterfaceError(
                f"{label}: compound TASK3 must contain at least two joint steps"
            )
    elif task == "TASK2":
        if len(case["steps"]) != 1 or primitive_count(case) != 1:
            raise InterfaceError(f"{label}: TASK2 must contain exactly one primitive")
    if not is_compound and task == "TASK3" and primitive_count(case) < 2:
        raise InterfaceError(f"{label}: TASK3 must contain at least two primitives")
    if task != "TASK2" and "checkpoint_doses" in case:
        raise InterfaceError(f"{label}: checkpoint_doses is only valid for TASK2")
    if task != "TASK2":
        return
    checkpoint_doses = effective_checkpoint_doses(case)
    if not checkpoint_doses or not all(
        is_positive_number(dose) for dose in checkpoint_doses
    ):
        raise InterfaceError(f"{label}: checkpoint_doses must contain positive numbers")
    if any(dose not in protocol["allowed_doses"] for dose in checkpoint_doses):
        raise InterfaceError(
            f"{label}: checkpoint_doses must be selected from protocol.allowed_doses"
        )
    if any(
        right <= left for left, right in zip(checkpoint_doses, checkpoint_doses[1:])
    ):
        raise InterfaceError(f"{label}: checkpoint_doses must be strictly increasing")
    step_dose = case["steps"][0]["dose"]
    if any(dose > step_dose for dose in checkpoint_doses):
        raise InterfaceError(f"{label}: checkpoint_doses cannot exceed the step dose")
    if checkpoint_doses[-1] != step_dose:
        raise InterfaceError(f"{label}: final checkpoint must equal the step dose")
    held_contract = protocol.get("task2_checkpoint_contract")
    if held_contract is not None:
        if not isinstance(held_contract, dict):
            raise InterfaceError("protocol.task2_checkpoint_contract must be an object")
        if held_contract.get("explicit_checkpoint_doses_required") is True \
                and "checkpoint_doses" not in case:
            raise InterfaceError(f"{label}: held Task2 requires explicit checkpoint_doses")
        expected_dose = held_contract.get("video_dose")
        expected_checkpoints = held_contract.get("checkpoint_doses")
        if step_dose != expected_dose:
            raise InterfaceError(
                f"{label}: held Task2 step dose must equal {expected_dose}"
            )
        if checkpoint_doses != expected_checkpoints:
            raise InterfaceError(
                f"{label}: held Task2 checkpoint_doses must equal {expected_checkpoints}"
            )


def require_sha256(value: Any, label: str) -> None:
    """Require one canonical lowercase SHA256 string."""

    if (
        not isinstance(value, str) or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise InterfaceError(f"{label} must be a lowercase SHA256")


def formal_case_identity(case: dict[str, Any], label: str) -> dict[str, Any] | None:
    """Return a complete formal identity, or mark a wholly old case diagnostic."""

    present = {field for field in FORMAL_CASE_IDENTITY_FIELDS if field in case}
    if not present:
        return None
    missing = sorted(set(FORMAL_CASE_IDENTITY_FIELDS) - present)
    if missing:
        raise InterfaceError(f"{label} has partial formal identity; missing={missing}")
    if not isinstance(case["state_id"], str) or not case["state_id"]:
        raise InterfaceError(f"{label}.state_id must be a non-empty string")
    if not isinstance(case["probe_case_id"], str) or not case["probe_case_id"]:
        raise InterfaceError(f"{label}.probe_case_id must be a non-empty string")
    require_sha256(case["first_frame_sha256"], f"{label}.first_frame_sha256")
    require_sha256(case["probe_plan_sha256"], f"{label}.probe_plan_sha256")
    return {field: case[field] for field in FORMAL_CASE_IDENTITY_FIELDS}


def compiled_case_identity(case: dict[str, Any]) -> dict[str, Any]:
    """Build fields copied unchanged into every compiled record kind."""

    identity = formal_case_identity(case, str(case.get("case_id", "case")))
    if identity is None:
        return {"case_identity_status": CASE_IDENTITY_UNBOUND}
    contract = {"case_id": case["case_id"], **identity}
    return {
        **identity, "case_identity_status": CASE_IDENTITY_BOUND,
        "case_identity_contract": contract,
        "case_identity_contract_sha256": canonical_json_sha256(contract),
    }


def benchmark_action_binding(case: dict[str, Any]) -> dict[str, str]:
    """Carry the benchmark route label separately from native primitives."""

    value = case.get("benchmark_action_label")
    return {"benchmark_action_label": str(value)} if value is not None else {}


def validate_case(case: dict[str, Any], protocol: dict[str, Any], label: str) -> None:
    """Validate one canonical benchmark case."""

    require_keys(case, ["case_id", "task", "seed", "steps"], label)
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise InterfaceError(f"{label}.case_id must be a non-empty string")
    benchmark_label = case.get("benchmark_action_label")
    if benchmark_label is not None and (
        not isinstance(benchmark_label, str) or not benchmark_label
    ):
        raise InterfaceError(f"{label}.benchmark_action_label must be non-empty")
    if case["task"] not in protocol["task_rules"]:
        raise InterfaceError(f"{label}.task must be TASK1, TASK2, or TASK3")
    if not isinstance(case["seed"], int) or isinstance(case["seed"], bool):
        raise InterfaceError(f"{label}.seed must be an integer")
    if not isinstance(case["steps"], list) or not case["steps"]:
        raise InterfaceError(f"{label}.steps must be a non-empty list")
    formal_case_identity(case, label)
    initial_pose = case.get("initial_state_pose_sha256")
    if initial_pose is not None and (
        not isinstance(initial_pose, str)
        or len(initial_pose) != 64
        or any(char not in "0123456789abcdef" for char in initial_pose)
    ):
        raise InterfaceError(f"{label}.initial_state_pose_sha256 must be lowercase SHA256")
    for index, step in enumerate(case["steps"]):
        if not isinstance(step, dict):
            raise InterfaceError(f"{label}.steps[{index}] must be an object")
        validate_step(step, protocol, f"{label}.steps[{index}]")
    validate_task_shape(case, protocol, label)


def validate_case_collection(cases: dict[str, Any], protocol: dict[str, Any]) -> None:
    """Validate a case file and enforce unique case identifiers."""

    require_keys(cases, ["schema_version", "cases"], "case collection")
    if not isinstance(cases["cases"], list) or not cases["cases"]:
        raise InterfaceError("case collection must contain at least one case")
    identifiers: list[str] = []
    for index, case in enumerate(cases["cases"]):
        if not isinstance(case, dict):
            raise InterfaceError(f"cases[{index}] must be an object")
        validate_case(case, protocol, f"cases[{index}]")
        identifiers.append(case["case_id"])
    if len(identifiers) != len(set(identifiers)):
        raise InterfaceError("case_id values must be unique")


def round_half_up(value: float) -> int:
    """Round a non-negative number to the nearest integer, with halves upward."""

    if value < 0:
        raise InterfaceError("Cannot quantize a negative tick count")
    return int(math.floor(value + 0.5))


def quantize_dose(base_ticks: int, dose: float) -> tuple[float, int, float]:
    """Convert a requested dimensionless dose into actual native ticks."""

    requested_ticks = base_ticks * dose
    realized_ticks = max(1, round_half_up(requested_ticks))
    realized_dose = realized_ticks / base_ticks
    return requested_ticks, realized_ticks, realized_dose


def add_vectors(left: list[Any], right: list[Any], label: str) -> list[float]:
    """Add equal-length numeric vectors used by verified multi-hot adapters."""

    if len(left) != len(right):
        raise InterfaceError(f"Cannot merge unequal vector lengths for {label}")
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in left + right):
        raise InterfaceError(f"Cannot merge non-numeric vectors for {label}")
    return [a + b for a, b in zip(left, right)]


def merge_native_payloads(adapter: dict[str, Any], actions: list[str]) -> dict[str, Any]:
    """Merge simultaneous actions using only adapter-declared numeric channels."""

    rule = adapter["simultaneous"]
    fields = rule["merge_fields"]
    payloads = [adapter["actions"][action] for action in actions]
    merged: dict[str, Any] = {}
    for field in fields:
        vectors = [payload.get(field) for payload in payloads]
        if any(not isinstance(vector, list) for vector in vectors):
            raise InterfaceError(f"{adapter['adapter_id']} cannot merge field {field}")
        combined = [0] * len(vectors[0])
        for vector in vectors:
            combined = add_vectors(combined, vector, field)
        merged[field] = combined
    return merged


def native_payload(adapter: dict[str, Any], actions: list[str]) -> dict[str, Any]:
    """Return the exact native payload for a single or simultaneous step."""

    if len(actions) == 1:
        return copy.deepcopy(adapter["actions"][actions[0]])
    return merge_native_payloads(adapter, actions)


def sequential_unsupported_reason(adapter: dict[str, Any], case: dict[str, Any]) -> str | None:
    """Return why an ordered multi-step request lacks verified runner support."""

    if len(case["steps"]) == 1:
        return None
    capability = adapter["sequential"]
    if not capability["supported"]:
        return capability["reason"]
    if len(case["steps"]) > capability["max_steps"]:
        return f"sequence has {len(case['steps'])} steps; verified maximum is {capability['max_steps']}"
    if adapter["timing"]["conditioning_frames"] == 0 and not case.get("initial_state_pose_sha256"):
        return "conditioning_frames=0 requires an explicit initial_state_pose_sha256"
    return None


def unsupported_reason(adapter: dict[str, Any], case: dict[str, Any]) -> str | None:
    """Return a deterministic N/A reason before attempting compilation."""

    for step in case["steps"]:
        for action in step["actions"]:
            if adapter["actions"][action] is None:
                return f"{action}: {adapter['unsupported_reasons'][action]}"
        if len(step["actions"]) > 1 and not adapter["simultaneous"]["supported"]:
            return adapter["simultaneous"]["reason"]
    return sequential_unsupported_reason(adapter, case)


def case_mode(case: dict[str, Any]) -> str:
    """Classify a schedule without changing its order."""

    if len(case["steps"]) == 1:
        return "simultaneous" if len(case["steps"][0]["actions"]) > 1 else "single"
    if any(len(step["actions"]) > 1 for step in case["steps"]):
        return "mixed"
    return "sequential"


def frame_at_tick(adapter: dict[str, Any], tick: int) -> int:
    """Map a native control boundary to an output-frame boundary."""

    timing = adapter["timing"]
    return timing["conditioning_frames"] + tick * timing["native_tick_frames"]


def measurement_checkpoints(
    adapter: dict[str, Any], case: dict[str, Any]
) -> list[dict[str, Any]]:
    """Freeze Task 2 endpoint frames before the one terminal observation tail."""

    if case.get("task") != "TASK2" or "checkpoint_doses" not in case:
        return []
    base_ticks = int(adapter["timing"]["base_control_ticks"])
    rows: list[dict[str, Any]] = []
    for dose in effective_checkpoint_doses(case):
        _, native_ticks, _ = quantize_dose(base_ticks, dose)
        rows.append({
            "dose_requested": dose,
            "native_tick": native_ticks,
            "frame_index": frame_at_tick(adapter, native_ticks) - 1,
        })
    return rows


def compiled_checkpoint_binding(
    adapter: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    """Return Task 2 checkpoint fields shared by supported and N/A records."""

    if case.get("task") != "TASK2" or "checkpoint_doses" not in case:
        return {}
    return {
        "measurement_contract_version": TASK2_MEASUREMENT_CONTRACT_VERSION,
        "checkpoint_doses": effective_checkpoint_doses(case),
        "measurement_checkpoints": measurement_checkpoints(adapter, case),
    }


def compiled_atomic_unit_binding(adapter: dict[str, Any]) -> dict[str, Any]:
    """Carry the exact measurable atom into every v3 compiled record."""

    unit = adapter.get("measurable_atomic_unit")
    if not isinstance(unit, dict):
        return {}
    return {
        "measurable_atomic_unit": copy.deepcopy(unit),
        "measurable_atomic_unit_sha256": canonical_json_sha256(unit),
    }


def operator_window(
    adapter: dict[str, Any], start_tick: int, end_tick: int,
) -> dict[str, Any]:
    """Map control ticks to inclusive pre/post action state-frame indices."""

    return {
        "start_frame": frame_at_tick(adapter, start_tick) - 1,
        "end_frame": frame_at_tick(adapter, end_tick) - 1,
        "end_is_inclusive": True,
        "definition": "frame_at_tick_minus_one_state_boundaries",
        "tail_included_in_unit_pose": False,
    }


def compile_step(
    adapter: dict[str, Any], step: dict[str, Any], step_index: int, start_tick: int
) -> dict[str, Any]:
    """Compile one sequential step and preserve requested versus realized dose."""

    base_ticks = adapter["timing"]["base_control_ticks"]
    requested, realized, realized_dose = quantize_dose(base_ticks, step["dose"])
    end_tick = start_tick + realized
    control_hz = adapter["timing"]["fps"] / adapter["timing"]["native_tick_frames"]
    payload = native_payload(adapter, step["actions"])
    relative_frames = realized * adapter["timing"]["native_tick_frames"]
    return {
        "step_index": step_index,
        "mode": "simultaneous" if len(step["actions"]) > 1 else "single",
        "canonical_actions": list(step["actions"]),
        "dose_requested": step["dose"],
        "native_ticks_requested_float": requested,
        "native_ticks": realized,
        "dose_realized": realized_dose,
        "start_tick": start_tick,
        "end_tick_exclusive": end_tick,
        "start_frame": frame_at_tick(adapter, start_tick),
        "end_frame_exclusive": frame_at_tick(adapter, end_tick),
        "operator_window": operator_window(adapter, start_tick, end_tick),
        "relative_start_frame": 0,
        "relative_end_frame_exclusive": relative_frames,
        "per_step_reference_frame": adapter["sequential"]["per_step_reference_frame"],
        "duration_seconds_realized": realized / control_hz,
        "native_payload": payload,
        "native_payload_sha256": canonical_json_sha256(payload),
    }


def repeat_payload(payload: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Expand one native payload over a fixed number of native control ticks."""

    return [copy.deepcopy(payload) for _ in range(count)]


def build_timeline(
    adapter: dict[str, Any], compiled_steps: list[dict[str, Any]], tail_ticks: int
) -> list[dict[str, Any]]:
    """Construct the complete native timeline, including the observation tail."""

    timeline: list[dict[str, Any]] = []
    for step in compiled_steps:
        timeline.extend(repeat_payload(step["native_payload"], step["native_ticks"]))
    timeline.extend(repeat_payload(adapter["actions"]["NO_OP"], tail_ticks))
    return timeline


def request_contract(case: dict[str, Any]) -> dict[str, Any]:
    """Freeze the canonical request even when compilation is structural N/A."""

    contract = {
        "case_id": case["case_id"], "task": case["task"], "seed": case["seed"],
        "steps": [{
            "step_index": index, "canonical_actions": list(step["actions"]),
            "dose_requested": step["dose"],
        } for index, step in enumerate(case["steps"])],
    }
    if case["task"] == "TASK2" and "checkpoint_doses" in case:
        contract["checkpoint_doses"] = effective_checkpoint_doses(case)
    if case.get("initial_state_pose_sha256"):
        contract["initial_state_pose_sha256"] = case["initial_state_pose_sha256"]
    identity = formal_case_identity(case, str(case.get("case_id", "case")))
    if identity is not None:
        contract.update(identity)
    contract.update(benchmark_action_binding(case))
    return contract


def na_record(adapter: dict[str, Any], case: dict[str, Any], reason: str) -> dict[str, Any]:
    """Create an explicit N/A record instead of silently changing the request."""

    request = request_contract(case)
    return {
        "case_id": case["case_id"],
        "task": case["task"],
        "seed": case["seed"],
        "adapter_id": adapter["adapter_id"],
        "adapter_fingerprint": canonical_json_sha256(adapter),
        "adapter_fingerprint_source": "canonical_adapter_json_sha256",
        "model_id": adapter["model_id"],
        "interface_track": adapter["interface_track"],
        "action_reference_frame": adapter["action_reference_frame"],
        "sequential_capability": copy.deepcopy(adapter["sequential"]),
        "sequential_capability_sha256": canonical_json_sha256(adapter["sequential"]),
        "request_contract": request,
        "request_contract_sha256": canonical_json_sha256(request),
        "status": "N/A",
        "na_kind": "STRUCTURAL_NA",
        "reason": reason,
        **compiled_atomic_unit_binding(adapter),
        **compiled_checkpoint_binding(adapter, case),
        **benchmark_action_binding(case),
        **compiled_case_identity(case),
    }


def compile_steps(adapter: dict[str, Any], case: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Compile all ordered steps and return the final command tick."""

    compiled_steps: list[dict[str, Any]] = []
    next_tick = 0
    for index, step in enumerate(case["steps"]):
        compiled = compile_step(adapter, step, index, next_tick)
        compiled_steps.append(compiled)
        next_tick = compiled["end_tick_exclusive"]
    return compiled_steps, next_tick


def observation_plan(
    adapter: dict[str, Any], protocol: dict[str, Any], command_end_tick: int
) -> tuple[int, float, int]:
    """Quantize the fixed response tail and compute the required frame count."""

    base_ticks = adapter["timing"]["base_control_ticks"]
    _, tail_ticks, tail_dose = quantize_dose(base_ticks, protocol["observation_tail_dose"])
    observation_end_tick = command_end_tick + tail_ticks
    required_frames = frame_at_tick(adapter, observation_end_tick)
    return tail_ticks, tail_dose, required_frames


def compiled_windows(
    adapter: dict[str, Any], command_end_tick: int, tail_ticks: int, tail_dose: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build separate command and observation windows using half-open indexes."""

    command = {
        "start_tick": 0,
        "end_tick_exclusive": command_end_tick,
        "start_frame": frame_at_tick(adapter, 0),
        "end_frame_exclusive": frame_at_tick(adapter, command_end_tick),
    }
    observation = {
        "start_frame": 0,
        "end_frame_exclusive": frame_at_tick(adapter, command_end_tick + tail_ticks),
        "tail_ticks": tail_ticks,
        "tail_dose_realized": tail_dose,
    }
    return command, observation


def schedule_contract(
    steps: list[dict[str, Any]], command: dict[str, Any],
    observation: dict[str, Any], required_frames: int,
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Freeze absolute sequence boundaries separately from atomic primitives."""

    boundaries = [{
        key: step[key] for key in (
            "step_index", "canonical_actions", "start_tick", "end_tick_exclusive",
            "start_frame", "end_frame_exclusive", "operator_window",
        )
    } for step in steps]
    contract = {
        "ordered_step_boundaries": boundaries,
        "command_window": copy.deepcopy(command),
        "observation_window": copy.deepcopy(observation),
        "evaluation_window": {"start_frame": 0, "end_frame_exclusive": required_frames},
        "terminal_tail_applied_once": True,
    }
    if checkpoints:
        contract["measurement_checkpoints"] = copy.deepcopy(checkpoints)
    return contract


def supported_record(
    adapter: dict[str, Any],
    case: dict[str, Any],
    compiled_steps: list[dict[str, Any]],
    command_end_tick: int,
    tail_ticks: int,
    tail_dose: float,
    required_frames: int,
) -> dict[str, Any]:
    """Assemble one supported record from already-compiled components."""

    timeline = build_timeline(adapter, compiled_steps, tail_ticks)
    command, observation = compiled_windows(adapter, command_end_tick, tail_ticks, tail_dose)
    checkpoints = measurement_checkpoints(adapter, case)
    schedule = schedule_contract(
        compiled_steps, command, observation, required_frames, checkpoints
    )
    base_ticks = adapter["timing"]["base_control_ticks"]
    request = request_contract(case)
    record = {
        "case_id": case["case_id"],
        "task": case["task"],
        "seed": case["seed"],
        "adapter_id": adapter["adapter_id"],
        "adapter_fingerprint": canonical_json_sha256(adapter),
        "adapter_fingerprint_source": "canonical_adapter_json_sha256",
        "model_id": adapter["model_id"],
        "interface_track": adapter["interface_track"],
        "action_reference_frame": adapter["action_reference_frame"],
        "sequential_capability": copy.deepcopy(adapter["sequential"]),
        "sequential_capability_sha256": canonical_json_sha256(adapter["sequential"]),
        "request_contract": request,
        "request_contract_sha256": canonical_json_sha256(request),
        "representation": adapter["representation"],
        "status": "SUPPORTED",
        "schedule_mode": case_mode(case),
        "base_control_ticks": base_ticks,
        "default_strength": copy.deepcopy(adapter["default_strength"]),
        "steps": compiled_steps,
        "command_window": command,
        "observation_window": observation,
        "schedule_contract": schedule,
        "schedule_contract_sha256": canonical_json_sha256(schedule),
        "default_output_frames": adapter["timing"]["frames"],
        "required_output_frames": required_frames,
        "generation_extension_required": required_frames > adapter["timing"]["frames"],
        "native_tick_frames": adapter["timing"]["native_tick_frames"],
        "fps": adapter["timing"]["fps"],
        "native_timeline": timeline,
        **compiled_atomic_unit_binding(adapter),
        **compiled_checkpoint_binding(adapter, case),
        **benchmark_action_binding(case),
        **compiled_case_identity(case),
    }
    if case.get("initial_state_pose_sha256"):
        record["initial_state_pose_sha256"] = case["initial_state_pose_sha256"]
    return record


def compile_case(adapter: dict[str, Any], case: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Compile one canonical case for one adapter."""

    reason = unsupported_reason(adapter, case)
    if reason:
        record = na_record(adapter, case, reason)
    else:
        steps, command_end_tick = compile_steps(adapter, case)
        tail_ticks, tail_dose, required_frames = observation_plan(
            adapter, protocol, command_end_tick,
        )
        capability = adapter["sequential"]
        if (
            required_frames > adapter["timing"]["frames"]
            and not capability["generation_extension_supported"]
        ):
            record = na_record(
                adapter, case,
                "required output exceeds the native horizon and no verified generation extension exists",
            )
        else:
            record = supported_record(
                adapter, case, steps, command_end_tick, tail_ticks, tail_dose,
                required_frames,
            )
    if protocol["schema_version"] == COMPOUND_PROTOCOL_VERSION:
        record.update({
            "compound_protocol_version": COMPOUND_PROTOCOL_VERSION,
            "compound_protocol_fingerprint": canonical_json_sha256(protocol),
        })
    return record


def compile_all(
    adapters: dict[str, dict[str, Any]], cases: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Compile every requested case for every selected adapter."""

    records = [
        compile_case(adapter, case, protocol)
        for adapter in adapters.values()
        for case in cases["cases"]
    ]
    return {
        "schema_version": "strideworld-compiled-actions-lite-v1",
        "protocol_version": protocol["schema_version"],
        "adapter_count": len(adapters),
        "case_count": len(cases["cases"]),
        "record_count": len(records),
        "records": records,
    }


def validate_all(
    protocol: dict[str, Any], adapters: dict[str, dict[str, Any]], cases: dict[str, Any]
) -> None:
    """Run all static checks before compilation or server inference."""

    validate_protocol(protocol)
    for name, adapter in adapters.items():
        validate_adapter(adapter, protocol, f"adapter[{name}]")
    validate_case_collection(cases, protocol)
