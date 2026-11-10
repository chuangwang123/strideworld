"""Report Task 3 endpoint errors against matched same-case calibration scales.

This module does not run tracking, pose estimation, relation estimation or any
replacement gate. Translation and rotation stay separate. Every planned pair
has weight 1 / (number of cases * planned pairs in its case), including missing
and unmeasurable pairs. The source document is only an observed plan unless an
independent ``expected_pairs`` manifest is supplied.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "strideworld-paper-composition-v1"
TRANSLATION_ACTIONS = ("W", "S", "A", "D")
ROTATION_ACTIONS = ("YL", "YR")
ACTIONS = TRANSLATION_ACTIONS + ROTATION_ACTIONS
IDENTITY_FIELDS = ("model", "scene_id", "case_id", "seed", "pair_id")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_composition_document(path: str | Path) -> dict[str, Any]:
    """Load either input document, rejecting duplicate JSON object keys."""
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(document, dict):
        raise ValueError("report must be a JSON object")
    return document


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"invalid seed: {value!r}")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"invalid seed: {value!r}") from error
    if parsed < 0 or str(parsed) != str(value):
        raise ValueError(f"noncanonical seed: {value!r}")
    return parsed


def _identity(pair: Mapping[str, Any], model: str) -> tuple[str, str, str, int, str]:
    if not isinstance(pair, Mapping):
        raise ValueError("each pair must be a mapping")
    if pair.get("model", model) != model:
        raise ValueError(f"pair model mismatch: {pair.get('model')!r} != {model!r}")
    for field in ("scene_id", "case_id", "pair_id"):
        if not isinstance(pair.get(field), str) or not pair[field]:
            raise ValueError(f"pair {field} must be a nonempty string")
    return model, pair["scene_id"], pair["case_id"], _seed(pair.get("seed")), pair["pair_id"]


def _index_pairs(pairs: Sequence[Mapping[str, Any]], model: str, label: str) -> dict:
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise ValueError(f"{label} must be a pair list")
    index = {}
    for pair in pairs:
        key = _identity(pair, model)
        if key in index:
            raise ValueError(f"duplicate {label} pair: {key!r}")
        index[key] = pair
    return index


def _unit_index(document: Mapping[str, Any]) -> dict:
    result = {}
    scenes = document.get("scenes", {})
    if not isinstance(scenes, Mapping):
        raise ValueError("unit document scenes must be a mapping")
    for scene_id, scene in scenes.items():
        if not isinstance(scene, Mapping) or not isinstance(scene.get("cases", {}), Mapping):
            raise ValueError(f"malformed unit scene: {scene_id!r}")
        for case_id, case in scene.get("cases", {}).items():
            if not isinstance(case, Mapping) or not isinstance(case.get("seeds", {}), Mapping):
                raise ValueError(f"malformed unit case: {scene_id}/{case_id}")
            for raw_seed, seed in case.get("seeds", {}).items():
                key = (scene_id, case_id, _seed(raw_seed))
                if key in result:
                    raise ValueError(f"duplicate unit seed: {key!r}")
                if not isinstance(seed, Mapping) or not isinstance(seed.get("actions", {}), Mapping):
                    raise ValueError(f"malformed unit seed: {key!r}")
                result[key] = seed.get("actions", {})
    return result


def _operator(matrix: Any) -> list[list[float]] | None:
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 3:
        return None
    if any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in matrix):
        return None
    if any(not _number(value) for row in matrix for value in row):
        return None
    m = [[float(value) for value in row] for row in matrix]
    close = lambda a, b: math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-8)
    if not all(close(value, expected) for value, expected in zip(m[2], (0.0, 0.0, 1.0))):
        return None
    a, b, c, d = m[0][0], m[0][1], m[1][0], m[1][1]
    if not all((close(a*a+c*c, 1.0), close(b*b+d*d, 1.0), close(a*b+c*d, 0.0), close(a*d-b*c, 1.0))):
        return None
    return m


def _median_nonnegative(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    lower, upper = ordered[middle - 1:middle + 1]
    # Avoid overflow in (lower + upper) / 2 and underflow in lower/2 + upper/2.
    return lower + (upper - lower) / 2.0


def _scales(actions: Mapping[str, Any], translation_floor: float, rotation_floor: float,
            calibration_horizon: int) -> dict[str, Any]:
    values = {}
    reasons = []
    for action in ACTIONS:
        item = actions.get(action)
        horizons = item.get("unit_horizons", {}) if isinstance(item, Mapping) else {}
        unit = horizons.get(str(calibration_horizon)) if isinstance(horizons, Mapping) else None
        label = f"k{calibration_horizon}"
        if not isinstance(unit, Mapping):
            reasons.append(f"missing_{label}_reference:{action}")
            continue
        if unit.get("status") != "UNIT_IDENTIFIED":
            reasons.append(f"{label}_reference_unavailable:{action}")
            continue
        if unit.get("dose_horizon", calibration_horizon) != calibration_horizon:
            reasons.append(f"{label}_horizon_mismatch:{action}")
            continue
        matrix = _operator(unit.get("operator_se2"))
        if matrix is None:
            reasons.append(f"invalid_{label}_operator:{action}")
            continue
        # U_K is a per-dose operator fitted on K1..K, not a K-dose endpoint.
        # Its matrix displacement is intentionally distinct from xi_bar/unit_one.
        value = (math.hypot(matrix[0][2], matrix[1][2]) if action in TRANSLATION_ACTIONS
                 else abs(math.degrees(math.atan2(matrix[0][1], matrix[0][0]))))
        if not math.isfinite(value):
            reasons.append(f"nonfinite_{label}_scale:{action}")
            continue
        values[action] = value  # A legal zero is a member of the fixed median.
    translation = _median_nonnegative([values[a] for a in TRANSLATION_ACTIONS]) if all(a in values for a in TRANSLATION_ACTIONS) else None
    rotation = _median_nonnegative([values[a] for a in ROTATION_ACTIONS]) if all(a in values for a in ROTATION_ACTIONS) else None
    if translation is not None and translation <= translation_floor:
        reasons.append("translation_scale_unresolved")
    if rotation is not None and rotation <= rotation_floor:
        reasons.append("rotation_scale_unresolved")
    return {"translation_scale_cm": translation, "rotation_scale_deg": rotation,
            "reference_values": values, "failure_reasons": reasons}


def _endpoint(pair: Mapping[str, Any], planned: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Require the actual final boundary of both routes, never a valid prefix."""
    reasons = []
    routes = {}
    for member in ("alpha", "beta"):
        route = pair.get(member)
        if not isinstance(route, Mapping):
            reasons.append(f"missing_route:{member}")
            continue
        actions = route.get("actions")
        frames = route.get("frame_boundaries")
        if (not isinstance(actions, list) or not actions
                or any(action not in ACTIONS for action in actions)
                or not isinstance(frames, list) or len(frames) != len(actions) + 1
                or any(isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 for frame in frames)
                or any(b <= a for a, b in zip(frames, frames[1:]))):
            reasons.append(f"invalid_route_boundaries:{member}")
            continue
        expected_route = planned.get(member, {})
        if not isinstance(expected_route, Mapping):
            raise ValueError(f"expected {member} route must be a mapping")
        for field in ("actions", "frame_boundaries"):
            if field in expected_route and expected_route[field] != route[field]:
                reasons.append(f"planned_route_mismatch:{member}:{field}")
        routes[member] = route
    if len(routes) != 2:
        return None, reasons
    final_step = len(routes["alpha"]["actions"])
    if len(routes["beta"]["actions"]) != final_step:
        reasons.append("route_final_step_mismatch")
    if "expected_final_step" in planned and planned["expected_final_step"] != final_step:
        reasons.append("planned_final_step_mismatch")
    errors = pair.get("relation_errors_by_step")
    if not isinstance(errors, list) or len(errors) != final_step:
        reasons.append("missing_complete_relation_endpoint")
        return None, reasons
    for step, error in enumerate(errors, 1):
        if not isinstance(error, Mapping) or type(error.get("step")) is not int or error["step"] != step:
            reasons.append("invalid_relation_step_sequence")
            return None, reasons
        for member in ("alpha", "beta"):
            action_field = f"{member}_action"
            if action_field in error and (step > len(routes[member]["actions"]) or error[action_field] != routes[member]["actions"][step - 1]):
                reasons.append(f"relation_action_mismatch:{member}")
    final = errors[-1]
    for field in ("translation_error_cm", "angular_error_deg"):
        if not _number(final.get(field)) or final[field] < 0:
            reasons.append(f"invalid_endpoint_error:{field}")
    if reasons:
        return None, sorted(set(reasons))
    return {"final_step": final_step,
            "alpha_final_frame": routes["alpha"]["frame_boundaries"][-1],
            "beta_final_frame": routes["beta"]["frame_boundaries"][-1],
            "translation_error_cm": float(final["translation_error_cm"]),
            "angular_error_deg": float(final["angular_error_deg"])}, []


def _weighted_quantile(rows: list[dict[str, Any]], field: str, q: float) -> float | None:
    """Inverse empirical CDF, with the same planned-case weights as the scores."""
    values = sorted((row[field], row["weight"]) for row in rows if row["measurable"])
    if not values:
        return None
    target = q * math.fsum(weight for _, weight in values)
    cumulative = 0.0
    for value, weight in values:
        cumulative += weight
        if cumulative >= target:
            return value
    return values[-1][0]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coverage = math.fsum(row["weight"] for row in rows if row["measurable"])
    result = {
        "n_expected_pairs": len(rows),
        "n_measurable_pairs": sum(row["measurable"] for row in rows),
        "n_cases": len({(row["scene_id"], row["case_id"]) for row in rows}),
        "coverage": coverage,
        "support_pct": 100.0 * coverage,
        "failure_counts": dict(sorted(Counter(reason for row in rows for reason in row["failure_reasons"]).items())),
        "source_failure_counts": dict(sorted(Counter(reason for row in rows for reason in row["source_failure_reasons"]).items())),
    }
    for family in ("translation", "rotation"):
        score = math.fsum(row["weight"] * row[f"{family}_score"] for row in rows)
        result[f"{family}_score"] = score
        result[f"conditional_{family}_score"] = score / coverage if coverage > 0 else None
        for percentile in (50, 90):
            result[f"{family}_r_p{percentile}"] = _weighted_quantile(rows, f"{family}_r", percentile / 100.0)
    return result


def build_composition_report(
    task3_document: Mapping[str, Any],
    unit_document: Mapping[str, Any],
    *,
    expected_pairs: Sequence[Mapping[str, Any]] | None = None,
    calibration_horizon: int = 4,
    min_translation_scale_cm: float = 0.0,
    min_rotation_scale_deg: float = 0.0,
) -> dict[str, Any]:
    """Create rows and macro scores from existing Task 3 and unit exports.

    ``calibration_horizon`` is explicitly 4 or 8. The Task 3 export must declare
    that same integer in ``definition.unit_horizon``. A K8 report requires the
    original Task 3 core to recompute route predictions and raw relation errors
    using K8 operators. Changing only this report's denominator is rejected;
    it is not a valid K4/K8 comparison. This reporter never recomputes routes.

    ``expected_pairs`` freezes the planned identities (model may be omitted).
    Optional ``alpha``/``beta`` actions and frame_boundaries, or
    ``expected_final_step``, additionally verify the frozen endpoint. Duplicate
    identities and unplanned source pairs raise ValueError. Missing planned
    pairs remain in the denominator with zero scores. Without a separate plan,
    source-document rows define the plan; this cannot detect absent rows or
    prove that the document represents a complete external experiment.

    Floors are inclusive failure thresholds in cm/degree and must be frozen
    across models by the caller. The 0 defaults reject only unresolved zero
    scales; they do not claim to encode a measurement-resolution budget.
    """
    if not isinstance(task3_document, Mapping) or not isinstance(unit_document, Mapping):
        raise ValueError("both documents must be mappings")
    if type(calibration_horizon) is not int or calibration_horizon not in (4, 8):
        raise ValueError("calibration_horizon must be the integer 4 or 8")
    source_definition = task3_document.get("definition")
    source_horizon = source_definition.get("unit_horizon") if isinstance(source_definition, Mapping) else None
    if type(source_horizon) is not int or source_horizon != calibration_horizon:
        raise ValueError(
            "Task 3 definition.unit_horizon must match calibration_horizon; "
            "recompute Task 3 predictions and raw relation errors with the requested "
            "operators, rather than changing only the denominator"
        )
    model = task3_document.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Task 3 model must be a nonempty string")
    if unit_document.get("model") != model:
        raise ValueError("Task 3 and unit document model mismatch")
    for floor in (min_translation_scale_cm, min_rotation_scale_deg):
        if not _number(floor) or floor < 0:
            raise ValueError("scale floors must be finite and nonnegative")
    source = _index_pairs(task3_document.get("pairs", []), model, "source")
    planned = source if expected_pairs is None else _index_pairs(expected_pairs, model, "expected")
    if not planned:
        raise ValueError("composition plan must contain at least one pair")
    extra = source.keys() - planned.keys()
    if extra:
        raise ValueError(f"unplanned source pairs: {sorted(extra)!r}")
    units = _unit_index(unit_document)
    counts = Counter((key[1], key[2]) for key in planned)
    scale_cache = {}
    rows = []
    manifest = []
    for key, expected in planned.items():
        identity = dict(zip(IDENTITY_FIELDS, key))
        entry = dict(identity)
        for field in ("expected_final_step", "alpha", "beta"):
            if field in expected:
                value = expected[field]
                entry[field] = ({name: value[name] for name in ("actions", "frame_boundaries") if name in value}
                                if field in ("alpha", "beta") and isinstance(value, Mapping) else value)
        manifest.append(entry)
        row = {**identity, "weight": 1.0 / (len(counts) * counts[(key[1], key[2])]),
               "measurable": False, "quality_pass": False,
               "translation_r": None, "rotation_r": None,
               "translation_score": 0.0, "rotation_score": 0.0,
               "translation_scale_cm": None, "rotation_scale_deg": None,
               "reference_values": {}, "endpoint": None,
               "failure_reasons": [], "source_failure_reasons": []}
        pair = source.get(key)
        if pair is None:
            row["failure_reasons"] = ["missing_expected_pair"]
            rows.append(row)
            continue
        row["relation_family"] = pair.get("relation_family")
        row["route_family"] = pair.get("route_family")
        row["quality_pass"] = pair.get("quality_pass") is True
        reasons = []
        if not row["quality_pass"]:
            reasons.append("task3_quality_gate_failed")
        for member in ("alpha", "beta"):
            if not isinstance(pair.get(member), Mapping) or pair[member].get("quality_pass") is not True:
                reasons.append(f"task3_quality_gate_failed:{member}")
        source_reasons = pair.get("failure_reasons", [])
        row["source_failure_reasons"] = sorted(set(str(reason) for reason in source_reasons)) if isinstance(source_reasons, list) else [str(source_reasons)]
        unit_key = key[1:4]
        if unit_key not in scale_cache:
            scale_cache[unit_key] = _scales(units.get(unit_key, {}), min_translation_scale_cm,
                                           min_rotation_scale_deg, calibration_horizon)
        scale = scale_cache[unit_key]
        for field in ("translation_scale_cm", "rotation_scale_deg", "reference_values"):
            row[field] = scale[field]
        reasons.extend(scale["failure_reasons"])
        endpoint, endpoint_reasons = _endpoint(pair, expected)
        row["endpoint"] = endpoint
        reasons.extend(endpoint_reasons)
        if not reasons:
            translation_r = endpoint["translation_error_cm"] / scale["translation_scale_cm"]
            rotation_r = endpoint["angular_error_deg"] / scale["rotation_scale_deg"]
            if not math.isfinite(translation_r) or not math.isfinite(rotation_r):
                reasons.append("nonfinite_relative_error")
            else:
                row.update(measurable=True, translation_r=translation_r, rotation_r=rotation_r,
                           translation_score=100.0 / (1.0 + translation_r),
                           rotation_score=100.0 / (1.0 + rotation_r))
        row["failure_reasons"] = sorted(set(reasons))
        rows.append(row)
    manifest_json = json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return {
        "schema_version": SCHEMA_VERSION, "model": model,
        "plan": {"source": "source_document_pairs" if expected_pairs is None else "provided_expected_pairs",
                 "external_plan_validated": expected_pairs is not None,
                 "completeness_scope": "source_document_only" if expected_pairs is None else "provided_manifest_only",
                 "n_expected_pairs": len(planned), "n_source_pairs": len(source),
                 "manifest_sha256": hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()},
        "definition": {
            "calibration_horizon": calibration_horizon,
            "source_unit_horizon": source_horizon,
            "horizon_contract": "source Task 3 predictions/raw relation errors must already use the requested calibration horizon; changing only the denominator is invalid",
            "endpoint": "final relation_errors_by_step entry at both complete route boundaries",
            "gate": "original pair and alpha/beta quality_pass; motion gate retained",
            "unit_binding": f"same model/scene_id/case_id/seed; K{calibration_horizon} operator_se2",
            "translation_scale": "median(norm(U_K[:2, 2])) across fixed W,S,A,D",
            "rotation_scale": "median(abs(degrees(atan2(U_K[0,1], U_K[0,0])))) across fixed YL,YR",
            "min_translation_scale_cm": float(min_translation_scale_cm),
            "min_rotation_scale_deg": float(min_rotation_scale_deg),
            "scale_policy": "all six references required; legal zero references retained; each median must exceed floor",
            "relative_error": "raw endpoint error / corresponding scale; no route-length divisor",
            "score": "100/(1+r); no clipping; unavailable pair scores are 0 in both columns",
            "aggregation": "equal planned pairs within scene/case, then equal cases; seeds remain planned pairs",
            "conditional_score": "same planned-case weights renormalized over shared measurable pairs",
            "decomposition": "score = (support_pct/100) * conditional_score when support is positive",
            "percentiles": "unclipped r; inverse weighted empirical CDF on measurable pairs, using planned-case weights",
        },
        "rows": rows, "summary": _summary(rows),
    }
