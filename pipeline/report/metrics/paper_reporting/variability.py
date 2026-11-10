"""Complete-plan variation of signed K4 action units; standard library only.

The two diagnostics are deliberately separate from direction accuracy. Raw
variance retains the source unit squared and is only a group-level diagnostic.
Paper tables use ``relative_sd_pct = 100 * sample_sd(u) / mean(abs(u))``.
This is a relative standard deviation, not a percentage variance.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ACTIONS = ("W", "S", "A", "D", "YL", "YR")
FAMILIES = {"translation": ACTIONS[:4], "rotation": ACTIONS[4:]}
METRICS = ("relative_sd_pct", "relative_variance")
SCHEMA_VERSION = "strideworld-unit-variability-v2"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_unit_report(path: str | Path) -> dict[str, Any]:
    """Load a unit export without silently overwriting duplicate cells.

    Nonstandard JSON NaN/Infinity values are retained as floats so the report
    builder can mark the affected planned cell unavailable with a reason.
    """
    with Path(path).open(encoding="utf-8") as stream:
        report = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(report, dict):
        raise ValueError("unit report must be a JSON object")
    return report


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _plan_cases(plan: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(plan, Mapping) or not plan:
        raise ValueError("plan must be a nonempty mapping of scene IDs to case ID lists")
    result = {}
    for scene, cases in plan.items():
        if not isinstance(scene, str) or not scene:
            raise ValueError("planned scene IDs must be nonempty strings")
        if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence) or not cases:
            raise ValueError(f"scene {scene!r} must have a nonempty case ID list")
        if any(not isinstance(case, str) or not case for case in cases):
            raise ValueError(f"scene {scene!r} has an invalid case ID")
        if len(set(cases)) != len(cases):
            raise ValueError(f"duplicate planned case in scene {scene!r}")
        result[scene] = tuple(cases)
    return result


def _seeds(expected_seeds: Sequence[int]) -> tuple[int, ...]:
    if isinstance(expected_seeds, (str, bytes)) or not isinstance(expected_seeds, Sequence):
        raise ValueError("expected_seeds must be a sequence of distinct nonnegative integers")
    if len(expected_seeds) < 2 or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in expected_seeds
    ):
        raise ValueError("expected_seeds must contain at least two nonnegative integers")
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("duplicate expected seed")
    return tuple(expected_seeds)


def _floors(min_scale: Mapping[str, float] | float) -> dict[str, float]:
    if isinstance(min_scale, Mapping):
        if set(min_scale) != set(FAMILIES):
            raise ValueError("min_scale must supply exactly 'translation' and 'rotation'")
        floors = dict(min_scale)
    else:
        floors = {family: min_scale for family in FAMILIES}
    if any(not _finite_number(value) or value < 0 for value in floors.values()):
        raise ValueError("min_scale values must be finite and nonnegative")
    return {family: float(value) for family, value in floors.items()}


def _index_units(report: Mapping[str, Any]) -> dict[tuple[str, str, int, str], Any]:
    """Index export cells, refusing seed aliases that could hide duplicates."""
    scenes = report.get("scenes", {})
    if not isinstance(scenes, Mapping):
        raise ValueError("unit report 'scenes' must be a mapping")
    index = {}
    for scene, scene_data in scenes.items():
        if not isinstance(scene_data, Mapping) or not isinstance(scene_data.get("cases", {}), Mapping):
            raise ValueError(f"malformed scene: {scene!r}")
        for case, case_data in scene_data.get("cases", {}).items():
            if not isinstance(case_data, Mapping) or not isinstance(case_data.get("seeds", {}), Mapping):
                raise ValueError(f"malformed case: {scene!r}/{case!r}")
            seen_seeds = set()
            for raw_seed, seed_data in case_data.get("seeds", {}).items():
                try:
                    seed = int(raw_seed)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(f"invalid source seed: {raw_seed!r}") from error
                if isinstance(raw_seed, bool) or seed < 0 or str(raw_seed) != str(seed):
                    raise ValueError(f"noncanonical source seed: {raw_seed!r}")
                if seed in seen_seeds:
                    raise ValueError(f"duplicate source seed: {scene}/{case}/{seed}")
                seen_seeds.add(seed)
                if not isinstance(seed_data, Mapping) or not isinstance(seed_data.get("actions", {}), Mapping):
                    raise ValueError(f"malformed seed: {scene}/{case}/{seed}")
                for action, action_data in seed_data.get("actions", {}).items():
                    key = (scene, case, seed, action)
                    if key in index:
                        raise ValueError(f"duplicate unit cell: {key!r}")
                    index[key] = action_data
    return index


def _cell(index: Mapping, scene: str, case: str, seed: int, action: str, horizon: int) -> dict[str, Any]:
    cell: dict[str, Any] = {"case_id": case, "seed": seed, "signed_unit": None, "reason": None}
    source = index.get((scene, case, seed, action))
    if source is None:
        cell["reason"] = "missing_planned_unit"
        return cell
    if not isinstance(source, Mapping) or not isinstance(source.get("unit_horizons"), Mapping):
        cell["reason"] = "malformed_unit_horizons"
        return cell
    unit = source["unit_horizons"].get(str(horizon))
    if not isinstance(unit, Mapping):
        cell["reason"] = "missing_calibration_horizon"
        return cell
    # The outer status also depends on K8/K16 and must not gate a legal K4.
    if unit.get("status") != "UNIT_IDENTIFIED":
        cell["reason"] = "calibration_not_identified"
        reasons = unit.get("failure_reasons", [])
        cell["source_failure_reasons"] = [str(reason) for reason in reasons] if isinstance(reasons, list) else [str(reasons)]
        return cell
    if unit.get("dose_horizon", horizon) != horizon:
        cell["reason"] = "calibration_horizon_mismatch"
        return cell
    value = unit.get("unit_one")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        cell["reason"] = "invalid_signed_unit"
        return cell
    if not _finite_number(value):
        cell["reason"] = "nonfinite_signed_unit"
        return cell
    # This is signed projection of xi_bar, not the first-dose observation or
    # its legacy non_degenerate flag. A reliably estimated zero is retained.
    cell["signed_unit"] = float(value)
    cell["source_unit"] = unit.get("unit")
    return cell


def _group(cells: list[dict[str, Any]], min_scale: float, **identity: Any) -> dict[str, Any]:
    values = [cell["signed_unit"] for cell in cells if cell["reason"] is None]
    reasons = sorted({cell["reason"] for cell in cells if cell["reason"] is not None})
    if len(cells) < 2:
        reasons.append("fewer_than_two_planned_members")
    if len(values) != len(cells):
        reasons.append("incomplete_planned_group")
    source_units = {cell.get("source_unit") for cell in cells if cell["reason"] is None and cell.get("source_unit") is not None}
    if len(source_units) > 1:
        reasons.append("inconsistent_source_units")
    result = {
        **identity,
        "status": "unavailable",
        "reasons": reasons,
        "n": len(values),
        "n_valid": len(values),
        "n_expected": len(cells),
        "statistics_scope": "complete_plan" if len(values) == len(cells) else "observed_subset_only",
        "source_unit": next(iter(source_units)) if len(source_units) == 1 else None,
        "mean_signed": None,
        "mean_abs": None,
        "denominator": None,
        "min_scale": min_scale,
        "sample_variance": None,
        "sample_sd": None,
        "relative_variance": None,
        "relative_sd_pct": None,
        "values": cells,
    }
    if values:
        result["mean_signed"] = statistics.mean(values)
        result["mean_abs"] = result["denominator"] = statistics.mean(abs(value) for value in values)
    if len(values) >= 2:
        try:
            result["sample_variance"] = statistics.variance(values)
            result["sample_sd"] = statistics.stdev(values)
            if not all(_finite_number(result[key]) for key in ("sample_variance", "sample_sd")):
                raise OverflowError("raw sample statistics are not finite")
        except (OverflowError, ValueError):
            result["sample_variance"] = result["sample_sd"] = None
            reasons.append("sample_statistics_overflow")
    if reasons:
        return result
    denominator = result["denominator"]
    if denominator is None or denominator <= min_scale:
        reasons.append("scale_unresolved")
        return result
    # Normalizing first avoids overflow/underflow when squaring the scale.
    normalized = [value / denominator for value in values]
    relative_variance = statistics.variance(normalized)
    relative_sd = 100.0 * statistics.stdev(normalized)
    if not math.isfinite(relative_variance) or not math.isfinite(relative_sd):
        reasons.append("relative_statistics_nonfinite")
        return result
    result.update(status="complete", relative_variance=relative_variance, relative_sd_pct=relative_sd)
    return result


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _summarize(groups: list[dict[str, Any]], plan: Mapping[str, Sequence[str]], kind: str) -> dict[str, Any]:
    by_action = {}
    for action in ACTIONS:
        selected = [group for group in groups if group["action"] == action]
        by_scene = {}
        for scene in plan:
            planned = [group for group in selected if group["scene_id"] == scene]
            complete = [group for group in planned if group["status"] == "complete"]
            by_scene[scene] = {
                **{metric: _mean([group[metric] for group in complete]) for metric in METRICS},
                "complete_groups": len(complete),
                "planned_groups": len(planned),
            }
        complete_scene_ids = [scene for scene, data in by_scene.items() if data["complete_groups"]]
        by_action[action] = {
            **{metric: _mean([by_scene[scene][metric] for scene in complete_scene_ids]) for metric in METRICS},
            "complete_groups": sum(group["status"] == "complete" for group in selected),
            "planned_groups": len(selected),
            "complete_scenes": len(complete_scene_ids),
            "planned_scenes": len(plan),
            "by_scene": by_scene,
        }
    summary = {}
    for family, actions in FAMILIES.items():
        missing_actions = [action for action in actions if by_action[action]["relative_sd_pct"] is None]
        selected = [group for group in groups if group["action"] in actions]
        summary[family] = {
            "status": "available" if not missing_actions else "unavailable",
            **{metric: None if missing_actions else statistics.mean(by_action[action][metric] for action in actions) for metric in METRICS},
            "complete_groups": sum(group["status"] == "complete" for group in selected),
            "planned_groups": len(selected),
            "missing_actions": missing_actions,
            "complete_scene_action_groups": sum(by_action[action]["complete_scenes"] for action in actions),
            "planned_scene_action_groups": len(plan) * len(actions),
            "failure_reason_counts": dict(sorted(Counter(reason for group in selected for reason in group["reasons"]).items())),
            "by_action": {action: by_action[action] for action in actions},
        }
    summary["aggregation_order"] = (
        "complete seed groups -> cases equally within scene -> available scenes equally -> fixed actions equally"
        if kind == "seed" else
        "complete case groups -> seeds equally within scene -> available scenes equally -> fixed actions equally"
    )
    return summary


def build_variability_report(
    unit_report: Mapping[str, Any],
    plan: Mapping[str, Sequence[str]],
    *,
    expected_seeds: Sequence[int] = (0, 1, 2),
    min_scale: Mapping[str, float] | float = 0.0,
    horizon: int = 4,
) -> dict[str, Any]:
    """Build seed and case variation tables from a frozen external plan.

    ``plan`` maps each planned scene to all its planned case IDs. Observed
    exports never determine or shrink that plan. ``expected_seeds`` defaults
    to the frozen three-seed experiment. All six fixed actions are required
    in family aggregates; an action with no complete group makes its family
    aggregate unavailable. Available scenes are equally weighted and their
    coverage remains explicit.

    ``min_scale`` is a nonnegative scalar or a mapping with ``translation``
    and ``rotation`` keys, in the source signed-unit scales. Zero (the default)
    rejects exact zero denominators only: it is numerical protection, not a
    measurement-resolution claim. Positive calibrated thresholds must be
    supplied by the caller and applied identically across models.
    """
    if not isinstance(unit_report, Mapping):
        raise ValueError("unit_report must be a mapping")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    planned = _plan_cases(plan)
    seeds = _seeds(expected_seeds)
    floors = _floors(min_scale)
    index = _index_units(unit_report)
    seed_groups, case_groups = [], []
    for scene, cases in planned.items():
        for action in ACTIONS:
            floor = floors["translation" if action in FAMILIES["translation"] else "rotation"]
            for case in cases:
                cells = [_cell(index, scene, case, seed, action, horizon) for seed in seeds]
                seed_groups.append(_group(cells, floor, scene_id=scene, case_id=case, action=action))
            for seed in seeds:
                cells = [_cell(index, scene, case, seed, action, horizon) for case in cases]
                case_groups.append(_group(cells, floor, scene_id=scene, seed=seed, action=action))
    source_definition = unit_report.get("definition", {})
    source_definition = source_definition if isinstance(source_definition, Mapping) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "model": unit_report.get("model"),
        "definition": {
            "horizon": horizon,
            "signed_unit_source": f"unit_horizons.{horizon}.unit_one",
            "sample_variance_ddof": 1,
            "relative_sd_pct": "100 * sample_sd(signed u, ddof=1) / mean(abs(u))",
            "relative_variance": "sample_variance(signed u, ddof=1) / mean(abs(u))**2",
            "min_scale": floors,
            "min_scale_rule": "mean(abs(u)) must be strictly greater than the family min_scale; no epsilon is added",
            "resolution_policy": "zero_thresholds_are_numerical_protection_only" if not any(floors.values()) else "caller_supplied_thresholds; calibration provenance is caller responsibility",
            "source_pose_support": {key: source_definition.get(key) for key in ("min_pose_inlier_ratio", "min_pose_anchor_coverage")},
            "complete_group_rule": "all planned signed units are identified and finite; n >= 2; resolvable denominator",
            "missing_policy": "NA; no zero imputation; no missing-action reweighting",
            "raw_statistics": "group details only; source unit for mean/sd and source unit squared for variance",
            "direction_policy": "signed units are preserved; direction correctness is a separate metric",
        },
        "plan": {"scenes": {scene: list(cases) for scene, cases in planned.items()}, "expected_seeds": list(seeds), "actions": list(ACTIONS)},
        "groups": {"seed": seed_groups, "case": case_groups},
        "summary": {"seed": _summarize(seed_groups, planned, "seed"), "case": _summarize(case_groups, planned, "case")},
    }
