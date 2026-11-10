"""StrideWorld Task 3 route consistency metrics.

The implementation follows the final Task 3 specification.  A route is
measured at its initial frame and at every segment boundary; its prediction is
the ordered product of the Task 2 atomic operators for the same model/state/
seed.  The module is intentionally independent of the legacy Task 3 manifest
evaluator: the ready index is the fixed input contract for this dataset.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from metrics.task2.task2_metrics import (
    ACTIONS,
    ROTATION_ACTIONS,
    TRANSLATION_ACTIONS,
    evaluate_cell,
    pose_metadata,
    read_json,
    record_path,
)
from metrics.tools.constants import FROZEN_METRIC_SETTINGS, FROZEN_METRIC_SETTINGS_SHA256
from metrics.tools.pose import is_rigid_transform
from metrics.tools.se2 import (
    distance_se2,
    inverse_se2,
    project_se3_to_se2,
)


SCHEMA_VERSION = "strideworld-task3-final-v2"
TRANSLATION_FLOOR_CM = float(FROZEN_METRIC_SETTINGS["translation_floor_cm"])
ROTATION_FLOOR_DEG = float(FROZEN_METRIC_SETTINGS["rotation_floor_deg"])
NORMALIZED_PASS_TAU = 0.5
DEFAULT_MIN_POSE_INLIER_RATIO = float(FROZEN_METRIC_SETTINGS["min_pose_inlier_ratio"])
DEFAULT_MIN_POSE_ANCHOR_COVERAGE = float(FROZEN_METRIC_SETTINGS["min_pose_anchor_coverage"])


def scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def ready_records(
    data_root: str | Path, *, model: str | None = None, seed: int | None = None,
) -> list[dict[str, Any]]:
    """Read ready Task 3 rows from the immutable index."""
    index = Path(data_root) / "common" / "index" / "by_host" / "eval.ready.jsonl"
    rows: list[dict[str, Any]] = []
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("primary_key", {})
        if key.get("task") != "task3":
            continue
        if model is not None and key.get("model") != model:
            continue
        if seed is not None and int(key.get("seed")) != int(seed):
            continue
        rows.append(row)
    rows.sort(key=lambda r: tuple(str(r["primary_key"].get(k)) for k in
                                  ("model", "scene_id", "case_id", "seed", "route_id")))
    if not rows:
        raise ValueError("ready index filter selected no Task 3 records")
    return rows


def route_identity(row: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    key = row["primary_key"]
    return (str(key.get("model")), str(key.get("scene_id")), str(key.get("case_id")),
            int(key.get("seed")), str(key.get("route_id")))


def pair_identity(row: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    key = row["primary_key"]
    route = row.get("route", {})
    record = row.get("_record", {})
    pair = row.get("pair_id") or route.get("pair_id") or record.get("route", {}).get("pair_id")
    if pair is None:
        pair = str(key.get("route_id", "")).rsplit("_", 1)[0]
    return (str(key.get("model")), str(key.get("scene_id")), str(key.get("case_id")),
            int(key.get("seed")), str(pair))


def pair_rows(rows: Iterable[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Pair Task 3 A/B rows without silently dropping denominator entries."""
    groups: dict[tuple[str, str, str, int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = row["primary_key"]
        route = row.get("route", {})
        route_id = str(key.get("route_id"))
        member = row.get("member") or route.get("member") or ("A" if route_id.endswith("_A") else "B")
        pair = row.get("pair_id") or route.get("pair_id") or route_id.rsplit("_", 1)[0]
        if member not in {"A", "B"}:
            raise ValueError(f"invalid Task 3 pair member: {member!r}")
        group_key = (
            str(key.get("model")), str(key.get("scene_id")), str(key.get("case_id")),
            int(key.get("seed")), str(pair),
        )
        if str(member) in groups[group_key]:
            raise ValueError(f"duplicate Task 3 pair member: {group_key}|{member}")
        groups[group_key][str(member)] = row
    incomplete = [key for key, group in groups.items() if set(group) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"incomplete Task 3 A/B pairs: {incomplete[:5]}")
    pairs = [(group["A"], group["B"]) for _, group in sorted(groups.items())]
    return sorted(pairs, key=lambda pair: (route_identity(pair[0]), route_identity(pair[1])))


def classify_relation(route_a: Mapping[str, Any], route_b: Mapping[str, Any]) -> str:
    """Classify current pairs: G1 is inverse, G2--G4 are order pairs."""
    a = [str(x) for x in route_a.get("expanded_unit_actions", [])]
    b = [str(x) for x in route_b.get("expanded_unit_actions", [])]
    return "order" if Counter(a) == Counter(b) else "inverse"


def index_route_pose_files(pose_root: str | Path) -> dict[tuple[str, str, str, int, str], Path]:
    """Index Task 3 pose artifacts by exact route identity."""
    output: dict[tuple[str, str, str, int, str], Path] = {}
    for path in sorted(Path(pose_root).rglob("pose.npz")):
        metadata = pose_metadata(path)
        fields = (metadata.get("model"), metadata.get("scene_id"), metadata.get("case_id"),
                  metadata.get("seed"), metadata.get("route_id"))
        if any(value is None for value in fields):
            continue
        key = (str(fields[0]), str(fields[1]), str(fields[2]), int(fields[3]), str(fields[4]))
        output.setdefault(key, path)
    return output


def _record(row: Mapping[str, Any], data_root: str | Path) -> dict[str, Any]:
    cached = row.get("_record")
    return cached if isinstance(cached, dict) else read_json(record_path(data_root, row))


def _route(row: Mapping[str, Any], data_root: str | Path) -> dict[str, Any]:
    return _record(row, data_root).get("route", {})


def _measurable_mask(payload: Mapping[str, np.ndarray], n: int, *, ratio: float,
                    coverage: float) -> np.ndarray:
    matrices = np.asarray(payload["camera_to_world_cv_rh"])
    stored = np.asarray(payload.get("measurable", np.zeros(n)), dtype=bool)
    camera = np.asarray(payload.get("camera_valid", []), dtype=bool)
    cov = np.asarray(payload.get("pose_anchor_coverage", []), dtype=float)
    inlier = np.asarray(payload.get("pose_inlier_ratio", []), dtype=float)
    if len(camera) == n and len(cov) == n and len(inlier) == n:
        return camera & (cov >= coverage) & (inlier >= ratio)
    return stored if len(stored) == n else np.zeros(n, dtype=bool)


def _segment_boundaries(record: Mapping[str, Any]) -> list[int]:
    timing = record.get("timing", {})
    unit_frames = timing.get("unit_boundary_frames")
    expanded = record.get("route", {}).get("expanded_unit_actions", [])
    if isinstance(unit_frames, list) and len(unit_frames) == len(expanded) + 1:
        return [int(x) for x in unit_frames[1:]]
    values = timing.get("segment_boundary_frames")
    if values is not None:
        return [int(x) for x in values]
    units = record.get("route", {}).get("segment_boundary_units", [])
    frames = timing.get("unit_boundary_frames", [])
    return [int(frames[int(x)]) for x in units]


def _operator_start(record: Mapping[str, Any]) -> int:
    values = record.get("timing", {}).get("unit_boundary_frames", [])
    return int(values[0]) if isinstance(values, list) and values else 0


def _segment_actions(route: Mapping[str, Any]) -> list[str]:
    expanded = route.get("expanded_unit_actions")
    if isinstance(expanded, list) and expanded:
        return [str(action).upper() for action in expanded]
    return [str(segment["action"]).upper() for segment in route.get("segments", [])]


def _empty_side(row: Mapping[str, Any], status: str, reasons: list[str]) -> dict[str, Any]:
    return {"route_id": str(row["primary_key"].get("route_id")), "status": status,
            "failure_reasons": reasons, "boundaries": [], "final": None,
            "actions": [], "observed_se2": None, "nondegenerate": False,
            "boundary_quality": []}


def _load_route_pose(pose_path: str | Path, *, ratio: float, coverage: float) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(pose_path, allow_pickle=False) as payload:
        matrices = np.asarray(payload["camera_to_world_cv_rh"], dtype=float)
        n = len(matrices)
        quality = {name: np.asarray(payload.get(name, np.full(n, np.nan))) for name in
                   ("camera_valid", "pose_anchor_coverage", "pose_inlier_ratio")}
        return matrices, _measurable_mask(payload, n, ratio=ratio, coverage=coverage), quality


def _observed_segments(matrices: np.ndarray, actions: list[str], boundaries: list[int], operator_start: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    start_inv = np.linalg.inv(matrices[operator_start]); observed=[]; prefix=[]; prev=np.eye(3)
    for index, (action, frame) in enumerate(zip(actions, boundaries), 1):
        planar, leakage = project_se3_to_se2(start_inv @ matrices[frame]); distances = distance_se2(prev, planar)
        observed.append({"segment": index, "action": action, "frame_index": int(frame), "se2": planar.tolist(),
                         "translation_cm": float(np.linalg.norm(planar[:2, 2])),
                         "yaw_deg": float(np.degrees(np.arctan2(planar[0, 1], planar[0, 0]))),
                         "delta_translation_cm": distances["translation_cm"], "delta_yaw_deg": distances["yaw_deg"], "leakage": leakage})
        prefix.append({"segment": index, "action": action, "frame_index": int(frame), "observed_se2": planar.tolist()}); prev=planar
    return observed, prefix


def _nondeg_segments(observed: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    result=[]; reasons=[]
    for item in observed:
        floor = ROTATION_FLOOR_DEG if item["action"] in ROTATION_ACTIONS else TRANSLATION_FLOOR_CM
        value = item["delta_yaw_deg"] if item["action"] in ROTATION_ACTIONS else item["delta_translation_cm"]; ok=bool(value >= floor)
        result.append({"segment": item["segment"], "action": item["action"], "d_t_cm": float(item["delta_translation_cm"]), "d_r_deg": float(item["delta_yaw_deg"]), "value": float(value), "floor": floor, "pass": ok})
        if not ok: reasons.append(f"static_shortcut_segment:{item['segment']}")
    return result, reasons


def evaluate_route_side(row: Mapping[str, Any], pose_path: str | Path | None, data_root: str | Path, *, min_pose_inlier_ratio: float = DEFAULT_MIN_POSE_INLIER_RATIO, min_pose_anchor_coverage: float = DEFAULT_MIN_POSE_ANCHOR_COVERAGE) -> dict[str, Any]:
    """Measure one route using the specification's hard M_t boundary gate."""
    route = _route(row, data_root); actions = _segment_actions(route)
    if pose_path is None: return _empty_side(row, "GEOMETRY_UNMEASURABLE", ["missing_pose_artifact"])
    try: matrices, measurable, quality = _load_route_pose(pose_path, ratio=min_pose_inlier_ratio, coverage=min_pose_anchor_coverage)
    except Exception as exc: return _empty_side(row, "GEOMETRY_UNMEASURABLE", [f"pose_load_error:{type(exc).__name__}"])
    record = _record(row, data_root)
    boundaries = _segment_boundaries(record); operator_start = _operator_start(record); reasons=[]
    hard_reasons=[]
    if not boundaries: hard_reasons.append("missing_segment_boundaries")
    if len(boundaries) != len(actions): hard_reasons.append("segment_boundary_action_count_mismatch")
    if any(action not in ACTIONS for action in actions): hard_reasons.append("unsupported_segment_action")
    if operator_start < 0 or operator_start >= len(matrices): hard_reasons.append("operator_start_out_of_range")
    if boundaries and (min(boundaries) < 0 or max(boundaries) >= len(matrices)): hard_reasons.append("segment_boundary_out_of_range")
    boundary_quality=[]
    for frame in [operator_start] + boundaries:
        in_range = 0 <= frame < len(matrices); finite = bool(in_range and np.isfinite(matrices[frame]).all())
        valid_transform = bool(in_range and is_rigid_transform(matrices[frame]))
        item={"frame_index":frame,"measurable":bool(in_range and measurable[frame]),"finite":finite,"valid_transform":valid_transform}
        for name, values in quality.items():
            value = scalar(values[frame]) if in_range else None
            if isinstance(value, (float, int, np.number)) and not np.isfinite(value): value = None
            item[name] = value
        boundary_quality.append(item)
        if not in_range or not valid_transform: hard_reasons.append(f"invalid_boundary_frame:{frame}")
        elif not item["measurable"]: hard_reasons.append(f"unmeasurable_boundary_frame:{frame}")
    reasons.extend(hard_reasons)
    if reasons and hard_reasons:
        result=_empty_side(row, "GEOMETRY_UNMEASURABLE", reasons); result.update({"actions":actions,"operator_start":operator_start,"frame_boundaries":boundaries,"boundary_quality":boundary_quality}); return result
    observed, prefix = _observed_segments(matrices, actions, boundaries, operator_start); nondeg, gate_reasons = _nondeg_segments(observed); reasons.extend(gate_reasons); final=observed[-1]["se2"] if observed else None
    return {"route_id":str(row["primary_key"].get("route_id")),"status":"MEASURED","failure_reasons":reasons,"actions":actions,"operator_start":operator_start,"frame_boundaries":boundaries,"boundary_quality":boundary_quality,"boundaries":observed,"prefix_observed":prefix,"final":final,"observed_se2":final,"nondegenerate":not gate_reasons,"nondegeneracy":nondeg,"pose_path":str(pose_path)}

def compose_units(actions: Iterable[str], operators: Mapping[str, np.ndarray]) -> np.ndarray:
    """Compose atomic operators in execution order (right multiplication)."""
    result = np.eye(3, dtype=float)
    for action in actions:
        result = result @ np.asarray(operators[action], dtype=float)
    return result


def _metric(observed: np.ndarray | None, predicted: np.ndarray | None) -> dict[str, Any]:
    if observed is None or predicted is None:
        return {"translation_cm": None, "yaw_deg": None, "d_n": None, "score": None,
                "pass": None}
    value = distance_se2(observed, predicted)
    return {"translation_cm": value["translation_cm"], "yaw_deg": value["yaw_deg"],
            "d_n": None, "score": None, "pass": None}


def _normalise(metric: dict[str, Any], u_t: float | None, u_r: float | None,
               tau: float) -> dict[str, Any]:
    if metric["translation_cm"] is None or metric["yaw_deg"] is None or not u_t or not u_r:
        return metric
    d_n = math.sqrt((metric["translation_cm"] / max(abs(u_t), 1e-9)) ** 2 +
                    (metric["yaw_deg"] / max(abs(u_r), 1e-9)) ** 2)
    metric["d_n"] = float(d_n)
    metric["score"] = float(math.exp(-d_n))
    metric["pass"] = bool(d_n <= tau)
    return metric


def _route_status(side_a: Mapping[str, Any], side_b: Mapping[str, Any],
                  missing_atomic: list[str]) -> tuple[str, list[str]]:
    reasons = list(side_a.get("failure_reasons", [])) + list(side_b.get("failure_reasons", []))
    if side_a.get("status") != "MEASURED" or side_b.get("status") != "MEASURED":
        return "GEOMETRY_UNMEASURABLE", reasons
    if missing_atomic:
        return "PAIRED_ATOMIC_CELL_UNMEASURABLE", reasons + ["missing_atomic:" + ",".join(missing_atomic)]
    if not side_a.get("nondegenerate") or not side_b.get("nondegenerate"):
        return "STATIC_SHORTCUT", reasons
    return "MEASURED", reasons


def _atomic_cell(key: tuple[str, str, str, int, str], *, data_root: str | Path, pose_index: Mapping, rows: Mapping, cache: dict, ratio: float, coverage: float) -> dict[str, Any]:
    if key in cache: return cache[key]
    pose=pose_index.get(key); row=rows.get(key)
    if pose is None or row is None: cell={"status":"UNIT_NOT_IDENTIFIABLE","task3_unit":{"status":"UNIT_NOT_IDENTIFIABLE","U_a_se2":None}}
    else:
        try: cell=evaluate_cell(pose, _record(row, data_root),row,min_pose_inlier_ratio=ratio,min_pose_anchor_coverage=coverage)
        except Exception as exc: cell={"status":"UNIT_NOT_IDENTIFIABLE","failure_reasons":[f"evaluation_error:{type(exc).__name__}"],"task3_unit":{"status":"UNIT_NOT_IDENTIFIABLE","U_a_se2":None}}
    cache[key]=cell; return cell


def _pair_operators(actions: set[str], identity: tuple[str,str,str,int], *, data_root: str | Path, pose_index: Mapping, rows: Mapping, cache: dict, ratio: float, coverage: float) -> tuple[dict[str,np.ndarray], list[str]]:
    operators={}; missing=[]; model,scene,case,seed=identity
    for action in sorted(actions):
        cell=_atomic_cell((model,scene,case,seed,action),data_root=data_root,pose_index=pose_index,rows=rows,cache=cache,ratio=ratio,coverage=coverage); unit=cell.get("task3_unit",{}); matrix=unit.get("U_a_se2")
        if unit.get("status")!="UNIT_IDENTIFIED" or matrix is None or not unit.get("non_degenerate"): missing.append(action)
        else: operators[action]=np.asarray(matrix,dtype=float)
    return operators,missing


def _pair_normalizers(identity: tuple[str,str,str,int], *, data_root: str | Path, pose_index: Mapping, rows: Mapping, cache: dict, ratio: float, coverage: float) -> tuple[float|None,float|None]:
    model,scene,case,seed=identity
    for action in ACTIONS: _atomic_cell((model,scene,case,seed,action),data_root=data_root,pose_index=pose_index,rows=rows,cache=cache,ratio=ratio,coverage=coverage)
    def values(actions):
        return [abs(float(cache[(model,scene,case,seed,a)]["task3_unit"]["u_a"])) for a in actions if cache[(model,scene,case,seed,a)].get("task3_unit",{}).get("status")=="UNIT_IDENTIFIED" and cache[(model,scene,case,seed,a)].get("task3_unit",{}).get("non_degenerate") and cache[(model,scene,case,seed,a)].get("task3_unit",{}).get("u_a") is not None]
    t,r=values(TRANSLATION_ACTIONS),values(ROTATION_ACTIONS)
    return (float(np.median(t)) if len(t) == len(TRANSLATION_ACTIONS) else None,
            float(np.median(r)) if len(r) == len(ROTATION_ACTIONS) else None)


def _side_ablation(route, side, operators, u_t, u_r, tau):
    result=[]
    for index, observed in enumerate(side.get("boundaries", [])):
        units = index + 1
        predicted = compose_units(route["expanded_unit_actions"][:units], operators)
        metric = _normalise(_metric(np.asarray(observed["se2"]), predicted), u_t, u_r, tau)
        result.append({"segment": index + 1, "units": units, "action": observed["action"], **metric})
    return result


def _self_mean(alpha, beta):
    output={}
    for field in ("translation_cm","yaw_deg","d_n","score"):
        values=[x[field] for x in (alpha,beta) if x.get(field) is not None]; output[field]=float(np.mean(values)) if values else None
    output["pass"]=bool(alpha.get("pass") is True and beta.get("pass") is True); return {"error":output}


def evaluate_pair(row_a: Mapping[str, Any], row_b: Mapping[str, Any], *, data_root: str | Path, route_pose_index: Mapping[tuple[str,str,str,int,str],Path], atomic_pose_index: Mapping[tuple[str,str,str,int,str],Path], atomic_rows: Mapping[tuple[str,str,str,int,str],Mapping[str,Any]], atomic_cache: dict[tuple[str,str,str,int,str],dict[str,Any]], min_pose_inlier_ratio: float = DEFAULT_MIN_POSE_INLIER_RATIO, min_pose_anchor_coverage: float = DEFAULT_MIN_POSE_ANCHOR_COVERAGE, normalized_tau: float = NORMALIZED_PASS_TAU) -> dict[str, Any]:
    key_a,key_b=route_identity(row_a),route_identity(row_b); route_a,route_b=_route(row_a,data_root),_route(row_b,data_root)
    side_a=evaluate_route_side(row_a,route_pose_index.get(key_a),data_root,min_pose_inlier_ratio=min_pose_inlier_ratio,min_pose_anchor_coverage=min_pose_anchor_coverage); side_b=evaluate_route_side(row_b,route_pose_index.get(key_b),data_root,min_pose_inlier_ratio=min_pose_inlier_ratio,min_pose_anchor_coverage=min_pose_anchor_coverage)
    model,scene,case,seed,pair=pair_identity(row_a); identity=(model,scene,case,seed); actions=set(_segment_actions(route_a)+_segment_actions(route_b)); operators,missing=_pair_operators(actions,identity,data_root=data_root,pose_index=atomic_pose_index,rows=atomic_rows,cache=atomic_cache,ratio=min_pose_inlier_ratio,coverage=min_pose_anchor_coverage); status,reasons=_route_status(side_a,side_b,missing); u_t,u_r=_pair_normalizers(identity,data_root=data_root,pose_index=atomic_pose_index,rows=atomic_rows,cache=atomic_cache,ratio=min_pose_inlier_ratio,coverage=min_pose_anchor_coverage)
    relation={"observed_se2":None,"predicted_se2":None,"error":_metric(None,None)}; self_a={"error":_metric(None,None)}; self_b={"error":_metric(None,None)}; predicted_a=predicted_b=None; ablation=[]
    if status in {"MEASURED","STATIC_SHORTCUT"} and not missing:
        predicted_a=compose_units(route_a.get("expanded_unit_actions",[]),operators); predicted_b=compose_units(route_b.get("expanded_unit_actions",[]),operators); observed_a=np.asarray(side_a["final"]); observed_b=np.asarray(side_b["final"]); obs_rel=inverse_se2(observed_a)@observed_b; pred_rel=inverse_se2(predicted_a)@predicted_b; relation={"observed_se2":obs_rel.tolist(),"predicted_se2":pred_rel.tolist(),"error":_normalise(_metric(obs_rel,pred_rel),u_t,u_r,normalized_tau)}; self_a={"error":_normalise(_metric(observed_a,predicted_a),u_t,u_r,normalized_tau)}; self_b={"error":_normalise(_metric(observed_b,predicted_b),u_t,u_r,normalized_tau)}; ablation={"alpha":_side_ablation(route_a,side_a,operators,u_t,u_r,normalized_tau),"beta":_side_ablation(route_b,side_b,operators,u_t,u_r,normalized_tau)}
    return {"model":model,"scene_id":scene,"case_id":case,"seed":seed,"pair_id":pair,"state_id":f"{scene}_{case}_state0000","route_family":str(route_a.get("family","unknown")),"relation_family":classify_relation(route_a,route_b),"route_id_alpha":str(row_a["primary_key"].get("route_id")),"route_id_beta":str(row_b["primary_key"].get("route_id")),"actions_alpha":route_a.get("expanded_unit_actions",[]),"actions_beta":route_b.get("expanded_unit_actions",[]),"status":status,"failure_reasons":reasons,"alpha":side_a,"beta":side_b,"predicted_alpha_se2":predicted_a.tolist() if predicted_a is not None else None,"predicted_beta_se2":predicted_b.tolist() if predicted_b is not None else None,"relation":relation,"self_consistency":{"alpha":self_a,"beta":self_b,"mean":_self_mean(self_a["error"],self_b["error"]) if predicted_a is not None else {"error":_metric(None,None)}},"normalizers":{"u_t_cm":u_t,"u_r_deg":u_r,"tau":normalized_tau,"source":"task2_unit_k1_to_k4"},"ablation":ablation}

def _summary_metrics(routes: list[Mapping[str, Any]]) -> dict[str, Any]:
    measured = [r for r in routes if r.get("status") == "MEASURED"]
    measurable = [r for r in routes if r.get("status") in {"MEASURED", "STATIC_SHORTCUT"}]
    normalized = [r for r in measured if r.get("relation", {}).get("error", {}).get("d_n") is not None]
    def vals(path: tuple[str, ...]) -> list[float]:
        out = []
        for route in measured:
            value: Any = route
            for token in path:
                value = value.get(token) if isinstance(value, Mapping) else None
            if value is not None and np.isfinite(float(value)):
                out.append(float(value))
        return out
    def stat(path: tuple[str, ...]) -> dict[str, Any]:
        values = vals(path)
        return {"median": float(np.median(values)) if values else None,
                "p90": float(np.percentile(values, 90)) if values else None, "n": len(values)}
    total = len(routes)
    summary = {
        "n_pairs": total, "n_measured": len(measured),
        "n_normalized": len(normalized),
        "n_geometry_measurable": sum(r.get("status") not in {"GEOMETRY_UNMEASURABLE", "GENERATION_FAILED", "INTERFACE_UNSUPPORTED"} for r in routes),
        "n_atomic_measurable": sum(r.get("status") not in {"PAIRED_ATOMIC_CELL_UNMEASURABLE", "GEOMETRY_UNMEASURABLE", "GENERATION_FAILED", "INTERFACE_UNSUPPORTED"} for r in routes),
        "n_static_shortcut": sum(r.get("status") == "STATIC_SHORTCUT" for r in routes),
        "interface_coverage": float(sum(r.get("status") != "INTERFACE_UNSUPPORTED" for r in routes) / max(total, 1)),
        "measurability_coverage": float(len(measurable) / max(total, 1)),
        "normalization_coverage": float(len(normalized) / max(total, 1)),
        "e2e_success": float(sum(r.get("status") == "MEASURED" and r.get("relation", {}).get("error", {}).get("pass") is True for r in routes) / max(total, 1)),
        "relation": {"translation_cm": stat(("relation", "error", "translation_cm")),
                     "yaw_deg": stat(("relation", "error", "yaw_deg")), "d_n": stat(("relation", "error", "d_n")),
                     "score": stat(("relation", "error", "score")),
                     "pass_rate": (float(sum(r.get("relation", {}).get("error", {}).get("pass") is True for r in normalized) / len(normalized))
                                   if normalized else None)},
        "self_alpha": {"translation_cm": stat(("self_consistency", "alpha", "error", "translation_cm")),
                        "yaw_deg": stat(("self_consistency", "alpha", "error", "yaw_deg")), "d_n": stat(("self_consistency", "alpha", "error", "d_n")),
                        "score": stat(("self_consistency", "alpha", "error", "score"))},
        "self_beta": {"translation_cm": stat(("self_consistency", "beta", "error", "translation_cm")),
                       "yaw_deg": stat(("self_consistency", "beta", "error", "yaw_deg")), "d_n": stat(("self_consistency", "beta", "error", "d_n")),
                       "score": stat(("self_consistency", "beta", "error", "score"))},
        "self_mean": {"translation_cm": stat(("self_consistency", "mean", "error", "translation_cm")),
                       "yaw_deg": stat(("self_consistency", "mean", "error", "yaw_deg")),
                       "d_n": stat(("self_consistency", "mean", "error", "d_n")),
                       "score": stat(("self_consistency", "mean", "error", "score"))},
    }
    summary["self_mean_d_n_median"] = summary["self_mean"]["d_n"]["median"]
    summary["route_category"] = {"trajectory_family": Counter(str(r.get("route_family")) for r in routes),
                                  "relation_family": Counter(str(r.get("relation_family")) for r in routes)}
    return summary


def aggregate_routes(routes: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        groups[(str(route.get("model")), str(route.get("relation_family")))].append(route)
    return {f"{model}|{family}": _summary_metrics(group) for (model, family), group in sorted(groups.items())}


def build_document(routes: list[dict[str, Any]], *, data_root: str | Path,
                   task2_pose_root: str | Path, task3_pose_root: str | Path,
                   worker_profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "spec": {"translation_floor_cm": TRANSLATION_FLOOR_CM,
        "rotation_floor_deg": ROTATION_FLOOR_DEG, "normalized_pass_tau": NORMALIZED_PASS_TAU,
        "atomic_operator_dose_max": int(FROZEN_METRIC_SETTINGS["task3_operator_dose_max"]),
        "frozen_settings_sha256": FROZEN_METRIC_SETTINGS_SHA256,
        "composition": "right_multiply_in_execution_order",
        "boundary_policy": "all_operator_boundaries_require_M_t"},
        "data_root": str(data_root), "task2_pose_root": str(task2_pose_root), "task3_pose_root": str(task3_pose_root),
        "run_profile": dict(worker_profile or {}), "n_pairs": len(routes), "routes": routes,
        "model_relation_summary": aggregate_routes(routes),
    }


def _flatten(route: Mapping[str, Any]) -> dict[str, Any]:
    error = route.get("relation", {}).get("error", {})
    sa = route.get("self_consistency", {}).get("alpha", {}).get("error", {})
    sb = route.get("self_consistency", {}).get("beta", {}).get("error", {})
    return {"model": route.get("model"), "scene_id": route.get("scene_id"), "case_id": route.get("case_id"),
            "state_id": route.get("state_id"), "seed": route.get("seed"), "pair_id": route.get("pair_id"), "route_family": route.get("route_family"),
            "relation_family": route.get("relation_family"), "route_id_alpha": route.get("route_id_alpha"),
            "route_id_beta": route.get("route_id_beta"), "status": route.get("status"),
            "failure_reasons": ";".join(route.get("failure_reasons", [])),
            "relation_translation_cm": error.get("translation_cm"), "relation_yaw_deg": error.get("yaw_deg"),
            "relation_d_n": error.get("d_n"), "relation_score": error.get("score"), "relation_pass": error.get("pass"),
            "self_alpha_translation_cm": sa.get("translation_cm"), "self_alpha_yaw_deg": sa.get("yaw_deg"),
            "self_alpha_d_n": sa.get("d_n"), "self_alpha_score": sa.get("score"),
            "self_beta_translation_cm": sb.get("translation_cm"), "self_beta_yaw_deg": sb.get("yaw_deg"),
            "self_beta_d_n": sb.get("d_n"), "self_beta_score": sb.get("score"),
            "u_t_cm": route.get("normalizers", {}).get("u_t_cm"), "u_r_deg": route.get("normalizers", {}).get("u_r_deg")}


def write_outputs(document: Mapping[str, Any], output_root: str | Path, *, invocation: str = "") -> dict[str, str]:
    root = Path(output_root); root.parent.mkdir(parents=True, exist_ok=True); root.mkdir(parents=True, exist_ok=True)
    json_path = root / "task3.json"; csv_path = root / "task3.routes.csv"; model_path = root / "task3.model_relation.csv"; report_path = root / "TASK3_REPORT.md"
    json_path.write_text(json.dumps({**document, "invocation": invocation}, ensure_ascii=False, indent=2, allow_nan=False))
    rows = [_flatten(route) for route in document.get("routes", [])]
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries = []
    for key, value in document.get("model_relation_summary", {}).items():
        model, family = key.split("|", 1); summaries.append({"model": model, "relation_family": family, **value})
    with model_path.open("w", newline="") as handle:
        fields = ["model", "relation_family", "n_pairs", "n_measured", "n_normalized", "n_geometry_measurable", "n_atomic_measurable", "interface_coverage", "measurability_coverage", "normalization_coverage", "e2e_success", "n_static_shortcut", "relation", "self_alpha", "self_beta", "self_mean_d_n_median"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fields})
    report_path.write_text(render_markdown(document, invocation=invocation), encoding="utf-8")
    return {"json": str(json_path), "routes_csv": str(csv_path), "model_csv": str(model_path), "report": str(report_path)}


def render_markdown(document: Mapping[str, Any], *, invocation: str = "") -> str:
    lines = ["# StrideWorld Task.3 Evaluation Report", "", "## Evaluation Inputs", "",
             f"- Data root: `{document.get('data_root')}`",
             f"- Task2 atomic operators: `{document.get('task2_pose_root')}`",
             f"- Task3 route poses: `{document.get('task3_pose_root')}`",
             f"- Number of route pairs: {document.get('n_pairs', 0)}",
             "- Current trajectory classification: G1 is inverse (different action multisets); G2/G3/G4 are order (same action multiset, different execution order).", "",
             "## Computation", "",
             f"Each route reads the observed SE(2) at the initial frame and at every unit-action boundary; every boundary must satisfy `M_t` (valid camera, `pose_inlier_ratio>={document.get('run_profile', {}).get('min_pose_inlier_ratio', DEFAULT_MIN_POSE_INLIER_RATIO)}`, `pose_anchor_coverage>={document.get('run_profile', {}).get('min_pose_anchor_coverage', DEFAULT_MIN_POSE_ANCHOR_COVERAGE)}`). The predicted trajectory uses the short-range atomic operators of the same model/scene/case/seed, estimated from Task 2 checkpoints k=1..4, right-multiplied in execution order; the complex route itself is not truncated to 4 steps. The relation error is the translation and yaw distance between `Delta_obs = Pa_obs^-1 Pb_obs` and `Delta_pred = Pa_pred^-1 Pb_pred`; the self error compares the observed and predicted endpoints of each side. The normalized distance is `sqrt((D_t/u_t)^2+(D_r/u_r)^2)` and the score is `exp(-d_n)`; this report evaluates the pass rate at `tau=0.5`. Each unit action must also translate at least 1 cm or rotate at least 0.25 deg; otherwise it is marked `STATIC_SHORTCUT`.", "",
             "## Results by Model and Relation Family", "", "|Model|Relation family|Pairs|Raw measurable|Normalized measurable|Interface coverage|Measurability coverage|Normalization coverage|E2E pass rate|Relation d_n median|Relation score median|self d_n median|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, summary in document.get("model_relation_summary", {}).items():
        model, family = key.split("|", 1); rel = summary.get("relation", {}); self_d = summary.get("self_mean_d_n_median")
        lines.append(f"|{model}|{family}|{summary.get('n_pairs')}|{summary.get('n_measured')}|{summary.get('n_normalized')}|{summary.get('interface_coverage', 0):.3f}|{summary.get('measurability_coverage', 0):.3f}|{summary.get('normalization_coverage', 0):.3f}|{summary.get('e2e_success', 0):.3f}|{rel.get('d_n', {}).get('median')}|{rel.get('score', {}).get('median')}|{self_d}|")
    lines += ["", "## Status Definitions", "", "`MEASURED` means all complex-route boundaries satisfy `M_t` and the Task 2 k=1..4 atomic operators actually required by the route are available; `GEOMETRY_UNMEASURABLE` means the pose is missing, a boundary frame is invalid, or any boundary fails `M_t`; `PAIRED_ATOMIC_CELL_UNMEASURABLE` means a required short-range atomic operator is missing or unidentified; `STATIC_SHORTCUT` means the non-degeneracy gate failed.", "", "## Reproduction Command", "", "```bash", invocation or "python metrics/task3/run_task3_pool.py --help", "```", "", "## Output Files", "", "This directory also holds the per-route JSON/CSV, the CSV aggregated by model and relation family, and this report."]
    return "\n".join(lines) + "\n"


__all__ = ["DEFAULT_MIN_POSE_ANCHOR_COVERAGE", "DEFAULT_MIN_POSE_INLIER_RATIO", "aggregate_routes", "build_document", "classify_relation", "compose_units", "evaluate_pair", "evaluate_route_side", "index_route_pose_files", "pair_rows", "ready_records", "render_markdown", "write_outputs"]
