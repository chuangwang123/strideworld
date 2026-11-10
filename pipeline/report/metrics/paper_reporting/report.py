"""Manifest-bound paper reports; no generation, tracking, PnP or remote work.

Run ``python -m metrics.paper_reporting --example-manifest`` for the input
contract. A normalized index supplies the route identities and boundaries;
explicit case counts validate its planned size, not an independent full-data
acceptance. Freeze and hash that index before model reporting. Missing atomic
cells and missing Task 3 result pairs retain their planned zero contribution.
Missing visual receipts are NA, never substituted from another video set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .composition import build_composition_report, load_composition_document
from .variability import build_variability_report


ACTIONS = ("W", "S", "A", "D", "YL", "YR")
HEADERS = (
    ("geometry_fidelity", "Ray score ↑"),
    ("direction_accuracy", "Dir. rate ↑"),
    ("atomic_consistency", "Atomic score ↑"),
    ("support", "Support ↑"),
    ("composition_translation", "Trans. score ↑"),
    ("composition_rotation", "Yaw score ↑"),
    ("imaging_quality", "MUSIQ ↑"),
    ("aesthetic_quality", "Aesthetic ↑"),
)


def example_manifest() -> dict[str, Any]:
    """Return a deliberately non-runnable template: replace each SHA placeholder."""
    return {
        "schema_version": "strideworld-paper-manifest-v1", "calibration_horizon": 4,
        "evaluation_doses": [8, 12, 16], "plan": {"scene": ["case_a", "case_b"]},
        "expected_seeds": [0, 1, 2], "task3_pairs_per_case": 4,
        "pose_quality": {"min_pose_inlier_ratio": 0.1, "min_pose_anchor_coverage": 0.1},
        "task3_quality": {"track_ratio_min": 0.1, "motion_ratio_min": 0.1},
        "ray_reference_deg": 1.0, "scale_floors": {"translation": 0.0, "rotation": 0.0},
        "models": [{"model": "model-name", "results_root": "/absolute/results/model-name",
                    "task2_units": "/absolute/task2_unit_horizons.json",
                    "task3_pairs": "/absolute/task3_case_unit4.json",
                    "expected_source_sha256": {"normalized_index": "REPLACE_WITH_FROZEN_SHA256",
                                               "task2_units": "REPLACE_WITH_FROZEN_SHA256",
                                               "task3_pairs": "REPLACE_WITH_FROZEN_SHA256",
                                               "task1": "REQUIRED_IF_TASK1_FILE_EXISTS"}}],
        "notes": ["Paths may be relative to the manifest directory. No automatic source relocation.",
                  "For relocation supply a model's source_results_root_relocation mapping old absolute root to this results_root.",
                  "normalized_index defaults to results_root/details/normalized_index.jsonl; task1 defaults to results_root/task1/task1.json.",
                  "task3_pairs_per_case is an integer or scene-to-case-to-count mapping; it includes all Task 3 seeds combined.",
                  "Index actions/frame boundaries plus expected per-case pair counts define the source plan; this is not additional independent full-data acceptance.",
                  "K8 requires evaluation_doses strictly above 8 AND Task 3 raw errors recomputed with K8, declared in definition.unit_horizon.",
                  "Optional visual_receipt requires expected_visual_manifest_sha256, a PASS receipt with matching model/visual_manifest_sha256 and means in [0,1], and expected_source_sha256.visual_receipt.",
                  "Receipt means are imaging_quality_mean and aesthetic_quality_mean. A PASS multi-model receipt may supply models[model].manifest_sha256 instead of the single-model visual_manifest_sha256 field. Missing receipt leaves both visual columns NA.",
                  "Output must be a new or empty directory; existing reports are never overwritten.",
                  "Zero scale floors are numerical guards, not calibrated measurement resolution. Freeze one policy across all models."],
    }


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{label} must be an explicit SHA256")
    return value.lower()


def _path(value: Any, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("input paths must be nonempty strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _verified_path(entry: Mapping, name: str, value: Any, base: Path) -> tuple[Path, str]:
    path = _path(value, base)
    expected = _digest(entry.get("expected_source_sha256", {}).get(name), f"expected_source_sha256.{name}")
    actual = _hash(path)
    if actual != expected:
        raise ValueError(f"{name} source SHA256 mismatch: {path}")
    return path, actual


def _validate_manifest(manifest: Mapping) -> None:
    required = ("calibration_horizon", "evaluation_doses", "plan", "expected_seeds", "task3_pairs_per_case",
                "pose_quality", "task3_quality", "ray_reference_deg", "scale_floors", "models")
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(f"missing explicit manifest fields: {missing}")
    horizon = manifest["calibration_horizon"]
    if type(horizon) is not int or horizon not in (4, 8):
        raise ValueError("calibration_horizon must explicitly be 4 or 8")
    doses = manifest["evaluation_doses"]
    if not isinstance(doses, list) or not doses or any(type(k) is not int or not horizon < k <= 16 for k in doses) or doses != sorted(set(doses)):
        raise ValueError("evaluation_doses must be unique increasing held-out doses after calibration and <=16")
    plan = manifest["plan"]
    if not isinstance(plan, Mapping) or not plan:
        raise ValueError("plan must be a nonempty scene-to-case mapping")
    for scene, cases in plan.items():
        if not isinstance(scene, str) or not scene or not isinstance(cases, list) or not cases or any(not isinstance(c, str) or not c for c in cases) or len(cases) != len(set(cases)):
            raise ValueError("plan contains malformed or duplicate cases")
    seeds = manifest["expected_seeds"]
    if not isinstance(seeds, list) or len(seeds) < 2 or any(type(s) is not int or s < 0 for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("expected_seeds must contain at least two distinct nonnegative integer seeds")
    for policy, names in (("pose_quality", ("min_pose_inlier_ratio", "min_pose_anchor_coverage")),
                          ("task3_quality", ("track_ratio_min", "motion_ratio_min"))):
        for name in names:
            value = manifest[policy].get(name)
            if not _finite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid explicit threshold {policy}.{name}")
    if not _finite(manifest["ray_reference_deg"]) or manifest["ray_reference_deg"] <= 0:
        raise ValueError("ray_reference_deg must be positive and explicit")
    if set(manifest["scale_floors"]) != {"translation", "rotation"} or any(not _finite(v) or v < 0 for v in manifest["scale_floors"].values()):
        raise ValueError("scale_floors must explicitly supply nonnegative translation and rotation")
    models = manifest["models"]
    if not isinstance(models, list) or not models or any(not isinstance(m, Mapping) or not isinstance(m.get("model"), str) or not m["model"] for m in models):
        raise ValueError("models must be a nonempty list with model names")
    if len({m["model"] for m in models}) != len(models):
        raise ValueError("duplicate model in manifest")
    for scene, cases in plan.items():
        for case in cases:
            _pair_count(manifest, scene, case)


def _pair_count(manifest: Mapping, scene: str, case: str) -> int:
    specification = manifest["task3_pairs_per_case"]
    try:
        count = specification[scene][case] if isinstance(specification, Mapping) else specification
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing planned pair count for {scene}/{case}") from error
    if type(count) is not int or count <= 0:
        raise ValueError("planned pair counts must be positive integers")
    return count


def _relocate(path: str, entry: Mapping, root: Path) -> Path:
    source = Path(path)
    if not source.is_absolute():
        raise ValueError(f"source provenance path must be absolute: {path}")
    relocations = entry.get("source_results_root_relocation", {})
    if not isinstance(relocations, Mapping):
        raise ValueError("source_results_root_relocation must map old roots to the current root")
    matches = []
    for old, new in relocations.items():
        old_path, new_path = Path(old), Path(new)
        if not old_path.is_absolute() or not new_path.is_absolute() or new_path.resolve() != root:
            raise ValueError("relocation must explicitly map an absolute source root to this results_root")
        try:
            matches.append(root / source.relative_to(old_path))
        except ValueError:
            pass
    if len(matches) > 1:
        raise ValueError("ambiguous overlapping source relocations")
    return (matches[0] if matches else source).resolve()


def _source_root(document: Mapping, entry: Mapping, root: Path, name: str) -> None:
    declared = document.get("source", {}).get("results_root")
    if not isinstance(declared, str) or _relocate(declared, entry, root) != root:
        raise ValueError(f"{name} source.results_root mismatch; explicit relocation required")


def _same_thresholds(actual: Mapping, expected: Mapping, label: str) -> None:
    for key, value in expected.items():
        if not _finite(actual.get(key)) or not math.isclose(actual[key], value, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{label} threshold mismatch: {key}")


def _load_index(path: Path, model: str, manifest: Mapping) -> tuple[list[dict], dict]:
    rows, records, identities = [], {}, set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        # Use the same duplicate-key rule as JSON documents without a temp file.
        def unique(pairs):
            out = {}
            for key, value in pairs:
                if key in out:
                    raise ValueError(f"duplicate normalized JSON key at line {line_number}: {key}")
                out[key] = value
            return out
        row = json.loads(line, object_pairs_hook=unique)
        key = row.get("primary_key", {})
        if key.get("model") != model:
            raise ValueError("normalized index model mismatch")
        if key.get("task") not in ("task2", "task3"):
            continue
        scene, case, seed = key.get("scene_id"), key.get("case_id"), key.get("seed")
        if scene not in manifest["plan"] or case not in manifest["plan"][scene] or type(seed) is not int or seed not in manifest["expected_seeds"]:
            raise ValueError("normalized row falls outside frozen case/seed plan")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or Path(record_id).name != record_id:
            raise ValueError("invalid normalized record_id")
        identity = (key["task"], scene, case, seed, key.get("route_id"))
        if record_id in records or identity in identities:
            raise ValueError("duplicate normalized record or primary identity")
        if row.get("status") == "GENERATION_READY":
            _digest(row.get("video_sha256"), "normalized video_sha256")
        records[record_id] = row
        identities.add(identity)
        rows.append(row)
    return rows, records


def _expected_pairs(rows: list[dict], manifest: Mapping, model: str) -> list[dict]:
    groups = {}
    for row in rows:
        key = row["primary_key"]
        if key["task"] != "task3":
            continue
        record = row.get("_record", {})
        route = record.get("route", row.get("route", {}))
        member = row.get("member", route.get("member"))
        pair_id = row.get("pair_id", route.get("pair_id"))
        if member not in ("A", "B") or not isinstance(pair_id, str) or not pair_id:
            raise ValueError("Task 3 index must explicitly bind pair_id and member A/B")
        group_key = (key["scene_id"], key["case_id"], key["seed"], pair_id)
        group = groups.setdefault(group_key, {})
        if member in group:
            raise ValueError("duplicate Task 3 pair member")
        actions = route.get("expanded_unit_actions")
        frames = record.get("timing", {}).get("unit_boundary_frames", row.get("unit_boundary_frames"))
        if not isinstance(actions, list) or not actions or any(a not in ACTIONS for a in actions) or not isinstance(frames, list) or len(frames) != len(actions) + 1 or any(type(f) is not int or f < 0 for f in frames) or any(b <= a for a, b in zip(frames, frames[1:])):
            raise ValueError("Task 3 frozen index lacks valid actions/boundaries")
        group[member] = {"record_id": row["record_id"], "video_sha256": row.get("video_sha256"),
                         "generation_status": row.get("status"),
                         "actions": actions, "frame_boundaries": frames}
    counts = Counter((key[0], key[1]) for key in groups)
    for scene, cases in manifest["plan"].items():
        for case in cases:
            if counts[(scene, case)] != _pair_count(manifest, scene, case):
                raise ValueError(f"source index pair count differs from frozen plan: {scene}/{case}")
    output = []
    for (scene, case, seed, pair_id), members in sorted(groups.items()):
        if set(members) != {"A", "B"}:
            raise ValueError("incomplete planned Task 3 A/B pair in normalized index")
        output.append({"model": model, "scene_id": scene, "case_id": case, "seed": seed, "pair_id": pair_id,
                       "alpha": members["A"], "beta": members["B"]})
    return output


def _bind_units(units: Mapping, records: Mapping, manifest: Mapping) -> None:
    for scene, scene_data in units.get("scenes", {}).items():
        for case, case_data in scene_data.get("cases", {}).items():
            for seed, seed_data in case_data.get("seeds", {}).items():
                for action, item in seed_data.get("actions", {}).items():
                    locator = item.get("video_locator", {})
                    row = records.get(locator.get("record_id"))
                    identified = item.get("unit_horizons", {}).get(str(manifest["calibration_horizon"]), {}).get("status") == "UNIT_IDENTIFIED"
                    if row is None:
                        if identified:
                            raise ValueError("identified unit video locator is absent from normalized index")
                        continue  # An unavailable planned unit cannot contribute a value.
                    key = row["primary_key"]
                    if (key["task"], key["scene_id"], key["case_id"], str(key["seed"]), key["route_id"]) != ("task2", scene, case, str(seed), f"atomic_{action}"):
                        raise ValueError("unit video identity mismatch")
                    if (not identified and row.get("status") != "GENERATION_READY"
                            and locator.get("video_sha256") is None and row.get("video_sha256") is None):
                        continue
                    if _digest(locator.get("video_sha256"), "unit video_sha256") != _digest(row.get("video_sha256"), "normalized video_sha256"):
                        raise ValueError("unit video SHA256 mismatch")


def _bind_pairs(source: Mapping, expected: list[dict]) -> None:
    planned = {(p["scene_id"], p["case_id"], p["seed"], p["pair_id"]): p for p in expected}
    for pair in source.get("pairs", []):
        identity = tuple(pair.get(k) for k in ("scene_id", "case_id", "seed", "pair_id"))
        if identity not in planned:
            raise ValueError("Task 3 result contains an unplanned pair")
        for member in ("alpha", "beta"):
            actual, required = pair.get(member, {}), planned[identity][member]
            if actual.get("record_id") != required["record_id"]:
                raise ValueError("Task 3 pair record_id mismatch")
            if (pair.get("quality_pass") is not True and required["generation_status"] != "GENERATION_READY"
                    and actual.get("video_sha256") is None and required.get("video_sha256") is None):
                continue
            if _digest(actual.get("video_sha256"), "Task 3 pair video_sha256") != _digest(required["video_sha256"], "planned pair video_sha256"):
                raise ValueError("Task 3 pair video SHA256 mismatch")


def _evaluate_atomic(*args, **kwargs):
    # Reporting/manifest inspection stays usable without the pose dependencies.
    from .atomic import evaluate_atomic_row
    return evaluate_atomic_row(*args, **kwargs)


def _atomic(rows: list[dict], units: Mapping, root: Path, manifest: Mapping) -> dict:
    index = {}
    for row in rows:
        key = row["primary_key"]
        if key["task"] == "task2":
            action = str(key["route_id"]).removeprefix("atomic_")
            if action not in ACTIONS:
                raise ValueError("unrecognized planned atomic route")
            index[(key["scene_id"], key["case_id"], key["seed"], action)] = row
    output = []
    plan, seeds = manifest["plan"], manifest["expected_seeds"]
    for scene, cases in plan.items():
        for case in cases:
            for seed in seeds:
                for action in ACTIONS:
                    weight = 1 / (len(ACTIONS) * len(plan) * len(cases) * len(seeds))
                    row = index.get((scene, case, seed, action))
                    if row is None:
                        value = {"atomic_score": 0.0, "direction_rate": 0.0, "n_scored": 0,
                                 "n_points": len(manifest["evaluation_doses"]), "failure_reasons": ["missing_planned_atomic_record"]}
                    else:
                        value = _evaluate_atomic(row, root, units,
                            calibration_horizon=manifest["calibration_horizon"], evaluation_doses=manifest["evaluation_doses"],
                            translation_floor=manifest["scale_floors"]["translation"], rotation_floor=manifest["scale_floors"]["rotation"],
                            **manifest["pose_quality"])
                    output.append({"scene_id": scene, "case_id": case, "seed": seed, "action": action,
                                   "weight": weight, **value})
    return {"rows": output, "summary": {
        "atomic_consistency": math.fsum(r["weight"] * r["atomic_score"] for r in output),
        "direction_accuracy": 100 * math.fsum(r["weight"] * r["direction_rate"] for r in output),
        "support_pct": 100 * math.fsum(r["weight"] * r["n_scored"] / r["n_points"] for r in output),
        "n_planned_records": len(output),
        "aggregation": "seeds equally within case -> cases equally within scene -> scenes equally -> fixed six actions equally",
        "failure_counts": dict(Counter(reason for r in output for reason in set(r["failure_reasons"])))}}


def _geometry(document: Mapping | None, expected: list[dict], records: Mapping, manifest: Mapping,
              entry: Mapping, root: Path) -> dict:
    observed = {}
    if document is not None:
        # The standalone evaluator exports constants, whereas the official full
        # evaluation wrapper exports only summary/cases. Successful rows carry
        # the effective thresholds in both schemas.
        if "constants" in document:
            _same_thresholds(document["constants"], manifest["pose_quality"], "Task 1 constants")
        for result in document.get("cases", []):
            row = result.get("row")
            if row is None:
                status = result.get("status")
                record_id = result.get("record_id")
                source = records.get(record_id)
                if not isinstance(status, str) or not status or status == "ok":
                    raise ValueError("Task 1 case without a row must have an explicit failure status")
                if source is None or source["primary_key"]["task"] != "task3":
                    raise ValueError("failed Task 1 record is not bound to a shared Task 3 index video")
                key = source["primary_key"]
                for field in ("model", "scene_id", "case_id", "seed", "route_id"):
                    if field in result and str(result[field]) != str(key.get(field)):
                        raise ValueError(f"failed Task 1 row identity mismatch: {field}")
                if "video_sha256" in result and _digest(result["video_sha256"], "failed Task 1 video_sha256") != _digest(source.get("video_sha256"), "normalized video_sha256"):
                    raise ValueError("failed Task 1 video SHA256 mismatch")
                if record_id in observed:
                    raise ValueError("duplicate Task 1 record")
                observed[record_id] = {"failure_reason": f"task1:{status}"}
                continue
            if not isinstance(row, Mapping):
                raise ValueError("Task 1 row must be a mapping")
            _same_thresholds(row, manifest["pose_quality"], "Task 1 row")
            source_path = row.get("input_pose_npz", result.get("pose_path"))
            if not isinstance(source_path, str):
                raise ValueError("Task 1 row lacks input pose provenance")
            pose_path = _relocate(source_path, entry, root)
            record_id = row.get("record_id", pose_path.parent.name)
            if "record_id" in result and result["record_id"] != record_id:
                raise ValueError("Task 1 result/row record_id mismatch")
            source = records.get(record_id)
            if source is None or source["primary_key"]["task"] != "task3" or pose_path != (root / "artifacts" / record_id / "pose.npz").resolve():
                raise ValueError("Task 1 record is not bound to a shared Task 3 index video")
            key = source["primary_key"]
            if row.get("model") != entry["model"] or row.get("case_id") != key["case_id"] or str(row.get("seed")) != str(key["seed"]):
                raise ValueError("Task 1 row identity mismatch")
            for identity_field in ("scene_id", "route_id"):
                if identity_field in row and row[identity_field] != key[identity_field]:
                    raise ValueError(f"Task 1 row identity mismatch: {identity_field}")
            if "video_sha256" in row and _digest(row["video_sha256"], "Task 1 video_sha256") != _digest(source.get("video_sha256"), "normalized video_sha256"):
                raise ValueError("Task 1 video SHA256 mismatch")
            if record_id in observed:
                raise ValueError("duplicate Task 1 record")
            observed[record_id] = row
    case_counts = Counter((pair["scene_id"], pair["case_id"]) for pair in expected)
    rows = []
    for pair in expected:
        for member in ("alpha", "beta"):
            record_id = pair[member]["record_id"]
            row = observed.get(record_id)
            error = row.get("e_rig_ray_deg_mean") if row else None
            valid = bool(row and _finite(error) and error >= 0 and _finite(row.get("e_rig_ray_deg_n")) and row["e_rig_ray_deg_n"] > 0)
            weight = 1 / (len(case_counts) * case_counts[(pair["scene_id"], pair["case_id"])] * 2)
            rows.append({"record_id": record_id, "scene_id": pair["scene_id"], "case_id": pair["case_id"],
                         "weight": weight, "measurable": valid, "r": error / manifest["ray_reference_deg"] if valid else None,
                         "score": 100 / (1 + error / manifest["ray_reference_deg"]) if valid else 0.0,
                         "failure_reason": None if valid else (row.get("failure_reason", "missing_or_unmeasurable_geometry") if row else "missing_or_unmeasurable_geometry")})
    return {"rows": rows, "summary": {"geometry_fidelity": math.fsum(r["weight"] * r["score"] for r in rows),
        "support_pct": 100 * math.fsum(r["weight"] for r in rows if r["measurable"]),
        "n_expected_records": len(rows), "ray_reference_deg": manifest["ray_reference_deg"],
        "aggregation": "planned shared-task3 videos equally within case, then cases equally; unavailable scores zero"}}


def _visual(entry: Mapping, base: Path) -> dict:
    empty = {"imaging_quality": None, "aesthetic_quality": None, "status": "NA", "reason": "no_visual_receipt"}
    if not entry.get("visual_receipt"):
        return empty
    path, digest = _verified_path(entry, "visual_receipt", entry["visual_receipt"], base)
    expected = _digest(entry.get("expected_visual_manifest_sha256"), "expected_visual_manifest_sha256")
    receipt = load_composition_document(path)
    if receipt.get("status") != "PASS":
        raise ValueError("visual receipt must have status PASS")
    selected = receipt
    if "models" in receipt:
        models = receipt["models"]
        if isinstance(models, list):
            indexed = {}
            for item in models:
                if not isinstance(item, Mapping) or not isinstance(item.get("model"), str) or item["model"] in indexed:
                    raise ValueError("invalid or duplicate model in visual receipt")
                indexed[item["model"]] = item
            models = indexed
        if not isinstance(models, Mapping) or not isinstance(models.get(entry["model"]), Mapping):
            raise ValueError("visual receipt model/manifest mismatch")
        selected = models[entry["model"]]
        if "model" in selected and selected["model"] != entry["model"]:
            raise ValueError("visual receipt model/manifest mismatch")
        if "status" in selected and selected["status"] != "PASS":
            raise ValueError("visual receipt model must have status PASS")
        if "failed" in selected and selected["failed"] != 0:
            raise ValueError("PASS visual receipt model has failed rows")
    elif receipt.get("model") != entry["model"]:
        raise ValueError("visual receipt model/manifest mismatch")
    digest_fields = [selected[key] for key in ("visual_manifest_sha256", "manifest_sha256") if key in selected]
    if not digest_fields or any(_digest(value, "receipt visual manifest SHA256") != expected for value in digest_fields):
        raise ValueError("visual receipt model/manifest mismatch")
    values = {}
    for name in ("imaging_quality", "aesthetic_quality"):
        value = selected.get(f"{name}_mean")
        if not _finite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid visual mean: {name}")
        values[name] = 100 * value
    return {**values, "status": "PASS", "model": entry["model"], "source_sha256": digest,
            "receipt_scope": "multi_model" if "models" in receipt else "single_model",
            "visual_manifest_sha256": expected}


def build_report(manifest: Mapping, *, base_dir: str | Path = ".") -> dict:
    """Validate sources first, then build scores from the explicitly frozen plan."""
    _validate_manifest(manifest)
    base = Path(base_dir).resolve()
    reports, main = [], []
    for entry in manifest["models"]:
        model, root = entry["model"], _path(entry.get("results_root"), base)
        paths, hashes = {}, {}
        for name, value in (("normalized_index", entry.get("normalized_index", str(root / "details/normalized_index.jsonl"))),
                            ("task2_units", entry.get("task2_units")), ("task3_pairs", entry.get("task3_pairs"))):
            paths[name], hashes[name] = _verified_path(entry, name, value, base)
        units, pairs = load_composition_document(paths["task2_units"]), load_composition_document(paths["task3_pairs"])
        for name, document in (("task2_units", units), ("task3_pairs", pairs)):
            if document.get("model") != model:
                raise ValueError(f"{name} model mismatch")
            _source_root(document, entry, root, name)
        _same_thresholds(units.get("definition", {}), manifest["pose_quality"], "unit report")
        _same_thresholds(pairs.get("definition", {}).get("quality_gate", {}), {
            "trackable_pose_point_ratio_min": manifest["task3_quality"]["track_ratio_min"],
            "per_action_motion_min": manifest["task3_quality"]["motion_ratio_min"]}, "Task 3 original gate")
        rows, records = _load_index(paths["normalized_index"], model, manifest)
        expected = _expected_pairs(rows, manifest, model)
        _bind_units(units, records, manifest)
        _bind_pairs(pairs, expected)
        composition = build_composition_report(pairs, units, expected_pairs=expected,
            calibration_horizon=manifest["calibration_horizon"],
            min_translation_scale_cm=manifest["scale_floors"]["translation"],
            min_rotation_scale_deg=manifest["scale_floors"]["rotation"])
        composition["plan"]["source"] = "frozen_hashed_normalized_index_with_explicit_case_counts"
        composition["plan"]["completeness_scope"] = "source_index_plan; not independent additional full-data acceptance"
        atomic = _atomic(rows, units, root, manifest)
        variability = build_variability_report(units, manifest["plan"], expected_seeds=manifest["expected_seeds"],
            min_scale=manifest["scale_floors"], horizon=manifest["calibration_horizon"])
        task1_path = _path(entry.get("task1", str(root / "task1/task1.json")), base)
        geometry_document = None
        if task1_path.exists():
            _, hashes["task1"] = _verified_path(entry, "task1", str(task1_path), base)
            geometry_document = load_composition_document(task1_path)
        geometry = _geometry(geometry_document, expected, records, manifest, entry, root)
        visual = _visual(entry, base)
        cs = composition["summary"]
        main_row = {"model": model, "geometry_fidelity": geometry["summary"]["geometry_fidelity"],
                    **{name: atomic["summary"][name] for name in ("direction_accuracy", "atomic_consistency")},
                    "support": cs["support_pct"], "composition_translation": cs["translation_score"],
                    "composition_rotation": cs["rotation_score"],
                    **{name: visual[name] for name in ("imaging_quality", "aesthetic_quality")}}
        main.append(main_row)
        reports.append({"model": model, "results_root": str(root), "source_sha256": hashes,
                        "atomic": atomic, "geometry": geometry, "composition": composition,
                        "variability": variability, "visual": visual})
    return {"schema_version": "strideworld-paper-report-v1", "manifest": dict(manifest),
            "main_table": main, "models": reports,
            "groups": {"Spatial": ["geometry_fidelity"], "Atomic actions": ["direction_accuracy", "atomic_consistency"],
                       "Composition": ["support", "composition_translation", "composition_rotation"],
                       "Visual quality": ["imaging_quality", "aesthetic_quality"]}}


def _display(value: Any) -> str:
    return "NA" if value is None else f"{value:.1f}" if isinstance(value, (float, int)) else str(value)


def _tex(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")


def _write_table(output: Path, name: str, headers: list[str], values: list[list[Any]], *, groups=False) -> None:
    shown = [[_display(value) for value in row] for row in values]
    with (output / f"{name}.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(shown)
    markdown = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    markdown += ["| " + " | ".join(value.replace("|", r"\|") for value in row) + " |" for row in shown]
    if groups:
        markdown = ["Spatial · Atomic actions · Composition · Visual quality", ""] + markdown
    with (output / f"{name}.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(markdown) + "\n")
    latex = [r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\hline"]
    if groups:
        latex.append(r"Model & \multicolumn{1}{c}{Spatial} & \multicolumn{2}{c}{Atomic actions} & \multicolumn{3}{c}{Composition} & \multicolumn{2}{c}{Visual quality} \\")
    latex += [" & ".join(_tex(h) for h in headers) + r" \\", r"\hline"]
    latex += [" & ".join(_tex(value) for value in row) + r" \\" for row in shown]
    latex += [r"\hline", r"\end{tabular}"]
    with (output / f"{name}.tex").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(latex) + "\n")


def write_report(report: Mapping, output_dir: str | Path) -> None:
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be new or empty; existing reports are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    _write_table(output, "main_table", ["Model"] + [h for _, h in HEADERS],
                 [[row["model"]] + [row[k] for k, _ in HEADERS] for row in report["main_table"]], groups=True)
    for kind in ("seed", "case"):
        values = []
        for model in report["models"]:
            summary = model["variability"]["summary"][kind]
            t, r = summary["translation"], summary["rotation"]
            values.append([model["model"], t["relative_sd_pct"], r["relative_sd_pct"],
                           f"{t['complete_groups']}/{t['planned_groups']}", f"{r['complete_groups']}/{r['planned_groups']}"])
        _write_table(output, f"{kind}_variability", ["Model", "Trans. RSD (%) ↓", "Turn RSD (%) ↓", "Translation Groups", "Turning Groups"], values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path)
    parser.add_argument("--example-manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.example_manifest:
        print(json.dumps(example_manifest(), ensure_ascii=False, indent=2))
        return 0
    if args.manifest is None or args.output_dir is None:
        parser.error("--manifest and --output-dir are required")
    manifest = load_composition_document(args.manifest)
    try:
        report = build_report(manifest, base_dir=args.manifest.resolve().parent)
        report["manifest_sha256"] = _hash(args.manifest)
        write_report(report, args.output_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"report": str(args.output_dir.resolve() / "report.json"), "models": len(report["models"])}))
    return 0
