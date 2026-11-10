#!/usr/bin/env python3
"""Compile canonical StrideWorld cases into model-native timelines."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from interface import InterfaceError, compile_all, discover_adapters, read_json, validate_all


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse the small public command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--adapter-dir", type=Path, default=ROOT / "adapters")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--models", nargs="*", help="Adapter filename stems; default: all")
    parser.add_argument(
        "--namespace-case-ids-by-model", action="store_true",
        help=(
            "Compile exactly one adapter and rewrite execution case IDs as "
            "<adapter-key>::<probe_case_id>"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_adapters(
    adapters: dict[str, dict], requested: list[str] | None
) -> dict[str, dict]:
    """Select adapters by stable filename while rejecting misspellings."""

    if not requested:
        return adapters
    unknown = sorted(set(requested) - adapters.keys())
    if unknown:
        raise InterfaceError(f"Unknown adapter names: {', '.join(unknown)}")
    return {name: adapters[name] for name in requested}


def namespace_cases_for_adapter(cases: dict, adapter_key: str) -> dict:
    """Materialize collision-free execution IDs while preserving probe IDs."""

    output = copy.deepcopy(cases)
    for case in output.get("cases", []):
        probe_case_id = str(case.get("probe_case_id", case.get("case_id", "")))
        if case.get("probe_case_id") is not None and str(
            case.get("case_id")
        ) != probe_case_id:
            raise InterfaceError(
                "Namespaced compilation requires model-agnostic "
                "case_id == probe_case_id"
            )
        case["case_id"] = f"{adapter_key}::{probe_case_id}"
    return output


def write_json(path: Path, value: dict) -> None:
    """Write deterministic, human-readable JSON output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Load, validate, compile, and report a short summary."""

    args = parse_args()
    protocol = read_json(args.protocol)
    adapters = select_adapters(discover_adapters(args.adapter_dir), args.models)
    cases = read_json(args.cases)
    namespace = None
    if args.namespace_case_ids_by_model:
        if len(adapters) != 1:
            raise InterfaceError(
                "--namespace-case-ids-by-model requires exactly one selected adapter"
            )
        adapter_key, adapter = next(iter(adapters.items()))
        cases = namespace_cases_for_adapter(cases, adapter_key)
        namespace = {
            "kind": "adapter_key_probe_case_v1", "adapter_key": adapter_key,
            "model_id": adapter["model_id"],
        }
    validate_all(protocol, adapters, cases)
    compiled = compile_all(adapters, cases, protocol)
    if namespace is not None:
        compiled["execution_case_namespace"] = namespace
    write_json(args.output, compiled)
    supported = sum(record["status"] == "SUPPORTED" for record in compiled["records"])
    print(f"Wrote {len(compiled['records'])} records: {supported} supported, "
          f"{len(compiled['records']) - supported} N/A -> {args.output}")


if __name__ == "__main__":
    main()
