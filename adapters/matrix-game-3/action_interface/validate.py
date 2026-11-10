#!/usr/bin/env python3
"""Validate the Lite protocol, adapters, cases, and compiled invariants."""

from __future__ import annotations

from pathlib import Path

from interface import compile_all, discover_adapters, read_json, validate_all


ROOT = Path(__file__).resolve().parent


def validate_record(record: dict) -> None:
    """Check invariants that only exist after timeline compilation."""

    if record["status"] == "N/A":
        if not record.get("reason"):
            raise AssertionError(f"N/A record lacks a reason: {record['case_id']}")
        return
    command = record["command_window"]
    observation = record["observation_window"]
    if command["end_frame_exclusive"] >= observation["end_frame_exclusive"]:
        raise AssertionError(f"Observation tail is missing: {record['case_id']}")
    expected_ticks = observation["end_frame_exclusive"] - record["native_tick_frames"] * len(
        record["native_timeline"]
    )
    if expected_ticks != command["start_frame"]:
        raise AssertionError(f"Timeline/frame mapping mismatch: {record['case_id']}")
    for index, step in enumerate(record["steps"]):
        if step["step_index"] != index:
            raise AssertionError(f"Step order changed: {record['case_id']}")
        if step["end_tick_exclusive"] <= step["start_tick"]:
            raise AssertionError(f"Empty command step: {record['case_id']}")


def validate_compiled(records: list[dict]) -> None:
    """Validate every supported and N/A record."""

    for record in records:
        validate_record(record)


def main() -> None:
    """Run all checks using the checked-in example cases."""

    protocol = read_json(ROOT / "protocol.json")
    adapters = discover_adapters(ROOT / "adapters")
    cases = read_json(ROOT / "example_cases.json")
    validate_all(protocol, adapters, cases)
    compiled = compile_all(adapters, cases, protocol)
    validate_compiled(compiled["records"])
    supported = sum(record["status"] == "SUPPORTED" for record in compiled["records"])
    print(f"PASS: {len(adapters)} adapters, {len(cases['cases'])} cases, "
          f"{supported} supported records, {len(compiled['records']) - supported} explicit N/A")


if __name__ == "__main__":
    main()
