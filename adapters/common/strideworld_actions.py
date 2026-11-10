#!/usr/bin/env python3
"""Deterministic StrideWorld action expansion and camera trajectory compiler.

The benchmark unit is intentionally model-native: profiles choose how many
decoded frames represent one unit and the native motion amplitude.  This
module never reads future/GT frames and never estimates a trajectory from the
conditioning image.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np


ACTIONS = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "YAW_LEFT",
    "YAW_RIGHT",
)
SHORT_TO_CANONICAL = {
    "W": "MOVE_FORWARD",
    "S": "MOVE_BACKWARD",
    "A": "MOVE_LEFT",
    "D": "MOVE_RIGHT",
    "YL": "YAW_LEFT",
    "YR": "YAW_RIGHT",
}


def expand_record_actions(record: Mapping[str, Any]) -> list[str]:
    units: list[str] = []
    for step in record["steps"]:
        actions = step["canonical_actions"]
        dose = float(step["dose"])
        rounded = round(dose)
        if len(actions) != 1 or dose != rounded or rounded <= 0:
            raise ValueError(f"record {record.get('record_id')} has a non-discrete step")
        action = actions[0]
        if action not in ACTIONS:
            raise ValueError(f"unsupported action: {action}")
        units.extend([action] * rounded)
    native = [SHORT_TO_CANONICAL[item] for item in record["v10_plan"]["expanded_unit_actions"]]
    if units != native:
        raise ValueError(f"record {record.get('record_id')} action expansion drift")
    if len(units) != int(record["route_length"]):
        raise ValueError(f"record {record.get('record_id')} route length drift")
    return units


def rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )


def pose_at(
    actions: Iterable[str],
    unit_time: float,
    *,
    translation_per_unit: float,
    yaw_degrees_per_unit: float,
) -> np.ndarray:
    """Return global C2W in the right/down/forward camera convention."""
    actions = list(actions)
    t = max(0.0, min(float(unit_time), float(len(actions))))
    complete = min(int(math.floor(t)), len(actions))
    fraction = 0.0 if complete == len(actions) else t - complete
    rotation = np.eye(3, dtype=np.float64)
    position = np.zeros(3, dtype=np.float64)

    def advance(action: str, amount: float) -> None:
        nonlocal rotation, position
        if action in {"MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT"}:
            local = {
                "MOVE_FORWARD": np.array([0.0, 0.0, 1.0]),
                "MOVE_BACKWARD": np.array([0.0, 0.0, -1.0]),
                "MOVE_LEFT": np.array([-1.0, 0.0, 0.0]),
                "MOVE_RIGHT": np.array([1.0, 0.0, 0.0]),
            }[action]
            position = position + rotation @ local * translation_per_unit * amount
        else:
            sign = -1.0 if action == "YAW_LEFT" else 1.0
            rotation = rotation @ rotation_y(sign * yaw_degrees_per_unit * amount)

    for action in actions[:complete]:
        advance(action, 1.0)
    if complete < len(actions) and fraction:
        advance(actions[complete], fraction)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def sample_c2w(
    actions: Iterable[str],
    sample_unit_times: Iterable[float],
    *,
    translation_per_unit: float,
    yaw_degrees_per_unit: float,
    dtype: str = "float64",
) -> np.ndarray:
    result = np.stack(
        [
            pose_at(
                actions,
                unit_time,
                translation_per_unit=translation_per_unit,
                yaw_degrees_per_unit=yaw_degrees_per_unit,
            )
            for unit_time in sample_unit_times
        ]
    ).astype(dtype)
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite camera trajectory")
    return result


def pixel_intrinsics(width: int, height: int, hfov_degrees: float = 84.0) -> np.ndarray:
    if width <= 0 or height <= 0 or not (0.0 < hfov_degrees < 180.0):
        raise ValueError("invalid image geometry")
    focal = width / (2.0 * math.tan(math.radians(hfov_degrees) / 2.0))
    return np.asarray(
        ((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )


def canonical_record_sha(record: Mapping[str, Any]) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
