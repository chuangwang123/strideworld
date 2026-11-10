"""Frozen constants shared by the StrideWorld metric implementations."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Final, Mapping


_SETTINGS = {
    "pose_ransac_threshold_deg": 1.0,
    "min_pose_inlier_ratio": 0.3,
    "min_pose_anchor_coverage": 0.3,
    "min_pose_points": 12,
    "min_world_points": 12,
    "min_track_confidence": 0.5,
    "tracker_visibility_threshold": 0.9,
    "translation_floor_cm": 1.0,
    "rotation_floor_deg": 0.25,
    "depth_bins": 3,
    "dose_max": 16,
    "checkpoint_doses": tuple(range(1, 17)),
    "unit_estimation_dose_max": 12,
    "task3_operator_dose_max": 4,
    "power_law_dose_max": 16,
    "robust_scale": 1.4826,
    "epsilon": 1e-9,
}

FROZEN_METRIC_SETTINGS: Final[Mapping[str, object]] = MappingProxyType(_SETTINGS)
FROZEN_METRIC_SETTINGS_SHA256: Final[str] = hashlib.sha256(
    json.dumps(_SETTINGS, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


__all__ = ["FROZEN_METRIC_SETTINGS", "FROZEN_METRIC_SETTINGS_SHA256"]
