"""NumPy-only, fixed-scale WorldStereo camera controls.

Coordinates are OpenCV camera-to-world: x right, y down, z forward.
Every dose-one action has the same displacement AND number of output
keyframe intervals in atomic and shared records.  No per-video centering
or magnitude normalization is applied here.  At inference call the native
get_camera_embedding(..., normalize=False, is_w2c=True) explicitly.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ACTION_NAMES = (
    "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT",
    "YAW_LEFT", "YAW_RIGHT",
)
CHECKPOINT_DOSES = tuple(range(1, 17))
_VECTORS = {
    "MOVE_FORWARD": (0., 0., 1.), "MOVE_BACKWARD": (0., 0., -1.),
    "MOVE_LEFT": (-1., 0., 0.), "MOVE_RIGHT": (1., 0., 0.),
    "YAW_LEFT": (0., 0., 0.), "YAW_RIGHT": (0., 0., 0.),
    "NO_OP": (0., 0., 0.),
}


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def rotation_y(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    c, s = math.cos(radians), math.sin(radians)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]], dtype=np.float64)


def compile_trajectory(
    actions: Sequence[str],
    doses: Sequence[int | float] | None = None,
    *,
    translation_per_unit: float = 0.025,
    yaw_degrees_per_unit: float = 0.5,
    keyframe_intervals_per_unit: int = 2,
    tail_hold_intervals: int = 0,
) -> dict[str, Any]:
    """Compile action units to a continuous C2W/W2C keyframe grid.

    Returns float32 ``c2w`` and ``w2c`` arrays of shape (F,4,4), JSON-safe
    timing metadata, and ``pipeline_num_frames=4*(F-1)+1``.  WorldStereo's
    keyframe decoder produces F output images, not pipeline_num_frames.
    Default strengths are conservative adapter candidates, not GPU acceptance.
    """
    if isinstance(actions, (str, bytes)) or len(actions) == 0:
        raise ValueError("actions must be a nonempty sequence")
    if doses is None:
        doses = [1] * len(actions)
    if len(doses) != len(actions):
        raise ValueError("actions and doses must have equal lengths")
    intervals = _integer(keyframe_intervals_per_unit, "keyframe_intervals_per_unit")
    tail = _integer(tail_hold_intervals, "tail_hold_intervals", 0)
    translation = _positive(translation_per_unit, "translation_per_unit")
    yaw_degrees = _positive(yaw_degrees_per_unit, "yaw_degrees_per_unit")
    parsed: list[tuple[str, int]] = []
    for action, dose in zip(actions, doses):
        if action not in _VECTORS:
            raise ValueError(f"unsupported action: {action!r}")
        parsed.append((action, _integer(dose, "dose")))

    poses = [np.eye(4, dtype=np.float64)]
    R, t = np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    steps, units = [], []
    for step_index, (action, dose) in enumerate(parsed):
        start_index = len(poses) - 1
        v = np.asarray(_VECTORS[action], dtype=np.float64) * translation
        yaw = {"YAW_LEFT": -yaw_degrees, "YAW_RIGHT": yaw_degrees}.get(action, 0.)
        for _ in range(dose):
            unit_start = len(poses) - 1
            R0, t0 = R.copy(), t.copy()
            for interval in range(1, intervals + 1):
                alpha = interval / intervals
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = R0 @ rotation_y(yaw * alpha)
                pose[:3, 3] = t0 + R0 @ (v * alpha)
                poses.append(pose)
            R, t = poses[-1][:3, :3].copy(), poses[-1][:3, 3].copy()
            units.append({"action": action, "start_frame_index": unit_start,
                          "end_frame_index": len(poses) - 1})
        steps.append({"step_index": step_index, "canonical_action": action, "dose": dose,
                      "start_frame_index": start_index, "end_frame_index": len(poses) - 1})
    formal_frames = len(poses)
    poses.extend([poses[-1].copy() for _ in range(tail)])
    c2w64 = np.stack(poses)
    w2c64 = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    w2c64[:, :3, :3] = c2w64[:, :3, :3].transpose(0, 2, 1)
    w2c64[:, :3, 3] = -np.einsum("fij,fj->fi", w2c64[:, :3, :3], c2w64[:, :3, 3])
    atomic16 = len(parsed) == 1 and parsed[0][1] == 16 and parsed[0][0] in ACTION_NAMES
    return {
        "schema_version": "worldstereo-camera-fixed-scale-v1",
        "c2w": c2w64.astype(np.float32), "w2c": w2c64.astype(np.float32),
        "steps": steps, "unit_boundaries": units,
        "translation_per_unit": translation, "yaw_degrees_per_unit": yaw_degrees,
        "keyframe_intervals_per_unit": intervals,
        "formal_frame_count": formal_frames, "output_frame_count": len(poses),
        "pipeline_num_frames": 4 * (len(poses) - 1) + 1,
        "terminal_unscored_hold_intervals": tail,
        "checkpoint_doses": list(CHECKPOINT_DOSES) if atomic16 else [],
        "checkpoint_frame_indices": [d * intervals for d in CHECKPOINT_DOSES] if atomic16 else [],
        "camera_reference_frame": "initial_camera_opencv_c2w_body_actions",
        "camera_embedding_normalize": False,
        "cross_model_metric_unit_claim": False,
    }


def compile_record(record: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Accept the existing campaign's one-action steps without relabeling."""
    raw_steps = record.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("record needs nonempty steps")
    actions, doses = [], []
    for step in raw_steps:
        if not isinstance(step, Mapping):
            raise ValueError("each step must be a mapping")
        value = step.get("canonical_actions", step.get("actions"))
        if value is None:
            value = [step.get("action", step.get("canonical_action"))]
        if not isinstance(value, list) or len(value) != 1:
            raise ValueError("each step must contain exactly one canonical action")
        actions.append(value[0])
        doses.append(step.get("dose", step.get("dose_requested", 1)))
    result = compile_trajectory(actions, doses, **kwargs)
    result["record_id"] = record.get("record_id")
    return result


def make_camera_only_render_inputs(image_chw: np.ndarray, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Build explicit no-PCD conditioning, matching official mask polarity.

    image_chw is float RGB in [-1,1].  Unknown future render RGB uses official
    normalized background 0; mask=1 means unknown/no point, mask=0 means valid.
    Only frame zero has the real image and mask zero.  This does not claim
    that every possible no-PCD input was covered by training dropout.
    """
    frames = _integer(frame_count, "frame_count")
    image = np.asarray(image_chw)
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) < 1:
        raise ValueError("image_chw must have shape (3,H,W)")
    if not np.isfinite(image).all() or image.min() < -1 or image.max() > 1:
        raise ValueError("image_chw must contain finite normalized RGB in [-1,1]")
    h, w = image.shape[1:]
    render = np.zeros((1, 3, frames, h, w), dtype=np.float32)
    mask = np.ones((1, 1, frames, h, w), dtype=np.float32)
    render[0, :, 0] = image
    mask[0, :, 0] = 0
    return render, mask


def metadata_without_arrays(control: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in control.items() if not isinstance(value, np.ndarray)}


def compile_camera_controls(blocks: Sequence[str], profile: Mapping[str, Any]) -> dict[str, Any]:
    """Worker API: every semantic block is one complete dose-one unit."""
    actions = ["NO_OP" if block == "NOOP" else block for block in blocks]
    result = compile_trajectory(
        actions,
        translation_per_unit=profile["translation_per_block"],
        yaw_degrees_per_unit=profile["yaw_degrees_per_block"],
        keyframe_intervals_per_unit=profile.get("keyframe_intervals_per_block", 2),
        tail_hold_intervals=profile.get("tail_hold_intervals", 0),
    )
    result["output_frames"] = result["output_frame_count"]
    result["semantic_block_count"] = len(actions)
    result["block_boundary_frame_indices"] = [u["end_frame_index"] for u in result["unit_boundaries"]]
    if len(actions) == 16 and len(set(actions)) == 1 and actions[0] in ACTION_NAMES:
        result["checkpoint_doses"] = list(CHECKPOINT_DOSES)
        result["checkpoint_frame_indices"] = [d * result["keyframe_intervals_per_unit"] for d in CHECKPOINT_DOSES]
    return result
