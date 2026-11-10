#!/usr/bin/env python3
"""Benchmark actions -> WorldCrafter camera trajectory and event list, at the command of the paper.

Every benchmark action spans 16 generated frames. A translation action moves the camera by 16/33 model units and a
yaw action turns it by 4 degrees, both spread evenly over the 16 frames with the official sampler of the pinned
WorldCrafter checkout (worldcrafter.camera.sample_chunk). The trajectory includes the pose at the end of the last
action and is padded with that pose to a whole number of native 33-frame windows; frames after the last action are
not scored. The result is the global camera-to-world array (N, 3, 4) that the official inference.py reads with
--camera-path, and one event string per action.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "sources/WorldCrafter"     # official checkout, see model_settings.json
FRAMES_PER_ACTION = 16
TRANSLATION_PER_ACTION = 16 / 33
YAW_DEGREES_PER_ACTION = 4.0
ACTION_PREFIX = {
    "MOVE_FORWARD": "forward", "MOVE_BACKWARD": "backward", "MOVE_LEFT": "left", "MOVE_RIGHT": "right",
    "YAW_LEFT": "yaw_left", "YAW_RIGHT": "yaw_right",
}


@dataclass(frozen=True)
class Schedule:
    camera: np.ndarray                  # (raw_frame_count, 3, 4) float64, global camera-to-world
    events: tuple[str, ...]             # one official event string per benchmark action
    logical_endpoint_frame: int         # frame of the pose at the end of the last action
    formal_frame_count: int             # scored frames: [0, formal_frame_count)
    raw_frame_count: int                # generated frames (whole 33-frame windows)
    checkpoint_frames: tuple[int, ...]  # frame index of every action boundary


def event_for_action(action: str, translation: float = TRANSLATION_PER_ACTION,
                     yaw_degrees: float = YAW_DEGREES_PER_ACTION) -> str:
    return ACTION_PREFIX[action] + format(yaw_degrees if action.startswith("YAW_") else translation, ".17g")


def compile_schedule(actions: list[str], translation: float = TRANSLATION_PER_ACTION,
                     yaw_degrees: float = YAW_DEGREES_PER_ACTION) -> Schedule:
    sys.path.insert(0, str(SOURCE))
    try:
        from worldcrafter.camera import CHUNK_FRAMES, action_from_event, relative_poses, sample_chunk
    finally:
        sys.path.pop(0)
    world = np.eye(4, dtype=np.float64)
    parts, events = [], []
    fractions = np.arange(FRAMES_PER_ACTION, dtype=np.float64) / FRAMES_PER_ACTION
    for action in actions:
        event = event_for_action(action, translation, yaw_degrees)
        poses, world = sample_chunk(world, action_from_event(event), fractions=fractions)
        parts.append(poses)
        events.append(event)
    endpoint = FRAMES_PER_ACTION * len(actions)
    sampled = np.concatenate((*parts, world[None]), axis=0)
    formal = len(sampled)
    raw = math.ceil(formal / CHUNK_FRAMES) * CHUNK_FRAMES
    if raw > formal:
        sampled = np.concatenate((sampled, np.repeat(world[None], raw - formal, axis=0)))
    camera = np.ascontiguousarray(sampled[:, :3, :4], dtype=np.float64)
    for start in range(0, raw, CHUNK_FRAMES):
        relative = relative_poses(camera[start:start + CHUNK_FRAMES])
        if relative.shape != (CHUNK_FRAMES, 3, 4) or not np.isfinite(relative).all():
            raise ValueError("invalid relative camera input")
    return Schedule(camera=camera, events=tuple(events), logical_endpoint_frame=endpoint, formal_frame_count=formal,
                    raw_frame_count=raw,
                    checkpoint_frames=tuple(FRAMES_PER_ACTION * i for i in range(len(actions) + 1)))


def official_argv(image_path: str, camera_path: str, prompt: str, seed: int, raw_frame_count: int,
                  output_path: str) -> list[str]:
    """Command line of the official inference.py for one video (Fast model, 6 steps, 640x384, 16 fps)."""
    return ["python", "inference.py", "--mode", "i2v", "--model-type", "fast", "--model-path", "${WORLDCRAFTER_FAST_MODEL}",
            "--image-path", image_path, "--camera-path", camera_path, "--no-gravity-align", "--prompt", prompt,
            "--num-chunks", str(raw_frame_count // 33), "--num-inference-steps", "6", "--guidance-scale", "1.0",
            "--seed", str(seed), "--output-path", output_path]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", nargs="+", choices=sorted(ACTION_PREFIX))
    parser.add_argument("--camera-out", type=Path, required=True)
    args = parser.parse_args()
    schedule = compile_schedule(args.actions)
    np.save(args.camera_out, schedule.camera, allow_pickle=False)
    print("\n".join(schedule.events))
