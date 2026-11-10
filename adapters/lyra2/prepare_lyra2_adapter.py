#!/usr/bin/env python3
"""Compile a StrideWorld campaign to the pinned Lyra 2 HOST trajectory profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


FRAMES_PER_DOSE = 10
TRANSLATION_PER_DOSE = 0.1
YAW_DEGREES_PER_DOSE = 2.0
SUPPORTED_LENGTHS = {8, 11, 14, 16}
SUPPORTED_ACTIONS = {
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "YAW_LEFT",
    "YAW_RIGHT",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def expand_actions(record: dict) -> list[str]:
    actions: list[str] = []
    for step in record["steps"]:
        canonical = list(step["canonical_actions"])
        if len(canonical) != 1 or canonical[0] not in SUPPORTED_ACTIONS:
            raise ValueError(f"{record['record_id']}: unsupported action step {canonical}")
        dose = float(step["dose"])
        units = int(round(dose))
        if units < 1 or not math.isclose(dose, units, abs_tol=1e-6):
            raise ValueError(f"{record['record_id']}: dose must be a positive integer, got {dose}")
        actions.extend([canonical[0]] * units)
    if len(actions) not in SUPPORTED_LENGTHS:
        raise ValueError(f"{record['record_id']}: semantic length {len(actions)} not in {sorted(SUPPORTED_LENGTHS)}")
    return actions


def build_c2w(actions: list[str]) -> tuple[np.ndarray, list[int], int]:
    translations = {
        "MOVE_FORWARD": (0.0, 0.0, TRANSLATION_PER_DOSE),
        "MOVE_BACKWARD": (0.0, 0.0, -TRANSLATION_PER_DOSE),
        "MOVE_LEFT": (-TRANSLATION_PER_DOSE, 0.0, 0.0),
        "MOVE_RIGHT": (TRANSLATION_PER_DOSE, 0.0, 0.0),
        "YAW_LEFT": (0.0, 0.0, 0.0),
        "YAW_RIGHT": (0.0, 0.0, 0.0),
    }
    yaw = {"YAW_LEFT": -YAW_DEGREES_PER_DOSE, "YAW_RIGHT": YAW_DEGREES_PER_DOSE}
    rotation = np.eye(3, dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    poses = [np.eye(4, dtype=np.float32)]
    boundaries = [0]

    for action in actions:
        start_rotation, start_translation = rotation.copy(), translation.copy()
        local_translation = np.asarray(translations[action], dtype=np.float32)
        angle = yaw.get(action, 0.0)
        for frame in range(1, FRAMES_PER_DOSE + 1):
            alpha = frame / FRAMES_PER_DOSE
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = start_rotation @ rotation_y(angle * alpha)
            pose[:3, 3] = start_translation + start_rotation @ (local_translation * alpha)
            poses.append(pose)
        rotation = start_rotation @ rotation_y(angle)
        translation = start_translation + start_rotation @ local_translation
        boundaries.append(len(poses) - 1)

    active_frames = len(poses)
    native_frames = 1 + 80 * math.ceil((active_frames - 1) / 80)
    poses.extend(poses[-1].copy() for _ in range(native_frames - active_frames))
    return np.asarray(poses, dtype=np.float32), boundaries, active_frames


def camera_intrinsics(path: Path) -> tuple[np.ndarray, int, int]:
    camera = json.loads(path.read_text(encoding="utf-8"))
    raw = camera.get("intrinsics", camera.get("K"))
    if isinstance(raw, dict):
        width = int(raw.get("width") or camera["width"])
        height = int(raw.get("height") or camera["height"])
        matrix = [[raw["fx"], 0.0, raw["cx"]], [0.0, raw["fy"], raw["cy"]], [0.0, 0.0, 1.0]]
    else:
        width, height = int(camera["width"]), int(camera["height"])
        matrix = raw
    K = np.asarray(matrix, dtype=np.float32)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"invalid camera intrinsics: {path}")
    return K, width, height


def replace_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source.resolve())


def materialize(campaign_path: Path, assets: Path, output_root: Path, limit: int | None) -> dict:
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    records = list(campaign["records"])
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("campaign contains no selected records")
    seeds = {int(record["seed"]) for record in records}
    if len(seeds) != 1:
        raise ValueError(f"one bundle must contain exactly one seed, got {sorted(seeds)}")

    bundle = output_root / "lyra2"
    if (bundle / "manifest.json").exists():
        raise FileExistsError(f"refusing to replace existing bundle: {bundle}")
    for name in ("images", "trajectories", "captions"):
        (bundle / name).mkdir(parents=True, exist_ok=True)

    manifest_records = []
    for index, record in enumerate(records):
        record_id = str(record["record_id"])
        if not record_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in record_id):
            raise ValueError(f"unsafe record id: {record_id!r}")
        actions = expand_actions(record)
        poses, boundaries, active_frames = build_c2w(actions)

        image_uri = str(record["condition"]["image_uri"])
        if not image_uri.startswith("data://"):
            raise ValueError(f"{record_id}: expected data:// image URI, got {image_uri}")
        image = assets / image_uri.removeprefix("data://")
        scene = str(record["benchmark_scene_id"])
        case = str(record["case_id"])
        camera = assets / "cameras" / scene / f"{case}.json"
        if not image.is_file() or not camera.is_file():
            raise FileNotFoundError(f"{record_id}: missing image or camera sidecar")
        expected_image_sha = str(record["condition"]["image_sha256"])
        if file_sha256(image) != expected_image_sha:
            raise ValueError(f"{record_id}: source image SHA mismatch")

        K, width, height = camera_intrinsics(camera)
        image_destination = bundle / "images" / f"{record_id}.png"
        trajectory_destination = bundle / "trajectories" / f"{record_id}.npz"
        caption_destination = bundle / "captions" / f"{record_id}.json"
        replace_symlink(image, image_destination)
        np.savez_compressed(
            trajectory_destination,
            w2c=np.linalg.inv(poses).astype(np.float32),
            intrinsics=np.repeat(K[None], len(poses), axis=0),
            image_height=height,
            image_width=width,
        )
        caption_destination.write_text(
            json.dumps({"0": record["condition"]["prompt"]}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_records.append(
            {
                "record_id": record_id,
                "record_index": int(record.get("record_index", index)),
                "kind": record.get("kind"),
                "scene_id": record.get("scene_id"),
                "case_id": case,
                "route_id": record.get("route_id"),
                "seed": int(record["seed"]),
                "actions": actions,
                "semantic_length": len(actions),
                "num_frames": len(poses),
                "valid_prefix_frames": active_frames,
                "score_valid_prefix": [0, active_frames - 1],
                "tail_hold_frames": len(poses) - active_frames,
                "dose_boundary_frames": boundaries,
                "source_image": str(image.resolve()),
                "source_image_sha256": expected_image_sha,
                "native_input_sha256": file_sha256(trajectory_destination),
            }
        )

    manifest = {
        "schema_version": "strideworld-lyra2-h1-adapter-v1",
        "profile": "strideworld_10f_dmd_t010_yaw2",
        "status": "CANDIDATE_REQUIRES_SMOKE_ACCEPTANCE",
        "model": "lyra2",
        "seed": seeds.pop(),
        "source_campaign": str(campaign_path.resolve()),
        "record_count": len(manifest_records),
        "length_counts": dict(sorted(Counter(record["semantic_length"] for record in manifest_records).items())),
        "native_frame_counts": dict(sorted(Counter(record["num_frames"] for record in manifest_records).items())),
        "adapter": {
            "frames_per_dose": FRAMES_PER_DOSE,
            "translation_per_dose": TRANSLATION_PER_DOSE,
            "yaw_degrees_per_dose": YAW_DEGREES_PER_DOSE,
            "pose_scale": 1.0,
            "camera_convention": "integrate c2w, store w2c",
            "native_frame_rule": "1+80*k",
            "padding": "hold final pose; excluded from score_valid_prefix",
        },
        "records": manifest_records,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def self_check() -> None:
    for length, expected_frames, expected_tail in ((8, 81, 0), (11, 161, 50), (14, 161, 20), (16, 161, 0)):
        poses, boundaries, active = build_c2w(["MOVE_FORWARD"] * length)
        assert poses.shape == (expected_frames, 4, 4)
        assert active == 1 + FRAMES_PER_DOSE * length
        assert len(poses) - active == expected_tail
        assert boundaries == list(range(0, active, FRAMES_PER_DOSE))
    poses, _, _ = build_c2w(["YAW_RIGHT"] * 16)
    angle = math.degrees(math.acos(np.clip((np.trace(poses[160, :3, :3]) - 1) / 2, -1, 1)))
    assert math.isclose(angle, 16.0, abs_tol=1e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    self_check()
    if args.self_check:
        print("PASS: Lyra adapter trajectory arithmetic")
        return
    if not all((args.campaign, args.assets, args.output_root)):
        raise SystemExit("--campaign, --assets and --output-root are required")
    manifest = materialize(args.campaign, args.assets, args.output_root, args.limit)
    print(json.dumps({key: manifest[key] for key in ("profile", "record_count", "length_counts", "native_frame_counts")}, indent=2))


if __name__ == "__main__":
    main()
