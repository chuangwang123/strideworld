#!/usr/bin/env python3
"""Rebuild only Lyra trajectories for the frozen 3,640-record campaign.

The source bundle already contains the corrected per-case intrinsics and
source-image bindings.  This script preserves those fields, replaces only the
camera trajectory with the reviewed 10-frame/dose profile, and writes one
immutable manifest per seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np


PROFILE = "strideworld_10f_dmd_t010_yaw2"
FRAMES_PER_DOSE = 10
TRANSLATION_PER_DOSE = 0.1
YAW_DEGREES_PER_DOSE = 2.0
ACTION_MAP = {
    "W": "MOVE_FORWARD",
    "S": "MOVE_BACKWARD",
    "A": "MOVE_LEFT",
    "D": "MOVE_RIGHT",
    "YL": "YAW_LEFT",
    "YR": "YAW_RIGHT",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rotation_y(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def build_c2w(
    actions: list[str], translation_per_dose: float, yaw_degrees_per_dose: float
) -> tuple[np.ndarray, list[int], int]:
    delta = {
        "MOVE_FORWARD": (0.0, 0.0, translation_per_dose),
        "MOVE_BACKWARD": (0.0, 0.0, -translation_per_dose),
        "MOVE_LEFT": (-translation_per_dose, 0.0, 0.0),
        "MOVE_RIGHT": (translation_per_dose, 0.0, 0.0),
        "YAW_LEFT": (0.0, 0.0, 0.0),
        "YAW_RIGHT": (0.0, 0.0, 0.0),
    }
    yaw = {"YAW_LEFT": -yaw_degrees_per_dose, "YAW_RIGHT": yaw_degrees_per_dose}
    rotation = np.eye(3, dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    poses = [np.eye(4, dtype=np.float32)]
    boundaries = [0]
    for action in actions:
        r0, t0 = rotation.copy(), translation.copy()
        dt = np.asarray(delta[action], dtype=np.float32)
        angle = yaw.get(action, 0.0)
        for frame in range(1, FRAMES_PER_DOSE + 1):
            alpha = frame / FRAMES_PER_DOSE
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = r0 @ rotation_y(angle * alpha)
            pose[:3, 3] = t0 + r0 @ (dt * alpha)
            poses.append(pose)
        rotation = r0 @ rotation_y(angle)
        translation = t0 + r0 @ dt
        boundaries.append(len(poses) - 1)
    active = len(poses)
    native = 1 + 80 * math.ceil((active - 1) / 80)
    poses.extend(poses[-1].copy() for _ in range(native - active))
    return np.asarray(poses, dtype=np.float32), boundaries, active


def safe_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and os.path.realpath(target) == str(source.resolve()):
            return
        raise FileExistsError(f"refusing to replace {target}")
    target.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--translation-per-dose", type=float, default=TRANSLATION_PER_DOSE)
    parser.add_argument("--yaw-degrees-per-dose", type=float, default=YAW_DEGREES_PER_DOSE)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--trust-source-image-sha", action="store_true")
    args = parser.parse_args()
    if args.translation_per_dose <= 0 or args.yaw_degrees_per_dose <= 0:
        raise ValueError("trajectory amplitudes must be positive")
    selected_seeds = tuple(sorted({int(x) for x in args.seeds.split(",") if x != ""}))
    if not selected_seeds or any(seed not in {0, 1, 2} for seed in selected_seeds):
        raise ValueError(f"invalid seeds: {selected_seeds}")
    source = args.source_bundle.resolve()
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    records = source_manifest["records"]
    if len(records) != 3640:
        raise ValueError(f"expected 3640 records, found {len(records)}")
    if Counter(int(r["seed"]) for r in records) != Counter({0: 1960, 1: 840, 2: 840}):
        raise ValueError("unexpected seed counts")
    if args.finalize_only:
        summaries = []
        total = 0
        for seed in (0, 1, 2):
            manifest_path = args.output_root.resolve() / f"seed_{seed}" / "lyra2" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            expected = {0: 1960, 1: 840, 2: 840}[seed]
            if manifest["profile"] != args.profile or manifest["seed"] != seed or manifest["record_count"] != expected:
                raise ValueError(f"invalid completed seed manifest: {manifest_path}")
            total += expected
            summaries.append({"seed": seed, "records": expected, "manifest": str(manifest_path), "sha256": digest(manifest_path)})
        receipt = {"status": "PASS", "profile": args.profile, "source_records": len(records), "output_records": total, "bundles": summaries}
        (args.output_root / "BUILD_COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        return
    seen: set[str] = set()
    image_hash_cache: dict[str, str] = {}
    summaries = []
    for seed in selected_seeds:
        bundle = args.output_root.resolve() / f"seed_{seed}" / "lyra2"
        manifest_path = bundle / "manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"refusing to replace existing bundle: {bundle}")
        for name in ("images", "trajectories", "captions"):
            (bundle / name).mkdir(parents=True, exist_ok=True)
        output_records = []
        for record in records:
            if int(record["seed"]) != seed:
                continue
            rid = str(record["record_id"])
            if rid in seen:
                raise ValueError(f"duplicate record id: {rid}")
            seen.add(rid)
            plan = record["v10_plan"]
            raw_actions = list(plan["expanded_unit_actions"])
            try:
                actions = [ACTION_MAP[a] for a in raw_actions]
            except KeyError as exc:
                raise ValueError(f"{rid}: unsupported action {exc.args[0]}") from exc
            semantic_length = len(actions)
            if semantic_length not in {8, 11, 14, 16}:
                raise ValueError(f"{rid}: bad semantic length {semantic_length}")
            poses, boundaries, active = build_c2w(
                actions, args.translation_per_dose, args.yaw_degrees_per_dose
            )
            expected = {8: (81, 81), 11: (161, 111), 14: (161, 141), 16: (161, 161)}[semantic_length]
            if (len(poses), active) != expected:
                raise AssertionError((rid, len(poses), active, expected))
            old_npz = source / "trajectories" / f"{rid}.npz"
            old = np.load(old_npz)
            intrinsics = np.asarray(old["intrinsics"], dtype=np.float32)
            if intrinsics.ndim == 3:
                intrinsics = intrinsics[0]
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                raise ValueError(f"{rid}: invalid corrected intrinsics")
            height, width = int(old["image_height"]), int(old["image_width"])
            trajectory = bundle / "trajectories" / f"{rid}.npz"
            np.savez_compressed(
                trajectory,
                w2c=np.linalg.inv(poses).astype(np.float32),
                intrinsics=np.repeat(intrinsics[None], len(poses), axis=0),
                image_height=height,
                image_width=width,
            )
            caption = source / "captions" / f"{rid}.json"
            if not caption.is_file():
                raise FileNotFoundError(caption)
            safe_link(caption, bundle / "captions" / caption.name)
            image = Path(record["source_image"])
            if not image.is_file():
                raise FileNotFoundError(image)
            image_key = str(image.resolve())
            if not args.trust_source_image_sha:
                image_hash_cache.setdefault(image_key, digest(image))
                if image_hash_cache[image_key] != record["source_image_sha256"]:
                    raise ValueError(f"{rid}: source image SHA mismatch")
            safe_link(image, bundle / "images" / f"{rid}.png")
            output_records.append(
                {
                    **record,
                    "actions": actions,
                    "semantic_length": semantic_length,
                    "num_frames": len(poses),
                    "valid_prefix_frames": active,
                    "score_valid_prefix": [0, active - 1],
                    "tail_hold_frames": len(poses) - active,
                    "dose_boundary_frames": boundaries,
                    "native_input_sha256": digest(trajectory),
                }
            )
        manifest = {
            "schema_version": "strideworld-lyra2-h1-adapter-v1",
            "profile": args.profile,
            "status": "READY_FOR_PINNED_H200_RUN",
            "model": "lyra2",
            "seed": seed,
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": digest(source_manifest_path),
            "record_count": len(output_records),
            "length_counts": dict(sorted(Counter(r["semantic_length"] for r in output_records).items())),
            "native_frame_counts": dict(sorted(Counter(r["num_frames"] for r in output_records).items())),
            "adapter": {
                "frames_per_dose": FRAMES_PER_DOSE,
                "translation_per_dose": args.translation_per_dose,
                "yaw_degrees_per_dose": args.yaw_degrees_per_dose,
                "pose_scale": 1.0,
                "intrinsics": "preserved from corrected calibrated bundle",
                "native_frame_rule": "1+80*k",
                "padding": "hold final pose; excluded from score_valid_prefix",
            },
            "records": output_records,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        summaries.append({"seed": seed, "records": len(output_records), "manifest": str(manifest_path), "sha256": digest(manifest_path)})
    expected_seen = sum({0: 1960, 1: 840, 2: 840}[seed] for seed in selected_seeds)
    if len(seen) != expected_seen:
        raise AssertionError((len(seen), expected_seen))
    receipt = {"status": "PASS", "profile": args.profile, "source_records": len(records), "output_records": len(seen), "seeds": list(selected_seeds), "bundles": summaries}
    receipt_name = "BUILD_COMPLETE.json" if selected_seeds == (0, 1, 2) else "BUILD_SEEDS_" + "-".join(map(str, selected_seeds)) + ".json"
    (args.output_root / receipt_name).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
