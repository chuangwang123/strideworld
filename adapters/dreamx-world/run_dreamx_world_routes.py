#!/usr/bin/env python3
"""Run one frozen final Task1/Task3 route through native DreamX AR inference.

The portable case manifest is the sole trajectory source.  A route of ``n``
external dose-1 actions is lowered to one native DreamX invocation with:

* the exact ordered ``w/s/a/d/j/l`` ``action_seq``;
* ``action_speed_list = [5] + [4] * (n - 1)``;
* ``num_output_latent_frames = n + 1``; and
* ``num_output_pixel_frames = 4n + 1``.

No route is padded, split, stitched, retimed, or sampled at runtime.  final
uses lengths 5/8/11/14; each produces a latent count divisible by the
pinned DreamX three-frame AR block.  The output uses the upstream global
relative camera convention and disables the optional Lab post-process; this
avoids the measured contrast/sharpness collapse caused by the former
``color_correction_strength=1.0``/``--chunk_relative`` combination.  This
runner is intentionally separate from the frozen v7 runner and does not alter
its 5/8 contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the audited source and weight pins from the atomic DreamX runner.  The
# portable route contract below remains independent from the legacy v7 runner.
import run_dreamx_world_d16 as atomic  # noqa: E402


ROOT = atomic.ROOT
MODEL_ID = atomic.MODEL_ID
ContractError = atomic.ContractError
file_sha256 = atomic.file_sha256
require_file = atomic.require_file
validate_source = atomic.validate_source
validate_assets = atomic.validate_assets

PORTABLE_CASE_SCHEMA = "strideworld-portable-case-v1"
RECORD_KIND = "SHARED_TASK1_TASK3"
GENERATION_RECORD_SCHEMA = "strideworld-dreamx-ar-complex-final-record-v1"
SUPPORTED_EXACT_ROUTE_LENGTHS = (8, 11, 14)
COLOR_CORRECTION_STRENGTH = 0.0
CHUNK_RELATIVE = False
ACTION_TOKEN = dict(atomic.ACTION_TOKEN)
CORE_ACTIONS = tuple(ACTION_TOKEN)
FPS = atomic.FPS
NATIVE_SPEED = atomic.NATIVE_SPEED
TRANSLATION_PER_PIXEL_FRAME = atomic.TRANSLATION_PER_PIXEL_FRAME
YAW_DEGREES_PER_PIXEL_FRAME = atomic.YAW_DEGREES_PER_PIXEL_FRAME
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def read_object(path: Path) -> dict[str, Any]:
    require_file(path)
    require(not path.is_symlink(), f"refusing symlink JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid JSON: {path}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_embedded_sha(value: dict[str, Any], key: str, label: str) -> str:
    require(key in value, f"{label}: missing {key}")
    unhashed = dict(value)
    stored = unhashed.pop(key)
    require(isinstance(stored, str) and SHA256.fullmatch(stored) is not None,
            f"{label}: invalid {key}")
    require(stored == canonical_sha256(unhashed), f"{label}: content SHA drift")
    return stored


def require_safe_id(value: Any, label: str) -> str:
    require(isinstance(value, str) and SAFE_ID.fullmatch(value) is not None,
            f"unsafe or missing {label}: {value!r}")
    return value


def resolve_input_image(data_root: Path, condition: dict[str, Any]) -> Path:
    uri = condition.get("image_uri")
    require(isinstance(uri, str) and uri.startswith("data://"),
            "portable condition has invalid image_uri")
    relative = uri.removeprefix("data://")
    require(relative and not Path(relative).is_absolute(), "invalid data URI path")
    root = data_root.resolve()
    unresolved_image = root / relative
    require(not unresolved_image.is_symlink(),
            f"refusing symlink input image: {unresolved_image}")
    image = unresolved_image.resolve()
    require(root in image.parents, "condition image escapes data root")
    require_file(image)
    require(not image.is_symlink(), f"refusing symlink input image: {image}")
    expected_size = condition.get("image_byte_size")
    require(isinstance(expected_size, int) and not isinstance(expected_size, bool)
            and expected_size > 0, "invalid condition image_byte_size")
    require(image.stat().st_size == expected_size, "condition image byte-size drift")
    expected_sha = condition.get("image_sha256")
    require(isinstance(expected_sha, str) and SHA256.fullmatch(expected_sha) is not None,
            "invalid condition image_sha256")
    require(file_sha256(image) == expected_sha, "condition image SHA256 drift")
    return image


def route_timing(route_length: int) -> dict[str, Any]:
    require(isinstance(route_length, int) and not isinstance(route_length, bool),
            "DreamX route length must be an integer")
    require(route_length in SUPPORTED_EXACT_ROUTE_LENGTHS,
            "DreamX final route length must be one of "
            f"{list(SUPPORTED_EXACT_ROUTE_LENGTHS)}")
    latent_frames = route_length + 1
    require(latent_frames % 3 == 0,
            "DreamX final latent count violates native three-frame block grid")
    pixel_frames = 4 * route_length + 1
    speed_list = [5] + [4] * (route_length - 1)
    boundaries = [4 * index for index in range(1, route_length + 1)]
    require(sum(speed_list) - 1 == pixel_frames - 1,
            "DreamX first-frame-relative timing formula mismatch")
    return {
        "route_length": route_length,
        "num_output_latent_frames": latent_frames,
        "num_output_pixel_frames": pixel_frames,
        "action_speed_list": speed_list,
        "action_boundary_frame_indices": boundaries,
    }


def validate_case_structure(case: dict[str, Any]) -> None:
    require(case.get("schema_version") == PORTABLE_CASE_SCHEMA,
            "wrong portable case schema")
    validate_embedded_sha(case, "case_content_sha256", "portable case")
    for key in ("benchmark_scene_id", "scene_id", "case_id", "state_id"):
        require_safe_id(case.get(key), key)
    require(case.get("seed") in (0, 1, 2), "portable case must be frozen to seed 0, 1 or 2")  # was == 0
    trajectories = case.get("trajectories")
    require(isinstance(trajectories, list) and len(trajectories) == 14,
            "final case must contain the frozen 14 trajectories")
    order = case.get("trajectory_order")
    ids = [row.get("record_id") for row in trajectories if isinstance(row, dict)]
    require(len(ids) == len(trajectories), "portable trajectory must be an object")
    require(all(isinstance(value, str) for value in ids),
            "portable trajectory record_id is missing")
    require(len(set(ids)) == len(ids), "portable case has duplicate record_id")
    require(order == ids, "portable trajectory_order drift")
    counts = case.get("counts")
    require(counts == {
        "task2_atomic": 6,
        "shared_task1_task3": 8,
        "total": 14,
    }, "final case count contract drift")
    for ordinal, row in enumerate(trajectories, start=1):
        validate_embedded_sha(row, "record_content_sha256", f"trajectory {ordinal}")
        require(row.get("ordinal_in_case") == ordinal,
                f"trajectory {ordinal}: ordinal drift")


def compile_record(
    case_manifest_path: Path,
    record_id: str,
    data_root: Path,
) -> dict[str, Any]:
    record_id = require_safe_id(record_id, "record_id")
    require_file(case_manifest_path)
    require(not case_manifest_path.is_symlink(),
            f"refusing symlink case manifest: {case_manifest_path}")
    case_manifest_path = case_manifest_path.resolve()
    case = read_object(case_manifest_path)
    validate_case_structure(case)
    matches = [
        row for row in case["trajectories"] if row.get("record_id") == record_id
    ]
    require(len(matches) == 1, f"record_id is not uniquely present: {record_id}")
    record = matches[0]
    require(record.get("kind") == RECORD_KIND,
            f"record is not {RECORD_KIND}: {record_id}")
    for key in ("benchmark_scene_id", "scene_id", "case_id", "state_id", "seed"):
        require(record.get(key) == case.get(key), f"record/case {key} drift")
    require(record.get("seed") in (0, 1, 2), "complex record must use seed 0, 1 or 2")  # was == 0
    require(record.get("metric_consumers") == [
        "TASK1_VISIBLE_WORLD_STABILITY", "TASK3_OPEN_COMPOSITION",
    ], "complex record metric-consumer drift")
    require(record.get("output_namespace") == "shared_task1_task3",
            "complex record output namespace drift")
    output_relpath = record.get("output_relpath")
    require(isinstance(output_relpath, str) and output_relpath,
            "complex record output_relpath is missing")
    output_path = Path(output_relpath)
    require(not output_path.is_absolute() and ".." not in output_path.parts,
            "complex record has unsafe output_relpath")

    route_id = require_safe_id(record.get("route_id"), "route_id")
    source_route_sha = record.get("source_route_content_sha256")
    require(isinstance(source_route_sha, str)
            and SHA256.fullmatch(source_route_sha) is not None,
            "complex record source-route SHA drift")
    steps = record.get("steps")
    require(isinstance(steps, list), "complex record steps must be a list")
    route_length = record.get("route_length")
    require(route_length == len(steps), "complex route_length/steps mismatch")
    timing = route_timing(route_length)
    action_seq: list[str] = []
    normalized_steps: list[dict[str, Any]] = []
    for expected_index, step in enumerate(steps, start=1):
        require(isinstance(step, dict), f"route step {expected_index} is not an object")
        require(set(step) == {"step_index", "canonical_actions", "dose"},
                f"route step {expected_index} fields drift")
        require(step.get("step_index") == expected_index,
                f"route step {expected_index} index drift")
        actions = step.get("canonical_actions")
        require(isinstance(actions, list) and len(actions) == 1,
                f"route step {expected_index} must contain exactly one action")
        action = actions[0]
        require(action in ACTION_TOKEN,
                f"route step {expected_index} has unsupported action: {action!r}")
        dose = step.get("dose")
        require(isinstance(dose, (int, float)) and not isinstance(dose, bool)
                and float(dose) == 1.0,
                f"route step {expected_index} must use dose1")
        action_seq.append(ACTION_TOKEN[action])
        normalized_steps.append({
            "step_index": expected_index,
            "canonical_actions": [action],
            "dose": 1.0,
            "boundary_frame_index": timing["action_boundary_frame_indices"][
                expected_index - 1
            ],
        })

    condition = case.get("condition")
    require(isinstance(condition, dict), "portable case condition is missing")
    prompt = condition.get("prompt")
    require(isinstance(prompt, str) and prompt.strip(), "condition prompt is empty")
    prompt_sha = condition.get("prompt_sha256")
    require(isinstance(prompt_sha, str) and SHA256.fullmatch(prompt_sha) is not None,
            "invalid condition prompt_sha256")
    require(hashlib.sha256(prompt.encode("utf-8")).hexdigest() == prompt_sha,
            "condition prompt SHA256 drift")
    image = resolve_input_image(data_root, condition)
    return {
        "task_id": record_id,
        "record_id": record_id,
        "record_content_sha256": record["record_content_sha256"],
        "case_manifest_path": str(case_manifest_path),
        "case_manifest_sha256": file_sha256(case_manifest_path),
        "case_content_sha256": case["case_content_sha256"],
        "benchmark_scene_id": case["benchmark_scene_id"],
        "scene_id": case["scene_id"],
        "case_id": case["case_id"],
        "state_id": case["state_id"],
        "seed": record["seed"],  # was 0
        "route_id": route_id,
        "route_steps": normalized_steps,
        "route_length": route_length,
        "source_route_content_sha256": source_route_sha,
        "output_relpath": output_relpath,
        "metric_consumers": list(record["metric_consumers"]),
        "image_path": str(image),
        "image_sha256": condition["image_sha256"],
        "caption": prompt,
        "prompt_sha256": prompt_sha,
        "action_seq": action_seq,
        **timing,
    }


def build_command(
    native_root: Path,
    checkpoint: Path,
    base_assets: Path,
    input_json: Path,
    output_directory: Path,
    num_output_latent_frames: int,
    *,
    python_executable: str | None = None,
) -> list[str]:
    return [
        python_executable or sys.executable,
        str(native_root / "inference_ar_forcing.py"),
        "--config_path",
        str(native_root / "configs/dreamx-ar/causal_camera_forcing_5b.yaml"),
        "--model_name",
        str(base_assets),
        "--transformer_path",
        str(native_root / "configs/dreamx-ar"),
        "--base_checkpoint_path",
        str(checkpoint),
        "--data_path",
        str(input_json),
        "--output_folder",
        str(output_directory),
        "--num_output_frames",
        str(num_output_latent_frames),
        "--fps",
        str(FPS),
        "--seed",
        os.environ.get("SW_TASK2_SEED", "0"),  # was "0"
        "--color_correction_strength",
        str(COLOR_CORRECTION_STRENGTH),
    ]


def parse_fraction(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def probe_video(path: Path, expected_frames: int) -> dict[str, Any]:
    require_file(path)
    if shutil.which("ffprobe"):
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
                "-of", "json", str(path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        require(len(streams) == 1, "DreamX output must contain one video stream")
        stream = streams[0]
        frames = int(stream.get("nb_read_frames") or stream.get("nb_frames"))
        fps = parse_fraction(stream["avg_frame_rate"])
        width, height = int(stream["width"]), int(stream["height"])
        duration = float(stream["duration"])
        backend = "ffprobe"
    else:
        import av

        with av.open(str(path)) as container:
            require(len(container.streams.video) == 1,
                    "DreamX output must contain one video stream")
            stream = container.streams.video[0]
            width, height = stream.width, stream.height
            fps = float(stream.average_rate)
            frames = sum(1 for _ in container.decode(video=0))
        duration = frames / fps
        backend = "pyav_decode"
    require(frames == expected_frames,
            f"DreamX final output frame mismatch: {frames} != {expected_frames}")
    require(abs(fps - FPS) <= 1e-6, f"DreamX output FPS mismatch: {fps}")
    require((width, height) == (1280, 704), "DreamX output resolution mismatch")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "byte_size": path.stat().st_size,
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "probe_backend": backend,
    }


def output_directory(output_root: Path, output_relpath: str) -> Path:
    root = output_root.resolve()
    destination = (root / output_relpath).resolve()
    require(root in destination.parents, "record output path escapes output root")
    return destination


def build_evidence(
    compiled: dict[str, Any],
    source_evidence: dict[str, Any],
    asset_evidence: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    return {
        "schema_version": GENERATION_RECORD_SCHEMA,
        "status": "PREFLIGHT_PASS",
        "model_id": MODEL_ID,
        "task": RECORD_KIND,
        "record_id": compiled["record_id"],
        "record_content_sha256": compiled["record_content_sha256"],
        "benchmark_scene_id": compiled["benchmark_scene_id"],
        "scene_id": compiled["scene_id"],
        "case_id": compiled["case_id"],
        "state_id": compiled["state_id"],
        "seed": compiled["seed"],  # was 0
        "metric_consumers": compiled["metric_consumers"],
        "generation_multiplicity": 1,
        "portable_case": {
            "manifest_path": compiled["case_manifest_path"],
            "manifest_sha256": compiled["case_manifest_sha256"],
            "case_content_sha256": compiled["case_content_sha256"],
        },
        "source": source_evidence,
        "assets": asset_evidence,
        "input": {
            "path": compiled["image_path"],
            "sha256": compiled["image_sha256"],
            "prompt": compiled["caption"],
            "prompt_sha256": compiled["prompt_sha256"],
        },
        "route": {
            "route_id": compiled["route_id"],
            "length": compiled["route_length"],
            "steps": compiled["route_steps"],
            "action_tokens": compiled["action_seq"],
            "external_dose_per_step": 1.0,
            "zero_based_action_boundary_frame_indices": compiled[
                "action_boundary_frame_indices"
            ],
            "source_route_content_sha256": compiled[
                "source_route_content_sha256"
            ],
        },
        "native_inference": {
            "entrypoint": "inference_ar_forcing.py",
            "single_native_invocation": True,
            "ordered_continuous_action_sequence": True,
            "stitching": False,
            "padding": False,
            "retiming": False,
            "action_token_mapping": ACTION_TOKEN,
            "action_seq": compiled["action_seq"],
            "action_speed_list": compiled["action_speed_list"],
            "num_output_latent_frames": compiled["num_output_latent_frames"],
            "num_output_pixel_frames": compiled["num_output_pixel_frames"],
            "num_frame_per_ar_block": 3,
            "fps": FPS,
            "sample_height_width": [704, 1280],
            "native_speed": NATIVE_SPEED,
            "translation_per_pixel_frame": TRANSLATION_PER_PIXEL_FRAME,
            "yaw_degrees_per_pixel_frame": YAW_DEGREES_PER_PIXEL_FRAME,
            "color_correction_strength": COLOR_CORRECTION_STRENGTH,
            "chunk_relative": CHUNK_RELATIVE,
            "postprocess_policy": "disabled_to_preserve_native_global_output",
            "camera_pose_reference": "global_first_frame",
        },
        "strideworld_mapping": {
            "supported_exact_route_lengths": list(SUPPORTED_EXACT_ROUTE_LENGTHS),
            "external_dose_per_route_step": 1.0,
            "num_output_latent_frames_formula": "n+1",
            "num_output_pixel_frames_formula": "4n+1",
            "action_speed_list_formula": "[5]+[4]*(n-1)",
            "effective_transitions": 4 * compiled["route_length"],
            "no_tail_padding": True,
        },
        "planned_output_directory": str(destination),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-assets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_manifest = args.case_manifest
    data_root = args.data_root.resolve()
    native_root = args.native_root.resolve()
    checkpoint = args.checkpoint.resolve()
    base_assets = args.base_assets.resolve()
    output_root = args.output_root.resolve()

    # Validate the cheap portable contract before touching large model assets.
    compiled = compile_record(case_manifest, args.record_id, data_root)
    destination = output_directory(output_root, compiled["output_relpath"])
    evidence = build_evidence(
        compiled,
        validate_source(native_root),
        validate_assets(checkpoint, base_assets),
        destination,
    )
    if args.preflight_only:
        print(json.dumps(evidence, indent=2, ensure_ascii=False))
        return 0

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(bool(visible) and "," not in visible,
            "set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    require(not destination.exists(),
            f"refusing existing/partial final output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    input_json = destination / "dreamx_input.json"
    planned_record = destination / "generation_record.planned.json"
    input_json.write_text(
        json.dumps([compiled], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    command = build_command(
        native_root,
        checkpoint,
        base_assets,
        input_json,
        destination,
        compiled["num_output_latent_frames"],
    )
    evidence["execution"] = {
        "cuda_visible_devices": visible,
        "python": sys.executable,
        "command": command,
        "input_json": str(input_json),
    }
    planned_record.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    compat = ROOT / "scripts/compat/dreamx"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(compat) + (
        f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else ""
    )
    subprocess.run(command, cwd=native_root, env=environment, check=True)

    image = Path(compiled["image_path"])
    expected_native_video = destination / (
        f"{compiled['task_id']}_{image.parent.name}_{image.stem}.mp4"
    )
    canonical_video = destination / "raw_generated.mp4"
    require_file(expected_native_video)
    require(not canonical_video.exists(), "raw_generated.mp4 unexpectedly exists")
    shutil.copy2(expected_native_video, canonical_video)
    require(file_sha256(expected_native_video) == file_sha256(canonical_video),
            "canonical video copy differs from native DreamX output")

    evidence["output"] = probe_video(
        canonical_video, compiled["num_output_pixel_frames"],
    )
    evidence["generated_video_sha256"] = evidence["output"]["sha256"]
    evidence["generated_video_byte_size"] = canonical_video.stat().st_size
    evidence["status"] = "COMPLETED_MACHINE_VALIDATED_G3_PENDING"
    generation_record = destination / "generation_record.json"
    require(not generation_record.exists(), "generation_record.json unexpectedly exists")
    generation_record.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": evidence["status"],
        "record": str(generation_record),
        "video": str(canonical_video),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
