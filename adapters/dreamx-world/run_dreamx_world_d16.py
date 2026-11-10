#!/usr/bin/env python3
"""Fail-closed DreamX-World-5B AR dose-16 StrideWorld runner.

This runner intentionally invokes the pinned upstream AR-forcing entry point
without modifying DreamX sampling, camera scale, speed, or CFG.  The optional
Lab color correction and chunk-relative pose rewrite are disabled: a fixed-seed
audit showed that the former full-strength combination progressively reduced
contrast and Laplacian sharpness, so the benchmark keeps the native global
output for reproducibility.
StrideWorld dose is lowered only to the legal AR temporal grid: one dose is four
pixel-frame transitions. The native AR grid generates 18 latent frames / 69
pixel frames; Task2 scores the first 65 frames (16 units), leaving four tail
frames unscored. Dose checkpoints 1/2/3/5/8/16 are at frames 4/8/12/20/32/64.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from camera_geometry import camera_geometry, missing_camera_geometry


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "dreamx_world"
SOURCE_COMMIT = "a1f4c6e5e45600718e5236955f2e0702e53fc275"
CHECKPOINT_REPO = "GD-ML/DreamX-World-5B"
CHECKPOINT_REVISION = "67487c4a61466bb7166d30b7187dd465e0ac9f6c"
CHECKPOINT_SHA256 = "fba4fd99fe1955b3fd9b2fe452a8029c9a625dc42441b5a482481f877520ebef"
CHECKPOINT_SIZE = 21_132_560_104
BASE_ASSET_REPO = "stdstu123/Yume-5B-720P"
BASE_ASSET_REVISION = "b117c778405246a0b932d7ec5873b01a4840da3a"
SOURCE_SHA256 = {
    "inference_ar_forcing.py":
        "e83988c4886a5f6c7cf71f2689a757b919ba3a58b8c6eb578b5e9c43d9db8fcd",
    "utils/trajectory_processor.py":
        "cb51e9df7e95a93c241719cab71509213be1bd8fc69f690f2090f664ef009e2d",
    "pipeline/pipeline_causal_camera.py":
        "f9b74c99dbcd6eb1e2c9907c39b1e196dbaa7d2377301dc89a7e07c33f5ba78f",
    "configs/dreamx-ar/config.json":
        "c618d961987187ba675823413431c388ac700ed721d877c623dc1c63e027af0e",
    "configs/dreamx-ar/causal_camera_forcing_5b.yaml":
        "b1c56bedb44c9c39891746e36e5a837f34ce920c3aaad9907cbe28558fb7bc1c",
    "configs/dreamx-ar/default_config.yaml":
        "0d48b43ba9e2d6c4a96c5af1681f779d82479d228921a920787b8d1b948050f8",
}
BASE_ASSET_SHA256 = {
    "models_t5_umt5-xxl-enc-bf16.pth":
        "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
    "Wan2.2_VAE.pth":
        "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36",
    "google/umt5-xxl/special_tokens_map.json":
        "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d",
    "google/umt5-xxl/spiece.model":
        "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",
    "google/umt5-xxl/tokenizer.json":
        "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b",
    "google/umt5-xxl/tokenizer_config.json":
        "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05",
}
BASE_ASSET_SIZE = {
    "models_t5_umt5-xxl-enc-bf16.pth": 11_361_920_418,
    "Wan2.2_VAE.pth": 2_818_839_170,
    "google/umt5-xxl/special_tokens_map.json": 6_623,
    "google/umt5-xxl/spiece.model": 4_548_313,
    "google/umt5-xxl/tokenizer.json": 16_837_417,
    "google/umt5-xxl/tokenizer_config.json": 61_728,
}
# Keep these historical names as the v6 compatibility surface.  Existing
# smoke tooling and tests patch them when constructing a synthetic frozen
# campaign, so do not silently repoint them at v7.
ASSIGNMENT_SHA256 = "66265546a7eca132fd43643199772aa56dcaaa0b2bab1fc8f1a911f22b5902fb"
INPUTS_SHA256 = "1455cb8f0fe948385e79468d437722c54168125447d048186b0a126fed859006"
V7_ASSIGNMENT_SHA256 = (
    "21c2e8050f2b1e76c091402a33cc0fd9311a167e5c9c3db537240389173ff277"
)
V7_INPUTS_SHA256 = (
    "4251822a0a5cf2584c5059319e3f10d4712e800a831dd7df046fa8865a1d30fb"
)
ACTION_TOKEN = {
    "MOVE_FORWARD": "w",
    "MOVE_BACKWARD": "s",
    "MOVE_LEFT": "a",
    "MOVE_RIGHT": "d",
    "YAW_LEFT": "j",
    "YAW_RIGHT": "l",
}
CHECKPOINT_DOSES = (1, 2, 3, 5, 8, 16)
CHECKPOINT_FRAME_INDICES = (4, 8, 12, 20, 32, 64)
NUM_OUTPUT_LATENT_FRAMES = 18
NUM_OUTPUT_PIXEL_FRAMES = 69
FORMAL_OUTPUT_PIXEL_FRAMES = 65
FPS = 16
NATIVE_SPEED = 1.5
TRANSLATION_PER_PIXEL_FRAME = 0.075
YAW_DEGREES_PER_PIXEL_FRAME = 1.5
COLOR_CORRECTION_STRENGTH = 0.0
CHUNK_RELATIVE = False


class ContractError(RuntimeError):
    """Raised before inference when immutable DreamX evidence disagrees."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, size: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if size is not None and path.stat().st_size != size:
        raise ContractError(f"file size mismatch: {path}")


def validate_hf_metadata(path: Path, revision: str, sha256: str) -> None:
    require_file(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or lines[0] != revision or lines[1] != sha256:
        raise ContractError(f"Hugging Face metadata mismatch: {path}")


def validate_source(native_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(native_root), "rev-parse", "HEAD"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise ContractError(f"DreamX source commit mismatch: {commit}")
    dirty = subprocess.run(
        ["git", "-C", str(native_root), "status", "--porcelain"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    if dirty:
        raise ContractError("DreamX native checkout must be clean")
    evidence = []
    for relative, expected in SOURCE_SHA256.items():
        path = native_root / relative
        require_file(path)
        actual = file_sha256(path)
        if actual != expected:
            raise ContractError(f"DreamX source SHA256 mismatch: {relative}")
        evidence.append({"path": str(path), "sha256": actual})
    return {"commit": commit, "files": evidence}


def validate_assets(checkpoint: Path, base_assets: Path) -> dict[str, Any]:
    require_file(checkpoint, CHECKPOINT_SIZE)
    validate_hf_metadata(
        checkpoint.parent / ".cache/huggingface/download/model.safetensors.metadata",
        CHECKPOINT_REVISION, CHECKPOINT_SHA256,
    )
    checkpoint_actual = file_sha256(checkpoint)
    if checkpoint_actual != CHECKPOINT_SHA256:
        raise ContractError("DreamX AR checkpoint SHA256 mismatch")

    revision_metadata = (
        base_assets / ".cache/huggingface/download/.gitattributes.metadata"
    )
    require_file(revision_metadata)
    if revision_metadata.read_text(encoding="utf-8").splitlines()[0] != BASE_ASSET_REVISION:
        raise ContractError("Wan2.2 base-asset snapshot revision mismatch")
    asset_evidence = []
    for relative, expected in BASE_ASSET_SHA256.items():
        path = base_assets / relative
        require_file(path, BASE_ASSET_SIZE[relative])
        actual = file_sha256(path)
        if actual != expected:
            raise ContractError(f"Wan2.2 base-asset SHA256 mismatch: {relative}")
        asset_evidence.append({
            "path": str(path), "byte_size": path.stat().st_size, "sha256": actual,
        })
    result = {
        "checkpoint": {
            "repo_id": CHECKPOINT_REPO, "revision": CHECKPOINT_REVISION,
            "path": str(checkpoint), "byte_size": checkpoint.stat().st_size,
            "sha256": checkpoint_actual, "variant": "AR-720p",
        },
        "base_assets": {
            "source_repo_id": BASE_ASSET_REPO,
            "source_snapshot_revision": BASE_ASSET_REVISION,
            "read_role": "Wan2.2 T5 encoder, UMT5 tokenizer, and Wan2.2 VAE only",
            "files": asset_evidence,
        },
    }


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"top-level JSON must be an object: {path}")
    return value


def resolve_frozen_campaign(
    assignment_path: Path,
    inputs_path: Path,
) -> dict[str, str]:
    """Identify one of the two immutable atomic campaigns.

    The legacy aliases remain mutable only for the existing synthetic CPU
    fixtures.  The production v7 pair is separately pinned so accepting v7
    cannot weaken or reinterpret the already-deployed v6 contract.
    """

    assignment_sha = file_sha256(assignment_path)
    inputs_sha = file_sha256(inputs_path)
    # StrideWorld v9 materializes a hash-bound assignment/input pair from the
    # frozen campaign. Accept that exact derived pair only when launched with
    # the bound controller campaign SHA; legacy v6/v7 aliases remain pinned.
    if (assignment_path.name == f"atomic_assignment.seed{os.environ['SW_TASK2_SEED']}.json"
            and inputs_path.name == "generation_inputs.v9.json"
            and os.environ.get("SW_CAMPAIGN_SHA256", "")
            and len(os.environ["SW_CAMPAIGN_SHA256"]) == 64):
        assignment = read_json(assignment_path)
        if (assignment.get("inputs_sha256") == inputs_sha
                and assignment_sha == os.environ["SW_TASK2_ASSIGNMENT_SHA256"]
                and inputs_sha == os.environ["SW_TASK2_INPUTS_SHA256"]):
            return {"campaign_id": "strideworld_v9_materialized", "assignment_sha256": assignment_sha, "inputs_sha256": inputs_sha}
    supported = (
        ("v6_inverse4_seed0", ASSIGNMENT_SHA256, INPUTS_SHA256),
        ("v7_full6_seed0", V7_ASSIGNMENT_SHA256, V7_INPUTS_SHA256),
    )
    for campaign_id, expected_assignment, expected_inputs in supported:
        if assignment_sha == expected_assignment and inputs_sha == expected_inputs:
            return {
                "campaign_id": campaign_id,
                "assignment_sha256": assignment_sha,
                "inputs_sha256": inputs_sha,
            }
    if assignment_sha not in {row[1] for row in supported}:
        raise ContractError("frozen assignment SHA256 mismatch")
    raise ContractError("frozen assignment/input pair mismatch")


def compile_case(
    action: str,
    scene_id: str,
    assignment_path: Path,
    inputs_path: Path,
    data_root: Path,
) -> dict[str, Any]:
    if action not in ACTION_TOKEN:
        raise ContractError(f"unsupported DreamX action: {action}")
    campaign = resolve_frozen_campaign(assignment_path, inputs_path)
    assignment = read_json(assignment_path)
    inputs = read_json(inputs_path)
    if assignment.get("inputs_sha256") != campaign["inputs_sha256"]:
        raise ContractError("assignment/input-manifest binding mismatch")
    seed_policy = assignment.get("seed_policy", {})
    if seed_policy.get("kind") != "fixed_across_campaign" or seed_policy.get("seed") != int(os.environ["SW_TASK2_SEED"]):
        raise ContractError("campaign seed must equal SW_TASK2_SEED")
    seed = int(os.environ["SW_TASK2_SEED"])
    if seed not in (0, 1, 2):
        raise ContractError("v10 Task2 runner supports seeds 0, 1, and 2")
    scene_rows = [row for row in assignment.get("scenes", []) if row.get("scene_id") == scene_id]
    if len(scene_rows) != 1:
        raise ContractError(f"scene must occur exactly once in assignment: {scene_id}")
    scene = scene_rows[0]
    if action not in scene.get("task2_actions", []):
        raise ContractError(f"action is not assigned to {scene_id}: {action}")
    input_rows = [row for row in inputs.get("inputs", []) if row.get("scene_id") == scene_id]
    if len(input_rows) != 1:
        raise ContractError(f"scene must occur exactly once in inputs: {scene_id}")
    input_row = input_rows[0]
    if input_row.get("case_id") not in (None, scene.get("case_id")):
        raise ContractError(f"assignment/input case mismatch for {scene_id}")
    state_ids = scene.get("state_ids", [])
    if not isinstance(state_ids, list) or len(state_ids) != 1:
        raise ContractError(f"scene must bind exactly one state: {scene_id}")
    if input_row.get("state_id") not in (None, state_ids[0]):
        raise ContractError(f"assignment/input state mismatch for {scene_id}")
    uri = input_row.get("image_uri", "")
    if not isinstance(uri, str) or not uri.startswith("data://"):
        raise ContractError(f"invalid data URI for {scene_id}")
    image = (data_root / uri.removeprefix("data://")).resolve()
    root = data_root.resolve()
    if root not in image.parents:
        raise ContractError("resolved image escapes data root")
    require_file(image)
    actual_image_sha = file_sha256(image)
    if actual_image_sha != input_row.get("image_sha256"):
        raise ContractError(f"first-frame SHA256 mismatch for {scene_id}")
    prompt = input_row.get("scene_prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ContractError(f"scene prompt is empty for {scene_id}")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != input_row.get("prompt_sha256"):
        raise ContractError(f"scene prompt SHA256 mismatch for {scene_id}")
    if NUM_OUTPUT_LATENT_FRAMES % 3:
        raise ContractError("DreamX AR latent count must be divisible by 3")
    if (NUM_OUTPUT_LATENT_FRAMES - 1) * 4 + 1 != NUM_OUTPUT_PIXEL_FRAMES:
        raise ContractError("DreamX latent/pixel frame formula mismatch")
    probe_case_id = f"{scene_id}__task2__{action.lower()}__d16__seed{seed}"
    result = {
        "seed": seed,
        "task_id": probe_case_id,
        "probe_case_id": probe_case_id,
        "case_id": scene["case_id"],
        "scene_id": scene_id,
        "state_id": state_ids[0],
        "frozen_campaign": campaign,
        "image_path": str(image),
        "image_sha256": actual_image_sha,
        "caption": prompt,
        "action_seq": [ACTION_TOKEN[action]],
        "action_speed_list": [1],
    }
    geometry = camera_geometry(
        input_row, output_pixel_frames=NUM_OUTPUT_PIXEL_FRAMES,
        output_latent_frames=NUM_OUTPUT_LATENT_FRAMES,
        checkpoint_pixels=list(CHECKPOINT_FRAME_INDICES),
        data_root=data_root, state_id=state_ids[0],
    )
    # Keep geometry out of the official native payload when source metadata is
    # absent; real campaign manifests that bind camera.json carry this field.
    if geometry["status"] == "READY":
        result["camera_geometry"] = geometry
    return result


def parse_fraction(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def native_case_payload(compiled: dict[str, Any]) -> dict[str, Any]:
    """Keep the upstream DreamX JSON byte-shape compatible with v6."""

    return {
        key: value for key, value in compiled.items()
        if key not in {"frozen_campaign", "camera_geometry", "seed"}
    }


def probe_video(path: Path) -> dict[str, Any]:
    if shutil.which("ffprobe"):
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
                "-of", "json", str(path),
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        stream = json.loads(result.stdout)["streams"][0]
        frames = int(stream.get("nb_read_frames") or stream.get("nb_frames"))
        fps = parse_fraction(stream["avg_frame_rate"])
        width, height = int(stream["width"]), int(stream["height"])
        duration = float(stream["duration"])
        probe_backend = "ffprobe"
    else:
        import av

        with av.open(str(path)) as container:
            video_stream = container.streams.video[0]
            width, height = video_stream.width, video_stream.height
            fps = float(video_stream.average_rate)
            frames = sum(1 for _ in container.decode(video=0))
        duration = frames / fps
        probe_backend = "pyav_decode"
    if frames != NUM_OUTPUT_PIXEL_FRAMES:
        raise ContractError(f"DreamX output frame mismatch: {frames}")
    if abs(fps - FPS) > 1e-6:
        raise ContractError(f"DreamX output FPS mismatch: {fps}")
    if (width, height) != (1280, 704):
        raise ContractError("DreamX output resolution mismatch")
    return {
        "path": str(path), "sha256": file_sha256(path), "frames": frames,
        "fps": fps, "width": width, "height": height,
        "duration_seconds": duration, "probe_backend": probe_backend,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=sorted(ACTION_TOKEN))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-assets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    native_root = args.native_root.resolve()
    checkpoint = args.checkpoint.resolve()
    base_assets = args.base_assets.resolve()
    assignment_path = args.assignment.resolve()
    inputs_path = args.inputs.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    compiled = compile_case(
        args.action, args.scene_id, assignment_path, inputs_path, data_root,
    )
    image = Path(compiled["image_path"])
    evidence = {
        "schema_version": "strideworld-dreamx-ar-d16-record-v2",
        "model_id": MODEL_ID,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "probe_case_id": compiled["probe_case_id"],
        "case_id": compiled["case_id"],
        "scene_id": compiled["scene_id"],
        "state_id": compiled["state_id"],
        "task": "TASK2",
        "action": args.action,
        "seed": compiled["seed"],
        "frozen_campaign": {
            "assignment_path": str(assignment_path),
            "assignment_sha256": compiled["frozen_campaign"]["assignment_sha256"],
            "inputs_path": str(inputs_path),
            "inputs_sha256": compiled["frozen_campaign"]["inputs_sha256"],
            "campaign_id": compiled["frozen_campaign"]["campaign_id"],
        },
        "source": validate_source(native_root),
        "assets": validate_assets(checkpoint, base_assets),
        "input": {
            "path": str(image), "sha256": compiled["image_sha256"],
            "prompt": compiled["caption"],
        },
        "condition_image_sha256": compiled["image_sha256"],
        "camera_geometry": compiled.get(
            "camera_geometry",
            missing_camera_geometry(output_pixel_frames=NUM_OUTPUT_PIXEL_FRAMES,
                                    output_latent_frames=NUM_OUTPUT_LATENT_FRAMES,
                                    checkpoint_pixels=list(CHECKPOINT_FRAME_INDICES)),
        ),
        "native_inference": {
            "entrypoint": "inference_ar_forcing.py",
            "conditioning": "PRoPE action_seq",
            "action_token": ACTION_TOKEN[args.action],
            "action_speed_list_semantics": "single-segment frame-budget ratio",
            "native_speed": NATIVE_SPEED,
            "translation_per_pixel_frame": TRANSLATION_PER_PIXEL_FRAME,
            "yaw_degrees_per_pixel_frame": YAW_DEGREES_PER_PIXEL_FRAME,
            "num_output_latent_frames": NUM_OUTPUT_LATENT_FRAMES,
            "num_output_pixel_frames": NUM_OUTPUT_PIXEL_FRAMES,
            "num_frame_per_ar_block": 3,
            "fps": FPS,
            "sample_height_width": [704, 1280],
            "denoising_step_list": [1000, 750, 500, 250],
            "guidance_scale": 3.0,
            "timestep_shift": 5.0,
            "weight_dtype": "bfloat16",
            "color_correction_strength": COLOR_CORRECTION_STRENGTH,
            "chunk_relative": CHUNK_RELATIVE,
            "postprocess_policy": "disabled_to_preserve_native_global_output",
            "camera_pose_reference": "global_first_frame",
        },
        "strideworld_dose": {
            "dose_unit": "four native pixel-frame transitions",
            "held_dose": 16,
            "checkpoint_doses": list(CHECKPOINT_DOSES),
            "checkpoint_frame_indices": list(CHECKPOINT_FRAME_INDICES),
            "formal_output_pixel_frames": FORMAL_OUTPUT_PIXEL_FRAMES,
            "native_output_pixel_frames": NUM_OUTPUT_PIXEL_FRAMES,
            "realized_seconds_scored": FORMAL_OUTPUT_PIXEL_FRAMES / FPS,
            "unscored_tail_frames": NUM_OUTPUT_PIXEL_FRAMES - FORMAL_OUTPUT_PIXEL_FRAMES,
            "no_tail_padding": False,
        },
        "status": "PREFLIGHT_PASS",
    }
    if args.preflight_only and args.finalize_existing:
        raise ContractError("preflight-only and finalize-existing are mutually exclusive")
    if args.preflight_only:
        print(json.dumps(evidence, indent=2, ensure_ascii=False))
        return 0

    case_dir = output_root / compiled["task_id"]
    input_json = case_dir / "dreamx_input.json"
    native_compiled = native_case_payload(compiled)
    expected_video = case_dir / (
        f"{compiled['task_id']}_{image.parent.name}_{image.stem}.mp4"
    )
    if args.finalize_existing:
        require_file(input_json)
        if json.loads(input_json.read_text(encoding="utf-8")) != [native_compiled]:
            raise ContractError("existing DreamX smoke input JSON mismatch")
        planned_record = case_dir / "generation_record.planned.json"
        require_file(planned_record)
        planned = json.loads(planned_record.read_text(encoding="utf-8"))
        if planned.get("probe_case_id") != compiled["probe_case_id"]:
            raise ContractError("planned DreamX smoke record identity mismatch")
        evidence["execution"] = planned.get("execution")
        canonical_video = case_dir / "raw_generated.mp4"
        if not canonical_video.exists():
            shutil.copy2(expected_video, canonical_video)
        evidence["output"] = probe_video(canonical_video)
        evidence["generated_video_sha256"] = evidence["output"]["sha256"]
        evidence["status"] = "COMPLETED_MACHINE_VALIDATED_G3_PENDING"
        record = case_dir / "generation_record.json"
        record.write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        print(json.dumps({"status": evidence["status"], "record": str(record)}, indent=2))
        return 0

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise ContractError("set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    if case_dir.exists() and any(case_dir.iterdir()):
        raise ContractError(f"refusing non-empty smoke directory: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=False)
    input_json.write_text(
        json.dumps([native_compiled], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable, str(native_root / "inference_ar_forcing.py"),
        "--config_path", str(native_root / "configs/dreamx-ar/causal_camera_forcing_5b.yaml"),
        "--model_name", str(base_assets),
        "--transformer_path", str(native_root / "configs/dreamx-ar"),
        "--base_checkpoint_path", str(checkpoint),
        "--data_path", str(input_json),
        "--output_folder", str(case_dir),
        "--num_output_frames", str(NUM_OUTPUT_LATENT_FRAMES),
        "--fps", str(FPS), "--seed", str(compiled["seed"]),
        "--color_correction_strength", str(COLOR_CORRECTION_STRENGTH),
    ]
    evidence["execution"] = {
        "cuda_visible_devices": visible, "python": sys.executable,
        "command": command, "input_json": str(input_json),
    }
    planned_record = case_dir / "generation_record.planned.json"
    planned_record.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    environment = os.environ.copy()
    compat = ROOT / "scripts/compat/dreamx"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(compat) + (
        f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else ""
    )
    subprocess.run(command, cwd=native_root, env=environment, check=True)
    require_file(expected_video)
    canonical_video = case_dir / "raw_generated.mp4"
    shutil.copy2(expected_video, canonical_video)
    if file_sha256(expected_video) != file_sha256(canonical_video):
        raise ContractError("canonical video copy differs from native DreamX output")
    evidence["output"] = probe_video(canonical_video)
    evidence["generated_video_sha256"] = evidence["output"]["sha256"]
    evidence["status"] = "COMPLETED_MACHINE_VALIDATED_G3_PENDING"
    record = case_dir / "generation_record.json"
    record.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": evidence["status"], "record": str(record)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
