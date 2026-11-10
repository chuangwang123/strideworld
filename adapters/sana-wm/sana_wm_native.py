#!/usr/bin/env python3
"""Run variable-horizon SANA-WM final requests with one model load.

This worker reuses the SHA-pinned, manually accepted streaming deployment but
removes its old fixed ``w-24``/25-frame restriction.  Every request carries an
explicit official trajectory DSL and a legal ``8k+1`` native horizon.  The
model is loaded once, request-local solver/VAE state is reset between videos,
and no RGB clips are concatenated.

The worker is intentionally an execution backend, not a trajectory sampler.
The outer StrideWorld worker must freeze and validate every DSL before invoking
this file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import time
from types import ModuleType
from typing import Any


SCHEMA = "strideworld-sana-wm-streaming-final-request-shard-v1"
RECORD_SCHEMA = "strideworld-sana-wm-streaming-final-worker-v1"
REQUEST_SCHEMA = "strideworld-sana-wm-streaming-final-request-v1"
SOURCE_REVISION = "6c3aeea94945692a902df8db25327eba040b5d91"
WEIGHT_REVISION = "c7694938c854b1fcc29468e9514f1ea2d0c4ba8e"
STAGE1_TEXT_ENCODER_REVISION = "569d9809d0c8b6722d4d31b5a77a2ec7a400650a"
PINNED_NATIVE_WORKER_SHA256 = (
    "538294dd26744ade374b056853b8dff1ceb7468624b45e3c9765edfed151d3e5"
)
SEED = int(os.environ["SW_TASK2_SEED"])
if SEED not in (0, 1, 2):
    raise RuntimeError("v10 native requires seed 0, 1, or 2")
FPS = 16
# The LTX-2 VAE alone accepts 8k+1 frames, but the accepted streaming
# deployment freezes refiner_block_size=3 as well.  Its actual joint horizon
# is therefore 8*3*k+1.  Keep only the two horizons needed by final rather
# than advertising 8k+1 values which fail inside the refiner chunk runner.
LEGAL_NATIVE_FRAMES = (25, 49, 73)
TOKEN_PATTERN = re.compile(r"^(none|w|s|j|l|a|d)-([1-9][0-9]*)$")
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class WorkerContractError(RuntimeError):
    """Raised before inference when a final request is not exact."""


@dataclass(frozen=True)
class Request:
    request_id: str
    action_dsl: str
    num_frames: int
    formal_frame_count: int
    image: Path
    image_sha256: str
    prompt_file: Path
    prompt_sha256: str
    intrinsics_file: Path
    intrinsics_sha256: str
    output_dir: Path
    name: str
    benchmark_json: Path
    control: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size <= 0:
        raise WorkerContractError(f"{label} is missing/unsafe: {resolved}")
    return resolved


def require_hash(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise WorkerContractError(f"{label} must be a lowercase SHA256")
    return text


def validate_bound_file(row: dict[str, Any], key: str, sha_key: str) -> tuple[Path, str]:
    path = require_file(Path(str(row.get(key, ""))), key)
    expected = require_hash(row.get(sha_key), sha_key)
    if sha256(path) != expected:
        raise WorkerContractError(f"{key} SHA256 mismatch: {path}")
    return path, expected


def validate_dsl(value: Any, num_frames: int, formal_frame_count: int) -> str:
    if not isinstance(value, str) or not value:
        raise WorkerContractError("action_dsl must be non-empty")
    runs = value.split(",")
    parsed: list[tuple[str, int]] = []
    for run in runs:
        match = TOKEN_PATTERN.fullmatch(run)
        if match is None:
            raise WorkerContractError(f"invalid SANA trajectory DSL run: {run!r}")
        parsed.append((match.group(1), int(match.group(2))))
    if sum(count for _, count in parsed) != num_frames - 1:
        raise WorkerContractError("trajectory DSL ticks do not match num_frames - 1")
    noop_indices = [index for index, (token, _) in enumerate(parsed) if token == "none"]
    if noop_indices and noop_indices != [len(parsed) - 1]:
        raise WorkerContractError("NO_OP is allowed only as terminal unscored padding")
    formal_action_ticks = sum(count for token, count in parsed if token != "none")
    if formal_action_ticks <= 0:
        raise WorkerContractError("trajectory DSL contains no scored action ticks")
    if formal_frame_count != formal_action_ticks + 1:
        raise WorkerContractError(
            "formal_frame_count must equal condition frame plus scored action ticks"
        )
    terminal_padding_ticks = sum(count for token, count in parsed if token == "none")
    if terminal_padding_ticks != num_frames - formal_frame_count:
        raise WorkerContractError("terminal NO_OP count differs from unscored horizon padding")
    return value


def load_requests(path: Path) -> tuple[dict[str, Any], tuple[Request, ...]]:
    manifest_path = require_file(path, "request manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise WorkerContractError(f"invalid request manifest: {error}") from error
    expected = {
        "schema_version": SCHEMA,
        "source_revision": SOURCE_REVISION,
        "weight_revision": WEIGHT_REVISION,
        "stage1_text_encoder_revision": STAGE1_TEXT_ENCODER_REVISION,
        "seed": SEED,
        "fps": FPS,
    }
    if not isinstance(payload, dict):
        raise WorkerContractError("request manifest root must be an object")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise WorkerContractError(f"request manifest mismatch for {key}")
    rows = payload.get("requests")
    if not isinstance(rows, list) or not rows:
        raise WorkerContractError("request manifest must contain requests")
    requests: list[Request] = []
    seen_ids: set[str] = set()
    seen_outputs: set[Path] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise WorkerContractError(f"request {index} is not an object")
        request_id = str(row.get("request_id", ""))
        if not SAFE_ID.fullmatch(request_id) or request_id in seen_ids:
            raise WorkerContractError(f"invalid/duplicate request_id: {request_id!r}")
        seen_ids.add(request_id)
        num_frames = row.get("num_frames")
        if isinstance(num_frames, bool) or num_frames not in LEGAL_NATIVE_FRAMES:
            raise WorkerContractError(
                f"request {request_id} num_frames must be one of {LEGAL_NATIVE_FRAMES}"
            )
        formal_frames = row.get("formal_frame_count")
        if (
            isinstance(formal_frames, bool)
            or not isinstance(formal_frames, int)
            or not 2 <= formal_frames <= num_frames
        ):
            raise WorkerContractError(f"request {request_id} formal frame count is invalid")
        action_dsl = validate_dsl(row.get("action_dsl"), num_frames, formal_frames)
        image, image_sha = validate_bound_file(row, "image", "image_sha256")
        prompt, prompt_sha = validate_bound_file(row, "prompt_file", "prompt_sha256")
        intrinsics, intrinsics_sha = validate_bound_file(
            row, "intrinsics_file", "intrinsics_sha256",
        )
        output_dir = Path(str(row.get("output_dir", "")))
        if not output_dir.is_absolute():
            raise WorkerContractError(f"request {request_id} output_dir must be absolute")
        output_dir = output_dir.resolve()
        if output_dir in seen_outputs:
            raise WorkerContractError(f"duplicate output_dir: {output_dir}")
        seen_outputs.add(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise WorkerContractError(f"refusing non-empty output_dir: {output_dir}")
        name = str(row.get("name", ""))
        if not SAFE_ID.fullmatch(name):
            raise WorkerContractError(f"request {request_id} has unsafe output name")
        benchmark_json = Path(str(row.get("benchmark_json", ""))).resolve()
        if benchmark_json.parent != output_dir:
            raise WorkerContractError("benchmark_json must be inside output_dir")
        control = row.get("control")
        if not isinstance(control, dict):
            raise WorkerContractError(f"request {request_id} control is missing")
        expected_control = {
            "action_dsl": action_dsl,
            "native_num_frames": num_frames,
            "formal_frame_count": formal_frames,
            "terminal_unscored_noop_ticks": num_frames - formal_frames,
            "terminal_unscored_coast_ticks": num_frames - formal_frames,
            "terminal_tail_policy": "official_none_key_release_coast_not_static_noop",
        }
        for key, expected_value in expected_control.items():
            if control.get(key) != expected_value:
                raise WorkerContractError(
                    f"request {request_id} control mismatch for {key}"
                )
        requests.append(Request(
            request_id=request_id,
            action_dsl=action_dsl,
            num_frames=num_frames,
            formal_frame_count=formal_frames,
            image=image,
            image_sha256=image_sha,
            prompt_file=prompt,
            prompt_sha256=prompt_sha,
            intrinsics_file=intrinsics,
            intrinsics_sha256=intrinsics_sha,
            output_dir=output_dir,
            name=name,
            benchmark_json=benchmark_json,
            control=control,
        ))
    return payload, tuple(requests)


def load_pinned_worker(path: Path) -> ModuleType:
    worker_path = require_file(path, "pinned SANA persistent worker")
    observed = sha256(worker_path)
    if observed != PINNED_NATIVE_WORKER_SHA256:
        raise WorkerContractError(
            "pinned SANA persistent worker SHA256 mismatch: "
            f"expected={PINNED_NATIVE_WORKER_SHA256} actual={observed}"
        )
    sys.path.insert(0, str(worker_path.parent))
    spec = importlib.util.spec_from_file_location("strideworld_pinned_sana_worker", worker_path)
    if spec is None or spec.loader is None:
        raise WorkerContractError("cannot import pinned SANA persistent worker")
    module = importlib.util.module_from_spec(spec)
    # ``dataclasses`` resolves string annotations through sys.modules while
    # the pinned worker is imported.  importlib does not insert modules made
    # via module_from_spec automatically.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The frozen worker constructs RefinerSettings(seed=SEED). Override only
    # its in-memory seed before build_pipeline; pinned source bytes stay intact.
    module.SEED = SEED
    return module


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    module = load_pinned_worker(args.pinned_worker)
    if getattr(module, "REFINER_BLOCK_SIZE", None) != 3:
        raise WorkerContractError("pinned SANA refiner_block_size drifted from 3")
    if any((frames - 1) % (8 * module.REFINER_BLOCK_SIZE) for frames in LEGAL_NATIVE_FRAMES):
        raise WorkerContractError("declared native horizons violate VAE/refiner alignment")
    manifest, requests = load_requests(args.request_manifest.resolve())
    source_root = args.source_root.resolve()
    weights_root = args.weights_root.resolve()
    text_root = args.stage1_text_encoder_root.resolve()
    sentencepiece_root = args.sentencepiece_runtime_root.resolve()
    for path, label in (
        (source_root, "source root"),
        (weights_root, "weights root"),
        (text_root, "stage1 text encoder root"),
        (sentencepiece_root, "SentencePiece root"),
    ):
        if not path.is_dir():
            raise WorkerContractError(f"{label} is missing: {path}")

    # Reuse the exact accepted source/weight/runtime audits and pipeline
    # constructor from the pinned worker; only request horizons are generalized.
    import run_sana_wm_streaming_prefix_smoke as contract  # type: ignore

    contract.validate_static_contract()
    source_audit = contract.validate_source(source_root)
    persistent_source_audit = module.validate_persistent_source_contract(source_root)
    weights_audit = contract.validate_weights(weights_root)
    text_audit = contract.validate_stage1_text_encoder(text_root)
    sentencepiece_audit = contract.validate_sentencepiece_runtime(sentencepiece_root)
    deployment = contract.validate_deployment_manifest(args.deployment_manifest.resolve())
    sentencepiece_site = Path(sentencepiece_audit["site_packages"])
    if str(sentencepiece_site) not in sys.path:
        raise WorkerContractError("isolated SentencePiece site-packages is absent from sys.path")

    record_path = args.worker_record.resolve()
    if record_path.exists():
        raise WorkerContractError(f"refusing existing worker record: {record_path}")
    started = time.perf_counter()
    native, torch, pipeline, build_audit = module.build_pipeline(
        source_root, weights_root, text_root, sentencepiece_site,
    )
    record: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA,
        "status": "RUNNING",
        "formal_score_eligible": False,
        "batch_authorization": False,
        "source": source_audit,
        "persistent_source_contract": persistent_source_audit,
        "weights": weights_audit,
        "stage1_text_encoder": text_audit,
        "sentencepiece_runtime": sentencepiece_audit,
        "deployment": deployment,
        "request_manifest": {
            "path": str(args.request_manifest.resolve()),
            "sha256": sha256(args.request_manifest.resolve()),
            "request_count": len(requests),
            "contract": manifest,
        },
        "pipeline": build_audit,
        "seed_binding": {"generator_seed": SEED, "refiner_seed": module.SEED,
                         "pinned_worker_in_memory_seed_override": True},
        "model_load_seconds": time.perf_counter() - started,
        "requests": [],
    }
    write_json_atomic(record_path, record)

    previous_prompt: str | None = None
    for request_index, request in enumerate(requests):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        prompt = request.prompt_file.read_text(encoding="utf-8", errors="strict").strip()
        contract.validate_prompt(prompt)
        cache_audit = module.clear_prompt_caches_if_changed(
            pipeline, previous_prompt, prompt,
        )
        state_audit = module.restore_request_state(pipeline, torch.device("cuda"))
        if cache_audit["prompt_changed"]:
            torch.cuda.empty_cache()
        image = native.Image.open(request.image).convert("RGB")
        c2w = native.action_string_to_c2w(
            request.action_dsl,
            translation_speed=module.TRANSLATION_SPEED,
            rotation_speed_deg=module.ROTATION_SPEED_DEG,
        )
        if c2w.shape != (request.num_frames, 4, 4):
            raise WorkerContractError(
                f"request {request.request_id} trajectory shape mismatch: {c2w.shape}"
            )
        cropped, src_size, resized_size, crop_offset = native.resize_and_center_crop(image)
        intrinsics = native.load_intrinsics(request.intrinsics_file, request.num_frames)
        intrinsics_vec4 = native.transform_intrinsics_for_crop(
            intrinsics, src_size, resized_size, crop_offset,
        )
        params = native.GenerationParams(
            num_frames=request.num_frames,
            fps=FPS,
            cfg_scale=module.CFG_SCALE,
            flow_shift=module.FLOW_SHIFT,
            seed=SEED,
            negative_prompt="",
            sampling_algo="self_forcing",
            num_cached_blocks=2,
            sink_token=True,
            num_frame_per_block=module.NUM_FRAME_PER_BLOCK,
            denoising_step_list=list(module.DENOISING_STEP_LIST),
        )
        output_path = request.output_dir / f"{request.name}_streaming.mp4"
        request_started = time.perf_counter()
        result, solver_audit = module.generate_with_isolated_model_patch(
            pipeline,
            cropped,
            prompt,
            c2w,
            intrinsics_vec4,
            params,
            output_path=output_path,
            streaming_crf=18,
            streaming_preset="medium",
            streaming_encoder="libx264",
            output_mode="mp4",
            profile_cuda=False,
        )
        request_seconds = time.perf_counter() - request_started
        if int(result["n_pixel_frames"]) != request.num_frames - 1:
            raise WorkerContractError(
                f"request {request.request_id} native frame count mismatch"
            )
        benchmark = {
            "schema_version": REQUEST_SCHEMA,
            "request_id": request.request_id,
            "request_index": request_index,
            "seed": SEED,
            "action_dsl": request.action_dsl,
            "num_frames": request.num_frames,
            "formal_frame_count": request.formal_frame_count,
            "control": request.control,
            "model_load_count_for_worker": 1,
            "request_state_reset": state_audit,
            "solver_state_reset": solver_audit,
            "prompt_cache": cache_audit,
            "input_bindings": {
                "image_sha256": request.image_sha256,
                "prompt_sha256": request.prompt_sha256,
                "intrinsics_sha256": request.intrinsics_sha256,
            },
            "output_path": str(output_path),
            "output_sha256": sha256(output_path),
            "n_pixel_frames": int(result["n_pixel_frames"]),
            "wall_seconds": float(result["wall_seconds"]),
            "request_end_to_end_seconds": request_seconds,
            "frames_per_second": float(result["frames_per_second"]),
            "realtime_factor": float(result["realtime_factor"]),
        }
        write_json_atomic(request.benchmark_json, benchmark)
        record["requests"].append(benchmark)
        write_json_atomic(record_path, record)
        previous_prompt = prompt

    record["status"] = "COMPLETED_PENDING_OUTER_VIDEO_VALIDATION"
    record["total_wall_seconds"] = time.perf_counter() - started
    record["pipeline"]["pipeline_load_count"] = 1
    write_json_atomic(record_path, record)
    print(json.dumps({
        "status": record["status"],
        "request_count": len(requests),
        "pipeline_load_count": 1,
        "worker_record": str(record_path),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned-worker", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--weights-root", required=True, type=Path)
    parser.add_argument("--stage1-text-encoder-root", required=True, type=Path)
    parser.add_argument("--sentencepiece-runtime-root", required=True, type=Path)
    parser.add_argument("--deployment-manifest", required=True, type=Path)
    parser.add_argument("--request-manifest", required=True, type=Path)
    parser.add_argument("--worker-record", required=True, type=Path)
    try:
        run(parser.parse_args())
    except (WorkerContractError, OSError, RuntimeError) as error:
        print(f"ERROR [sana_wm_native]: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
