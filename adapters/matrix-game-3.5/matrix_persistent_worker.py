#!/usr/bin/env python3
"""Experimental one-GPU Matrix worker with a persistent Wan model.

Launch this through the official single-GPU Accelerate config. Outputs are
prototype artifacts, not accepted campaign generation records; compare them
with the public CLI before adopting this path for a full evaluation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import pickle
import shutil
import sys
import time

from validate_model_inputs import canonical_sha, rows, sha, validate_row
from validate_worldplay_outputs import atomic_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sources/Matrix-Game-3.5"
MODEL = "matrix_game_35_distilled_fp"
PER_CASE_ARGS = frozenset({
    "dataset_index_path", "val_dataset_index_path", "dataset_base_path",
    "camera_params_path", "depth_path", "vipe_prompt_path",
    "memory_latent_cache_dir", "dataset_cache_dir",
    "num_validation_blocks", "validation_seed",
})
SUBSTITUTIONS = {
    "${MATRIX_DISTILLED_CHECKPOINT}": ROOT / "models/Matrix-Game-3.5-Distilled/first-person.safetensors",
    "${WAN22_TI2V_5B}": ROOT / "models/Wan2.2-TI2V-5B",
    "${DA3_NESTED_GIANT_LARGE}": ROOT / "models/DA3NESTED-GIANT-LARGE-1.1",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_cases(all_cases: list[dict], record_ids: list[str], shard: int | None, limit: int | None) -> list[dict]:
    if record_ids and shard is not None:
        raise ValueError("choose --record-id or --shard-index, not both")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate --record-id")
    by_id = {row["record_id"]: row for row in all_cases}
    if len(by_id) != len(all_cases):
        raise ValueError("duplicate record ID in input cases")
    if record_ids:
        missing = set(record_ids) - by_id.keys()
        if missing:
            raise ValueError(f"unknown record IDs: {sorted(missing)}")
        cases = [by_id[record_id] for record_id in record_ids]
    elif shard is not None:
        cases = all_cases[shard::2]
    else:
        raise ValueError("explicit --record-id or --shard-index is required")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        cases = cases[:limit]
    return cases


def resolved_argv(row: dict, output: Path) -> list[str]:
    command = list(row["run_argv"])
    if command[:2] != ["python", "infer_distilled.py"]:
        raise ValueError("Matrix input must invoke the pinned public CLI")
    if command.count("--output") != 1:
        raise ValueError("expected exactly one --output argument")
    command[0] = sys.executable
    for index, arg in enumerate(command):
        for placeholder, path in SUBSTITUTIONS.items():
            arg = arg.replace(placeholder, str(path))
        arg = arg.replace("${OUTPUT_ROOT}", str(output.parent))
        if "${" in arg:
            raise ValueError(f"unresolved placeholder: {arg}")
        command[index] = arg
    expected = output.parent / f"{row['record_id']}.mp4"
    if Path(command[command.index("--output") + 1]) != expected or output != expected:
        raise ValueError("output path does not match input record")
    return command


def model_signature(args) -> str:
    """Reject any config drift beyond per-case inputs, seed, and block count."""
    return canonical_sha({key: str(value) for key, value in vars(args).items() if key not in PER_CASE_ARGS})


def seed_like_public_runner(args, accelerator, torch, np, derive_seed) -> None:
    args.rank = accelerator.process_index
    args.dataset_pass_id = 0
    args.start_epoch = 0
    args.start_global_step = 0
    seed = derive_seed(args.seed, "trainer", accelerator.process_index, 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)


def capture_rng(torch, np) -> tuple:
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        np.random.get_state(),
        random.getstate(),
    )


def restore_rng(state: tuple, torch, np) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])
    np.random.set_state(state[2])
    random.setstate(state[3])


def rng_digests(torch, np) -> dict:
    state = capture_rng(torch, np)
    digest = lambda data: hashlib.sha256(data).hexdigest()
    return {
        "torch_cpu": digest(state[0].cpu().numpy().tobytes()),
        "torch_cuda": [digest(value.cpu().numpy().tobytes()) for value in (state[1] or [])],
        "numpy": digest(pickle.dumps(state[2], protocol=4)),
        "python": digest(pickle.dumps(state[3], protocol=4)),
    }


@contextmanager
def case_dataloader_lifetime(accelerator):
    """Release only this call's prepared loaders, retaining the resident model.

    Accelerate 1.14.0 appends each prepared DataLoader to this registry and has
    no per-loader public cleanup API. Its free_memory() also clears models,
    optimizers and schedulers, so restoring the registry prefix is narrower.
    """
    registry = getattr(accelerator, "_dataloaders", None)
    if not isinstance(registry, list):
        raise RuntimeError("unsupported Accelerate dataloader registry")
    prefix = tuple(registry)
    counts = {"before": len(prefix)}
    try:
        yield counts
    finally:
        if accelerator._dataloaders is not registry or any(
            first is not second for first, second in zip(registry[:len(prefix)], prefix)
        ) or len(registry) < len(prefix):
            raise RuntimeError("Accelerate dataloader registry changed unexpectedly")
        counts["prepared"] = len(registry) - len(prefix)
        del registry[len(prefix):]
        counts["after"] = len(registry)


def assert_inference_model_state(model) -> None:
    if model.training or getattr(model, "step_count", None) != 0:
        raise RuntimeError("persistent model training/step state changed")


@contextmanager
def cached_depth_estimator(causal_inference, enabled: bool):
    """Optional A/B-only cache; each case still gets a fresh frustum handler."""
    if not enabled:
        yield
        return
    original = causal_inference._init_registration_estimator
    cached = []
    cached_device = []

    def get_estimator(args, device=None):
        if not cached:
            cached.append(original(args, device=device))
            cached_device.append(str(device))
            print("[persistent] depth estimator loaded once", flush=True)
        elif str(device) != cached_device[0]:
            raise RuntimeError("cached depth estimator device changed")
        return cached[0]

    causal_inference._init_registration_estimator = get_estimator
    try:
        yield
    finally:
        causal_inference._init_registration_estimator = original


def parse_cli(row: dict, output: Path):
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    import infer_distilled

    command = resolved_argv(row, output)
    cli = infer_distilled.build_parser().parse_args(command[2:])
    infer_distilled.validate_cli(cli)
    if (
        cli.image.resolve() != Path(row["condition_image"]).resolve()
        or cli.camera.resolve() != Path(row["camera_path"]).resolve()
        or cli.prompt_file is None
        or cli.prompt_file.resolve() != Path(row["prompt_path"]).resolve()
        or cli.checkpoint.resolve() != SUBSTITUTIONS["${MATRIX_DISTILLED_CHECKPOINT}"].resolve()
        or Path(cli.wan_dir).resolve() != SUBSTITUTIONS["${WAN22_TI2V_5B}"].resolve()
        or Path(cli.tokenizer_dir).resolve() != (SUBSTITUTIONS["${WAN22_TI2V_5B}"] / "google/umt5-xxl").resolve()
        or Path(cli.da3_dir).resolve() != SUBSTITUTIONS["${DA3_NESTED_GIANT_LARGE}"].resolve()
        or cli.accelerate_config.resolve() != infer_distilled.DEFAULT_ACCELERATE_CONFIG.resolve()
        or cli.camera_convention != "c2w"
        or cli.keep_workspace
        or cli.dry_run
    ):
        raise ValueError("public CLI arguments differ from the pinned input record")
    if int(row["num_blocks"]) != infer_distilled.load_inference_config(cli.config).num_blocks:
        raise ValueError("case block count differs from public config")
    for path_key, hash_key in (
        ("condition_image", "condition_image_sha256"),
        ("camera_path", "camera_sha256"),
        ("prompt_path", "prompt_sha256"),
    ):
        if sha(Path(row[path_key])) != row[hash_key]:
            raise ValueError(f"input hash changed: {row[path_key]}")
    if sha(cli.config) != row["config_sha256"]:
        raise ValueError("config hash changed")
    return cli, command


def validate_input_receipts(input_root: Path, all_cases: list[dict], cases: list[dict]) -> None:
    manifest = json.loads((input_root / "MANIFEST.json").read_text())
    identity = manifest.pop("build_sha256")
    if canonical_sha(manifest) != identity:
        raise ValueError("input manifest hash mismatch")
    if (
        manifest["source_validation"] is not True
        or manifest["future_rgb_used"] is not False
        or manifest["models"][MODEL] != len(all_cases)
        or manifest["group_count"] != len({row["group_id"] for row in all_cases})
    ):
        raise ValueError("input manifest contract mismatch")
    for row in cases:
        validate_row(MODEL, row)
        path = input_root / MODEL / "receipts" / f"{row['record_id']}.json"
        receipt = json.loads(path.read_text())
        recorded = receipt.pop("adapter_input_sha256")
        if canonical_sha(receipt) != recorded:
            raise ValueError(f"input receipt hash mismatch: {row['record_id']}")
        if receipt.pop("schema_version") != "strideworld-model-native-input-v1" or receipt.pop("model_id") != MODEL:
            raise ValueError(f"input receipt model mismatch: {row['record_id']}")
        if receipt != row:
            raise ValueError(f"input row/receipt mismatch: {row['record_id']}")


def build_case_runtime(cli, case_id: str):
    import infer as base_infer
    import infer_distilled
    from distilled_config import load_inference_config
    from examples.wanvideo.pipeline.mosaic.causal_config import build_runtime_args

    config = load_inference_config(cli.config)
    os.environ["DA3_MODEL_PATH"] = base_infer.resolve_weight("da3", cli.da3_dir)
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    wan_dir = Path(base_infer.resolve_weight("wan", cli.wan_dir))
    tokenizer_dir = Path(base_infer.resolve_weight("tokenizer", cli.tokenizer_dir))
    tmp_root = SOURCE / ".cache/infer_runs" / f"persistent_{os.getpid()}_{case_id}"
    if tmp_root.exists():
        raise FileExistsError(f"stale case workspace must be inspected: {tmp_root}")
    workspace = Path(base_infer.build_input_workspace(
        infer_distilled._workspace_args(cli, config.num_blocks), str(tmp_root),
    ))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE) + os.pathsep + env.get("PYTHONPATH", "")
    index = Path(base_infer.build_index(str(workspace), sys.executable, env))
    args = build_runtime_args(
        config, checkpoint=cli.checkpoint, dataset_index=index,
        workspace=workspace, wan_dir=wan_dir, tokenizer_dir=tokenizer_dir,
        memory_cache_dir=tmp_root / "memory_latents",
    )
    args.dataset_cache_dir = str(tmp_root / "dataset_cache")
    return args, tmp_root


def run(args) -> int:
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    all_cases = rows(args.input_root / MODEL / "cases.jsonl")
    cases = selected_cases(all_cases, args.record_id, args.shard_index, args.limit)
    if not cases:
        raise ValueError("no selected Matrix cases")
    validate_input_receipts(args.input_root, all_cases, cases)
    outputs = []
    for row in cases:
        output = args.output_root / "records" / row["record_id"] / f"{row['record_id']}.mp4"
        cli, command = parse_cli(row, output)
        outputs.append((row, output, cli, command))
    if args.preflight_only:
        print(json.dumps({"status": "PREFLIGHT_ONLY", "records": [row["record_id"] for row in cases]}))
        return 0

    import accelerate
    import numpy as np
    import torch
    if str(SOURCE / "tools") not in sys.path:
        sys.path.insert(0, str(SOURCE / "tools"))
    from diffsynth.core.data.unified_dataset import _derive_seed
    from examples.wanvideo.pipeline.mosaic import causal_inference
    from examples.wanvideo.pipeline.mosaic.datasets import build_mosaic_validation_dataset
    from run_matrix_2gpu import video_info
    from run_distilled_inference import build_model

    if torch.cuda.device_count() != 1 or "GPU" not in torch.cuda.get_device_name(0):
        raise RuntimeError("persistent worker requires exactly one visible GPU")
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=1,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if accelerator.num_processes != 1 or accelerator.mixed_precision != "bf16":
        raise RuntimeError("requires official single-process BF16 Accelerate config")

    model = None
    first_signature = None
    after_model_rng = None
    with cached_depth_estimator(causal_inference, args.reuse_depth_estimator):
        for row, output, cli, command in outputs:
            directory = output.parent
            directory.mkdir(parents=True, exist_ok=True)
            receipt = directory / "PERSISTENT_PROTOTYPE.json"
            if output.exists() or receipt.exists():
                raise FileExistsError(f"prototype output already exists: {directory}")
            tmp_root = None
            dataset = None
            runtime = None
            started = time.monotonic()
            try:
                runtime, tmp_root = build_case_runtime(cli, row["record_id"])
                seed_like_public_runner(runtime, accelerator, torch, np, _derive_seed)
                signature = model_signature(runtime)
                if first_signature is not None and signature != first_signature:
                    raise ValueError("model/static inference configuration changed between cases")
                dataset = build_mosaic_validation_dataset(runtime, train_dataset=None)
                if len(dataset) != 1:
                    raise RuntimeError(f"expected one input sample, got {len(dataset)}")
                loaded_this_case = model is None
                load_started = time.monotonic()
                if model is None:
                    model = build_model(runtime, accelerator)
                    model.eval()
                    first_signature = signature
                    after_model_rng = capture_rng(torch, np)
                    print(f"[persistent] Wan model loaded once; signature={signature}", flush=True)
                model_load_seconds = time.monotonic() - load_started if loaded_this_case else 0.0
                restore_rng(after_model_rng, torch, np)
                prepared_seconds = time.monotonic() - started
                assert_inference_model_state(model)
                rng_before = rng_digests(torch, np)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                inference_started = time.monotonic()
                with case_dataloader_lifetime(accelerator) as loader_counts:
                    causal_inference.run_distilled_inference(accelerator, dataset, model, output, runtime)
                torch.cuda.synchronize()
                inference_seconds = time.monotonic() - inference_started
                assert_inference_model_state(model)
                dataset = None
                runtime = None
                gc.collect()
                stream = video_info(output, row)
                atomic_json(receipt, {
                    "status": "PROTOTYPE_SUCCEEDED", "record_id": row["record_id"],
                    "record_sha256": row["record_sha256"], "argv": command,
                    "model_signature_sha256": signature,
                    "depth_estimator_reused": args.reuse_depth_estimator,
                    "video_path": output.name, "video_sha256": sha(output),
                    "video_stream": stream, "prepared_seconds": prepared_seconds,
                    "model_loaded_this_case": loaded_this_case,
                    "model_load_seconds": model_load_seconds,
                    "inference_seconds": inference_seconds,
                    "total_seconds": time.monotonic() - started,
                    "pre_inference_rng_sha256": rng_before,
                    "accelerate_version": accelerate.__version__,
                    "accelerate_dataloader_counts": loader_counts,
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "after_case_gpu_allocated_bytes": torch.cuda.memory_allocated(),
                    "after_case_gpu_reserved_bytes": torch.cuda.memory_reserved(),
                    "finished_at": now(),
                })
                print(f"[persistent] SUCCEEDED {row['record_id']} {now()}", flush=True)
                shutil.rmtree(tmp_root)
            except Exception as exc:
                atomic_json(directory / "FAILURE.json", {
                    "record_id": row["record_id"], "error": repr(exc),
                    "workspace": str(tmp_root) if tmp_root else None, "at": now(),
                })
                raise
            finally:
                dataset = None
                runtime = None
                gc.collect()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record-id", action="append", default=[])
    parser.add_argument("--shard-index", type=int, choices=(0, 1))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--reuse-depth-estimator", action="store_true")
    args = parser.parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
