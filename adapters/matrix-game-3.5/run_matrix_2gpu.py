#!/usr/bin/env python3
"""Execute pinned Matrix public CLI inputs, one independent record per GPU."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import av

from validate_model_inputs import canonical_sha, sha, rows, validate_row
from validate_worldplay_outputs import atomic_json

ROOT = Path(__file__).resolve().parents[1]
MODEL = "matrix_game_35_distilled_fp"
SOURCE = ROOT / "sources/Matrix-Game-3.5"


def now():
    return datetime.now(timezone.utc).isoformat()


def video_info(path: Path, row: dict) -> dict:
    with av.open(str(path)) as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError("expected one video stream")
        source = streams[0]
        rate = source.average_rate or source.base_rate or source.guessed_rate
        stream = {
            "codec_name": source.codec_context.name,
            "width": int(source.codec_context.width),
            "height": int(source.codec_context.height),
            "nb_read_frames": str(sum(1 for _ in container.decode(video=0))),
            "r_frame_rate": str(rate),
        }
    frames = int(stream["nb_read_frames"])
    if (stream["width"], stream["height"]) != (1280, 704):
        raise ValueError(f"Matrix video geometry mismatch: {stream}")
    if stream["codec_name"] != "h264" or Fraction(stream["r_frame_rate"]) != 16:
        raise ValueError(f"Matrix video encoding mismatch: {stream}")
    # The pinned rollout emits 21 latent frames/block, plus its initial anchor.
    # Wan VAE temporal decode is 4 * (latent_count - 1) + 1.
    if frames != row["raw_frame_count"]:
        raise ValueError(f"expected {row['raw_frame_count']} frames, got {frames}")
    if row["formal_frame_count"] > frames:
        raise ValueError("formal evaluation window exceeds video")
    return stream


def prepare(input_root: Path, output: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((input_root / "MANIFEST.json").read_text())
    recorded = manifest.pop("build_sha256")
    if canonical_sha(manifest) != recorded:
        raise ValueError("input build identity mismatch")
    manifest["build_sha256"] = recorded
    if manifest["future_rgb_used"] is not False or not manifest["source_validation"]:
        raise ValueError("requires source-validated, first-frame-only inputs")
    cases = rows(input_root / MODEL / "cases.jsonl")
    if len(cases) != manifest["models"][MODEL] or len({r["record_id"] for r in cases}) != len(cases):
        raise ValueError("input record count/identity mismatch")
    if len({r["group_id"] for r in cases}) != manifest["group_count"]:
        raise ValueError("input group count mismatch")
    receipts = {}
    checked = set()
    for row in cases:
        validate_row(MODEL, row)
        native = input_root / MODEL / "receipts" / f"{row['record_id']}.json"
        receipt = json.loads(native.read_text())
        identity = receipt.pop("adapter_input_sha256")
        if canonical_sha(receipt) != identity:
            raise ValueError("input receipt hash mismatch")
        if receipt.pop("model_id") != MODEL or receipt.pop("schema_version") != "strideworld-model-native-input-v1":
            raise ValueError("input receipt model mismatch")
        if receipt != row:
            raise ValueError("input row/receipt mismatch")
        for key, hash_key in (("condition_image", "condition_image_sha256"), ("prompt_path", "prompt_sha256")):
            p = Path(row[key])
            if p not in checked:
                if sha(p) != row[hash_key]:
                    raise ValueError(f"input hash mismatch: {p}")
                checked.add(p)
        receipts[row["record_id"]] = sha(native)
    checkpoint = ROOT / "models/Matrix-Game-3.5-Distilled/first-person.safetensors"
    expected_checkpoint = (checkpoint.parent / "SHA256SUMS").read_text().split()[0]
    if sha(checkpoint) != expected_checkpoint:
        raise ValueError("distilled checkpoint hash mismatch")
    source_files = {
        str(p.relative_to(SOURCE)): sha(p)
        for p in sorted(SOURCE.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".yaml")
        and not any(x.startswith(".") or x == "__pycache__" for x in p.relative_to(SOURCE).parts)
    }
    plan = {
        "schema_version": "strideworld-matrix-execution-plan-v1",
        "model_id": MODEL, "input_root": str(input_root),
        "input_manifest_sha256": sha(input_root / "MANIFEST.json"),
        "cases_sha256": sha(input_root / MODEL / "cases.jsonl"),
        "input_build_sha256": recorded, "campaign_sha256": manifest["campaign_sha256"],
        "record_count": len(cases), "group_count": manifest["group_count"],
        "source_commit": "fbf7def0693ae14f745ba35bf2a26215d4ef991d",
        "source_files_sha256": source_files, "checkpoint_sha256": expected_checkpoint,
        "input_receipt_sha256": receipts,
        "runner_sha256": sha(Path(__file__)),
    }
    plan["plan_sha256"] = canonical_sha(plan)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "EXECUTION_PLAN.json"
    if target.exists() and json.loads(target.read_text()) != plan:
        raise ValueError("output already belongs to a different execution plan")
    atomic_json(target, plan)
    return plan, cases


def validate_result(result_path: Path, row: dict, plan: dict) -> dict:
    result = json.loads(result_path.read_text())
    for key, expected in {
        "status": "SUCCEEDED", "model_id": MODEL, "record_id": row["record_id"],
        "plan_sha256": plan["plan_sha256"], "record_sha256": row["record_sha256"],
        "input_receipt_sha256": plan["input_receipt_sha256"][row["record_id"]],
        "metric_frame_slice_half_open": row["metric_frame_slice_half_open"],
    }.items():
        if result.get(key) != expected:
            raise ValueError(f"result {key} mismatch")
    for name in ("video", "native_generation_record", "executed_action_record"):
        relative = Path(result[f"{name}_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe artifact path")
        artifact = result_path.parent / relative
        if not artifact.is_file() or not artifact.stat().st_size or sha(artifact) != result[f"{name}_sha256"]:
            raise ValueError(f"artifact mismatch: {artifact}")
    video_info(result_path.parent / result["video_path"], row)
    return result


def run_shard(index: int, cases: list[dict], plan: dict, output: Path) -> None:
    for row in cases[index::2]:
        directory = output / "records" / row["record_id"]
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "generation_record.json"
        if result_path.exists():
            validate_result(result_path, row, plan)
            print(f"GPU{index} SKIP_ACCEPTED {row['record_id']}", flush=True)
            continue
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(index), "OMP_NUM_THREADS": "8"})
        substitutions = {
            "${MATRIX_DISTILLED_CHECKPOINT}": str(ROOT / "models/Matrix-Game-3.5-Distilled/first-person.safetensors"),
            "${WAN22_TI2V_5B}": str(ROOT / "models/Wan2.2-TI2V-5B"),
            "${DA3_NESTED_GIANT_LARGE}": str(ROOT / "models/DA3NESTED-GIANT-LARGE-1.1"),
            "${OUTPUT_ROOT}": str(directory),
        }
        command = list(row["run_argv"])
        command[0] = sys.executable
        for i, arg in enumerate(command):
            for key, value in substitutions.items():
                arg = arg.replace(key, value)
            if "${" in arg:
                raise ValueError(f"unresolved placeholder: {arg}")
            command[i] = arg
        started = now()
        print(f"GPU{index} START {row['record_id']} {started}", flush=True)
        with (directory / "inference.log").open("a", buffering=1) as log:
            code = subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            atomic_json(directory / "FAILURE.json", {"created_at": now(), "returncode": code, "argv": command})
            raise RuntimeError(f"GPU{index} {row['record_id']} failed, exit {code}")
        video = Path(command[command.index("--output") + 1])
        stream = video_info(video, row)
        native = directory / "native_generation_record.json"
        atomic_json(native, {
            "schema_version": "strideworld-matrix-native-generation-v1", "argv": command,
            "started_at": started, "finished_at": now(), "gpu_worker_index": index,
            "hostname": socket.gethostname(), "plan_sha256": plan["plan_sha256"],
            "video_stream": stream, "seed": row["seed"], "num_blocks": row["num_blocks"],
            "schedule": [1000, 667, 333], "student_cfg_scale": 3.0,
        })
        actions = directory / "executed_action_record.json"
        atomic_json(actions, {
            "schema_version": "strideworld-matrix-executed-actions-v1", "record_id": row["record_id"],
            "actions": row["actions"], "camera_path": row["camera_path"],
            "camera_sha256": row["camera_sha256"], "config_sha256": row["config_sha256"],
            "future_rgb_used": False,
        })
        result = {
            "schema_version": "strideworld-generation-result-v1", "status": "SUCCEEDED",
            "model_id": MODEL, "record_id": row["record_id"], "group_id": row["group_id"],
            "record_sha256": row["record_sha256"], "plan_sha256": plan["plan_sha256"],
            "campaign_sha256": plan["campaign_sha256"],
            "input_receipt_sha256": plan["input_receipt_sha256"][row["record_id"]],
            "metric_frame_slice_half_open": row["metric_frame_slice_half_open"],
            "formal_frame_count": row["formal_frame_count"], "native_frame_count": int(stream["nb_read_frames"]),
        }
        for name, path in (("video", video), ("native_generation_record", native), ("executed_action_record", actions)):
            result[f"{name}_path"] = path.name
            result[f"{name}_sha256"] = sha(path)
        atomic_json(result_path, result)
        print(f"GPU{index} SUCCEEDED {row['record_id']} {now()}", flush=True)


def telemetry(output: Path, stop: threading.Event) -> None:
    with (output / "GPU_TELEMETRY.csv").open("a", buffering=1) as log:
        while not stop.is_set():
            sample = subprocess.run([
                "nvidia-smi", "--query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True)
            log.write(sample.stdout or f"{now()},ERROR,{sample.stderr.strip()}\n")
            stop.wait(30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan, cases = prepare(args.input_root.resolve(), args.output_root.resolve())
    if args.validate_only:
        print(json.dumps({"status": "VALIDATED_CPU_ONLY", "plan_sha256": plan["plan_sha256"], "records": len(cases)}))
        return 0
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
    if len(devices) != 2 or any("GPU" not in x for x in devices):
        raise RuntimeError(f"requires exactly two GPU GPUs: {devices}")
    stop = threading.Event()
    sampler = threading.Thread(target=telemetry, args=(args.output_root, stop), daemon=True)
    sampler.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_shard, index, cases, plan, args.output_root) for index in range(2)]
            for future in futures:
                future.result()
    finally:
        stop.set()
        sampler.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
