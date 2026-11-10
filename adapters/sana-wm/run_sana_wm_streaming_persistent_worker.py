#!/usr/bin/env python3
"""Run multiple exact SANA-WM streaming requests with one model load.

This worker imports the pinned upstream streaming pipeline without modifying
that checkout.  Model components and the compiled refiner stay resident for
the whole shard.  Per-request solvers, RNGs, refiner KV runners, causal-VAE
decoders, and output writers are still constructed by the upstream
``generate_streaming`` call.  Before every request we restore the VAE encoder
to CUDA and clear its request-local encode/decode caches.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


# These environment defaults must be installed before importing torch or Sana.
os.environ.setdefault("DISABLE_XFORMERS", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Network lookup is never an allowed fallback for a benchmark run.  These are
# assignments, not defaults, so a caller cannot silently re-enable the Hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


SOURCE_REVISION = "6c3aeea94945692a902df8db25327eba040b5d91"
WEIGHT_REVISION = "c7694938c854b1fcc29468e9514f1ea2d0c4ba8e"
STAGE1_TEXT_ENCODER_REPO = "Efficient-Large-Model/gemma-2-2b-it"
STAGE1_TEXT_ENCODER_REVISION = "569d9809d0c8b6722d4d31b5a77a2ec7a400650a"
SENTENCEPIECE_VERSION = "0.2.1"
CONFIG_RELATIVE = "configs/sana_wm/sana_wm_streaming_1600m_720p.yaml"
PIPELINE_RELATIVE = "inference_video_scripts/wm/inference_sana_wm.py"
DECODER_RELATIVE = "diffusion/model/ltx2/streaming_decoder.py"
SCHEMA = "strideworld-sana-wm-streaming-request-shard-v1"
RECORD_SCHEMA = "strideworld-sana-wm-streaming-persistent-worker-v3"
REQUEST_RECORD_SCHEMA = "strideworld-sana-wm-streaming-persistent-request-v3"
CLAIM_SCHEMA = "strideworld-sana-wm-streaming-shard-claim-v1"
RUNTIME_FINGERPRINT_SCHEMA = "strideworld-sana-wm-streaming-runtime-fingerprint-v1"
SHARED_AUDIT_SCHEMA = "strideworld-sana-wm-streaming-shared-audit-v1"
REQUEST_IDENTITY_SCHEMA = "strideworld-sana-wm-streaming-request-identity-v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
SEED = 0
FPS = 16
NUM_FRAMES = 25
ACTION_TICKS = 24
CFG_SCALE = 1.0
FLOW_SHIFT = 8.0
TRANSLATION_SPEED = 0.025
ROTATION_SPEED_DEG = 0.6
DENOISING_STEP_LIST = (1000, 960, 889, 727, 0)
NUM_FRAME_PER_BLOCK = 3
REFINER_BLOCK_SIZE = 3
REFINER_KV_MAX_FRAMES = 11
ACTIONS = {
    "MOVE_FORWARD": "w-24",
    "MOVE_BACKWARD": "s-24",
    "MOVE_LEFT": "j-24",
    "MOVE_RIGHT": "l-24",
    "YAW_LEFT": "a-24",
    "YAW_RIGHT": "d-24",
}
SMOKE_CAMPAIGN_ACTION = "SIX_ACTION_SMOKE"


class WorkerContractError(RuntimeError):
    """Raised before model loading when a persistent shard is not frozen."""


@dataclass(frozen=True)
class Request:
    request_id: str
    request_index: int
    action: str
    image: Path
    image_sha256: str
    prompt_file: Path
    prompt_sha256: str
    intrinsics_file: Path
    intrinsics_sha256: str
    output_dir: Path
    name: str
    benchmark_json: Path
    request_manifest: Path
    request_manifest_sha256: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def read_process_identity(pid: int) -> dict[str, Any]:
    """Bind a worker claim to this Linux process rather than a reusable PID."""

    process_root = Path("/proc") / str(pid)
    try:
        stat_text = (process_root / "stat").read_text(encoding="utf-8")
        cmdline_raw = (process_root / "cmdline").read_bytes()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8",
        ).strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError) as exc:
        raise WorkerContractError("cannot bind worker process identity") from exc
    right_paren = stat_text.rfind(")")
    if right_paren < 0:
        raise WorkerContractError("invalid worker /proc stat")
    fields = stat_text[right_paren + 2:].split()
    if len(fields) <= 19 or fields[0] == "Z" or not cmdline_raw:
        raise WorkerContractError("worker process is not live")
    command = [
        token.decode("utf-8", errors="surrogateescape")
        for token in cmdline_raw.rstrip(b"\0").split(b"\0")
    ]
    return {
        "pid": pid,
        "boot_id": boot_id,
        "start_time_ticks": int(fields[19]),
        "cmdline": command,
        "command_sha256": canonical_sha256(command),
    }


@contextmanager
def claim_shard(args: argparse.Namespace):
    """Acquire the per-campaign/action lease before any audit or model load."""

    campaign_sha = require_hash(
        args.campaign_manifest_sha256, "campaign manifest SHA256",
    )
    manifest_sha = require_hash(
        args.request_manifest_sha256, "request manifest SHA256",
    )
    token = str(args.launch_token)
    if len(token) != 64 or any(char not in "0123456789abcdef" for char in token):
        raise WorkerContractError("launch token must be 64 lowercase hex characters")
    if args.campaign_action not in (*ACTIONS, SMOKE_CAMPAIGN_ACTION):
        raise WorkerContractError("campaign action is invalid")
    gpu_uuid = str(args.physical_gpu_uuid)
    if not re.fullmatch(r"GPU-[0-9a-f-]{36}", gpu_uuid):
        raise WorkerContractError("physical GPU UUID is invalid")
    global_fd = int(args.model_global_lease_fd)
    if global_fd < 3:
        raise WorkerContractError("model-global inherited lease FD is invalid")
    try:
        global_stat = os.fstat(global_fd)
        # Same inherited open-file description already owns the exclusive
        # flock; reasserting it is nonblocking and must succeed.
        fcntl.flock(global_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as exc:
        raise WorkerContractError(
            "worker did not inherit the model-global campaign lease"
        ) from exc
    lease_path = args.shard_lease.resolve()
    claim_path = args.claim_record.resolve()
    if lease_path.is_symlink() or claim_path.is_symlink():
        raise WorkerContractError("shard lease/claim must not be symlinked")
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        lease_path,
        os.O_RDWR | os.O_CREAT | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0),
        0o600,
    )
    handle = os.fdopen(descriptor, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkerContractError(
                f"another worker owns the shard lease: {lease_path}"
            ) from exc
        identity = read_process_identity(os.getpid())
        claim = {
            "schema_version": CLAIM_SCHEMA,
            "status": "CLAIMED_LEASE_HELD",
            "campaign_manifest_sha256": campaign_sha,
            "campaign_action": args.campaign_action,
            "request_manifest": str(args.request_manifest.resolve()),
            "request_manifest_sha256": manifest_sha,
            "physical_gpu_uuid": gpu_uuid,
            "launch_token": token,
            "shard_lease": str(lease_path),
            "worker_record": str(args.worker_record.resolve()),
            "model_global_lease": {
                "fd": global_fd,
                "device": global_stat.st_dev,
                "inode": global_stat.st_ino,
            },
            "process_identity": identity,
        }
        write_json_atomic(claim_path, claim)
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({
            "campaign_manifest_sha256": campaign_sha,
            "campaign_action": args.campaign_action,
            "launch_token": token,
            "claim_record": str(claim_path),
            "pid": os.getpid(),
        }, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield claim
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def validate_live_request_bindings(request: Request) -> dict[str, str]:
    """Re-hash every request input at the actual consumption/publish boundary."""

    bindings = {
        "request_manifest_sha256": sha256(request.request_manifest),
        "image_sha256": sha256(request.image),
        "prompt_sha256": sha256(request.prompt_file),
        "intrinsics_sha256": sha256(request.intrinsics_file),
    }
    expected = {
        "request_manifest_sha256": request.request_manifest_sha256,
        "image_sha256": request.image_sha256,
        "prompt_sha256": request.prompt_sha256,
        "intrinsics_sha256": request.intrinsics_sha256,
    }
    if bindings != expected:
        raise WorkerContractError(
            f"live request input drift: {request.request_id}"
        )
    return bindings


def build_runtime_fingerprint(
    args: argparse.Namespace, *, deployment: dict[str, Any],
    shared_audit: dict[str, Any], manifest_path: Path, manifest_sha: str,
) -> dict[str, Any]:
    worker = Path(__file__).resolve()
    fingerprint = {
        "schema_version": RUNTIME_FINGERPRINT_SCHEMA,
        "campaign_manifest_sha256": require_hash(
            args.campaign_manifest_sha256, "campaign manifest SHA256",
        ),
        "campaign_action": args.campaign_action,
        "physical_gpu_uuid": args.physical_gpu_uuid,
        "worker": {"path": str(worker), "sha256": sha256(worker)},
        "deployment": {
            "path": str(args.deployment_manifest.resolve()),
            "sha256": deployment.get("sha256"),
        },
        "shared_audit_artifact": {
            "path": str(args.shared_audit_artifact.resolve()),
            "sha256": shared_audit.get("sha256"),
        },
        "request_manifest": {
            "path": str(manifest_path), "sha256": manifest_sha,
        },
        "source": {
            "root": str(args.source_root.resolve()), "revision": SOURCE_REVISION,
        },
        "weights": {
            "root": str(args.weights_root.resolve()), "revision": WEIGHT_REVISION,
        },
        "stage1_text_encoder": {
            "root": str(args.stage1_text_encoder_root.resolve()),
            "revision": STAGE1_TEXT_ENCODER_REVISION,
        },
        "sentencepiece": {
            "root": str(args.sentencepiece_runtime_root.resolve()),
            "version": SENTENCEPIECE_VERSION,
        },
    }
    return fingerprint


def ensure_runtime_provenance(
    record_path: Path, runtime_fingerprint: dict[str, Any],
) -> dict[str, str]:
    provenance_path = record_path.with_suffix(record_path.suffix + ".provenance.json")
    payload = {
        "schema_version": "strideworld-sana-wm-streaming-worker-provenance-v1",
        "worker_record": str(record_path),
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": canonical_sha256(runtime_fingerprint),
    }
    if provenance_path.exists():
        if provenance_path.is_symlink() or json.loads(
            provenance_path.read_text(encoding="utf-8")
        ) != payload:
            raise WorkerContractError("worker runtime provenance drifted")
    else:
        write_json_atomic(provenance_path, payload)
    return {"path": str(provenance_path), "sha256": sha256(provenance_path)}


def validate_runtime_provenance(
    binding: dict[str, Any], record_path: Path,
    runtime_fingerprint: dict[str, Any],
) -> dict[str, str]:
    expected_path = record_path.with_suffix(record_path.suffix + ".provenance.json")
    if not isinstance(binding, dict) or binding.get("path") != str(expected_path):
        raise WorkerContractError("worker provenance path binding mismatch")
    provenance_path = require_absolute_file(expected_path, "worker provenance")
    digest = require_hash(binding.get("sha256"), "worker provenance SHA256")
    if sha256(provenance_path) != digest:
        raise WorkerContractError("worker provenance SHA256 mismatch")
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkerContractError("worker provenance is invalid JSON") from exc
    if payload != {
        "schema_version": "strideworld-sana-wm-streaming-worker-provenance-v1",
        "worker_record": str(record_path),
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": canonical_sha256(runtime_fingerprint),
    }:
        raise WorkerContractError("worker provenance content mismatch")
    return {"path": str(provenance_path), "sha256": digest}


def stat_payload(path: Path) -> dict[str, int | str]:
    """Return the immutable-file stat tuple bound by a shared full audit."""

    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise WorkerContractError(f"shared-audit file is missing or symlinked: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "byte_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "mode": stat.st_mode,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "uid": stat.st_uid,
        "gid": stat.st_gid,
        "nlink": stat.st_nlink,
    }


def stage1_required_file_contract() -> tuple[dict[str, tuple[int, str]], str]:
    """Load the small, deployment-bound stage-1 file contract.

    Loading the auditor does not read model payloads.  It gives workers the
    canonical path/size/SHA set that the shared artifact must cover, so a
    truncated or hand-crafted artifact cannot suppress any 10.48GB audit row.
    """

    auditor_path = Path(__file__).resolve().with_name(
        "audit_sana_stage1_text_encoder_snapshot.py"
    )
    if not auditor_path.is_file() or auditor_path.is_symlink():
        raise WorkerContractError("stage-1 auditor is missing or symlinked")
    spec = importlib.util.spec_from_file_location(
        "strideworld_sana_worker_stage1_contract", auditor_path,
    )
    if spec is None or spec.loader is None:
        raise WorkerContractError("cannot load stage-1 auditor contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = getattr(module, "REQUIRED_FILE_CONTRACT", None)
    manifest_name = getattr(module, "MANIFEST_NAME", None)
    if not isinstance(contract, dict) or not contract or not isinstance(
        manifest_name, str,
    ):
        raise WorkerContractError("stage-1 auditor contract is invalid")
    normalized: dict[str, tuple[int, str]] = {}
    for relative, raw in contract.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(raw, tuple)
            or len(raw) != 2
        ):
            raise WorkerContractError("stage-1 auditor file contract is invalid")
        size, digest = raw
        if not isinstance(size, int) or size <= 0:
            raise WorkerContractError("stage-1 auditor byte-size contract is invalid")
        normalized[relative] = (size, require_hash(
            digest, f"stage-1 auditor SHA256: {relative}",
        ))
    return normalized, manifest_name


def validate_shared_stage1_bindings(
    bindings_by_path: dict[str, dict[str, Any]], stage1_root: Path,
) -> dict[str, Any]:
    """Replace the repeated full hash pass without weakening its file set.

    Payload bytes are not reread: their artifact rows must carry the canonical
    size/SHA256 and the exact live stat tuple.  The small metadata and cached
    manifest are hashed live, and an inexpensive directory scan still rejects
    missing, extra or partial payloads.
    """

    contract, manifest_name = stage1_required_file_contract()
    root = stage1_root.resolve()
    expected_payloads = set(contract)
    actual_payloads = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(root).parts
        and path.name != manifest_name
    }
    if actual_payloads != expected_payloads:
        raise WorkerContractError("shared stage-1 payload set drift")
    if any(path.is_file() for path in root.rglob("*.incomplete")):
        raise WorkerContractError("shared stage-1 snapshot has incomplete payloads")

    payload_bytes = 0
    for relative, (expected_size, expected_sha) in contract.items():
        payload_path = (root / relative).resolve()
        payload_row = bindings_by_path.get(str(payload_path))
        if not isinstance(payload_row, dict):
            raise WorkerContractError(
                f"shared audit misses stage-1 payload binding: {relative}"
            )
        if (
            payload_row.get("byte_size") != expected_size
            or payload_row.get("content_sha256") != expected_sha
        ):
            raise WorkerContractError(
                f"shared stage-1 payload identity mismatch: {relative}"
            )
        payload_bytes += expected_size

        metadata_path = (
            root / ".cache" / "huggingface" / "download"
            / f"{relative}.metadata"
        ).resolve()
        metadata_row = bindings_by_path.get(str(metadata_path))
        if not isinstance(metadata_row, dict):
            raise WorkerContractError(
                f"shared audit misses stage-1 metadata binding: {relative}"
            )
        if metadata_row.get("content_sha256") != sha256(metadata_path):
            raise WorkerContractError(
                f"shared stage-1 metadata SHA256 mismatch: {relative}"
            )

    manifest_path = (root / manifest_name).resolve()
    manifest_row = bindings_by_path.get(str(manifest_path))
    if not isinstance(manifest_row, dict):
        raise WorkerContractError("shared audit misses stage-1 manifest binding")
    if manifest_row.get("content_sha256") != sha256(manifest_path):
        raise WorkerContractError("shared stage-1 manifest SHA256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkerContractError("shared stage-1 manifest is invalid JSON") from exc
    rows = manifest.get("files") if isinstance(manifest, dict) else None
    if (
        not isinstance(rows, list)
        or len(rows) != len(contract)
        or manifest.get("revision") != STAGE1_TEXT_ENCODER_REVISION
        or manifest.get("payload_file_count") != len(contract)
        or manifest.get("payload_bytes") != payload_bytes
    ):
        raise WorkerContractError("shared stage-1 manifest contract mismatch")
    indexed = {row.get("path"): row for row in rows if isinstance(row, dict)}
    if set(indexed) != expected_payloads:
        raise WorkerContractError("shared stage-1 manifest file set mismatch")
    for relative, (expected_size, expected_sha) in contract.items():
        row = indexed[relative]
        if row.get("byte_size") != expected_size or row.get("sha256") != expected_sha:
            raise WorkerContractError(
                f"shared stage-1 cached identity mismatch: {relative}"
            )
    return {
        "payload_file_count": len(contract),
        "payload_bytes": payload_bytes,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
    }


def validate_shared_audit_artifact(
    path: Path,
    expected_sha256: str,
    *,
    source_root: Path,
    weights_root: Path,
    stage1_text_encoder_root: Path,
    sentencepiece_runtime_root: Path,
    deployment_manifest: Path,
) -> dict[str, Any]:
    """Validate a once-computed full audit using hash-bound immutable stats.

    The artifact records the content SHA/Hub identity established by one full
    audit and the exact size/mtime/ctime/inode/device tuple for every bound
    file.  Workers still validate live source, weight-manifest, SentencePiece,
    deployment, request and model-loading contracts; only the repeated 10.48GB
    stage-1 rehash and Python/wheel audit are reused.
    """

    artifact_path = require_absolute_file(path, "shared audit artifact")
    expected = require_hash(expected_sha256, "shared audit artifact SHA256")
    actual = sha256(artifact_path)
    if actual != expected:
        raise WorkerContractError("shared audit artifact SHA256 mismatch")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkerContractError("shared audit artifact is invalid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SHARED_AUDIT_SCHEMA
        or payload.get("status") != "COMPLETE_EXACT_SHARED_AUDIT"
    ):
        raise WorkerContractError("shared audit artifact schema/status mismatch")
    roots = payload.get("roots")
    expected_roots = {
        "source_root": str(source_root.resolve()),
        "weights_root": str(weights_root.resolve()),
        "stage1_text_encoder_root": str(stage1_text_encoder_root.resolve()),
        "sentencepiece_runtime_root": str(sentencepiece_runtime_root.resolve()),
    }
    if roots != expected_roots:
        raise WorkerContractError("shared audit root binding mismatch")
    identity = payload.get("identity", {})
    if identity != {
        "source_revision": SOURCE_REVISION,
        "weight_revision": WEIGHT_REVISION,
        "stage1_text_encoder_revision": STAGE1_TEXT_ENCODER_REVISION,
        "sentencepiece_version": SENTENCEPIECE_VERSION,
    }:
        raise WorkerContractError("shared audit identity mismatch")
    deployment = payload.get("deployment", {})
    if (
        deployment.get("path") != str(deployment_manifest.resolve())
        or deployment.get("sha256") != sha256(deployment_manifest.resolve())
    ):
        raise WorkerContractError("shared audit deployment binding mismatch")
    if payload.get("python_executable") != str(Path(sys.executable).resolve()):
        raise WorkerContractError("shared audit Python interpreter mismatch")
    bindings = payload.get("file_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise WorkerContractError("shared audit contains no file bindings")
    if payload.get("file_binding_count") != len(bindings):
        raise WorkerContractError("shared audit binding count mismatch")
    if payload.get("file_bindings_sha256") != canonical_sha256(bindings):
        raise WorkerContractError("shared audit binding manifest SHA256 mismatch")
    seen: set[str] = set()
    bindings_by_path: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(bindings):
        if not isinstance(row, dict):
            raise WorkerContractError(f"shared audit binding {index} is invalid")
        path_text = str(row.get("path", ""))
        if path_text in seen:
            raise WorkerContractError(f"duplicate shared audit binding: {path_text}")
        seen.add(path_text)
        bindings_by_path[path_text] = row
        live = stat_payload(Path(path_text))
        for key, value in live.items():
            if row.get(key) != value:
                raise WorkerContractError(
                    f"shared audit stat drift for {path_text}: {key}"
                )
        content_sha = row.get("content_sha256")
        hub_identity = row.get("hub_content_identity")
        if content_sha is None and hub_identity is None:
            raise WorkerContractError(
                f"shared audit binding lacks content identity: {path_text}"
            )
        if content_sha is not None:
            require_hash(content_sha, f"shared audit content SHA256: {path_text}")
        if hub_identity is not None and not isinstance(hub_identity, str):
            raise WorkerContractError(
                f"shared audit Hub identity is invalid: {path_text}"
            )
    stage1_live = validate_shared_stage1_bindings(
        bindings_by_path, stage1_text_encoder_root,
    )
    audits = payload.get("audits")
    if not isinstance(audits, dict):
        raise WorkerContractError("shared audit result payload is missing")
    stage1 = audits.get("stage1_text_encoder")
    if (
        not isinstance(stage1, dict)
        or stage1.get("revision") != STAGE1_TEXT_ENCODER_REVISION
        or stage1.get("payload_file_count") != stage1_live["payload_file_count"]
        or stage1.get("payload_bytes") != stage1_live["payload_bytes"]
    ):
        raise WorkerContractError("shared stage-1 audit result mismatch")
    if (
        stage1_live["payload_file_count"] != stage1["payload_file_count"]
        or stage1_live["payload_bytes"] != stage1["payload_bytes"]
        or stage1_live["manifest"] != stage1.get("manifest")
        or stage1_live["manifest_sha256"] != stage1.get("manifest_sha256")
    ):
        raise WorkerContractError("shared stage-1 live/artifact mismatch")
    if audits.get("offline_tokenizer_preflight", {}).get(
        "result", {},
    ).get("status") != "PASS_OFFLINE_LOCAL_TOKENIZERS":
        raise WorkerContractError("shared offline tokenizer preflight is absent")
    return {
        "path": str(artifact_path),
        "sha256": actual,
        "status": payload["status"],
        "file_binding_count": len(bindings),
        "stage1_text_encoder": stage1,
        "python_runtime": audits.get("python_runtime"),
        "offline_tokenizer_preflight": audits.get("offline_tokenizer_preflight"),
    }


def require_hash(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise WorkerContractError(f"{label} must be a lowercase SHA256")
    return text


def require_absolute_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise WorkerContractError(f"{label} must be absolute: {path}")
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise WorkerContractError(f"{label} is missing or empty: {resolved}")
    return resolved


def validate_file_binding(row: dict[str, Any], path_key: str, sha_key: str) -> Path:
    path = require_absolute_file(Path(str(row.get(path_key, ""))), path_key)
    expected = require_hash(row.get(sha_key), sha_key)
    actual = sha256(path)
    if actual != expected:
        raise WorkerContractError(f"{path_key} SHA256 mismatch: {path}")
    return path


def request_identity(request: Request) -> dict[str, Any]:
    """Return the exact immutable request identity carried into final records."""

    return {
        "schema_version": REQUEST_IDENTITY_SCHEMA,
        "request_id": request.request_id,
        "request_index": request.request_index,
        "action": request.action,
        "action_dsl": ACTIONS[request.action],
        "seed": SEED,
        "image": {
            "path": str(request.image), "sha256": request.image_sha256,
        },
        "prompt_file": {
            "path": str(request.prompt_file), "sha256": request.prompt_sha256,
        },
        "intrinsics_file": {
            "path": str(request.intrinsics_file),
            "sha256": request.intrinsics_sha256,
        },
        "output_dir": str(request.output_dir),
        "name": request.name,
        "benchmark_json": str(request.benchmark_json),
        "request_manifest": {
            "path": str(request.request_manifest),
            "sha256": request.request_manifest_sha256,
        },
    }


def load_requests(
    path: Path, *, allow_existing_output_dirs: bool = False,
) -> tuple[dict[str, Any], tuple[Request, ...]]:
    manifest_path = require_absolute_file(path, "request manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkerContractError(f"invalid request manifest JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
        raise WorkerContractError("persistent request manifest schema mismatch")
    expected = {
        "source_revision": SOURCE_REVISION,
        "weight_revision": WEIGHT_REVISION,
        "stage1_text_encoder_revision": STAGE1_TEXT_ENCODER_REVISION,
        "seed": SEED,
        "fps": FPS,
        "num_frames": NUM_FRAMES,
        "native_action_ticks": ACTION_TICKS,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise WorkerContractError(f"request manifest mismatch for {key}")
    rows = payload.get("requests")
    if not isinstance(rows, list) or not rows:
        raise WorkerContractError("request manifest must contain at least one request")
    manifest_sha = sha256(manifest_path)
    requests: list[Request] = []
    seen_ids: set[str] = set()
    seen_outputs: set[Path] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise WorkerContractError(f"request {index} is not an object")
        request_id = str(raw.get("request_id", ""))
        if not SAFE_ID.fullmatch(request_id) or request_id in seen_ids:
            raise WorkerContractError(f"invalid or duplicate request_id: {request_id!r}")
        seen_ids.add(request_id)
        action = str(raw.get("action", ""))
        if action not in ACTIONS or raw.get("action_dsl") != ACTIONS.get(action):
            raise WorkerContractError(f"request {request_id} action mapping mismatch")
        image = validate_file_binding(raw, "image", "image_sha256")
        image_sha = require_hash(raw.get("image_sha256"), "image_sha256")
        prompt_file = validate_file_binding(raw, "prompt_file", "prompt_sha256")
        prompt_sha = require_hash(raw.get("prompt_sha256"), "prompt_sha256")
        intrinsics_file = validate_file_binding(
            raw, "intrinsics_file", "intrinsics_sha256",
        )
        intrinsics_sha = require_hash(
            raw.get("intrinsics_sha256"), "intrinsics_sha256",
        )
        output_dir = Path(str(raw.get("output_dir", "")))
        if not output_dir.is_absolute():
            raise WorkerContractError(f"request {request_id} output_dir must be absolute")
        output_dir = output_dir.resolve()
        if output_dir in seen_outputs:
            raise WorkerContractError(f"duplicate output_dir: {output_dir}")
        seen_outputs.add(output_dir)
        if (
            not allow_existing_output_dirs
            and output_dir.exists()
            and any(output_dir.iterdir())
        ):
            raise WorkerContractError(f"refusing non-empty output_dir: {output_dir}")
        name = str(raw.get("name", ""))
        if not SAFE_ID.fullmatch(name):
            raise WorkerContractError(f"request {request_id} has unsafe output name")
        benchmark_json = Path(str(raw.get("benchmark_json", "")))
        if not benchmark_json.is_absolute():
            raise WorkerContractError(
                f"request {request_id} benchmark_json must be absolute"
            )
        benchmark_json = benchmark_json.resolve()
        if benchmark_json.parent != output_dir:
            raise WorkerContractError(
                f"request {request_id} benchmark_json must be inside output_dir"
            )
        requests.append(Request(
            request_id=request_id,
            request_index=index,
            action=action,
            image=image,
            image_sha256=image_sha,
            prompt_file=prompt_file,
            prompt_sha256=prompt_sha,
            intrinsics_file=intrinsics_file,
            intrinsics_sha256=intrinsics_sha,
            output_dir=output_dir,
            name=name,
            benchmark_json=benchmark_json,
            request_manifest=manifest_path,
            request_manifest_sha256=manifest_sha,
        ))
    return payload, tuple(requests)


def validate_completed_request(
    request: Request, *, runtime_fingerprint: dict[str, Any],
    worker_record_path: Path, worker_provenance: dict[str, str],
) -> dict[str, Any] | None:
    """Return a completed request record, or ``None`` when it never started.

    Final request directories are published with one atomic rename.  Therefore
    a non-empty final directory must be a complete, hash-consistent unit; a
    partial or hand-edited directory fails closed instead of being overwritten.
    """

    if not request.output_dir.exists():
        return None
    if not request.output_dir.is_dir() or request.output_dir.is_symlink():
        raise WorkerContractError(
            f"completed output path is not a regular directory: {request.output_dir}"
        )
    if not any(request.output_dir.iterdir()):
        return None
    benchmark_path = request.benchmark_json
    output_path = request.output_dir / f"{request.name}_streaming.mp4"
    validate_live_request_bindings(request)
    validate_runtime_provenance(
        worker_provenance, worker_record_path, runtime_fingerprint,
    )
    for path, label in (
        (benchmark_path, "benchmark"), (output_path, "native video"),
    ):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise WorkerContractError(
                f"completed request {request.request_id} has invalid {label}"
            )
    try:
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkerContractError(
            f"completed request {request.request_id} has invalid benchmark JSON"
        ) from exc
    expected = {
        "schema_version": REQUEST_RECORD_SCHEMA,
        "request_id": request.request_id,
        "request_index": request.request_index,
        "action": request.action,
        "action_dsl": ACTIONS[request.action],
        "seed": SEED,
        "output_path": str(output_path),
        "n_pixel_frames": NUM_FRAMES - 1,
        "request_manifest": {
            "path": str(request.request_manifest),
            "sha256": request.request_manifest_sha256,
        },
        "worker_record": str(worker_record_path),
        "worker_provenance": worker_provenance,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": canonical_sha256(runtime_fingerprint),
        "input_bindings": {
            "image": {
                "path": str(request.image), "sha256": request.image_sha256,
            },
            "prompt_file": {
                "path": str(request.prompt_file),
                "sha256": request.prompt_sha256,
            },
            "intrinsics_file": {
                "path": str(request.intrinsics_file),
                "sha256": request.intrinsics_sha256,
            },
        },
    }
    identity = request_identity(request)
    expected["request_identity"] = identity
    expected["request_identity_sha256"] = canonical_sha256(identity)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise WorkerContractError(
                f"completed request {request.request_id} benchmark mismatch: {key}"
            )
    expected_hash = require_hash(
        payload.get("output_sha256"),
        f"completed request output SHA256: {request.request_id}",
    )
    if sha256(output_path) != expected_hash:
        raise WorkerContractError(
            f"completed request {request.request_id} output SHA256 mismatch"
        )
    return payload


def completion_set(
    requests: tuple[Request, ...], records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    by_id = {row.get("request_id"): row for row in records}
    if set(by_id) != {request.request_id for request in requests}:
        raise WorkerContractError("worker completion request set mismatch")
    rows: list[dict[str, str]] = []
    for request in requests:
        record = by_id[request.request_id]
        rows.append({
            "request_id": request.request_id,
            "request_identity_sha256": record["request_identity_sha256"],
            "benchmark_json": str(request.benchmark_json),
            "benchmark_json_sha256": sha256(request.benchmark_json),
            "output_path": record["output_path"],
            "output_sha256": record["output_sha256"],
        })
    return rows


def validate_completed_shard(
    requests: tuple[Request, ...], *, record_path: Path,
    runtime_fingerprint: dict[str, Any], worker_provenance: dict[str, str],
) -> dict[str, Any] | None:
    """Validate a final worker record and its exact 44 benchmark/video join."""

    existing_outputs = [
        request.output_dir for request in requests
        if request.output_dir.exists() and any(request.output_dir.iterdir())
    ]
    if not record_path.exists():
        if existing_outputs:
            raise WorkerContractError(
                "completed outputs exist without a bound final worker record"
            )
        return None
    if record_path.is_symlink() or not record_path.is_file():
        raise WorkerContractError("worker record is not a regular file")
    try:
        worker_record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkerContractError("worker record is invalid JSON") from exc
    status = worker_record.get("status")
    if (
        worker_record.get("schema_version") != RECORD_SCHEMA
        or status not in {"RUNNING", "COMPLETED_PENDING_OUTER_VIDEO_VALIDATION"}
        or worker_record.get("runtime_fingerprint") != runtime_fingerprint
        or worker_record.get("runtime_fingerprint_sha256")
        != canonical_sha256(runtime_fingerprint)
        or worker_record.get("worker_provenance") != worker_provenance
    ):
        raise WorkerContractError("final worker runtime provenance mismatch")
    records = [
        validate_completed_request(
            request,
            runtime_fingerprint=runtime_fingerprint,
            worker_record_path=record_path,
            worker_provenance=worker_provenance,
        )
        for request in requests
    ]
    completed = [record for record in records if record is not None]
    if worker_record.get("requests") != completed:
        raise WorkerContractError("worker request records differ from live outputs")
    if status == "RUNNING":
        if len(completed) == len(requests):
            raise WorkerContractError("RUNNING worker record already has all outputs")
        if "completion_set" in worker_record or "completion_set_sha256" in worker_record:
            raise WorkerContractError("partial worker record has a completion fingerprint")
        return worker_record
    if any(record is None for record in records):
        raise WorkerContractError("final worker record has missing output requests")
    joined = completion_set(requests, completed)
    if (
        worker_record.get("completion_set") != joined
        or worker_record.get("completion_set_sha256") != canonical_sha256(joined)
    ):
        raise WorkerContractError("final worker completion-set fingerprint mismatch")
    return worker_record


def source_revision(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def validate_persistent_source_contract(source_root: Path) -> dict[str, Any]:
    """Bind the upstream statements and reset boundaries used by this worker."""

    pipeline_path = source_root / PIPELINE_RELATIVE
    decoder_path = source_root / DECODER_RELATIVE
    if not pipeline_path.is_file() or not decoder_path.is_file():
        raise WorkerContractError("persistent SANA source files are missing")
    pipeline_text = pipeline_path.read_text(encoding="utf-8")
    pipeline_markers = (
        "exposes :meth:`generate` for repeated sampling",
        "def _offload_vae_encoder_for_streaming(self)",
        'encoder.to("cpu")',
        "generator = torch.Generator(device=self.device).manual_seed(params.seed)",
        "refiner_runner = RefinerChunkRunner(",
        "vae_streaming_decoder = CausalVaeStreamingDecoder(self.vae)",
    )
    missing = [marker for marker in pipeline_markers if marker not in pipeline_text]
    if missing:
        raise WorkerContractError(
            f"persistent SANA pipeline assumptions drifted: {missing}"
        )
    decoder_text = decoder_path.read_text(encoding="utf-8")
    decoder_markers = (
        "def reset(self)",
        "self._vae.clear_decoder_cache()",
        "reset = self._first_chunk",
        "reset_cache=reset",
    )
    missing = [marker for marker in decoder_markers if marker not in decoder_text]
    if missing:
        raise WorkerContractError(
            f"persistent SANA decoder reset assumptions drifted: {missing}"
        )
    return {
        "pipeline_path": str(pipeline_path.resolve()),
        "pipeline_sha256": sha256(pipeline_path),
        "decoder_path": str(decoder_path.resolve()),
        "decoder_sha256": sha256(decoder_path),
        "model_reuse": "upstream_pipeline_constructed_once",
        "per_request_state": (
            "new_solver_generator_refiner_runner_and_decoder_plus_explicit_vae_cache_clear"
        ),
    }


def restore_request_state(pipeline: Any, device: Any) -> dict[str, bool]:
    """Restore only state that upstream intentionally offloads between calls."""

    vae = getattr(pipeline, "vae", None)
    if vae is None:
        raise WorkerContractError("persistent pipeline has no VAE")
    encoder = getattr(vae, "encoder", None)
    if encoder is None:
        raise WorkerContractError("persistent pipeline VAE has no encoder")
    encoder.to(device)
    clear_encoder = getattr(vae, "clear_encoder_cache", None)
    clear_decoder = getattr(vae, "clear_decoder_cache", None)
    if not callable(clear_encoder) or not callable(clear_decoder):
        raise WorkerContractError("causal VAE request-cache clear methods are missing")
    clear_encoder()
    clear_decoder()
    return {
        "vae_encoder_restored": True,
        "vae_encoder_cache_cleared": True,
        "vae_decoder_cache_cleared": True,
    }


def generate_with_isolated_model_patch(
    pipeline: Any, *args: Any, **kwargs: Any,
) -> tuple[Any, dict[str, bool]]:
    """Run one request without retaining the solver's ``forward_long`` patch.

    The pinned SelfForcing solver patches ``pipeline.model.forward_long`` in
    its constructor.  A persistent process must therefore restore the exact
    pre-request callable even when generation raises, otherwise every request
    adds another closure and retains stale camera tensors on the GPU.
    """

    model = getattr(pipeline, "model", None)
    if model is None or not hasattr(model, "forward_long"):
        raise WorkerContractError("persistent pipeline model has no forward_long")
    original_forward_long = model.forward_long
    try:
        result = pipeline.generate_streaming(*args, **kwargs)
    finally:
        model.forward_long = original_forward_long
    if model.forward_long is not original_forward_long:
        raise WorkerContractError("failed to restore model.forward_long")
    return result, {
        "forward_long_restored": True,
        "solver_patch_retained_after_request": False,
    }


def clear_prompt_caches_if_changed(
    pipeline: Any, previous_prompt: str | None, prompt: str,
) -> dict[str, Any]:
    changed = previous_prompt is not None and previous_prompt != prompt
    cleared: list[str] = []
    if changed:
        for name in (
            "_streaming_stage1_prompt_cache", "_streaming_refiner_prompt_cache",
        ):
            cache = getattr(pipeline, name, None)
            if not isinstance(cache, dict):
                raise WorkerContractError(f"persistent pipeline cache missing: {name}")
            cache.clear()
            cleared.append(name)
    return {"prompt_changed": changed, "cleared": cleared}


def load_stage1_text_encoder_locally(
    model_builder: Any,
    torch_module: Any,
    stage1_text_encoder_root: Path,
    *,
    name: str = "T5",
    device: Any = "cuda",
) -> tuple[Any, Any]:
    """Mirror the pinned native Gemma loader with no remote fallback."""

    if name != "gemma-2-2b-it":
        raise WorkerContractError(f"unexpected stage-1 text encoder requested: {name}")
    tokenizer = model_builder.AutoTokenizer.from_pretrained(
        str(stage1_text_encoder_root), local_files_only=True,
    )
    tokenizer.padding_side = "right"
    text_encoder = (
        model_builder.AutoModelForCausalLM.from_pretrained(
            str(stage1_text_encoder_root),
            torch_dtype=torch_module.bfloat16,
            local_files_only=True,
        )
        .get_decoder()
        .to(device)
    )
    return tokenizer, text_encoder


def build_pipeline(
    source_root: Path, weights_root: Path, stage1_text_encoder_root: Path,
    sentencepiece_site: Path,
) -> tuple[Any, Any, Any, Any]:
    if source_revision(source_root) != SOURCE_REVISION:
        raise WorkerContractError("SANA source revision mismatch")
    sys.path.insert(0, str(source_root))
    import pyrallis
    import torch
    from diffusion.model import builder as model_builder
    from inference_video_scripts.wm import inference_sana_wm as native_base
    from inference_video_scripts.wm import inference_sana_wm_streaming as native

    import sentencepiece
    sentencepiece_path = Path(sentencepiece.__file__).resolve()
    try:
        sentencepiece_path.relative_to(sentencepiece_site.resolve())
    except ValueError as exc:
        raise WorkerContractError(
            f"SentencePiece imported outside isolated target: {sentencepiece_path}"
        ) from exc
    if sentencepiece.__version__ != SENTENCEPIECE_VERSION:
        raise WorkerContractError("SentencePiece runtime version drift")

    native._apply_fast_defaults()
    logger = native.get_root_logger()
    native._apply_precision_args(
        SimpleNamespace(stage1_precision="bf16", refiner_precision="bf16"), logger,
    )
    path_args = SimpleNamespace(
        streaming_root=weights_root,
        config=str(source_root / CONFIG_RELATIVE),
        model_path=None,
        causal_vae_path=None,
        refiner_root=None,
        refiner_gemma_root=None,
    )
    config_path, model_path, causal_vae_path, refiner_root, gemma_root = (
        native._resolve_streaming_paths(path_args)
    )
    config = pyrallis.parse(
        config_class=native.InferenceConfig, config_path=str(config_path), args=[],
    )
    config.vae.vae_type = "LTX2VAE_diffusers_causal"
    config.vae.vae_pretrained = str(causal_vae_path)
    if config.text_encoder.text_encoder_name != "gemma-2-2b-it":
        raise WorkerContractError(
            "pinned SANA config no longer requests gemma-2-2b-it"
        )
    refiner = native.RefinerSettings(
        root=str(refiner_root),
        gemma_root=str(gemma_root),
        sink_size=1,
        seed=SEED,
        block_size=REFINER_BLOCK_SIZE,
        kv_max_frames=REFINER_KV_MAX_FRAMES,
    )
    device = torch.device("cuda")
    original_loader = native_base.get_tokenizer_and_text_encoder
    if original_loader is not model_builder.get_tokenizer_and_text_encoder:
        raise WorkerContractError("stage-1 text-encoder import binding drifted")

    def load_pinned_stage1_text_encoder(name: str = "T5", device: Any = "cuda"):
        return load_stage1_text_encoder_locally(
            model_builder,
            torch,
            stage1_text_encoder_root,
            name=name,
            device=device,
        )

    native_base.get_tokenizer_and_text_encoder = load_pinned_stage1_text_encoder
    try:
        pipeline = native.SanaWMPipeline(
            config=config,
            model_path=str(model_path),
            device=device,
            refiner=refiner,
            offload_vae=False,
            offload_refiner=False,
            offload_text_encoder=False,
            logger=logger,
        )
    finally:
        native_base.get_tokenizer_and_text_encoder = original_loader
    if native_base.get_tokenizer_and_text_encoder is not original_loader:
        raise WorkerContractError("failed to restore native stage-1 loader")
    compile_mode = os.environ.get(
        "SANA_WM_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs",
    ).strip()
    compile_dynamic = os.environ.get(
        "SANA_WM_TORCH_COMPILE_DYNAMIC", "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    compile_targets = {
        item.strip().lower()
        for item in os.environ.get("SANA_WM_TORCH_COMPILE_TARGETS", "refiner")
        .replace("+", ",").split(",") if item.strip()
    }
    if compile_targets - {"vae", "refiner"}:
        raise WorkerContractError("unsupported SANA_WM_TORCH_COMPILE_TARGETS")
    if "vae" in compile_targets:
        compile_targets.remove("vae")
    if "refiner" not in compile_targets:
        raise WorkerContractError("exact persistent worker requires compiled refiner")
    pipeline.refiner.transformer = torch.compile(
        pipeline.refiner.transformer, mode=compile_mode, dynamic=compile_dynamic,
    )
    build_audit = {
        "config": str(config_path),
        "model_path": str(model_path),
        "causal_vae_path": str(causal_vae_path),
        "refiner_root": str(refiner_root),
        "gemma_root": str(gemma_root),
        "stage1_text_encoder": {
            "repo_id": STAGE1_TEXT_ENCODER_REPO,
            "revision": STAGE1_TEXT_ENCODER_REVISION,
            "root": str(stage1_text_encoder_root.resolve()),
            "local_files_only": True,
            "native_loader_restored": True,
            "tokenizer_class": "transformers.AutoTokenizer",
            "model_class": "transformers.AutoModelForCausalLM",
            "torch_dtype": "bfloat16",
            "projection": "get_decoder",
        },
        "sentencepiece": {
            "version": sentencepiece.__version__,
            "path": str(sentencepiece_path),
            "site_packages": str(sentencepiece_site.resolve()),
            "isolated": True,
        },
        "compile_mode": compile_mode,
        "compile_dynamic": compile_dynamic,
        "compile_targets": ["refiner"],
        "pipeline_load_count": 1,
    }
    return native, torch, pipeline, build_audit


def _run_claimed(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    weights_root = args.weights_root.resolve()
    stage1_text_encoder_root = args.stage1_text_encoder_root.resolve()
    sentencepiece_runtime_root = args.sentencepiece_runtime_root.resolve()
    if (
        not source_root.is_dir()
        or not weights_root.is_dir()
        or not stage1_text_encoder_root.is_dir()
        or not sentencepiece_runtime_root.is_dir()
    ):
        raise WorkerContractError(
            "source_root, weights_root, stage1_text_encoder_root, and "
            "sentencepiece_runtime_root must be directories"
        )
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_sana_wm_streaming_prefix_smoke as contract

    contract.validate_static_contract()
    source_audit = contract.validate_source(source_root)
    persistent_source_audit = validate_persistent_source_contract(source_root)
    weights_audit = contract.validate_weights(weights_root)
    sentencepiece_runtime_audit = contract.validate_sentencepiece_runtime(
        sentencepiece_runtime_root
    )
    sentencepiece_site = Path(sentencepiece_runtime_audit["site_packages"])
    if str(sentencepiece_site) not in sys.path:
        raise WorkerContractError(
            "isolated SentencePiece site-packages is absent from worker sys.path"
        )
    deployment_manifest = args.deployment_manifest.resolve()
    deployment = contract.validate_deployment_manifest(deployment_manifest)
    shared_path = args.shared_audit_artifact
    shared_sha = args.shared_audit_artifact_sha256
    if (shared_path is None) != (shared_sha is None):
        raise WorkerContractError(
            "shared audit artifact path and SHA256 must be supplied together"
        )
    if shared_path is not None:
        shared_audit = validate_shared_audit_artifact(
            shared_path.resolve(),
            shared_sha,
            source_root=source_root,
            weights_root=weights_root,
            stage1_text_encoder_root=stage1_text_encoder_root,
            sentencepiece_runtime_root=sentencepiece_runtime_root,
            deployment_manifest=deployment_manifest,
        )
        stage1_text_encoder_audit = shared_audit["stage1_text_encoder"]
    else:
        shared_audit = None
        stage1_text_encoder_audit = contract.validate_stage1_text_encoder(
            stage1_text_encoder_root
        )
    manifest_path = args.request_manifest.resolve()
    manifest_sha = sha256(manifest_path)
    if manifest_sha != require_hash(
        args.request_manifest_sha256, "request manifest SHA256",
    ):
        raise WorkerContractError("request manifest SHA256 differs from launch binding")
    manifest, requests = load_requests(
        manifest_path, allow_existing_output_dirs=args.resume,
    )
    expected_actions = (
        set(ACTIONS)
        if args.campaign_action == SMOKE_CAMPAIGN_ACTION
        else {args.campaign_action}
    )
    if (
        args.campaign_action not in (*ACTIONS, SMOKE_CAMPAIGN_ACTION)
        or {request.action for request in requests} != expected_actions
        or (
            args.campaign_action == SMOKE_CAMPAIGN_ACTION
            and len(requests) != len(ACTIONS)
        )
    ):
        raise WorkerContractError("campaign action/request shard mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.physical_gpu_uuid:
        raise WorkerContractError(
            "worker CUDA visibility is not the frozen physical GPU UUID"
        )
    record_path = args.worker_record.resolve()
    if record_path.exists() and not args.resume:
        raise WorkerContractError(f"refusing to overwrite worker record: {record_path}")

    if shared_audit is None:
        raise WorkerContractError("campaign worker requires a shared audit artifact")
    runtime_fingerprint = build_runtime_fingerprint(
        args,
        deployment=deployment,
        shared_audit=shared_audit,
        manifest_path=manifest_path,
        manifest_sha=manifest_sha,
    )
    runtime_fingerprint_sha = canonical_sha256(runtime_fingerprint)
    worker_provenance = ensure_runtime_provenance(
        record_path, runtime_fingerprint,
    )

    previous_record: dict[str, Any] | None = None
    if record_path.exists():
        try:
            previous_record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkerContractError("resume worker record is invalid JSON") from exc
        if (
            not isinstance(previous_record, dict)
            or previous_record.get("schema_version") != RECORD_SCHEMA
            or previous_record.get("request_manifest", {}).get("path")
            != str(manifest_path)
            or previous_record.get("request_manifest", {}).get("sha256")
            != manifest_sha
            or previous_record.get("deployment", {}).get("sha256")
            != deployment.get("sha256")
            or previous_record.get("runtime_fingerprint") != runtime_fingerprint
            or previous_record.get("runtime_fingerprint_sha256")
            != runtime_fingerprint_sha
            or previous_record.get("worker_provenance") != worker_provenance
        ):
            raise WorkerContractError("resume worker record binding mismatch")
        previous_shared = previous_record.get("shared_audit_artifact")
        if shared_audit is not None and (
            not isinstance(previous_shared, dict)
            or previous_shared.get("sha256") != shared_audit.get("sha256")
        ):
            raise WorkerContractError("resume shared-audit binding mismatch")

    completed: list[dict[str, Any]] = []
    pending: list[tuple[int, Request]] = []
    for request_index, request in enumerate(requests):
        request_record = validate_completed_request(
            request,
            runtime_fingerprint=runtime_fingerprint,
            worker_record_path=record_path,
            worker_provenance=worker_provenance,
        ) if args.resume else None
        if request_record is None:
            pending.append((request_index, request))
        else:
            if request_record.get("request_index") != request_index:
                raise WorkerContractError(
                    f"completed request index mismatch: {request.request_id}"
                )
            completed.append(request_record)

    if args.resume and not pending:
        if previous_record is None:
            raise WorkerContractError("resume found outputs but no bound worker record")
        previous_record["status"] = "COMPLETED_PENDING_OUTER_VIDEO_VALIDATION"
        previous_record["requests"] = completed
        joined = completion_set(requests, completed)
        previous_record["completion_set"] = joined
        previous_record["completion_set_sha256"] = canonical_sha256(joined)
        previous_record["resume_noop"] = True
        write_json_atomic(record_path, previous_record)
        print(json.dumps({
            "status": previous_record["status"],
            "request_count": len(requests),
            "resumed_completed_count": len(completed),
            "pending_count": 0,
            "pipeline_load_count_this_process": 0,
            "worker_record": str(record_path),
        }))
        return

    started = time.perf_counter()
    native, torch, pipeline, build_audit = build_pipeline(
        source_root, weights_root, stage1_text_encoder_root, sentencepiece_site,
    )
    load_seconds = time.perf_counter() - started
    process_attempt = (
        int(previous_record.get("process_attempt_count", 0)) + 1
        if previous_record is not None else 1
    )
    record: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA,
        "status": "RUNNING",
        "formal_score_eligible": False,
        "batch_authorization": False,
        "source": source_audit,
        "persistent_source_contract": persistent_source_audit,
        "weights": weights_audit,
        "stage1_text_encoder": stage1_text_encoder_audit,
        "sentencepiece_runtime": sentencepiece_runtime_audit,
        "shared_audit_artifact": shared_audit,
        "deployment": deployment,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha,
        "worker_provenance": worker_provenance,
        "request_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "request_count": len(requests),
            "contract": manifest,
        },
        "pipeline": build_audit,
        "model_load_seconds": load_seconds,
        "process_attempt_count": process_attempt,
        "resumed_completed_count": len(completed),
        "requests": completed,
    }
    write_json_atomic(record_path, record)

    previous_prompt: str | None = None
    for request_index, request in pending:
        validate_live_request_bindings(request)
        request.output_dir.parent.mkdir(parents=True, exist_ok=True)
        if request.output_dir.exists():
            if any(request.output_dir.iterdir()):
                raise WorkerContractError(
                    f"refusing incomplete final output_dir: {request.output_dir}"
                )
            request.output_dir.rmdir()
        attempt_dir = request.output_dir.with_name(
            f"{request.output_dir.name}.tmp.worker-{os.getpid()}"
        )
        if attempt_dir.exists():
            raise WorkerContractError(f"request attempt directory exists: {attempt_dir}")
        attempt_dir.mkdir()
        prompt_bytes = request.prompt_file.read_bytes()
        image_bytes = request.image.read_bytes()
        if (
            hashlib.sha256(prompt_bytes).hexdigest() != request.prompt_sha256
            or hashlib.sha256(image_bytes).hexdigest() != request.image_sha256
        ):
            raise WorkerContractError(
                f"request input changed while being read: {request.request_id}"
            )
        prompt = prompt_bytes.decode("utf-8", errors="strict").strip()
        contract.validate_prompt(prompt)
        cache_audit = clear_prompt_caches_if_changed(
            pipeline, previous_prompt, prompt,
        )
        state_audit = restore_request_state(pipeline, torch.device("cuda"))
        if cache_audit["prompt_changed"]:
            torch.cuda.empty_cache()

        image = native.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        c2w = native.action_string_to_c2w(
            ACTIONS[request.action],
            translation_speed=TRANSLATION_SPEED,
            rotation_speed_deg=ROTATION_SPEED_DEG,
        )
        if c2w.shape != (NUM_FRAMES, 4, 4):
            raise WorkerContractError(
                f"request {request.request_id} native trajectory shape mismatch: {c2w.shape}"
            )
        cropped, src_size, resized_size, crop_offset = native.resize_and_center_crop(image)
        intrinsics = native.load_intrinsics(request.intrinsics_file, NUM_FRAMES)
        intrinsics_vec4 = native.transform_intrinsics_for_crop(
            intrinsics, src_size, resized_size, crop_offset,
        )
        params = native.GenerationParams(
            num_frames=NUM_FRAMES,
            fps=FPS,
            cfg_scale=CFG_SCALE,
            flow_shift=FLOW_SHIFT,
            seed=SEED,
            negative_prompt="",
            sampling_algo="self_forcing",
            num_cached_blocks=2,
            sink_token=True,
            num_frame_per_block=NUM_FRAME_PER_BLOCK,
            denoising_step_list=list(DENOISING_STEP_LIST),
        )
        final_output_path = request.output_dir / f"{request.name}_streaming.mp4"
        attempt_output_path = attempt_dir / f"{request.name}_streaming.mp4"
        request_started = time.perf_counter()
        result, solver_state_audit = generate_with_isolated_model_patch(
            pipeline,
            cropped,
            prompt,
            c2w,
            intrinsics_vec4,
            params,
            output_path=attempt_output_path,
            streaming_crf=18,
            streaming_preset="medium",
            streaming_encoder="libx264",
            output_mode="mp4",
            profile_cuda=False,
        )
        request_seconds = time.perf_counter() - request_started
        if int(result["n_pixel_frames"]) != NUM_FRAMES - 1:
            raise WorkerContractError(
                f"request {request.request_id} native frame count mismatch"
            )
        frozen_request_identity = request_identity(request)
        benchmark = {
            "schema_version": REQUEST_RECORD_SCHEMA,
            "request_id": request.request_id,
            "request_index": request_index,
            "action": request.action,
            "action_dsl": ACTIONS[request.action],
            "seed": SEED,
            "request_manifest": {
                "path": str(request.request_manifest),
                "sha256": request.request_manifest_sha256,
            },
            "input_bindings": {
                "image": {
                    "path": str(request.image), "sha256": request.image_sha256,
                },
                "prompt_file": {
                    "path": str(request.prompt_file),
                    "sha256": request.prompt_sha256,
                },
                "intrinsics_file": {
                    "path": str(request.intrinsics_file),
                    "sha256": request.intrinsics_sha256,
                },
            },
            "request_identity": frozen_request_identity,
            "request_identity_sha256": canonical_sha256(frozen_request_identity),
            "worker_record": str(record_path),
            "worker_provenance": worker_provenance,
            "runtime_fingerprint": runtime_fingerprint,
            "runtime_fingerprint_sha256": runtime_fingerprint_sha,
            "model_load_count_for_worker": 1,
            "worker_process_attempt": process_attempt,
            "request_state_reset": state_audit,
            "solver_state_reset": solver_state_audit,
            "prompt_cache": cache_audit,
            "output_path": str(final_output_path),
            "output_sha256": sha256(attempt_output_path),
            "n_pixel_frames": int(result["n_pixel_frames"]),
            "wall_seconds": float(result["wall_seconds"]),
            "request_end_to_end_seconds": request_seconds,
            "frames_per_second": float(result["frames_per_second"]),
            "realtime_factor": float(result["realtime_factor"]),
        }
        validate_live_request_bindings(request)
        attempt_benchmark = attempt_dir / request.benchmark_json.name
        write_json_atomic(attempt_benchmark, benchmark)
        os.replace(attempt_dir, request.output_dir)
        record["requests"].append(benchmark)
        write_json_atomic(record_path, record)
        previous_prompt = prompt

    record["status"] = "COMPLETED_PENDING_OUTER_VIDEO_VALIDATION"
    record["total_wall_seconds"] = time.perf_counter() - started
    record["pipeline"]["pipeline_load_count"] = 1
    joined = completion_set(requests, record["requests"])
    record["completion_set"] = joined
    record["completion_set_sha256"] = canonical_sha256(joined)
    write_json_atomic(record_path, record)
    print(json.dumps({
        "status": record["status"],
        "request_count": len(requests),
        "pipeline_load_count": 1,
        "process_attempt_count": process_attempt,
        "resumed_completed_count": len(completed),
        "worker_record": str(record_path),
    }))


def run(args: argparse.Namespace) -> None:
    # The lease is the first stateful operation.  It is acquired before source
    # audits, large-file reads, torch import, CUDA initialization, or model load.
    with claim_shard(args):
        _run_claimed(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--weights-root", required=True, type=Path)
    parser.add_argument("--stage1-text-encoder-root", required=True, type=Path)
    parser.add_argument("--sentencepiece-runtime-root", required=True, type=Path)
    parser.add_argument("--shared-audit-artifact", type=Path)
    parser.add_argument("--shared-audit-artifact-sha256")
    parser.add_argument("--deployment-manifest", required=True, type=Path)
    parser.add_argument("--request-manifest", required=True, type=Path)
    parser.add_argument("--request-manifest-sha256", required=True)
    parser.add_argument("--worker-record", required=True, type=Path)
    parser.add_argument("--campaign-manifest-sha256", required=True)
    parser.add_argument(
        "--campaign-action", required=True,
        choices=(*ACTIONS, SMOKE_CAMPAIGN_ACTION),
    )
    parser.add_argument("--physical-gpu-uuid", required=True)
    parser.add_argument("--launch-token", required=True)
    parser.add_argument("--shard-lease", required=True, type=Path)
    parser.add_argument("--claim-record", required=True, type=Path)
    parser.add_argument("--model-global-lease-fd", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
