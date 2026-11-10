"""Minimal model-independent loader for frozen WorldStereo2 records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_records(manifest_path, *, shard_index=0, shard_count=1):
    """Shard by stable original record index without changing route/action order."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    manifest = json.loads(Path(manifest_path).read_text())
    rows = manifest["records"]
    if len(rows) != 1664 or len({r["record_id"] for r in rows}) != 1664:
        raise ValueError("campaign is not the frozen unique 1664-record refresh inventory")
    if [r["record_index"] for r in rows] != sorted({r["record_index"] for r in rows}) or not all(0 <= r["record_index"] < 3640 for r in rows):
        raise ValueError("source record order changed")
    if any(r["seed"] not in (0, 1, 2) for r in rows):
        raise ValueError("campaign seed drift")
    return [r for r in rows if r["record_index"] % shard_count == shard_index]


def generation_inputs(record, data_root, *, include_intrinsics=False, verify_image=True):
    """Return an explicit model-input allowlist, never the whole source record.

    The inference runner must initialize its own torch/NumPy/Python generators
    with the returned seed before *each* record. It must separately compile
    semantic_blocks using a frozen native camera profile before model inference.
    """
    uri = record["condition"]["image_uri"]
    if not uri.startswith("data://"):
        raise ValueError("expected data:// image URI")
    relative = Path(uri.removeprefix("data://"))
    root = Path(data_root).resolve()
    image = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not image.is_relative_to(root):
        raise ValueError("source image escapes data root")
    if not image.is_file():
        raise FileNotFoundError(image)
    if verify_image:
        digest = hashlib.sha256()
        with image.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != record["condition"]["image_sha256"]:
            raise ValueError(f"source RGB hash mismatch: {record['record_id']}")
    result = {"record_id": record["record_id"], "image_path": str(image),
              "prompt": record["condition"]["prompt"], "seed": record["seed"],
              "semantic_blocks": list(record["semantic_blocks"])}
    if include_intrinsics:
        result["camera_intrinsics"] = record["camera_intrinsics"]
    return result
