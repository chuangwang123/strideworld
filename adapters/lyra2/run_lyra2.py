#!/usr/bin/env python3
"""Run the frozen StrideWorld 528-record bundle with Lyra 2.0.

The official Lyra custom-camera entry point accepts a folder of images, a
folder of trajectories and one ``num_frames`` value per process.  This wrapper
therefore groups the frozen records by native trajectory length, stages only
symlinks, and invokes the official entry point once per group.  It never edits
the frozen manifest or changes the route geometry.

By default this is a dry-run.  Pass ``--run`` to generate videos.  A first
smoke can use ``--limit 2 --lengths 81,401``; the same command without a limit
covers all 528 records.  ``--data-root`` is deliberately explicit because the
manifest was authored on HOST while generation may run on site-b.
"""

from __future__ import annotations

from resume_guard import is_complete, seal_output, quarantine_partial
import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_BUNDLE = "<path>/lyra2"
DEFAULT_SOURCE = "<path>/Lyra-2"
DEFAULT_WEIGHTS = "<path>/Lyra-2.0"
DEFAULT_DA3 = "<path>/DA3NESTED-GIANT-LARGE"
DEFAULT_DA3_SRC = "<path>/src"
DEFAULT_DATA = "<path>/data"
DEFAULT_PACKAGE = "<path>/lyra_alaya"
DEFAULT_HF_HOME = "<path>/hf_cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--manifest", default=None, help="lyra2 bundle manifest.json")
    p.add_argument("--bundle", default=DEFAULT_BUNDLE)
    p.add_argument("--source-root", default=DEFAULT_SOURCE)
    p.add_argument("--weights-root", default=DEFAULT_WEIGHTS)
    p.add_argument("--da3-model", default=DEFAULT_DA3)
    p.add_argument("--da3-source", default=DEFAULT_DA3_SRC)
    p.add_argument("--package-root", default=DEFAULT_PACKAGE,
                   help="shared lightweight dependencies (loguru, etc.)")
    p.add_argument("--hf-home", default=DEFAULT_HF_HOME,
                   help="offline HuggingFace cache containing the official google/umt5-xxl tokenizer")
    p.add_argument("--data-root", default=DEFAULT_DATA)
    p.add_argument("--work-root", default=None, help="temporary staged folders")
    p.add_argument("--output-root", default=None, help="final mp4 root")
    p.add_argument("--gpus", default="0", help="CUDA ids, e.g. 4 or 4,5,6,7")
    p.add_argument("--limit", type=int, default=None, help="limit records after filters")
    p.add_argument("--ids", default=None, help="comma-separated exact record IDs (smoke convenience)")
    p.add_argument("--lengths", default=None, help="comma-separated native frame lengths")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--pose-scale", type=float, default=1.0)
    p.add_argument("--sampling-steps", type=int, default=35)
    p.add_argument("--no-dmd", action="store_true", help="disable official 4-step DMD LoRA")
    p.add_argument("--use-moge-scale", action=argparse.BooleanOptionalAction, default=False,
                   help="align DA3 depth to MoGe scale; disabled in the calibrated StrideWorld profile")
    p.add_argument("--run", action="store_true", help="actually invoke Lyra; otherwise print commands")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep-stage", action="store_true")
    return p.parse_args()


def prepare_ffmpeg(work: Path, env: dict[str, str]) -> None:
    """Expose the bundled imageio-ffmpeg binary under the expected name.

    The base site-b image has no system ``ffmpeg`` command, while the official
    Lyra writer (ffmpegcv) checks that exact executable name at import time.
    Keep this compatibility shim inside generated staging rather than
    modifying the host environment.
    """
    if shutil.which("ffmpeg", path=env.get("PATH")):
        return
    try:
        import imageio_ffmpeg

        binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return
    if not binary.is_file():
        return
    bindir = work / "ffmpeg_bin"
    bindir.mkdir(parents=True, exist_ok=True)
    for name in ("ffmpeg", "ffprobe"):
        target = bindir / name
        if target.is_symlink() or target.exists():
            if target.is_symlink() and os.path.realpath(target) == str(binary):
                continue
            target.unlink()
        target.symlink_to(binary)
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")


def resolve_source_image(record: dict, data_root: Path) -> Path:
    """Resolve an HOST-authored source_image on the current host."""
    original = Path(str(record["source_image"]))
    if original.is_file():
        return original
    text = str(original)
    marker = "/data/"
    if marker in text:
        candidate = data_root / text.split(marker, 1)[1]
        if candidate.is_file():
            return candidate
    # Last-resort lookup is constrained to the known scene directory and exact
    # basename, so a duplicate image cannot silently be selected.
    matches = list(data_root.glob(f"MovieRenders_Sencen*/{original.name}"))
    if len(matches) == 1 and matches[0].is_file():
        return matches[0]
    raise FileNotFoundError(f"cannot resolve source image for {record['record_id']}: {original}")


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.path.realpath(dst) == str(src):
            return
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def select_records(manifest: dict, args: argparse.Namespace) -> dict[int, list[dict]]:
    records = list(manifest.get("records", []))
    if not records:
        raise ValueError("manifest contains no records")
    lengths = None
    if args.lengths:
        lengths = {int(x) for x in args.lengths.split(",") if x.strip()}
    ids = {x.strip() for x in args.ids.split(",") if x.strip()} if args.ids else None
    groups: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        if ids is not None and rec["record_id"] not in ids:
            continue
        n = int(rec["num_frames"])
        if lengths is not None and n not in lengths:
            continue
        if args.limit is not None and sum(len(v) for v in groups.values()) >= args.limit:
            break
        groups[n].append(rec)
    if not groups:
        raise ValueError("filters selected no records")
    return dict(sorted(groups.items()))


def stage_group(group: list[dict], *, bundle: Path, work: Path, data_root: Path, nframes: int) -> Path:
    stage = work / f"length_{nframes}"
    # The official entry point scans the whole folder rather than accepting a
    # record list.  Always rebuild this generated staging directory so a
    # previous dry-run or interrupted batch cannot leak extra samples into the
    # next invocation (which would make the model pick the wrong trajectory).
    if stage.exists() or stage.is_symlink():
        if stage.is_dir() and not stage.is_symlink():
            shutil.rmtree(stage)
        else:
            stage.unlink()
    for sub in ("images", "trajectories", "captions"):
        (stage / sub).mkdir(parents=True, exist_ok=True)
    for rec in group:
        rid = rec["record_id"]
        link(resolve_source_image(rec, data_root), stage / "images" / f"{rid}.png")
        link(bundle / "trajectories" / f"{rid}.npz", stage / "trajectories" / f"{rid}.npz")
        link(bundle / "captions" / f"{rid}.json", stage / "captions" / f"{rid}.json")
    return stage


def build_command(args: argparse.Namespace, stage: Path, group: list[dict], output: Path) -> list[str]:
    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
    cp = len(gpu_ids)
    py_args = [
        "-m", "lyra_2._src.inference.lyra2_custom_traj_inference",
        "--input_image_path", str(stage / "images"),
        "--trajectory_path", str(stage / "trajectories"),
        "--captions_path", str(stage / "captions"),
        "--checkpoint_dir", "checkpoints/model",
        # The released Lyra config registers the experiment under ``lyra2``;
        # using the unregistered descriptive name makes Hydra fail before any
        # weights are loaded.
        "--experiment", "lyra2",
        "--output_path", str(output),
        "--num_samples", str(len(group)),
        "--num_frames", str(int(group[0]["num_frames"])),
        "--resolution", "480,832",
        "--fps", str(args.fps),
        "--seed", str(args.seed),
        "--pose_scale", str(args.pose_scale),
        "--num_sampling_step", str(args.sampling_steps),
        "--da3_model_name", str(args.da3_model),
        "--da3_frame_interval", "8",
        "--da3_max_history_frames", "10",
    ]
    py_args.append("--use_moge_scale" if args.use_moge_scale else "--no-use_moge_scale")
    if not args.no_dmd:
        py_args.append("--use_dmd")
    if cp > 1:
        return ["torchrun", "--standalone", "--nproc_per_node", str(cp), *py_args,
                "--context_parallel_size", str(cp)]
    return [sys.executable, *py_args]


def make_runtime_root(args: argparse.Namespace, work: Path) -> Path:
    """Create a throw-away cwd matching the official Lyra layout.

    Lyra resolves its config and DMD LoRA paths relative to the current
    directory.  The released source and weights are intentionally kept in
    separate trees, so a tiny symlink view avoids copying either large tree.
    """
    runtime = work / "official_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for name, src in (("lyra_2", Path(args.source_root) / "lyra_2"),
                      ("checkpoints", Path(args.weights_root) / "checkpoints")):
        dst = runtime / name
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() and os.path.realpath(dst) == str(src):
                continue
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.symlink_to(src, target_is_directory=True)
    return runtime


def main() -> int:
    args = parse_args()
    bundle = Path(args.bundle).resolve()
    manifest_path = Path(args.manifest or (bundle / "manifest.json")).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.seed != manifest.get("seed"): raise ValueError("CLI seed differs from immutable native manifest")
    groups = select_records(manifest, args)
    work = Path(args.work_root or (bundle.parent / "_staging" / "lyra2_strideworld_528")).resolve()
    output_root = Path(args.output_root or (bundle.parent / "generated" / "lyra2_strideworld_528")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    gpu_ids = ",".join(x.strip() for x in args.gpus.split(",") if x.strip())
    print(json.dumps({"manifest": str(manifest_path), "records": sum(map(len, groups.values())),
                      "groups": {str(k): len(v) for k, v in groups.items()},
                      "gpus": gpu_ids, "run": args.run, "output_root": str(output_root)}, indent=2))
    runtime_root = None
    env = os.environ.copy()
    if args.run:
        work.mkdir(parents=True, exist_ok=True)
        runtime_root = make_runtime_root(args, work)
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        env["HF_HOME"] = str(args.hf_home)
        # ``snapshot_download(cache_dir=...)`` stores models--* directly under
        # this directory, so use it as HF_HUB_CACHE (not HF_HOME/hub).
        env["HF_HUB_CACHE"] = str(args.hf_home)
        # Never let a batch job silently block on the public network after the
        # required tokenizer cache has been prepared.
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(args.source_root), str(args.da3_source), str(args.package_root), env.get("PYTHONPATH", "")]))
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        prepare_ffmpeg(work, env)
    status = {"schema_version": "strideworld-lyra2-batch-status-v1", "manifest": str(manifest_path),
              "seed": args.seed, "records_requested": sum(map(len, groups.values())), "groups": []}
    for nframes, group in groups.items():
        # In dry-run mode this is only the command's future path; do not create
        # or mutate staging, since a prior live run may be using another path.
        stage = (stage_group(group, bundle=bundle, work=work, data_root=Path(args.data_root), nframes=nframes)
                 if args.run else work / f"length_{nframes}")
        out = output_root / f"length_{nframes}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = build_command(args, stage, group, out)
        row = {"num_frames": nframes, "record_ids": [r["record_id"] for r in group],
               "command": cmd, "status": "PLANNED"}
        print("[Lyra2]", " ".join(map(str, cmd)))
        if args.run:
            if args.resume:
                pending = [r for r in group if not is_complete(out / f"{r['record_id']}.mp4", r, args.seed, nframes, 832, 480)]
                if not pending:
                    row["status"] = "SKIPPED_COMPLETE"
                    status["groups"].append(row)
                    continue
                if len(pending) != len(group):
                    # Official runner accepts a folder, so stage a filtered view.
                    stage = stage_group(pending, bundle=bundle, work=work / "pending", data_root=Path(args.data_root), nframes=nframes)
                    cmd = build_command(args, stage, pending, out)
                    row["record_ids"] = [r["record_id"] for r in pending]
                    row["command"] = cmd
            for r in (pending if args.resume else group):
                quarantine_partial(out / f"{r['record_id']}.mp4")
            assert runtime_root is not None
            proc = subprocess.run(cmd, cwd=str(runtime_root), env=env)
            row["returncode"] = proc.returncode
            for r in (pending if args.resume else group):
                video = out / f"{r['record_id']}.mp4"
                try: seal_output(video, r, args.seed, nframes, 832, 480)
                except Exception: pass
            missing_outputs = [
                r["record_id"] for r in (pending if args.resume else group)
                if not is_complete(out / f"{r['record_id']}.mp4", r, args.seed, nframes, 832, 480)
            ]
            if missing_outputs:
                row["missing_outputs"] = missing_outputs
            row["status"] = "COMPLETED" if proc.returncode == 0 and not missing_outputs else "FAILED"
            if proc.returncode != 0 or missing_outputs:
                if proc.returncode == 0 and missing_outputs:
                    row["returncode"] = 2
                status["groups"].append(row)
                (output_root / "batch_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                return row["returncode"]
        status["groups"].append(row)
        (output_root / "batch_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    status["status"] = "COMPLETED" if args.run else "MATERIALIZED_NOT_RUN"
    (output_root / "batch_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if args.run and not args.keep_stage:
        # Staging contains only symlinks; remove it after successful completion.
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
