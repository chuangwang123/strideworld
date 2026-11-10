#!/usr/bin/env python3
"""Evoke r5 recipe scaled to the full 3,640-record StrideWorld plan.

Same official argv capture (prepare_evoke_smoke_argv_r1 env), same D16 inclusive endpoint rewrite as
reports/evoke_direct_d16_r1, same worker body as run_evoke_compat_smoke_r1.worker plus the r5 eval()
correction (adapters/evoke_eval_mode_r5.py). Differences: inputs/outputs live under the rerun root,
any number of shards, resumable (valid records are skipped), and per-job shard ranges.
Frozen benchmark_refresh sources are only imported, never modified.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

W = Path('<path>/benchmark_refresh_20261019')
T = Path('<path>/evoke')
INPUT = T / 'inputs_full/evoke_turbo_i2v'
PREP = T / 'argv'
SMOKE_ARGV = W / 'reports/evoke_direct_d16_r1/argv_preparation'


class Captured(Exception):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def prepare(out_root: Path):
    """CPU only: capture the official batch argv for all rows, then apply the D16 endpoint rewrite."""
    if (PREP / 'jobs.jsonl').exists():
        raise SystemExit('argv already prepared: ' + str(PREP / 'jobs.jsonl'))
    capture_dir = PREP / 'capture_only'
    capture_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update(MODE='i2v', JSONL=str(INPUT / 'evoke_i2v.jsonl'), MAX_CASES='0', NSHARD='1', SHARD='0',
        OUT_ROOT=str(capture_dir), BASE_CKPT=str(W / 'models/Evoke/evoke-base'),
        TRANSFORMER_PATH=str(W / 'models/Evoke/evoke-turbo'), NUM_CHUNKS='4', IS_STAGE2='1',
        STAGE2_STEPS='1 1 1', GUIDANCE_SCALE='1', HEIGHT='384', WIDTH='640', FPS='24',
        RESTRICT='0', GEO_WARP_STAGE0_ONLY='1', WARP_SIGMA_MAX='0.135', NOISE_CENTER='0',
        RENDER_MODE='backward_zbuf', BW_FILL_ITERS='12', DEPTH_BACKEND='vigeo',
        VIGEO_WEIGHTS=str(W / 'models/ViGeo1.1'), ZBUF_DESPECKLE='0', WARP_MODE='fixed_mem',
        VAE_DECODE_TYPE='persistent', IN_PROCESS_BATCH='1', DUMP_GEO='1', SAVE_SEGMENTS='1',
        BG_POSTPROC='0', JOYSTICK_HUD='off', POSE_EXTEND_MODE='clamp', CAP_MAX_DEG='0',
        CAP_MAX_TRANS='0', POSE_SMOOTH_WIN='1')
    spec = importlib.util.spec_from_file_location('evoke_official_batch', W / 'sources/Evoke/scripts/inference/infer_batch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.subprocess.run

    def capture(argv, **kwargs):
        if len(argv) == 4 and argv[2] == '--argv_jsonl':
            raise Captured()
        raise RuntimeError('unexpected subprocess in CPU argv preparation')

    module.subprocess.run = capture
    try:
        module.main()
    except Captured:
        pass
    finally:
        module.subprocess.run = original
    jobs = [json.loads(x) for x in (capture_dir / '_argv_batch_shard0.jsonl').read_text().splitlines()]
    rows = {r['record_id']: r for r in map(json.loads, (INPUT / 'cases.jsonl').read_text().splitlines())}
    assert len(jobs) == len(rows) == 3640, (len(jobs), len(rows))
    smoke_rows = {}
    for shard in (0, 1):
        for line in (SMOKE_ARGV / f'shard_{shard}.jsonl').read_text().splitlines():
            job = json.loads(line)
            smoke_rows[job['name']] = job['input_row']
    contract = next(iter(smoke_rows.values()))['endpoint_contract']
    for job in jobs:
        row = dict(rows[job['name']])
        argv = job['argv']
        argv[argv.index('--num_frames') + 1] = str(33 * row['num_chunks'])
        argv[argv.index('--output_folder') + 1] = str(out_root / 'native' / job['name'])
        job['log'] = str(out_root / 'logs' / f"{job['name']}.log")
        endpoint = 8 * len(row['actions'])
        row.update(previous_formal_frame_count=row['formal_frame_count'], formal_frame_count=endpoint + 1,
                   metric_frame_slice_half_open=[0, endpoint + 1],
                   checkpoint_frame_indices=list(range(0, endpoint + 1, 8)), endpoint_contract=contract,
                   unscored_tail_frames=row['raw_frame_count'] - (endpoint + 1))
        assert endpoint < row['raw_frame_count']
        job['input_row'] = row
    jobs.sort(key=lambda j: j['name'])
    (PREP / 'jobs.jsonl').write_text(''.join(json.dumps(j) + '\n' for j in jobs))
    # Cross-check the 28 accepted smoke rows: identical except for the rerun root paths.
    def norm(value):
        text = json.dumps(value, sort_keys=True)
        for old, new in ((str(W / 'adapter_inputs_smoke_evoke_compat_r1'), 'INPUTROOT'), (str(T / 'inputs_full'), 'INPUTROOT'),
                         (str(W / 'runs/evoke_direct_d16_r1'), 'OUTROOT'), (str(out_root), 'OUTROOT')):
            text = text.replace(old, new)
        return json.loads(text)
    by_name = {j['name']: j for j in jobs}
    diffs = []
    for shard in (0, 1):
        for line in (SMOKE_ARGV / f'shard_{shard}.jsonl').read_text().splitlines():
            old = json.loads(line)
            new = by_name[old['name']]
            if norm(old['argv']) != norm(new['argv']):
                diffs.append((old['name'], 'argv'))
            o, n = norm(old['input_row']), norm(new['input_row'])
            for key in set(o) | set(n):
                if o.get(key) != n.get(key):
                    diffs.append((old['name'], key))
    save(PREP / 'PREPARED.json', dict(jobs=len(jobs), jobs_sha256=sha(PREP / 'jobs.jsonl'), output_root=str(out_root),
                                       smoke_cross_check_records=len(smoke_rows), smoke_cross_check_diffs=diffs))
    print(json.dumps(dict(jobs=len(jobs), smoke_diffs=diffs[:20], n_smoke_diffs=len(diffs))))


def all_jobs():
    return [json.loads(x) for x in (PREP / 'jobs.jsonl').read_text().splitlines()]


def engine():
    sys.path.insert(0, str(W / 'adapters'))
    from evoke_torch24_annotations import install as install_annotations
    install_annotations()
    sys.path.insert(0, str(W / 'sources/Evoke'))
    spec = importlib.util.spec_from_file_location('official_evoke_infer', W / 'sources/Evoke/scripts/inference/infer_single.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    from evoke_eval_mode_r5 import install as install_eval
    return install_eval(mod)


def record_ok(out_root: Path, job) -> bool:
    path = out_root / 'records' / job['name'] / 'generation_record.json'
    if not path.exists():
        return False
    try:
        rec = json.loads(path.read_text())
        return Path(rec['video_path']).exists() and sha(rec['video_path']) == rec['video_sha256']
    except Exception:
        return False


def worker(out_root: Path, subset: Path, uuid: str, label: str):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == uuid
    import torch
    assert torch.cuda.device_count() == 1 and 'GPU' in torch.cuda.get_device_name(0)
    mod = engine()
    import cv2
    session = f'{os.getpid()}-{time.time_ns()}'
    model_id = None
    names = [x for x in subset.read_text().split() if x]
    by_name = {j['name']: j for j in all_jobs()}
    for index, name in enumerate(names):
        job = by_name[name]
        if record_ok(out_root, job):
            print('SKIP_DONE', label, name, flush=True)
            continue
        row = job['input_row']
        started = time.time()
        native = out_root / 'native' / name
        if native.exists():
            stale = out_root / 'native_incomplete' / f'{name}.{session}'
            stale.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(native), str(stale))
        log = Path(job['log'])
        log.parent.mkdir(parents=True, exist_ok=True)
        print('START', label, name, flush=True)
        try:
            with log.open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                mod.main(job['argv'])
        finally:
            # --save_chunk_segments patches pipe._decode_chunk_persistent_cache with a per-case closure that wraps
            # whatever method is current. The official in-process batch path restores the method after every case;
            # calling main() directly does not, so closures stacked across cases (old segment dumpers kept running and
            # per-case time grew ~1 s per completed case). Restore the class method and clear causal caches, exactly as
            # infer_single's batch path does.
            for cached in list(mod._PIPE_CACHE.values()):
                cached_pipe = cached[0]
                cached_pipe.__dict__.pop('_decode_chunk_persistent_cache', None)
                for clear in (getattr(cached_pipe.vae, 'clear_cache', None), getattr(cached_pipe.transformer, 'clear_kv_cache', None)):
                    if clear is not None:
                        try:
                            clear()
                        except Exception:
                            pass
        assert len(mod._PIPE_CACHE) == 1, 'pipeline cache unexpectedly changed'
        pipe = next(iter(mod._PIPE_CACHE.values()))[0]
        assert '_decode_chunk_persistent_cache' not in pipe.__dict__, 'segment hook not restored'
        if model_id is None:
            model_id = id(pipe.transformer)
        assert model_id == id(pipe.transformer) and not pipe.transformer.training
        video = native / 'geo_pred.mp4'
        cap = cv2.VideoCapture(str(video))
        count = 0
        digest = hashlib.sha256()
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape == (384, 640, 3)
            digest.update(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
            count += 1
        cap.release()
        assert count == row['raw_frame_count'] and abs(fps - 24) < .02, (count, row['raw_frame_count'], fps)
        gc.collect()
        torch.cuda.empty_cache()
        save(out_root / 'records' / name / 'generation_record.json', dict(
            schema='strideworld-evoke-r5-full-rerun-v1', formal_output_accepted=False,
            worker_source_sha256=sha(__file__), segment_hook_restored_per_case=True,
            recipe='evoke_direct_d16_r5 (eval-corrected, 8 frames/unit, inclusive D16 endpoint)',
            record_id=name, worker=label, pid=os.getpid(), session=session, gpu_uuid=uuid,
            model_signature=model_id, model_eval=not pipe.transformer.training,
            first_case_in_session=model_id is not None and index == 0, elapsed_seconds=time.time() - started,
            video_path=str(video), video_sha256=sha(video), decoded_rgb_sha256=digest.hexdigest(),
            native_frame_count=count, formal_frame_count=row['formal_frame_count'],
            camera_sha256=sha(row['camera_path']), condition_image_sha256=sha(row['condition_image']),
            input_row=row, argv=job['argv'], log_sha256=sha(log),
            cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved()))
        print('DONE', label, name, round(time.time() - started, 1), flush=True)


def run(out_root: Path, subsets: list[Path]):
    """One worker per visible GPU; subsets[i] (a file of record names) goes to GPU i."""
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True).strip().splitlines()
    cards = [x.split(',', 1) for x in lines]
    assert len(cards) == len(subsets), (len(cards), len(subsets))
    assert len({c[0].strip() for c in cards}) == len(cards) and all('GPU' in c[1] for c in cards)
    out_root.mkdir(parents=True, exist_ok=True)
    tag = time.strftime('%m%d-%H%M%S')
    save(out_root / f'GPU_ALLOCATION_{tag}.json', dict(cards=cards, hostname=os.uname().nodename, subsets=[str(s) for s in subsets]))
    children, handles = [], []
    for i, (card, subset) in enumerate(zip(cards, subsets)):
        uuid = card[0].strip()
        log = (out_root / f'WORKER_{subset.stem}_{tag}.log').open('w')
        handles.append(log)
        children.append(subprocess.Popen([sys.executable, __file__, 'worker', '--out', str(out_root), '--subset', str(subset),
                                          '--uuid', uuid, '--label', subset.stem],
                                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid), stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in children]
    for h in handles:
        h.close()
    save(out_root / f'JOB_TERMINAL_{tag}.json', dict(returncodes=codes, subsets=[str(s) for s in subsets]))
    if any(codes):
        raise SystemExit('worker failed ' + str(codes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'worker', 'run'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--subset', type=Path)
    parser.add_argument('--subsets', type=Path, nargs='*')
    parser.add_argument('--uuid')
    parser.add_argument('--label', default='w')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.out)
    elif args.mode == 'worker':
        worker(args.out, args.subset, args.uuid, args.label)
    else:
        run(args.out, args.subsets)
