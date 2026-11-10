#!/usr/bin/env python3
"""One GPU, one stable shard, one engine load for the existing full campaign.

Generation is impossible without an explicit, genuine smoke acceptance JSON.
Run shards 0..7 in the eight-GPU job and 8..11 in the four-GPU job; all use
--num-shards 12 and the same manifest/config/output/acceptance paths.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import gc
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from full_common import (COMMIT, REPRO_ROOT, PROFILE, canonical, digest, require, load, sha,
    relative_path, validate_campaign, verify_acceptance, verify_row_files, durable_save)
from common import camera_for_alaya, input_files
from official_runner import verify_video, assert_native_import, link_exact
from audit_smoke import expected_action


def now():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=REPRO_ROOT/'artifacts_full_official_v2/manifest.json')
    p.add_argument('--source-root', type=Path, default=REPRO_ROOT/f'AlayaWorld-{COMMIT}')
    p.add_argument('--config', type=Path)
    p.add_argument('--output-root', type=Path, default=REPRO_ROOT/'outputs_full_official_v2')
    p.add_argument('--smoke-acceptance-json', type=Path)
    p.add_argument('--shard-index', type=int, required=True)
    p.add_argument('--num-shards', type=int, default=12)
    p.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    p.add_argument('--compile', choices=['none', 'default', 'reduce-overhead', 'max-autotune'], default='none')
    p.add_argument('--crf', type=int, default=18)
    p.add_argument('--resume', action='store_true', help='verify completed hashes; archive and retry incomplete attempts')
    p.add_argument('--dry-run', action='store_true', help='CPU file/hash/shard check; write nothing and load no model')
    return p.parse_args()


def partition(records, shard_index, num_shards):
    require(num_shards > 0 and 0 <= shard_index < num_shards, 'invalid shard index/count')
    require([r['alaya_official_v1']['campaign_index'] for r in records] == list(range(len(records))), 'campaign index order drift')
    return [r for index, r in enumerate(records) if index % num_shards == shard_index]


def output_directory(root, row):
    directory = (root / relative_path((row.get('output') or row['v10_plan'].get('output', {}))['attempt_relpath'])).resolve()
    require(directory.is_relative_to(root), 'output path escaped output root')
    return directory


def bind_output_run(path, binding):
    """Atomic create-or-compare prevents jobs with incompatible shards/inputs mixing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.binding-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(canonical(binding)+'\n'); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            require(canonical(load(path)) == canonical(binding), f'output root belongs to a different run: {path}')
    finally:
        os.unlink(temp)


def event(progress, progress_path, event_path, **fields):
    progress.update(fields)
    progress['updated_at'] = now()
    durable_save(progress_path, progress)
    with event_path.open('a') as stream:
        stream.write(canonical({'time': progress['updated_at'], **fields})+'\n')
        stream.flush(); os.fsync(stream.fileno())


def verify_completed(row, directory, binding):
    record_path = directory/'record.json'
    if not record_path.is_file():
        return None
    previous = load(record_path)
    generation = previous.get('alaya_official_v1', {}).get('generation', {})
    if generation.get('status') != 'COMPLETED':
        return None
    require(previous['record_id'] == row['record_id'] and
            previous['alaya_official_v1']['source_record_sha256'] == row['alaya_official_v1']['source_record_sha256'],
            'completed record identity changed')
    require(canonical(previous['v10_plan']) == canonical(row['v10_plan']) and
            canonical(previous.get('output', previous['v10_plan'].get('output', {}))) == canonical(row.get('output', row['v10_plan'].get('output', {}))) and previous['seed'] == row['seed'],
            'completed v10 keys or output paths changed')
    require(generation.get('run_binding') == binding, 'completed result belongs to another run')
    require(previous['alaya_official_v1']['input_files'] == row['alaya_official_v1']['input_files'], 'completed input hashes changed')
    for key in ('video', 'dose12_preview'):
        if key not in generation:
            continue
        saved = generation[key]
        file = directory/('video.mp4' if key == 'video' else 'dose12_prefix_97.mp4')
        require(file.is_file() and sha(file) == saved['sha256'], f'completed video hash mismatch: {file}')
        actual = verify_video(file, row['alaya_official_v1']['expected_frames'] if key == 'video' else 97, 960, 544)
        require(actual['decoded_frames'] == saved['decoded_frames'], 'completed frame count changed')
    require('video' in generation, 'completed record has no main video')
    return generation


def archive_incomplete(directory):
    files = [directory/name for name in ('record.json', 'video.mp4', 'video.pending.mp4',
                                        'dose12_prefix_97.mp4', 'dose12_prefix_97.pending.mp4')
             if (directory/name).exists()]
    if files:
        archive = directory/'_partial'/f'{time.time_ns()}-{os.getpid()}'
        archive.mkdir(parents=True)
        for file in files:
            file.rename(archive/file.name)


def checked_write_video(engine, path, frames, crf):
    temporary = path.with_name(path.stem+'.pending.mp4')
    engine.write_video(temporary, frames, crf=crf)
    info = verify_video(temporary, len(frames), 960, 544)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    info['path'] = str(path)
    return info


def audit_native_row(row, metadata, scales, native_action):
    data = row['alaya_official_v1']; actions = row['v10_plan']['expanded_unit_actions']
    expected_poses, rounds, tail = camera_for_alaya(actions)
    require(rounds == data['rounds'] == math.ceil(len(actions)/4), 'dynamic round mismatch')
    require(data['expected_frames'] == 1+rounds*32 and data['unscored_tail_frames'] == tail, 'output timing mismatch')
    np.testing.assert_allclose(metadata['cam_c2w'].cpu().numpy(), expected_poses, atol=1e-6)
    np.testing.assert_allclose(metadata['cam_c2w_raw'].cpu().numpy(), expected_poses, atol=1e-6)
    np.testing.assert_allclose(metadata['intrinsic'].cpu().numpy()[2], [0, 0, 1], atol=0)
    require((metadata['pose_orig_w'], metadata['pose_orig_h']) == (960., 544.) and metadata['has_camera'] is True,
            'prepared image/camera metadata mismatch')
    all_actions = native_action(metadata['cam_c2w'], action_scale=scales, vae_temporal_stride=8).cpu().numpy()[0]
    target = all_actions[17:17+4*rounds]
    scale_array = np.array([float(x) for x in scales.split(',')])
    expected = np.stack([expected_action(a, PROFILE)/scale_array for a in actions])
    np.testing.assert_allclose(target[:len(actions)], expected, atol=1e-4, rtol=1e-4)
    if len(actions) < len(target):
        np.testing.assert_allclose(target[len(actions):], 0, atol=1e-5)
    return {'target_actions_6d': target.tolist(), 'max_abs_by_dim': np.abs(target).max(axis=0).tolist(),
            'mean_abs_by_dim': np.abs(target).mean(axis=0).tolist(),
            'scored_units': len(actions), 'unscored_tail_frames': tail}


def main():
    args = parse_args()
    if args.config is None: args.config = args.manifest.parent/'official_config.yaml'
    for key in ('manifest', 'source_root', 'config', 'output_root'):
        setattr(args, key, getattr(args, key).resolve())
    manifest = load(args.manifest)
    require(manifest['schema_version'] == 'alaya-full-campaign-v4', 'wrong full manifest schema')
    require(canonical(manifest['profile']) == canonical(PROFILE), 'manifest profile drift')
    require(sha(manifest['source_campaign']) == manifest['source_campaign_sha256'], 'original source campaign changed')
    summary = validate_campaign(manifest['records'])
    rows = partition(manifest['records'], args.shard_index, args.num_shards)
    for row in rows:
        verify_row_files(row)
    acceptance = verify_acceptance(args.smoke_acceptance_json) if args.smoke_acceptance_json else None
    if args.dry_run:
        print(canonical({'status': 'CPU_SHARD_CONTRACT_PASS_NOT_GENERATED', 'shard_index': args.shard_index,
            'num_shards': args.num_shards, 'assigned': len(rows), 'seed_counts': dict(Counter(r['seed'] for r in rows)),
            'generation_gate': 'VALID_ACCEPTANCE_SUPPLIED' if acceptance else 'ACCEPTANCE_REQUIRED',
            'record_ids_sha256': digest([r['record_id'] for r in rows]), 'campaign': summary}))
        return
    acceptance = verify_acceptance(args.smoke_acceptance_json)  # mandatory before output creation or GPU imports
    require(len(args.gpus.split(',')) == 1 and args.gpus.strip(), 'each full worker requires one GPU')
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1, 'launch full workers independently, not via multi-rank torchrun')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('ALAYA_USE_FA3', '0')
    sys.path.insert(0, str(args.source_root)); os.chdir(args.source_root)
    # Only now import the GPU stack. The acceptance check above cannot be bypassed
    # by --resume, a preexisting output file, or an empty shard.
    import torch
    from alaya.config.loader import load_config
    from alaya.control.action import c2w_to_adaln_action_vectors
    from alaya.inference.pipeline import FlashAlayaPipeline
    from alaya.inference.rollout_utils import (build_engine, free_generation_models,
        load_input_sample, plan_rollout, check_input_resolution)
    provenance = [assert_native_import(x, args.source_root) for x in
                  (FlashAlayaPipeline, build_engine, c2w_to_adaln_action_vectors)]
    cfg = load_config(str(args.config)); cfg.validation.save_joystick = False
    mode = next(iter(cfg.validation.modes.values()))
    K = int(mode.layout.output_latent_frames)
    N = int(cfg.layout.history_latent_frames if mode.layout.history_latent_frames is None else mode.layout.history_latent_frames)
    gap = int(float(mode.layout.max_gap_sec or 0)*cfg.sample.fps/cfg.sample.temporal_stride)
    cond = int(mode.layout.condition_latent_frames)
    require((K, N, gap, cond, int(cfg.sample.temporal_stride), int(cfg.layout.sink_latent_frames),
             int(cfg.sample.width), int(cfg.sample.height), float(cfg.sample.fps)) == (4,16,0,1,8,1,960,544,24.),
            'full worker requires the accepted OFFICIAL_V1 native-eight-frame inference layout')
    binding = {'manifest': str(args.manifest), 'manifest_sha256': sha(args.manifest),
        'profile': PROFILE, 'config_sha256': sha(args.config), 'acceptance': acceptance,
        'num_shards': args.num_shards, 'compile': args.compile, 'crf': args.crf, 'source_files': provenance}
    bind_output_run(args.output_root/'campaign_run.json', binding)
    state_dir = args.output_root/'workers'/f'shard_{args.shard_index:02d}_of_{args.num_shards:02d}'
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # duplicate shard cannot run concurrently
    progress_path, events_path = state_dir/'progress.json', state_dir/'events.jsonl'
    progress = {'status': 'VERIFYING', 'started_at': now(), 'pid': os.getpid(),
        'shard_index': args.shard_index, 'num_shards': args.num_shards, 'assigned': len(rows),
        'assigned_record_ids': [r['record_id'] for r in rows], 'completed': 0, 'verified_skipped': 0,
        'run_binding': binding, 'results': {}}
    event(progress, progress_path, events_path, status='VERIFYING')
    pending = []
    for row in rows:
        directory = output_directory(args.output_root, row)
        prior = verify_completed(row, directory, binding) if args.resume else None
        if prior is not None:
            progress['verified_skipped'] += 1
            progress['results'][row['record_id']] = {'status': 'SKIPPED_HASH_AND_DECODE_VERIFIED', 'record': str(directory/'record.json')}
        else:
            if (directory/'record.json').exists() or (directory/'video.mp4').exists():
                require(args.resume, f'existing attempt requires --resume: {directory}')
            pending.append(row)
    event(progress, progress_path, events_path, status='LOADING' if pending else 'COMPLETED')
    if not pending:
        print(canonical({'status': 'COMPLETED', 'progress': str(progress_path)})); return
    engine = build_engine(cfg, compile_mode=args.compile, compile_aux=False, bank_taehv=False, verbose=True)
    from inference.da3_patch import apply_da3_robust_scale
    require(apply_da3_robust_scale(), 'official DA3 robust-scale patch failed')
    import depth_anything_3.utils.pose_align as pose_align
    progress['da3_pose_align'] = {'path': pose_align.__file__, 'sha256': sha(pose_align.__file__)}
    for pending_index, row in enumerate(pending):
        directory = output_directory(args.output_root, row); directory.mkdir(parents=True, exist_ok=True)
        result = {'status': 'RUNNING', 'started_at': now(), 'seed': row['seed'],
            'rounds': row['alaya_official_v1']['rounds'], 'run_binding': binding,
            'smoke_profile_visual_pass': True, 'per_video_visual_status': 'NOT_REVIEWED'}
        record_lock = (directory/'record.lock').open('a')
        fcntl.flock(record_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.resume: archive_incomplete(directory)
            native = verify_row_files(row)
            require(verify_acceptance(args.smoke_acceptance_json) == acceptance, 'acceptance changed during worker run')
            require(sha(args.config) == binding['config_sha256'] and sha(args.manifest) == binding['manifest_sha256'],
                    'config or manifest changed during worker run')
            event(progress, progress_path, events_path, status='GENERATING', current_record_id=row['record_id'], current_chunk=0)
            random.seed(row['seed']); np.random.seed(row['seed']); torch.manual_seed(row['seed']); torch.cuda.manual_seed_all(row['seed'])
            pixels, caption, metadata = load_input_sample(native['input_prefix'], image_target_hw=(544,960))
            check_input_resolution(pixels, cfg)
            result['action_stats'] = audit_native_row(row, metadata, cfg.control.action_scale, c2w_to_adaln_action_vectors)
            rounds = native['rounds']
            pixels, metadata, planned, maximum, needed = plan_rollout(cfg, pixels, metadata,
                rounds_cap=rounds, K=K, N=N, gap_steps=gap, cond_end=cond)
            require(planned == rounds, f'trajectory too short for full route: {planned}/{rounds}/{maximum}')
            pipe = FlashAlayaPipeline(engine, control_modes=list(mode.control), use_memory=bool(mode.use_memory),
                action_cfg_scale=float(mode.action_cfg_scale), flex_attn=args.compile!='none', seed=row['seed'], ttc=False)
            cache = pipe.initialize_cache(pixels, caption, metadata, rounds=rounds, K=K, cond_end=cond, needed_latents=needed)
            started = time.perf_counter()
            for chunk in range(rounds):
                pred = pipe.generate(chunk, cache); pipe.finalize(chunk, cache, pred)
                torch.cuda.synchronize()
                event(progress, progress_path, events_path, current_chunk=chunk+1)
                print(f'[full OFFICIAL_V1 shard {args.shard_index}] {row["record_id"]} seed={row["seed"]} chunk={chunk+1}/{rounds}', flush=True)
            result['generate_seconds'] = round(time.perf_counter()-started, 3)
            event(progress, progress_path, events_path, status='DECODING')
            # Same reuse semantics as OFFICIAL_V1: no per-case checkpoint/model reload.
            # Release generation models only on the worker's last pending case.
            if pending_index == len(pending)-1: free_generation_models(engine)
            decoded = pipe.decode(cache, keep_history_prefix=False).cpu()
            require(tuple(decoded.shape) == (32*rounds,544,960,3), f'unexpected official decode shape: {tuple(decoded.shape)}')
            initial = ((pixels[0].detach().cpu().clamp(-1,1)+1)*127.5).to(torch.uint8).permute(1,2,0)
            frames = torch.cat([initial[None], decoded], dim=0)
            event(progress, progress_path, events_path, status='WRITING_AND_VERIFYING')
            result['video'] = checked_write_video(engine, directory/'video.mp4', frames, args.crf)
            if len(row['v10_plan']['expanded_unit_actions']) >= 12:
                result['dose12_preview'] = checked_write_video(engine, directory/'dose12_prefix_97.mp4', frames[:97], args.crf)
                result['dose12_preview']['derivation'] = 'first97 frames of full rollout; no extra generation'
            for file in input_files(native['input_prefix']): link_exact(file, directory/'native'/file.name)
            result.update({'status': 'COMPLETED', 'completed_at': now()})
            saved = copy.deepcopy(row); saved['alaya_official_v1']['generation'] = result
            durable_save(directory/'record.json', saved)
            progress['completed'] += 1
            progress['results'][row['record_id']] = {'status': 'COMPLETED', 'record': str(directory/'record.json'),
                                                   'video_sha256': result['video']['sha256']}
            event(progress, progress_path, events_path, status='RUNNING')
            del pred, cache, pipe, pixels, metadata, decoded, initial, frames
            gc.collect(); torch.cuda.empty_cache()
        except BaseException as exc:
            result.update({'status': 'FAILED', 'failed_at': now(), 'error': f'{type(exc).__name__}: {exc}'})
            saved = copy.deepcopy(row); saved['alaya_official_v1']['generation'] = result
            durable_save(directory/'record.json', saved)
            progress['results'][row['record_id']] = {'status': 'FAILED', 'error': result['error']}
            event(progress, progress_path, events_path, status='FAILED', error=result['error'])
            raise
        finally:
            record_lock.close()
    require(progress['completed']+progress['verified_skipped'] == len(rows), 'not all assigned records completed')
    event(progress, progress_path, events_path, status='COMPLETED', current_record_id=None, current_chunk=None)
    print(canonical({'status': 'COMPLETED', 'progress': str(progress_path)}))
    lock.close()


if __name__ == '__main__':
    main()
