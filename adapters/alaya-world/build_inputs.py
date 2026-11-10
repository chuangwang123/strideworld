#!/usr/bin/env python3
"""Build an official-contract bundle: unchanged official baseline, seven + seven routes.

No upstream code is edited. The camera/action conversion is checked against the
pinned upstream implementation. The full atomic dose is 16; dose 12 is an
inspection point/prefix within each full rollout.
"""
from __future__ import annotations
import argparse
import shutil
import tarfile
from pathlib import Path
import numpy as np
from PIL import Image
from common import (COMMIT, HOST_ROOT, REPRO_ROOT, PROFILE, REFERENCE_PROFILES, ROUTES,
                    load, save, sha, profile_poses, camera_for_alaya, transformed_intrinsic,
                    input_files, file_records)


def prepare_image(source: Path, destination: Path, k_pixel):
    with Image.open(source) as raw:
        image = raw.convert('RGB')
        w, h = image.size
        kn, transform = transformed_intrinsic(k_pixel, w, h)
        rw, rh = transform['resized_size_wh']
        left, top = transform['crop_left_top']
        # The pixel representation is newly materialized and its hash is recorded.
        image = image.resize((rw, rh), Image.Resampling.BILINEAR)
        image = image.crop((left, top, left+960, top+544))
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
    return kn, transform


def source_archive_check(source: Path, archive: Path):
    """Verify unpacked files against the pinned archive, without changing either."""
    count, mismatches = 0, []
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 2 or '..' in parts:
                raise ValueError(f'unsafe archive member {member.name}')
            rel = Path(*parts[1:])
            target = source / rel
            stream = tar.extractfile(member)
            import hashlib
            archive_hash = hashlib.sha256(stream.read()).hexdigest()
            if not target.is_file() or sha(target) != archive_hash:
                mismatches.append(str(rel))
            count += 1
    if mismatches:
        raise ValueError(f'upstream differs from pinned archive: {mismatches[:20]}')
    return {'archive_sha256': sha(archive), 'archive_files_checked': count,
            'matches_pinned_archive': True, 'runtime_source': str(source), 'commit': COMMIT}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=REPRO_ROOT)
    p.add_argument('--source-root', type=Path)
    p.add_argument('--output', type=Path, help='new bundle directory; must not exist')
    p.add_argument('--campaign', type=Path, default=HOST_ROOT/'work/strideworld_refresh_20260915/runtime/native_inputs/alaya-world/native_campaign.json')
    p.add_argument('--case-root', type=Path, default=HOST_ROOT/'data/cases/ue5_scene_03/case_01')
    p.add_argument('--paths-config', type=Path, help='config supplying only paths; defaults to root/official_config.yaml')
    return p.parse_args()


def main():
    import torch
    import yaml
    args = parse_args()
    root = args.root.resolve()
    source = (args.source_root or root/f'AlayaWorld-{COMMIT}').resolve()
    output = (args.output or root/'artifacts_official_v1').resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}; choose --output NEW_PATH')
    source_proof = source_archive_check(source, root/'official-ea03cfb.tar.gz')
    campaign = load(args.campaign)
    candidates = [r for r in campaign['records'] if r['seed'] == 0
                  and r['v10_plan']['scene_id'] == 'ue5_scene_03' and r['v10_plan']['case_id'] == 'case_01']
    canonical = {r['v10_plan']['route_id']: r for r in candidates if r['v10_plan']['route_id'] in ROUTES}
    if len(canonical) != 7:
        raise ValueError('source campaign must contain all six atomics and G4_003_A for seed 0')
    for route, actions in ROUTES.items():
        if canonical[route]['v10_plan']['expanded_unit_actions'] != actions:
            raise ValueError(f'common route mismatch: {route}')
    user_image = args.case_root/'rgb.png'
    user_camera = load(args.case_root/'camera.json')
    with Image.open(user_image) as image:
        sw, sh = image.size
    if (sw, sh) != (user_camera['width'], user_camera['height']):
        raise ValueError('source camera/image dimensions disagree')
    user_sha = sha(user_image)
    for row in canonical.values():
        if row['condition']['image_sha256'] != user_sha:
            raise ValueError('source campaign does not bind to the current UE5 RGB')
    user_prompt = canonical['atomic_W']['condition']['prompt']
    official_prefix = source/'playground/case1/case1'
    official_meta = torch.load(str(official_prefix)+'_camera.pt', map_location='cpu', weights_only=False)
    official_image = Path(str(official_prefix)+'_image.png')
    with Image.open(official_image) as image:
        ow, oh = image.size
    original_kn = np.asarray(official_meta['intrinsic'], dtype=np.float64)
    if original_kn.shape != (3, 3):
        raise ValueError('official case1 intrinsic must be 3x3')
    original_k_pixel = np.diag([ow, oh, 1]) @ original_kn
    output.mkdir(parents=True)

    # Baseline is a byte-for-byte triplet copy, no synthetic poses, no preprocessing.
    baseline_prefix = output/'inputs/official_baseline/case1'
    baseline_prefix.parent.mkdir(parents=True)
    baseline_files = []
    for src_file, dst_file in zip(input_files(official_prefix), input_files(baseline_prefix)):
        shutil.copyfile(src_file, dst_file)
        if sha(src_file) != sha(dst_file):
            raise RuntimeError('baseline file copy hash mismatch')
        baseline_files.append({'path': str(dst_file), 'sha256': sha(dst_file), 'source': str(src_file),
                               'source_sha256': sha(src_file)})
    baseline = {'record_id': 'official_case1_original_trajectory', 'group': 'official_baseline',
                'case_uid': 'official/case1', 'route_id': 'original_trajectory', 'task': 'baseline',
                'seed': 0, 'rounds': 4, 'expected_frames': 129, 'input_prefix': str(baseline_prefix),
                'input_files': baseline_files, 'original_triplet_unchanged': True,
                'skill_sec': 0, 'ttc': False, 'motion_status': 'NOT_RUN',
                'baseline_scope': 'official case1 image, prompt and camera; CLI-equivalent --rounds 4 --seed 0 --skill-sec 0 --no-joystick --compile none'}
    save(baseline_prefix.parent/'manifest.json', {'schema_version': 'alaya-original-baseline-official_v1', 'records': [baseline]})

    groups = [('official_case1', 'official/case1', official_image,
               Path(str(official_prefix)+'_prompt.txt').read_text(), original_k_pixel),
              ('ue5_scene_03_case_01', 'ue5_scene_03/case_01', user_image, user_prompt, np.asarray(user_camera['K']))]
    for tag, case_uid, image_source, prompt, k_pixel in groups:
        records = []
        for route, actions in ROUTES.items():
            rid = f'{tag}__{route}'
            prefix = output/'inputs'/tag/'cases'/rid/rid
            prefix.parent.mkdir(parents=True)
            normalized_k, transform = prepare_image(image_source, Path(str(prefix)+'_image.png'), k_pixel)
            cameras, rounds, padding = camera_for_alaya(actions)
            metadata = {'cam_c2w': torch.from_numpy(cameras), 'cam_c2w_raw': torch.from_numpy(cameras.copy()),
                        'intrinsic': torch.from_numpy(normalized_k), 'intrinsic_raw': torch.from_numpy(normalized_k.copy()),
                        'has_camera': True, 'pose_orig_w': 960.0, 'pose_orig_h': 544.0,
                        'frame_start': 0, 'frame_end': len(cameras)-1,
                        'source': PROFILE['id'], 'video_id': rid,
                        'source_image_size_wh': transform['source_size_wh'], 'image_transform': transform,
                        'conditioning_pose_frames': 129, 'action_boundary_pixel': 128,
                        'generated_frame_start': 129, 'trailing_hold_frames': 8+padding}
            torch.save(metadata, str(prefix)+'_camera.pt')
            Path(str(prefix)+'_prompt.txt').write_text(prompt.strip()+'\n')
            row = {'record_id': rid, 'group': tag, 'seed': 0, 'route_id': route,
                   'task': 'task2' if route.startswith('atomic_') else 'task3', 'case_uid': case_uid,
                   'actions': actions, 'common_units': len(actions), 'full_dose': 16 if route.startswith('atomic_') else None,
                   'input_prefix': str(prefix), 'source_image': str(image_source),
                   'condition_sha256': sha(image_source), 'prepared_image_sha256': sha(str(prefix)+'_image.png'),
                   'camera_sha256': sha(str(prefix)+'_camera.pt'), 'input_files': file_records(prefix),
                   'rounds': rounds, 'expected_frames': 1+32*rounds,
                   'unit_boundary_frames': [8*i for i in range(len(actions)+1)],
                   'unit_boundary_camera_frames': [128+8*i for i in range(len(actions)+1)],
                   'inspection_frames': {'dose12': 96, 'dose16': 128} if route.startswith('atomic_') else {'dose12': 96, 'final': 8*len(actions)},
                   'unscored_tail_frames': padding, 'control_profile': PROFILE,
                   'image_transform': transform, 'motion_status': 'OFFICIAL_CONTRACT_NOT_RUN'}
            records.append(row)
            save(prefix.parent/'record.json', row)
        save(output/'inputs'/tag/'manifest.json', {'schema_version': 'alaya-smoke7-official_v1',
             'record_count': 7, 'profile': PROFILE, 'records': records})

    # Explicit model-specific amplitudes prevent the V3 global lateral-value leak.
    for model, profile in REFERENCE_PROFILES.items():
        for route, actions in ROUTES.items():
            task = 'task2' if route.startswith('atomic_') else 'task3'
            destination = output/'alignment'/model/'ue5_scene_03/case_01/seed_0'/task/route
            destination.mkdir(parents=True)
            cameras = profile_poses(actions, profile)
            np.save(destination/'c2w.npy', cameras)
            if model == 'lyra-2.0':
                target = 161
                padded = np.concatenate([cameras, np.repeat(cameras[-1:], target-len(cameras), axis=0)])
                np.savez_compressed(destination/'trajectory.npz', w2c=np.linalg.inv(padded).astype(np.float32),
                     intrinsics=np.repeat(np.asarray(user_camera['K'], dtype=np.float32)[None], target, axis=0),
                     image_width=sw, image_height=sh)
            else:
                np.save(destination/'poses.npy', cameras)
                # Existing Lingbot camera control export convention, not a UE5 calibration estimate.
                np.save(destination/'intrinsics.npy', np.repeat(np.array([[415.5298,415.6922,415.77786,239.77779]], dtype=np.float32), len(cameras), axis=0))
            save(destination/'record.json', {'model': model, 'case_uid': 'ue5_scene_03/case_01',
                 'seed': 0, 'route_id': route, 'actions': actions, 'condition_sha256': user_sha,
                 'unit_boundary_frames': [profile['frames_per_unit']*i for i in range(len(actions)+1)],
                 'native_profile': profile, 'status': 'CONTROL_EXPORT_ONLY_NOT_GENERATED',
                 'cross_model_metric_unit_claim': False})

    config = yaml.safe_load((source/'configs/infer.yaml').read_text())
    old = yaml.safe_load((args.paths_config or root/'official_config.yaml').read_text())
    config['paths'] = old['paths']
    config['validation']['save_joystick'] = False
    config['run']['output_dir'] = str(output/'outputs')
    (output/'official_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    source_proof.update({'repo': 'https://github.com/AlayaLab/AlayaWorld',
        'config_overrides': ['paths', 'validation.save_joystick', 'run.output_dir'],
        'model_path': config['paths']['model'], 'model_bytes': Path(config['paths']['model']).stat().st_size,
        'external_da3_source': config['paths'].get('da3_repo'),
        'external_da3_pristine_status': 'NOT_ASSERTED'})
    save(output/'OFFICIAL_SOURCE.json', source_proof)
    save(output/'COMMON_ROUTES.json', {'routes': [{'route_id': r, 'actions': a} for r,a in ROUTES.items()],
        'case_uid': 'ue5_scene_03/case_01', 'seed': 0, 'source_campaign': str(args.campaign),
        'source_campaign_sha256': sha(args.campaign), 'source_combo_record': canonical['G4_003_A']['record_id'],
        'profile': PROFILE, 'condition_sha256': user_sha})
    save(output/'BUILD_STATE.json', {'status': 'BUILT_NOT_AUDITED_NOT_GENERATED', 'bundle': str(output),
        'baseline': 1, 'official_smoke': 7, 'user_smoke': 7, 'seed': 0, 'profile': PROFILE})
    print(f'BUILT_NOT_GENERATED {output}: original baseline + 7 official-image + 7 new-case trajectories')


if __name__ == '__main__':
    main()
