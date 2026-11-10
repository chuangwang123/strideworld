#!/usr/bin/env python3
"""Prepare the existing 3640-row campaign without changing its identity or output paths.

This builds inputs only. Generation requires a separate genuine smoke acceptance.
No data is built merely by importing this module or invoking --help.
"""
from __future__ import annotations
import argparse
import copy
import math
from pathlib import Path
import numpy as np
from PIL import Image
from full_common import (COMMIT, HOST_ROOT, REPRO_ROOT, PROFILE, CAMPAIGN, load, sha,
    canonical, digest, require, validate_campaign, source_image, durable_save)
from common import camera_for_alaya, file_records
from build_inputs import prepare_image, source_archive_check


def preflight(records, data_root, authoritative_plan=None):
    summary = validate_campaign(records)
    if authoritative_plan is not None:
        from full_common import validate_authoritative_plan
        summary['authoritative_plan'] = validate_authoritative_plan(records, authoritative_plan)
    cases = {}
    for row in records:
        image_path = source_image(row, data_root)
        key = row['condition']['image_uri']
        if key not in cases:
            camera_path = image_path.parent/'camera.json'
            camera = load(camera_path)
            with Image.open(image_path) as image:
                width, height = image.size
            require((width, height) == (camera['width'], camera['height']), f'camera/image dimensions differ: {image_path}')
            k = np.asarray(camera['K'], dtype=np.float64)
            require(k.shape == (3, 3) and np.isfinite(k).all()
                    and np.allclose(k[2], [0, 0, 1]) and k[0, 0] > 0 and k[1, 1] > 0,
                    f'invalid actual-pixel K: {camera_path}')
            cases[key] = {'source_image': str(image_path), 'source_image_sha256': sha(image_path),
                'source_camera': str(camera_path), 'source_camera_sha256': sha(camera_path),
                'source_width': width, 'source_height': height, 'k_pixel': k.tolist(),
                'cache_key': digest({'uri': key})[:24]}
        require(cases[key]['source_image_sha256'] == row['condition']['image_sha256'],
                f'campaign RGB SHA does not match new data: {row["record_id"]}')
    case_images = {(row['scene_id'], row['case_id']): row['condition']['image_uri'] for row in records}
    require(len(case_images) == 140 and len(set(case_images.values())) == 140,
            'expected one distinct new source image per each of 140 cases')
    summary.update({'unique_source_images': len(cases), 'unique_case_images': len(set(case_images.values())),
                    'rounds_distribution': {str(r): sum(math.ceil(len(x['v10_plan']['expanded_unit_actions'])/4) == r for x in records)
                         for r in sorted({math.ceil(len(x['v10_plan']['expanded_unit_actions'])/4) for x in records})}})
    return cases, summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, default=CAMPAIGN)
    p.add_argument('--data-root', type=Path, default=HOST_ROOT/'data')
    p.add_argument('--root', type=Path, default=REPRO_ROOT)
    p.add_argument('--source-root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true', help='validate source hashes and shape, write nothing')
    p.add_argument('--authoritative-plan', type=Path, default=HOST_ROOT/'work/strideworld_refresh_20260915/common/index/by_model/alaya-world.plan.jsonl')
    return p.parse_args()


def main():
    args = parse_args()
    campaign_path = args.campaign.resolve()
    campaign_sha = sha(campaign_path)
    campaign = load(campaign_path)
    records = campaign['records']
    require(all('alaya_official_v1' not in row for row in records), 'input campaign already contains reserved alaya_official_v1 fields')
    cases, summary = preflight(records, args.data_root.resolve(), args.authoritative_plan.resolve())
    if args.dry_run:
        print(canonical({'status': 'CPU_SOURCE_PREFLIGHT_PASS_NO_FILES_WRITTEN', **summary}))
        return
    import torch
    source = (args.source_root or args.root/f'AlayaWorld-{COMMIT}').resolve()
    output = (args.output or args.root/'artifacts_full_official_v1').resolve()
    require(not output.exists(), f'refusing to overwrite bundle: {output}')
    source_proof = source_archive_check(source, args.root/'official-ea03cfb.tar.gz')
    output.mkdir(parents=True)
    durable_save(output/'BUILD_STATE.json', {'status': 'BUILDING', **summary, 'profile': PROFILE})
    # Each source image is resized/cropped exactly once, shared by all seeds/routes.
    for case in cases.values():
        cached = output/'case_cache'/case['cache_key']/'image.png'
        normalized_k, transform = prepare_image(Path(case['source_image']), cached, case['k_pixel'])
        require(sha(case['source_image']) == case['source_image_sha256'], 'source image changed during preparation')
        require(sha(case['source_camera']) == case['source_camera_sha256'], 'source camera changed during preparation')
        case.update({'prepared_image': str(cached), 'prepared_image_sha256': sha(cached),
                     'normalized_intrinsic': normalized_k.tolist(), 'image_transform': transform})
        durable_save(cached.parent/'case.json', case)
    built = []
    for index, original in enumerate(records):
        row = copy.deepcopy(original)  # retain all v10_plan/primary-key/output fields verbatim
        actions = row['v10_plan']['expanded_unit_actions']
        case = cases[row['condition']['image_uri']]
        prefix = output/'inputs'/f'{index:06d}-{digest(row["record_id"])[:12]}'/'case'
        prefix.parent.mkdir(parents=True)
        Path(str(prefix)+'_image.png').symlink_to(case['prepared_image'])
        poses, rounds, tail = camera_for_alaya(actions)
        k = np.asarray(case['normalized_intrinsic'], dtype=np.float32)
        metadata = {'cam_c2w': torch.from_numpy(poses), 'cam_c2w_raw': torch.from_numpy(poses.copy()),
            'intrinsic': torch.from_numpy(k), 'intrinsic_raw': torch.from_numpy(k.copy()),
            'has_camera': True, 'pose_orig_w': 960.0, 'pose_orig_h': 544.0,
            'frame_start': 0, 'frame_end': len(poses)-1, 'source': PROFILE['id'], 'video_id': row['record_id'],
            'conditioning_pose_frames': 129, 'action_boundary_pixel': 128,
            'generated_frame_start': 129, 'trailing_hold_frames': 8+tail,
            'source_image_size_wh': case['image_transform']['source_size_wh'],
            'image_transform': case['image_transform']}
        torch.save(metadata, str(prefix)+'_camera.pt')
        Path(str(prefix)+'_prompt.txt').write_text(row['condition']['prompt'].strip()+'\n')
        row['alaya_official_v1'] = {'campaign_index': index, 'source_record_sha256': digest(original),
            'profile': PROFILE, 'actions': actions, 'common_units': len(actions),
            'rounds': rounds, 'expected_frames': 1+32*rounds, 'unscored_tail_frames': tail,
            'input_prefix': str(prefix), 'input_files': file_records(prefix),
            'source_image': case['source_image'], 'source_camera': case['source_camera'],
            'source_camera_sha256': case['source_camera_sha256'],
            'prepared_image_sha256': case['prepared_image_sha256'], 'image_transform': case['image_transform'],
            'unit_boundary_frames': [8*i for i in range(len(actions)+1)],
            'unit_boundary_camera_frames': [128+8*i for i in range(len(actions)+1)],
            'dose12_output_frame': 96 if len(actions)>=12 else None,
            'generation_status': 'NOT_RUN_REQUIRES_SMOKE_ACCEPTANCE'}
        require(canonical({k: v for k, v in row.items() if k != 'alaya_official_v1'}) == canonical(original), 'source row mutated')
        durable_save(prefix.parent/'record.json', row)
        built.append(row)
        if (index+1)%100 == 0:
            durable_save(output/'BUILD_STATE.json', {'status': 'BUILDING', 'records_built': index+1, **summary, 'profile': PROFILE})
            print(f'[full OFFICIAL_V1 build] {index+1}/{len(records)}', flush=True)
    require(sha(campaign_path) == campaign_sha, 'source campaign changed during build')
    manifest = {'schema_version': 'alaya-full-campaign-v4', 'profile': PROFILE, **summary,
        'source_campaign': str(campaign_path), 'source_campaign_sha256': campaign_sha,
        'source_campaign_metadata': {k: v for k, v in campaign.items() if k != 'records'},
        'official_source': source_proof, 'source_root': str(source),
        'build_status': 'BUILT_INPUTS_ONLY_REQUIRES_SMOKE_ACCEPTANCE', 'records': built}
    durable_save(output/'manifest.json', manifest)
    durable_save(output/'BUILD_STATE.json', {'status': manifest['build_status'], **summary,
        'profile': PROFILE, 'manifest': str(output/'manifest.json'), 'manifest_sha256': sha(output/'manifest.json')})
    print(canonical({'status': manifest['build_status'], 'manifest': str(output/'manifest.json'), **summary}))


if __name__ == '__main__':
    main()
