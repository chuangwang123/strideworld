"""[rerun copy, smoke=False] New Evoke smoke identity with eight-frame action timing and resize-consistent K."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import build_model_inputs as base
from strideworld_actions import sample_c2w, pixel_intrinsics

def camera(item, profile, raw):
    poses = sample_c2w(item['actions'], np.minimum(np.arange(raw)/8.0, item['route_length']),
                      translation_per_unit=profile['translation_per_unit'],
                      yaw_degrees_per_unit=profile['yaw_degrees_per_unit'], dtype='float32')
    # Official infer_single applies direct PIL resize, NOT a cover crop.
    K = pixel_intrinsics(item['condition_width'], item['condition_height'], 84.)
    K[0] *= 640 / item['condition_width']
    K[1] *= 384 / item['condition_height']
    return poses, K

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    campaign = json.loads(base.DEFAULT_CAMPAIGN.read_text())
    profile = json.loads(base.DEFAULT_PROFILES.read_text())['models']['evoke_turbo_i2v']
    selection = argparse.Namespace(smoke=False, group=[], limit=0)
    records = base.select_records(campaign, selection)
    rows = []
    root = args.output/'evoke_turbo_i2v'
    for record in records:
        item = base.common(record, args.output, base.DEFAULT_DATA_ROOT, verify_source=True)
        formal = 8 * item['route_length']; chunks = math.ceil((formal+3)/36); raw = 36*chunks-3
        poses, K = camera(item, profile, raw)
        path = root/'cases'/item['record_id']/'pose.npz'
        base.save_npz(path, cam_c2w=poses, intrinsics=np.repeat(K[None], raw, axis=0))
        row = dict(item, num_chunks=chunks, raw_frame_count=raw, formal_frame_count=formal,
                   metric_frame_slice_half_open=[0,formal], unscored_tail_frames=raw-formal,
                   camera_path=str(path.resolve()), camera_sha256=base.file_sha(path),
                   camera_adapter='evoke-eight-frames-per-unit-direct-resize-K-v1',
                   native_translation_per_unit=profile['translation_per_unit'],
                   yaw_degrees_per_unit=profile['yaw_degrees_per_unit'])
        row['jsonl_row'] = dict(name=item['record_id'], image_path=item['condition_image'],
            pose_path=str(path.resolve()), prompt_path=item['prompt_path'], pose_fps=24,
            pose_source_resolution=[384,640], pose_type='vipe', seed=item['seed'])
        rows.append(row)
    base.finalize_model('evoke_turbo_i2v', rows, root)
    manifest = dict(schema_version='strideworld-model-input-build-v1',
        campaign_id=campaign['campaign_id'], campaign_sha256=campaign['source_plan_sha256'],
        campaign_document_sha256=base.canonical_sha(campaign), profile_sha256=base.file_sha(base.DEFAULT_PROFILES),
        builder_sha256=base.file_sha(Path(__file__)), base_builder_sha256=base.file_sha(Path(base.__file__)),
        record_count=len(rows), group_count=len({r['group_id'] for r in rows}),
        models={'evoke_turbo_i2v':len(rows)}, source_validation=True, future_rgb_used=False,
        compatibility_only=True, formal_output_accepted=False)
    manifest['build_sha256'] = base.canonical_sha(manifest)
    base.write_json(args.output/'MANIFEST.json',manifest)
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()
