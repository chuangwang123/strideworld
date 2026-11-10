#!/usr/bin/env python3
"""Re-scoring of existing pose artifacts under the common relaxed protocol (CPU only)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--smoke', action='store_true')
    opt = p.parse_args()
    config = json.loads(opt.config.read_text())
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    if not opt.smoke and (root / 'summary.json').exists():
        raise RuntimeError('Completed ablation already exists; refusing to overwrite')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, config['repo'])
    import numpy as np
    from metrics import run_full_evaluation as rf
    from metrics.task1.evaluate_task1_final import _case_result
    from metrics.task2.task2_metrics import evaluate_cell
    from metrics.task3.task3_metrics import evaluate_pair, pair_rows
    from metrics.tools.constants import FROZEN_METRIC_SETTINGS as settings
    from metrics.tools.constants import FROZEN_METRIC_SETTINGS_SHA256 as settings_sha

    for key, expected in config['overrides'].items():
        assert settings[key] == expected, (key, settings[key], expected)
    args = argparse.Namespace(
        model=config['model'], data_root=Path(config['index']), output_root=root,
        gpus='', workers_per_gpu=1, anchor_workers=1,
        checkpoint=Path(config['checkpoint']), first_frame_max_l1=32.0,
        min_pose_inlier_ratio=settings['min_pose_inlier_ratio'],
        min_pose_anchor_coverage=settings['min_pose_anchor_coverage'],
        normalized_pass_tau=0.5, bootstrap_samples=2000,
    )
    started = time.time()
    invocation = shlex.join([sys.executable, *sys.argv])
    originals = {str(Path(config['source_results']) / name): sha(Path(config['source_results']) / name)
                 for name in ['summary.json', 'StrideWorld_three_task_summary.md']}
    provenance = {
        **config, 'started_at': stamp(), 'input_sha256': sha(args.data_root),
        'frozen_settings': dict(settings), 'frozen_settings_sha256': settings_sha,
        'kind': 'user_requested_eight_model_common_protocol_rescore', 'gpu_used_for_rescore': False,
        'video_generation_rerun': False, 'pose_estimation_rerun': bool(config.get('supplemental_pose_results')),
        'original_report_hashes_before': originals, 'invocation': invocation,
    }
    rows = rf.load_eval_index(args.data_root, args.model)
    assert len(rows) == 3640
    old = [json.loads(line) for line in
           (Path(config['source_results']) / 'details/pose_worker_results.jsonl').read_text().splitlines()]
    old_by_id = {r['record_id']: r for r in old}
    assert len(old_by_id) == 3640 and set(old_by_id) == {r['record_id'] for r in rows}
    supplements = []
    if config.get('supplemental_pose_results'):
        supplement_path = Path(config['supplemental_pose_results'])
        assert supplement_path.is_file(), f'Missing GPU repair manifest: {supplement_path}'
        supplements = [json.loads(line) for line in supplement_path.read_text().splitlines()]
        expected_ids = set(old_by_id) if config.get('rebuild_all_poses') else {k for k,v in old_by_id.items() if v['status'] != 'ok'}
        assert {r['record_id'] for r in supplements} == expected_ids
        assert len(supplements) == len(expected_ids)
        for item in supplements:
            if item['status'] == 'ok' or config.get('rebuild_all_poses'):
                old_by_id[item['record_id']] = item
        provenance['supplemental_pose_results_sha256'] = sha(supplement_path)
        provenance['recomputed_pose_count'] = sum(r['status']=='ok' for r in supplements)
        provenance['first_frame_gate_note'] = 'First-frame pixel-difference gate kept at 32; tracking and poses recomputed after correcting the independent index with the actual generation preprocessing; see pose_repair/repair_provenance.json'
    atomic = [r for r in rows if r['route_id'].startswith('atomic_')]
    complex_rows = [r for r in rows if not r['route_id'].startswith('atomic_')]
    assert len(atomic) == 2520 and len(complex_rows) == 1120
    assert Counter(r['seed'] for r in rows) == {0: 1960, 1: 840, 2: 840}
    poses, results = {}, []
    check_rows = ([next(r for r in atomic if old_by_id[r['record_id']]['status']=='ok'),
                   next(r for r in complex_rows if old_by_id[r['record_id']]['status']=='ok')] if opt.smoke else rows)
    required = {'camera_valid', 'pose_anchor_coverage', 'pose_inlier_ratio', 'camera_to_world_cv_rh'}
    for row in check_rows:
        old_item = old_by_id[row['record_id']]
        if old_item['status'] != 'ok':
            results.append(dict(old_item))
            continue
        path = Path(old_item['pose'])
        with np.load(path, allow_pickle=False) as z:
            assert required.issubset(z.files), (str(path), required - set(z.files))
            camera = np.asarray(z['camera_valid'], dtype=bool)
            coverage, ratio = z['pose_anchor_coverage'], z['pose_inlier_ratio']
            assert camera.shape == coverage.shape == ratio.shape
            measured = camera & (coverage >= args.min_pose_anchor_coverage) & (ratio >= args.min_pose_inlier_ratio)
            count = int(measured.sum())
        poses[row['record_id']] = path
        results.append({'status': 'ok', 'record_id': row['record_id'], 'pose': str(path),
                        'resumed': True, 'source_gpu': old_item.get('gpu'),
                        'source_measurable_frames': old_item.get('measurable_frames'),
                        'measurable_frames': count, 'n_frames': int(camera.size),
                        'evaluation_thresholds': config['overrides'],
                        'mode': 'cpu_metrics_only_reusing_original_pose'})
    if opt.smoke:
        ar, cr = check_rows
        t1 = _case_result(poses[cr['record_id']], None, args.min_pose_inlier_ratio, args.min_pose_anchor_coverage)
        t2 = evaluate_cell(poses[ar['record_id']], ar['_record'], ar,
                           min_pose_inlier_ratio=args.min_pose_inlier_ratio,
                           min_pose_anchor_coverage=args.min_pose_anchor_coverage)
        assert t1['status'] == 'ok'
        # Exercise one actual paired route with all of its source atomic poses.
        route_index = {tuple(r['primary_key'][k] for k in ('model','scene_id','case_id','seed','route_id')):
                       Path(old_by_id[r['record_id']]['pose']) for r in complex_rows if old_by_id[r['record_id']]['status']=='ok'}
        atomic_rows, atomic_index = {}, {}
        for r in atomic:
            k=r['primary_key']; key=(k['model'],k['scene_id'],k['case_id'],k['seed'],r['route_id'].split('_',1)[1])
            atomic_rows[key]=r
            if old_by_id[r['record_id']]['status']=='ok':
                atomic_index[key]=Path(old_by_id[r['record_id']]['pose'])
        ra, rb = next(iter(pair_rows(complex_rows)))
        t3 = evaluate_pair(ra, rb, data_root=args.data_root, route_pose_index=route_index,
                          atomic_pose_index=atomic_index, atomic_rows=atomic_rows, atomic_cache={},
                          min_pose_inlier_ratio=args.min_pose_inlier_ratio,
                          min_pose_anchor_coverage=args.min_pose_anchor_coverage, normalized_tau=.5)
        print(json.dumps({'smoke': 'PASS', 'variant': config['variant'], 'task1': t1['status'],
                          'task2': t2['status'], 'task3': t3['status'], 'settings_sha256': settings_sha}), flush=True)
        return
    rf._write_json(root / 'ablation_provenance.json', provenance)
    rf._write_jsonl(root / 'details/normalized_index.jsonl', rows)
    rf._write_jsonl(root / 'details/pose_worker_results.jsonl', results)
    # Each record links to its own immutable source or isolated repaired artifact.
    cache_dir=root/'artifacts'; cache_dir.mkdir(exist_ok=True)
    for record_id,path in poses.items():
        link=cache_dir/rf._safe(record_id)
        if link.exists() or link.is_symlink():
            assert link.is_symlink() and link.resolve()==path.parent.resolve()
        else:
            link.symlink_to(path.parent,target_is_directory=True)
    print(json.dumps({'stage': 'preflight_complete', 'variant': config['variant'], 'cached_poses': len(poses)}), flush=True)
    t1, cases = rf._evaluate_task1(complex_rows, poses, root, args)
    print(json.dumps({'stage': 'task1_complete', 'successful': t1['successful'], 'failed': t1['failed']}), flush=True)
    t2, cells = rf._evaluate_task2(atomic, poses, root, args, invocation)
    print(json.dumps({'stage': 'task2_complete', 'cells': len(cells)}), flush=True)
    t3, routes = rf._evaluate_task3(complex_rows, atomic, poses, root, args, invocation)
    assert len(cells) == 2520 and len(routes) == 560
    assert not any('evaluation_error:' in f for c in cells for f in c.get('failure_reasons', []))
    assert not any(r['status'] == 'EVALUATION_ERROR' for r in routes)
    assert not any(r['status'] == 'STATIC_SHORTCUT' for r in routes)
    source_routes=json.loads((Path(config['source_results'])/'task3/task3.json').read_text())['routes']
    identity=lambda r:(r['state_id'],r['seed'],r['pair_id'])
    new_routes={identity(r):r for r in routes}
    if config.get('allow_task3_protocol_upgrade'):
        old_spec=json.loads((Path(config['source_results'])/'task3/task3.json').read_text())['spec']
        assert 'atomic_operator_dose_max' not in old_spec, 'Only explicit legacy k16→k4 source may skip old-score equality'
    for old_route in ([] if config.get('rebuild_all_poses') or config.get('allow_task3_protocol_upgrade') else source_routes):
        if old_route['status']!='MEASURED':
            continue
        new_route=new_routes[identity(old_route)]
        assert new_route['status']=='MEASURED'
        for key in ['translation_cm','yaw_deg','d_n']:
            old_value=rf._path_value(old_route,'relation.error.'+key)
            if old_value is not None:
                assert math.isclose(old_value,rf._path_value(new_route,'relation.error.'+key),rel_tol=1e-10,abs_tol=1e-10), (identity(old_route),key)
    for route in routes:
        error=route.get('relation',{}).get('error',{}); units=route.get('normalizers',{})
        if error.get('d_n') is not None:
            calculated=math.hypot(error['translation_cm']/max(abs(units['u_t_cm']),1e-9),error['yaw_deg']/max(abs(units['u_r_deg']),1e-9))
            assert math.isclose(error['d_n'],calculated,rel_tol=1e-10,abs_tol=1e-10)
            assert error['pass']==bool(calculated<=units['tau'])
    rf._create_plots(root, t1, cells, routes)
    report = rf.render_report(args.model, args.data_root, root, args.gpus, results, t1, t2, cells, t3, routes, invocation)
    banner = ('> User-specified threshold-adjusted version; stored separately from the results with the original default thresholds. '
              f"Changes in this run: {config['description']}. Original videos and existing poses are reused; tracking after input-geometry correction is recomputed separately (if any); this stage recomputes the metrics on CPU.\n\n")
    report = banner + report
    report = report.replace('|GPU|``|', '|Compute resources (this run)|CPU (original poses reused, no GPU run)|')
    report = report.replace('per GPU: 1 persistent worker(s)', 'Existing poses reused; recomputed on CPU')
    report = report.replace('Frozen config SHA256', 'Threshold config SHA256 (this run)')
    report = report.replace('usable for formal statistics', 'usable for statistics under this threshold adjustment')
    report = report.replace('Assessment: Task 1 pose coverage and rigidity measurement are reliable; ', 'Summary under this threshold setting: ')
    if config['variant']=='h3_pose010':
        report=report.replace('number of route pairs whose complex-route boundaries are all measurable and whose required k=1..4 atomic operators are available',
                              'number of route pairs whose complex-route boundaries are all measurable, whose required k=1..4 atomic operators are available, and that are not static shortcuts')
    if settings['translation_floor_cm']==0.0 and settings['rotation_floor_deg']==0.0:
        report=report.replace('First 4 checkpoints incomplete or motion below the resolution floor','First 4 checkpoints incomplete (motion-magnitude floor removed in this run)')
        report=report.replace('Static shortcuts additionally excluded','Static magnitude gate removed in this run; matches the geometry+atomic measurable count')
        report=report.replace('non-static-shortcut, ','')
        report=report.replace('non-static-shortcut and ','')
        report=report.replace('non-static and ','')
    report_path = root / 'StrideWorld_three_task_summary.md'
    report_path.write_text(report)
    for p in [root/'task2/TASK2_REPORT.md', root/'task3/TASK3_REPORT.md']:
        if p.exists():
            content=p.read_text()
            if settings['translation_floor_cm']==0.0 and settings['rotation_floor_deg']==0.0:
                content=content.replace('Each unit action must also translate at least 1 cm or rotate at least 0.25 deg; otherwise it is marked `STATIC_SHORTCUT`.',
                    'This run removes the motion-magnitude floor (translation and rotation floors are both 0); static shortcuts are not judged by the original 1 cm / 0.25 deg floor.')
            p.write_text(banner+content)
    summary = {'model':args.model,'data_root':str(args.data_root),'output_root':str(root),
               'pose':{'expected':len(results),'successful':len(poses),'failed':len(results)-len(poses),'mode':'reused_plus_isolated_repair' if supplements else 'reused','recomputed':sum(r['status']=='ok' for r in supplements)},
               'task1':t1,'task2':{'expected':len(cells),
                 'identified':sum(c['status']=='UNIT_IDENTIFIED' for c in cells),
                 'unit_k12':sum(c['status']=='UNIT_IDENTIFIED' for c in cells),
                 'task3_operator_k4':sum(c['task3_unit']['status']=='UNIT_IDENTIFIED' and c['task3_unit']['non_degenerate'] for c in cells),
                 'power_law_k16':sum(c['power_law']['status']=='FIT_AVAILABLE' and c['unit']['non_degenerate'] for c in cells)},
               'task3':{'expected':len(routes),'statuses':dict(Counter(r['status'] for r in routes)),
                 'normalized':sum(r['status']=='MEASURED' and rf._path_value(r,'relation.error.d_n') is not None for r in routes),
                 'e2e_pass_count':sum(r['status']=='MEASURED' and r.get('relation',{}).get('error',{}).get('pass') is True for r in routes)},
               'report':str(report_path),'invocation':invocation,'ablation':config['description'],
               'frozen_settings_sha256':settings_sha,'elapsed_seconds':round(time.time()-started,2)}
    assert all(sha(path)==digest for path,digest in originals.items()), 'Original reports changed'
    provenance.update(finished_at=stamp(), elapsed_seconds=summary['elapsed_seconds'], original_report_hashes_unchanged=True)
    rf._write_json(root/'ablation_provenance.json',provenance)
    rf._write_json(root/'summary.json',summary)
    print(json.dumps({'stage':'complete','variant':config['variant'],'elapsed_seconds':summary['elapsed_seconds'],
                      'task3':summary['task3'],'report':str(report_path)}),flush=True)


if __name__ == '__main__':
    main()
