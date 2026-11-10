#!/usr/bin/env python3
"""CPU re-scoring of existing pose artifacts at two pose-quality gates (the paper uses the 0.1 gate)."""
from __future__ import annotations

import argparse, collections, hashlib, json, math, os, shlex, sys, time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    opt=p.parse_args(); config=json.loads(opt.config.read_text())
    os.environ['CUDA_VISIBLE_DEVICES']=''
    repo=Path(config['repo']); sys.path.insert(0,str(repo))
    from metrics import run_full_evaluation as rf
    from metrics.task1.evaluate_task1_final import _case_result
    from metrics.task2.task2_metrics import evaluate_cell
    from metrics.task3.task3_metrics import evaluate_pair,pair_rows

    expected_commit=config.get('code_commit','release')   # code version recorded in the outputs
    if config.get('source_pin'):   # provenance pin of the original run (optional in the release)
        pin=json.loads(Path(config['source_pin']).read_text())
        assert (pin.get('git_commit') or pin.get('commit'))==config.get('code_commit')
    index=Path(config['index']); source=Path(config['source_results'])
    normalized_source=Path(config['normalized_index']) if config.get('normalized_index') else None
    if normalized_source is not None:
        rows=[json.loads(line) for line in normalized_source.read_text().splitlines() if line.strip()]
        assert all(r.get('primary_key',{}).get('model')==config['model'] for r in rows)
        assert all('_record' in r and '_source_transform' in r for r in rows)
    else:
        rows=rf.load_eval_index(index,config['model'])
    assert len(rows)==3640
    assert collections.Counter(r['seed'] for r in rows)=={0:1960,1:840,2:840}
    atomic=[r for r in rows if r['route_id'].startswith('atomic_')]
    complex_rows=[r for r in rows if not r['route_id'].startswith('atomic_')]
    assert len(atomic)==2520 and len(complex_rows)==1120
    worker=source/'details/pose_worker_results.jsonl'
    items=[json.loads(line) for line in worker.read_text().splitlines() if line.strip()]
    by_id={str(item['record_id']):item for item in items}
    assert len(items)==len(by_id)==3640 and set(by_id)=={r['record_id'] for r in rows}
    poses={}
    status=collections.Counter()
    required={'camera_valid','pose_anchor_coverage','pose_inlier_ratio','camera_to_world_cv_rh'}
    for row in rows:
        item=by_id[row['record_id']]; status[str(item.get('status'))]+=1
        if item.get('status')!='ok': continue
        pose=Path(str(item['pose']))
        poses[row['record_id']]=pose
    assert len(poses)==3640,status

    # Full index/worker identity is checked above.  Decompress a deterministic
    # stratified sample here; the official scorers still read every pose during
    # the full run.  This avoids a second complete pass during preflight.
    sample_indices=[0,2519,2520,len(rows)-1]
    for index_value in sample_indices:
        row=rows[index_value]; pose=poses[row['record_id']]
        with np.load(pose,allow_pickle=False) as z:
            assert required.issubset(z.files),(pose,required-set(z.files))
            lengths={len(np.asarray(z[k])) for k in required}; assert len(lengths)==1,(pose,lengths)
            valid=np.asarray(z['camera_valid'],bool); matrices=np.asarray(z['camera_to_world_cv_rh'],float)
            assert np.isfinite(matrices[valid]).all(),pose
        manifest=pose.parent/'run_manifest.json'
        if manifest.is_file():
            m=json.loads(manifest.read_text()); assert m.get('record_id')==row['record_id']
            assert m.get('video_sha256')==row.get('video_sha256')
            assert Path(str(m.get('video'))).resolve()==Path(str(row['video_path'])).resolve()

    preflight={'status':'PASS','model':config['model'],'rows':len(rows),'poses':len(poses),
               'atomic':len(atomic),'complex':len(complex_rows),'worker_statuses':dict(status),
               'index_sha256':sha(index),'normalized_index_sha256':sha(normalized_source) if normalized_source else None,
               'worker_sha256':sha(worker),'repo':str(repo),
               'code_commit':expected_commit,'source_results':str(source),
               'pose_integrity_samples':len(sample_indices)}
    write_json(Path(config['output_root'])/'INPUT_PREFLIGHT.json',preflight)

    # Exercise the current code on one real sample from each task before a full rescore.
    sample_complex=complex_rows[0]; sample_atomic=atomic[0]
    t1=_case_result(poses[sample_complex['record_id']],None,0.1,0.1)
    t2=evaluate_cell(poses[sample_atomic['record_id']],sample_atomic['_record'],sample_atomic,
                     min_pose_inlier_ratio=0.1,min_pose_anchor_coverage=0.1)
    route_index={(r['primary_key']['model'],r['primary_key']['scene_id'],r['primary_key']['case_id'],r['seed'],r['route_id']):poses[r['record_id']] for r in complex_rows}
    atomic_rows={}; atomic_index={}
    for r in atomic:
        k=r['primary_key']; key=(k['model'],k['scene_id'],k['case_id'],r['seed'],r['route_id'].split('_',1)[1])
        atomic_rows[key]=r; atomic_index[key]=poses[r['record_id']]
    ra,rb=pair_rows(complex_rows)[0]
    t3=evaluate_pair(ra,rb,data_root=index,route_pose_index=route_index,atomic_pose_index=atomic_index,
                     atomic_rows=atomic_rows,atomic_cache={},min_pose_inlier_ratio=0.1,
                     min_pose_anchor_coverage=0.1,normalized_tau=0.5)
    smoke={'status':'PASS','task1_status':t1.get('status'),'task2_status':t2.get('status'),
           'task3_status':t3.get('status')}
    write_json(Path(config['output_root'])/'CPU_SMOKE.json',smoke)
    if opt.preflight_only:
        print(json.dumps({'preflight':preflight,'smoke':smoke},ensure_ascii=False,indent=2)); return

    invocation=shlex.join([sys.executable,*sys.argv])
    variants=(('gated_010',0.1),('ungated_000',0.0))
    done={}
    for tag,threshold in variants:
        out=Path(config['output_root'])/tag
        marker=out/'VARIANT_COMPLETE.json'
        if marker.is_file():
            done[tag]=json.loads(marker.read_text()); continue
        out.mkdir(parents=True,exist_ok=True)
        args=SimpleNamespace(model=config['model'],data_root=index,output_root=out,gpus='CPU',workers_per_gpu=0,
             anchor_workers=0,checkpoint=Path('/dev/null'),first_frame_max_l1=32.0,
             min_pose_inlier_ratio=threshold,min_pose_anchor_coverage=threshold,
             normalized_pass_tau=0.5,bootstrap_samples=2000)
        results=[]
        for row in rows:
            pose=poses[row['record_id']]
            results.append({'status':'ok','record_id':row['record_id'],'pose':str(pose),'resumed':True,
                            'threshold':threshold,'mode':'cpu_metrics_only_reusing_verified_pose'})
        rf._write_jsonl(out/'details/normalized_index.jsonl',rows)
        rf._write_jsonl(out/'details/pose_worker_results.jsonl',results)
        t1_full,cases=rf._evaluate_task1(complex_rows,poses,out,args)
        t2_full,cells=rf._evaluate_task2(atomic,poses,out,args,invocation)
        t3_full,routes=rf._evaluate_task3(complex_rows,atomic,poses,out,args,invocation)
        assert len(cells)==2520 and len(routes)==560
        assert t1_full['successful']+t1_full['failed']==1120
        assert not any(r.get('status')=='EVALUATION_ERROR' for r in routes)
        rf._create_plots(out,t1_full,cells,routes)
        report=rf.render_report(config['model'],index,out,'CPU',results,t1_full,t2_full,cells,t3_full,routes,invocation)
        (out/'StrideWorld_three_task_summary.md').write_text(report)
        summary={'model':config['model'],'code_commit':expected_commit,'thresholds':[threshold,threshold],
                 'pose':{'expected':3640,'successful':3640},'task1':t1_full,
                 'task2':{'expected':2520,'identified':sum(c['status']=='UNIT_IDENTIFIED' for c in cells),
                          'unit_k12':sum(c['status']=='UNIT_IDENTIFIED' for c in cells),
                          'task3_operator_k4':sum(c['task3_unit']['status']=='UNIT_IDENTIFIED' and c['task3_unit']['non_degenerate'] for c in cells),
                          'power_law_k16':sum(c['power_law']['status']=='FIT_AVAILABLE' and c['unit']['non_degenerate'] for c in cells)},
                 'task3':{'expected':560,'statuses':dict(collections.Counter(r['status'] for r in routes)),
                          'normalized':sum(r['status']=='MEASURED' and rf._path_value(r,'relation.error.d_n') is not None for r in routes),
                          'e2e_pass_count':sum(r['status']=='MEASURED' and r.get('relation',{}).get('error',{}).get('pass') is True for r in routes)},
                 'index_sha256':sha(index),'normalized_index_sha256':sha(normalized_source) if normalized_source else None,
                 'source_worker_sha256':sha(worker),'invocation':invocation}
        write_json(out/'summary.json',summary)
        receipt={'status':'PASS','model':config['model'],'variant':tag,'thresholds':[threshold,threshold],
                 'fixed_denominators':{'task1':1120,'task2':2520,'task3':560},
                 'summary_sha256':sha(out/'summary.json'),'report_sha256':sha(out/'StrideWorld_three_task_summary.md')}
        write_json(marker,receipt); done[tag]=receipt
        print(json.dumps({'stage':'variant_complete','model':config['model'],'variant':tag,'task3':summary['task3']},ensure_ascii=False),flush=True)
    final={'status':'PASS','model':config['model'],'code_commit':expected_commit,'variants':done,
           'source_results':str(source),'completed_unix':time.time()}
    write_json(Path(config['output_root'])/'EVALUATION_COMPLETE.json',final)
    print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
