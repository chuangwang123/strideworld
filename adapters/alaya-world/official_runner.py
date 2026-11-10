#!/usr/bin/env python3
"""Official API reproduction plus two exactly seven-route smoke sets, seed 0.

--phase baseline needs no generated bundle: it reads the original case1 triplet.
--phase all runs that baseline first, then the two manifests with one loaded
engine per rank. No upstream source or trajectory is modified at runtime.
"""
from __future__ import annotations
import argparse
import gc
import inspect
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from common import COMMIT, REPRO_ROOT, ROUTES, PROFILE, load, save, sha, input_files, file_records


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=REPRO_ROOT)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--source-root',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--phase',choices=['baseline','official','user','smoke','all'],default='all')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--seed',type=int,choices=[0],default=0)
    p.add_argument('--gpus',default=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3'))
    p.add_argument('--compile',choices=['none','default','reduce-overhead','max-autotune'],default='none')
    p.add_argument('--crf',type=int,default=18)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true',help='skip only fully verified sidecar/video hashes')
    return p.parse_args()


def plan(args):
    if args.num_shards<1 or not 0<=args.shard_index<args.num_shards:
        raise ValueError('require num_shards >= 1 and 0 <= shard_index < num_shards')
    args.root=args.root.resolve()
    args.bundle=(args.bundle or args.root/'artifacts_official_v1').resolve()
    args.source_root=(args.source_root or args.root/f'AlayaWorld-{COMMIT}').resolve()
    args.config=(args.config or args.root/'official_config.yaml').resolve()
    args.output_root=(args.output_root or args.root/'outputs_official_v1').resolve()
    records=[]
    if args.phase in ('baseline','all'):
        prefix=args.source_root/'playground/case1/case1'
        records.append({'record_id':'official_case1_original_trajectory','group':'official_baseline',
            'case_uid':'official/case1','route_id':'original_trajectory','task':'baseline',
            'seed':0,'rounds':4,'expected_frames':129,'input_prefix':str(prefix),
            'input_files':file_records(prefix),'original_triplet_unchanged':True,
            'skill_sec':0,'ttc':False,'scope':'unchanged original case1 image, prompt, camera; four rounds'})
    if args.phase in ('official','user','smoke','all'):
        from audit_smoke import audit_bundle
        audit_bundle(args.bundle,args.source_root,write=False)
        groups={'official':('official_case1',),'user':('ue5_scene_03_case_01',),
                'smoke':('official_case1','ue5_scene_03_case_01'),
                'all':('official_case1','ue5_scene_03_case_01')}[args.phase]
        for group in groups:
            manifest=load(args.bundle/'inputs'/group/'manifest.json')
            rows={r['route_id']:r for r in manifest['records']}
            for route in ROUTES:
                record=dict(rows[route]);record['group']=group
                records.append(record)
    expected={'baseline':1,'official':7,'user':7,'smoke':14,'all':15}[args.phase]
    if len(records)!=expected:raise ValueError('case count mismatch')
    for record in records:
        if record['seed']!=0 or record['rounds']!=4:raise ValueError('all cases require seed0 and full four rounds')
        for item in record['input_files']:
            if sha(item['path'])!=item['sha256']:raise ValueError(f'input changed: {item["path"]}')
    for index,record in enumerate(records):record['phase_record_index']=index
    # Shards are independent workers. Run baseline, official, user as separate
    # launch waves when global group ordering is required with multiple shards.
    return [record for index,record in enumerate(records) if index%args.num_shards==args.shard_index]


def assert_native_import(obj, source):
    origin=Path(inspect.getfile(obj)).resolve()
    if not origin.is_relative_to(source):raise RuntimeError(f'wrong imported source: {origin}')
    return {'path':str(origin),'sha256':sha(origin)}


def verify_video(path, expected_frames, width, height):
    """Decode via PyAV and inspect real frame count, not just write() success."""
    import av
    count=0;rate=None
    with av.open(str(path)) as container:
        stream=container.streams.video[0];rate=float(stream.average_rate)
        for frame in container.decode(stream):
            if (frame.width,frame.height)!=(width,height):raise RuntimeError('video dimensions changed')
            count+=1
    if count!=expected_frames or abs(rate-24)>1e-4:
        raise RuntimeError(f'encoded video mismatch frames={count}, rate={rate}')
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,
            'decoded_frames':count,'fps':rate,'width':width,'height':height}


def canonical_directory(output, row):
    scene,case=row['case_uid'].split('/',1)
    return output/'rollouts/alaya-world'/scene/case/'seed_0'/row['task']/row['route_id']/'attempt_001'


def link_exact(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.is_symlink():
        if dst.resolve()==src.resolve():return
        raise FileExistsError(dst)
    if dst.exists():raise FileExistsError(dst)
    dst.symlink_to(src.resolve())


def main():
    args=parse_args();records=plan(args)
    if args.dry_run:
        print({'status':'INPUT_CONTRACT_PASS_NOT_GENERATED','phase':args.phase,'records':len(records),
               'seed':0,'rounds_each':4,'shard_index':args.shard_index,'num_shards':args.num_shards,
               'record_ids':[r['record_id'] for r in records],'output_root':str(args.output_root)});return 0
    if not records:
        print(f'[OFFICIAL_V1] empty shard {args.shard_index}/{args.num_shards}; nothing to generate',flush=True)
        return 0
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpus
    os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')
    sys.path.insert(0,str(args.source_root));os.chdir(args.source_root)
    import torch
    import torch.distributed as dist
    from alaya.config.loader import load_config
    from alaya.control.action import c2w_to_adaln_action_vectors
    from alaya.inference.pipeline import FlashAlayaPipeline
    from alaya.inference.rollout_utils import (bootstrap_context_parallel,build_engine,check_input_resolution,
        free_generation_models,load_input_sample,plan_rollout,silence_worker_rank)
    provenance=[assert_native_import(x,args.source_root) for x in (FlashAlayaPipeline,build_engine,c2w_to_adaln_action_vectors)]
    cp_size,rank0,seed,compile_mode=bootstrap_context_parallel(0,args.compile)
    if not rank0:silence_worker_rank()
    cfg=load_config(str(args.config));cfg.validation.save_joystick=False
    mode=next(iter(cfg.validation.modes.values()))
    K=int(mode.layout.output_latent_frames);N=int(cfg.layout.history_latent_frames if mode.layout.history_latent_frames is None else mode.layout.history_latent_frames)
    gap=int(float(mode.layout.max_gap_sec or 0)*cfg.sample.fps/cfg.sample.temporal_stride)
    cond=int(mode.layout.condition_latent_frames);stride=int(cfg.sample.temporal_stride)
    if (K,N,gap,cond,stride,int(cfg.layout.sink_latent_frames))!=(4,16,0,1,8,1):raise ValueError('unsupported timing config; audit assumes official native8 layout')
    report={'schema_version':'alaya-official-run-official_v1','phase':args.phase,'status':'LOADING',
        'seed':0,'config':str(args.config),'config_sha256':sha(args.config),'context_parallel':cp_size,
        'shard_index':args.shard_index,'num_shards':args.num_shards,
        'source_files':provenance,'profile':PROFILE,'motion_status':'NOT_VISUALLY_VALIDATED',
        'baseline_scope':'original triplet, skill_sec=0, ttc=False, no joystick', 'cases':[]}
    suffix=f'_shard{args.shard_index:02d}-of-{args.num_shards:02d}' if args.num_shards>1 else ''
    report_path=args.output_root/f'report_{args.phase}{suffix}.json'
    if rank0:save(report_path,report)
    engine=build_engine(cfg,compile_mode=compile_mode,compile_aux=False,bank_taehv=False,verbose=rank0)
    from inference.da3_patch import apply_da3_robust_scale
    if not apply_da3_robust_scale():raise RuntimeError('official DA3 robust-scale patch failed; refusing silent fallback')
    if rank0:
        import depth_anything_3.utils.pose_align as pose_align
        report['da3_pose_align']={'path':pose_align.__file__,'sha256':sha(pose_align.__file__)}
        report['status']='RUNNING';save(report_path,report)

    for index,row in enumerate(records):
        rid=row['record_id'];directory=canonical_directory(args.output_root,row)
        sidecar=directory/'record.json';out=args.output_root/'native_outputs'/row['group']/f'{rid}_rounds-4.mp4'
        result={'record_id':rid,'group':row['group'],'record_index':index,'status':'RUNNING',
            'input_prefix':row['input_prefix'],'input_files':row['input_files'],'seed':0,'rounds':4,
            'motion_status':'NOT_VISUALLY_VALIDATED','scope':row.get('scope','official-contract aligned trajectory')}
        skip=False
        if args.resume and sidecar.is_file():
            old=load(sidecar)
            skip=(old.get('generation_status')=='COMPLETED' and old.get('config_sha256')==sha(args.config)
                  and old.get('input_files')==row['input_files'] and out.is_file()
                  and old.get('video',{}).get('sha256')==sha(out))
        if cp_size>1:
            decision=[skip if rank0 else None];dist.broadcast_object_list(decision,src=0);skip=decision[0]
        if skip:
            if rank0:
                result.update({'status':'SKIPPED_VERIFIED','record':str(sidecar),'output':str(out)})
                report['cases'].append(result);save(report_path,report)
            continue
        if sidecar.exists() or out.exists():raise FileExistsError(f'output exists without verified resume: {directory}')
        if rank0:print(f'[OFFICIAL_V1] {index+1}/{len(records)} {row["group"]}/{rid}',flush=True)
        try:
            pixels,caption,metadata=load_input_sample(row['input_prefix'],image_target_hw=(int(cfg.sample.height),int(cfg.sample.width)))
            check_input_resolution(pixels,cfg)
            pixels,metadata,rounds,max_rounds,needed=plan_rollout(cfg,pixels,metadata,rounds_cap=4,K=K,N=N,gap_steps=gap,cond_end=cond)
            if rounds!=4:raise RuntimeError(f'trajectory too short: {rounds}/{max_rounds}')
            all_actions=c2w_to_adaln_action_vectors(metadata['cam_c2w'],action_scale=cfg.control.action_scale,vae_temporal_stride=stride).cpu().numpy()[0]
            target=all_actions[17:33]
            if not np.isfinite(target).all() or np.max(np.abs(target))<=1e-9:raise RuntimeError('target actions absent or invalid')
            result['action_stats']={'target_actions_6d':target.tolist(),
                'max_abs_by_dim':np.abs(target).max(axis=0).tolist(),'mean_abs_by_dim':np.abs(target).mean(axis=0).tolist(),
                'nonzero_latents':int(np.count_nonzero(np.linalg.norm(target,axis=1)>1e-8))}
            pipe=FlashAlayaPipeline(engine,control_modes=list(mode.control),use_memory=bool(mode.use_memory),
                action_cfg_scale=float(mode.action_cfg_scale),flex_attn=(compile_mode!='none'),seed=0,ttc=False)
            cache=pipe.initialize_cache(pixels,caption,metadata,rounds=4,K=K,cond_end=cond,needed_latents=needed)
            started=time.perf_counter()
            for ar in range(4):
                pred=pipe.generate(ar,cache);pipe.finalize(ar,cache,pred)
                if rank0:print(f'[OFFICIAL_V1] {rid} chunk {ar+1}/4',flush=True)
            torch.cuda.synchronize()
            result['generate_seconds']=round(time.perf_counter()-started,3)
            # All ranks wait while rank0 decodes. On the final case release the
            # generation models exactly as the official CLI does before decoding.
            if index==len(records)-1:free_generation_models(engine)
            error=None
            if rank0:
                try:
                    native=pipe.decode(cache,keep_history_prefix=False).cpu()
                    if tuple(native.shape)!=(128,int(cfg.sample.height),int(cfg.sample.width),3):raise RuntimeError(f'bad native decode {tuple(native.shape)}')
                    initial=((pixels[0].detach().cpu().clamp(-1,1)+1)*127.5).to(torch.uint8).permute(1,2,0)
                    frames=torch.cat([initial[None],native],dim=0)
                    out.parent.mkdir(parents=True,exist_ok=True)
                    engine.write_video(out,frames,crf=args.crf)
                    video=verify_video(out,129,int(cfg.sample.width),int(cfg.sample.height))
                    result.update({'status':'COMPLETED','output':str(out),'video':video})
                    if row['group']=='official_baseline':
                        raw=directory/'native/official_generated_128.mp4';raw.parent.mkdir(parents=True,exist_ok=True)
                        engine.write_video(raw,native,crf=args.crf)
                        result['official_native_video']=verify_video(raw,128,int(cfg.sample.width),int(cfg.sample.height))
                    else:
                        preview=directory/'dose12_prefix_97.mp4';preview.parent.mkdir(parents=True,exist_ok=True)
                        engine.write_video(preview,frames[:97],crf=args.crf)
                        result['dose12_preview']=verify_video(preview,97,int(cfg.sample.width),int(cfg.sample.height))
                        result['dose12_preview']['derivation']='first97 decoded frames of the full rollout; no extra generation'
                    link_exact(out,directory/'video.mp4')
                    for source_file in input_files(row['input_prefix']):link_exact(source_file,directory/'native'/source_file.name)
                    saved=dict(row);saved.update({'schema_version':'alaya-aligned-smoke-record-official_v1','generation_status':'COMPLETED',
                        'motion_status':'NOT_VISUALLY_VALIDATED','config_sha256':sha(args.config),'video':video,
                        'video_path':str(directory/'video.mp4'),'generation_report':str(report_path),
                        'action_stats':result['action_stats'],'runtime_source':str(args.source_root)})
                    save(sidecar,saved);result['record']=str(sidecar)
                    del frames,native,initial
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            if cp_size>1:
                messages=[error];dist.broadcast_object_list(messages,src=0);error=messages[0]
            if error:raise RuntimeError(error)
            del pred,cache,pipe,pixels,metadata
            gc.collect();torch.cuda.empty_cache()
        except BaseException as exc:
            if rank0:
                result.update({'status':'FAILED','error':f'{type(exc).__name__}: {exc}'})
                report['cases'].append(result);report['status']='FAILED';save(report_path,report)
            raise
        if rank0:report['cases'].append(result);save(report_path,report)
    if rank0:
        report['status']='GENERATION_COMPLETED_MOTION_REVIEW_REQUIRED';save(report_path,report)
        print(f'[OFFICIAL_V1] generation complete; motion review still required: {report_path}',flush=True)
    if cp_size>1:dist.barrier();dist.destroy_process_group()
    return 0


if __name__=='__main__':raise SystemExit(main())
