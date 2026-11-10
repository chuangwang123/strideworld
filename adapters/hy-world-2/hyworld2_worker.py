#!/usr/bin/env python3
"""Independent I0-only WorldStereo rollouts with frozen predicted geometry."""
from __future__ import annotations
import argparse, datetime, fcntl, hashlib, json, os, random, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from worldstereo_campaign import load_records, generation_inputs

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def write_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(p)

def decode_video(path):
    import cv2
    cap=cv2.VideoCapture(str(path));frames=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    if not frames: raise ValueError('zero decoded frames')
    return frames

def balanced_development_tasks(selected_rows, add_noop, repeat_first, probe_values, shard_count):
    """Expand one global dev plan before modulo sharding, with one repeat owner."""
    if shard_count < 1 or not selected_rows:
        raise ValueError('balanced development requires selected records and positive shard count')
    rows=json.loads(json.dumps(selected_rows))
    first=rows[0]
    if add_noop:
        r=json.loads(json.dumps(first));r['record_id']+='__NOOP_DEVELOPMENT'
        r['output_relpath']+='__NOOP_DEVELOPMENT';r['semantic_blocks']=['NO_OP']*8
        r['kind']='DEVELOPMENT_NOOP';rows.append(r)
    translation_rows=[r for r in selected_rows if r['kind']=='TASK2_ATOMIC' and r['semantic_blocks'][0].startswith('MOVE_')]
    for value in probe_values:
        for baseline in translation_rows:
            r=json.loads(json.dumps(baseline));suffix='__TRANSLATION_PROBE_'+format(value,'.6g').replace('.','p')
            r['probe_of']=r['record_id'];r['record_id']+=suffix;r['output_relpath']+=suffix
            r['kind']='DEVELOPMENT_TRANSLATION_PROBE';r['translation_probe_value']=value;rows.append(r)
    if repeat_first:
        r=json.loads(json.dumps(first));r['repeat_of']=r['record_id'];r['record_id']+='__REPEAT_DEVELOPMENT'
        r['output_relpath']+='__REPEAT_DEVELOPMENT';r['kind']='DEVELOPMENT_REPEAT'
        # Baseline remains task ordinal 0. Put its repeat at the latest later
        # ordinal owned by shard 0, without padding or duplicating other tasks.
        repeat_ordinal=(len(rows)//shard_count)*shard_count
        if repeat_ordinal <= 0:
            raise ValueError('not enough development tasks to repeat the first record later on its same GPU')
        rows.insert(repeat_ordinal,r)
    ids=[r['record_id'] for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError('balanced development plan contains duplicate record IDs')
    for ordinal,r in enumerate(rows):r['development_task_ordinal']=ordinal
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source', required=True); p.add_argument('--weights', required=True)
    p.add_argument('--manifest', required=True); p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True); p.add_argument('--profile', required=True)
    p.add_argument('--geometry-root',required=True,help='Frozen I0-only DA3 depth caches, one directory per case')
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--ids',default=''); p.add_argument('--dry-run',action='store_true')
    p.add_argument('--add-noop',action='store_true')
    p.add_argument('--repeat-first',action='store_true')
    p.add_argument('--translation-probe-values',default='',help='Development-only additional translation strengths on selected atomic records')
    p.add_argument('--balanced-development-shards',action='store_true',help='Expand one explicit development plan, then assign task ordinal modulo shard count')
    a=p.parse_args(); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lock=open(out/f'.worker_{a.shard_index}.lock','w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base_profile=json.loads(Path(a.profile).read_text()); ph=sha(a.profile)
    if a.shard_count<1 or not 0<=a.shard_index<a.shard_count:raise ValueError('invalid shard index/count')
    if a.balanced_development_shards and (not a.ids or base_profile['status']!='DEVELOPMENT_NOT_YET_SMOKE_ACCEPTED'):
        raise ValueError('balanced development sharding requires explicit --ids and a development profile')
    probe_values=[float(v) for v in a.translation_probe_values.split(',') if v]
    if probe_values and (not a.ids or base_profile['status']!='DEVELOPMENT_NOT_YET_SMOKE_ACCEPTED'):
        raise ValueError('translation probes require explicit development IDs/profile')
    if any(not np.isfinite(v) or v<=0 for v in probe_values) or len(set(probe_values))!=len(probe_values):
        raise ValueError('invalid or duplicate translation probe')
    weights_provenance=Path(a.weights)/'worldstereo-camera/config_provenance.json'
    if not a.dry_run:
        provenance=json.loads(weights_provenance.read_text())
        ready=Path(provenance['weights_ready_path'])
        if sha(ready)!=provenance['weights_ready_sha256']:raise ValueError('weights verification identity changed')
        if sha(provenance['verification_path'])!=provenance['verification_sha256']:raise ValueError('weight hashes changed')
    compile_mode = 'eager_identity_recovery_20260916'
    contract={'profile':ph,'worker':sha(__file__),'controls':sha(HERE/'hyworld2_camera_controls.py'),
              'rgbd_renderer':sha(HERE/'hyworld2_rgbd_render.py'),
              'manifest':sha(a.manifest),'weights_config':sha(Path(a.weights)/'worldstereo-camera/config.json') if not a.dry_run else 'pending',
              'weights_provenance':sha(weights_provenance) if not a.dry_run else 'pending',
              'source_commit':base_profile['source_commit'],'development_translation_probe_values':probe_values,
              'torch_compile_mode':compile_mode,
              'torch_compile_bypass_reason':'The recovered runtime exposes Triton but torch._inductor rejects it as non-working; auxiliary modules run eagerly with identical weights and inputs.'}
    contract_sha=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    rows=load_records(a.manifest)
    if a.ids:
        ids=a.ids.split(','); rows=[r for r in rows if r['record_id'] in ids]
        if {r['record_id'] for r in rows}!=set(ids): raise ValueError('requested IDs missing')
    if a.balanced_development_shards:
        rows=balanced_development_tasks(rows,a.add_noop,a.repeat_first,probe_values,a.shard_count)
        rows=[r for ordinal,r in enumerate(rows) if ordinal%a.shard_count==a.shard_index]
    else:
        rows=[r for r in rows if r['record_index']%a.shard_count==a.shard_index]
        if a.add_noop:
            r=json.loads(json.dumps(rows[0])); r['record_id']+='__NOOP_DEVELOPMENT'
            r['output_relpath']+='__NOOP_DEVELOPMENT';r['semantic_blocks']=['NO_OP']*8
            r['kind']='DEVELOPMENT_NOOP';rows.append(r)
        if a.repeat_first:
            r=json.loads(json.dumps(rows[0]));r['repeat_of']=r['record_id'];r['record_id']+='__REPEAT_DEVELOPMENT'
            r['output_relpath']+='__REPEAT_DEVELOPMENT';r['kind']='DEVELOPMENT_REPEAT';rows.append(r)
        translation_rows=[r for r in rows if r['kind']=='TASK2_ATOMIC' and r['semantic_blocks'][0].startswith('MOVE_')]
        for value in probe_values:
            for baseline in translation_rows:
                r=json.loads(json.dumps(baseline));suffix='__TRANSLATION_PROBE_'+format(value,'.6g').replace('.','p')
                r['probe_of']=r['record_id'];r['record_id']+=suffix;r['output_relpath']+=suffix
                r['kind']='DEVELOPMENT_TRANSLATION_PROBE';r['translation_probe_value']=value;rows.append(r)
    from hyworld2_camera_controls import compile_trajectory
    if a.dry_run:
        # Contract-only static validation deliberately does not require the
        # derived DA3 cache.  Geometry is created and hash-checked inside the
        # formal job before any model inference starts.
        for r in rows:
            generation_inputs(r,a.data_root,include_intrinsics=True)
        write_json(out/f'preflight_{a.shard_index}.json',{'records':len(rows),'profile_sha256':ph,
            'image_bindings_verified':True,'geometry_cache_checked':False,
            'geometry_cache_check_stage':'formal_job_before_model_load',
            'gpu_imported':False,'model_loaded':False,
            'seeds':sorted({int(r['seed']) for r in rows}),
            'unit_counts':sorted({len(r['semantic_blocks']) for r in rows})})
        return
    inputs=[]
    for r in rows:
        inp=generation_inputs(r,a.data_root,include_intrinsics=True)
        geometry_directory=Path(a.geometry_root)/r['case_id']
        inp['geometry_directory']=str(geometry_directory)
        inp['geometry_metadata_sha256']=sha(geometry_directory/'metadata.json')
        inp['geometry_depth_sha256']=sha(geometry_directory/'depth.npz')
        inputs.append((r,inp))
    write_json(out/f'preflight_{a.shard_index}.json',{'records':len(rows),'profile_sha256':ph,
        'image_bindings_verified':True,'model_inputs':['first_frame_RGB','frozen_caption','intrinsics','prescribed_camera','depth_predicted_only_from_same_first_RGB'],
        'gt_depth_input':False,'estimated_depth_from_same_i0':True,
        'memory_input':False,'seed_reset_each_record':True})
    import torch, torch.distributed as dist
    # The frozen source unconditionally wraps the text encoder and VAE with
    # torch.compile.  On the recovered HOST runtime, Inductor raises
    # TritonMissing before the first inference even though Triton is importable.
    # Keep the same modules, weights, dtypes and model calls, but make the two
    # compile wrappers identity operations for this explicitly recorded repair.
    original_torch_compile = torch.compile
    torch.compile = lambda module, *args, **kwargs: module
    from diffusers import UniPCMultistepScheduler
    # Four independent models must not inherit one scheduler rendezvous endpoint.
    os.environ['MASTER_ADDR']='127.0.0.1';os.environ['MASTER_PORT']=str(29700+a.shard_index)
    os.environ['RANK']='0';os.environ['WORLD_SIZE']='1';os.environ['LOCAL_RANK']='0'
    dist.init_process_group('nccl',rank=0,world_size=1); torch.cuda.set_device(0)
    sys.path.insert(0,str(Path(a.source)/'hyworld2/worldgen'))
    from models.worldstereo_wrapper import WorldStereo
    from models.camera import get_camera_embedding
    device=torch.device('cuda:0'); start=time.time()
    model=WorldStereo.from_pretrained(a.weights,subfolder='worldstereo-camera',local_files_only=True,device=device)
    pipe=model.pipeline
    # The wrapper disables rescaling, but this adapter supplies uint8 PIL RGB.
    # CLIP needs [0,1] before its channel normalization; keep VAE/PIL input intact.
    if base_profile.get('clip_do_rescale') is not True:
        raise ValueError('PIL RGB input requires explicit CLIP rescaling in this adapter')
    pipe.image_processor.do_rescale=True
    if abs(float(pipe.image_processor.rescale_factor)-1/255)>1e-10:
        raise ValueError('unexpected CLIP rescaling factor')
    # Native loading permits missing keys; camera-specific weights must actually be present.
    from safetensors import safe_open
    state=pipe.transformer.state_dict()
    with safe_open(str(Path(a.weights)/'worldstereo-camera/model.safetensors'),framework='pt',device='cpu') as checkpoint:
        checkpoint_keys=set(checkpoint.keys())
        expected_control={k for k in state if k.startswith('controlnet.')}
        missing_control=expected_control-checkpoint_keys
        if missing_control:raise RuntimeError('camera checkpoint missing control weights: '+str(sorted(missing_control)[:10]))
        projection='controlnet.controlnet_mask_embedding.mask_zero_proj.weight'
        reference=checkpoint.get_tensor(projection).to(device=state[projection].device,dtype=state[projection].dtype)
        if not torch.equal(reference,state[projection]) or not torch.isfinite(reference).all() or reference.abs().max()==0:
            raise RuntimeError('camera projection weights were not loaded or are invalid')
        projection_abs_max=float(reference.abs().max());del reference
    del state
    scheduler_config=dict(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    write_json(out/f'loaded_{a.shard_index}.json',{'elapsed':time.time()-start,
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'pipeline':type(pipe).__name__,
        'control_checkpoint_keys_verified':len(expected_control),'camera_projection_abs_max':projection_abs_max,
        'torch_compile_mode':compile_mode,'original_torch_compile':repr(original_torch_compile)})
    completed=0
    for ordinal,(r,inp) in enumerate(inputs):
        record_contract=dict(contract,geometry_metadata=inp['geometry_metadata_sha256'],
                             geometry_depth=inp['geometry_depth_sha256'])
        contract_sha=hashlib.sha256(json.dumps(record_contract,sort_keys=True).encode()).hexdigest()
        profile=dict(base_profile)
        if 'translation_probe_value' in r:
            profile['translation_per_block']=r['translation_probe_value']
            profile['id']+='__translation_'+format(r['translation_probe_value'],'.6g')
        effective_ph=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()
        dest=out/'records'/r['output_relpath'];dest.mkdir(parents=True,exist_ok=True)
        done=dest/'completion.json'; video=dest/'native_generated_full_horizon.mp4'
        if done.exists():
            d=json.loads(done.read_text())
            if d.get('contract_sha256')!=contract_sha or d['image_sha256']!=r['condition']['image_sha256']:
                raise RuntimeError('resume contract changed; use a new output root')
            if sha(video)!=d['video_sha256']:raise RuntimeError('completed video corrupted')
            completed+=1;continue
        seed=inp['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        pipe.scheduler=UniPCMultistepScheduler.from_config(scheduler_config)
        controls=compile_trajectory(inp['semantic_blocks'],
            translation_per_unit=profile['translation_per_block'],
            yaw_degrees_per_unit=profile['yaw_degrees_per_block'],
            keyframe_intervals_per_unit=profile['keyframe_intervals_per_block'],
            tail_hold_intervals=profile.get('tail_hold_intervals',0))
        if r['kind']=='TASK2_ATOMIC':
            controls['checkpoint_doses']=list(range(1,17))
            controls['checkpoint_frame_indices']=[d*profile['keyframe_intervals_per_block'] for d in range(1,17)]
        F=int(controls['output_frame_count']); H=int(profile['height']); W=int(profile['width'])
        image=Image.open(inp['image_path']).convert('RGB')
        sw,sh=image.size
        if (sw,sh)!=(inp['camera_intrinsics']['source_width'],inp['camera_intrinsics']['source_height']):
            raise RuntimeError('camera and image size mismatch')
        image=image.resize((W,H),Image.Resampling.BICUBIC)
        image.save(dest/'conditioning_frame.png')
        K=np.array(inp['camera_intrinsics']['K_pixel'],dtype=np.float32)
        # OpenCV resize maps pixel centres as (u+0.5)*s-0.5.
        A=np.array([[W/sw,0,.5*(W/sw-1)],[0,H/sh,.5*(H/sh-1)],[0,0,1]],np.float32)
        Kcv=A@K
        # Preserve native asymmetric ray grid (x-0.5, y+0.5) with explicit K conversion.
        Knative=Kcv.copy();Knative[0,2]-=.5;Knative[1,2]+=.5
        ks=torch.tensor(np.repeat(Knative[None],F,axis=0),device=device)
        w2c=torch.tensor(controls['w2c'],dtype=torch.float32,device=device)
        emb=get_camera_embedding(ks,w2c,F,H,W,normalize=False,is_w2c=True)
        if not torch.isfinite(emb).all():raise ValueError('invalid camera embedding')
        direction_norm=torch.linalg.vector_norm(emb[:,:3],dim=1)
        if not torch.allclose(direction_norm,torch.ones_like(direction_norm),atol=1e-4,rtol=1e-4):
            raise ValueError('native camera embedding invalid or silently zeroed')
        embedding_moment_rms=float(emb[:,3:].square().mean().sqrt())
        del direction_norm
        from hyworld2_rgbd_render import build_render_inputs
        renderer_profile=dict(profile,expected_source_image_sha256=r['condition']['image_sha256'])
        rv,rm,geometry_meta=build_render_inputs(np.asarray(image).copy(),Kcv,w2c,
                                               inp['geometry_directory'],device,renderer_profile)
        if geometry_meta['geometry_metadata_sha256']!=inp['geometry_metadata_sha256'] or geometry_meta['depth_npz_sha256']!=inp['geometry_depth_sha256']:
            raise ValueError('frozen geometry cache changed after worker preflight')
        rv=rv.to(dtype=torch.bfloat16);rm=rm.to(dtype=torch.bfloat16)
        if tuple(rv.shape)!=(1,3,F,H,W) or tuple(rm.shape)!=(1,1,F,H,W):
            raise ValueError('point-cloud render input shape mismatch')
        write_json(dest/'geometry_conditioning.json',geometry_meta)
        if profile.get('save_condition_render',False):
            import imageio.v2 as imageio
            render_frames=((rv[0].permute(1,2,3,0).float().cpu().numpy()+1)*127.5).round().clip(0,255).astype(np.uint8)
            imageio.mimwrite(dest/'conditioning_render.mp4',render_frames,fps=profile['export_fps'],codec='libx264',quality=8,macro_block_size=1)
            Image.fromarray(render_frames[-1]).save(dest/'conditioning_render_last.png')
            del render_frames
        np.savez_compressed(dest/'prescribed_camera.npz',w2c=controls['w2c'],K_cv=Kcv,K_native=Knative,image_transform=A)
        write_json(dest/'executed_record.json',{'record_id':r['record_id'],'kind':r['kind'],
            'case_id':r['case_id'],'seed':seed,'semantic_blocks':inp['semantic_blocks'],
            'profile':profile,'profile_sha256':ph,'effective_profile_sha256':effective_ph,
            'weights_provenance':provenance,
            'development_probe_of':r.get('probe_of'),'camera_controls':{k:v for k,v in controls.items() if k not in ('w2c','c2w')},
            'model_image_sha256':sha(dest/'conditioning_frame.png'),'image_sha256':r['condition']['image_sha256'],
            'torch_compile_mode':compile_mode,
            'source_camera_is_not_generated_ground_truth':True,'model_variant':'worldstereo-camera',
            'geometry_source':'single current I0 only; frozen DA3 predicted depth, no GT or other images',
            'geometry_conditioning':geometry_meta,
            'memory_reset':'camera pipeline has no memory bank; scheduler and RNG reset each record',
            'native_num_frames':4*(F-1)+1,'actual_output_keyframes':F,
            'render_first_frame':'condition_RGB','render_future_value':'I0 point-cloud reprojection; unknown background 0',
            'unknown_render_mask_value':1,'known_render_mask_value':0,
            'camera_normalize':False,'native_pixel_grid_correction':[-.5,.5],
            'prescribed_embedding_moment_rms':embedding_moment_rms})
        t=time.time();print(json.dumps({'state':'generating','record_id':r['record_id'],'frames':F,'ordinal':ordinal}),flush=True)
        def denoise_progress(pipeline, step, timestep, callback_kwargs):
            if step==0 or (step+1)%10==0 or step+1==profile['steps']:
                progress={'state':'generating','record_id':r['record_id'],'step':step+1,
                          'steps':profile['steps'],'elapsed_seconds':time.time()-t,
                          'complete':completed,'assigned':len(inputs)}
                write_json(out/f'status_{a.shard_index}.json',progress)
                print(json.dumps(progress),flush=True)
            return callback_kwargs
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                arr=model(image=image,render_video=rv,render_mask=rm,camera_embedding=emb,
                    extrinsics=None,intrinsics=None,prompt=inp['prompt'],
                    negative_prompt=profile['negative_prompt'],height=H,width=W,
                    num_frames=4*(F-1)+1,num_inference_steps=profile['steps'],guidance_scale=profile['guidance_scale'],
                    generator=torch.Generator(device=device).manual_seed(seed),
                    latent_cond_mode='first_frame_only',output_type='np',
                    callback_on_step_end=denoise_progress,callback_on_step_end_tensor_inputs=[]).frames[0]
            arr=np.asarray(arr)
            if arr.shape!=(F,H,W,3) or not np.isfinite(arr).all():raise ValueError(f'output shape or finite {arr.shape}')
            arr=np.clip(np.rint(arr*255),0,255).astype(np.uint8)
            import imageio.v2 as imageio
            temp=video.with_name('partial.mp4')
            imageio.mimwrite(temp,arr,fps=profile['export_fps'],codec='libx264',quality=8,macro_block_size=1)
            frames=decode_video(temp)
            if len(frames)!=F or frames[0].shape!=(H,W,3):raise ValueError('strict decode mismatch')
            temp.replace(video)
            Image.fromarray(arr[0]).save(dest/'native_first_frame.png');Image.fromarray(arr[-1]).save(dest/'native_last_frame.png')
            write_json(done,{'record_id':r['record_id'],'profile_sha256':ph,'effective_profile_sha256':effective_ph,'contract_sha256':contract_sha,'contract':record_contract,'image_sha256':r['condition']['image_sha256'],
                'video_sha256':sha(video),'frames':F,'height':H,'width':W,'elapsed_seconds':time.time()-t,
                'seed':seed,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'formal_metric_scoring_allowed':False})
            if r.get('repeat_of'):
                first=next(x for x in rows if x['record_id']==r['repeat_of'])
                baseline=decode_video(out/'records'/first['output_relpath']/'native_generated_full_horizon.mp4')
                delta=np.abs(np.stack(baseline).astype(np.float32)-np.stack(frames).astype(np.float32))
                write_json(dest/'independence_repeat.json',{'repeat_of':r['repeat_of'],
                    'mean_absolute_decoded_pixel_difference':float(delta.mean()),
                    'max_absolute_decoded_pixel_difference':float(delta.max()),
                    'identical_decoded_video':bool(delta.max()==0),
                    'scope':'same seed and controls after intervening different records, same process'})
            completed+=1
            write_json(out/f'status_{a.shard_index}.json',{'state':'running','complete':completed,'assigned':len(inputs),
                'last_record':r['record_id'],'last_seconds':time.time()-t})
            print(json.dumps({'state':'complete','record_id':r['record_id'],'seconds':time.time()-t}),flush=True)
        except Exception:
            write_json(dest/'failure.json',{'record_id':r['record_id'],'traceback':traceback.format_exc()})
            raise
        finally:
            del emb,rv,rm,w2c,ks;torch.cuda.empty_cache()
    write_json(out/f'status_{a.shard_index}.json',{'state':'succeeded','complete':completed,'assigned':len(inputs)})
    dist.destroy_process_group()

if __name__=='__main__':main()
