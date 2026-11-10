#!/usr/bin/env python3
"""Independent H3-World rollouts with fixed native video and input provenance."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from build_campaign import validate_manifest
from h3_controls import canonical_hash, compile_controls


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def inside(root,relative):
    root=Path(root).resolve();p=(root/relative).resolve()
    if Path(relative).is_absolute() or '..' in Path(relative).parts or not p.is_relative_to(root):
        raise ValueError('path escapes root')
    return p


def verify_runtime(path,pinned):
    if sha(path)!=pinned: raise ValueError('runtime contract changed')
    contract=json.loads(Path(path).read_text())
    for f in contract['files']:
        if sha(f['path'])!=f['sha256']: raise ValueError('runtime file changed: '+f['path'])
    for f in contract.get('verified_weight_files',[]):
        st=Path(f['path']).stat()
        if st.st_size!=f['bytes'] or st.st_mtime_ns!=f['mtime_ns_at_freeze']:
            raise ValueError('verified weight changed after freeze: '+f['path'])
    return contract


def verify_video(path,expected_frames):
    import av
    count=0
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        if abs(float(stream.average_rate)-24)>1e-6: raise ValueError('wrong output fps')
        for frame in container.decode(stream):
            if (frame.width,frame.height)!=(832,480): raise ValueError('wrong native dimensions')
            count+=1
    if count!=expected_frames: raise ValueError(f'native frame count {count} != {expected_frames}')
    return count


@contextmanager
def model_load_guard():
    # Full eight-GPU jobs serialize weight initialization to avoid eight
    # simultaneous checkpoint copies in host RAM. Sampling stays parallel.
    lock_path=os.environ.get('H3_MODEL_LOAD_LOCK')
    if not lock_path:
        yield
        return
    with open(lock_path,'a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True,type=Path);ap.add_argument('--weights',required=True,type=Path)
    ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--manifest-sha256',required=True)
    ap.add_argument('--parent-manifest',required=True,type=Path);ap.add_argument('--data-root',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--runtime-contract',type=Path);ap.add_argument('--runtime-contract-sha256')
    ap.add_argument('--shard-index',type=int,default=0);ap.add_argument('--shard-count',type=int,default=1)
    ap.add_argument('--smoke-selection',choices=['all','directions','long-route','requested7'],default='all')
    ap.add_argument('--accepted-profile',type=Path)
    ap.add_argument('--accepted-profile-sha256')
    ap.add_argument('--smoke-ids',default=''); ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    if not 0<=a.shard_index<a.shard_count: raise ValueError('invalid shard')
    if sha(a.manifest)!=a.manifest_sha256: raise ValueError('manifest hash differs from launch contract')
    doc=json.loads(a.manifest.read_text());rows=validate_manifest(doc,a.parent_manifest)
    is_smoke=bool(a.smoke_ids)
    if a.smoke_ids:
        wanted=set(a.smoke_ids.split(',')); rows=[r for r in rows if r['record_id'] in wanted]
        if {r['record_id'] for r in rows}!=wanted:raise ValueError('unknown smoke ids')
    if a.smoke_selection!='all' and not is_smoke: raise ValueError('smoke selection on full campaign')
    if not is_smoke and not a.dry_run:
        if not a.accepted_profile: raise ValueError('full generation requires observed smoke acceptance')
        if not a.accepted_profile_sha256 or sha(a.accepted_profile)!=a.accepted_profile_sha256:
            raise ValueError('smoke acceptance changed after submission')
        accepted=json.loads(a.accepted_profile.read_text())
        if accepted.get('accepted') is not True or accepted.get('profile_sha256')!=doc['profile']['profile_sha256']:
            raise ValueError('profile not accepted')
        requested_units=max(len(r['semantic_blocks']) for r in rows)
        observed_units=accepted.get('max_units_verified',0)
        cpu_units=accepted.get('max_units_cpu_checked',observed_units)
        if type(cpu_units) is not int or cpu_units < requested_units:
            raise ValueError('CPU control compilation does not cover the requested horizon')
        if type(observed_units) is not int or observed_units < min(requested_units,14):
            raise ValueError('longest observed smoke horizon is insufficient')
        if (accepted.get('motion_review_completed') is not True
                and accepted.get('control_mapping_review_completed') is not True):
            raise ValueError('observed control mapping review must precede full generation')
        if accepted.get('six_direction_review_completed') is not True:
            raise ValueError('six-direction control review must precede full generation')
        evidence=accepted.get('evidence',[])
        if not evidence:raise ValueError('smoke acceptance lacks concrete evidence')
        for item in evidence:
            if sha(item['path'])!=item['sha256']:raise ValueError('accepted smoke evidence changed')
    if a.smoke_selection=='directions': rows=[r for r in rows if len(r['semantic_blocks'])==8]
    if a.smoke_selection=='long-route': rows=[r for r in rows if len(r['semantic_blocks'])==14]
    if a.smoke_selection=='requested7': rows=[r for r in rows if r['kind']!='SMOKE_NOOP']
    rows=[r for r in rows if r['record_index']%a.shard_count==a.shard_index]
    a.output.mkdir(parents=True,exist_ok=True)
    lock=open(a.output/f'.worker_{a.shard_index}.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=a.output/f'status_{a.shard_index}.json'
    def status(state,**fields):
        write_json(status_path,{'state':state,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               'assigned':len(rows),'shard_index':a.shard_index,**fields})
    status('validating_inputs')
    images={}
    for row in rows:
        uri=row['condition']['image_uri']
        if not uri.startswith('data://'): raise ValueError('unexpected source image URI')
        path=inside(a.data_root,uri[7:])
        if str(path) not in images:
            images[str(path)]=sha(path)
        if images[str(path)]!=row['condition']['image_sha256']: raise ValueError('I0 hash mismatch')
        if hashlib.sha256(row['condition']['prompt'].encode()).hexdigest()!=row['condition']['prompt_sha256']:
            raise ValueError('prompt hash mismatch')
        inside(a.output/'records',row['output_relpath'])
    write_json(a.output/f'preflight_{a.shard_index}.json',{'passed':True,'records':len(rows),
        'manifest_sha256':a.manifest_sha256,'image_hashes_verified':len(images),
        'profile_sha256':doc['profile']['profile_sha256'],'expected_seed':rows[0]['seed'] if rows else None})
    if a.dry_run: status('preflight_passed');return
    if not a.runtime_contract or not a.runtime_contract_sha256:
        raise ValueError('inference requires a pinned runtime contract')
    runtime=verify_runtime(a.runtime_contract,a.runtime_contract_sha256)
    if runtime['source_root']!=str(a.source.resolve()) or runtime['weights_root']!=str(a.weights.resolve()):
        raise ValueError('source/weights roots differ from runtime contract')
    for name in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']:
        os.environ[name]='1'
    os.environ['ABOT_DIFFSYNTH_ROOT']=str(a.source/'DiffSynth-Studio-h3-v2')
    os.environ['ABOT_HEIGHT']='480';os.environ['ABOT_WIDTH']='832'
    os.environ['DIFFSYNTH_SKIP_DOWNLOAD']='True'
    for n in ['HF_HOME','XDG_CACHE_HOME','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TMPDIR']:
        path=a.output/'cache'/f'worker{a.shard_index}'/n.lower();path.mkdir(parents=True,exist_ok=True)
        os.environ[n]=str(path)
    sys.path.insert(0,str(a.source/'code/abot'))
    import infer as official
    import numpy as np
    import torch
    from diffsynth.utils.data.audio_video import write_video_audio
    if not torch.cuda.is_available(): raise RuntimeError('no allocated CUDA GPU')
    torch.cuda.set_device(0)
    checkpoint=a.weights/'H3-World/step-10000.safetensors'
    if sha(checkpoint)!='ddd9187b920b1e52c2d090f4e264fd83d8d433efc2a5b159e58883aeaf96e526':
        raise ValueError('official LoRA checksum mismatch')
    # Verify frozen native vocabulary against the loaded official annotator.
    if doc['profile']['mode']=='native_keys':
        for row in rows:
            expected=official.S.annotate_from_keys9(np.array(row['controls']['keys9'],dtype=np.int64))
            if expected!=row['controls']['action_script']: raise ValueError('native vocabulary mismatch')
    status('waiting_model_load_slot')
    with model_load_guard():
        status('loading_model')
        started=time.time();pipe=official.load_pipeline('cuda:0')
        lora=official.load_checkpoint_lora(checkpoint)
        pairs=sum('.lora_A.' in k for k in lora)
        before=sum(len(m.lora_A_weights) for m in pipe.dit.modules() if hasattr(m,'lora_A_weights'))
        pipe.load_lora(pipe.dit,state_dict=lora,hotload=True)
        after=sum(len(m.lora_A_weights) for m in pipe.dit.modules() if hasattr(m,'lora_A_weights'))
        if after-before!=pairs: raise RuntimeError('incomplete LoRA attachment')
        del lora
        write_json(a.output/f'loaded_{a.shard_index}.json',{'load_seconds':time.time()-started,
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'lora_pairs':pairs,
            'dit_source':official._dit.__file__,'pipeline_source':official._pipe.__file__,
            'runtime_contract_sha256':a.runtime_contract_sha256,'single_gpu_cpu_offload':True})
    completed=0
    for row in rows:
        dest=inside(a.output/'records',row['output_relpath']);dest.mkdir(parents=True,exist_ok=True)
        video_path=dest/'native_generated_full_horizon.mp4';done=dest/'completion.json'
        controls=compile_controls(row['semantic_blocks'],doc['profile']['mode'])
        if done.exists():
            d=json.loads(done.read_text())
            if d.get('runtime_contract_sha256')!=a.runtime_contract_sha256 or d.get('record_content_sha256')!=row['record_content_sha256']:
                raise ValueError('resume provenance differs')
            if d.get('seed')!=row['seed'] or d.get('profile_sha256')!=doc['profile']['profile_sha256'] or d.get('manifest_sha256')!=a.manifest_sha256:
                raise ValueError('resume seed/profile/manifest differs')
            for filename,key in [('controls.json','controls_sha256'),('executed_record.json','executed_record_sha256')]:
                if sha(dest/filename)!=d[key]: raise ValueError('resume execution evidence changed')
            expected_indices={0,controls['final_action_frame'],controls['num_frames']-1,*controls['dose_checkpoint_frames'].values()}
            if set(d.get('native_frame_sha256',{}))!={str(i) for i in expected_indices}:
                raise ValueError('resume native checkpoint inventory differs')
            for i,digest in d['native_frame_sha256'].items():
                if sha(dest/f'native_frame_{int(i):04d}.png')!=digest:raise ValueError('resume checkpoint image changed')
            if sha(video_path)!=d['video_sha256']:raise ValueError('resume video corrupted')
            verify_video(video_path,controls['num_frames']);completed+=1;continue
        seed=row['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        # __call__ resets both native schedulers on every independent record.
        image_path=inside(a.data_root,row['condition']['image_uri'][7:])
        image=official.load_first_frame(image_path);image.save(dest/'conditioning_frame.png')
        if image.size!=(832,480): raise ValueError('unexpected preprocessing dimensions')
        with official.Image.open(image_path) as source_image: sw,sh=source_image.size
        ksrc=row['source_intrinsics_provenance_only']
        if (sw,sh)!=(ksrc['source_width'],ksrc['source_height']): raise ValueError('source image/intrinsics size mismatch')
        scale=max(832/sw,480/sh);rw,rh=round(sw*scale),round(sh*scale)
        left,top=(rw-832)//2,(rh-480)//2;sx,sy=rw/sw,rh/sh
        transform=np.array([[sx,0,.5*(sx-1)-left],[0,sy,.5*(sy-1)-top],[0,0,1]])
        preprocessing={'source_width':sw,'source_height':sh,'resized_width':rw,'resized_height':rh,
            'crop_left':left,'crop_top':top,'target_width':832,'target_height':480,
            'actual_scale_x':sx,'actual_scale_y':sy,'pixel_center_convention':'first_pixel_center_is_0_0',
            'source_to_output_pixel_transform':transform.tolist(),
            'K_output_provenance_only':(transform@np.array(ksrc['K_pixel'])).tolist(),
            'intrinsics_used_for_generation':False}
        write_json(dest/'controls.json',controls)
        execution={'record':row,'native_config':doc['profile'],'input_image_sha256':row['condition']['image_sha256'],
            'conditioning_frame_sha256':sha(dest/'conditioning_frame.png'),
            'manifest_sha256':a.manifest_sha256,'runtime_contract_sha256':a.runtime_contract_sha256,
            'model_input_allowlist':['I0_RGB','fixed_scene_prompt','action_script','negative_action_script','seed'],
            'preprocessing':'official scale-to-cover Lanczos and center crop',
            'preprocessing_geometry':preprocessing,
            'seed_reset_each_record':True,'native_seed_rand_device':'cpu',
            'accepted_profile_sha256':a.accepted_profile_sha256 if not is_smoke else None,
            'intrinsics_depth_gt_history_used':False,'sampling':'whole-sequence bidirectional native model; not causal streaming'}
        write_json(dest/'executed_record.json',execution)
        torch.cuda.reset_peak_memory_stats()
        began=time.time()
        def progress(timesteps):
            for i,t in enumerate(timesteps):
                status('generating',record_id=row['record_id'],seed=seed,complete=completed,
                       step=i+1,steps=len(timesteps),elapsed_seconds=time.time()-began)
                yield t
            status('decoding',record_id=row['record_id'],seed=seed,complete=completed)
        try:
            video,audio=pipe(prompt=row['condition']['prompt'],negative_prompt=' ',
                height=480,width=832,num_frames=controls['num_frames'],num_inference_steps=50,
                seed=seed,keyframes=[image],keyframe_indices=[0],cfg_scale=1.0,
                action_script=controls['action_script'],negative_action_script=controls['negative_action_script'],
                action_pad_used_to=controls['action_pad_used_to'],
                progress_bar_cmd=progress)
            if len(video)!=controls['num_frames']: raise ValueError('pipeline returned wrong frame count')
            # Raw first output is kept. The provided image is never pasted into video.
            image_array=np.asarray(image,dtype=np.float32)
            first_mae=float(np.abs(np.asarray(video[0],dtype=np.float32)-image_array).mean())
            chosen=sorted({0,controls['final_action_frame'],controls['num_frames']-1,*controls['dose_checkpoint_frames'].values()})
            for idx in chosen: video[idx].save(dest/f'native_frame_{idx:04d}.png')
            temp=video_path.with_name(f'native_generated_full_horizon.{os.getpid()}.tmp.mp4')
            write_video_audio(video=video,audio=audio,output_path=str(temp),fps=24,audio_sample_rate=32000)
            temp.replace(video_path);count=verify_video(video_path,controls['num_frames'])
            receipt={'record_id':row['record_id'],'seed':seed,'record_content_sha256':row['record_content_sha256'],
                'profile_sha256':doc['profile']['profile_sha256'],'manifest_sha256':a.manifest_sha256,
                'runtime_contract_sha256':a.runtime_contract_sha256,'video_path':str(video_path),
                'accepted_profile_sha256':a.accepted_profile_sha256 if not is_smoke else None,
                'video_sha256':sha(video_path),'frame_count':count,'fps':24,'full_decode_checked':True,
                'controls_sha256':sha(dest/'controls.json'),'executed_record_sha256':sha(dest/'executed_record.json'),
                'native_frame_sha256':{str(i):sha(dest/f'native_frame_{i:04d}.png') for i in chosen},
                'first_native_frame_vs_conditioning_rgb_mae_255':first_mae,
                'elapsed_seconds':time.time()-began,'max_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'max_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
                'cuda_peak_scope':'reset immediately before this record generation',
                'formal_metric_scoring_allowed':False}
            write_json(done,receipt);completed+=1
            del video,audio
            status('record_complete',record_id=row['record_id'],seed=seed,complete=completed)
        except Exception as exc:
            write_json(dest/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'seed':seed})
            status('failed',record_id=row['record_id'],seed=seed,complete=completed,error=repr(exc))
            raise
    status('complete',complete=completed)
    write_json(a.output/f'worker_summary_{a.shard_index}.json',{'complete':completed==len(rows),'ready':completed,
        'expected':len(rows),'manifest_sha256':a.manifest_sha256,'runtime_contract_sha256':a.runtime_contract_sha256})


if __name__=='__main__': main()
