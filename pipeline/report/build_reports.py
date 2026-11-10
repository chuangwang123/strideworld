"""Read-only, pinned CPU reporting from current immutable pose artifacts."""
from pathlib import Path
import argparse,collections,concurrent.futures,fcntl,hashlib,json,os,subprocess,sys,time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

def one(config,model):
 cfg=json.loads(Path(config).read_text());repo=Path(cfg['repo']);out=Path(cfg['root'])/model;source=Path(cfg['models'][model])
 out.mkdir(parents=True,exist_ok=True)
 lock=(out/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'COMPLETE.json').exists():return json.loads((out/'COMPLETE.json').read_text())
 for rel,digest in cfg['code_sha256'].items():assert sha(repo/rel)==digest,(rel,sha(repo/rel))
 env={**os.environ,'PYTHONPATH':os.pathsep.join([str(repo)]+[x for x in os.environ.get('PYTHONPATH','').split(os.pathsep) if x]),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
 index=source/'details/normalized_index.jsonl';rr=rows(index);workers=rows(source/'details/pose_worker_results.jsonl');byid={x['record_id']:x for x in workers}
 assert len(rr)==len(byid)==3640 and {x['record_id'] for x in rr}==set(byid)
 assert all(x['primary_key']['model']==model for x in rr)
 assert collections.Counter(x['primary_key']['task'] for x in rr)=={'task2':2520,'task3':1120}
 view=out/'source_view';(view/'details').mkdir(parents=True,exist_ok=True);(view/'artifacts').mkdir(exist_ok=True)
 def link(src,dst):
  if dst.exists() or dst.is_symlink():assert dst.resolve()==src.resolve(),(src,dst)
  else:dst.symlink_to(src)
 for name in ['normalized_index.jsonl','pose_worker_results.jsonl']:link(source/'details'/name,view/'details'/name)
 for name in ['task1','task2']:link(source/name,view/name)
 for x in rr:
  w=byid[x['record_id']];assert w['status']=='ok',w
  p=Path(w['pose']);assert p.is_file(),p
  link(p.parent,view/'artifacts'/x['record_id'])
 # Pin source-to-view mapping. Symlinks only: source videos/poses are unchanged.
 plan={}
 for x in rr:
  k=x['primary_key'];plan.setdefault(k['scene_id'],set()).add(k['case_id'])
 plan={s:sorted(v) for s,v in sorted(plan.items())}
 assert len(plan)==30 and sum(map(len,plan.values()))==140
 source_binding={'model':model,'results_root':str(source),'view':str(view),'normalized_index_sha256':sha(index),'pose_worker_results_sha256':sha(source/'details/pose_worker_results.jsonl'),'n_poses':3640,'plan':plan,'code_sha256':cfg['code_sha256'],'started_unix':time.time()}
 save(out/'SOURCE_BINDING.json',source_binding)
 unit=out/'task2_unit_horizons.json';pair=out/'task3_case_unit4.json'
 stages=[('unit',[sys.executable,str(repo/'metrics/task2/export_unit_horizons.py'),'--results-root',str(view),'--output',str(unit),'--model',model,'--min-pose-inlier-ratio','0.1','--min-pose-anchor-coverage','0.1']),('pair',[sys.executable,str(repo/'metrics/task3/evaluate_from_unit_json.py'),'--task2-json',str(unit),'--results-root',str(view),'--output-json',str(pair),'--track-ratio-threshold','0.1','--motion-ratio-threshold','0.1'])]
 for name,cmd in stages:
  print('STAGE',model,name,flush=True)
  with (out/(name+'.log')).open('w') as f:subprocess.run(cmd,env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=2400)
 entry={'model':model,'results_root':str(view),'task2_units':str(unit),'task3_pairs':str(pair),'task1':str(source/'task1/task1.json'),'expected_source_sha256':{'normalized_index':sha(index),'task2_units':sha(unit),'task3_pairs':sha(pair),'task1':sha(source/'task1/task1.json')}}
 manifest={'schema_version':'strideworld-paper-manifest-v1','calibration_horizon':4,'evaluation_doses':[12,16],'plan':plan,'expected_seeds':[0,1,2],'task3_pairs_per_case':4,'pose_quality':{'min_pose_inlier_ratio':0.1,'min_pose_anchor_coverage':0.1},'task3_quality':{'track_ratio_min':0.1,'motion_ratio_min':0.1},'ray_reference_deg':1.0,'scale_floors':{'translation':0.0,'rotation':0.0},'models':[entry]}
 save(out/'reporting_manifest.json',manifest)
 with (out/'report.log').open('w') as f:subprocess.run([sys.executable,'-m','metrics.paper_reporting','--manifest',str(out/'reporting_manifest.json'),'--output',str(out/'report')],env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=2400)
 report=json.loads((out/'report/report.json').read_text());v=report['models'][0]
 compact={'model':model,'main':report['main_table'][0],'source_sha256':v['source_sha256'],'manifest_sha256':report['manifest_sha256'],'source_binding':source_binding,'geometry_summary':v['geometry']['summary'],'atomic_summary':v['atomic']['summary'],'composition':v['composition'],'variability':v['variability'],'depth':{}}
 geometry=json.loads((source/'task1/task1.json').read_text())
 for strata in ['near','mid','far']:
  values=[]
  for x in geometry.get('cases',[]):
   r=x.get('row',{});err=r.get(strata+'_e_rig_ray_deg_mean');n=r.get(strata+'_e_rig_ray_deg_n',0)
   if isinstance(err,(int,float)) and err>=0 and n>0:values.append(100/(1+err))
  compact['depth'][strata]={'planned':1120,'measured':len(values),'score_fixed_denominator':sum(values)/1120}
 # Preserve exact normalized-index video bindings for cross-source visual check.
 save(out/'video_bindings.json',{x['record_id']:x['video_sha256'] for x in rr})
 save(out/'PAPER_INPUT.json',compact)
 final={'status':'PASS','model':model,'source_results':str(source),'reporter_commit':'release','core_commit':'release','n_videos':3640,'n_atomic_records':2520,'n_pairs':560,'calibration_horizon':4,'evaluation_doses':[12,16],'manifest_sha256':sha(out/'reporting_manifest.json'),'report_sha256':sha(out/'report/report.json'),'paper_input_sha256':sha(out/'PAPER_INPUT.json'),'video_bindings_sha256':sha(out/'video_bindings.json'),'completed_unix':time.time()}
 save(out/'COMPLETE.json',final);print('MODEL_COMPLETE',json.dumps(final),flush=True);return final

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--model');a=p.parse_args()
 if a.model:one(a.config,a.model);return
 c=json.loads(Path(a.config).read_text());root=Path(c['root']);root.mkdir(parents=True,exist_ok=True)
 lock=(root/'PIPELINE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def run(m):
  with (root/(m+'.log')).open('a') as f:
   q=subprocess.run([sys.executable,__file__,'--config',a.config,'--model',m],stdout=f,stderr=subprocess.STDOUT)
  return m,q.returncode
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:status=dict(pool.map(run,c['models']))
 final={'status':'PASS' if all(x==0 for x in status.values()) else 'FAIL','exit_codes':status,'config_sha256':sha(a.config),'completed_unix':time.time()};save(root/'PIPELINE_COMPLETE.json',final);print(json.dumps(final),flush=True)
 if final['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
