"""Strict video completion/seed binding; no model or CUDA import."""
import hashlib,json,time
from pathlib import Path
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(rec,seed):
 return {'record_id':rec['record_id'],'seed':int(seed),'source_image_sha256':rec['source_image_sha256'],'native_input_sha256':rec['native_input_sha256']}
def inspect_video(path,frames,width,height):
 import av
 with av.open(str(path)) as container:
  stream=container.streams.video[0];stream.codec_context.options={'err_detect':'explode'};n=0
  for f in container.decode(stream):
   if (f.width,f.height)!=(width,height):raise ValueError('decoded geometry mismatch')
   n+=1
 if n!=frames:raise ValueError(f'decoded frame count {n}, expected {frames}')
 return {'frames':n,'width':width,'height':height}
def seal_output(path,rec,seed,frames,width,height):
 path=Path(path);d={**binding(rec,seed),'decoded':inspect_video(path,frames,width,height),'video_sha256':digest(path),'status':'DECODED_AND_BOUND'}
 stamp=path.with_suffix('.completion.json');tmp=stamp.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(stamp)
def is_complete(path,rec,seed,frames,width,height):
 path=Path(path)
 try:
  d=json.loads(path.with_suffix('.completion.json').read_text())
  if any(d.get(k)!=v for k,v in binding(rec,seed).items()):return False
  if d.get('video_sha256')!=digest(path):return False
  return inspect_video(path,frames,width,height)==d.get('decoded')
 except Exception:return False

def quarantine_partial(path):
 path=Path(path)
 if path.exists():
  dst=path.parent/'partial_attempts';dst.mkdir(exist_ok=True);suffix=str(time.time_ns())
  path.rename(dst/(path.name+'.'+suffix))
  stamp=path.with_suffix('.completion.json')
  if stamp.exists():stamp.rename(dst/(stamp.name+'.'+suffix))
