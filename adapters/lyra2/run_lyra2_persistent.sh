#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_ROOT=${1:?usage: run_lyra2_persistent.sh BUNDLE_ROOT OUTPUT_ROOT GROUP_INDEX ASSIGNMENT}
OUT=${2:?usage: run_lyra2_persistent.sh BUNDLE_ROOT OUTPUT_ROOT GROUP_INDEX ASSIGNMENT}
GROUP=${3:?usage: run_lyra2_persistent.sh BUNDLE_ROOT OUTPUT_ROOT GROUP_INDEX ASSIGNMENT}
ASSIGNMENT=${4:?usage: run_lyra2_persistent.sh BUNDLE_ROOT OUTPUT_ROOT GROUP_INDEX ASSIGNMENT}
[[ "$GROUP" =~ ^[0-3]$ ]] || { echo "group must be 0..3" >&2; exit 2; }

PROFILE=${SW_LYRA_PROFILE:-strideworld_10f_dmd_t010_yaw2}
ROOT=${SW_ROOT:-/<path>/anon}
PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PY=${SW_LYRA_PYTHON:-$ROOT/lgt/envs/cosmos-framework-5e67049-cu128-persistent/bin/python}
SOURCE=${SW_LYRA_SOURCE_ROOT:-$ROOT/projects/worldbench_16ue5_6eth3d_v9_20260906/lyra_build/lyra_source_patched}
WEIGHTS=${SW_LYRA_WEIGHTS_ROOT:-$ROOT/models/strideworld/weights/Lyra-2.0}
DA3_SOURCE=${SW_DA3_SOURCE_ROOT:-$ROOT/lgt/third_party/strideworld/Depth-Anything-3/src}
DA3_MODEL=${SW_DA3_MODEL_ROOT:-$ROOT/models/strideworld/weights/AlayaWorld/DA3NESTED-GIANT-LARGE-1.1}
PKG=${SW_LYRA_PACKAGE_ROOT:-$ROOT/projects/strideworld/python_pkgs/lyra_py313_extra}
RUNNER=$PACKAGE_DIR/lyra2_persistent_batch.py

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#GPU_NAMES[@]} -eq 8 ]] || { echo "expected 8 visible GPUs, found ${#GPU_NAMES[@]}" >&2; exit 2; }
for name in "${GPU_NAMES[@]}"; do [[ "$name" == *GPU* ]] || { echo "expected GPU, found: $name" >&2; exit 2; }; done

"$PY" - "$BUNDLE_ROOT" "$ASSIGNMENT" "$PROFILE" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,assignment,profile=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
a=json.load(open(assignment)); assert a["status"]=="PASS" and a["profile"]==profile and a["record_count"]==3640 and a["num_shards"]==32
seen=set()
for seed,expected in ((0,1960),(1,840),(2,840)):
 p=root/f"seed_{seed}/lyra2/manifest.json"; d=json.load(open(p))
 assert d["profile"]==profile and d["seed"]==seed and len(d["records"])==expected
 h=hashlib.sha256(p.read_bytes()).hexdigest(); assert h==a["manifest_sources"][str(seed)]["sha256"]
 for r in d["records"]: assert r["record_id"] not in seen; seen.add(r["record_id"])
assert len(seen)==3640
assigned=[x["record_id"] for shard in a["shards"] for x in shard["records"]]
assert len(assigned)==len(set(assigned))==3640 and set(assigned)==seen
PY

mkdir -p "$OUT/logs/group_$GROUP" "$OUT/receipts" "$OUT/sample_manifests/group_$GROUP" "$OUT/runtime"
exec 8>"$OUT/.group_${GROUP}.lock"
flock -n 8

# Copy immutable weights once per allocated node. Eight workers then read local
# storage instead of independently streaming the same 91 GiB tree from GPFS.
WEIGHT_MANIFEST=$WEIGHTS/strideworld_download_manifest.jsonl
CACHE_KEY=$(sha256sum "$WEIGHT_MANIFEST" | awk '{print substr($1,1,16)}')
CACHE_ROOT=/tmp/strideworld_lyra2_${CACHE_KEY}
CACHE_LOCK=/tmp/strideworld_lyra2_${CACHE_KEY}.lock
exec 9>"$CACHE_LOCK"
flock 9
if [[ ! -f "$CACHE_ROOT/CACHE_COMPLETE.json" ]]; then
  CACHE_TMP=${CACHE_ROOT}.tmp.$$
  rm -rf "$CACHE_TMP" "$CACHE_ROOT"
  # Keep the optimization fail-fast: a full source copy plus 10 GiB of
  # headroom must fit on the node-local filesystem before touching it.
  WEIGHTS_BYTES=$(du -sb "$WEIGHTS" | awk '{print $1}')
  DA3_BYTES=$(du -sb "$DA3_MODEL" | awk '{print $1}')
  REQUIRED_BYTES=$((WEIGHTS_BYTES + DA3_BYTES + 10 * 1024 * 1024 * 1024))
  AVAILABLE_BYTES=$(df -PB1 /tmp | awk 'NR==2 {print $4}')
  if (( AVAILABLE_BYTES < REQUIRED_BYTES )); then
    echo "node-local cache insufficient: available=$AVAILABLE_BYTES required=$REQUIRED_BYTES" >&2
    exit 72
  fi
  mkdir -p "$CACHE_TMP"
  cp -a "$WEIGHTS" "$CACHE_TMP/weights"
  cp -a "$DA3_MODEL" "$CACHE_TMP/da3"
  "$PY" - "$WEIGHTS" "$DA3_MODEL" "$CACHE_TMP/weights" "$CACHE_TMP/da3" "$CACHE_TMP/CACHE_COMPLETE.json" "$CACHE_KEY" <<'PY'
import json,sys,time
from pathlib import Path
def stats(root):
 files=[p for p in Path(root).rglob('*') if p.is_file()]
 return {"files":len(files),"bytes":sum(p.stat().st_size for p in files)}
sw,sd,dw,dd=map(Path,sys.argv[1:5]); receipt=Path(sys.argv[5]); key=sys.argv[6]
a,b,c,d=stats(sw),stats(sd),stats(dw),stats(dd)
assert a==c,(a,c); assert b==d,(b,d)
receipt.write_text(json.dumps({"status":"PASS","cache_key":key,"weights":a,"da3":b,"created_at":time.time()},indent=2)+"\n")
PY
  mv "$CACHE_TMP" "$CACHE_ROOT"
fi
flock -u 9

RUNTIME=$OUT/runtime/group_${GROUP}_${HOSTNAME}
mkdir -p "$RUNTIME"
ln -sfn "$SOURCE/lyra_2" "$RUNTIME/lyra_2"
ln -sfn "$CACHE_ROOT/weights/checkpoints" "$RUNTIME/checkpoints"

"$PY" - "$BUNDLE_ROOT" "$OUT" "$GROUP" "$ASSIGNMENT" <<'PY'
import json,sys
from pathlib import Path
bundle,out,group,assignment=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]),Path(sys.argv[4])
manifests={}; records={}
for seed in (0,1,2):
 d=json.load(open(bundle/f"seed_{seed}/lyra2/manifest.json")); manifests[seed]=d
 for r in d["records"]: records[r["record_id"]]=(seed,r)
a=json.load(open(assignment))
for slot in range(8):
 shard=group*8+slot; target=a["shards"][shard]; assert target["shard"]==shard
 samples=[]
 for item in target["records"]:
  seed,r=records[item["record_id"]]
  assert seed==item["seed"] and int(r["num_frames"])==item["num_frames"]
  root=bundle/f"seed_{seed}/lyra2"
  video=out/f"generated/seed_{seed}/shard_{shard}/length_{r['num_frames']}/{r['record_id']}.mp4"
  samples.append({"image_path":r["source_image"],"trajectory_path":str(root/"trajectories"/(r["record_id"]+".npz")),"captions_path":str(root/"captions"/(r["record_id"]+".json")),"output_path":str(video),"num_frames":int(r["num_frames"]),"seed":seed,"record":r})
 p=out/f"sample_manifests/group_{group}/gpu_{slot}.json"; tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(samples)+"\n"); tmp.replace(p)
PY

PIDS=()
for slot in 0 1 2 3 4 5 6 7; do
  manifest=$OUT/sample_manifests/group_$GROUP/gpu_${slot}.json
  (
    cd "$RUNTIME"
    export CUDA_VISIBLE_DEVICES="$slot"
    export HF_HOME="$CACHE_ROOT/weights/hf_cache"
    export HF_HUB_CACHE="$CACHE_ROOT/weights/hf_cache"
    export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
    export PYTHONPATH="$SOURCE:$DA3_SOURCE:$PKG:$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$PY" -u "$RUNNER" \
      --sample_manifest "$manifest" \
      --checkpoint_dir checkpoints/model --experiment lyra2 \
      --output_path "$OUT/generated" --resolution 480,832 --fps 16 \
      --pose_scale 1.0 --num_sampling_step 35 \
      --da3_model_name "$CACHE_ROOT/da3" --no-use_moge_scale --use_dmd
  ) >"$OUT/logs/group_$GROUP/gpu_${slot}.log" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
"$PY" - "$OUT" "$GROUP" "$status" "$PROFILE" "$ASSIGNMENT" <<'PY'
import json,sys,time
from pathlib import Path
out,group,status,profile,assignment=Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4],Path(sys.argv[5])
assigned=complete=0; missing=[]
for slot in range(8):
 samples=json.load(open(out/f"sample_manifests/group_{group}/gpu_{slot}.json")); assigned+=len(samples)
 for s in samples:
  stamp=Path(s["output_path"]).with_suffix('.completion.json')
  if stamp.is_file(): complete+=1
  else: missing.append(s["record"]["record_id"])
r={"status":"PASS" if status==0 and complete==assigned else "FAIL","profile":profile,"group":group,"assigned":assigned,"complete":complete,"worker_status":status,"missing":missing,"assignment":str(assignment),"time":time.time()}
p=out/"receipts"/f"group_{group}.json"; tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(r,indent=2)+"\n"); tmp.replace(p)
print(json.dumps(r,indent=2)); raise SystemExit(0 if r["status"]=="PASS" else 1)
PY
