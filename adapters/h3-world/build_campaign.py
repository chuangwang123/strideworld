from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from h3_controls import profile,canonical_hash
def validate_manifest(document,parent_path):
    if document.get("schema")!="strideworld-h3-campaign-v1": raise ValueError("bad schema")
    if document["profile"]!=profile(document["profile"]["mode"]): raise ValueError("profile mismatch")
    rows=document.get("records",[])
    seed=int(document["campaign_id"][-1])
    # a extra-seed subset campaign carries its own record count; frozen campaigns keep the fixed counts.
    expected=int(document['counts']['records']) if str(document.get('campaign_id','')).startswith('w3b_') else {0:1960,1:840,2:840}[seed]
    if len(rows)!=expected: raise ValueError("record count")
    if len({r.get("record_index") for r in rows})!=len(rows): raise ValueError("record order")
    if any(r.get("seed")!=seed for r in rows): raise ValueError("seed mismatch")
    plan=document.get("v10_plan",{})
    if plan.get("authoritative") is not True or plan.get("source_campaign_sha256") is None:
        raise ValueError("missing authoritative v10_plan pair contract")
    if plan.get("controls_compiled_at_runtime") is not True or plan.get("cpu_control_horizon_checked")!=16:
        raise ValueError("missing control compilation contract")
    for r in rows:
        blocks=r.get("semantic_blocks",[])
        if r.get("seed") not in (0,1,2) or not (1<=len(blocks)<=16): raise ValueError("bad horizon/seed")
        if not r.get("condition",{}).get("image_uri","").startswith("data://"): raise ValueError("image uri")
        flat=[]
        for step in r.get("steps",[]):
            acts=step.get("canonical_actions",[])
            dose=step.get("dose",1)
            if len(acts)==1 and isinstance(dose,(int,float)) and int(dose)==dose and dose>=1:
                flat.extend(acts*int(dose))
            else:
                flat.extend(acts)
        if flat!=blocks: raise ValueError("semantic blocks do not match steps "+str(r.get("record_id")))
        if r.get("kind")=="TASK2_ATOMIC":
            if not r.get("canonical_action") or not r.get("checkpoint_doses"): raise ValueError("atomic control fields missing "+str(r.get("record_id")))
        else:
            if not r.get("route_id") or not r.get("task2_dependency_ids"):
                raise ValueError("pair route/dependency fields missing "+str(r.get("record_id")))
        q=dict(r); old=q.pop("record_content_sha256",None)
        if old!=canonical_hash(q): raise ValueError("record hash "+str(r.get("record_id")))
    return rows
