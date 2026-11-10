"""Appendix analyses for the response-gated Pace score: shared references (the model-wide stride references of the appendix),
measurable cohort and video length (the response-profile discussion of the appendix), and paired scene-bootstrap intervals.
Inputs: the --data directory of compute_metrics.py (adds per_checkpoint_scores.csv) and outputs/per_video_response.csv.gz.
Every model of the registry (registry_release.json) is added after the thirteen to the shared-reference and cohort tables,
using its per-checkpoint scores from checkpoint_scores.py (per_checkpoint_out), its main_per_video export, its per-video
Response (extra_model.py, per_video_response_out) and its frames per 16 steps (registry `frames`).
python3 appendix_analyses.py [--data <data13>] [--out outputs]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from compute_metrics import ORDER, KEY, load, respond
import registry as reg

FRAMES = {'matrix-game-2': 64, 'matrix-game-3': 64, 'matrix-game-3.5': 128, 'sana-wm': 64, 'dreamx-world': 64,
          'hunyuan-world-1.5': 64, 'hy-world-2': 32, 'lingbot-world-v2': 64, 'alaya-world': 128, 'h3-world': 272,
          'lyra2': 160, 'evoke-turbo': 128, 'worldcrafter-fast': 256}
FRAMES.update({m.key: m.frames for m in reg.MODELS})

ap = argparse.ArgumentParser(); ap.add_argument('--data', default=str(reg.BASE)); ap.add_argument('--out', default=str(reg.OUT))
a = ap.parse_args(); out = Path(a.out)
V = pd.read_csv(out / 'per_video_response.csv.gz')
C = pd.read_csv(Path(a.data) / 'per_checkpoint_scores.csv')
MPV = pd.read_csv(Path(a.data) / 'main_per_video.csv')
ALL = list(ORDER)
if reg.MODELS:
    C = pd.concat([C] + [pd.read_csv(out / m.per_checkpoint_out) for m in reg.MODELS])
    MPV = pd.concat([MPV] + [reg.read_csv(m, 'main_per_video') for m in reg.MODELS])
    V = pd.concat([V] + [pd.read_csv(out / m.per_video_response_out) for m in reg.MODELS]); ALL += reg.KEYS
C = C.merge(V[KEY + ['resp']], on=KEY, how='left')
assert C.resp.notna().all()
# ---- shared references ----
rows = []
for m in ALL:
    g = C[C.model_key == m]
    pv = g.groupby('record_id').agg(weight=('weight', 'first'), action=('action', 'first'), resp=('resp', 'first'),
                                    main=('main', lambda x: x.fillna(0).mean()), fixed=('fixed', lambda x: x.fillna(0).mean()),
                                    axis_only=('axis', lambda x: x.fillna(0).mean()))
    w = pv.weight / pv.weight.sum()
    r = dict(model_key=m)
    for c in ('main', 'fixed', 'axis_only'):
        r[c + '_ungated'] = (w * pv[c]).sum(); r[c] = (w * pv[c] * pv.resp).sum()
    tr = pv.action.isin(['W', 'S', 'A', 'D'])
    for nm, sel in (('trans', tr), ('turn', ~tr)):
        ww = pv.weight[sel] / pv.weight[sel].sum()
        r['d_' + nm] = (ww * (pv.fixed[sel] - pv.main[sel]) * pv.resp[sel]).sum()
    rows.append(r)
RS = pd.DataFrame(rows).set_index('model_key').loc[ALL]
RS['rank_main'] = RS.main.rank(ascending=False).astype(int); RS['rank_fixed'] = RS.fixed.rank(ascending=False).astype(int)
RS['rank_axis'] = RS.axis_only.rank(ascending=False).astype(int)
RS.to_csv(out / 'reference_scores.csv')
print(RS.round(1).to_string())
print('spearman main-fixed', spearmanr(RS.main, RS.fixed), 'kendall', kendalltau(RS.main, RS.fixed))
print('ungated spearman main-fixed', spearmanr(RS.main_ungated, RS.fixed_ungated))
# ---- measurable cohort and length ----
rows = []
for m in ALL:
    S = MPV[(MPV.model_key == m) & MPV.both_scorable.astype(bool)].merge(V[KEY + ['resp']], on=KEY)
    w = S.weight / S.weight.sum()
    s12 = (w * S.main12 * S.resp).sum(); s16 = (w * S.main16 * S.resp).sum()
    rows.append(dict(model_key=m, frames=FRAMES[m], n=len(S), step12=s12, step16=s16, fall=s12 - s16, mean=(s12 + s16) / 2))
L = pd.DataFrame(rows).set_index('model_key').loc[ALL]
L.to_csv(out / 'length_cohort.csv')
print(L.round(1).to_string())
print('fall vs frames', spearmanr(L.frames, L.fall), 'step16 vs frames', spearmanr(L.frames, L.step16))
# ---- paired scene bootstrap ----
A, R = load(a.data, 'all_scores.json', 'initial_view_depths.csv', 'obs/*.observations.csv')
A['resp'] = respond(A); A['hold'] = A.atomic * A.resp
scenes = sorted(A.scene_id.unique())
R['rs'] = R.rscore.where(R.route_valid, 0)
P = R.groupby(['model_key', 'scene_id', 'case_id', 'pair_id']).agg(pv=('pair_valid', 'first'), pt=('pair_trans', 'first'),
                                                                    py=('pair_yaw', 'first')).reset_index()
P['ps'] = ((P.pt + P.py) / 2).where(P.pv, 0)
hs = A.groupby(['model_key', 'scene_id']).apply(lambda g: pd.Series(dict(wh=(g.weight * g.hold).sum(), w=g.weight.sum())))
rs = R.groupby(['model_key', 'scene_id']).rs.agg(['sum', 'count'])
ps = P.groupby(['model_key', 'scene_id']).ps.agg(['sum', 'count'])


def metric(m, mult):
    h = hs.loc[m].reindex(scenes); r = rs.loc[m].reindex(scenes); p = ps.loc[m].reindex(scenes)
    hold = (h.wh * mult).sum() / (h.w * mult).sum()
    route = (r['sum'] * mult).sum() / (r['count'] * mult).sum(); pair = (p['sum'] * mult).sum() / (p['count'] * mult).sum()
    return dict(hold=hold, route=route, pair=pair, composition=(route + pair) / 2)


rng = np.random.default_rng(20261027)
comps = [('alaya-world', 'lyra2', 'hold'), ('hy-world-2', 'lyra2', 'composition'), ('hy-world-2', 'lyra2', 'route'),
         ('hy-world-2', 'lyra2', 'pair'), ('lyra2', 'hy-world-2', 'hold')]
one = np.ones(len(scenes))
res = []
for m1, m2, k in comps:
    d0 = metric(m1, one)[k] - metric(m2, one)[k]
    bs = []
    for _ in range(2000):
        idx = rng.integers(0, len(scenes), len(scenes)); mult = np.bincount(idx, minlength=len(scenes)).astype(float)
        bs.append(metric(m1, mult)[k] - metric(m2, mult)[k])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    res.append(dict(m1=m1, m2=m2, metric=k, diff=d0, lo=lo, hi=hi))
B = pd.DataFrame(res); B.to_csv(out / 'paired_intervals.csv', index=False)
print(B.round(2).to_string())
