"""Scene-bootstrap 95% intervals for the control columns of Table 2 (the thirteen, then the registered models of
registry_release.json in registry order) and rank intervals.
Resamples the 30 scenes with replacement (2,000 replicates, seed 20261027); every model uses the same resample, so
ranks are compared within a replicate. Response/Hold keep the per-video weights; Valid routes, Route, Pair and
Composition average routes/pairs of the resampled scenes; Repeat/Views average the per-scene group RSDs of each
translation action over the resampled scenes, then the four actions (the aggregation for the stride-variation table of the appendix (Table H.5)).
python3 bootstrap_table2.py --out outputs/bootstrap_table2.csv
"""
import argparse

import numpy as np
import pandas as pd

from compute_metrics import ORDER, load, respond
import registry as reg


def per_scene(A, R, U):
    A = A.assign(resp=respond(A)); A['wr'] = A.weight * A.resp; A['wh'] = A.weight * A.atomic * A.resp
    a = A.groupby(['model_key', 'scene_id']).agg(w=('weight', 'sum'), wr=('wr', 'sum'), wh=('wh', 'sum'))
    R = R.assign(rs=R.rscore.where(R.route_valid, 0), v=R.route_valid.astype(float))
    r = R.groupby(['model_key', 'scene_id']).agg(nr=('rs', 'size'), rs=('rs', 'sum'), v=('v', 'sum'))
    P = R.groupby(['model_key', 'scene_id', 'case_id', 'pair_id']).agg(pv=('pair_valid', 'first'), pt=('pair_trans', 'first'),
                                                                        py=('pair_yaw', 'first')).reset_index()
    P['ps'] = ((P.pt + P.py) / 2).where(P.pv, 0)
    p = P.groupby(['model_key', 'scene_id']).agg(np_=('ps', 'size'), ps=('ps', 'sum'))
    # group RSDs per (model, action, scene) for translations, exactly as rsd_tables
    U = U[U.action.isin(['W', 'S', 'A', 'D'])]
    ncase = U.groupby('scene_id').case_id.nunique()

    def rsd(u):
        u = np.asarray(u, float); den = np.mean(np.abs(u))
        return np.nan if (len(u) < 2 or not den > 0) else 100 * np.std(u, ddof=1) / den
    out = {}
    for kind, fixed, planned in (('view', ['model_key', 'scene_id', 'seed', 'action'], lambda g: ncase[g.scene_id.iloc[0]]),
                                 ('repeat', ['model_key', 'scene_id', 'case_id', 'action'], lambda g: 3)):
        rows = []
        for key, g in U.groupby(fixed):
            ok = len(g) == planned(g) and g.unit_available.all() and g.signed_command_component.notna().all()
            rows.append(dict(zip(fixed, key), rsd=rsd(g.signed_command_component) if ok else np.nan))
        G = pd.DataFrame(rows).dropna()
        out[kind] = G.groupby(['model_key', 'action', 'scene_id']).rsd.mean()
    return a.join(r).join(p), out


def arrays(S, V, models, scenes):
    """Per model: scene arrays for the ratio metrics and (action x scene) RSD matrices."""
    out = {}
    for mk in models:
        s = S.loc[mk].reindex(scenes)
        d = {c: s[c].values.astype(float) for c in ('w', 'wr', 'wh', 'nr', 'rs', 'v', 'np_', 'ps')}
        for kind in ('repeat', 'view'):
            x = V[kind].loc[mk]
            d[kind] = np.vstack([x.loc[a].reindex(scenes).values for a in ('W', 'S', 'A', 'D')])
        out[mk] = d
    return out


def metrics(Arr, models, w):
    res = {}
    for mk in models:
        d = Arr[mk]
        route = (d['rs'] @ w) / (d['nr'] @ w); pair = (d['ps'] @ w) / (d['np_'] @ w)
        r = dict(response=100 * (d['wr'] @ w) / (d['w'] @ w), hold=(d['wh'] @ w) / (d['w'] @ w),
                 valid_routes=100 * (d['v'] @ w) / (d['nr'] @ w), route=route, pair=pair, composition=(route + pair) / 2)
        for kind in ('repeat', 'view'):
            M = d[kind]; ok = ~np.isnan(M)
            r[kind] = np.mean(np.nansum(np.where(ok, M, 0) * w, axis=1) / (ok * w).sum(axis=1))
        res[mk] = r
    return pd.DataFrame(res).T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(reg.OUT / 'bootstrap_table2.csv'))
    ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args()
    A, R = load(reg.BASE, 'all_scores.json', 'initial_view_depths.csv', 'obs/*.observations.csv')
    X = [reg.load_model(m) for m in reg.MODELS]
    U = pd.concat([pd.read_csv(reg.BASE / 'unit_estimates.csv')] + [reg.read_csv(m, 'units') for m in reg.MODELS])
    S, V = per_scene(pd.concat([A] + [x[0] for x in X]), pd.concat([R] + [x[1] for x in X]), U)
    models = ORDER + reg.KEYS; scenes = sorted(A.scene_id.unique())
    Arr = arrays(S, V, models, scenes)
    point = metrics(Arr, models, np.ones(len(scenes)))
    T = pd.read_csv(reg.OUT / 'main_table_values.csv', index_col=0)
    chk = pd.concat([T] + ([pd.read_csv(reg.OUT / 'extra_values.csv', index_col=0)] if reg.MODELS else []))
    for c, cc in (('response', 'response'), ('hold', 'hold'), ('valid_routes', 'valid_routes'), ('route', 'route'), ('pair', 'pair'),
                  ('composition', 'composition'), ('repeat', 'repeat_translation'), ('view', 'view_translation')):
        print('reproduce', c, float((point[c] - chk.loc[models, cc]).abs().max()))
    rng = np.random.default_rng(20261027)
    reps = []
    for b in range(a.B):
        idx = rng.integers(0, len(scenes), len(scenes)); m = np.bincount(idx, minlength=len(scenes)).astype(float)
        reps.append(metrics(Arr, models, m).assign(b=b))
    Bt = pd.concat(reps)
    out = point.copy()
    for c in point.columns:
        q = Bt.groupby(level=0)[c].quantile([.025, .975]).unstack()
        out[c + '_lo'] = q[0.025]; out[c + '_hi'] = q[0.975]
    for c, asc in (('composition', False), ('hold', False)):
        rk = Bt.groupby('b')[c].rank(ascending=asc)
        Bt[c + '_rank'] = rk
        q = Bt.groupby(level=0)[c + '_rank'].quantile([.025, .975]).unstack()
        out[c + '_rank_lo'] = q[0.025]; out[c + '_rank_hi'] = q[0.975]
        out[c + '_rank'] = point[c].rank(ascending=asc)
    out.to_csv(a.out)
    pd.set_option('display.width', 250)
    print(out.round(1).to_string())


if __name__ == '__main__':
    main()
