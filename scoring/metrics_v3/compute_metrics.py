"""StrideWorld metrics: Response, Hold (response-gated), Composition (route + pair), Valid routes,
Repeat / Cross-view variation (unchanged RSD), recomputed from the frozen per-video and per-route exports.

Inputs (the per-video measurements, assembled by scoring/assemble.py), directory given by --data (13 benchmark models) and --sft (SFT labels):
  all_scores.json            per-video single-action records and per-route records
  unit_estimates.csv         per-video K4 single-step estimates (strides)
  initial_view_depths.csv    median valid depth of each initial view after the adapter crop
  obs/<model>.observations.csv  per-video cumulative motion at doses 1..16
SFT directory: all_scores_sft.json, unit_estimates_sft.csv, initial_view_depths_sft.csv, obs/<label>.observations.csv.

Definitions (paper Sec. 4):
  responds   u_last >= 2 deg (turns) or >= 2% of the view's median depth (translations), where u_last is the signed
             motion along the commanded axis at step 16 or at the last step with a recovered pose.
  Response   scene-balanced share of responding single-action videos (same weights as Atomic).
  Hold       Atomic (mean of step-12 and step-16 scores, unmeasurable = 0) with non-responding videos set to 0.
  Route      mean over all 1,120 routes of (route_trans + route_yaw) / 2, invalid routes = 0.
  Pair       mean over all 560 pairs of (pair_trans + pair_yaw) / 2, invalid pairs = 0.
  Composition (Route + Pair) / 2;  Valid routes = share of routes passing the validity checks.
  Repeat / Cross-view variation: RSD of the signed axial step, unchanged definition (Table H.5 of the appendix), reproduced exactly.
Outputs in --out: main_table_values.csv, per_video_response.csv, response_by_action.csv, results_profile.csv,
sft_values.csv, gate_sensitivity.csv.
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ['matrix-game-2', 'matrix-game-3', 'matrix-game-3.5', 'sana-wm', 'dreamx-world', 'hunyuan-world-1.5', 'hy-world-2',
         'lingbot-world-v2', 'alaya-world', 'h3-world', 'lyra2', 'evoke-turbo', 'worldcrafter-fast']
KEY = ['model_key', 'scene_id', 'case_id', 'seed', 'action']


def load(data, scores_name, depth_name, obs_glob):
    d = json.load(open(Path(data) / scores_name))
    S = pd.DataFrame([r for r in d if 'pair_id' not in r]).rename(columns={'model': 'model_key', 'scene': 'scene_id', 'case': 'case_id'})
    R = pd.DataFrame([r for r in d if 'pair_id' in r]).rename(columns={'model': 'model_key', 'scene': 'scene_id', 'case': 'case_id'})
    o = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(Path(data) / obs_glob)))]).sort_values(KEY + ['dose'])
    m = o[o.measurable.astype(bool)]
    last = m.groupby(KEY).agg(last_dose=('dose', 'max'), u_last=('u_signed', 'last')).reset_index()
    dep = pd.read_csv(Path(data) / depth_name)[['model_key', 'scene_id', 'case_id', 'depth_cm']]
    A = S.merge(last, on=KEY, how='left').merge(dep, on=['model_key', 'scene_id', 'case_id'], how='left')
    assert len(A) == len(S) and A.depth_cm.notna().all()
    A['turn'] = A.action.isin(['YL', 'YR'])
    A['vis'] = np.where(A.turn, A.u_last, 100 * A.u_last / A.depth_cm)
    R['rscore'] = (R.route_trans + R.route_yaw) / 2
    return A, R


def respond(A, tau_t=2.0, tau_r=2.0):
    return (A.vis >= np.where(A.turn, tau_r, tau_t)) & A.u_last.notna()


def model_values(A, R, resp):
    A = A.assign(resp=resp)
    T = {}
    for mk, g in A.groupby('model_key'):
        w = g.weight / g.weight.sum()
        r = R[R.model_key == mk]
        p = r.groupby(['case_id', 'pair_id']).agg(pv=('pair_valid', 'first'), pt=('pair_trans', 'first'), py=('pair_yaw', 'first'))
        p['ps'] = (p.pt + p.py) / 2
        T[mk] = dict(response=100 * (w * g.resp).sum(), atomic=(w * g.atomic).sum(), hold=(w * g.atomic * g.resp).sum(),
                     valid_routes=100 * r.route_valid.mean(),
                     valid_routes_no_motion=100 * (r.route_quality_pass_without_motion & r.calibration_pass).mean(),
                     route=r.rscore.where(r.route_valid, 0).mean(), pair=p.ps.where(p.pv, 0).mean(),
                     valid_pairs=100 * p.pv.mean(),
                     route_trans_valid=r.route_trans[r.route_valid].mean(), pair_trans_valid=p.pt[p.pv].mean(),
                     route_err_median=r.route_translation_r[r.route_valid].median(),
                     route_err_q25=r.route_translation_r[r.route_valid].quantile(.25),
                     route_err_q75=r.route_translation_r[r.route_valid].quantile(.75),
                     pair_err_median=(100 / p.pt[p.pv] - 1).median())
    T = pd.DataFrame(T).T
    T['composition'] = (T.route + T.pair) / 2
    return T


def rsd_tables(U, resp_df=None):
    """Repeat and cross-view RSD exactly as Table H.5 of the appendix (complete groups; members -> scene -> actions)."""
    U = U.copy(); U['turn'] = U.action.isin(['YL', 'YR'])
    ncase = U.groupby('scene_id').case_id.nunique()

    def rsd(u):
        u = np.asarray(u, float); den = np.mean(np.abs(u))
        return np.nan if (len(u) < 2 or not den > 0) else 100 * np.std(u, ddof=1) / den
    out = {}
    for kind, fixed, planned in [('view', ['model_key', 'scene_id', 'seed', 'action'], lambda g: ncase[g.scene_id.iloc[0]]),
                                 ('repeat', ['model_key', 'scene_id', 'case_id', 'action'], lambda g: 3)]:
        rows = []
        for key, g in U.groupby(fixed):
            ok = len(g) == planned(g) and g.unit_available.all() and g.signed_command_component.notna().all()
            rows.append(dict(zip(fixed, key), rsd=rsd(g.signed_command_component) if ok else np.nan))
        G = pd.DataFrame(rows); G['turn'] = G.action.isin(['YL', 'YR'])
        s = G.dropna().groupby(['model_key', 'action', 'scene_id']).rsd.mean().groupby(['model_key', 'action']).mean().reset_index()
        s['turn'] = s.action.isin(['YL', 'YR'])
        t = s.groupby(['model_key', 'turn']).rsd.mean().unstack()
        out[f'{kind}_translation'] = t[False]; out[f'{kind}_turn'] = t[True]
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--sft', default=''); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A, R = load(a.data, 'all_scores.json', 'initial_view_depths.csv', 'obs/*.observations.csv')
    resp = respond(A)
    T = model_values(A, R, resp)
    frozen_atomic = [58.4, 63.2, 74.1, 76.5, 46.9, 77.8, 87.1, 83.0, 28.8, 26.5, 88.4, 74.9, 78.1]  # Atomic of the scoring reports (consistency check)
    # Values printed by the scoring report of a run can be passed for the same check: SW_FROZEN_ATOMIC='evoke-turbo=74.3'.
    import os
    for item in filter(None, os.environ.get('SW_FROZEN_ATOMIC', '').split(',')):
        k, v = item.split('=')
        frozen_atomic[ORDER.index(k.strip())] = float(v)
    assert np.allclose(T.loc[ORDER, 'atomic'].round(1).values, frozen_atomic), \
        ('Atomic does not reproduce', dict(zip(ORDER, T.loc[ORDER, 'atomic'].round(1).values)))
    U = pd.read_csv(Path(a.data) / 'unit_estimates.csv')
    T = T.join(rsd_tables(U))
    T.loc[ORDER].to_csv(out / 'main_table_values.csv')
    A.assign(resp=resp)[KEY + ['record_id', 'weight', 'atomic', 'measurable12', 'measurable16', 'u_last', 'last_dose',
                                'depth_cm', 'vis', 'resp']].to_csv(out / 'per_video_response.csv.gz', index=False, compression='gzip')
    rb = A.assign(resp=resp).groupby(['model_key', 'action']).apply(
        lambda g: 100 * (g.weight * g.resp).sum() / g.weight.sum()).unstack().loc[ORDER]
    rb.to_csv(out / 'response_by_action.csv')
    sens = []
    for f in (0.25, 0.5, 1.0, 1.5, 2.0):
        Tf = model_values(A, R, respond(A, 2.0 * f, 2.0 * f))
        for mk in ORDER:
            sens.append(dict(model_key=mk, factor=f, response=Tf.loc[mk, 'response'], hold=Tf.loc[mk, 'hold']))
    pd.DataFrame(sens).to_csv(out / 'gate_sensitivity.csv', index=False)
    if a.sft:
        As, Rs = load(a.sft, 'all_scores_sft.json', 'initial_view_depths_sft.csv', 'obs/*.observations.csv')
        rs = respond(As)
        Ts = model_values(As, Rs, rs)
        eth = As.scene_id.str.startswith('eth3d'); ethr = Rs.scene_id.str.startswith('eth3d')
        Te = model_values(As[eth], Rs[ethr], rs[eth]).add_suffix('_eth3d')
        Ts.join(Te).to_csv(out / 'sft_values.csv')
    print(T.loc[ORDER, ['response', 'hold', 'valid_routes', 'route', 'pair', 'composition', 'repeat_translation',
                        'view_translation']].round(1).to_string())


if __name__ == '__main__':
    main()
