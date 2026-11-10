"""F3 diagnostics: why routes miss the end predicted by chained strides.

For every completed route (all scored records, 15 models) and every swap pair (G2-G4, both routes completed):
  err        route end error in strides, ||t(H_hat^-1 H)|| / b_t (equals route_translation_r)
  scale      observed / predicted displacement from the start (1 = the route keeps the strides' length)
  rep_floor  error a perfectly composing model would show if the route were a fresh generation: the route predicted
             from seed-1 / seed-2 strides against the seed-0 prediction (mean of the two), in strides
  swap_gap   distance between the two ends of a swap pair (same actions, other order), in strides;
             swap_pred = the same from chained strides (~0: the strides nearly commute)
  swap_rep   swap gap expected from repeat variation alone (route A from seed s1, route B from seed s2 strides)
SE(2) chaining: U_a = exp(rho_x, rho_z, -theta) from unit_estimates (reproduces predicted_endpoint_se2 to 1e-13).
Writes out/f3_diag/{routes.csv, swaps.csv, by_model.csv}.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import os as _os


def _dp(name, rest=''):
    """Data path under SW_DATA, with the folder names of SW_BASE_DIR, SW_SOLAR_DIR and SW_DEFORM_DIR."""
    env = {'base': 'SW_BASE_DIR', 'extra': 'SW_SOLAR_DIR', 'deform': 'SW_DEFORM_DIR'}[name]
    return _os.path.join(_os.environ['SW_DATA'], _os.environ.get(env, name)) + rest

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / 'out' / 'f3_diag'
X1 = Path(_dp('base'))
X2 = Path(_dp('extra'))


def exp_se2(rx, rz, th):
    V = np.eye(2) if abs(th) < 1e-9 else np.array([[np.sin(th), -(1 - np.cos(th))], [1 - np.cos(th), np.sin(th)]]) / th
    t = V @ np.array([rx, rz])
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, t[0]], [s, c, t[1]], [0, 0, 1]])


def main(units=None, scores=None, out=None):
    out = out or OUT
    out.mkdir(parents=True, exist_ok=True)
    units = units or [X1 / 'unit_estimates.csv', X2 / 'unit_estimates_solar.csv']
    scores = scores or [X1 / 'all_scores.json', X2 / 'all_scores_solar.json']
    U = pd.concat([pd.read_csv(u) for u in units])
    U = U[U.unit_available.astype(bool)]
    idx = {(r.model_key, r.scene_id, r.case_id, int(r.seed), r.action): r for r in U.itertuples()}

    def chain(m, sc, cs, seed, steps):
        H = np.eye(3)
        for a in steps:
            u = idx.get((m, sc, cs, seed, a))
            if u is None:
                return None
            H = H @ exp_se2(u.rho_x_cm, u.rho_z_cm, -u.theta_rad)
        return H

    def tn(M):
        return float(np.hypot(*M[:2, 2]))

    recs = []
    for p in scores:
        recs += [x for x in json.load(open(p)) if 'pair_id' in x]
    rows, pairs = [], {}
    for x in recs:
        pairs.setdefault((x['model'], x['scene'], x['case'], x['pair_id']), {})[x['member']] = x
        if not x['route_valid']:
            continue
        m, sc, cs, bt = x['model'], x['scene'], x['case'], x['b_t']
        H, P0 = np.array(x['observed_endpoint_se2']), np.array(x['predicted_endpoint_se2'])
        rep = [tn(np.linalg.inv(P0) @ Ps) / bt for s in (1, 2) if (Ps := chain(m, sc, cs, s, x['steps'])) is not None]
        rows.append(dict(model=m, scene=sc, case=cs, pair=x['pair_id'], member=x['member'], family=x['family'],
                         err=tn(np.linalg.inv(P0) @ H) / bt, scale=tn(H) / max(tn(P0), 1e-9),
                         pred_len=tn(P0) / bt, rep_floor=np.mean(rep) if rep else np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(out / 'routes.csv', index=False)
    srows = []
    for (m, sc, cs, pid), d in pairs.items():
        if 'A' not in d or 'B' not in d or d['A']['family'] == 'G1':
            continue
        a, b = d['A'], d['B']
        if not (a['route_valid'] and b['route_valid']):
            continue
        bt = a['b_t']
        Ha, Hb = np.array(a['observed_endpoint_se2']), np.array(b['observed_endpoint_se2'])
        Pa, Pb = np.array(a['predicted_endpoint_se2']), np.array(b['predicted_endpoint_se2'])
        rep = []
        for s1, s2 in [(0, 1), (0, 2), (1, 2)]:
            A1, B2 = chain(m, sc, cs, s1, a['steps']), chain(m, sc, cs, s2, b['steps'])
            if A1 is not None and B2 is not None:
                rep.append(tn(np.linalg.inv(A1) @ B2) / bt)
        srows.append(dict(model=m, scene=sc, case=cs, pair=pid, family=a['family'],
                          swap_gap=tn(np.linalg.inv(Ha) @ Hb) / bt, swap_gap_cm=tn(np.linalg.inv(Ha) @ Hb),
                          swap_pred=tn(np.linalg.inv(Pa) @ Pb) / bt, swap_rep=np.median(rep) if rep else np.nan))
    S = pd.DataFrame(srows)
    S.to_csv(out / 'swaps.csv', index=False)
    g = R.groupby('model').agg(n_routes=('err', 'size'), end_err=('err', 'median'), rep_floor=('rep_floor', 'median'),
                               scale=('scale', 'median'))
    gs = S.groupby('model').agg(n_swaps=('swap_gap', 'size'), swap_gap=('swap_gap', 'median'),
                                swap_pred=('swap_pred', 'median'), swap_rep=('swap_rep', 'median'),
                                swap_gt1=('swap_gap', lambda v: 100 * np.mean(v > 1)))
    G = g.join(gs)
    G['err_over_floor'] = G.end_err / G.rep_floor
    G['swap_over_rep'] = G.swap_gap / G.swap_rep
    G.to_csv(out / 'by_model.csv')
    print(G.round(2).sort_values('end_err').to_string())
    print('swap_pred pooled median %.3f, p95 %.3f' % (S.swap_pred.median(), S.swap_pred.quantile(0.95)))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'sft':      # fine-tuning labels (n0, j256, b2, hy15ref)
        XS = Path(_os.environ['SW_DATA']) / _os.environ.get('SW_SFT_DIR', 'sft')
        main([XS / 'unit_estimates_sft.csv'], [XS / 'all_scores_sft.json'], HERE / 'out' / 'f3_diag_sft')
    else:
        main()
