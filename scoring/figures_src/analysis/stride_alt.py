"""Stride definition check (App.: alternative stride definitions).

The frozen stride is U_a = Exp(1/4 sum_{k=1..4} Log(H_{a,k}) / k), which weights the first four steps by
0.52 / 0.27 / 0.15 / 0.06. Alternatives, from the same observation tables (H_k = SE2(t_x, t_z, -yaw_k); this reproduces
the frozen unit_estimates to 1e-9):
  K4    the frozen definition (recomputed; must match unit_estimates)
  M4    Exp(Log(H_4) / 4)                  unweighted mean step over steps 1-4
  M25   Exp(Log(H_1^-1 H_5) / 4)            unweighted mean step over steps 2-5 (ignores the first step)
F2: Table 2's Ratio = u_signed(16) / (16 x signed command component of the stride) over the videos measurable at step
16 whose stride moves the commanded way (hold_ratio.py); median per model and the share outside [0.8, 1.25].
F3 (M4 only): routes re-predicted by chaining M4 strides; b_t, b_r recomputed from M4; end error of completed routes
(the frozen completion set) and Near = completed and max(r_t, r_yaw) <= 1 over all 1,120 routes.
Writes out/stride_alt/{units.csv, f2_ratio.csv, f3_m4.csv}.
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
OUT = HERE / 'out' / 'stride_alt'
X1 = Path(_dp('base'))
X2 = Path(_dp('extra'))
KEY = ['model_key', 'scene_id', 'case_id', 'seed', 'action']
SIGN = {'W': (1, 1), 'S': (1, -1), 'A': (0, -1), 'D': (0, 1)}   # (component index, sign) -> filled from data below


def V(th):
    return np.eye(2) if abs(th) < 1e-9 else np.array([[np.sin(th), -(1 - np.cos(th))],
                                                      [1 - np.cos(th), np.sin(th)]]) / th


def log_se2(H):
    ph = float(np.arctan2(H[1, 0], H[0, 0]))
    return np.r_[np.linalg.solve(V(ph), H[:2, 2]), ph]


def exp_se2(x):
    t = V(x[2]) @ x[:2]
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([[c, -s, t[0]], [s, c, t[1]], [0, 0, 1.0]])


def se2(x, z, ph):
    c, s = np.cos(ph), np.sin(ph)
    return np.array([[c, -s, x], [s, c, z], [0, 0, 1.0]])


def units_from_obs():
    obs = pd.concat([pd.read_csv(f) for f in sorted((X1 / 'obs').glob('*.observations.csv'))] +
                    [pd.read_csv(f) for f in sorted((X2 / 'obs').glob('*.observations.csv'))])
    rows = []
    for k, g in obs[obs.dose <= 16].groupby(KEY):
        g = g.set_index('dose')
        H = {d: se2(g.at[d, 't_x'], g.at[d, 't_z'], -np.radians(g.at[d, 'theta_y_deg'])) for d in g.index}
        meas = {d: bool(g.at[d, 'measurable']) for d in g.index}
        r = dict(zip(KEY, k))
        if all(meas.get(d, False) for d in (1, 2, 3, 4)):
            r['K4'] = np.mean([log_se2(H[d]) / d for d in (1, 2, 3, 4)], axis=0)
            r['M4'] = log_se2(H[4]) / 4
        if all(meas.get(d, False) for d in (1, 5)):
            r['M25'] = log_se2(np.linalg.inv(H[1]) @ H[5]) / 4
        r['u16'] = g.at[16, 'u_signed'] if 16 in g.index and meas.get(16, False) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def cmd(action, x):
    """Signed command component of a twist (as unit_estimates.signed_command_component; turns in degrees)."""
    if x is None or not isinstance(x, np.ndarray):
        return np.nan
    if action == 'W':
        return x[1]
    if action == 'S':
        return -x[1]
    if action == 'D':
        return x[0]
    if action == 'A':
        return -x[0]
    if action == 'YL':
        return np.degrees(x[2])
    if action == 'YR':
        return -np.degrees(x[2])
    return np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A = units_from_obs()
    Uf = pd.concat([pd.read_csv(X1 / 'unit_estimates.csv'), pd.read_csv(X2 / 'unit_estimates_solar.csv')])
    Uf['seed'] = Uf.seed.astype(int)
    A = A.merge(Uf[KEY + ['unit_available', 'rho_x_cm', 'rho_z_cm', 'theta_rad', 'signed_command_component']],
                on=KEY, how='left')
    # check K4 against the frozen units and the sign convention of the command component
    ok = A.K4.notna() & A.unit_available.fillna(False).astype(bool)
    d = np.array([np.abs(np.r_[r.K4[0] - r.rho_x_cm, r.K4[1] - r.rho_z_cm, r.K4[2] + r.theta_rad]).max()
                  for r in A[ok].itertuples()])
    sc = A[ok].apply(lambda r: cmd(r.action, r.K4), axis=1)
    fc = A[ok].signed_command_component.where(~A[ok].action.isin(['YL', 'YR']),
                                              np.degrees(A[ok].signed_command_component))
    print('K4 vs frozen twist: max abs diff %.2e; command component max abs diff %.2e' % (d.max(),
                                                                                       np.nanmax(np.abs(sc - fc))))
    for c in ('K4', 'M4', 'M25'):
        A[f'cmd_{c}'] = [cmd(a, x) if isinstance(x, np.ndarray) else np.nan for a, x in zip(A.action, A[c])]
    keep = A.drop(columns=['K4', 'M4', 'M25']).copy()
    for c in ('K4', 'M4', 'M25'):
        for i, nm in enumerate(('x', 'z', 'phi')):
            keep[f'{c}_{nm}'] = [x[i] if isinstance(x, np.ndarray) else np.nan for x in A[c]]
    keep.to_csv(OUT / 'units.csv', index=False)
    # F2 ratio (videos measurable at 16 with a positive frozen command component, as hold_ratio.py)
    F = A[A.u16.notna() & A.unit_available.fillna(False).astype(bool)].copy()
    F = F[F.cmd_K4 > 0]
    rows = []
    for m, g in F.groupby('model_key'):
        row = dict(model=m, n=len(g))
        for c in ('K4', 'M4', 'M25'):
            gg = g[g[f'cmd_{c}'] > 0]
            r = gg.u16 / (16 * gg[f'cmd_{c}'])
            row[f'ratio_{c}'] = r.median()
            row[f'n_{c}'] = len(gg)
        rows.append(row)
    R = pd.DataFrame(rows).set_index('model')
    H = pd.read_csv(HERE.parent / 'metrics_v3' / 'outputs' / 'hold_ratio.csv').set_index('model_key')
    R['frozen_hold_med'] = H.hold_med
    for c in ('K4', 'M4', 'M25'):
        R[f'out_{c}'] = ~R[f'ratio_{c}'].between(0.8, 1.25)
    R.to_csv(OUT / 'f2_ratio.csv')
    # F3 under M4
    S0 = keep[keep.seed == 0]
    tw = {(r.model_key, r.scene_id, r.case_id, r.action): np.r_[r.M4_x, r.M4_z, r.M4_phi] for r in S0.itertuples()
          if np.isfinite(r.M4_x)}
    tw_k = {(r.model_key, r.scene_id, r.case_id, r.action): np.r_[r.K4_x, r.K4_z, r.K4_phi] for r in S0.itertuples()
            if np.isfinite(r.K4_x)}
    rows = []
    for p in [X1 / 'all_scores.json', X2 / 'all_scores_solar.json']:
        for x in json.load(open(p)):
            if 'pair_id' not in x:
                continue
            m, sc_, cs = x['model'], x['scene'], x['case']
            row = dict(model=m, valid=x['route_valid'], near_frozen=bool(x['route_valid']) and max(
                x.get('route_translation_r') or 0, x.get('route_yaw_r') or 0) <= 1)
            if x['route_valid']:
                O = np.array(x['observed_endpoint_se2'])
                for nm, T in (('m4', tw), ('k4', tw_k)):
                    us = [T.get((m, sc_, cs, a)) for a in ('W', 'S', 'A', 'D', 'YL', 'YR')]
                    if any(u is None for u in us) or any(T.get((m, sc_, cs, a)) is None for a in x['steps']):
                        continue
                    bt = np.median([np.hypot(*exp_se2(u)[:2, 2]) for u in us[:4]])
                    br = np.median([abs(np.degrees(u[2])) for u in us[4:]])
                    P = np.eye(3)
                    for a in x['steps']:
                        P = P @ exp_se2(T[(m, sc_, cs, a)])
                    E = np.linalg.inv(P) @ O
                    rt = np.hypot(*E[:2, 2]) / bt
                    ry = abs(np.degrees(np.arctan2(E[1, 0], E[0, 0]))) / br
                    row[f'err_{nm}'] = rt
                    row[f'near_{nm}'] = max(rt, ry) <= 1
                    row[f'bt_{nm}'] = bt
                row['err_frozen'] = x['route_translation_r']
            rows.append(row)
    T = pd.DataFrame(rows)
    G = T.groupby('model').agg(near_frozen=('near_frozen', lambda v: 100 * v.mean()),
                               near_k4=('near_k4', lambda v: 100 * v.fillna(False).astype(bool).mean()),
                               near_m4=('near_m4', lambda v: 100 * v.fillna(False).astype(bool).mean()),
                               err_frozen=('err_frozen', 'median'), err_k4=('err_k4', 'median'),
                               err_m4=('err_m4', 'median'))
    G.to_csv(OUT / 'f3_m4.csv')
    with pd.option_context('display.width', 250):
        print(R.round(3).to_string())
        print(G.round(2).to_string())


if __name__ == '__main__':
    main()
