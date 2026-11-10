"""Response profiles for Figs. E.1 and E.2: for each model and action, |u_k| of every single-action
video divided by its own first-step |u_1|, median over the videos measurable at all 16 steps, and a free-intercept
least-squares fit log m = log a + gamma log k over k = 1..16 (R^2 in log space). Also the share of the 420 planned
videos with a valid pose at each step (orange curves). Same procedure as the printed panels of the three models added
in 10b (reproduced exactly with --check), applied to the registered models (--extra: every model of registry_release.json).

python3 response_profiles.py --obs '<dir>/obs/<model>.observations.csv' [--obs ...] [--extra] --out <csv prefix>
"""
import argparse, glob

import numpy as np
import pandas as pd

import registry as reg

KEY = ['model_key', 'scene_id', 'case_id', 'seed', 'action']


def profiles(o):
    o = o.sort_values(KEY + ['dose'])
    rows, fits = [], []
    for (mk, act), g in o.groupby(['model_key', 'action']):
        valid = g.pivot_table(index=['scene_id', 'case_id', 'seed'], columns='dose', values='measurable', aggfunc='first')
        val = g.pivot_table(index=['scene_id', 'case_id', 'seed'], columns='dose', values='u_signed', aggfunc='first')
        valid = valid.reindex(columns=range(1, 17)).fillna(False).astype(bool)
        val = val.reindex(index=valid.index, columns=range(1, 17))
        full = valid.all(axis=1)
        v = val[full].abs()
        v = v[v[1] > 0]
        norm = v.div(v[1], axis=0)
        prof = norm.median(axis=0).values
        k = np.arange(1, 17)
        X = np.c_[np.ones(16), np.log(k)]; y = np.log(prof)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        fits.append(dict(model_key=mk, action=act, n=len(norm), gamma=beta[1], a=np.exp(beta[0]), r2=r2))
        frac = g[g.measurable.astype(bool)].groupby('dose').size().reindex(range(1, 17), fill_value=0) / 420
        for kk in range(1, 17):
            rows.append(dict(model_key=mk, action=act, dose=kk, response=prof[kk - 1],
                             fit=np.exp(beta[0]) * kk ** beta[1], valid_pose=frac.loc[kk]))
    return pd.DataFrame(rows), pd.DataFrame(fits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obs', action='append', default=[]); ap.add_argument('--out', required=True)
    ap.add_argument('--extra', action='store_true', help='add every model of the registry')
    a = ap.parse_args()
    o = [pd.read_csv(f) for g in a.obs for f in sorted(glob.glob(g))]
    o = pd.concat(o + ([reg.read_obs(m) for m in reg.MODELS] if a.extra else []))
    P, F = profiles(o)
    P.to_csv(a.out + '_profiles.csv', index=False); F.to_csv(a.out + '_fits.csv', index=False)
    print(F.assign(gamma=F.gamma.round(2), r2=F.r2.round(3)).to_string())


if __name__ == '__main__':
    main()
