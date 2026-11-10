"""Per-checkpoint Hold scores (steps 12 and 16) under the video's own step (main) and under the two shared references of
Appendix H (fixed: component-wise median of the model's single-step estimates for the action; axis: only the commanded
component of that median), recomputed from the observation tables and unit estimates.

Translations: error = |t_obs - t(U^k)| / (k |t(U)|); turns: |theta_obs - k theta(U)| / (k |theta(U)|); score 100/(1+r);
a checkpoint is scorable when its pose is measurable and the video has a unit. SE(2) uses R = [[c, s], [-s, c]] in the
x-z plane, with V adjusted accordingly (reproduces the frozen predicted_endpoint_se2).

--check verifies the 13-model per_checkpoint_scores.csv (main_r, fixed_r, axis_r) before writing the extra models.
python3 checkpoint_scores.py --check <dir13>      every model of the registry (registry_release.json) -> its per_checkpoint_out
python3 checkpoint_scores.py --check <dir13> --dir <extra dir> --obs 'obs/*.observations.csv' --units <units.csv> --out <csv>
"""
import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd

import registry as reg

KEY = ['model_key', 'scene_id', 'case_id', 'seed', 'action']


def exp_t(rx, rz, th):
    th = np.asarray(th, float); small = np.abs(th) < 1e-9
    s, c = np.sin(th), np.cos(th)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(small, 1.0, s / th); b = np.where(small, 0.0, (1 - c) / th)
    return a * rx + b * rz, -b * rx + a * rz


def errors(m, z):
    k = m.dose.values; turn = m.action.isin(['YL', 'YR']).values
    px, pz = exp_t(k * z.rho_x_cm, k * z.rho_z_cm, k * z.theta_rad)
    ux, uz = exp_t(z.rho_x_cm, z.rho_z_cm, z.theta_rad)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_tr = np.hypot(m.t_x - px, m.t_z - pz) / (k * np.hypot(ux, uz))
        r_tu = np.abs(np.radians(m.theta_y_deg) - k * z.theta_rad) / (k * np.abs(z.theta_rad))
    return np.where(turn, r_tu, r_tr)


def checkpoint_table(obs, units, doses=(12, 16)):
    u = units.copy()
    med = u[u.unit_available.astype(bool)].groupby(['model_key', 'action'])[['rho_x_cm', 'rho_z_cm', 'theta_rad']].median()
    base = u[KEY + ['record_id', 'weight', 'unit_available', 'rho_x_cm', 'rho_z_cm', 'theta_rad']]
    rows = []
    for d in doses:
        b = base.assign(dose=d)
        rows.append(b)
    m = pd.concat(rows).merge(obs[KEY + ['dose', 't_x', 't_z', 'theta_y_deg', 'measurable']], on=KEY + ['dose'], how='left')
    m['measurable'] = m.measurable.fillna(False).astype(bool)
    m['scorable'] = m.measurable & m.unit_available.astype(bool)
    fx = m[KEY[:1] + ['action']].join(med, on=['model_key', 'action'])
    ax = fx.copy()
    ax.loc[m.action.isin(['W', 'S']).values, ['rho_x_cm', 'theta_rad']] = 0
    ax.loc[m.action.isin(['A', 'D']).values, ['rho_z_cm', 'theta_rad']] = 0
    ax.loc[m.action.isin(['YL', 'YR']).values, ['rho_x_cm', 'rho_z_cm']] = 0
    for nm, z in (('main', m), ('fixed', fx), ('axis', ax)):
        r = errors(m, z)
        m[nm + '_r'] = np.where(m.scorable, r, np.nan)
        m[nm] = np.where(m.scorable, 100 / (1 + m[nm + '_r']), 0.0)
    return m[KEY + ['record_id', 'weight', 'dose', 'scorable', 'main', 'fixed', 'axis', 'main_r', 'fixed_r', 'axis_r']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', help='13-model data dir with per_checkpoint_scores.csv, obs/ and unit_estimates.csv')
    ap.add_argument('--dir', help='one export dir (with --units, --out) instead of the registry')
    ap.add_argument('--obs', default='obs/*.observations.csv'); ap.add_argument('--units'); ap.add_argument('--out')
    a = ap.parse_args()
    if a.check:
        P = Path(a.check)
        ref = pd.read_csv(P / 'per_checkpoint_scores.csv')
        obs = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(P / 'obs/*.observations.csv')))])
        mine = checkpoint_table(obs, pd.read_csv(P / 'unit_estimates.csv'))
        j = ref.merge(mine, on=KEY + ['dose'], suffixes=('', '_new'))
        assert len(j) == len(ref)
        for c in ('main_r', 'fixed_r', 'axis_r'):
            ok = j[c].notna() & j[c + '_new'].notna()
            print(c, 'max abs diff', float((j[c][ok] - j[c + '_new'][ok]).abs().max()), 'nan mismatches',
                  int((j[c].isna() != j[c + '_new'].isna()).sum()), 'of', len(j))
    if a.dir:
        P = Path(a.dir)
        obs = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(P / a.obs)))])
        T = checkpoint_table(obs, pd.read_csv(P / a.units))
        T.to_csv(a.out, index=False)
        print('wrote', a.out, len(T))
        return
    for m in reg.MODELS:
        T = checkpoint_table(reg.read_obs(m), reg.read_csv(m, 'units'))
        T.to_csv(m.out('per_checkpoint_out'), index=False)
        print('wrote', m.out('per_checkpoint_out'), len(T))


if __name__ == '__main__':
    main()
