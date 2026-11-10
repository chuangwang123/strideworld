"""Measurable range of the benchmark: how often the pose of the next step is still recovered, as a function of how far the
camera has already moved from the initial view (single-action videos of all registered models, all doses).

For every single-action video and step k >= 2 whose pose at step k-1 was recovered, the outcome is whether the pose at
step k is recovered; it is binned by the motion from the start observed at step k-1: planar displacement in percent of
the initial view's median depth (translations) or absolute yaw (turns). Also writes the share of videos with a recovered
pose at steps 12 and 16 per model and action type.

python3 envelope.py            # -> outputs/measurable_envelope.csv, outputs/measurable_by_model.csv (prints both)
"""
import numpy as np
import pandas as pd

import registry as reg

KEY = ['model_key', 'scene_id', 'case_id', 'seed', 'action']
T_BINS = [0, 10, 25, 50, 75, 100, 150, np.inf]
Y_BINS = [0, 5, 10, 20, 30, 45, 60, np.inf]


def load():
    obs = [pd.read_csv(f) for f in sorted((reg.BASE / 'obs').glob('*.observations.csv'))]
    dep = [pd.read_csv(reg.BASE / 'initial_view_depths.csv')]
    for m in reg.MODELS:
        obs.append(reg.read_obs(m))
        dep.append(reg.read_csv(m, 'depths'))
    O = pd.concat(obs, ignore_index=True)
    D = pd.concat(dep, ignore_index=True)[['model_key', 'scene_id', 'case_id', 'depth_cm']].drop_duplicates()
    O = O.merge(D, on=['model_key', 'scene_id', 'case_id'], how='left')
    assert O.depth_cm.notna().all()
    O['turn'] = O.action.isin(['YL', 'YR'])
    O['disp_pct'] = 100 * np.hypot(O.t_x, O.t_z) / O.depth_cm
    O['yaw_abs'] = O.theta_y_deg.abs()
    O['meas'] = O.measurable.astype(bool)
    O = O.sort_values(KEY + ['dose'])
    g = O.groupby(KEY)
    O['prev_meas'] = g.meas.shift(1)
    O['prev_disp'] = g.disp_pct.shift(1)
    O['prev_yaw'] = g.yaw_abs.shift(1)
    return O


def main():
    O = load()
    H = O[(O.dose > 1) & (O.prev_meas == True)]
    rows = []
    for turn, col, bins, unit in [(False, 'prev_disp', T_BINS, '% depth'), (True, 'prev_yaw', Y_BINS, 'deg')]:
        h = H[H.turn == turn]
        b = pd.cut(h[col], bins, right=False)
        for iv, grp in h.groupby(b, observed=True):
            rows.append(dict(kind='turn' if turn else 'translation', unit=unit, lo=iv.left, hi=iv.right,
                             p_next_recovered=grp.meas.mean(), n_steps=len(grp), n_models=grp.model_key.nunique()))
    E = pd.DataFrame(rows)
    E.to_csv(reg.OUT / 'measurable_envelope.csv', index=False)
    M = (O[O.dose.isin([12, 16])].assign(kind=lambda d: np.where(d.turn, 'turn', 'translation'))
         .groupby(['model_key', 'kind', 'dose']).meas.mean().unstack(['kind', 'dose']))
    M.columns = [f'{k}_step{d}' for k, d in M.columns]
    M.to_csv(reg.OUT / 'measurable_by_model.csv')
    print(E.round(3).to_string(index=False))
    print(M.round(3).to_string())


if __name__ == '__main__':
    main()
