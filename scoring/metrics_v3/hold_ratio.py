"""Fig. 3b input: per model, observed motion along the commanded axis at step 16 divided by 16 x the model's own
first-four-step step (signed command component of the K4 unit; radians for turns, converted to degrees). Uses the
videos measurable at step 16 whose first four steps move the commanded way (positive component), so that the ratio is
defined; reports the 25th, 50th and 75th percentiles over videos (unweighted). Reproduces the 13-model values of
figures_src/data/results_profile.csv exactly.

python3 hold_ratio.py --obs '<dir>/obs/*.observations.csv' --units <unit_estimates.csv> [--obs ... --units ...] [--extra] --out <csv>
--extra appends every model of the registry (registry_release.json), in registry order.
"""
import argparse, glob

import numpy as np
import pandas as pd

import registry as reg

KEY = ['model_key', 'scene_id', 'case_id', 'seed', 'action']


def ratios(obs, units, dose=16):
    o = obs[(obs.dose == dose) & obs.measurable.astype(bool)][KEY + ['u_signed']]
    u = units[units.unit_available.astype(bool)][KEY + ['signed_command_component']]
    m = o.merge(u, on=KEY, how='inner')
    m['step'] = np.where(m.action.isin(['YL', 'YR']), np.degrees(m.signed_command_component), m.signed_command_component)
    m = m[m.step > 0]
    m['ratio'] = m.u_signed / (dose * m.step)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obs', action='append', default=[])
    ap.add_argument('--units', action='append', default=[])
    ap.add_argument('--extra', action='store_true', help='add every model of the registry')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    obs = [pd.read_csv(f) for g in a.obs for f in sorted(glob.glob(g))]
    units = [pd.read_csv(f) for f in a.units]
    if a.extra:
        obs += [reg.read_obs(m) for m in reg.MODELS]; units += [reg.read_csv(m, 'units') for m in reg.MODELS]
    obs, units = pd.concat(obs), pd.concat(units)
    m = ratios(obs, units)
    q = m.groupby('model_key').ratio.quantile([.25, .5, .75]).unstack()
    q.columns = ['hold_q25', 'hold_med', 'hold_q75']
    q['n'] = m.groupby('model_key').size()
    q.to_csv(a.out)
    print(q.round(3))


if __name__ == '__main__':
    main()
