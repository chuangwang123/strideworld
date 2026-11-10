"""F3 route outcomes per model (Table 2: Near, Stall): the route classification used for the paper's main results.

A route is near if it is completed and max(translation, yaw) error <= 1 stride, far if completed otherwise,
unmeasurable if it failed for pose or calibration, and stalled otherwise. Writes out/findings_v7/F3_route_outcomes.csv.
"""
import json
import os as _os
from pathlib import Path

import pandas as pd


def _dp(name, rest=''):
    """Data path (SW_DATA; SW_BASE_DIR, SW_SOLAR_DIR)."""
    env = {'base': 'SW_BASE_DIR', 'extra': 'SW_SOLAR_DIR', 'deform': 'SW_DEFORM_DIR'}[name]
    return _os.path.join(_os.environ['SW_DATA'], _os.environ.get(env, name)) + rest


HERE = Path(__file__).resolve().parent
OUT = HERE / 'out' / 'findings_v7'
SCORES = [Path(_dp('base', '/all_scores.json')),
          Path(_dp('extra', '/all_scores_solar.json'))]


def route_outcomes():
    rows = []
    for p in SCORES:
        for x in json.load(open(p)):
            if 'pair_id' not in x:
                continue
            cond = {c.split(':')[0] for c in x.get('route_failure_conditions', [])}
            if x['route_valid']:
                r = max(x.get('route_translation_r') or 0, x.get('route_yaw_r') or 0)
                cat = 'near' if r <= 1 else 'far'
            elif cond & {'Pose', 'Calibration'}:
                cat = 'unmeasurable'
            else:
                cat = 'stalled'
            rows.append(dict(model=x['model'], cat=cat))
    D = pd.DataFrame(rows)
    T = D.groupby(['model', 'cat']).size().unstack(fill_value=0)
    n = T.sum(axis=1)
    T = T.div(n, axis=0) * 100
    T['n_routes'] = n
    return T


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    T = route_outcomes()
    T.to_csv(OUT / 'F3_route_outcomes.csv')
    print(T.round(1).to_string())
