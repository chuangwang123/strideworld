"""Write figures_src/data/results_profile.csv (Fig. 3) from the metrics_v3 outputs.

Columns: Response per action (W..YR), Hold ratio at step 16 (q25/median/q75, hold_ratio.py), valid routes with and
without the motion check, and median (IQR) endpoint error of valid routes and median pair error, in steps.
Rows: the 13 benchmark models in main-table order, then the registered models (registry_release.json) in registry order,
from outputs/extra_values.csv.
python3 fig_results_data.py [--out <csv>; default ../figures_src/data/results_profile.csv]
"""
import argparse
from pathlib import Path

import pandas as pd

from compute_metrics import ORDER
import registry as reg

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(HERE.parent / 'figures_src' / 'data' / 'results_profile.csv'))
args = ap.parse_args()
O = reg.OUT
T = pd.read_csv(O / 'main_table_values.csv', index_col=0)
RB = pd.read_csv(O / 'response_by_action.csv', index_col=0)
HR = pd.read_csv(O / 'hold_ratio.csv', index_col=0)
rows = {}
for m in ORDER:
    rows[m] = {**RB.loc[m, ['W', 'S', 'A', 'D', 'YL', 'YR']].to_dict(), **HR.loc[m, ['hold_q25', 'hold_med', 'hold_q75']].to_dict(),
               'valid': T.loc[m, 'valid_routes'], 'valid_nom': T.loc[m, 'valid_routes_no_motion'],
               'route_err_med': T.loc[m, 'route_err_median'], 'pair_err_med': T.loc[m, 'pair_err_median'],
               'route_err_q25': T.loc[m, 'route_err_q25'], 'route_err_q75': T.loc[m, 'route_err_q75']}
if reg.MODELS:
    E = pd.read_csv(O / 'extra_values.csv', index_col=0)
    for m in reg.KEYS:
        s = E.loc[m]
        rows[m] = {**{a: s[f'response_{a}'] for a in ['W', 'S', 'A', 'D', 'YL', 'YR']},
                   **HR.loc[m, ['hold_q25', 'hold_med', 'hold_q75']].to_dict(),
                   'valid': s.valid_routes, 'valid_nom': s.valid_routes_no_motion, 'route_err_med': s.route_err_median,
                   'pair_err_med': s.pair_err_median, 'route_err_q25': s.route_err_q25, 'route_err_q75': s.route_err_q75}
F = pd.DataFrame(rows).T
F.index.name = 'model_key'
out = Path(args.out)
if out.exists():
    old = pd.read_csv(out, index_col=0)
    common = old.index.intersection(F.index)
    print('max change on existing rows:', float((old.loc[common] - F.loc[common, old.columns].astype(float)).abs().max().max()))
F.to_csv(out)
print(F.loc[reg.KEYS].T if reg.KEYS else F.tail(1))
