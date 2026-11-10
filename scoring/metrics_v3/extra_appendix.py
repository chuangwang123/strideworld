"""Appendix quantities for the models scored after the first thirteen (registry registry_release.json, each on its own), from
their exported unit estimates and initial-view depths: median single-step magnitudes and the
depth-normalized cross-view RSD. Printed only. The same functions reproduce the 13 printed rows when
pointed at the 13-model inputs (--check); --units/--depths replace the registry by one explicit export.

python3 extra_appendix.py [--units <unit_estimates_x.csv> --depths <initial_view_depths_x.csv>]
       [--check-units <unit_estimates.csv> --check-depths <initial_view_depths.csv>]
"""
import argparse

import numpy as np
import pandas as pd

from compute_metrics import rsd_tables
import registry as reg


def magnitudes(U):
    U = U[U.unit_available.astype(bool)].copy()
    U['mag'] = np.where(U.action.isin(['YL', 'YR']), U.yaw_abs_deg, U.translation_cm)
    return U.groupby(['model_key', 'action']).mag.median().unstack()[['W', 'S', 'A', 'D', 'YL', 'YR']]


def depth_rsd(U, dep):
    U = U.merge(dep[['model_key', 'scene_id', 'case_id', 'depth_cm']], on=['model_key', 'scene_id', 'case_id'], how='left')
    assert U.depth_cm.notna().all()
    orig = rsd_tables(U)
    U = U.assign(signed_command_component=U.signed_command_component / U.depth_cm)
    norm = rsd_tables(U)
    return pd.DataFrame({'repeat': orig.repeat_translation, 'cross_view': orig.view_translation,
                         'depth_normalized': norm.view_translation,
                         'change': norm.view_translation - orig.view_translation})


if __name__ == '__main__':      # importable (tools/sync_rows.py uses magnitudes and depth_rsd)
    ap = argparse.ArgumentParser()
    ap.add_argument('--units'); ap.add_argument('--depths')
    ap.add_argument('--check-units'); ap.add_argument('--check-depths')
    a = ap.parse_args()
    if a.check_units:
        U, d = pd.read_csv(a.check_units), pd.read_csv(a.check_depths)
        print(magnitudes(U).round(2).to_string()); print(depth_rsd(U, d).round(1).to_string())
    pairs = ([(pd.read_csv(a.units), pd.read_csv(a.depths))] if a.units else
             [(reg.read_csv(m, 'units'), reg.read_csv(m, 'depths')) for m in reg.MODELS])
    for U, d in pairs:
        print(magnitudes(U).round(2).to_string()); print(depth_rsd(U, d).round(1).to_string())
