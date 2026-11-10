"""Cross-view variation net of seed noise (answers whether views matter beyond repeated-generation noise).
For each model, scene and translation action, with u[v, s] the axial step of view v and seed s (complete 3-seed groups):
between-view variance of the seed means minus the within-view variance / 3 estimates the variance due to the view;
its square root over the mean |u| is the view-only RSD (0 when negative). Averaged within scene, over scenes, then over
the four translations, as in the stride-variation table of the appendix (Table H.5); plus the share of the step's variance (views + seeds) that is due to the view.
Models: the 13 and the registered models (registry_release.json).
python3 view_excess.py --out outputs/view_excess.csv
"""
import argparse

import numpy as np
import pandas as pd

import registry as reg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(reg.OUT / 'view_excess.csv')); a = ap.parse_args()
    U = pd.concat([pd.read_csv(reg.BASE / 'unit_estimates.csv')] + [reg.read_csv(m, 'units') for m in reg.MODELS])
    U = U[U.action.isin(['W', 'S', 'A', 'D'])]
    rows = []
    for (mk, sc, act), g in U.groupby(['model_key', 'scene_id', 'action']):
        piv = g.pivot_table(index='case_id', columns='seed', values='signed_command_component', aggfunc='first')
        avail = g.pivot_table(index='case_id', columns='seed', values='unit_available', aggfunc='first')
        ok = (piv.notna() & avail.fillna(False).astype(bool)).all(axis=1) & (piv.shape[1] == 3)
        x = piv[ok].values
        if x.shape[0] < 2:
            continue
        den = np.mean(np.abs(x))
        if not den > 0:
            continue
        within = x.var(axis=1, ddof=1).mean()
        between = x.mean(axis=1).var(ddof=1)
        view_var = max(between - within / 3, 0.0)
        rows.append(dict(model_key=mk, scene_id=sc, action=act, view_rsd=100 * np.sqrt(view_var) / den,
                         seed_rsd=100 * np.sqrt(within) / den, view_share=view_var / (view_var + within) if view_var + within > 0 else np.nan))
    G = pd.DataFrame(rows)
    T = G.groupby(['model_key', 'action', 'scene_id'])[['view_rsd', 'seed_rsd', 'view_share']].mean() \
         .groupby(['model_key', 'action']).mean().groupby('model_key').mean()
    T.to_csv(a.out)
    print(T.round(2).to_string())


if __name__ == '__main__':
    main()
