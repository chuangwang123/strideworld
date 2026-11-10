"""LaTeX rows for appendix tables: pace profiles (Table E.1), stall location
and route scale, view effects net of generation noise, scene-bootstrap intervals of
Table 2 (Table H.2 of the appendix) and Spearman correlations between metrics.
Inputs: outputs/pace_switch_scale.csv, view_excess.csv, bootstrap_table2.csv, main_table_values.csv, extra_values.csv,
make_main_table.VISUAL and the registry's external values, unit estimates. Output: outputs/appendix_new_tables.tex (one
block per table). Rows: the thirteen, then the registered models (registry_release.json) in registry order.
"""
import numpy as np
import pandas as pd

from compute_metrics import ORDER
from make_main_table import NAMES, VISUAL
import registry as reg

O = reg.OUT
MODELS = ORDER + reg.KEYS
NM = dict(NAMES); NM.update(reg.names())
f1 = lambda v: f'{v:.1f}'
f2 = lambda v: f'{v:.2f}'
blocks = {}

P = pd.read_csv(O / 'pace_switch_scale.csv', index_col=0)
blocks['pace'] = [f"{NM[m]} & {int(P.loc[m, 'n_pace'])} & {f2(P.loc[m, 'step1'])} & {f2(P.loc[m, 's5_8'])} & {f2(P.loc[m, 's9_12'])} & "
                  f"{f2(P.loc[m, 's13_16'])} & {f2(P.loc[m, 'late_vs_2_5'])} \\\\" for m in MODELS]
blocks['stall_scale'] = [f"{NM[m]} & {f1(P.loc[m, 'fail_switch'])} & {f1(P.loc[m, 'fail_block'])} & {f1(P.loc[m, 'valid'])} & "
                         f"{f1(P.loc[m, 'valid_switch_exempt'])} & {int(P.loc[m, 'scale_n'])} & {f2(P.loc[m, 'scale_med'])} "
                         f"[{f2(P.loc[m, 'scale_q25'])}, {f2(P.loc[m, 'scale_q75'])}] \\\\" for m in MODELS]
V = pd.read_csv(O / 'view_excess.csv', index_col=0)
T = pd.concat([pd.read_csv(O / 'main_table_values.csv', index_col=0)]
              + ([pd.read_csv(O / 'extra_values.csv', index_col=0)] if reg.MODELS else []))
blocks['view_excess'] = [f"{NM[m]} & {f1(T.loc[m, 'repeat_translation'])} & {f1(T.loc[m, 'view_translation'])} & "
                         f"{f1(V.loc[m, 'view_rsd'])} & {f2(V.loc[m, 'view_share'])} \\\\" for m in MODELS]
B = pd.read_csv(O / 'bootstrap_table2.csv', index_col=0)
iv = lambda m, c: f"[{B.loc[m, c + '_lo']:.1f}, {B.loc[m, c + '_hi']:.1f}]"
rk = lambda m, c: (f"{int(B.loc[m, c + '_rank_lo'])}" if B.loc[m, c + '_rank_lo'] == B.loc[m, c + '_rank_hi']
                   else f"{B.loc[m, c + '_rank_lo']:.0f}--{B.loc[m, c + '_rank_hi']:.0f}")
blocks['bootstrap'] = [f"{NM[m]} & {iv(m, 'response')} & {iv(m, 'hold')} & {iv(m, 'valid_routes')} & {iv(m, 'composition')} & "
                       f"{iv(m, 'repeat')} & {iv(m, 'view')} & {rk(m, 'hold')} & {rk(m, 'composition')} \\\\" for m in MODELS]
# correlations
VIS = dict(VISUAL)
VIS.update({m.key: tuple(m.external[x] for x in ('scene_stability', 'musiq', 'aesthetic')) for m in reg.MODELS})
U = pd.concat([pd.read_csv(reg.BASE / 'unit_estimates.csv')] + [reg.read_csv(m, 'units') for m in reg.MODELS])
U = U[U.unit_available.astype(bool)]
X = T.loc[MODELS, ['response', 'hold', 'valid_routes', 'composition', 'repeat_translation', 'view_translation']].astype(float).copy()
X['scene'] = [VIS[m][0] for m in MODELS]; X['musiq'] = [VIS[m][1] for m in MODELS]
X['step'] = U[U.action == 'W'].groupby('model_key').translation_cm.median().loc[MODELS]
C = X.corr(method='spearman')
lab = ['Response', 'Hold', 'Valid', 'Composition', 'Repeat', 'Views', 'Scene stab.', 'MUSIQ', 'Fwd. step']
rows = []
for i, c in enumerate(X.columns):
    cells = [f'{C.iloc[i, j]:+.2f}'.replace('+', '$+$').replace('-', '$-$') if j < i else '' for j in range(len(X.columns) - 1)]
    rows.append(f"{lab[i]} & " + ' & '.join(cells) + r' \\')
blocks['metric_corr'] = rows[1:]
with open(O / 'appendix_new_tables.tex', 'w') as fh:
    for k, v in blocks.items():
        fh.write(f'% ---- {k}\n' + '\n'.join(v) + '\n')
print(open(O / 'appendix_new_tables.tex').read())
