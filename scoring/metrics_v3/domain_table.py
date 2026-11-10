"""Table F.1: Response, Hold, Valid routes and Composition on UE5 and ETH3D starts, for the 13 benchmark
models and the registered models (registry_release.json, registry order). Single-action weights are renormalized within each
domain (model_values does this per model).
Marks: best bold and second underlined per column (ties share a mark), except Response.

python3 domain_table.py [--data <data13>] [--out outputs]
"""
import argparse
from pathlib import Path

import pandas as pd

from compute_metrics import ORDER, load, respond, model_values
import registry as reg

NAMES = {'matrix-game-2': 'Matrix-Game 2.0', 'matrix-game-3': 'Matrix-Game 3.0', 'matrix-game-3.5': 'Matrix-Game 3.5',
         'sana-wm': 'SANA-WM', 'dreamx-world': 'DreamX-World', 'hunyuan-world-1.5': 'Hunyuan-World 1.5',
         'hy-world-2': 'HY-World 2.0', 'lingbot-world-v2': 'LingBot-World v2', 'alaya-world': 'AlayaWorld',
         'h3-world': 'H3-World', 'lyra2': 'Lyra 2.0', 'evoke-turbo': 'EVOKE', 'worldcrafter-fast': 'WorldCrafter'}
NAMES.update(reg.names())


def domains(A, R):
    out = {}
    for dom, eth in (('UE5', False), ('ETH3D', True)):
        a = A.scene_id.str.startswith('eth3d') == eth; r = R.scene_id.str.startswith('eth3d') == eth
        out[dom] = model_values(A[a], R[r], respond(A[a]))
    return pd.concat(out, axis=1)


ap = argparse.ArgumentParser()
ap.add_argument('--data', default=str(reg.BASE)); ap.add_argument('--out', default=str(reg.OUT))
a = ap.parse_args(); out = Path(a.out)
A, R = load(a.data, 'all_scores.json', 'initial_view_depths.csv', 'obs/*.observations.csv')
T = domains(A, R).loc[ORDER]
order = list(ORDER)
for m in reg.MODELS:
    Ae, Re = reg.load_model(m)
    Te = domains(Ae, Re); T = pd.concat([T, Te]); order += list(Te.index)
T.to_csv(out / 'domain_values.csv')
cols = [(d, c) for d in ('UE5', 'ETH3D') for c in ('response', 'hold', 'valid_routes', 'composition')]
Rr = T[cols].round(1)


def cell(m, col):
    v = Rr.loc[m, col]; s = f'{v:.1f}'
    if col[1] != 'response':
        u = sorted(Rr[col].unique(), reverse=True)
        if v == u[0]:
            s = r'\textbf{' + s + '}'
        elif v == u[1]:
            s = r'\underline{' + s + '}'
    return s


rows = [f'{NAMES[m]} & ' + ' & '.join(cell(m, c) for c in cols) + r' \\' for m in order]
(out / 'domain_table_rows.tex').write_text('\n'.join(rows) + '\n')
print('\n'.join(rows))
