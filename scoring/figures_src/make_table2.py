"""Table 2 (main results, grouped by interface) Writes tables/table2_body.tex and table2.csv."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MV = ROOT / 'metrics_v3' / 'outputs'
DEFORM = (Path(__file__).resolve().parent / 'data_f1' / 'deform_summary.csv')
OUTC = ROOT / 'figures_src' / 'out' / 'findings_v7' / 'F3_route_outcomes.csv'   # written by findings_v7.py
F3D = ROOT / 'figures_src' / 'out' / 'f3_diag' / 'by_model.csv'  # analysis/f3_diagnostics.py
GROUPS = [('Discrete keys', ['matrix-game-3', 'matrix-game-2', 'sana-wm', 'dreamx-world', 'hunyuan-world-1.5']),
          ('Camera path', ['hy-world-2', 'lyra2', 'lingbot-world-v2', 'solarwm-h3', 'worldcrafter-fast',
                           'matrix-game-3.5', 'alaya-world', 'evoke-turbo', 'hy-worldplay']),
          ('Text prompt', ['h3-world'])]   # membership of each interface; rows are sorted by Composition below
NAMES = {'matrix-game-2': 'Matrix-Game 2.0', 'matrix-game-3': 'Matrix-Game 3.0', 'matrix-game-3.5': 'Matrix-Game 3.5',
         'sana-wm': 'SANA-WM', 'dreamx-world': 'DreamX-World', 'hunyuan-world-1.5': 'HY-World 1.5',
         'hy-world-2': 'HY-World 2.0', 'lingbot-world-v2': 'LingBot-World v2', 'alaya-world': 'AlayaWorld',
         'h3-world': 'H3-World', 'lyra2': 'Lyra 2.0', 'evoke-turbo': 'EVOKE', 'worldcrafter-fast': 'WorldCrafter',
         'solarwm-h3': 'SolarWM', 'hy-worldplay': 'HY-WorldPlay'}
MUSIQ = {'matrix-game-2': 67.1, 'matrix-game-3': 72.1, 'matrix-game-3.5': 72.3, 'sana-wm': 69.6, 'dreamx-world': 74.3,
         'hunyuan-world-1.5': 71.6, 'hy-world-2': 72.9, 'lingbot-world-v2': 72.2, 'alaya-world': 71.5, 'h3-world': 63.3,
         'lyra2': 75.5, 'evoke-turbo': 69.6, 'worldcrafter-fast': 67.8, 'solarwm-h3': 73.0}
_REG = json.load(open(__import__('os').environ['SW_REGISTRY']))
MUSIQ.update({k: v['musiq'] for k, v in _REG.get('base_external', {}).items()})      # values kept in the registry
MUSIQ.update({m['key']: m['external']['musiq'] for m in _REG['models']})             # registered models

M = pd.read_csv(MV / 'main_table_values.csv').rename(columns={'Unnamed: 0': 'model'}).set_index('model')
for _m in _REG['models']:   # models scored after the first thirteen (registry), e.g. SolarWM, HY-WorldPlay
    _V = pd.read_csv(MV / _m['values_out']).rename(columns={'Unnamed: 0': 'model'}).set_index('model')
    M = pd.concat([M, _V[[c for c in M.columns if c in _V.columns]]])
D = pd.read_csv(DEFORM).set_index('model')
TW = pd.read_csv(MV / 'rigid_twin_ratio.csv').set_index('model')   # scoring/rigid_twin_ratio.py
Hd = pd.read_csv(MV / 'hold_ratio.csv').set_index('model_key')
T = pd.read_csv(OUTC).set_index('model')
F = pd.read_csv(F3D).set_index('model')
rows = []
for g, ms in GROUPS:
    for m in ms:
        if m not in M.index:    # a registered model whose run is not in the current data (e.g. before it is merged)
            continue
        off = D.loc[m, 'offset_med_deg'] if (m in D.index and D.loc[m, 'n_frames'] >= 1000) else np.nan
        rows.append(dict(group=g, model=m, name=NAMES[m], resp=M.loc[m, 'response'], offset=off,
                         vs_rigid=TW.loc[m, 'vs_rigid'],
                         early=D.loc[m, 'early'] if m in D.index else np.nan,
                         late=D.loc[m, 'late'] if m in D.index else np.nan,
                         pace=M.loc[m, 'hold'], ratio=Hd.loc[m, 'hold_med'], comp=M.loc[m, 'composition'],
                         near=T.loc[m, 'near'], stall=T.loc[m, 'stalled'], completed=M.loc[m, 'valid_routes'],
                         enderr=M.loc[m, 'route_err_median'], swap=F.loc[m, 'swap_gap'], floor=F.loc[m, 'rep_floor'],
                         scale=F.loc[m, 'scale'],
                         views=M.loc[m, 'view_translation'], musiq=MUSIQ[m]))
R = pd.DataFrame(rows)
# order by Composition score within each interface (Fig. F.2 and Table F.8 read this order from table2.csv)
R['gi'] = R.group.map([g for g, _ in GROUPS].index)
R = R.sort_values(['gi', 'comp'], ascending=[True, False]).drop(columns='gi').reset_index(drop=True)
GROUPS = [(g, R[R.group == g].model.tolist()) for g, _ in GROUPS]
R.to_csv(ROOT / 'tables' / 'table2.csv', index=False)


# Pace rule (Sec. 5.2): a model changes pace if its median ratio is outside 0.8-1.25 with the stride of Eq. (3) AND with
# equal step weights (analysis/stride_alt.py); such ratios are printed in red.
_F2ALT = pd.read_csv(ROOT / 'figures_src' / 'out' / 'stride_alt' / 'f2_ratio.csv').set_index('model')
EQW_STEADY = {m for m in _F2ALT.index if not bool(_F2ALT.loc[m, 'out_M4'])}
NOISE = {'alaya-world'}   # stride close to the measurement noise: not ranked in Pace/Composition
GREY_NOISE = {'ratio', 'pace', 'comp', 'near', 'stall', 'enderr', 'swap'}   # stride-based cells of these models are shown in grey


def rank_fmt(col, higher=True, fmt='{:.1f}', rank_noise=False):
    """Bold best, underline second. Models in NOISE are skipped unless the column does not depend on strides."""
    v = R[col].values.astype(float)
    order = np.argsort(-v if higher else v)
    order = [i for i in order if np.isfinite(v[i]) and (rank_noise or R.model[i] not in NOISE)]
    best, second = order[0], order[1]
    out = []
    for i, x in enumerate(v):
        if not np.isfinite(x):
            out.append('--')
            continue
        s = fmt.format(x)
        if i == best:
            s = r'\textbf{' + s + '}'
        elif i == second:
            s = r'\underline{' + s + '}'
        out.append(s)
    return out


def tie_fmt(col, fmt='{:.1f}'):
    """Lower is better; bold every model at the best printed value, underline those at the second."""
    txt = [fmt.format(x) for x in R[col].values.astype(float)]
    vals = sorted({float(t) for t in txt})
    return [r'\textbf{' + t + '}' if float(t) == vals[0] else r'\underline{' + t + '}' if float(t) == vals[1] else t
            for t in txt]


cols = {
    'resp': [f'{x:.1f}' for x in R.resp],
    'offset': rank_fmt('offset', higher=False, fmt='{:.2f}', rank_noise=True),   # Rigidity does not use strides
    'vs_rigid': tie_fmt('vs_rigid'),
    'pace': rank_fmt('pace'),
    'ratio': [f'{x:.2f}' if (m in NOISE or 0.8 <= x <= 1.25 or m in EQW_STEADY)
              else (r'\textcolor{alertred}{' + f'{x:.2f}' + '}') for x, m in zip(R.ratio, R.model)],
    'comp': rank_fmt('comp'),
    'near': rank_fmt('near'),
    'enderr': rank_fmt('enderr', higher=False, fmt='{:.2f}'),
    'stall': rank_fmt('stall', higher=False),
    'swap': rank_fmt('swap', higher=False, fmt='{:.2f}'),
    'musiq': [r'\textcolor{gray}{' + f'{x:.1f}' + '}' for x in R.musiq],
}
lines = []
lines.append(r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccccc@{}}')
lines.append(r'\toprule')
lines.append(r' & & \multicolumn{2}{c}{\cellcolor{fone}Rigidity} & \multicolumn{2}{c}{\cellcolor{ftwo}Pace} '
             r'& \multicolumn{5}{c}{\cellcolor{fthree}Composition} & \\')
lines.append(r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-11}')
lines.append(r'Model & Resp.\,(\%) & Offset\,(°)$\downarrow$ & vs.\,rigid$\downarrow$ & Ratio\,(${\to}1$) & Score$\uparrow$ & Stall\,(\%)$\downarrow$ & '
             r'End err.\,(str.)$\downarrow$ & Swap gap\,(str.)$\downarrow$ & Near\,(\%)$\uparrow$ & Score$\uparrow$ & \textcolor{gray}{MUSIQ} \\')
for g, ms in GROUPS:
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{12}{@{}l}{\textit{' + g + r'}}\\[-1pt]')
    for i, r in R[R.group == g].iterrows():
        grey = lambda c, v: (r'\textcolor{gray}{' + v + '}') if (r['model'] in NOISE and c in GREY_NOISE) else v
        cells = [r['name'] + (r'$^\dagger$' if r['model'] in NOISE else '')] + [grey(c, cols[c][i]) for c in ['resp', 'offset', 'vs_rigid', 'ratio', 'pace', 'stall', 'enderr', 'swap',
                                                    'near', 'comp', 'musiq']]
        lines.append(' & '.join(cells) + r' \\')
lines.append(r'\bottomrule')
lines.append(r'\end{tabular*}')
open(ROOT / 'tables' / 'table2_body.tex', 'w').write('\n'.join(lines).replace('°', r'$^\circ$') + '\n')
print(R.round(2).to_string())
