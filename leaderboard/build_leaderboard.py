"""Build leaderboard.json from the main-results table (results/table2.csv) and the stride check (f2_ratio.csv).

usage: python3 build_leaderboard.py --table results/table2.csv --f2 results/f2_ratio.csv --out leaderboard.json

Rules (as in Table 2 of the paper):
  * Pace: a model changes pace if its median pace ratio lies outside 0.8-1.25 with the weighted stride of the paper
    AND with equal step weights; outside with the weighted stride only -> "outside_eq1_only".
  * A model whose stride is close to the measurement noise (AlayaWorld) is listed but not ranked on stride-based
    columns (pace score, composition, near, stall, end error, swap gap). The Rigidity offset uses no stride and is
    ranked for every model with enough frames.
"""
import argparse, json, math
import csv

NOISE = {'alaya-world'}
STRIDE_BASED = {'pace', 'comp', 'near', 'stall', 'enderr', 'swap'}
DEFAULT_COMMAND = {'matrix-game-2', 'matrix-game-3', 'sana-wm', 'dreamx-world', 'hunyuan-world-1.5', 'evoke-turbo',
                   'hy-worldplay'}   # run at the default command per step of their release (App. A)
COLUMNS = [  # key, label, group, unit, higher_is_better, decimals
    ('resp', 'Resp.', 'Response', '%', True, 1),
    ('offset', 'Offset', 'Rigidity', 'deg', False, 2),
    ('ratio', 'Ratio', 'Pace', '', None, 2),
    ('pace', 'Score', 'Pace', '', True, 1),
    ('comp', 'Score', 'Composition', '', True, 1),
    ('near', 'Near', 'Composition', '%', True, 1),
    ('stall', 'Stall', 'Composition', '%', False, 1),
    ('enderr', 'End err.', 'Composition', 'strides', False, 2),
    ('swap', 'Swap gap', 'Composition', 'strides', False, 2),
    ('musiq', 'MUSIQ', 'Image quality', '', True, 1),
]


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', required=True)
    ap.add_argument('--f2', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    f2 = {r['model']: r for r in csv.DictReader(open(a.f2))}
    rows = []
    for r in csv.DictReader(open(a.table)):
        m = r['model']
        ratio = num(r['ratio'])
        if m in NOISE or ratio is None or 0.8 <= ratio <= 1.25:
            pace_flag = 'steady' if (ratio is not None and 0.8 <= ratio <= 1.25) else 'not_assessed'
        elif f2[m]['out_M4'] == 'False':
            pace_flag = 'outside_eq1_only'
        else:
            pace_flag = 'changes_pace'
        vals = {k: num(r[k]) for k, *_ in COLUMNS}
        rows.append(dict(key=m, name=r['name'], interface=r['group'],
                         command='release default' if m in DEFAULT_COMMAND else 'benchmark magnitude',
                         stride_near_noise=m in NOISE, pace_flag=pace_flag, completed=num(r['completed']),
                         vs_rigid=(None if num(r.get('vs_rigid')) is None else round(num(r['vs_rigid']), 2)),
                         values={k: (None if v is None else round(v, 4)) for k, v in vals.items()}))
    # ranks (1 = best) among ranked models; None where not ranked
    for k, _, _, _, hib, _ in COLUMNS:
        if hib is None or k == 'musiq':
            continue
        cand = [x for x in rows if x['values'][k] is not None and not (x['stride_near_noise'] and k in STRIDE_BASED)]
        cand.sort(key=lambda x: -x['values'][k] if hib else x['values'][k])
        for i, x in enumerate(cand, 1):
            x.setdefault('rank', {})[k] = i
    out = dict(
        benchmark='StrideWorld',
        version='v1.0',
        protocol=dict(start_views=140, scenes=30, rendered_scenes=24, real_scenes=6, requests_per_view=26,
                      videos_per_model=3640, steps=16, actions=['W', 'S', 'A', 'D', 'yaw-left', 'yaw-right']),
        columns=[dict(key=k, label=l, group=g, unit=u, higher_is_better=h, decimals=d) for k, l, g, u, h, d in COLUMNS],
        notes={
            'resp': 'Share of single-action videos in which the camera moves the commanded way (2 deg or 2% of the start view depth by step 16).',
            'offset': 'Median offset of held-out anchors from their rigid positions at 2-8 deg of image motion. Hover a value for the offset relative to a rigid counterpart of the start view along the same camera path (1 = rigid).',
            'ratio': 'Pace ratio: observed over predicted motion after 16 steps (1 = steady; 0.8-1.25 counts as steady).',
            'pace': 'Pace score: usable control of a held action (50 = off by one predicted path length).',
            'comp': 'Composition score: route and swap-pair scores (translation and yaw averaged); incomplete routes score 0.',
            'near': 'Routes ending within one stride of the prediction (%).',
            'stall': 'Routes with a stalled step (%).',
            'enderr': 'Median end error of completed routes (strides).',
            'swap': 'Median gap between the ends of a swap pair (strides).',
            'musiq': 'MUSIQ no-reference image quality (not part of the benchmark scores).',
            'stride_near_noise': 'Strides close to the measurement noise: stride-based values shown but not ranked.',
        },
        models=rows)
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f'{len(rows)} models -> {a.out}')


if __name__ == '__main__':
    main()
