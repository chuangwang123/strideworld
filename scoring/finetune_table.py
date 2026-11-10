"""The fine-tuning table (Table J.1 of the appendix): Base / SFT / Released columns of HY-World 1.5 and SANA-WM.

Every column is one run of the full benchmark (3,640 videos), scored with the functions that produce Table 2:
route classes from figures_src/route_outcomes.py (Stall, Near), compute_metrics.load / respond / model_values
(Completed, End err., Composition, Pace, Resp.), the swap gap of figures_src/analysis/f3_diagnostics.py, and the
Rigidity offset of metrics_v3/deform_analysis.summary on the run's per-frame residuals.

usage: python3 scoring/finetune_table.py DATA_RELEASE_DIR OUT.csv
"""
import gzip, json, os, shutil, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(__file__).resolve().parent
# column -> measurement directory under measurements/
RUNS = [('HY-World 1.5', 'Base', 'finetune/n0'), ('HY-World 1.5', 'SFT', 'finetune/b2'),
        ('HY-World 1.5', 'Released', 'hunyuan-world-1.5'),
        ('SANA-WM', 'Base', 'finetune/sana-n0'), ('SANA-WM', 'SFT', 'finetune/sana-sft'), ('SANA-WM', 'Released', 'sana-wm')]
# MUSIQ of each run (image quality, from the scoring report of the run; not part of any benchmark score)
MUSIQ = {'finetune/n0': 69.5, 'finetune/b2': 70.6, 'hunyuan-world-1.5': 71.6,
         'finetune/sana-n0': 74.2, 'finetune/sana-sft': 74.3, 'sana-wm': 69.6}


def swap_gap(recs):
    """Median distance between the ends of the two routes of a swap pair (G2-G4, both completed), in strides."""
    tn = lambda M: float(np.hypot(*M[:2, 2]))
    pairs = {}
    for x in recs:
        if 'pair_id' in x:
            pairs.setdefault((x['scene'], x['case'], x['pair_id']), {})[x['member']] = x
    gaps = []
    for d in pairs.values():
        if 'A' not in d or 'B' not in d or d['A']['family'] == 'G1':
            continue
        a, b = d['A'], d['B']
        if a['route_valid'] and b['route_valid']:
            Ha, Hb = np.array(a['observed_endpoint_se2']), np.array(b['observed_endpoint_se2'])
            gaps.append(tn(np.linalg.inv(Ha) @ Hb) / a['b_t'])
    return float(np.median(gaps)) if gaps else float('nan')


def rigidity_offset(p, lo=2.0, hi=8.0):
    """deform_analysis.summary on one run: median per-frame mean offset over frames at 2-8 deg of image motion."""
    F = pd.read_csv(p)
    V = pd.read_csv(p.parent / 'rigidity_videos.csv.gz')
    cnt = F.groupby('record_id').size().rename('got').reset_index()
    ok = V.merge(cnt, on='record_id', how='left')
    F = F[F.record_id.isin(ok[ok.got == ok.frames].record_id)]
    G = F[(F.t >= 1) & F.camera_valid & (F.n_scored >= 12) & np.isfinite(F.err_mean_deg)]
    return float(G[(G.par_med_deg >= lo) & (G.par_med_deg < hi)].err_mean_deg.median())


def main(rel, out):
    M = Path(rel) / 'measurements'
    os.environ.setdefault('SW_DATA', '/nonexistent')            # route_outcomes builds its default paths at import
    os.environ.setdefault('SW_REGISTRY', str(S / 'registry_release.json'))
    sys.path.insert(0, str(S / 'metrics_v3')); sys.path.insert(0, str(S / 'figures_src'))
    import compute_metrics as CM
    import route_outcomes as RO
    work = Path(tempfile.mkdtemp(prefix='finetune_table_'))
    rows = []
    try:
        for model, col, run in RUNS:
            recs = json.load(gzip.open(M / run / 'scores.json.gz', 'rt'))
            mk = {r['model'] for r in recs}; assert len(mk) == 1, mk
            mk = mk.pop()
            d = work / run.replace('/', '_'); (d / 'obs').mkdir(parents=True)
            json.dump(recs, open(d / 'all_scores.json', 'w'))
            pd.read_csv(M / run / 'observations.csv.gz').assign(model_key=mk).to_csv(d / 'obs' / f'{mk}.observations.csv', index=False)
            pd.read_csv(M / run / 'initial_view_depths.csv').assign(model_key=mk).to_csv(d / 'initial_view_depths.csv', index=False)
            RO.SCORES = [d / 'all_scores.json']
            T = RO.route_outcomes().loc[mk]
            A, R = CM.load(d, 'all_scores.json', 'initial_view_depths.csv', 'obs/*.observations.csv')
            V = CM.model_values(A, R, CM.respond(A)).loc[mk]
            rows.append(dict(model=model, column=col, run=run, videos=len(recs), stall=float(T.get('stalled', 0.0)),
                             completed=float(V['valid_routes']), end_err=float(V['route_err_median']),
                             swap_gap=swap_gap(recs), near=float(T.get('near', 0.0)), composition=float(V['composition']),
                             pace=float(V['hold']), response=float(V['response']),
                             offset=rigidity_offset(M / run / 'rigidity_perframe.csv.gz'), musiq=MUSIQ[run]))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    T = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(out, index=False)
    print(T.drop(columns=['run', 'videos']).round(2).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
