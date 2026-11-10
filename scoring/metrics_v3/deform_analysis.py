"""F1 (scene deformation) at matched image motion, from the per-frame re-extraction of the frozen pose artifacts.

Inputs (SW_DEFORM, required; written by scoring/assemble.py):
  perframe_all.csv.gz                     per-frame held-out-anchor residuals, camera motion, expected image motion
  perframe_all_videos.csv.gz              per-video check value (frozen Scene-stability score recomputed)
plus the frozen depths (initial_view_depths*.csv) for the depth normalisation.
Frame filter: t >= 1, recovered camera, >= 12 scored held-out anchors (the frozen frame gate up to the pose-quality
thresholds, which are not stored per frame).

python3 deform_analysis.py   -> outputs/deform_*.csv (printed summaries)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import registry as reg

D = Path(os.environ['SW_DEFORM'])
OUT = reg.OUT
BINS = [0, 0.5, 1, 2, 4, 8, 16, 32, 90]           # expected image motion of the held-out anchors (deg)
NAME = {'solarwm': 'solarwm-h3'}


def read_gz(p):
    """Read a gzip CSV that may be truncated (stream interrupted): keep complete lines only."""
    import io, zlib
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = []
    with open(p, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            try:
                out.append(d.decompress(b))
            except zlib.error:
                break
    txt = b''.join(out)
    txt = txt[:txt.rfind(b'\n') + 1]
    return pd.read_csv(io.BytesIO(txt))


def load():
    F, V = [], []
    for part in ['all']:
        p = D / f'perframe_{part}.csv.gz'
        if not p.exists():
            print('missing', p); continue
        f = read_gz(p)
        pv = D / f'perframe_{part}_videos.csv.gz'
        if pv.exists():
            v = read_gz(pv)
            # keep only videos whose frames are all present (an interrupted stream can cut the last ones)
            cnt = f.groupby(['model', 'record_id']).size().rename('got').reset_index()
            v = v.merge(cnt, on=['model', 'record_id'], how='left')
            ok = v[v.got == v.frames][['model', 'record_id']]
            print(part, 'videos', len(v), 'complete', len(ok))
            f = f.merge(ok, on=['model', 'record_id'])
            V.append(v)
        F.append(f)
    F = pd.concat(F, ignore_index=True)
    V = pd.concat(V, ignore_index=True) if V else None
    F['model'] = F.model.replace(NAME)
    dep = [pd.read_csv(reg.BASE / 'initial_view_depths.csv')]
    for m in reg.MODELS:
        dep.append(reg.read_csv(m, 'depths'))
    dep = pd.concat(dep, ignore_index=True)[['model_key', 'scene_id', 'case_id', 'depth_cm']].drop_duplicates()
    F = F.merge(dep.rename(columns={'model_key': 'model'}), on=['model', 'scene_id', 'case_id'], how='left')
    return F, V


def main():
    F, V = load()
    print('frames', len(F), 'videos', F.record_id.nunique(), 'models', F.model.nunique(), 'missing depth', F.depth_cm.isna().sum())
    G = F[(F.t >= 1) & F.camera_valid & (F.n_scored >= 12) & np.isfinite(F.err_mean_deg)].copy()
    G['disp_pct'] = 100 * np.hypot(G.tx, G.tz) / G.depth_cm
    G['yaw_abs'] = G.yaw_deg.abs()
    G['mot'] = G.par_med_deg
    G['bin'] = pd.cut(G.mot, BINS, right=False)
    # per-video and per-model residual (observation-weighted, as the frozen score)
    G['w_err'] = G.err_mean_deg * G.n_scored
    vid = G.groupby(['model', 'record_id']).agg(w=('w_err', 'sum'), n=('n_scored', 'sum'), mot=('mot', 'median'),
                                                disp=('disp_pct', 'median'), frames=('t', 'size'))
    vid['err'] = vid.w / vid.n
    vid['ray'] = 100 / (1 + vid.err)
    mod = vid.groupby('model').agg(ray=('ray', 'mean'), err_med=('err', 'median'), mot_med=('mot', 'median'),
                                   disp_med=('disp', 'median'), n=('err', 'size'))
    print(mod.sort_values('ray').round(3).to_string())
    # residual vs expected image motion (median of frame means per bin)
    B = (G.groupby(['model', 'bin'], observed=True)
         .agg(err_med=('err_mean_deg', 'median'), err_q25=('err_mean_deg', lambda s: s.quantile(.25)),
              err_q75=('err_mean_deg', lambda s: s.quantile(.75)), n=('err_mean_deg', 'size'), mot=('mot', 'median'))
         .reset_index())
    B['lo'] = B.bin.apply(lambda iv: iv.left); B = B.drop(columns='bin')
    B.to_csv(OUT / 'deform_by_motion.csv', index=False)
    P = B.pivot(index='model', columns='lo', values='err_med')
    print('\nmedian held-out residual (deg) by expected image motion (deg):')
    print(P.round(3).to_string())
    N = B.pivot(index='model', columns='lo', values='n')
    print(N.to_string())
    mod.to_csv(OUT / 'deform_by_model.csv')
    G[['model', 'record_id', 't', 'err_mean_deg', 'mot', 'disp_pct', 'yaw_abs', 'n_scored']].to_csv(
        OUT / 'deform_frames_compact.csv.gz', index=False)


if __name__ == '__main__':
    main()


def summary(F=None, lo=2.0, hi=8.0):
    """Per model: median held-out offset (deg) over frames whose anchors moved lo-hi deg in the image, frame counts,
    and the same for the first and last thirds of the videos. Writes outputs/deform_summary.csv."""
    if F is None:
        F, _ = load()
    G = F[(F.t >= 1) & F.camera_valid & (F.n_scored >= 12) & np.isfinite(F.err_mean_deg)].copy()
    G['T'] = G.groupby(['model', 'record_id']).t.transform('max')
    G['tau'] = G.t / G['T']
    b = G[(G.par_med_deg >= lo) & (G.par_med_deg < hi)]
    rows = []
    for m, g in b.groupby('model'):
        rows.append(dict(model=m, offset_med_deg=g.err_mean_deg.median(), n_frames=len(g),
                         n_videos=g.record_id.nunique(),
                         early=g[g.tau < 1 / 3].err_mean_deg.median(), late=g[g.tau >= 2 / 3].err_mean_deg.median()))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / 'deform_summary.csv', index=False)
    return S
