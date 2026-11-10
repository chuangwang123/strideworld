"""Rigidity offset relative to the rigid twins (Table 2, column "vs. rigid"; App. G, Table G.2).

For every model, 20 route videos are paired with a rigid twin: the start view and its ground-truth depth rendered
along the camera path recovered from the generated video, in the model's resolution, frame rate and codec, and scored
with the unchanged pipeline. Per model, over the frames whose twin is eligible, that are measurable in both videos and
whose image motion in the generated video lies in [2, 8] deg:

    measured = median per-frame mean offset of the twin      real = the same for the generated video
    ratio    = real / measured

usage: python3 scoring/rigid_twin_ratio.py DATA_RELEASE_DIR OUT.csv
"""
import sys
from pathlib import Path

import pandas as pd


def row(g, model, lo=2.0, hi=8.0):
    vids = g.groupby('record_id').floor_eligible.first()
    band = g.real_measurable.astype(bool) & g.real_par_med_deg.between(lo, hi)
    P = g[band & g.floor_eligible.astype(bool) & g.measurable.astype(bool)]
    mm, rm = P.err_mean_deg.median(), P.real_err_mean_deg.median()
    md, rd = P.err_med_deg.median(), P.real_err_med_deg.median()
    return dict(model=model, n_twins_used=int(vids.sum()), n_twins_total=int(len(vids)), n_frames=len(P),
                twin_offset_deg=mm, real_offset_deg=rm, vs_rigid=rm / mm, twin_offset_median_deg=md,
                real_offset_median_deg=rd, vs_rigid_median=rd / md)


def main(rel, out):
    rows = []
    for f in sorted((Path(rel) / 'measurements').glob('*/rigid_twin_perframe.csv.gz')):
        g = pd.read_csv(f)
        rows.append(row(g, f.parent.name))
    T = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(out, index=False)
    print(T.round(3).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
