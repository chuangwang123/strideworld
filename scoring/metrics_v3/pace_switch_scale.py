"""Appendices E and F (pace, switches, route scale), the 13 models and the registered models (registry_release.json, registry order):
 (1) Pace profile of held actions: per-step motion along the commanded axis (increment of u_k), divided by its mean over
     steps 1-4, median over the single-action videos measurable at all 16 steps with a positive steps-1-4 mean; and the
     ratio of the mean increment over steps 13-16 to that over steps 2-5 (median over videos), which is insensitive to a
     slow first step.
 (2) Where routes stall: share of route steps failing the 10% motion check on the first step after an action switch
     and inside an action block; valid routes when switch steps are exempt from the motion check (all other checks kept).
 (3) Route scale: |observed| / |predicted| endpoint displacement of valid routes whose predicted displacement exceeds
     two steps (b_t), median and quartiles.
python3 pace_switch_scale.py --out outputs/pace_switch_scale.csv
"""
import argparse, glob, json

import numpy as np
import pandas as pd

import registry as reg

KEY = ['model_key', 'scene_id', 'case_id', 'seed', 'action']


def pace(o):
    o = o[o.dose <= 16]
    piv = o.pivot_table(index=KEY, columns='dose', values='u_signed', aggfunc='first').reindex(columns=range(1, 17))
    val = o.pivot_table(index=KEY, columns='dose', values='measurable', aggfunc='first').reindex(
        index=piv.index, columns=range(1, 17)).fillna(False).astype(bool)
    u = piv[val.all(axis=1)]
    u0 = u.copy(); u0.insert(0, 0, 0.0)
    inc = u0.diff(axis=1).iloc[:, 1:]
    base = inc.iloc[:, 0:4].mean(axis=1); ok = base > 0
    r = inc[ok].div(base[ok], axis=0)
    early2 = inc[ok].iloc[:, 1:5].mean(axis=1)
    late = inc[ok].iloc[:, 12:16].mean(axis=1)
    return dict(n_pace=int(ok.sum()), step1=r.iloc[:, 0].median(), s5_8=r.iloc[:, 4:8].mean(axis=1).median(),
                s9_12=r.iloc[:, 8:12].mean(axis=1).median(), s13_16=r.iloc[:, 12:16].mean(axis=1).median(),
                late_vs_2_5=(late / early2)[early2 > 0].median())


def routes(R):
    out = {}
    for mk in sorted({r['model'] for r in R}):
        rows, steps = [], []
        for r in (x for x in R if x['model'] == mk):
            st = r['steps']; fails = set(r.get('failed_motion_steps') or [])
            for i in range(1, len(st) + 1):
                first = i == 1 or st[i - 2] != st[i - 1]
                steps.append((first, i in fails))
            inblock = [i for i in fails if i >= 2 and st[i - 2] == st[i - 1]]
            base_ok = bool(r['route_quality_pass_without_motion']) and bool(r['calibration_pass'])
            ratio = np.nan
            if r['route_valid']:
                P = np.array(r['predicted_endpoint_se2']); O = np.array(r['observed_endpoint_se2'])
                pn, on = np.hypot(P[0, 2], P[1, 2]), np.hypot(O[0, 2], O[1, 2])
                if pn / r['b_t'] > 2:
                    ratio = on / pn
            rows.append(dict(valid=bool(r['route_valid']), switch_exempt=base_ok and not inblock, ratio=ratio))
        S = pd.DataFrame(steps, columns=['first', 'fail']); T = pd.DataFrame(rows)
        rr = T.ratio.dropna()
        out[mk] = dict(fail_switch=100 * S[S['first']].fail.mean(), fail_block=100 * S[~S['first']].fail.mean(),
                       valid=100 * T.valid.mean(), valid_switch_exempt=100 * T.switch_exempt.mean(),
                       scale_n=len(rr), scale_med=rr.median(), scale_q25=rr.quantile(.25), scale_q75=rr.quantile(.75))
    return pd.DataFrame(out).T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(reg.OUT / 'pace_switch_scale.csv')); a = ap.parse_args()
    fs = sorted(glob.glob(str(reg.BASE / 'obs' / '*.observations.csv')))
    P = {pd.read_csv(f, nrows=1).model_key.iloc[0]: pace(pd.read_csv(f)) for f in fs}
    P.update({m.key: pace(reg.read_obs(m)) for m in reg.MODELS})
    P = pd.DataFrame(P).T
    R = [r for r in json.load(open(reg.BASE / 'all_scores.json')) if 'pair_id' in r]
    for m in reg.MODELS:
        R += [r for r in reg.records(m) if 'pair_id' in r]
    T = P.join(routes(R))
    T.to_csv(a.out)
    pd.set_option('display.width', 250)
    print(T.round(2).to_string())


if __name__ == '__main__':
    main()
