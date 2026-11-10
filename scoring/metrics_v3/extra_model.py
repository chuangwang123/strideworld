"""Metrics for the models scored after the first thirteen (registry registry_release.json), each exported with the same five
files as the 13 benchmark models.

Usage: python3 extra_model.py
For every registered model, in registry order, one row of outputs/extra_values.csv (index model_key) with the main-table
quantities (Response, Hold, Valid routes, Route, Pair, Composition, Repeat and cross-view translation RSD), the composition
diagnostics used in the text and Fig. 3, and Response per action (also written alone to the model's values_out, e.g.
solar_values.csv, when the registry sets it); and the per-model files per_video_response_out (per-video Response, as
per_video_response.csv.gz) and gate_sensitivity_out (Response and Hold with both thresholds scaled, as gate_sensitivity.csv).
"""
import pandas as pd

from compute_metrics import respond, model_values, rsd_tables
import registry as reg


def model_rows(m):
    A, R = reg.load_model(m)
    resp = respond(A)
    T = model_values(A, R, resp)
    U = reg.read_csv(m, 'units')
    T = T.join(rsd_tables(U))
    rb = A.assign(resp=resp).groupby(['model_key', 'action']).apply(lambda g: 100 * (g.weight * g.resp).sum() / g.weight.sum()).unstack()
    T = T.join(rb.add_prefix('response_'))
    sens = []
    for f in (0.25, 0.5, 1.0, 1.5, 2.0):
        Tf = model_values(A, R, respond(A, 2.0 * f, 2.0 * f))
        sens.append(dict(model_key=m.key, factor=f, response=Tf.loc[m.key, 'response'], hold=Tf.loc[m.key, 'hold']))
    return T, A.assign(resp=resp), pd.DataFrame(sens)


def main():
    reg.OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in reg.MODELS:
        T, V, G = model_rows(m)
        assert list(T.index) == [m.key], T.index
        if m.values_out:
            T.to_csv(reg.OUT / m.values_out)
        V.to_csv(m.out('per_video_response_out'), index=False)
        G.to_csv(m.out('gate_sensitivity_out'), index=False)
        rows.append(T)
        print(T.T.round(2).to_string())
    (pd.concat(rows) if rows else pd.DataFrame()).to_csv(reg.OUT / 'extra_values.csv')


if __name__ == '__main__':
    main()
