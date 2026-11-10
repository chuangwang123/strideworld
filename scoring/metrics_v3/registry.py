"""Registry of the models scored after the thirteen benchmark models (registry_release.json) and the input/output locations.

Every script loops over MODELS (registry order) instead of naming a model; their rows follow the thirteen in all tables.
Environment overrides:
  SW_DATA      data root (required): the layout written by scoring/assemble.py
  SW_REGISTRY  registry file (default scoring/registry_release.json)
  SW_OUT       outputs directory (default outputs/ next to this file)
Registry: base_dir and sft_dir (the 13-model and SFT exports) and the ordered list `models`; per model
  key                      row key in every output (index of outputs/extra_values.csv)
  name                     display name in the LaTeX rows
  source_key               optional: model key written inside the export files when it differs from key (rows relabelled)
  dir                      export directory, relative to the data root; the file names below are relative to it
  scores, depths, units    all_scores json, initial_view_depths csv, unit_estimates csv
  obs                      glob of its observation tables (every row must belong to source_key)
  main_per_video           main_per_video csv (measurable cohort in the response-profile discussion of the appendix)
  frames                   frames per 16 steps (the response-profile discussion of the appendix)
  external                 values not computed here: scene_stability, musiq, aesthetic (Table 2 and auxiliary metric-correlation analyses), source
  per_video_response_out, per_checkpoint_out, gate_sensitivity_out   per-model output names (in the outputs directory)
  values_out               optional one-row copy of its extra_values.csv row (solar_values.csv, backward compatibility)

python3 registry.py           check the registry and every input file, list the models
python3 registry.py --shell   DATA/BASE/SFT/OUT assignments for run_all.sh
python3 registry.py --md5     md5 lines (inputs.md5 format) of the registered models' input files
"""
import argparse, glob, hashlib, json, os, shlex
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REGISTRY = Path(os.environ.get('SW_REGISTRY') or HERE.parent / 'registry_release.json')
DATA = Path(os.environ['SW_DATA'])
OUT = Path(os.environ.get('SW_OUT') or HERE / 'outputs')
_REG = json.loads(REGISTRY.read_text())
BASE = DATA / _REG['base_dir']
SFT = DATA / _REG['sft_dir']
BASE_EXTERNAL = _REG.get('base_external', {})   # image-quality values of base models kept in the registry
INPUTS = ('scores', 'depths', 'units', 'main_per_video')
OUTPUTS = ('per_video_response_out', 'per_checkpoint_out', 'gate_sensitivity_out')
EXTERNAL = ('scene_stability', 'musiq', 'aesthetic')
SHARED_OUTPUTS = {'main_table_values.csv', 'per_video_response.csv.gz', 'response_by_action.csv', 'gate_sensitivity.csv',
                  'sft_values.csv', 'extra_values.csv', 'hold_ratio.csv', 'reference_scores.csv', 'length_cohort.csv',
                  'paired_intervals.csv', 'domain_values.csv', 'domain_table_rows.tex', 'response_profiles_profiles.csv',
                  'response_profiles_fits.csv', 'pace_switch_scale.csv', 'view_excess.csv', 'bootstrap_table2.csv',
                  'appendix_new_tables.tex', 'measurable_envelope.csv', 'measurable_by_model.csv'}


class Model:
    """One registered model; attributes are the registry fields, plus path (its export directory)."""

    def __init__(self, e):
        need = ('key', 'name', 'dir', 'obs', 'frames', 'external') + INPUTS + OUTPUTS
        miss = [f for f in need if f not in e] + [f'external.{f}' for f in EXTERNAL if f not in e.get('external', {})]
        unknown = [f for f in e if f not in need + ('source_key', 'values_out') and not f.startswith('_')]
        if miss or unknown:
            raise ValueError(f'{REGISTRY}: entry {e.get("key")!r} lacks {miss}, unknown fields {unknown}')
        self.__dict__.update(e)
        self.source_key = e.get('source_key', e['key'])
        self.values_out = e.get('values_out')
        self.frames = int(e['frames'])
        self.path = DATA / e['dir']

    def file(self, field):
        return self.path / getattr(self, field)

    def out(self, field):
        return OUT / getattr(self, field)

    def obs_files(self):
        fs = sorted(glob.glob(str(self.path / self.obs)))
        if not fs:
            raise FileNotFoundError(f'{self.key}: no observation table matches {self.path / self.obs}')
        return fs

    def outputs(self):
        return [getattr(self, f) for f in OUTPUTS] + ([self.values_out] if self.values_out else [])


MODELS = [Model(e) for e in _REG['models']]
KEYS = [m.key for m in MODELS]


def names():
    return {m.key: m.name for m in MODELS}


def relabel(df, m, col='model_key'):
    """Check that a table of m's export holds only m's rows; rename source_key to key."""
    keys = set(df[col].unique())
    if keys != {m.source_key}:
        raise ValueError(f'{m.key}: expected only {m.source_key!r} in {col}, found {sorted(keys)}')
    return df if m.source_key == m.key else df.assign(**{col: m.key})


def read_csv(m, field):
    return relabel(pd.read_csv(m.file(field)), m)


def read_obs(m):
    return relabel(pd.concat([pd.read_csv(f) for f in m.obs_files()]), m)


def records(m):
    """all_scores records of m (single-action videos and routes), relabelled."""
    d = json.load(open(m.file('scores')))
    keys = {r['model'] for r in d}
    if keys != {m.source_key}:
        raise ValueError(f'{m.key}: expected only {m.source_key!r} in {m.scores}, found {sorted(keys)}')
    if m.source_key != m.key:
        for r in d:
            r['model'] = m.key
    return d


def load_model(m):
    """compute_metrics.load on m's export (single-action table A, routes R), relabelled."""
    from compute_metrics import load
    A, R = load(m.path, m.scores, m.depths, m.obs)
    return relabel(A, m), relabel(R, m)


def check():
    """Registry consistency and, for every model, existence and model keys of its input files."""
    from compute_metrics import ORDER
    dup = {k for k in KEYS if KEYS.count(k) > 1} | (set(KEYS) & set(ORDER))
    if dup:
        raise ValueError(f'{REGISTRY}: keys repeated or among the thirteen: {sorted(dup)}')
    outs = [o for m in MODELS for o in m.outputs()]
    clash = {o for o in outs if outs.count(o) > 1} | (set(outs) & SHARED_OUTPUTS)
    if clash:
        raise ValueError(f'{REGISTRY}: output names repeated or shared: {sorted(clash)}')
    for p in (BASE, SFT):
        if not p.is_dir():
            raise FileNotFoundError(p)
    for m in MODELS:
        records(m)
        for f in ('depths', 'units', 'main_per_video'):
            relabel(pd.read_csv(m.file(f), usecols=['model_key']), m)
        for f in m.obs_files():
            relabel(pd.read_csv(f, usecols=['model_key']), m)
        print(f'{m.key:<16} {m.name:<14} {m.dir}: {m.scores}, {m.depths}, {m.units}, {m.main_per_video}, '
              f'{len(m.obs_files())} obs; frames {m.frames}; external {[m.external[x] for x in EXTERNAL]}')
    print(f'registry {REGISTRY}: {len(MODELS)} model(s) after the thirteen; data {DATA}; outputs {OUT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--shell', action='store_true'); ap.add_argument('--md5', action='store_true')
    a = ap.parse_args()
    if a.shell:
        for k, v in (('DATA', DATA), ('BASE', BASE), ('SFT', SFT), ('OUT', OUT)):
            print(f'{k}={shlex.quote(str(v))}')
    elif a.md5:
        for m in MODELS:
            for p in sorted([m.file(f) for f in INPUTS] + [Path(f) for f in m.obs_files()]):
                print(f'{hashlib.md5(p.read_bytes()).hexdigest()}  {p.relative_to(DATA)}')
    else:
        check()
