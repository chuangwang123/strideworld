# Scoring (per-video measurements -> tables)

All scripts read the data release and run on CPU.

Terminology: the paper’s rigid counterpart is the code’s rigid twin, and the paper’s Pace score is the code’s Hold score.

## Table 2

`reproduce_table2.sh DATA_RELEASE_DIR [WORK_DIR]` rebuilds Table 2 and checks it against `results/table2.csv`.

| Script | Output | Column of Table 2 |
|---|---|---|
| `assemble.py` | `WORK/{base,extra,<model>,sft,deform}` from `measurements/<model>/` | |
| `metrics_v3/compute_metrics.py`, `extra_model.py` | `main_table_values.csv`, per-model values | Resp., Pace score, Composition score, End err. |
| `metrics_v3/hold_ratio.py` | `hold_ratio.csv` | Pace ratio |
| `metrics_v3/deform_analysis.py` (`summary`) | `deform_summary.csv` | Rigidity offset |
| `rigid_twin_ratio.py` | `rigid_twin_ratio.csv` | vs. rigid |
| `figures_src/analysis/stride_alt.py` | `f2_ratio.csv` (three stride definitions) | marks of the pace ratio |
| `figures_src/analysis/f3_diagnostics.py` | `by_model.csv` | Swap gap |
| `figures_src/route_outcomes.py` | `F3_route_outcomes.csv` | Near, Stall |
| `figures_src/make_table2.py` | `tables/table2.csv`, `tables/table2_body.tex` | all |

MUSIQ comes from the scoring reports of the runs and is stored in `registry_release.json`. It is not part of any
benchmark score. `SW_FROZEN_ATOMIC` only feeds a consistency check inside `compute_metrics.py`.

## the fine-tuning table (Table J.1 of the appendix)

`reproduce_finetune.sh DATA_RELEASE_DIR [OUT.csv]` runs `finetune_table.py` and checks the result against
`results/finetune_table.csv`. Each column (Base, SFT, Released, for HY-World 1.5 and SANA-WM) is one run of the full
benchmark, scored with the functions behind Table 2.

## Appendix analyses

| Script | Data | What it prints |
|---|---|---|
| `seed_tables.py DATA` | `seeds/` | Composition score per seed, rank agreement between seeds, spread of the route ends across seeds against the end error |
| `perturbation_summary.py DATA OUT_DIR` | `perturbations/` | share of videos flagged by each test under controlled changes of an ideal rendering (`flag_rates.csv`), and the summary figure with the response of the Rigidity offset to known scene changes |
| `metrics_v3/bootstrap_table2.py`, `domain_table.py`, `response_profiles.py`, `pace_switch_scale.py`, `view_excess.py`, `envelope.py`, `checkpoint_scores.py`, `appendix_analyses.py`, `appendix_new_tables.py`, `extra_appendix.py` | assembled layout of `reproduce_table2.sh` | intervals, the split by rendered and real scenes, response per action, and the remaining appendix tables |

The scripts in `metrics_v3/` use the environment of `reproduce_table2.sh`: `SW_DATA` (assembled data root),
`SW_BASE_DIR`, `SW_SOLAR_DIR`, `SW_SFT_DIR`, `SW_DEFORM`, `SW_REGISTRY`, `SW_OUT`. Run them after
`reproduce_table2.sh` with the same values.
