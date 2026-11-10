# StrideWorld: Benchmarking Scene Rigidity and Action Consistency in Interactive Video World Models

Chuang Wang, Rui Nie, Jie Song, Junyuan Gao, Jiahe Song, Haitao Zhou, Boyu Zhu, Xing Li, Jiang Wu, Conghui He, Qian Yu, and Weijia Li

[Data](https://huggingface.co/datasets/strideworld-release/strideworld) ·
[Leaderboard](https://huggingface.co/spaces/strideworld-release/strideworld-leaderboard)

The arXiv link and BibTeX entry will be added after the preprint is posted.

StrideWorld tests an interactive video world model on three properties while actions drive the camera. Every video
starts from a start view with known depth and camera. We recover the camera of every generated frame against that
view and score the video against two references that are fixed before generation: the geometry of the start view, and
the stride of each action, measured on the model's own first steps.

| Test | Question | Columns of Table 2 |
|---|---|---|
| Rigidity | Does the scene keep its 3D structure as the camera moves? | Offset of held-out anchors from their rigid positions, and the same offset relative to a rigid counterpart along the same camera path |
| Pace | Does a held action keep the stride of its first steps? | Pace ratio after 16 steps, Pace score |
| Composition | Do the strides of single actions predict where a route ends? | Stall, End error, Swap gap, Near, Composition score |

The benchmark has 140 start views (104 views of 24 rendered scenes and 36 photographs of 6 ETH3D scenes) and 26
requests per view (6 single actions with 3 seeds each, and 8 routes that form 4 pairs). This gives 3,640 videos per
model and 54,600 videos for the 15 models of the paper.

The full generated videos are not included in the release. The leaderboard includes eight annotated short clips
as examples (`leaderboard/media/`).

## Layout

```
manifests/requests.jsonl   the 3,640 requests (view, seed, route, action sequence), the same for every model
adapters/                  request -> model input for each of the 15 models, and the settings each model was run with
pipeline/                  generated video -> per-video measurements (camera recovery, anchors, strides, routes)
scoring/                   per-video measurements -> the tables of the paper
leaderboard/               the leaderboard page and the format of a submission
```

## Download scope

To reproduce the tables, download only `measurements/`, `results/`, `seeds/` and `perturbations/` from the data release (about 300 MB). The `start_views/` directory is about 6.6 GB and is needed only when running the measurement pipeline on generated videos.

## Reproduce the tables (CPU, a few minutes)

```bash
pip install -r requirements.txt
bash scoring/reproduce_table2.sh /path/to/strideworld-data ./work     # Table 2, all 15 models
bash scoring/reproduce_finetune.sh /path/to/strideworld-data            # the fine-tuning table (Table J.1 of the appendix)
python3 scoring/seed_tables.py /path/to/strideworld-data              # further seeds (appendix)
python3 scoring/perturbation_summary.py /path/to/strideworld-data ./work/perturbations   # controlled changes (appendix)
```

The first two scripts end with a comparison against `results/table2.csv` and `results/finetune_table.csv` of the data
release. Both scripts print `REPRODUCED` when differences are within the absolute tolerance of `1e-9`
(with zero relative tolerance). Small floating-point rounding differences can occur across platforms and library versions.

`reproduce_table2.sh` assembles the per-model measurement files (`scoring/assemble.py`) and runs the scoring code:
`metrics_v3/compute_metrics.py` (response, Pace score, Composition score), `hold_ratio.py` (pace ratio),
`deform_analysis.py` (Rigidity offset), `rigid_twin_ratio.py` (offset relative to the rigid rendering),
`figures_src/analysis` (stride definitions, swap gap) and `figures_src/route_outcomes.py` (Near, Stall). The other
scripts in `scoring/metrics_v3` compute the appendix analyses from the same assembled layout.

## Measurement pipeline (GPU)

`pipeline/` holds the code that turned the generated videos into the per-video measurements (see
`pipeline/README.md`).

1. **measure**: CoTracker3 tracks the anchors of the start view through the video. The camera of every frame is
   recovered from the pose anchors (EPnP with RANSAC, then nonlinear refinement) against the depth of the start view.
   Held-out anchors give the Rigidity residuals.
2. **common**, **dualgate**: one protocol for all models, scored at the pose-quality gate used in the paper.
3. **report**: strides from the first four steps, route and pair evaluation against the strides, per-video export.

## Evaluate a new model

1. Generate one video per line of `manifests/requests.jsonl` from its start view, executing the actions in order,
   and record the frame index at the end of every action unit. The shared rules are in `adapters/README.md`, and
   `adapters/` shows how each of the 15 models was driven.
2. Run the measurement pipeline on the videos.
3. Add the model to a registry (`scoring/registry_release.json` shows the format), run the scoring, and build its
   leaderboard row with `leaderboard/build_leaderboard.py`.

## Licence

Code: Apache License 2.0 (`LICENSE`). Data: see the data release. Third-party models and trackers keep their own
licences.
