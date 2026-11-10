# Measurement pipeline (video -> per-video measurements)

The code is kept in the states it was run with for the paper. `pipeline/measure` is the base; the later stages are
small patches on it, so the differences between stages stay visible:

| stage | code | what it does | entry point |
|---|---|---|---|
| 1 measure | `measure/` | CoTracker3 tracks the start view's anchors; per-frame camera from the pose anchors (EPnP + RANSAC, refinement) against the start-view depth; first-frame check (L1 ≤ 32), pose gates (inlier ratio, anchor coverage), SE(2) observations at every step boundary, held-out-anchor residuals; task tables | `python -m metrics.run_full_evaluation` |
| 2 common | `measure/` + `rescore/patches/common_protocol.patch` | one relaxed protocol for all models: pose inlier ratio and anchor coverage ≥ 0.1, motion floor 0 (k = 4/12/16 and the first-frame gate unchanged) | `rescore/common_protocol_rescore.py --config configs/common.json` |
| 3 dualgate | `measure/` + `rescore/patches/dual_threshold.patch` | re-scores the same pose artifacts at two pose-quality gates; the paper uses the 0.1 gate (`gated_010`) | `rescore/dual_threshold_rescore.py --config configs/dualgate.json` |
| 4 report | `measure/` + `report/patches/report.patch` + `report/metrics/` | stride (unit) horizons k = 1..4, pair evaluation from the strides, per-model report | `report/build_reports.py --config configs/report.json --model <key>` |

```bash
bash pipeline/make_stages.sh            # writes pipeline/build/{measure,common,dualgate,report}
cd pipeline/build/measure
python -m metrics.run_full_evaluation --data-root <index.jsonl> --model <key> --output-root <out> \
    --gpus 0,1,2,3 --workers-per-gpu 1 --anchor-workers 4 \
    --checkpoint checkpoints/cotracker3/scaled_offline.pth --cotracker-root third_party/co-tracker \
    --first-frame-max-l1 32.0 --min-pose-inlier-ratio 0.30 --min-pose-anchor-coverage 0.30 \
    --normalized-pass-tau 0.50 --bootstrap-samples 2000
```

The index (`--data-root`) has one JSON line per generated video: the request fields of `manifests/requests.jsonl`
(`case_uid` = `view`, `seed`, `route_id`, `actions`), the video path, and the frame index at the end of every step
(`timing.unit_boundary_frames`). The configs in `configs/` use `<work>`/`<model>` placeholders for paths;
`configs/report.json` pins the SHA-256 of the report-stage code (`report/build_reports.py` checks it).

The report stage writes the per-video tables that the data release ships per model (`scores.json.gz`,
`main_per_video.csv`, `unit_estimates.csv`, `initial_view_depths.csv`, `observations.csv.gz`; formats in the data card),
and `scoring/` turns them into the paper's tables.
