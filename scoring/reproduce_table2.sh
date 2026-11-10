#!/usr/bin/env bash
# Reproduce Table 2 of the paper (main results) from the released per-video measurements.
#
#   bash scoring/reproduce_table2.sh DATA_RELEASE_DIR [WORK_DIR]
#
# Writes WORK_DIR (default ./work): the assembled measurement layout, all intermediate outputs, and
# scoring/tables/table2.csv + table2_body.tex. Then compares table2.csv with DATA_RELEASE_DIR/results/table2.csv.
# Needs python3 with numpy, pandas, scipy (requirements.txt). CPU only, a few minutes.
set -euo pipefail
REL=$(cd "$1" && pwd); WORK=${2:-$PWD/work}; mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
S=$(cd "$(dirname "$0")" && pwd)
python3 "$S/assemble.py" "$REL" "$WORK"
export SW_DATA="$WORK" SW_BASE_DIR=base SW_SOLAR_DIR=extra SW_SFT_DIR=sft SW_DEFORM="$WORK/deform/" \
       SW_REGISTRY="$S/registry_release.json" SW_OUT="$WORK/outputs"
export SW_FROZEN_ATOMIC='evoke-turbo=74.3,worldcrafter-fast=80.5,matrix-game-2=74.3,matrix-game-3=77.3'   # Atomic printed by the scoring reports of these runs (check only)
mkdir -p "$SW_OUT" "$S/figures_src/data_f1" "$S/tables"
cd "$S/metrics_v3"
python3 registry.py
python3 compute_metrics.py --data "$WORK/base" --sft "$WORK/sft" --out "$SW_OUT"        # Resp., Pace score, Composition, ...
python3 extra_model.py                                                                  # SolarWM, HY-WorldPlay
python3 hold_ratio.py --obs "$WORK/base/obs/*.observations.csv" --units "$WORK/base/unit_estimates.csv" --extra \
    --out "$SW_OUT/hold_ratio.csv"                                                      # F2 pace ratio
python3 -c "import deform_analysis as d; d.summary()"                                   # F1 offset
python3 "$S/rigid_twin_ratio.py" "$REL" "$SW_OUT/rigid_twin_ratio.csv" > /dev/null                   # Rigidity vs. rigid twins
cp "$SW_OUT/deform_summary.csv" "$S/figures_src/data_f1/deform_summary.csv"
mkdir -p "$S/metrics_v3/outputs"; cp "$SW_OUT"/*.csv "$S/metrics_v3/outputs/"           # make_table2 reads metrics_v3/outputs
cd "$S/figures_src"
python3 analysis/stride_alt.py > /dev/null                                              # F2 stride check (pace flags)
python3 analysis/f3_diagnostics.py > /dev/null                                          # swap gap, stride scale
python3 route_outcomes.py > /dev/null                                                   # Near, Stall
python3 make_table2.py > /dev/null
python3 - "$S/tables/table2.csv" "$REL/results/table2.csv" <<'PY'
import sys, pandas as pd, numpy as np
a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
num = [c for c in b.columns if b[c].dtype.kind == 'f']
ok = list(a.model) == list(b.model) and np.allclose(a[num].values, b[num].values, rtol=0, atol=1e-9, equal_nan=True)
d = np.nanmax(np.abs(a[num].values - b[num].values))
print(f'Table 2: {len(a)} models; max |difference| to the released table {d:.2e} ->', 'REPRODUCED' if ok else 'DIFFERENT')
sys.exit(0 if ok else 1)
PY
