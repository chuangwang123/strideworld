#!/usr/bin/env bash
# Reproduce the fine-tuning table (Table J.1 of the appendix) from the released per-video measurements.
#
#   bash scoring/reproduce_finetune.sh DATA_RELEASE_DIR [OUT.csv]
#
# Every column is one run of the full benchmark (3,640 videos), scored with the functions behind Table 2.
# Compares the result with DATA_RELEASE_DIR/results/finetune_table.csv. CPU only, under a minute.
set -euo pipefail
REL=$(cd "$1" && pwd); OUT=${2:-$PWD/finetune_table.csv}
S=$(cd "$(dirname "$0")" && pwd)
python3 "$S/finetune_table.py" "$REL" "$OUT"
python3 - "$OUT" "$REL/results/finetune_table.csv" <<'PY'
import sys, pandas as pd, numpy as np
a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
num = [c for c in b.columns if b[c].dtype.kind == 'f']
ok = list(a.run) == list(b.run) and np.allclose(a[num].values, b[num].values, rtol=0, atol=1e-9, equal_nan=True)
d = np.nanmax(np.abs(a[num].values - b[num].values))
print(f'the fine-tuning table (Table J.1 of the appendix): {len(a)} runs; max |difference| to the released table {d:.2e} ->', 'REPRODUCED' if ok else 'DIFFERENT')
sys.exit(0 if ok else 1)
PY
