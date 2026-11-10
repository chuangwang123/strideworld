#!/usr/bin/env bash
# Materialise the four code states the paper's measurements were produced with (see pipeline/README.md):
#   build/measure   stage 1, video -> camera, anchors, task tables   (= pipeline/measure)
#   build/common    stage 2, common relaxed protocol                  (measure + rescore/patches/common_protocol.patch)
#   build/dualgate  stage 3, dual-threshold re-scoring                (measure + rescore/patches/dual_threshold.patch)
#   build/report    stage 4, unit horizons, pair evaluation, report   (measure + report/patches/report.patch + report/metrics)
set -euo pipefail
P=$(cd "$(dirname "$0")" && pwd); B=${1:-$P/build}
rm -rf "$B"; mkdir -p "$B"
for s in measure common dualgate report; do cp -r "$P/measure" "$B/$s"; done
(cd "$B/common" && patch -s -p1 < "$P/rescore/patches/common_protocol.patch")
(cd "$B/dualgate" && patch -s -p1 < "$P/rescore/patches/dual_threshold.patch")
(cd "$B/report" && patch -s -p1 < "$P/report/patches/report.patch" && cp -r "$P/report/metrics/." metrics/)
echo "stages in $B: $(ls "$B" | tr '\n' ' ')"
