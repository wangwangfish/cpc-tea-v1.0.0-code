#!/usr/bin/env bash
# Density boundary: baseline vs head on three 1200-image training subsets.
#
# Paper: Section 9, where the correction collapses.
# Protocol: SGD 0.01 / wd 1e-4 / 224 px / 150 epochs / cuDNN off.
# Seeds: 42 123 456.
# Cost: 18 runs x ~20 min, plus the subset build.
#
# The subsets hold a fixed 1200 training images at realised imbalance 2.19 / 4.98 /
# 10.14, so imbalance is the only variable; validation is the shared full split.
# build_imb_subsets.py symlinks from data/tea8, so it costs no disk.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)

[ -d "$REPO/data/imb2/r2" ] || python3 "$HERE/build_imb_subsets.py"

{
  for r in 2 5 10; do
    for s in 42 123 456; do
      echo "results_imb2/base_r${r}_s${s}|${s}||$REPO/data/imb2/r${r}"
      echo "results_imb2/pph_r${r}_s${s}|${s}|--config r18 --head pph|$REPO/data/imb2/r${r}"
    done
  done
} | bash "$HERE/_launch.sh"
