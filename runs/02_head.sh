#!/usr/bin/env bash
# Prior-prototype head (PPH) on the baseline trunk, twelve seeds.
#
# Paper: Table 6 row 2; Table 5 fourth cell.
# Protocol: SGD 0.01 / wd 1e-4 / 224 px / 150 epochs / cuDNN off, ResNet-18 unless stated.
# Seeds: 42 123 456 789 1024 2024 7 13 21 33 55 89
# Cost: 12 runs x ~25 min.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
{
  for s in 42 123 456 789 1024 2024 7 13 21 33 55 89; do
    echo "results_pph/pph_s${s}|${s}|--config r18 --head pph|"
  done
} | bash "$HERE/_launch.sh"
