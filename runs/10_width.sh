#!/usr/bin/env bash
# Capacity axis: half and double channel width, twelve seeds each.
#
# Paper: Table 2; Section 6.
# Protocol: SGD 0.01 / wd 1e-4 / 224 px / 150 epochs / cuDNN off, ResNet-18 unless stated.
# Seeds: 42 123 456 789 1024 2024 7 13 21 33 55 89
# Cost: 24 runs x ~25-55 min (the 2x trunk is slower).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
{
  for w in 0.5 2.0; do
    for s in 42 123 456 789 1024 2024 7 13 21 33 55 89; do
      echo "results_width/w${w}_s${s}|${s}|--config r18 --width_mult ${w}|"
    done
  done
} | bash "$HERE/_launch.sh"
