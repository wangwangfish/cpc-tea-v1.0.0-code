#!/usr/bin/env bash
# Aggregation axis: GeM, multi-scale fusion, and both, six seeds each.
#
# Paper: Table 3.
# Protocol: SGD 0.01 / wd 1e-4 / 224 px / 150 epochs / cuDNN off, ResNet-18 unless stated.
# Seeds: 42 123 456 789 1024 2024
# Cost: 18 runs x ~25 min.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
{
  for h in gem ms ms_gem; do
    for s in 42 123 456 789 1024 2024; do
      echo "results_head/${h}_s${s}|${s}|--config r18 --head ${h}|"
    done
  done
} | bash "$HERE/_launch.sh"
