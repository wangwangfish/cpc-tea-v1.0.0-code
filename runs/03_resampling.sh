#!/usr/bin/env bash
# Class-balanced resampling, six seeds.
#
# Paper: Table 6 row 3.
# Protocol: SGD 0.01 / wd 1e-4 / 224 px / 150 epochs / cuDNN off, ResNet-18 unless stated.
# Seeds: 42 123 456 789 1024 2024
# Cost: 6 runs x ~25 min.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
{
  for s in 42 123 456 789 1024 2024; do
    echo "results_balance/sampler_s${s}|${s}|--config r18 --cls_balance sampler|"
  done
} | bash "$HERE/_launch.sh"
