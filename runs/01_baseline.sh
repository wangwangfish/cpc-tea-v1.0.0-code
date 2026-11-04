#!/usr/bin/env bash
# ResNet-18 baseline, twelve seeds.
#
# Paper: Table 6 row 1; reference for every paired contrast in the paper.
# Protocol: SGD 0.01 / wd 1e-4 / 224 px / 150 epochs / cuDNN off.
# Seeds: 42 123 456 789 1024 2024 (first batch) and 7 13 21 33 55 89 (power extension).
# Cost: 12 runs x ~25 min on one V100.
#
# The two batches landed in different directories on the server and were merged at
# analysis time, which is what makes n = 12 rather than n = 6. Reproduce them
# separately so the directory names match the shipped results.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

{
  for s in 42 123 456 789 1024 2024; do
    echo "results_paper/r18_sgd_s${s}|${s}||"
  done
  for s in 7 13 21 33 55 89; do
    echo "results_power/r18_s${s}|${s}||"
  done
} | bash "$HERE/_launch.sh"
