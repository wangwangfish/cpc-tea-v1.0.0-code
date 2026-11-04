#!/usr/bin/env bash
# Two ingredients of the head, crossed: linear vs angular readout, zero vs log-prior bias.
#
# Paper: Table 5. The fourth cell is the head itself, produced by 02_head.sh.
# Protocol: SGD 0.01 / wd 1e-4 / 224 px / 150 epochs / cuDNN off, ResNet-18 unless stated.
# Seeds: 42 123 456 789 1024 2024
# Cost: 18 runs x ~25 min.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
{
  for h in lin0 linp cos0; do
    for s in 42 123 456 789 1024 2024; do
      echo "results_quad/${h}_s${s}|${s}|--config r18 --head ${h}|"
    done
  done
} | bash "$HERE/_launch.sh"
