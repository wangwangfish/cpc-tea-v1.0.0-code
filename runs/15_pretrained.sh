#!/usr/bin/env bash
# ImageNet-initialised ResNet-18, thirty epochs.
#
# Paper: Section 10, the separate regime.
# Protocol: SGD 0.001 / wd 1e-4 / 224 px / 30 epochs / ImageNet init.
# Seeds: 42 123 456.
# Cost: 3 runs x ~8 min.
#
# Same script, same data, same evaluation as every other arm; only the initialisation
# and the schedule differ, which is what fine-tuning means. BASE replaces the launcher
# default rather than adding to it, which is why lr appears here and not in the task list.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

{
  for s in 42 123 456; do
    echo "results_ft244/ft_s${s}|${s}||"
  done
} | BASE="--optimizer sgd --lr 0.001 --wd 1e-4 --bs 32 --disable_cudnn --init_imagenet" \
    EPOCHS=30 \
    bash "$HERE/_launch.sh"
