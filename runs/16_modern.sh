#!/usr/bin/env bash
# Modern backbones on the same split, single seed — the ceiling check of Section 10.
#
# Paper: Section 10 ("a separate regime"). No statistics: one run per setting.
# Protocol: timm models at 224 px, 150 epochs, cuDNN on, seed 42.
# Cost: 24 runs x ~35 min.
#
# These are not comparable with the from-scratch ResNet-18 arms: different architecture,
# different optimiser defaults, one seed. They answer one question only — how much room
# is left above 95 on this split when the trunk is not trained from scratch.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
PY=${PY:-python3}
DATA=${DATA:-$REPO/data/tea8}
OUT=${OUT:-$REPO/runs_out}
GPU=${GPU:-0}

for arch in convnextv2_tiny davit_tiny mambaout_tiny repvit_m1_1; do
  for lr in 1e-4 5e-4 1e-3; do
    for wd in 0.05 0.1; do
      tag="$OUT/results_adapt/${arch}_lr${lr}_wd${wd}"
      [ -f "$tag/result.json" ] && { echo "skip  $arch lr=$lr wd=$wd"; continue; }
      mkdir -p "$tag"
      echo "run   $arch lr=$lr wd=$wd"
      CUDA_VISIBLE_DEVICES=$GPU $PY "$REPO/src/train_modern.py" \
        --model "$arch" --data_dir "$DATA" --output_dir "$tag" \
        --lr "$lr" --wd "$wd" --epochs 150 --seed 42 --workers 4 \
        > "$tag/train.log" 2>&1 || echo "FAILED $arch lr=$lr wd=$wd"
    done
  done
done
