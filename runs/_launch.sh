#!/usr/bin/env bash
# Generic distributor: reads "tag|seed|flags|data" lines on stdin and keeps up to JOBS
# runs in flight across the GPUs listed in GPU_LIST. A task whose result.json already
# exists is skipped, so the same call resumes an interrupted sweep.
#
#   tag    output directory, relative to $OUT, e.g. results_pph/pph_s42
#   seed   integer seed passed to --seed
#   flags  extra flags for src/train_lpr.py (may be empty, but the field must be there)
#   data   optional training root; defaults to $DATA
#
# Environment:
#   PY        python with torch (default: python3)
#   DATA      training root (default: <repo>/data/tea8)
#   OUT       output root (default: <repo>/runs_out)
#   JOBS      parallel runs (default: 8)
#   GPU_LIST  comma-separated device ids (default: 0,1,2,3,4,5,6,7)
#   EPOCHS    default 150
#   BASE      flags prepended to every task
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
PY=${PY:-python3}
if [ -d "$REPO/data/tea8" ]; then
  DATA=${DATA:-$REPO/data/tea8}
else
  DATA=${DATA:-$REPO/../GreenTea-8-1.0.0/tea8}   # dataset released as a separate folder
fi
OUT=${OUT:-$REPO/runs_out}
JOBS=${JOBS:-8}
EPOCHS=${EPOCHS:-150}
WORKERS=${WORKERS:-6}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
BASE=${BASE:---optimizer sgd --lr 0.01 --wd 1e-4 --disable_cudnn}

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
mkdir -p "$OUT"

n=0
while IFS='|' read -r tag seed flags data; do
  [ -z "${tag:-}" ] && continue
  d="$OUT/$tag"
  if [ -f "$d/result.json" ]; then
    echo "skip  $tag (already has result.json)"
    n=$((n + 1))
    continue
  fi
  g=${GPUS[$((n % ${#GPUS[@]}))]}
  root=${data:-$DATA}
  mkdir -p "$d"
  (
    CUDA_VISIBLE_DEVICES=$g $PY "$REPO/src/train_lpr.py" \
      --data_dir "$root" --output_dir "$d" --seed "$seed" \
      --epochs "$EPOCHS" --workers "$WORKERS" \
      $BASE $flags > "$d/train.log" 2>&1
    echo "end   $tag rc=$? (gpu $g)"
  ) &
  n=$((n + 1))
  if [ $((n % JOBS)) -eq 0 ]; then wait; fi
done
wait
echo "launcher finished after $n tasks; results under $OUT"
