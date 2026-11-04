#!/usr/bin/env bash
# Fail early and loudly: dataset present and unmodified, python has what it needs,
# at least one GPU is visible, and one forward pass runs.
#
#   bash runs/00_env_check.sh
#
# Exits non-zero on the first problem, so it can gate a sweep.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
PY=${PY:-python3}
if [ -d "$REPO/data/tea8" ]; then
  DATA=${DATA:-$REPO/data/tea8}
else
  DATA=${DATA:-$REPO/../GreenTea-8-1.0.0/tea8}   # dataset released as a separate folder
fi

echo "== dataset =="
if [ ! -d "$DATA/train" ]; then
  echo "missing $DATA/train -- point DATA= at the dataset root"; exit 1
fi
n_train=$(find "$DATA/train" -type f | wc -l)
n_val=$(find "$DATA/val" -type f | wc -l)
echo "train $n_train  val $n_val   (expected 1708 and 296)"
[ "$n_train" -eq 1708 ] || { echo "training set is not the one the paper used"; exit 1; }
[ "$n_val" -eq 296 ]   || { echo "validation set is not the one the paper used"; exit 1; }

if [ -f "$REPO/data/SHA256SUMS" ]; then
  echo "verifying checksums (2004 files, ~500 MB, about 10 s)"
  ( cd "$REPO" && sha256sum -c data/SHA256SUMS --quiet ) && echo "checksums ok"
else
  echo "no data/SHA256SUMS; skipping the file-level check"
fi

echo "== python =="
$PY - <<'EOF'
import sys
print('python', sys.version.split()[0])
import torch, torchvision, numpy, PIL
print('torch', torch.__version__, '| torchvision', torchvision.__version__)
print('numpy', numpy.__version__, '| pillow', PIL.__version__)
try:
    import scipy; print('scipy', scipy.__version__)
except ImportError:
    print('scipy MISSING -- analysis/ needs it, training does not')
try:
    import timm; print('timm', timm.__version__)
except ImportError:
    print('timm missing -- only needed by src/train_modern.py')
EOF

echo "== gpu =="
$PY -c "import torch; n=torch.cuda.device_count(); print('cuda', torch.version.cuda, '| devices', n); assert n>0, 'no GPU visible'" \
  || { echo "training needs a GPU; analysis/ does not"; exit 1; }

echo "== one forward pass =="
$PY - <<EOF
import sys, os
sys.path.insert(0, os.path.join("$REPO", "src"))
import torch
from models.cpc_resnet import CPCResNet, BasicBlock
m = CPCResNet(block=BasicBlock, layers=[2,2,2,2], num_classes=8,
              cpc_stages=[0,0,True,True], cpc_pos="both")
x = torch.randn(2, 3, 224, 224)
try:
    m = m.cuda(); x = x.cuda()
except AssertionError:
    pass
y = m(x)
print('logits', tuple(y.shape), '| params %.3f M' % (sum(p.numel() for p in m.parameters()) / 1e6))
assert y.shape == (2, 8)
EOF

echo
echo "environment looks usable. Next: bash runs/02_head.sh"
