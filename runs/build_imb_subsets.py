"""Build the imbalance ladder with a FIXED total, so imbalance is the only variable.

The previous version let the total shrink as the imbalance grew (1687 -> 1397 -> 1285),
which left baseline accuracy falling from 94.5 to 88.6 for two confounded reasons. Here
every rung holds exactly 1200 training images; only the shape of the class distribution
changes. Validation is untouched and shared, so all rungs stay comparable.

    w_i = 1 - (1 - 1/r) * i/(K-1)     ranks sorted by original count
    n_i = round(T * w_i / sum(w))      capped at the number of images available
"""
import os, shutil, json
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'data', 'tea8')
DST = os.path.join(REPO, 'data', 'imb2')
RATIOS = [2, 5, 10]
TOTAL = 1200
SEED = 0

classes = sorted(d for d in os.listdir(os.path.join(SRC, 'train'))
                 if os.path.isdir(os.path.join(SRC, 'train', d)))
counts = {c: len(os.listdir(os.path.join(SRC, 'train', c))) for c in classes}
order = sorted(classes, key=lambda c: -counts[c])
print('source:', {c: counts[c] for c in order})

rng = np.random.RandomState(SEED)
manifest = {}
for r in RATIOS:
    raw = np.array([1.0 - (1.0 - 1.0 / r) * (i / (len(order) - 1))
                    for i in range(len(order))])
    raw = raw / raw.sum()
    want = np.round(TOTAL * raw).astype(int)
    # cap at availability, then top the remainder up from classes with spare images
    take = np.minimum(want, np.array([counts[c] for c in order]))
    deficit = TOTAL - take.sum()
    if deficit > 0:
        spare = np.array([counts[c] for c in order]) - take
        for i in np.argsort(-spare):
            if deficit <= 0:
                break
            add = min(spare[i], deficit)
            take[i] += add
            deficit -= add
    root = os.path.join(DST, 'r%d' % r)
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, 'train'))
    kept = {}
    for c, k in zip(order, take):
        src_dir = os.path.join(SRC, 'train', c)
        files = sorted(os.listdir(src_dir))
        k = int(min(k, len(files)))
        sel = rng.choice(files, size=k, replace=False) if k < len(files) else files
        dst_dir = os.path.join(root, 'train', c)
        os.makedirs(dst_dir)
        for f in sel:
            os.symlink(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        kept[c] = k
    os.symlink(os.path.join(SRC, 'val'), os.path.join(root, 'val'))
    imb = max(kept.values()) / min(kept.values())
    manifest['r%d' % r] = dict(target=r, per_class=kept, total=sum(kept.values()),
                               realised=round(imb, 2))
    print('r%-2d total=%4d realised=%.2f  %s'
          % (r, sum(kept.values()), imb, ' '.join('%s=%d' % (c[:6], kept[c]) for c in order)))

json.dump(manifest, open(os.path.join(DST, 'manifest.json'), 'w'), indent=1)
print('wrote manifest; totals:', [manifest['r%d' % r]['total'] for r in RATIOS])
