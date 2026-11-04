# -*- coding: utf-8 -*-
"""Compare a rerun against the shipped per-run results, arm by arm.

    python analysis/compare_rerun.py runs_out results/raw

Reruns are not expected to reproduce a number bit for bit: cuDNN kernel choice, GPU
model and library version all move the last decimal, and the trainer records `best_acc`
rounded to two decimals. What this prints is the size of the difference, so a claim like
"within 0.3 points" can be checked rather than believed.

Exit code is 0 when every shared arm is within the tolerance (default 0.5 points of
top-1) and 1 otherwise.
"""
import json
import os
import re
import sys

SEED = re.compile(r'_s(\d+)$')
TOL = float(os.environ.get('TOL', '0.5'))


def collect(root):
    out = {}
    if not os.path.isdir(root):
        return out
    for d in sorted(os.listdir(root)):
        dd = os.path.join(root, d)
        if not os.path.isdir(dd):
            continue
        for run in sorted(os.listdir(dd)):
            p = os.path.join(dd, run, 'result.json')
            m = SEED.search(run)
            if os.path.exists(p) and m:
                acc = json.load(open(p)).get('best_acc')
                if acc is not None:
                    out[(d, run)] = float(acc)
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    new, old = collect(sys.argv[1]), collect(sys.argv[2])
    shared = sorted(set(new) & set(old))
    if not shared:
        print('no arm appears in both trees; nothing to compare')
        return 2

    print('%-34s %8s %8s %8s' % ('arm', 'shipped', 'rerun', 'delta'))
    print('-' * 62)
    worst, bad = (None, 0.0), 0
    for k in shared:
        delta = new[k] - old[k]
        if abs(delta) > abs(worst[1]):
            worst = (k, delta)
        if abs(delta) > TOL:
            bad += 1
        print('%-34s %8.2f %8.2f %+8.2f%s'
              % (k[1], old[k], new[k], delta, '   <- outside tolerance' if abs(delta) > TOL else ''))

    mean = sum(new[k] - old[k] for k in shared) / len(shared)
    print('-' * 62)
    print('%d arms compared, mean delta %+.2f, largest %s %+.2f, %d outside %+.2f'
          % (len(shared), mean, worst[0][1], worst[1], bad, TOL))
    print('only %d of %d shipped arms were rerun' % (len(shared), len(old)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
