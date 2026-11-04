# -*- coding: utf-8 -*-
"""Audit every derived number in the manuscript against the raw per-seed results.

For each claim the script recomputes the quantity from the JSON that the figures are built
from, then compares with the value printed in the LaTeX. Differences of 0.01 are rounding;
anything larger is a finding.
"""
import json
import math
import numpy as np

import os
DERIVED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "derived")


def _p(name):
    return os.path.join(DERIVED, name)

D = json.load(open(_p('figdata_paper.json')))
WP = json.load(open(_p('width_pairs.json')))
PC = json.load(open(_p('panel_contrasts.json')))

T = {2: 2.920, 5: 2.015, 11: 1.796}
rows = []


def check(name, got, claim, tol=0.011):
    ok = abs(got - claim) <= tol
    rows.append((ok, name, got, claim))
    print('%-58s computed %8.3f   claimed %8.3f   %s'
          % (name, got, claim, 'ok' if ok else '<<< MISMATCH'))


def mean(arm, key):
    return float(np.mean(list(D[key][arm].values())))


bT, bM, bW = mean('baseline', 'top1'), mean('baseline', 'macro'), mean('baseline', 'worst')
check('baseline top-1 (12 seeds)', bT, 94.59)
check('baseline macro (12 seeds)', bM, 93.39)
check('baseline worst (12 seeds)', bW, 59.23)
check('baseline sd of top-1 across seeds',
      float(np.std(list(D['top1']['baseline'].values()), ddof=1)), 0.41)

six = ['s42', 's123', 's456', 's789', 's1024', 's2024']
bT6 = float(np.mean([D['top1']['baseline'][s] for s in six]))
bM6 = float(np.mean([D['macro']['baseline'][s] for s in six]))
bW6 = float(np.mean([D['worst']['baseline'][s] for s in six]))
check('baseline top-1 (6 seeds)', bT6, 94.54)
check('baseline macro (6 seeds)', bM6, 93.30)
check('baseline worst (6 seeds)', bW6, 58.93)

# arms
for arm, n, t1, ma, wo in [('pph', 12, 95.47, 94.56, 67.86),
                           ('combo', 6, 95.95, 95.20, 71.43)]:
    check('%s top-1 level' % arm, mean(arm, 'top1'), t1)
    check('%s macro level' % arm, mean(arm, 'macro'), ma)
    check('%s worst level' % arm, mean(arm, 'worst'), wo)

# paired gains
def paired(arm, key, seeds):
    return float(np.mean([D[key][arm][s] - D[key]['baseline'][s] for s in seeds]))


pph_seeds = list(D['top1']['pph'].keys())
check('head top-1 paired gain', paired('pph', 'top1', pph_seeds), 0.87)
check('head macro paired gain', paired('pph', 'macro', pph_seeds), 1.17)
check('head worst paired gain', paired('pph', 'worst', pph_seeds), 8.63)
d = np.array([D['top1']['pph'][s] - D['top1']['baseline'][s] for s in pph_seeds])
check('head top-1 se', float(d.std(ddof=1) / math.sqrt(len(d))), 0.25)
check('head top-1 t', float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))), 3.48)
print('   head per-seed signs: %d positive, %d zero, %d negative'
      % ((d > 1e-9).sum(), (np.abs(d) < 1e-9).sum(), (d < -1e-9).sum()))

cs = list(D['top1']['combo'].keys())
check('combination top-1 paired gain', paired('combo', 'top1', cs), 1.41)
check('combination macro paired gain', paired('combo', 'macro', cs), 1.90)
check('combination worst paired gain', paired('combo', 'worst', cs), 12.50)

# width axis
for k, lvl, dt, sd, se, ub in [('0.5', 92.62, -1.97, 0.92, 0.27, -1.49),
                               ('2.0', 94.26, -0.34, 0.78, 0.22, +0.06)]:
    w = WP[k]
    n = w['n']
    dd = np.array(w['d'])
    check('width %s level' % k, w['level'], lvl)
    check('width %s delta' % k, dd.mean(), dt)
    check('width %s s_d' % k, dd.std(ddof=1), sd)
    check('width %s se' % k, dd.std(ddof=1) / math.sqrt(n), se)
    check('width %s UB' % k, dd.mean() + T[n - 1] * dd.std(ddof=1) / math.sqrt(n), ub)
    print('   width %s signs: %d negative, %d positive; t = %.2f'
          % (k, (dd < 0).sum(), (dd > 0).sum(), dd.mean() / (dd.std(ddof=1) / math.sqrt(n))))

# panel A and B bounds inside the tables
for name, lvl, dt, sd, se, ub, _ in PC['panelB']:
    check('panelB %-24s UB' % name, dt + T[2] * se, ub)
for name, dt, sd, se, ub in PC['panelA']:
    check('panelA %-24s UB' % name, dt + T[2] * se, ub)

# resolutions
check('one image on 296 = top-1 step', 100.0 / 296, 0.34)
check('one image on 28 = worst step', 100.0 / 28, 3.57)
check('top-1 step averaged over 12 seeds', 100.0 / 296 / 12, 0.03)
check('worst step averaged over 12 seeds', 100.0 / 28 / 12, 0.30)
check('parameters of the head', 8 * 512 + 8 + 1, 4105, tol=0)
check('parameters of the linear layer', 8 * 512 + 8, 4104, tol=0)
check('imbalance ratio 294/134', 294 / 134, 2.19)
check('width ratio 44.68/11.18', 44.68 / 11.18, 4.00, tol=0.01)
check('half width 11.18/2.801', 11.18 / 2.801, 3.99, tol=0.01)
check('worst-class gain in images out of 28', 8.63 * 28 / 100, 2.42, tol=0.05)
check('sampler fraction of the head gain', 6.55 / 8.63, 0.76, tol=0.01)
check('pretrained minus scratch (best pair)', 97.98 - 77.44, 20.54, tol=0.05)
check('selection bias top-1', 95.05 - 94.82, 0.23)
check('selection bias macro', 95.06 - 94.64, 0.42)
check('selection bias worst', 88.83 - 79.24, 9.59)
check('sign test, 12 of 12, two-sided', 2 * 0.5 ** 12, 0.0005, tol=0.0001)

bad = [r for r in rows if not r[0]]
print('\n%d checks, %d mismatches' % (len(rows), len(bad)))
for r in bad:
    print('  MISMATCH', r[1], r[2], r[3])
