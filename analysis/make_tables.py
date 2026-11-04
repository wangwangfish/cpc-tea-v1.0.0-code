# -*- coding: utf-8 -*-
"""Rebuild every table in the paper from the shipped per-run result files.

This script reads ``results/raw/`` only. It does not touch the files in
``results/derived/``, so a reader who distrusts those can still reproduce the numbers
that appear in the manuscript.

    python analysis/make_tables.py            # all panels
    python analysis/make_tables.py width      # one panel

Conventions, identical to the paper:
  * a level is the mean top-1 of that arm's runs;
  * a contrast is paired by seed against the arm's own reference, so only seeds present
    in both enter;
  * s_d is the standard deviation of the per-seed differences, se = s_d / sqrt(n);
  * UB = delta + t(0.95, n-1) * se, one-sided; UB_adj uses t(1 - 0.05/17, n-1) for the
    seventeen routes reported across the two panels.

Two details that are easy to get wrong, and that change the numbers:
  * the twelve-seed baseline is stored in two directories -- six seeds in
    ``results_paper`` and six in ``results_power`` -- because the second batch was a
    power extension. Merging them is what makes n = 12 rather than n = 6;
  * the operator panel has two references. Three routes were carried to six seeds and
    are paired against the six-seed C0 (94.82); the rest are paired on seeds
    42/123/456 against the three-seed C0 (95.27). Pairing everything against one of
    them silently shifts eleven deltas.

Levels are the two-decimal ``best_acc`` that the trainer writes into ``result.json``.
The manuscript's main table is recomputed from checkpoints as exact k/296 counts, which
can differ from these means by up to 0.005 per seed; ``docs/RESULTS.md`` says when each
convention applies.
"""
import json
import math
import os
import re
import sys

from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, 'results', 'raw')
ROUTES = 17
SEED = re.compile(r'_s(\d+)$')

# A reference can span several directories: the twelve-seed baseline does.
REFS = {
    'base12': [('results_paper', 'r18_sgd'), ('results_power', 'r18')],
    'c0_6':   [('results_mild', 'C0_v1_k0_l0.0_nodno')],
    'c0_3':   [('results_mild', 'C0_v1_k0_l0.0_nodno')],
}

# panel -> [(dir, prefix, reference name or None, label)]
PANELS = {
    'width': [
        ('results_width', 'w0.5', 'base12', r'$0.5\times$ width'),
        ('results_width', 'w2.0', 'base12', r'$2\times$ width'),
    ],
    'agg': [
        ('results_head', 'gem',    'base12', 'GeM pooling'),
        ('results_head', 'ms',     'base12', 'Multi-scale fusion'),
        ('results_head', 'ms_gem', 'base12', 'Both combined'),
    ],
    'head': [
        ('results_paper',   'r18_sgd',     'self:base12', 'Baseline, GAP + linear'),
        ('results_pph',     'pph',         'base12', 'Prior-prototype head'),
        ('results_balance', 'sampler',     'base12', 'Class-balanced resampling'),
        ('results_pph',     'pph_sampler', 'base12', 'Head $+$ resampling'),
    ],
    'operator': [
        ('results_mild',      'C0_v1_k0_l0.0_nodno', None, 'C0 reference, 8 CPC layers'),
        ('results_dual',      'M1',  'c0', 'Dual prototype, soft, delta 0.10'),
        ('results_dual',      'M1b', 'c0', 'Dual prototype, soft, delta 0.30'),
        ('results_dual',      'M2',  'c0', 'Dual prototype, max'),
        ('results_dual',      'M3c', 'c0', 'Dual prototype, mean (control)'),
        ('results_dual2',     'M4m', 'c0', 'Dual prototype, max, independent init'),
        ('results_dual2',     'M4s', 'c0', 'Dual prototype, soft, independent init'),
        ('results_cpcpos',    'A_conv2', 'c0', r'In-block placement, conv$_2$'),
        ('results_cpcpos',    'B_conv1', 'c0', r'In-block placement, conv$_1$'),
        ('results_cpcstages', 'C12', 'c0', '12 CPC layers'),
        ('results_cpcstages', 'C16', 'c0', '16 CPC layers'),
        ('results_mild', 'C1_soft_k0.01_l0.0_nodno', 'c0', 'Soft energy floor, kappa 0.01'),
        ('results_mild', 'C1_soft_k0.05_l0.0_nodno', 'c0', 'Soft energy floor, kappa 0.05'),
        ('results_mild', 'C2_mild_k0_l0.1_nodno',    'c0', 'Mild DC suppression, 0.10'),
        ('results_mild', 'C2_mild_k0_l0.3_nodno',    'c0', 'Mild DC suppression, 0.30'),
        ('results_mild', 'C3_v1_k0_l0.0_nodyes',     'c0', 'Prototype without weight decay'),
    ],
    'crossed': [
        ('results_quad', 'lin0', 'base6', 'Linear, zero bias'),
        ('results_quad', 'linp', 'base6', r'Linear, $\log \pi_c$ bias'),
        ('results_quad', 'cos0', 'base6', 'Angular, zero bias'),
        ('results_pph',  'pph',  'base6', r'Angular, $\log \pi_c$ bias (the head)'),
    ],
    'density': [
        ('results_imb2', 'base_r2',  None,      'Baseline, imbalance 2'),
        ('results_imb2', 'pph_r2',   'base_r2', 'Head, imbalance 2'),
        ('results_imb2', 'base_r5',  None,      'Baseline, imbalance 5'),
        ('results_imb2', 'pph_r5',   'base_r5', 'Head, imbalance 5'),
        ('results_imb2', 'base_r10', None,       'Baseline, imbalance 10'),
        ('results_imb2', 'pph_r10',  'base_r10', 'Head, imbalance 10'),
    ],
    'pretrained': [
        ('results_ft244', 'ft', 'base3', 'ImageNet init, 30 epochs'),
    ],
}
REFS['base3'] = [('results_paper', 'r18_sgd')]
REFS['base6'] = [('results_paper', 'r18_sgd')]
REFS['base_r2'] = [('results_imb2', 'base_r2')]
REFS['base_r5'] = [('results_imb2', 'base_r5')]
REFS['base_r10'] = [('results_imb2', 'base_r10')]


def load():
    """{(dir, prefix): {seed: best_acc}} for every run on disk."""
    arms = {}
    for d in sorted(os.listdir(RAW)):
        dd = os.path.join(RAW, d)
        if not os.path.isdir(dd):
            continue
        for run in sorted(os.listdir(dd)):
            p = os.path.join(dd, run, 'result.json')
            m = SEED.search(run)
            if not os.path.exists(p) or not m:
                continue
            acc = json.load(open(p)).get('best_acc')
            if acc is None:
                continue
            arms.setdefault((d, run[:m.start()]), {})[int(m.group(1))] = float(acc)
    return arms


def merged(arms, names):
    out = {}
    for d, prefix in names:
        out.update(arms.get((d, prefix), {}))
    return out


def c0_for(arms, n, mixed):
    """Six-seed C0 when the arm has six seeds, three-seed C0 when it has three.

    The reference is cut to the arm's seeds anyway, so passing both is harmless; this
    keeps the printed n honest about which reference was used.
    """
    ref = merged(arms, REFS['c0_6'])
    seeds = sorted(ref)
    if n >= 6:
        six = [s for s in seeds if s in (42, 123, 456, 789, 1024, 2024)]
        return {s: ref[s] for s in six}
    return {s: ref[s] for s in seeds if s in (42, 123, 456)}


def contrast(arm, ref):
    seeds = sorted(set(arm) & set(ref))
    if len(seeds) < 2:
        return None
    diffs = [arm[s] - ref[s] for s in seeds]
    n = len(diffs)
    delta = sum(diffs) / n
    s_d = statistics_stdev(diffs)
    se = s_d / math.sqrt(n)
    return {'n': n, 'delta': delta, 'sd': s_d, 'se': se,
            'ub': delta + stats.t.ppf(0.95, n - 1) * se,
            'ub_adj': delta + stats.t.ppf(1 - 0.05 / ROUTES, n - 1) * se,
            'level': sum(arm[s] for s in seeds) / n}


def statistics_stdev(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def reading(r):
    if r['ub'] < 0:
        return 'excludes a gain'
    if r['delta'] > 0:
        return 'unresolved, positive'
    return 'unresolved'


def panel(name, arms):
    print('\n### %s\n' % name)
    print('| arm | n | level | delta | s_d | se | UB | UB_adj | reading |')
    print('|---|---|---|---|---|---|---|---|---|')
    for d, prefix, refname, label in PANELS[name]:
        arm = arms.get((d, prefix), {})
        if not arm:
            print('| %s | - | | | | | | | MISSING %s/%s |' % (label, d, prefix))
            continue
        if isinstance(refname, str) and refname.startswith('self:'):
            ref = merged(arms, REFS[refname.split(':')[1]])
            keep = sorted(ref)
            print('| %s | %d | %.2f | --- | --- | --- | --- | --- | reference |'
                  % (label, len(keep), sum(ref[s] for s in keep) / len(keep)))
            continue
        if refname is None:
            print('| %s | %d | %.2f | --- | --- | --- | --- | --- | reference |'
                  % (label, len(arm), sum(arm.values()) / len(arm)))
            continue
        if refname == 'c0':
            ref = c0_for(arms, len(arm), False)
        else:
            ref = merged(arms, REFS[refname])
        r = contrast(arm, ref)
        if r is None:
            print('| %s | %d | %.2f | no shared seeds | | | | | |'
                  % (label, len(arm), sum(arm.values()) / len(arm)))
            continue
        print('| %s | %d | %.2f | %+.3f | %.3f | %.2f | %+.2f | %+.2f | %s |'
              % (label, r['n'], r['level'], r['delta'], r['sd'], r['se'],
                 r['ub'], r['ub_adj'], reading(r)))


def crossed_within(arms):
    """Table 5 quotes deltas against its own first row, not against the baseline."""
    first = arms.get(('results_quad', 'lin0'), {})
    print('\n### crossed, deltas against the first row (as Table 5 prints them)\n')
    print('| arm | n | level | delta vs linear-zero |')
    print('|---|---|---|---|')
    for d, prefix, _, label in PANELS['crossed']:
        arm = arms.get((d, prefix), {})
        seeds = sorted(set(arm) & set(first))
        if len(seeds) < 2:
            continue
        delta = sum(arm[s] - first[s] for s in seeds) / len(seeds)
        print('| %s | %d | %.2f | %+.2f |'
              % (label, len(seeds), sum(arm[s] for s in seeds) / len(seeds), delta))


def main():
    arms = load()
    print('# Tables rebuilt from results/raw (%d arms on disk)' % len(arms))
    want = sys.argv[1:] or list(PANELS)
    for name in want:
        if name not in PANELS:
            print('unknown panel %r; choose from %s' % (name, ', '.join(PANELS)))
            continue
        panel(name, arms)
        if name == 'crossed':
            crossed_within(arms)
    print('\nMultiplicity: %d routes, alpha = 0.05/%d = %.5f per contrast.'
          % (ROUTES, ROUTES, 0.05 / ROUTES))
    print('UB_adj is what the paper quotes wherever a family is read as a whole.')


if __name__ == '__main__':
    main()
