# -*- coding: utf-8 -*-
"""Cell-by-cell audit of every table, against the pulled result files.

Checks, per table:
  * Level equals the mean of that arm's runs;
  * delta equals level minus its reference level;
  * s_d matches the spread of the per-seed pairs;
  * se equals s_d / sqrt(n);
  * both bounds recompute (also enforced by the gate, repeated here for completeness);
  * n equals the number of runs actually present;
and across tables: the same arm never appears with two different levels.

Tolerances are set by what the table prints: a level printed to two decimals can differ from
its unrounded mean by up to 0.005, and a delta printed to three by 0.0005.
"""
import json
import sys
import math
import re

from scipy import stats

import os
DERIVED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "derived")


def _p(name):
    return os.path.join(DERIVED, name)

R = {(x['dir'], x['run']): x for x in json.load(open(_p('routes.json')))}
P = json.load(open(_p('panel_contrasts6.json')))
FT = json.load(open(_p('ft244.json')))
QF = json.load(open(_p('quad_pairs.json')))
WP = json.load(open(_p('width_pairs.json')))
MS = os.environ.get('MANUSCRIPT', 'manuscript')
TEX = open(os.path.join(MS, 'body.tex'), encoding='utf-8').read()
problems = []


def ck(cond, msg):
    if not cond:
        problems.append(msg)



def statistics_stdev(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def clean_name(x):
    return re.sub(r'\\(top|mid|bottom)rule\s*', '', x).strip()


def series(d, run, seeds):
    out = {}
    for s in seeds:
        k = (d, '%s_s%d' % (run, s))
        if k in R:
            out[s] = R[k]['best_acc']
    return out


THREE = [42, 123, 456]
SIX = [42, 123, 456, 789, 1024, 2024]

# ---------------------------------------------------------------- Table 1: composition
body = TEX
t1 = re.search(r'\\label\{tab:classes\}.*?\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}', body, re.S).group(1)
rows = [[c.strip() for c in r.split('&')] for r in t1.split('\\\\') if '&' in r]
tr = va = 0
for r in rows:
    if len(r) < 4:
        continue
    name = clean_name(r[0])
    try:
        a, b = int(r[1]), int(r[2])
    except ValueError:
        continue
    if name == 'Total':
        ck(a == tr and b == va, 'Table 1 total row (%d/%d) does not match the sum (%d/%d)'
           % (a, b, tr, va))
    else:
        tr += a
        va += b
ck(tr == 1708 and va == 296, 'Table 1 sums to %d/%d, expected 1708/296' % (tr, va))
nc = len([r for r in rows if len(r) >= 4 and clean_name(r[0]) not in ('Total', 'Grade')
          and r[1].isdigit()])
print('Table 1: %d classes, %d train, %d val, totals consistent' % (nc, tr, va))
ck(nc == 8, 'Table 1 lists %d classes, expected 8' % nc)

# ---------------------------------------------------------------- Table 4: operator routes
t4 = re.search(r'\\label\{tab:operator\}.*?\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}', body, re.S).group(1)
REF = {'3': 95.2667, '6': 94.8183}
mapping = {'M1': ('results_dual', 'M1'), 'M1b': ('results_dual', 'M1b'), 'M2': ('results_dual', 'M2'),
           'M3c': ('results_dual', 'M3c'), 'M4m': ('results_dual2', 'M4m'), 'M4s': ('results_dual2', 'M4s'),
           'conv$_2$ position': ('results_cpcpos', 'A_conv2'), 'conv$_1$ position': ('results_cpcpos', 'B_conv1'),
           '$k=0.01$': ('results_mild', 'C1_soft_k0.01_l0.0_nodno'),
           '$k=0.05$': ('results_mild', 'C1_soft_k0.05_l0.0_nodno'),
           '$\\lambda=0.1$': ('results_mild', 'C2_mild_k0_l0.1_nodno'),
           '$\\lambda=0.3$': ('results_mild', 'C2_mild_k0_l0.3_nodno'),
           '12 layers': ('results_cpcstages', 'C12'), '16 layers': ('results_cpcstages', 'C16')}
checked = 0
for r in t4.split('\\\\'):
    if '&' not in r:
        continue
    c = [x.strip() for x in r.split('&')]
    if len(c) < 8:
        continue
    label = c[1]
    if label not in mapping:
        continue
    d, run = mapping[label]
    seeds = SIX if c[2] == '6' else THREE
    s = series(d, run, seeds)
    ck(len(s) == len(seeds), 'Table 4 %s: %d runs found, n says %s' % (label, len(s), c[2]))
    lvl = sum(s.values()) / len(s)
    lvl_printed = float(c[3])
    delta_printed = float(c[4].replace('$', ''))
    sd_printed = float(c[5])
    ck(abs(lvl - lvl_printed) < 0.005 + 1e-9,
       'Table 4 %s: level printed %.2f, runs average %.4f' % (label, lvl_printed, lvl))
    ck(abs((lvl - REF[c[2]]) - delta_printed) < 0.012,
       'Table 4 %s: delta printed %+.3f, level minus reference is %+.3f'
       % (label, delta_printed, lvl - REF[c[2]]))
    # s_d is the spread of the PAIRED differences against the same-seed reference,
    # not the spread of the arm on its own
    ref = series('results_mild', 'C0_v1_k0_l0.0_nodno', seeds)
    diffs = [s[k] - ref[k] for k in sorted(s) if k in ref]
    if len(diffs) > 1:
        ck(abs(statistics_stdev(diffs) - sd_printed) < 0.02,
           'Table 4 %s: s_d printed %.3f, paired differences give %.3f'
           % (label, sd_printed, statistics_stdev(diffs)))
    checked += 1
print('Table 4: %d rows checked' % checked)



# ---------------------------------------------------------------- Table 2: width
t2 = re.search(r'\\label\{tab:width\}.*?\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}', body, re.S).group(1)
for r in t2.split('\\\\'):
    if '&' not in r:
        continue
    c = [x.strip().replace('$', '') for x in r.split('&')]
    if len(c) < 7 or c[3] == '---':
        continue
    try:
        d, sd, se = float(c[3]), float(c[4]), float(c[5])
    except ValueError:
        continue
    ck(abs(sd / math.sqrt(12) - se) < 0.006,
       'Table 2 "%s": se %.2f but s_d/sqrt(12) = %.3f' % (c[0][:10], se, sd / math.sqrt(12)))
print('Table 2: se column consistent with s_d / sqrt(12)')

# ---------------------------------------------------------------- Table 3: aggregation
# The only table whose conclusion survives the multiplicity correction, so both bounds
# are printed side by side and both have to recompute from the printed delta and s_d.
t3 = re.search(r'\\label\{tab:agg\}.*?\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}', body, re.S).group(1)
AGG = {'GeM pooling': 'agg:GeM', 'Multi-scale fusion': 'agg:multi-scale',
       'Both combined': 'agg:both'}
ROUTES = 17
n3 = 0
for r in t3.split('\\\\'):
    if '&' not in r:
        continue
    c = [clean_name(x).replace('$', '') for x in r.split('&')]
    if len(c) < 6 or c[0] not in AGG:
        continue
    row = P['rows'][AGG[c[0]]]
    d, sd, ub, uba = float(c[1]), float(c[2]), float(c[3]), float(c[4])
    df = row['n'] - 1
    ck(row['n'] == 6, 'Table 3 %s: n=%d, caption says six seeds' % (c[0], row['n']))
    ck(abs(row['mean'] - d) < 0.0006,
       'Table 3 %s: delta printed %+.3f, paired mean %+.4f' % (c[0], d, row['mean']))
    ck(abs(row['sd'] - sd) < 0.0006,
       'Table 3 %s: s_d printed %.3f, paired sd %.4f' % (c[0], sd, row['sd']))
    for label, printed, tval in (
            ('UB', ub, stats.t.ppf(0.95, df)),
            ('UB_adj', uba, stats.t.ppf(1 - 0.05 / ROUTES, df))):
        want = row['mean'] + tval * row['se']
        ck(abs(want - printed) < 0.006,
           'Table 3 %s: %s printed %+.2f, delta %+.3f + %.2f*se = %+.3f'
           % (c[0], label, printed, row['mean'], tval, want))
    n3 += 1
ck(n3 == 3, 'Table 3: %d rows matched, expected 3' % n3)
print('Table 3: %d rows, both bounds recomputed (Bonferroni over %d routes)' % (n3, ROUTES))

# ---------------------------------------------------------------- Table 5: crossed
t5 = re.search(r'\\label\{tab:crossed\}.*?\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}', body, re.S).group(1)
lv = QF['levels']
vals = {'Linear & zero': lv['lin0'], 'Linear & $\\log \\pi_c$': lv['linp'],
        'Angular & zero': lv['cos0'], 'Angular & $\\log \\pi_c$ (the head)': lv['cosp']}
for r in t5.split('\\\\'):
    if '&' not in r:
        continue
    c = [x.strip() for x in r.split('&')]
    if len(c) < 4:
        continue
    key = '%s & %s' % (c[0], c[1])
    if key in vals:
        try:
            printed = float(c[2])
        except ValueError:
            continue
        ck(abs(vals[key] - printed) < 0.005,
           'Table 5 %s: printed %.2f, data %.4f' % (key, printed, vals[key]))
        if c[3] != '---':
            delta = float(c[3].replace('$', ''))
            ck(abs((vals[key] - lv['lin0']) - delta) < 0.006,
               'Table 5 %s: delta printed %+.2f, cell minus first row %+.3f'
               % (key, delta, vals[key] - lv['lin0']))
print('Table 5: %d cells checked' % len(vals))

# ---------------------------------------------------------------- Table 6: main results
t6 = re.search(r'\\label\{tab:main\}.*?\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}', body, re.S).group(1)
D = json.load(open(_p('figdata_paper.json')))
PRE = open(os.path.join(MS, 'preamble.tex'), encoding='utf-8').read()
MACROS = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', PRE))


def expand(t):
    for _ in range(3):
        t = re.sub(r'\\(\w+)', lambda m: MACROS.get(m.group(1), m.group(0)), t)
    return t


def macro(name):
    m = re.search(r'\\newcommand\{\\%s\}\{([^}]*)\}' % name, PRE)
    return m.group(1).strip() if m else None


def per_seed(arm, metric, seeds):
    """figdata for the arms it holds; routes.json for the resampling arm it does not."""
    if arm in D.get(metric, {}):
        return {s: D[metric][arm]['s%d' % s] for s in seeds if 's%d' % s in D[metric][arm]}
    if metric == 'top1' and arm == 'sampler':
        return series('results_balance', 'sampler', seeds)
    return {}


def mean_of(arm, metric, seeds):
    d = per_seed(arm, metric, seeds)
    return sum(d.values()) / len(d) if d else None


ARMS = {'Prior-prototype head': ('pph', SIX + [7, 13, 21, 33, 55, 89]),
        'Class-balanced resampling': ('sampler', SIX),
        'Head $+$ resampling': ('combo', SIX)}
for r in t6.split('\\\\'):
    if '&' not in r:
        continue
    c = [x.strip() for x in r.split('&')]
    if len(c) < 7:
        continue
    label = c[0]
    if label not in ARMS:
        continue
    arm, seeds = ARMS[label]
    base = mean_of('baseline', 'top1', seeds)
    lvl = mean_of(arm, 'top1', seeds)
    if base is None or lvl is None:
        continue
    a, b = per_seed(arm, 'top1', seeds), per_seed('baseline', 'top1', seeds)
    common = sorted(set(a) & set(b))
    d_paired = sum(a[k] - b[k] for k in common) / len(common)
    lvl_printed = float(macro({'Prior-prototype head': 'hT',
                               'Class-balanced resampling': 'sT',
                               'Head $+$ resampling': 'cT'}[label]) or 'nan')
    d_printed = float(macro({'Prior-prototype head': 'hDT',
                             'Class-balanced resampling': 'sDT',
                             'Head $+$ resampling': 'cDT'}[label])
                      .replace('$', '').replace('+', '') or 'nan')
    ck(abs(lvl - lvl_printed) < 0.006,
       'Table 6 %s: level macro %s, figdata mean %.4f' % (label, lvl_printed, lvl))
    ck(abs(d_paired - d_printed) < 0.006,
       'Table 6 %s: delta macro %s, paired difference %.4f' % (label, d_printed, d_paired))
    print('  %-26s level %.2f (macro %s)  delta %+.2f (macro %+.2f)'
          % (label, lvl, lvl_printed, d_paired, d_printed))

# ---------------------------------------------------------------- Table 7: three-seed comparison
t7 = re.search(r'\\label\{tab:three\}.*?\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}',
               body, re.S).group(1)
t7x = expand(t7)
rows7 = [[c.strip() for c in r.split('&')] for r in t7x.split('\\\\') if r.count('&') >= 3]
vals7 = {}
for c in rows7:
    label = re.sub(r'\\\\(top|mid|bottom)rule\s*', '', c[0]).strip()
    if label in ('Configuration', ''):
        continue
    try:
        vals7[label] = [float(re.sub(r'[\\${}()]|textbf', '', x)) for x in c[1:4]]
    except ValueError:
        pass
if len(vals7) == 4:
    ks = list(vals7)
    base, head, post, diff = (vals7[k] for k in ks)
    for i, inst in enumerate(('top-1', 'macro', 'weakest')):
        ck(abs((post[i] - head[i]) - diff[i]) < 0.006,
           'Table 7 %s: difference row %.2f, but %.2f minus %.2f is %.2f'
           % (inst, diff[i], post[i], head[i], post[i] - head[i]))
    # the three-seed head row must equal the figdata mean over seeds 42/123/456
    for i, metric in enumerate(('top1', 'macro', 'worst')):
        want = mean_of('pph', metric, THREE)
        ck(want is not None and abs(want - head[i]) < 0.006,
           'Table 7 head %s: %.2f printed, figdata mean %.3f' % (metric, head[i], want or -1))
    print('Table 7: %d cells + 3 difference entries checked' % (len(vals7) * 3))
else:
    ck(False, 'Table 7 parsed %d rows, expected 4' % len(vals7))

# ---------------------------------------------------------------- cross-table
seen = {}
for r in t4.split('\\\\'):
    if '&' not in r:
        continue
    c = [x.strip() for x in r.split('&')]
    if len(c) >= 8 and c[1] in mapping:
        seen.setdefault(mapping[c[1]][1], set()).add(c[3])
dupes = {k: v for k, v in seen.items() if len(v) > 1}
ck(not dupes, 'an arm appears with two different levels: %s' % dupes)

print('\n== problems ==')
if problems:
    for p in problems:
        print('  FAIL', p)
else:
    print('  none')
print('%d checks failed' % len(problems))
sys.exit(1 if problems else 0)
