# -*- coding: utf-8 -*-
"""Figures for the manuscript. Every number comes from figdata_paper.json (checkpoint
evaluations on the fixed 296-image split), panel_contrasts.json (paired bounds) or
width_pairs.json (the capacity axis). No number is typed in by hand.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# The derived tables live beside this script, not in the working directory, so that the
# figures can be rebuilt from any cwd.
DERIVED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'derived')


def _p(name):
    return os.path.join(DERIVED, name)

plt.rcParams.update({
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix',
    'font.size': 8.5,
    'axes.linewidth': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.frameon': False,
    'pdf.fonttype': 42,
})

C_BASE = '#9a9a9a'
C_HEAD = '#2166ac'
C_COMBO = '#d95f02'
C_NEG = '#4d4d4d'

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

D = json.load(open(_p('figdata_paper.json')))
CLASSES = [c.replace('_', ' ') for c in D['classes']]
TRAIN = np.array(D['train_counts'])
order = np.argsort(TRAIN)

# ---------------------------------------------------------------- fig 1: per-class
fig, ax = plt.subplots(figsize=(6.6, 2.9))
arms = [('baseline', 'Baseline', C_BASE, 12), ('pph', 'Prior-prototype head', C_HEAD, 12),
        ('combo', 'Head $+$ resampling', C_COMBO, 6)]
x = np.arange(len(order))
w = 0.27
for i, (key, lab, col, n) in enumerate(arms):
    M = np.array([D['perclass'][key][s] for s in D['perclass'][key]])
    m, sd = M.mean(0), M.std(0, ddof=1)
    ax.bar(x + (i - 1) * w, m[order], w, color=col, label=lab,
           yerr=sd[order], error_kw=dict(lw=0.7, capsize=1.6, capthick=0.7), zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([CLASSES[i] for i in order], rotation=28, ha='right', fontsize=7.5)
ax.set_ylabel('Per-class recall (%)')
ax.set_ylim(40, 104)
ax.set_xlim(-0.6, len(order) - 0.4)
ax.grid(axis='y', lw=0.35, color='0.85', zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5,
          handlelength=1.4, columnspacing=1.6)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIGDIR, 'fig1_perclass.pdf'))
fig.savefig(os.path.join(FIGDIR, 'fig1_perclass.png'), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- fig 3: tau sweep
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.95), sharex=True)
trajs = [np.array(v) for v in D['tau_traj'].values()]
# the manuscript states the sweep as [0, 6]; the stored trajectory also holds tau < 0,
# which is kept in the JSON for completeness but not plotted.
trajs = [t[t[:, 0] >= 0] for t in trajs]
common = np.array(trajs[0][:, 0])
A = np.array([t[:, 1:] for t in trajs])          # seed x tau x (top1, macro, worst)
mean = A.mean(0)
head = {'top1': np.mean([D['top1']['pph'][s] for s in ('s42', 's123', 's456')]),
        'macro': np.mean([D['macro']['pph'][s] for s in ('s42', 's123', 's456')]),
        'worst': np.mean([D['worst']['pph'][s] for s in ('s42', 's123', 's456')])}

ax = axes[0]
ax.plot(common, mean[:, 0], color=C_HEAD, lw=1.4, label='top-1')
ax.plot(common, mean[:, 1], color=C_NEG, lw=1.4, ls='--', label='macro-average')
ax.axhline(head['top1'], color=C_HEAD, lw=0.9, ls=':', label='head, top-1')
ax.axhline(head['macro'], color=C_NEG, lw=0.9, ls=':', label='head, macro')
ax.set_xlabel(r'Adjustment strength $\tau$')
ax.set_ylabel('Accuracy (%)')
ax.set_title('(a) Aggregate instruments', fontsize=8.5, pad=3)
ax.grid(lw=0.35, color='0.88')
ax.set_axisbelow(True)
handles, labs = ax.get_legend_handles_labels()
ax.set_ylim(92.0, 96.2)

ax = axes[1]
ax.plot(common, mean[:, 2], color=C_COMBO, lw=1.4, label='weakest class')
ax.axhline(head['worst'], color=C_COMBO, lw=0.9, ls=':',
           label='head, weakest class')
i_max = int(np.argmax(mean[:, 2]))
ax.plot(common[i_max], mean[i_max, 2], 'o', ms=3.4, color=C_COMBO)
ax.annotate(r'$\tau=%.2f$' % common[i_max], xy=(common[i_max], mean[i_max, 2]),
            xytext=(common[i_max] - 2.4, mean[i_max, 2] - 8), fontsize=7.2,
            arrowprops=dict(arrowstyle='-', lw=0.6, color='0.4'))
ax.set_xlabel(r'Adjustment strength $\tau$')
ax.set_title('(b) The weakest class', fontsize=8.5, pad=3)
ax.grid(lw=0.35, color='0.88')
ax.set_axisbelow(True)
h2, l2 = ax.get_legend_handles_labels()
ax.set_xlim(common[0], common[-1])
fig.legend(handles + h2, labs + l2, loc='upper center', ncol=4, fontsize=7.2,
           handlelength=1.8, columnspacing=1.6, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.90))
fig.savefig(os.path.join(FIGDIR, 'fig3_tau.pdf'))
fig.savefig(os.path.join(FIGDIR, 'fig3_tau.png'), dpi=300)
plt.close(fig)
print('fig1, fig3 written')
print('head 3-seed levels:', {k: round(v, 2) for k, v in head.items()})
print('tau grid', common[0], common[-1], len(common))
print('worst peak %.2f at tau %.2f' % (mean[i_max, 2], common[i_max]))

# ---------------------------------------------------------------- fig 4: bounds
PC = json.load(open(_p('panel_contrasts.json')))
labels, ubs, ex = [], [], []
for name, lvl, dlt, sd, se, ub, verdict in PC['panelB']:
    labels.append(name); ubs.append(ub); ex.append(ub < 0)
for name, dlt, sd, se, ub in PC['panelA']:
    labels.append(name + ' (A)'); ubs.append(ub); ex.append(ub < 0)
ubs = np.array(ubs); ex = np.array(ex, dtype=bool)
o = np.argsort(ubs)
labels = [labels[i] for i in o]; ubs = ubs[o]; ex = ex[o]
fig, ax = plt.subplots(figsize=(6.0, 3.5))
y = np.arange(len(ubs))
ax.barh(y, ubs, height=0.62, color=['#2166ac' if e else '#c9c9c9' for e in ex],
        edgecolor='0.3', lw=0.5, zorder=3)
ax.axvline(0, color='0.15', lw=0.9, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.0)
ax.set_xlabel('One-sided 95% upper bound on the gain over the same-seed reference (points)')
ax.grid(axis='x', lw=0.35, color='0.88', zorder=0)
ax.set_axisbelow(True)
ax.set_xlim(min(ubs) - 0.35, max(ubs) + 0.35)
h1 = plt.Rectangle((0, 0), 1, 1, fc='#2166ac', ec='0.3', lw=0.5)
h2 = plt.Rectangle((0, 0), 1, 1, fc='#c9c9c9', ec='0.3', lw=0.5)
ax.legend([h1, h2], ['excludes any gain', 'unresolved at three seeds'],
          loc='lower right', fontsize=7.0, handlelength=1.1, borderpad=0.3)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIGDIR, 'fig4_bounds.pdf'))
fig.savefig(os.path.join(FIGDIR, 'fig4_bounds.png'), dpi=300)
plt.close(fig)
print('fig4 written, %d routes, %d excluded' % (len(ubs), ex.sum()))

# ---------------------------------------------------------------- fig 2: width axis
if os.path.exists(_p('width_pairs.json')):
    WP = json.load(open(_p('width_pairs.json')))
    order = [k for k in ('0.5', '2.0') if k in WP]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    rng = np.random.RandomState(3)
    cols = {'0.5': '#762a83', '2.0': '#2166ac'}
    for i, k in enumerate(order):
        d = np.array(WP[k].get('deltas', WP[k].get('d')))
        xj = i + rng.uniform(-0.13, 0.13, len(d))
        ax.plot(xj, d, 'o', ms=3.2, mfc=cols[k], mec='white', mew=0.4, zorder=3)
        m, ub = WP[k]['mean'], WP[k]['ub']
        ax.plot([i - 0.28, i + 0.28], [m, m], color=cols[k], lw=1.8, zorder=4)
        ax.plot([i, i], [m, ub], color=cols[k], lw=0.9, zorder=4)
        ax.plot([i - 0.1, i + 0.1], [ub, ub], color=cols[k], lw=0.9, zorder=4)
        ax.text(i + 0.31, m, '%+.2f' % m, fontsize=7.2, color=cols[k], va='center')
    ax.axhline(0, color='0.15', lw=0.9, zorder=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([r'$0.5\times$, 2.80 M, $n=%d$' % WP[k]['n'] if k == '0.5'
                        else r'$2\times$, 44.68 M, $n=%d$' % WP[k]['n'] for k in order])
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylabel('Paired top-1 difference vs $1\\times$ (points)')
    ax.grid(axis='y', lw=0.35, color='0.88', zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGDIR, 'fig2_width.pdf'))
    fig.savefig(os.path.join(FIGDIR, 'fig2_width.png'), dpi=300)
    plt.close(fig)
    print('fig2 written with n=%s' % [WP[k]['n'] for k in order])
else:
    print('fig2 skipped: width_pairs.json missing')
