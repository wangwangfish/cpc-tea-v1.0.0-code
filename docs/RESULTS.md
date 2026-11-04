# Results, and the two ways they are counted

This file exists because the paper quotes numbers at two levels of precision, and a reader
who recomputes one from the other will sometimes find a hundredth of a point missing.

## The two conventions

**`best_acc` in `result.json`** is the best validation top-1 seen during training, written
by the trainer and rounded to two decimals. Tables 2, 3, 4 and 5 are built from it, by
pairing runs seed by seed.

**Exact counts over 296 images** were recomputed afterwards from the saved checkpoints for
the main results (Table 6) and the cross-checks, as `k/296`. 279/296 is 94.2568, which the
`result.json` of that run records as 94.26. `results/derived/figdata_paper.json` holds the
exact counts; `analysis/reeval_exact.py` is the script that produced them.

The two conventions differ by at most 0.005 per run. On one route it moved a bound in the
second decimal — the `2×` width upper bound is +0.063 from exact counts and +0.065 from
the two-decimal ones — and the paper quotes +0.07, which is what a reader gets from the
shipped files. Rounding to the arithmetic a reader can redo was a deliberate choice over
quoting the prettier exact value.

Checkpoints are not distributed (427 files, 25 GB). Re-running an arm reproduces the
two-decimal level; the exact counts are in `figdata_paper.json`.

## Headline numbers

| arm | n | top-1 | macro | weakest class |
|---|---|---|---|---|
| Baseline, GAP + linear | 12 | 94.59 | 93.39 | 59.23 |
| Prior-prototype head | 12 | 95.47 | 94.56 | 67.86 |
| Class-balanced resampling | 6 | 95.27 | 94.21 | 65.48 |
| Head + resampling | 6 | 95.95 | 95.20 | 71.43 |
| Width 0.5× | 12 | 92.62 | — | — |
| Width 2× | 12 | 94.26 | — | — |

Reproduce with `python analysis/make_tables.py head` and `... width`.

## Paired contrasts

| contrast | n | delta | s_d | UB | UB_adj | reading |
|---|---|---|---|---|---|---|
| 0.5× width vs 1× | 12 | −1.969 | 0.919 | −1.49 | −1.07 | excludes a gain |
| 2× width vs 1× | 12 | −0.337 | 0.776 | +0.07 | +0.43 | unresolved |
| GeM vs 1× | 6 | −0.617 | 1.270 | +0.43 | +1.76 | unresolved |
| Multi-scale vs 1× | 6 | −0.842 | 0.872 | −0.12 | +0.79 | unresolved after correction |
| Both vs 1× | 6 | −2.193 | 1.124 | −1.27 | −0.09 | excludes a gain |
| Head vs 1× | 12 | +0.873 | 0.870 | +1.32 | +1.73 | positive, unresolved |

`UB` is one-sided 95 %: `delta + t(0.95, n-1) · s_d/sqrt(n)`. `UB_adj` corrects the same
bound for seventeen routes (`alpha = 0.05/17`). Both columns are what the paper prints; the
correction is what turns "two of the three aggregation variants exclude a gain" into "one
of them does".

## What changed between drafts, and why it is in the data

Three conclusions moved as seeds were added. The data for both the old and the new seed
counts is shipped, so the movement is checkable rather than assertable:

- **Crossed design, 3 → 6 seeds.** The interaction between readout geometry and prior
  placement was −1.01 on three seeds and −0.33 on six, so "the two ingredients are
  substitutes" was withdrawn.
- **Aggregation axis, 3 → 6 seeds.** Multi-scale fusion went from −0.34 to −0.84 and the
  combined variant from −2.25 to −2.19; at six seeds two of the three exclude a gain
  before correction, one after.
- **Operator routes, 3 → 6 seeds.** Three near-zero routes were re-run on six seeds. Two
  stayed unresolved and the twelve-layer route became a marginal exclusion.

A fourth change is not a result at all: one batch of twelve runs was launched with
`--config r18` where the route needed `--config cpc`, which quietly produced plain
ResNet-18 runs under CPC names. They are quarantined on the server and not shipped;
`docs/PROVENANCE.md` records what the flag did and how the rerun was verified.

## Reproducing a table

```bash
python analysis/make_tables.py            # all panels, markdown
python analysis/make_tables.py operator   # one panel
python analysis/verify_numbers.py         # 64 derived quantities against their sources
```

`verify_numbers.py` recomputes the quantities the prose reasons with — the per-image step
of top-1, the parameter counts, the imbalance ratio, the sign test — and fails loudly when
one drifts. It needs no GPU and no network.
