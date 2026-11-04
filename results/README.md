# Results

```
raw/      one directory per sweep, one subdirectory per run:
              <sweep>/<run>/result.json     hyperparameters and the headline accuracy
              <sweep>/<run>/history.json    per-epoch train loss and validation top-1
          history.json is here because a run that stops early, or plateaus and then
          jumps, is otherwise indistinguishable from one that trained normally.
derived/  the joined tables the analysis reads. Regenerable, shipped for convenience.
```

301 runs, 24 sweeps. No checkpoints: 427 files, 25 GB, reproducible from `src/` and
`data/`.

## result.json

One object per run. The fields that matter for reproduction are the flags; the rest are
outcomes.

| field | meaning |
|---|---|
| `config` | backbone selector; `use_cpc` is derived from its prefix, so `r18` and `cpc` are not interchangeable |
| `head`, `head_variant` | readout: `linear`, `pph`, `gem`, `ms`, `lin0`, `linp`, `cos0`, `cosp` |
| `optimizer`, `alpha` | `sgd` (lr in the sweep script) or `adamw` |
| `op`, `kappa`, `lambda_dir` | cosine-prototype operator variant and its two regularisers |
| `cpc_pos`, `cpc_stages` | where the operator sits; `0,0,1,1` is the published eight layers |
| `dual_agg`, `dual_delta`, `dual_init`, `dual_stages` | dual-prototype competition |
| `cls_balance`, `cb_beta` | counter-imbalance method |
| `aug`, `aug_rot`, `mixup`, `cutmix` | augmentation actually used |
| `n_cpc_layers`, `n_dual_layers`, `params_M` | **check these three before trusting an arm** |
| `best_acc`, `best_epoch`, `last_acc` | validation top-1, rounded to two decimals |
| `minutes`, `seed`, `epochs`, `resolution` | wall clock and schedule |

`n_cpc_layers` and `n_dual_layers` are there because `--config` silently controls whether
the operator is built at all. A run with `config: r18` under a CPC name has
`n_cpc_layers: 0` and `params_M: 11.181`; that is how the twelve mislabelled runs were
caught. They are quarantined and not shipped.

## Which sweep is which

| sweep | what it is | used by |
|---|---|---|
| `results_paper` | first batch: baseline, CPC, AdamW on six seeds | Table 6, every contrast |
| `results_power` | power extension: six more seeds of baseline and CPC | Table 6 (merged with the above) |
| `results_pph` | prior-prototype head, and head + resampling | Tables 5, 6 |
| `results_balance` | effective-number weighting, resampling, both | Table 6 |
| `results_width` | channel width 0.5× and 2× | Table 2 |
| `results_head` | GeM, multi-scale, both | Table 3 |
| `results_quad` | linear/angular × zero/log-prior, the crossed design | Table 5 |
| `results_mild`, `results_dual`, `results_dual2`, `results_cpcpos`, `results_cpcstages` | the fourteen operator routes and the C0 control | Table 4 |
| `results_imb2` | 1200-image subsets at imbalance 2.19 / 4.98 / 10.14 | Section 9 |
| `results_ft244` | ImageNet-initialised, 30 epochs | Section 10 |
| `results_sgdcheck`, `results_adapt` | timm backbones, single seed | Section 10 ceiling |
| `results_alpha`, `results_c4`, `results_confirm`, `results_recipe`, `results_sigma`, `results_v3`, `results_wn`, `results_awcos`, `results_balanced`, `results_imb` | screens and controls that are not in any table | — |

The last row is not waste: six of those screens are why the paper does not claim things it
would otherwise have claimed. They are shipped because "was this tried?" should be
answerable from data rather than from a promise.

## derived/

| file | contents |
|---|---|
| `routes.json` | 123 runs flattened to one row each, with the fields the operator panel pairs on |
| `panel_contrasts.json`, `panel_contrasts6.json` | per-route paired statistics at three and six seeds |
| `width_pairs.json`, `quad_pairs.json` | per-seed paired differences for the capacity axis and the crossed design, from exact counts |
| `figdata_paper.json` | every per-seed number the figures and the main table need, as exact k/296 |
| `ft244.json` | the three pretrained runs |

`analysis/make_tables.py` deliberately does **not** read these: it recomputes from
`raw/`, so the two paths can disagree and be noticed.
