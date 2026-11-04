# Provenance

Every number in the paper maps to a machine, a directory, a script and a run. This file
records that map, including the parts that went wrong.

## One machine

All reported results come from a single node: 8 × V100-SXM2-32GB, Ubuntu 24.04, the
`cpc` conda environment. Nothing in the manuscript comes from a second machine. This
matters because an earlier draft mixed in results from a different server whose
`train_lpr.py` was a different file (204 lines against 498); the two are not comparable and
the pretrained numbers were re-run here to remove the mixture.

## Code

`src/train_lpr.py`, 498 lines, md5 `ffd33fc7de487e76020cbcf09f20a3ea`. The file is
CRLF on the machine that ran the experiments and is shipped byte for byte, so that md5
still matches; `.gitattributes` pins it (`* -text`) for the same reason. The only edit after
the last reported run added `--init_imagenet` (default off), which the pretrained arm uses
and no other arm touches. The pre-edit file is not shipped; the flag is additive and the
ten extra lines are visible in `git`-style diff terms as the `if a.init_imagenet:` block.

## Data

`data/tea8`, 1708 / 296, byte-identical to what the trainer read (SHA-256 per file in
`data/SHA256SUMS`, verified after transfer). The paper's validation split is this
directory's `val/`, unchanged across every arm, including the density ladder in Section 9,
which shrinks only the training side.

## The map

| numbers in the paper | sweep | seeds | n |
|---|---|---|---|
| baseline 94.59 / 93.39 / 59.23 | `results_paper/r18_sgd_*` + `results_power/r18_*` | 42 123 456 + 789 1024 2024, and 7 13 21 33 55 89 | 12 |
| head 95.47 / 94.56 / 67.86 | `results_pph/pph_*` | the same twelve | 12 |
| resampling 95.27 / 94.21 / 65.48 | `results_balance/sampler_*` | 42 123 456 789 1024 2024 | 6 |
| head + resampling 95.95 / 95.20 / 71.43 | `results_pph/pph_sampler_*` | the same six | 6 |
| width 0.5× 92.62, −1.97, UB −1.49 | `results_width/w0.5_*` | the twelve | 12 |
| width 2× 94.26, −0.34, UB +0.07 | `results_width/w2.0_*` | the twelve | 12 |
| crossed 94.15 / 94.99 / 94.76 / 95.27 | `results_quad/{lin0,linp,cos0}_*` + `results_pph/pph_*` | 42 123 456 789 1024 2024 | 6 |
| aggregation −0.62 / −0.84 / −2.19 | `results_head/{gem,ms,ms_gem}_*` | the same six | 6 |
| operator routes, 14 | `results_mild`, `results_dual`, `results_dual2`, `results_cpcpos`, `results_cpcstages` | 42 123 456; three routes also 789 1024 2024 | 3 or 6 |
| density +0.23 / +0.34 / −0.12 | `results_imb2/{base,pph}_r{2,5,10}_*` | 42 123 456 | 3 |
| pretrained 97.30 | `results_ft244/ft_*` | 42 123 456 | 3 |
| modern backbones 79.05–93.58 | `results_sgdcheck`, `results_adapt` | 42 only | 1 |

## The mistake that is documented rather than hidden

At 02:00 on 4 November, twelve runs were launched with `--config r18` where the routes
needed `--config cpc`. `use_cpc` is derived from the config name, so those runs built a
plain ResNet-18 — no cosine-prototype layers, no dual prototypes, 11.181 M parameters —
and were filed under the names of CPC routes. The accuracy looked plausible, which is why
it survived a first pass.

They were caught by comparing `n_cpc_layers`, `n_dual_layers` and `params_M` against the
existing arms of the same family, then re-run with the flags from the original sweep
scripts. The rerun was verified the same way. The twelve bad runs are quarantined on the
server in `results_wrong_r18/` and are **not** in this release; the three families they
touched are in `results_dual`, `results_dual2` and `results_cpcstages` at six seeds.

The lesson is in `docs/ENVIRONMENT.md`: `--config` selects the backbone *and* whether the
operator is built at all, and the accuracy alone will not tell you which happened.

## Two protocol corrections, both visible in the data

**A cross-server subtraction was withdrawn.** An earlier draft reported the pretrained
backbone as "about three points" above the best from-scratch model. Those two numbers came
from different machines, different scripts and different epoch budgets, so they were not
subtractable. The pretrained arm was re-run on this machine with this script, and the
sentence became a statement about a separate regime rather than a subtraction.

**A six-seed rerun reversed one conclusion.** The interaction in the crossed design was
−1.01 on three seeds; the same design on six seeds gives −0.33 with an upper bound of
+0.66, so "the two ingredients are substitutes" was withdrawn in five places in the text.
The three-seed data is still in `results_quad` and `results_pph`, so the withdrawal can be
checked rather than taken on trust.

## What is not here

- checkpoints (25 GB) — reproducible from `src/` and `data/`
- the quarantined mislabelled batch — documented above, not distributed
- the scraping scripts — their output is the dataset
- the second server's results — withdrawn from the manuscript
