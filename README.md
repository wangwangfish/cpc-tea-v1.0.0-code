# Cosine-prototype head for small imbalanced green-tea grading

Code, data and per-run results for the paper *Classifier corrections, not extra capacity,
lift the weakest class when green tea training data are small* (submitted to *Processes*).

The dataset is the eight-class green-tea set used in the paper: 1708 training and 296
validation images, 134–294 images per class, 2.19:1 imbalance. Every number in the
manuscript is produced by the files in this repository.

```
data/                2004 images in data/tea8/, 8 classes; SHA-256 for each file
                      (the images ship as a Zenodo record, see "Getting the data")
src/                  train_lpr.py (the trainer), train_modern.py, models/
runs/                 one script per experiment family, plus the generic launcher
results/raw/          301 runs: the result.json and history.json of every arm
results/derived/      the joined tables the analysis and figures read
analysis/             rebuild the tables, verify the numbers, redraw the figures
docs/                 reproduction notes, environment, provenance, results
```

## Getting the data

The images are 478 MB, so they are not in this repository: Processes requires datasets
above 250 MB to sit in a repository that mints a DOI, and a half-gigabyte clone would make
the code awkward to use. Download the Zenodo record, unpack it so that `data/tea8/train`
and `data/tea8/val` exist, then verify:

```bash
cd <repo> && sha256sum -c data/SHA256SUMS --quiet && echo "dataset ok"
```

`data/README.md` has the composition and the licence; `data/SHA256SUMS` has all 2004
hashes, so any copy can be checked against this tree.

## Quick start

```bash
python -m pip install -r requirements.txt          # torch 2.5.1+cu124 and friends
python analysis/verify_numbers.py                  # 64 arithmetic checks, no GPU needed
python analysis/make_tables.py                     # rebuild Tables 2-7 from results/raw
bash runs/01_baseline.sh                           # one family, 12 runs, ~5 h on one V100
```

`analysis/verify_numbers.py` and `analysis/make_tables.py` run on a laptop in seconds and
touch nothing but `results/`. Start there: if those two agree with the paper, the data in
this repository is the data the paper was written from.

## What the paper claims, and where each claim is checked

| claim | table | rebuild with |
|---|---|---|
| Four times the parameters buys no detectable gain | Table 2 | `make_tables.py width` |
| Three feature-aggregation variants lose accuracy | Table 3 | `make_tables.py agg` |
| Fourteen operator and structural variants sit at or below their control | Table 4 | `make_tables.py operator` |
| The gain divides between readout geometry and prior placement | Table 5 | `make_tables.py crossed` |
| The head adds 0.87 top-1 and 8.63 weakest-class points | Table 6 | `make_tables.py head` |
| The same three seeds under a split-half evaluation | Table 7 | `verify_numbers.py` |
| The correction collapses below about 100 images per class | Section 9 | `make_tables.py density` |

## Two conventions that change the numbers

**Seeds are not interchangeable.** The twelve-seed baseline is stored in two directories
(`results_paper/r18_sgd_*`, six seeds, and `results_power/r18_*`, six more). The operator
panel has two references: three routes were carried to six seeds and are paired against
the six-seed control, the other eleven against the three-seed control. Pairing against the
wrong reference moves eleven deltas by up to 0.45 points without any run changing.
`analysis/make_tables.py` mirrors both choices and says which it used.

**Two rounding levels.** `result.json` stores `best_acc` to two decimals, which is what
Tables 2–5 use. The main table was recomputed from checkpoints as exact counts over 296
images (279/296 = 94.2568). The two differ by at most 0.005 per seed. Checkpoints are not
distributed — 427 files, 25 GB — so a rerun reproduces the two-decimal level and the
exact counts come from `results/derived/figdata_paper.json`. `docs/RESULTS.md` states
which convention applies to which table.

## Licence and citation

Code: MIT (`LICENSE`). Dataset: CC BY-NC 4.0 (`LICENSE-DATA`), non-commercial research
and teaching. The two licences are separate: the MIT grant over the code does not extend
to the photographs.

Cite the paper, not this repository — the manuscript is under review, so there is no DOI
yet. Once it is published, quote the article DOI together with the Zenodo DOI of the
release you actually used, so that the version behind a number is identifiable.

## What is deliberately missing

Training checkpoints (25 GB) and the raw scraping scripts. Checkpoints are reproducible
from the shipped code and data; the scrapers are not part of the study and their output
is the dataset itself. `docs/PROVENANCE.md` records which machine, directory and script
produced every number, including one batch of runs that was quarantined because a config
flag was wrong — the data for that batch is not shipped, but the mistake is documented.
