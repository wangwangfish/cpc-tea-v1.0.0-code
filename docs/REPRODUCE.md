# Reproducing the paper

Three levels, in order of cost. The first two need no GPU.

## 1. Check the arithmetic (seconds)

```bash
pip install scipy          # the only hard dependency for this step
python analysis/verify_numbers.py
```

Recomputes 64 quantities the manuscript reasons with, each against the file it came from:
per-image steps of top-1 and of weakest-class recall, parameter counts, the imbalance
ratio, the sampler's share of the head's gain, the selection bias between best-epoch and
last-epoch, a sign test. Exits non-zero if any of them drifts.

## 2. Rebuild the tables (seconds)

```bash
python analysis/make_tables.py            # all panels
python analysis/make_tables.py width      # Table 2 only
```

Reads `results/raw/` — the 301 shipped `result.json` files — and prints each arm with its
paired contrast, both bounds and a reading. It does not read `results/derived/`, so it is
an independent check on those files rather than a rerun of the same code.

If your numbers differ from the paper's, check the reference first: `make_tables.py`
pairs against the twelve-seed baseline for Tables 2, 3, 5 and 6, but against the
six-or-three-seed C0 for Table 4 depending on how many seeds the route has. Getting that
wrong moves a delta by up to 0.45 without any run being wrong.

## 3. Retrain (about 80 GPU-hours)

```bash
python -m pip install -r requirements.txt
bash runs/00_env_check.sh          # dataset hashes, GPU visibility, one forward pass
bash runs/02_head.sh               # one family: 12 runs, ~5 h on one V100
```

`runs/README.md` maps each script to the table it produces. Every script is resumable and
writes to `runs_out/`, never to `results/`, so a rerun cannot overwrite the archive.

Compare afterwards:

```bash
python analysis/compare_rerun.py runs_out results/raw
```

Expect agreement within a few tenths of a point, not exactly. cuDNN is disabled in the
reported runs precisely to keep that spread small, but kernel selection, GPU model and
library version still move the last digit. `docs/ENVIRONMENT.md` lists the versions the
paper used.

## Reproducing the figures

```bash
python analysis/plot_v3.py
```

Reads `results/derived/` and writes the seven figures. The bounds figure (Figure 3) caps
its x-axis at 4 points and marks the two adjusted bounds that fall outside, because
plotting them at +18.6 would compress everything the figure is for.

## What cannot be reproduced from here

- **Checkpoints.** 427 files, 25 GB, not distributed. `analysis/reeval_exact.py` is
  included for provenance: it is the script that produced the exact k/296 counts in
  `figdata_paper.json`, and it needs the `.pth` files to run.
- **The exact bit pattern of a training run.** See above; the paper's claims are all
  paired across seeds within one machine, which is the design that survives this.
- **The scraper output.** The images are the dataset; `data/tea8` ships them with a
  SHA-256 per file.

## If you only want one number

The claim the title rests on: doubling channel width, a four-fold parameter increase,
moves top-1 by −0.337 points on twelve paired seeds, with a one-sided bound on the gain of
+0.07. `python analysis/make_tables.py width` prints both in under a second.
