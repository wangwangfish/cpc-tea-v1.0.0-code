# Runs

One script per family of experiments in the paper. Each script only writes a task list
(`tag|seed|flags|data`) and pipes it into `_launch.sh`, so the line to check is the flag
line: it is the same command the sweep used on the server.

| script | paper | runs | seeds |
|---|---|---|---|
| `01_baseline.sh` | Table 6 row 1 | 12 | 42 123 456 789 1024 2024 + 7 13 21 33 55 89 |
| `02_head.sh` | Table 6 row 2, Table 5 cell 4 | 12 | the same twelve |
| `03_resampling.sh` | Table 6 row 3 | 6 | 42 123 456 789 1024 2024 |
| `04_head_resampling.sh` | Table 6 row 4 | 6 | 42 123 456 789 1024 2024 |
| `10_width.sh` | Table 2, Section 6 | 24 | the same twelve |
| `11_aggregation.sh` | Table 3 | 18 | 42 123 456 789 1024 2024 |
| `12_crossed.sh` | Table 5 | 18 | 42 123 456 789 1024 2024 |
| `13_operator.sh` | Table 4 | 44 | 42 123 456, three routes on six |
| `14_density.sh` | Section 9 | 18 | 42 123 456 |
| `15_pretrained.sh` | Section 10 | 3 | 42 123 456 |
| `16_modern.sh` | Section 10 ceiling | 24 | 42 only |

## Knobs

`_launch.sh` reads the environment, so nothing here has to be edited to move to another
machine:

```bash
PY=/opt/conda/envs/mine/bin/python \
GPU_LIST=0,1 \
JOBS=2 \
OUT=/scratch/results \
bash runs/02_head.sh
```

`OUT` defaults to `runs_out/` and never to `results/`. The shipped `results/raw/` is the
archive the paper was written from; a rerun must not be able to overwrite it. Compare the
two afterwards with `python analysis/compare_rerun.py runs_out results/raw`.

## Resume

Every task is skipped when its `result.json` already exists, so an interrupted sweep is
resumed by running the same script again. This is also how the original sweeps were
driven: `run_pph.sh` on the server waited for the resampling batch to clear the GPUs,
then filled all eight.

## Timing

About 25 minutes per run at 224 px on a V100-32GB (150 epochs, batch 32, 6 workers), 55
minutes for the `2×` width arm, 8 minutes for the 30-epoch pretrained arm. The scripts
above come to 185 runs, about 80 GPU-hours at those rates; eight V100s cover it in a day.

## Why some arms are missing from these scripts

`13_operator.sh` reproduces the fourteen routes that appear in Table 4 and their control.
The release also ships `results/raw/` for routes that were run and then dropped from the
paper — the weight-normalised convolution control, the C4 rotation experiments, the
σ-sweep, the recipe screen. They are kept because a reader checking "was this tried?" is
better served by the data than by a promise. They are not in any table, and no script here
reproduces them, because the exact flag sets for those are in `results/raw/*/result.json`.
