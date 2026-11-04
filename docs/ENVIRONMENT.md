# Environment

Every run behind the paper was produced in one conda environment on one machine. Nothing
in the code pins a GPU model, but it does depend on the CUDA build of torch, and on cuDNN
being switched off (`--disable_cudnn`), which is why the numbers here are reproducible to
about a tenth of a point rather than bit for bit.

## Versions actually used

| component | version |
|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.0-100 |
| GPU | 8 × Tesla V100-SXM2-32GB |
| CUDA runtime | 12.4 |
| Python | 3.10.20 |
| torch | 2.5.1+cu124 |
| torchvision | 0.20.1+cu124 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |
| Pillow | 12.3.0 |
| matplotlib | 3.10.9 |
| pandas | 2.3.3 |
| timm | 1.0.28 |

`src/train_lpr.py` needs torch, torchvision, numpy and Pillow only. scipy is needed by
`analysis/` (the t quantiles), matplotlib and pandas by `analysis/plot_v3.py`, and timm by
`src/train_modern.py` alone — the eight-backbone ceiling check. If you only want to
reproduce the from-scratch arms, timm is optional.

## Install

```bash
conda create -n cpc python=3.10 -y && conda activate cpc
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`requirements.txt` holds everything else at the versions above. The torch line is separate
because the wheel index depends on your driver; anything from cu121 up works, and CPU-only
works for `analysis/`.

## Things that will bite

**cuDNN is disabled in every reported run.** Not a preference: with cuDNN on, the same
seed gave 94.9 and 96.3 on different days at the same settings, which is larger than the
effects the paper measures. The flag costs about 15 % throughput and is passed by
`runs/_launch.sh` through `BASE`.

**Workers.** 6 per run, 8 runs in flight, on a 48-core box. More workers per run and the
GPUs starve; fewer and the dataloader is the bottleneck at 224 px.

**`--config` is not only a backbone choice.** `use_cpc` is derived from the config name
(`cfg.startswith("cpc")`), so `--config r18` on a route that expects `--config cpc`
silently produces a plain ResNet-18 with no CPC layers and no error. One batch of twelve
runs was wasted this way before the `n_cpc_layers` / `n_dual_layers` / `params_M` fields
in `result.json` were used as a check. Compare those three fields against the arm you
mean to reproduce; the accuracy alone will not tell you.

**Determinism.** `--seed` fixes python, numpy and torch RNGs and the sampler order, but
not kernel nondeterminism. Two runs of the same seed on different GPUs typically land
within 0.1 points; the paper's paired design assumes exactly this and never compares
across machines.
