# Dataset: eight-class green-tea grading

2004 photographs of dry tea leaf, one product per class, photographed on the same bench
under the same light with a phone camera. Split 1708 / 296. The imbalance is a property of
the product, not of the split: Longjing has 294 training images and Zhuyeqing 134, a ratio
of 2.19:1.

## Composition

| class | train | val | total |
|---|---|---|---|
| Anji_Baicha | 248 | 50 | 298 |
| Biluochun | 211 | 32 | 243 |
| Lishui_Xiangcha | 202 | 45 | 247 |
| Liu_an_Guapian | 175 | 30 | 205 |
| Longjing | 294 | 36 | 330 |
| Taiping_Houkui | 219 | 40 | 259 |
| Xinyang_Maojian | 225 | 35 | 260 |
| Zhuyeqing | 134 | 28 | 162 |
| **total** | **1708** | **296** | **2004** |

Files are `.jpg`, `.jpeg` and `.png` in the proportions 1524 / 16 / 464. They are the
originals as collected; nothing was resized, re-encoded or recoloured for the release. The
trainer resizes to 224 px on the fly.

## Reading it

The layout is what `torchvision.datasets.ImageFolder` expects, so the loader in
`src/train_lpr.py` takes the root and does the rest:

```
data/tea8/train/<class>/*.jpg
data/tea8/val/<class>/*.jpg
```

```python
from torchvision import datasets
ds = datasets.ImageFolder('data/tea8/train')
print(len(ds), ds.classes)      # 1708, ['Anji_Baicha', 'Biluochun', ...]
```

## Filenames

Filenames are the scraped originals and 299 of them contain characters that annoy shells:
spaces, `=`, `?`, parentheses (`2Q== (1).jpg`, `images_q=tbn_ANd9Gc...(1).jpg`). They are
kept because the published copy of this dataset uses them and renaming would break the
correspondence. Every script in this repository therefore quotes paths. If you write your
own loop over the files, quote the filenames.

## Where the files are

The images live in a separate folder, `GreenTea-8-1.0.0/`, released as its own artefact
next to the code repository; this directory holds only the manifest and the description.
The 2004 images are not tracked in the git repository — 478 MB is above the 250 MB
threshold at which *Processes* requires a DOI-minting repository, and it is also a lot to
clone for someone who only wants the analysis scripts. They are deposited as a Zenodo
record at <https://doi.org/10.5281/zenodo.23934586>; `data/SHA256SUMS` in this repository is
the manifest of that record, so a
downloaded copy can be verified against the code that used it.

Unpack so that `data/tea8/train/<class>/` and `data/tea8/val/<class>/` exist, then:

```bash
cd <repo> && sha256sum -c data/SHA256SUMS --quiet && echo ok
```

## Integrity

`SHA256SUMS` lists all 2004 files, 478 MB, in the format `sha256sum -c` expects:

```bash
cd <repo> && sha256sum -c data/SHA256SUMS --quiet && echo ok
```

`runs/00_env_check.sh` does this check and also asserts the two counts, so a truncated
copy fails before a sweep starts rather than after an hour of training.

## Provenance and licence

Collected by the authors from a single production line; the images are of loose leaf, no
people, no location metadata. Released under CC BY-NC 4.0 (`../docs/LICENSE-DATA.md`): research
and teaching use, no commercial redistribution. If you use it, cite the paper
(`../CITATION.cff`) — the dataset is not a separate publication, so there is no dataset DOI
of its own; the Zenodo record for this release carries one for the package as a whole.
