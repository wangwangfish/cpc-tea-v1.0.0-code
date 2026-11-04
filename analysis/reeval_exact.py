# -*- coding: utf-8 -*-
"""Re-evaluate every checkpoint behind the width axis and the crossed design, at full
precision, so that no paired number in the paper mixes two levels of rounding.

Why this exists: train_lpr.py writes best_acc into result.json rounded to two decimals,
while the main results in the paper were recomputed from the saved checkpoints as exact
counts over 296 images. Mixing the two produced a bound of +0.06 on one route and +0.07 on
the other. Everything below is recomputed the exact way, and the output replaces the
rounded values in width_pairs.json and quad_pairs.json.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
from train_lpr import build                       # noqa: E402
from models.head_variants import PriorPrototypeHead, QuadHead  # noqa: E402
from torchvision import datasets, transforms                   # noqa: E402
import torchvision.transforms as T                             # noqa: E402

ROOT = REPO + '/'
DEV = torch.device('cuda')
SEEDS = [42, 123, 456, 789, 1024, 2024, 7, 13, 21, 33, 55, 89]
train_cnt = [len(os.listdir(ROOT + 'data/tea8/train/' + c))
             for c in sorted(os.listdir(ROOT + 'dataset_v1/train'))]

tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
val = datasets.ImageFolder(ROOT + 'dataset_v1/val', tf)
loader = torch.utils.data.DataLoader(val, batch_size=128, shuffle=False, num_workers=4)


@torch.no_grad()
def top1(model):
    correct = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        correct += int((model(x).argmax(1) == y).sum())
    return 100.0 * correct / len(val)


def load(path, kind):
    ck = torch.load(os.path.join(ROOT, path, 'best.pth'), map_location='cpu')
    m, _ = build(ck['config'])
    m.fc = nn.Identity()
    if kind == 'base':
        m = nn.Sequential(m, nn.Flatten(), nn.Linear(512, 8))
        # baseline checkpoints keep their fc inside the model; rebuild instead
        m, _ = build(ck['config'])
    elif kind == 'pph':
        m = PriorPrototypeHead(m, train_cnt, mode='pph')
    elif kind == 'lin0':
        m = QuadHead(m, train_cnt, use_cos=False, use_prior=False)
    elif kind == 'linp':
        m = QuadHead(m, train_cnt, use_cos=False, use_prior=True)
    elif kind == 'cos0':
        m = QuadHead(m, train_cnt, use_cos=True, use_prior=False)
    elif kind == 'width':
        m, _ = build(ck['config'])
    m.load_state_dict(ck['model_state_dict'])
    return m.to(DEV).eval()


jobs = []
for s in SEEDS:
    for tag, path, kind in [
        ('base', 'results_paper/r18_sgd_s%d' % s, 'base'),
        ('w0.5', 'results_width/w0.5_s%d' % s, 'width'),
        ('w2.0', 'results_width/w2.0_s%d' % s, 'width'),
    ]:
        if os.path.exists(ROOT + path + '/best.pth'):
            jobs.append((tag, s, path, kind))
for s in SEEDS[:6]:
    for tag, kind in [('lin0', 'lin0'), ('linp', 'linp'), ('cos0', 'cos0'), ('cosp', 'pph')]:
        path = ('results_pph/pph_s%d' % s) if kind == 'pph' else ('results_quad/%s_s%d' % (kind, s))
        if os.path.exists(ROOT + path + '/best.pth'):
            jobs.append((tag, s, path, 'pph' if kind == 'pph' else kind))

print('checkpoints to evaluate:', len(jobs))
out = {}
for i, (tag, s, path, kind) in enumerate(jobs, 1):
    try:
        m = load(path, kind)
        acc = top1(m)
        out.setdefault(tag, {})[str(s)] = acc
        print('%3d/%d %-6s s%-5d %6.2f  (%d/%d)' % (i, len(jobs), tag, s, acc,
                                                    round(acc * 2.96), 296), flush=True)
    except Exception as ex:
        print('%3d/%d %-6s s%-5d FAILED %s' % (i, len(jobs), tag, s, ex), flush=True)

json.dump(out, open(ROOT + 'exact_acc.json', 'w'), indent=1)
print('wrote exact_acc.json with', sum(len(v) for v in out.values()), 'entries')
