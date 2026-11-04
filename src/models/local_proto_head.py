# -*- coding: utf-8 -*-
"""Local prototype readout (LPR).

Motivation: CPC makes each channel respond to a prototype-like pattern, but the
classifier still pools the whole map and applies a single linear layer, so
evidence that appears at only a few positions is diluted. LPR adds a second
path that matches local features against a small set of class prototypes and
aggregates the strongest responses.

    logits = logits_global + lam * logits_local
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalPrototypeHead(nn.Module):
    def __init__(self, in_dim, num_classes, protos_per_class=2, proj_dim=128,
                 topk=3, init_scale=10.0):
        super().__init__()
        self.num_classes = num_classes
        self.P = protos_per_class
        self.topk = topk

        # 1x1 projection to a compact embedding space
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
        )
        # one prototype bank, K*P vectors
        self.protos = nn.Parameter(torch.empty(num_classes * protos_per_class, proj_dim))
        nn.init.orthogonal_(self.protos)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, feat):
        h = self.proj(feat)                       # B, D, H, W
        h = F.normalize(h, dim=1, eps=1e-6)
        p = F.normalize(self.protos, dim=1, eps=1e-6)

        sim = torch.einsum("bdhw,kd->bkhw", h, p)  # B, K*P, H, W
        b, kp, hh, ww = sim.shape
        flat = sim.reshape(b, kp, hh * ww)

        k = min(self.topk, hh * ww)
        top = flat.topk(k, dim=2).values           # B, K*P, k
        per_proto = top.mean(dim=2)                # B, K*P
        per_class = per_proto.reshape(b, self.num_classes, self.P).mean(dim=2)
        return self.scale * per_class


class CosineClassifier(nn.Module):
    """GAP + cosine classifier, to separate operator effect from head effect."""

    def __init__(self, in_dim, num_classes, init_scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, feat):
        v = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        v = F.normalize(v, dim=1, eps=1e-6)
        w = F.normalize(self.weight, dim=1, eps=1e-6)
        return self.scale * (v @ w.t())


class WrapWithHead(nn.Module):
    """Attach an auxiliary head to an existing classifier via a feature hook."""

    def __init__(self, backbone, head, feat_dim=512, lam=1.0, lam_learnable=False):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self._feat = {}
        backbone.layer4.register_forward_hook(self._grab)
        if lam_learnable:
            self.lam = nn.Parameter(torch.tensor(float(lam)))
        else:
            self.register_buffer("lam", torch.tensor(float(lam)))

    def _grab(self, module, inp, out):
        self._feat["x"] = out

    def forward(self, x):
        logits_g = self.backbone(x)
        f = self._feat["x"]
        return logits_g + self.lam * self.head(f)
