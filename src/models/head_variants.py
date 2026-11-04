# -*- coding: utf-8 -*-
"""Head variants: replace the plain global-average-pool + linear head.

Motivation. Sixteen earlier interventions all changed how a convolution computes its
output. None touched how features are aggregated before the classifier, yet that is where
fine-grained information is lost: stage 4 is 512 x 7 x 7, and the 2-3% leaf-margin
differences we care about are gone by then.

  gem     generalised-mean pooling over stage 4 (one learnable exponent)
  ms      concatenate pooled stage 3 (256 x 14 x 14, more detail) with pooled stage 4
  ms_gem  both

Only the pooling and the final linear layer change; the backbone is untouched.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPool(nn.Module):
    """Generalised mean pooling: (mean x^p)^(1/p). p=1 gives average, p->inf gives max."""

    def __init__(self, p: float = 3.0, learnable: bool = True):
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.tensor(float(p)))
        else:
            self.p = None
            self.register_buffer("p_fixed", torch.tensor(float(p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p if self.p is not None else self.p_fixed
        p = p.clamp(min=1.0, max=10.0)
        x = x.clamp(min=1e-6)
        out = F.avg_pool2d(x.pow(p), (x.size(2), x.size(3)))
        return out.pow(1.0 / p).flatten(1)


class HeadVariant(nn.Module):
    """Backbone with its GAP+FC head replaced.

    The backbone's own avgpool/fc are bypassed (fc set to Identity by the caller), and
    features are read from stage 3 / stage 4 through permanent forward hooks.
    """

    def __init__(self, backbone: nn.Module, n_cls: int, mode: str = "gem",
                 c3: int = 256, c4: int = 512):
        super().__init__()
        self.bb = backbone
        self.mode = mode
        self.use_ms = mode in ("ms", "ms_gem")
        self.pool = GeMPool(3.0) if mode in ("gem", "ms_gem") else nn.AdaptiveAvgPool2d(1)

        self._feats = {}
        self.bb.layer4.register_forward_hook(
            lambda m, i, o: self._feats.__setitem__("f4", o))
        if self.use_ms:
            self.bb.layer3.register_forward_hook(
                lambda m, i, o: self._feats.__setitem__("f3", o))

        in_dim = c4 + (c3 if self.use_ms else 0)
        self.fc = nn.Linear(in_dim, n_cls)
        nn.init.normal_(self.fc.weight, 0.0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def _g(self, t: torch.Tensor) -> torch.Tensor:
        if isinstance(self.pool, nn.AdaptiveAvgPool2d):
            return self.pool(t).flatten(1)
        return self.pool(t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.bb(x)
        v = self._g(self._feats["f4"])
        if self.use_ms:
            v = torch.cat([v, self._g(self._feats["f3"])], dim=1)
        return self.fc(v)

class PriorPrototypeHead(nn.Module):
    """Cosine prototype classifier with a learnable class-prior bias.

        logits_c = s * cos(f(x), m_c) + b_c,   b_c initialised to log(pi_c)

    The class prior enters as a parameter rather than through resampling, so the training
    distribution is left untouched: no duplicated minority samples, no distorted batch
    statistics. Unlike the prototype matching tried inside the convolutions, this operates
    on the pooled feature, where a class-level direction is meaningful.
    """

    def __init__(self, backbone: nn.Module, class_counts, mode: str = "pph",
                 c4: int = 512, init_scale: float = 10.0):
        super().__init__()
        self.bb = backbone
        self.mode = mode
        self.pool = GeMPool(3.0) if mode == "pph_gem" else nn.AdaptiveAvgPool2d(1)
        self._feats = {}
        self.bb.layer4.register_forward_hook(
            lambda m, i, o: self._feats.__setitem__("f4", o))

        n_cls = len(class_counts)
        self.prototypes = nn.Parameter(torch.randn(n_cls, c4) * 0.02)
        cnt = torch.as_tensor(list(class_counts), dtype=torch.float32)
        prior = cnt / cnt.sum()
        self.bias = nn.Parameter(torch.log(prior.clamp_min(1e-8)))
        self.log_s = nn.Parameter(torch.tensor(float(np.log(init_scale))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.bb(x)
        f = self._feats["f4"]
        f = self.pool(f) if isinstance(self.pool, GeMPool) else self.pool(f).flatten(1)
        f = F.normalize(f, dim=1)
        m = F.normalize(self.prototypes, dim=1)
        return torch.exp(self.log_s) * (f @ m.t()) + self.bias

class QuadHead(nn.Module):
    """2x2 decomposition: linear vs cosine geometry x prior bias vs no prior bias.

    PPH mixes two ingredients. Without separating them the claim "the class prior in the
    parameters is what helps" has no support. This head exposes both switches:

        use_cos=False, use_prior=False   linear classifier, zero-initialised bias
        use_cos=False, use_prior=True    + log(pi) bias
        use_cos=True,  use_prior=False   cosine geometry only
        use_cos=True,  use_prior=True    both (= PPH)
    """

    def __init__(self, backbone, class_counts, use_cos=True, use_prior=True, c4=512):
        super().__init__()
        self.bb = backbone
        self.use_cos = use_cos
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._feats = {}
        self.bb.layer4.register_forward_hook(
            lambda m, i, o: self._feats.__setitem__("f4", o))
        n = len(class_counts)
        cnt = torch.as_tensor(list(class_counts), dtype=torch.float32)
        prior = cnt / cnt.sum()
        b0 = torch.log(prior.clamp_min(1e-8)) if use_prior else torch.zeros(n)
        self.bias = nn.Parameter(b0)
        if use_cos:
            self.prototypes = nn.Parameter(torch.randn(n, c4) * 0.02)
            self.log_s = nn.Parameter(torch.tensor(float(np.log(10.0))))
        else:
            self.lin = nn.Linear(c4, n)
            nn.init.normal_(self.lin.weight, 0.0, 0.01)
            nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        self.bb(x)
        f = self.pool(self._feats["f4"]).flatten(1)
        if self.use_cos:
            z = torch.exp(self.log_s) * (F.normalize(f, dim=1) @ F.normalize(self.prototypes, dim=1).t())
        else:
            z = self.lin(f)
        return z + self.bias
