# -*- coding: utf-8 -*-
"""Dual-prototype CPC: two matching directions per output channel, competed into one response.

    z_{c,1} = tau * cos(p_{c,1}, x),   z_{c,2} = tau * cos(p_{c,2}, x)
    y_c     = tau * AGG(z_{c,1}, z_{c,2})

AGG in {"max", "soft", "mean"}:
  max  : hard competition                      y = max(z1, z2)
  soft : smooth competition (beta -> inf gives max, beta -> 0 gives mean)
         y = (1/beta) * log( (exp(beta*z1) + exp(beta*z2)) / 2 )
  mean : structural control (tests whether any gain is just a wider linear model)

Everything else — patch normalisation, temperature, output channel count, absence of
bias, zero padding — is inherited unchanged from CPCConv2d, so a dual layer differs from
a single-prototype layer *only* in how many directions a channel may match.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cpc_conv import CPCConv2d


class CPCConv2dDual(CPCConv2d):
    # class-level defaults, snapshotted into the instance at construction time
    AGG = "soft"
    BETA = 10.0
    DELTA = 0.1
    INIT = "mirror"   # mirror = p +- delta (symmetric); independent = second drawn fresh

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # --- snapshot class settings so a second model cannot mutate this one ---
        self.agg = CPCConv2dDual.AGG
        self.beta = float(CPCConv2dDual.BETA)
        self.delta_rel = float(CPCConv2dDual.DELTA)
        self.init_mode = CPCConv2dDual.INIT
        if self.init_mode not in ("mirror", "independent"):
            raise ValueError("INIT must be mirror/independent, got %r" % (self.init_mode,))
        if self.agg not in ("max", "soft", "mean"):
            raise ValueError("AGG must be max/soft/mean, got %r" % (self.agg,))
        if self.groups != 1:
            raise NotImplementedError("CPCConv2dDual supports groups=1 only")

        # --- split the single prototype into two nearby directions ---
        # base was already Xavier-initialised by the parent constructor
        base = self.prototype.data.clone()              # (C_out, D)
        out_c, dim = base.shape
        if self.init_mode == "independent":
            # Mirror init keeps p0 and p1 symmetric; with a symmetric aggregation
            # (mean, and to first order soft) the two gradients stay mirror images and
            # the pair never separates -- measured: cos(p0,p1) stays at its initial
            # value through 150 epochs. Drawing the second direction independently
            # removes that fixed point, so 'multi-direction matching' is actually tested.
            bound = math.sqrt(6.0 / float(dim + out_c))
            second = torch.empty(out_c, dim).uniform_(-bound, bound)
            dual = torch.stack([base, second], dim=1)              # (C_out, 2, D)
        else:
            gen = torch.Generator().manual_seed(0)      # same split for every layer
            noise = torch.randn(out_c, dim, generator=gen)
            scale = base.std().clamp_min(1e-6) * self.delta_rel
            delta = noise * scale
            dual = torch.stack([base + delta, base - delta], dim=1)  # (C_out, 2, D)
        self.prototype = nn.Parameter(dual)

        # --- usage statistics: is the second prototype ever the winner? ---
        self._wins = 0.0
        self._count = 0.0

    def reset_stats(self) -> None:
        self._wins = 0.0
        self._count = 0.0

    def win_rate(self) -> float:
        return self._wins / max(self._count, 1.0)

    def _aggregate(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., 2) -> (...)"""
        if self.agg == "max":
            return z.max(dim=-1).values
        if self.agg == "mean":
            return z.mean(dim=-1)
        b = self.beta
        return (torch.logsumexp(b * z, dim=-1) - math.log(2.0)) / b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, H, W = x.shape
        K_h, K_w = self.kernel_size
        P_h, P_w = self.padding
        S_h, S_w = self.stride
        D_h, D_w = self.dilation

        H_out = (H + 2 * P_h - D_h * (K_h - 1) - 1) // S_h + 1
        W_out = (W + 2 * P_w - D_w * (K_w - 1) - 1) // S_w + 1

        if P_h > 0 or P_w > 0:
            x_padded = F.pad(x, (P_w, P_w, P_h, P_h))
        else:
            x_padded = x
        patches = F.unfold(x_padded, kernel_size=(K_h, K_w),
                           dilation=(D_h, D_w), stride=(S_h, S_w))
        patches = patches.permute(0, 2, 1)                       # (N, L, D)

        proto_norm = F.normalize(self.prototype, p=2, dim=2)     # (C_out, 2, D)
        out_c, two, dim = proto_norm.shape
        proto_flat = proto_norm.reshape(out_c * two, dim).t()     # (D, C_out*2)

        patches_norm = F.normalize(patches, p=2, dim=2)          # (N, L, D)
        cos = torch.matmul(patches_norm, proto_flat)             # (N, L, C_out*2)
        cos = cos.reshape(N, -1, out_c, two)                     # (N, L, C_out, 2)

        if not self.training:
            with torch.no_grad():
                self._wins += float((cos[..., 0] > cos[..., 1]).sum())
                self._count += float(cos[..., 0].numel())

        y = self.tau * self._aggregate(cos)                      # (N, L, C_out)
        y = y.permute(0, 2, 1).reshape(N, -1, H_out, W_out)
        return y
