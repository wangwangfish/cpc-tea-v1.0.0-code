#!/usr/bin/env python3
"""CPCConv2d V2 — targeted fixes for the two measured failure modes.

Motivation (all measured, see MECHANISM_V2_2026-11-01.md)
---------------------------------------------------------
1. LOST MAGNITUDE.  CPC computes a pure cosine.  Empirically, with *identical*
   ImageNet weights, cosine trails standard convolution by 0.90 pp, because
   L2 normalisation discards the per-channel magnitude that ImageNet learned
   (|w| varies 1.59x across channels in pretrained weights vs 1.08x in Xavier).
2. INSUFFICIENT CHANNEL DIFFERENTIATION.  From scratch, CPC lowers the channel
   redundancy (mean |corr|) from 0.50 to 0.32, i.e. it closes only 56.7% of the
   gap between from-scratch and ImageNet-pretrained representations.

Two knobs, both backward compatible with the published operator:

  alpha  -- partial normalisation
        y = tau * (p . x) / (||p|| * ||x||^alpha)
        alpha = 1  -> pure cosine  (the published CPC-Net, default)
        alpha = 0  -> (p.x)/||p||  (input magnitude fully retained)
        ||x|| is divided by sqrt(D) so the scale stays O(1) for alpha < 1.

  per_channel_tau  -- one learnable temperature per output channel instead of
        a single scalar.  Adds out_channels scalars per layer (<0.03% params)
        and lets each prototype choose its own sharpness, which is the direct
        analogue of the per-channel magnitude that cosine throws away.

Backward compatibility: alpha=1.0 and per_channel_tau=False reproduce the
published CPCConv2d bit-for-bit (the extra factor becomes ||x||^0 = 1).

Targeted swap (no edit to the published files):
    import models.cpc_resnet as cr
    from models.cpc_conv_v2 import CPCConv2dV2
    CPCConv2dV2.ALPHA = 0.8
    cr.CPCConv2d = CPCConv2dV2
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cpc_conv import CPCConv2d


class CPCConv2dV2(CPCConv2d):
    """Cosine prototype convolution with optional partial normalisation."""

    # variant configuration (set as class attributes before instantiation)
    ALPHA: float = 1.0
    LEARNABLE_ALPHA: bool = False
    PER_CHANNEL_TAU: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        out_c = self.out_channels
        tau_init = float(self.tau.item())

        if self.PER_CHANNEL_TAU:
            self.log_tau = nn.Parameter(
                torch.full((out_c,), math.log(tau_init)))

        if self.LEARNABLE_ALPHA:
            a = min(max(self.ALPHA, 1e-3), 1 - 1e-3)
            self.raw_alpha = nn.Parameter(torch.tensor(math.log(a / (1 - a))))
        else:
            self.register_buffer("alpha_fixed", torch.tensor(float(self.ALPHA)))

    # --- accessors -------------------------------------------------------
    def alpha_value(self) -> torch.Tensor:
        if self.LEARNABLE_ALPHA:
            return torch.sigmoid(self.raw_alpha)
        return self.alpha_fixed

    def tau_value(self) -> torch.Tensor:
        t = torch.exp(self.log_tau)
        if self.PER_CHANNEL_TAU:
            return t.view(1, 1, -1)          # (1,1,C) for broadcasting
        return t

    def extra_repr(self) -> str:
        tau = self.tau_value().detach()
        tau_s = f"{tau.mean().item():.3f}" + ("(per-ch)" if self.PER_CHANNEL_TAU else "")
        a = self.alpha_value().detach()
        a_s = f"{a.item():.3f}" + ("(learn)" if self.LEARNABLE_ALPHA else "")
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, tau={tau_s}, alpha={a_s}")

    # --- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _, H, W = x.shape
        K_h, K_w = self.kernel_size
        P_h, P_w = self.padding
        S_h, S_w = self.stride
        D_h, D_w = self.dilation

        H_out = (H + 2 * P_h - D_h * (K_h - 1) - 1) // S_h + 1
        W_out = (W + 2 * P_w - D_w * (K_w - 1) - 1) // S_w + 1

        x_padded = F.pad(x, (P_w, P_w, P_h, P_h)) if (P_h > 0 or P_w > 0) else x
        patches = F.unfold(x_padded, kernel_size=(K_h, K_w),
                           dilation=(D_h, D_w), stride=(S_h, S_w))
        patches = patches.permute(0, 2, 1)              # (N, L, D)

        proto_norm = F.normalize(self.prototype, p=2, dim=1)
        pnorm = patches.norm(p=2, dim=2, keepdim=True).clamp_min(1e-8)
        cos = (patches / pnorm) @ proto_norm.t()        # (N, L, C)

        alpha = self.alpha_value()
        if self.LEARNABLE_ALPHA or float(alpha) < 0.999:
            dim = patches.shape[2]
            mag = (pnorm / math.sqrt(dim)).pow(1.0 - alpha)   # (N,L,1)
            cos = cos * mag

        out = self.tau_value() * cos
        return out.permute(0, 2, 1).reshape(N, -1, H_out, W_out)
