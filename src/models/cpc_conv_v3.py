"""CPC v3: stronger removal of the dominant common mode.

Why (see MECHANISM_2026-11-01.md)
---------------------------------
A standard conv response factors as

    y_c(t) = ||w_c|| * ||x_t|| * cos(w_c, x_t)
              rho_c      m_t       direction

CPC drops both rho_c and m_t.  rho_c is free to drop because the following BN
re-supplies a learnable per-channel gain.  m_t is *not* free: it is the per-position
input energy, and BN (which normalises across the whole map) cannot restore it.

The measurement that motivates this module:

    * 3x3 patch vectors carry 76.6% of their energy in a single common direction,
      and that direction's strength varies per position (local contrast / DC level).
      Dividing by ||x_t|| lowers it only to 60.4%.
    * Consequence: on layer-4 features the top eigenvalue share of the channel
      covariance is 69.5% for from-scratch ResNet-18 and 34.0% for CPC-Net --
      i.e. CPC halves the dominant common mode, which is exactly the channel
      redundancy the CTD hypothesis talks about.
    * Pretraining removes the same mode on its own (channel mean |corr| drops
      0.4955 -> 0.1833), which is why CPC's benefit turns negative there.

If the mechanism is right, then removing that mode *more cleanly* should help in the
low-information regime and hurt in the high-information one -- i.e. it should make
the "CPC benefit vs. available information" curve steeper, without raising the
absolute ceiling.

Modes
-----
cos   : identical to the published CPCConv2d (kept bit-exact for control / QC)
dc    : project both patches and prototypes onto the DC-free subspace of every
        input channel (subtract the per-window mean), then cosine.  For a 1x1
        kernel the DC component *is* the signal, so this mode falls back to plain
        cosine there.
lcn   : local contrast normalisation of the input -- divide each channel by the
        mean absolute value over a 7x7 window -- followed by DC-free cosine.
        This removes the *relative* contrast rather than the absolute energy.

Parameter count is identical to Conv2d / CPCConv2d in every mode.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_pair(v: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(v, Iterable):
        return tuple(v)  # type: ignore[return-value]
    return (int(v), int(v))


class CPCConv2dV3(nn.Module):
    """Cosine-prototype convolution with configurable common-mode removal."""

    #: "cos" | "dc" | "lcn" | "soft" | "mild"
    MODE = "cos"
    #: soft mode: energy floor delta = KAPPA * mean patch norm (frozen per layer)
    KAPPA = 0.0
    #: mild mode: suppression strength on the analytic DC direction
    LAMBDA_DIR = 0.0
    #: window for local contrast normalisation (LCN mode only)
    LCN_WINDOW = 7
    #: learnable scalar temperature, same as the published operator
    TAU_INIT = 5.0

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, eps=1e-6, **_):
        super().__init__()
        if bias:
            raise ValueError("CPCConv2dV3 does not support bias; set bias=False")
        if groups != 1:
            raise ValueError("CPCConv2dV3 currently supports groups=1 only")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_pair(kernel_size)
        self.stride = _to_pair(stride)
        self.padding = _to_pair(padding)
        self.dilation = _to_pair(dilation)
        self.groups = groups
        self.eps = eps

        # Snapshot the class-level configuration onto the instance.  Without this,
        # every module would read the mutable class attribute at *forward* time, so
        # constructing a second model with a different mode would silently change the
        # behaviour of the first one (this bit us in QC).
        self.MODE = type(self).MODE
        self.KAPPA = type(self).KAPPA
        self.LAMBDA_DIR = type(self).LAMBDA_DIR

        dim = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.prototype = nn.Parameter(torch.empty(out_channels, dim))
        self.log_tau = nn.Parameter(torch.tensor(math.log(self.TAU_INIT),
                                                 dtype=torch.float32))
        # energy floor for MODE="soft"; stays 0.0 (= exact CPC) until calibrated
        self.register_buffer("delta", torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.prototype)

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    # ------------------------------------------------------------------ helpers
    @property
    def _use_dc(self) -> bool:
        """DC removal needs a window larger than one pixel to mean anything."""
        return self.MODE in ("dc", "lcn") and self.kernel_size[0] * self.kernel_size[1] > 1

    def _remove_dc_proto(self, p: torch.Tensor) -> torch.Tensor:
        if not self._use_dc:
            return p
        K = self.kernel_size[0] * self.kernel_size[1]
        q = p.view(self.out_channels, self.in_channels, K)
        q = q - q.mean(dim=2, keepdim=True)
        return q.reshape(self.out_channels, -1)

    @staticmethod
    def _remove_dc_patch(pat: torch.Tensor, C: int, K: int) -> torch.Tensor:
        N, L, _ = pat.shape
        q = pat.view(N, L, C, K)
        q = q - q.mean(dim=3, keepdim=True)
        return q.reshape(N, L, C * K)

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        K_h, K_w = self.kernel_size
        S_h, S_w = self.stride
        D_h, D_w = self.dilation
        P_h, P_w = self.padding

        if self.MODE == "lcn":
            w = self.LCN_WINDOW
            local = F.avg_pool2d(x.abs(), w, stride=1, padding=w // 2)
            x = x / (local + self.eps)

        if P_h or P_w:
            # Replicate padding (not zeros) whenever DC removal is active: with zero
            # padding a border patch mixes real values with zeros, so removing its local
            # mean leaves a spurious pattern -- a uniform image would produce a bright
            # frame and the operator would lose brightness-shift invariance at the
            # border.  Replication keeps every window "all real", so the DC-free
            # response is exactly invariant.  MODE="cos" keeps the original zero
            # padding so that it stays bit-identical to CPCConv2d.
            mode = "replicate" if self._use_dc else "constant"
            x_p = F.pad(x, (P_w, P_w, P_h, P_h), mode=mode)
        else:
            x_p = x
        H_out = (H + 2 * P_h - D_h * (K_h - 1) - 1) // S_h + 1
        W_out = (W + 2 * P_w - D_w * (K_w - 1) - 1) // S_w + 1

        patches = F.unfold(x_p, kernel_size=(K_h, K_w), dilation=(D_h, D_w),
                           stride=(S_h, S_w)).permute(0, 2, 1)          # N,L,D
        D = patches.shape[2]

        if self.MODE in ("soft", "mild"):
            # Both modes keep the ORIGINAL patch norm in the denominator, so a
            # low-energy patch stays low-response instead of being blown up to unit
            # length.  Only the numerator is modified.  lambda=0 / delta=0 reduce
            # exactly to the published operator.
            proto = F.normalize(self.prototype, p=2, dim=1, eps=self.eps)

            if self.MODE == "mild" and self.LAMBDA_DIR > 0:
                # analytic DC direction u = 1/sqrt(D); numerator uses A_lambda x
                u = torch.full((D,), 1.0 / math.sqrt(D), device=patches.device)
                proj = (patches * u).sum(dim=2, keepdim=True)           # u^T x
                num = torch.matmul(patches - self.LAMBDA_DIR * proj * u,
                                   proto.t())
            else:
                num = torch.matmul(patches, proto.t())

            pnorm = patches.pow(2).sum(dim=2, keepdim=True)             # ||x||^2
            den = (pnorm + (self.delta ** 2 if self.MODE == "soft"
                            else self.eps ** 2)).sqrt()
            out = self.tau * (num / den)
        else:
            if self._use_dc:
                patches = self._remove_dc_patch(patches, C, K_h * K_w)
            patches = F.normalize(patches, p=2, dim=2, eps=self.eps)
            proto = F.normalize(self._remove_dc_proto(self.prototype), p=2, dim=1,
                                eps=self.eps)
            out = self.tau * torch.matmul(patches, proto.t())

        return out.permute(0, 2, 1).reshape(N, self.out_channels, H_out, W_out)

    def mean_patch_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Mean L2 norm of the unfolded patches -- used to calibrate `delta`."""
        K_h, K_w = self.kernel_size
        P_h, P_w = self.padding
        x_p = F.pad(x, (P_w, P_w, P_h, P_h)) if (P_h or P_w) else x
        p = F.unfold(x_p, kernel_size=(K_h, K_w), dilation=self.dilation,
                     stride=self.stride)
        return p.pow(2).sum(dim=1).sqrt().mean()

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, mode={self.MODE}, "
                f"dc_removal={self._use_dc}")
