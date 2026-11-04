"""C4 orbit-pooled Cosine Prototype Convolution.

Motivation (theory, see THEORY_2026-11-01.md)
--------------------------------------------
Bordelon, Atanasov & Pehlevan (J. Stat. Mech. 2025) show that for targets lying in
the RKHS of the initial NTK, feature learning and the kernel regime share the same
scaling exponent: architectural changes to the *matching rule* can only alter the
rate, not the attainable error.  Every intervention we tried on the operator
(partial normalisation, per-channel temperature, learnable alpha, local prototype
read-out) failed in exactly that way -- it moved convergence, never the ceiling.

What does lower the attainable error under a fixed sample budget is a smaller
hypothesis class consistent with a *true* symmetry of the task.  Chen, Katsoulakis,
Rey-Bellet & Zhu (SIAM J. Imaging Sci. / arXiv 2305.13517, 2025) prove that for a
group of size |G| the sample requirement drops by a factor of |G|, and that this
gain cannot be reproduced by data augmentation alone.  Linmans et al. (GU-Net)
report the corresponding empirical factor of 8x on segmentation.

A tea leaf photographed on a bench has no canonical orientation: the label is
invariant to in-plane rotation.  The published pipeline uses only a horizontal
flip, so every orientation must be learned from data.  This module hard-wires the
cyclic group C4 = {0deg, 90deg, 180deg, 270deg}.

Design
------
    y_c(t) = tau * max_{k in C4}  cos( R_k p_c , x(t) )

* R_k p_c rotates the prototype inside the kernel plane, so the orbit is *generated*
  from the same tensor.  The parameter count is therefore identical to CPCConv2d
  and to the equivalent Conv2d -- only compute grows (x4).
* Taking the max over the orbit makes the response invariant to which member of the
  orbit matched.  This is the standard group max-pooling used by G-CNNs to convert
  equivariance into invariance.
* mode="mean" replaces max by the orbit average, a smoother (less aggressive)
  invariant that keeps more of the response magnitude.

Only 90-degree rotations are used, because they are the only rotations that map a
(k x k) kernel grid onto itself without resampling.  The group is therefore the
exact analogue of the rot90 augmentation used in the control arm, which makes the
"augmentation vs. equivariance" comparison a strict single-variable test.
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


class CPCConv2dC4(nn.Module):
    """Cosine-prototype convolution with C4 orbit pooling over the prototypes."""

    #: "max" -> invariant (hypothesis class / |G| ), "mean" -> orbit average
    MODE = "max"
    #: which rotations to include: 4 -> C4, 2 -> {0, 180}
    ORDER = 4
    #: learnable scalar temperature (same as the published operator)
    LEARNABLE_TAU = True
    TAU_INIT = 5.0

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, eps=1e-6, **_):
        super().__init__()
        if bias:
            raise ValueError("CPCConv2dC4 does not support bias; set bias=False")
        if groups != 1:
            raise ValueError("CPCConv2dC4 currently supports groups=1 only")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_pair(kernel_size)
        self.stride = _to_pair(stride)
        self.padding = _to_pair(padding)
        self.dilation = _to_pair(dilation)
        self.groups = groups
        self.eps = eps

        # identical parameter shape to CPCConv2d / Conv2d
        dim = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.prototype = nn.Parameter(torch.empty(out_channels, dim))

        tau = torch.tensor(math.log(self.TAU_INIT), dtype=torch.float32)
        if self.LEARNABLE_TAU:
            self.log_tau = nn.Parameter(tau)
        else:
            self.register_buffer("log_tau", tau)
        self.reset_parameters()

    # ------------------------------------------------------------------ init
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.prototype)

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    # ----------------------------------------------------------------- orbit
    def _orbit(self) -> torch.Tensor:
        """Return (G, C_out, D) stacked rotated copies of the prototypes."""
        K_h, K_w = self.kernel_size
        p = self.prototype.view(self.out_channels, self.in_channels, K_h, K_w)
        steps = {4: (0, 1, 2, 3), 2: (0, 2)}[self.ORDER]
        orbit = torch.stack([torch.rot90(p, k, dims=(-2, -1)) for k in steps], dim=0)
        return orbit.reshape(len(steps), self.out_channels, -1)

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _, H, W = x.shape
        K_h, K_w = self.kernel_size
        S_h, S_w = self.stride
        D_h, D_w = self.dilation
        P_h, P_w = self.padding

        x_p = F.pad(x, (P_w, P_w, P_h, P_h)) if (P_h or P_w) else x
        H_out = (H + 2 * P_h - D_h * (K_h - 1) - 1) // S_h + 1
        W_out = (W + 2 * P_w - D_w * (K_w - 1) - 1) // S_w + 1

        patches = F.unfold(x_p, kernel_size=(K_h, K_w), dilation=(D_h, D_w),
                           stride=(S_h, S_w)).permute(0, 2, 1)      # N,L,D
        patches = F.normalize(patches, p=2, dim=2, eps=self.eps)    # N,L,D

        orbit = F.normalize(self._orbit(), p=2, dim=2, eps=self.eps)  # G,C,D
        G = orbit.shape[0]
        orbit_cat = orbit.reshape(G * self.out_channels, -1)          # G*C, D
        cos = torch.matmul(patches, orbit_cat.t())                    # N, L, G*C
        cos = cos.view(N, -1, G, self.out_channels).permute(2, 0, 3, 1)  # G,N,C,L
        resp = cos.max(dim=0).values if self.MODE == "max" else cos.mean(dim=0)

        out = self.tau * resp
        return out.reshape(N, self.out_channels, H_out, W_out)

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, C{self.ORDER}-orbit {self.MODE}, "
                f"tau={self.TAU_INIT if not self.LEARNABLE_TAU else 'learnable'}")
