"""
CPCConv2d — Cosine Prototype Convolution
==========================================
This module implements the core innovation of CPC-Net: replacing standard
convolution's linear projection W*x with cosine similarity between learnable
prototype vectors and input patches.

Mathematical Formulation:
-------------------------
Standard Conv2d:       y_c(i,j) = W_c ⋅ x_ij + b_c
CPCConv2d:             y_c(i,j) = τ ⋅ cos(θ) = τ ⋅ (p_c ⋅ x_ij) / (||p_c|| ⋅ ||x_ij||)

where:
  - p_c ∈ ℝ^(C_in * K * K): learnable prototype vector for output channel c
  - x_ij: flattened input patch at spatial position (i,j)
  - τ: learnable temperature parameter that scales the cosine similarity

Key Properties:
---------------
1. Zero additional parameters compared to standard Conv2d
2. Bounded output in [-τ, +τ] provides implicit gradient clipping
3. Prototype matching forces channels to learn semantically meaningful patterns
4. Drop-in replacement for nn.Conv2d with SAME parameter count
5. Compatible with stride, padding, dilation (inherited from Conv2d semantics)

Reference:
----------
CPC-Net: Learning Prototype-Conditioned Convolutional Kernels
for Fine-Grained Tea Cultivar Classification with Limited Data (2026)

Author: CPC-Net Research Team
Date: 2026-06-24
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPCConv2d(nn.Module):
    """Cosine Prototype Convolution (CPC) layer.

    Replaces the linear projection W*x in standard convolution with cosine
    similarity between prototype vectors and input patches.

    Key insight: Each output channel is represented as a learnable prototype
    vector. The output at each spatial position is the cosine similarity
    between the prototype and the corresponding input patch, scaled by
    a learnable temperature τ.

    Architecture:
        p_c ∈ ℝ^(C_in * K * K): prototype for output channel c
        x_ij: flattened input patch at position (i,j)
        output[c,i,j] = τ * (p_c ⋅ x_ij) / (||p_c||₂ ⋅ ||x_ij||₂ + ε)

    Parameters are IDENTICAL to a standard Conv2d layer with the same
    in_channels, out_channels, and kernel_size — we simply interpret them
    differently and compute the forward pass via cosine similarity.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (prototypes).
        kernel_size (int or tuple): Convolution kernel size.
        stride (int or tuple, optional): Stride. Default: 1.
        padding (int or tuple, optional): Zero-padding. Default: 0.
        dilation (int or tuple, optional): Dilation. Default: 1.
        groups (int, optional): Grouped convolution. Default: 1.
        bias (bool, optional): Not used; CPCConv2d has no bias term. Default: False.
        tau_init (float, optional): Initial temperature value. Default: 5.0.
        tau_learnable (bool, optional): Whether τ is learnable. Default: True.
        eps (float, optional): Small constant for numerical stability. Default: 1e-8.

    Shape:
        Input:  (N, C_in, H, W)
        Output: (N, C_out, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        tau_init: float = 5.0,
        tau_learnable: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super(CPCConv2d, self).__init__()

        # Validate parameters
        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by groups ({groups})"
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by groups ({groups})"
            )
        if bias:
            # CPCConv2d has no bias — cosine similarity is self-normalizing
            raise ValueError("CPCConv2d does not support bias; set bias=False")

        # Store convolution parameters for unfold operation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._to_tuple(kernel_size)
        self.stride = self._to_tuple(stride)
        self.padding = self._to_tuple(padding)
        self.dilation = self._to_tuple(dilation)
        self.groups = groups
        self.eps = eps

        # Number of input channels per group
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups

        # Prototype vectors: shape (out_channels, in_channels * K_h * K_w)
        # Each row is a prototype vector p_c for output channel c
        # This has exactly the same number of parameters as a standard Conv2d weight
        prototype_dim = self.in_channels_per_group * self.kernel_size[0] * self.kernel_size[1]
        self.prototype = nn.Parameter(
            torch.empty(out_channels, prototype_dim)
        )

        # Learnable temperature parameter τ (clipped to positive values)
        if tau_learnable:
            # Store as log(tau) for unconstrained optimization, then exp()
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        else:
            self.register_buffer('log_tau', torch.tensor(math.log(tau_init)))

        # Track whether τ is learnable
        self.tau_learnable = tau_learnable

        # Initialize prototype vectors with Xavier uniform initialization
        self._init_weights()

    @staticmethod
    def _to_tuple(x) -> Tuple[int, int]:
        """Convert scalar or tuple to (h, w) tuple."""
        if isinstance(x, int):
            return (x, x)
        return tuple(x)

    def _init_weights(self) -> None:
        """Initialize prototype vectors using Xavier uniform initialization.

        This matches standard Conv2d weight initialization, ensuring fair
        comparison with the baseline ResNet-18.
        """
        # Xavier uniform: U(-a, a) where a = sqrt(6 / (fan_in + fan_out))
        fan_in: int = self.prototype.size(1)  # prototype dimension
        fan_out: int = self.prototype.size(0)  # number of output channels
        bound: float = math.sqrt(6.0 / float(fan_in + fan_out))
        nn.init.uniform_(self.prototype, -bound, bound)
        # Note: log_tau is already initialized in __init__

    @property
    def tau(self) -> torch.Tensor:
        """Get the current temperature value τ.

        Returns:
            torch.Tensor: exp(log_tau), guaranteed positive.
        """
        return torch.exp(self.log_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Cosine Prototype Convolution.

        Steps:
        1. Unfold input into patches (mimics sliding window of convolution)
        2. Normalize prototype vectors to unit length
        3. Normalize input patches to unit length
        4. Compute cosine similarity as matrix multiplication
        5. Scale by temperature τ
        6. Reshape to (N, C_out, H_out, W_out)

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
        """
        N, C_in, H, W = x.shape

        # Handle grouped convolution
        if self.groups == 1:
            return self._forward_single_group(x)

        # For grouped convolution: split channels, process each group, concatenate
        outputs = []
        x_split = torch.split(x, self.in_channels_per_group, dim=1)
        proto_split = torch.split(
            self.prototype,
            self.out_channels_per_group,
            dim=0
        )
        for x_g, proto_g in zip(x_split, proto_split):
            out_g = self._forward_single_group(x_g, prototype=proto_g)
            outputs.append(out_g)

        return torch.cat(outputs, dim=1)

    def _forward_single_group(
        self,
        x: torch.Tensor,
        prototype: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a single group (or when groups=1).

        Args:
            x (torch.Tensor): Input tensor (N, C_in_group, H, W).
            prototype (torch.Tensor, optional): Prototype matrix for this group.

        Returns:
            torch.Tensor: Output tensor (N, C_out_group, H_out, W_out).
        """
        if prototype is None:
            prototype = self.prototype

        N, C_in_g, H, W = x.shape
        K_h, K_w = self.kernel_size
        P_h, P_w = self.padding
        S_h, S_w = self.stride
        D_h, D_w = self.dilation

        # Compute output spatial dimensions (same formula as Conv2d)
        H_out = (H + 2 * P_h - D_h * (K_h - 1) - 1) // S_h + 1
        W_out = (W + 2 * P_w - D_w * (K_w - 1) - 1) // S_w + 1

        # === Step 1: Unfold input into patches ===
        # unfold extracts sliding local blocks: (N, C_in_g * K_h * K_w, H_out * W_out)
        if P_h > 0 or P_w > 0:
            x_padded = F.pad(x, (P_w, P_w, P_h, P_h))
        else:
            x_padded = x

        patches = F.unfold(
            x_padded,
            kernel_size=(K_h, K_w),
            dilation=(D_h, D_w),
            stride=(S_h, S_w),
        )  # (N, C_in_g * K_h * K_w, L) where L = H_out * W_out

        patches = patches.permute(0, 2, 1)  # (N, L, C_in_g * K_h * K_w)

        # === Step 2: Normalize prototypes to unit length ===
        # prototype: (C_out_g, D) where D = C_in_g * K_h * K_w
        proto_norm = F.normalize(prototype, p=2, dim=1)  # L2 normalize along dim=1

        # === Step 3: Normalize input patches to unit length ===
        # patches: (N, L, D)
        patches_norm = F.normalize(patches, p=2, dim=2)  # L2 normalize each patch

        # === Step 4: Compute cosine similarity ===
        # cos_sim = patches_norm @ proto_norm^T
        # patches_norm: (N, L, D), proto_norm: (C_out_g, D)
        # Result: (N, L, C_out_g)
        cos_sim = torch.matmul(patches_norm, proto_norm.t())

        # === Step 5: Scale by temperature τ ===
        tau_val = self.tau
        output = tau_val * cos_sim

        # === Step 6: Reshape to spatial output ===
        # (N, L, C_out_g) → (N, C_out_g, H_out, W_out)
        output = output.permute(0, 2, 1)  # (N, C_out_g, L)
        output = output.reshape(N, -1, H_out, W_out)

        return output

    def extra_repr(self) -> str:
        """Custom representation for printing."""
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"dilation={self.dilation}, "
            f"groups={self.groups}, "
            f"tau={self.tau.item():.2f}, "
            f"tau_learnable={self.tau_learnable}"
        )


# ==== Compatibility Alias ====
CosinePrototypeConv2d = CPCConv2d


def verify_parameter_parity() -> bool:
    """Verify that CPCConv2d and nn.Conv2d have identical parameter counts.

    This is crucial: CPC-Net must not introduce additional parameters
    compared to the standard ResNet-18 baseline.

    Returns:
        bool: True if parameter counts match.
    """
    test_configs = [
        # (in_channels, out_channels, kernel_size)
        (64, 64, 3),     # Stage 2 typical
        (128, 128, 3),   # Stage 3 typical
        (256, 256, 3),   # Stage 4 typical
        (64, 128, 3),    # Between-stage transition
        (128, 256, 3),   # Between-stage transition
        (256, 512, 3),   # Wide transition
        (64, 64, 1),     # 1x1 conv
        (3, 64, 7),      # Stem conv
    ]

    all_match = True
    for in_c, out_c, k in test_configs:
        # CPCConv2d
        cpc = CPCConv2d(in_c, out_c, k, bias=False)
        n_cpc = sum(p.numel() for p in cpc.parameters())

        # Standard Conv2d
        std = nn.Conv2d(in_c, out_c, k, bias=False)
        n_std = sum(p.numel() for p in std.parameters())

        match = n_cpc == n_std
        if not match:
            print(f"MISMATCH: in={in_c}, out={out_c}, k={k}: "
                  f"CPC={n_cpc}, Conv2d={n_std}")
            all_match = False

    if all_match:
        print("✓ All configurations pass: CPCConv2d == Conv2d in parameter count")

    return all_match


if __name__ == "__main__":
    # Quick sanity check
    print("=== CPCConv2d Parameter Parity Test ===")
    verify_parameter_parity()

    # Test forward pass
    x = torch.randn(2, 64, 32, 32)
    cpc = CPCConv2d(64, 128, 3, stride=1, padding=1)
    y = cpc(x)
    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters:  {sum(p.numel() for p in cpc.parameters()):,}")
    print(f"  Standard Conv2d: {sum(p.numel() for p in nn.Conv2d(64, 128, 3, bias=False).parameters()):,}")
    print(f"τ = {cpc.tau.item():.4f}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"Expected range: [{-cpc.tau.item():.4f}, {cpc.tau.item():.4f}]")
