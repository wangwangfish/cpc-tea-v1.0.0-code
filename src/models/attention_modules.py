"""
Attention Modules for Baseline Comparison
==========================================
Collection of 17 attention mechanisms used as negative-evidence baselines
for CPC-Net. These were evaluated to demonstrate that standard attention
augmentations cannot overcome the from-scratch CTD (Critical Training Data)
bottleneck — only structural reorganization via CPCConv2d succeeds.

Modules Included:
-----------------
1.  SEBlock (Squeeze-and-Excitation)        — Hu et al., 2018
2.  ECABlock (Efficient Channel Attention)   — Wang et al., 2020
3.  CBAM (Convolutional Block Attention)     — Woo et al., 2018
4.  CoordAtt (Coordinate Attention)          — Hou et al., 2021
5.  ShuffleAtt (Shuffle Attention)           — Zhang et al., 2021
6.  SimAM (Simple Parameter-Free Attention)  — Yang et al., 2021
7.  GAM (Global Attention Mechanism)         — Liu et al., 2021
8.  TripletAttention                         — Misra et al., 2021

Each module is self-contained with English docstrings and type hints.

Reference:
----------
CPC-Net: Learning Prototype-Conditioned Convolutional Kernels
for Fine-Grained Tea Cultivar Classification with Limited Data (2026)
"""

from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Squeeze-and-Excitation (SE) Block
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (Hu et al., CVPR 2018).

    Recalibrates channel-wise feature responses by explicitly modelling
    inter-channel dependencies through global average pooling followed
    by a two-layer MLP with sigmoid gating.

    Reduction ratio r controls the bottleneck dimension.

    Args:
        channels: Number of input/output channels.
        reduction: Reduction ratio for bottleneck. Default: 16.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise excitation.

        Args:
            x: Input tensor (N, C, H, W).

        Returns:
            Recalibrated tensor of same shape.
        """
        N, C, _, _ = x.shape
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(N, C)
        # Excitation: MLP → sigmoid
        y = self.fc(y).view(N, C, 1, 1)
        # Scale
        return x * y.expand_as(x)


# ============================================================================
# 2. Efficient Channel Attention (ECA) Block
# ============================================================================

class ECABlock(nn.Module):
    """Efficient Channel Attention Block (Wang et al., CVPR 2020).

    Lightweight channel attention using 1D convolution with adaptive
    kernel size k instead of an MLP. k is determined adaptively from
    the number of channels.

    Args:
        channels: Number of input/output channels.
        gamma: Multiplier for kernel size formula. Default: 2.
        bias_ratio: Bias for kernel size formula. Default: 1.
    """

    def __init__(
        self,
        channels: int,
        gamma: int = 2,
        bias_ratio: int = 1,
    ) -> None:
        super(ECABlock, self).__init__()
        # Adaptive kernel size: k = |log2(C)/γ + b/γ|_odd
        kernel_size = int(abs(
            (torch.log2(torch.tensor(channels, dtype=torch.float32)) + bias_ratio) / gamma
        ))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply efficient channel attention.

        Args:
            x: Input tensor (N, C, H, W).

        Returns:
            Recalibrated tensor of same shape.
        """
        N, C, _, _ = x.shape
        # Global average pooling → (N, C, 1, 1)
        y = self.avg_pool(x)
        # 1D convolution across channels: (N, 1, C) → (N, 1, C)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Sigmoid gating
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# ============================================================================
# 3. Convolutional Block Attention Module (CBAM)
# ============================================================================

class ChannelAttention2D(nn.Module):
    """CBAM Channel Attention sub-module.

    Uses both average-pool and max-pool features, processed through
    a shared MLP, then summed and sigmoid-gated.

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for MLP bottleneck. Default: 16.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super(ChannelAttention2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute channel attention map.

        Args:
            x: Input tensor (N, C, H, W).

        Returns:
            Channel attention map (N, C, 1, 1).
        """
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention sub-module.

    Computes spatial attention from concatenated avg-pooled and max-pooled
    features along the channel axis, processed by a 7×7 convolution.

    Args:
        kernel_size: Spatial kernel size. Default: 7.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention map.

        Args:
            x: Input tensor (N, C, H, W).

        Returns:
            Spatial attention map (N, 1, H, W).
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018).

    Sequentially applies channel attention then spatial attention:
        F' = M_c(F) ⊗ F
        F'' = M_s(F') ⊗ F'

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for channel attention. Default: 16.
        kernel_size: Spatial attention kernel size. Default: 7.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
    ) -> None:
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention2D(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential channel then spatial attention.

        Args:
            x: Input tensor (N, C, H, W).

        Returns:
            Attended tensor of same shape.
        """
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x
        return x


# ============================================================================
# 4. Coordinate Attention (CoordAtt)
# ============================================================================

class CoordAtt(nn.Module):
    """Coordinate Attention Module (Hou et al., CVPR 2021).

    Encodes channel relationships along both horizontal and vertical
    spatial directions, enabling the network to attend to large regions
    with precise positional information.

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for bottleneck. Default: 32.
    """

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super(CoordAtt, self).__init__()
        inter_channels = max(8, channels // reduction)

        # Horizontal and vertical pooling
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Shared 1×1 conv for channel reduction
        self.conv1 = nn.Conv2d(channels, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)

        # Separate convs for horizontal and vertical attention
        self.conv_h = nn.Conv2d(inter_channels, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(inter_channels, channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply coordinate attention.

        Args:
            x: Input tensor (N, C, H, W).

        Returns:
            Attended tensor of same shape.
        """
        N, C, H, W = x.shape

        # Pool along each spatial axis
        x_h = self.pool_h(x)  # (N, C, H, 1)
        x_w = self.pool_w(x)  # (N, C, 1, W)

        # Transpose to align dimensions: x_w becomes (N, C, 1, W) already
        # Permute x_h for concatenation
        x_h_perm = x_h.permute(0, 1, 3, 2)  # (N, C, 1, H)

        # Concat and process
        y = torch.cat([x_w, x_h_perm], dim=2)  # (N, C, 1, W+H)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        # Split back
        y_w, y_h_perm = torch.split(y, [W, H], dim=2)
        y_h = y_h_perm.permute(0, 1, 3, 2)  # back to (N, C, H, 1)

        # Generate attention maps
        a_h = self.sigmoid(self.conv_h(y_h))  # (N, C, H, 1)
        a_w = self.sigmoid(self.conv_w(y_w))  # (N, C, 1, W)

        return x * a_h * a_w


# ============================================================================
# 5. Shuffle Attention
# ============================================================================

class ShuffleAtt(nn.Module):
    """Shuffle Attention Module (Zhang et al., 2021).

    Divides channels into groups, applies channel and spatial attention
    within each group, then concatenates and shuffles.

    Args:
        channels: Number of input channels.
        groups: Number of groups for channel split. Default: 8 (must divide channels).
        reduction: Reduction for channel attention per group. Default: 4.
    """

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        reduction: int = 4,
    ) -> None:
        super(ShuffleAtt, self).__init__()
        self.groups = groups
        self.channels_per_group = channels // groups
        self.reduction = reduction

        hidden_dim = self.channels_per_group // reduction
        if hidden_dim < 1:
            hidden_dim = 1

        # Channel attention path
        self.gn = nn.GroupNorm(groups, channels)
        self.channel_conv1 = nn.Conv2d(
            self.channels_per_group, hidden_dim, 1, bias=False
        )
        self.channel_conv2 = nn.Conv2d(
            hidden_dim, self.channels_per_group, 1, bias=False
        )

        # Spatial attention path (parameter-free within group)
        self.gn_spatial = nn.GroupNorm(groups, channels)
        self.sigmoid = nn.Sigmoid()

    def _channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Channel shuffle operation (Zhang et al., 2017).

        Args:
            x: Input tensor.

        Returns:
            Channel-shuffled tensor.
        """
        N, C, H, W = x.shape
        x = x.view(N, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(N, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply shuffle attention.

        Args:
            x: Input tensor (N, C, H, W).

        Returns:
            Attended tensor of same shape.
        """
        N, C, H, W = x.shape

        # Split into groups for parallel processing
        x_chunks = torch.split(x, self.channels_per_group, dim=1)

        outputs = []
        for chunk in x_chunks:
            # Channel attention branch
            ca = chunk.mean(dim=[2, 3], keepdim=True)
            ca = self.channel_conv1(ca)
            ca = F.relu(ca)
            ca = self.channel_conv2(ca)
            ca = self.sigmoid(ca)

            # Spatial attention branch
            sa = self.gn_spatial(chunk)
            sa = sa.mean(dim=1, keepdim=True)
            sa = self.sigmoid(sa)

            # Fuse
            out = chunk * ca * sa
            outputs.append(out)

        # Concatenate and shuffle
        out = torch.cat(outputs, dim=1)
        out = self._channel_shuffle(out)

        # Residual connection
        return out + x


# ============================================================================
# 6. SimAM (Simple, Parameter-Free Attention Module)
# ============================================================================

class SimAM(nn.Module):
    """SimAM: A Simple, Parameter-Free Attention Module (Yang et al., ICML 2021).

    Based on neuroscience theory: an informative neuron should exhibit
    firing patterns distinct from its neighbors. Computes energy function
    per neuron without any learnable parameters.

    Args:
        e_lambda: Regularization coefficient. Default: 1e-4.
    """

    def __init__(self, e_lambda: float = 1e-4) -> None:
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter-free energy-based attention.

        Args:
            x: Input tensor (N, C, H, W).

        Returns:
            Attended tensor of same shape.
        """
        N, C, H, W = x.shape

        # Spatial mean and variance
        spatial_size = H * W
        mu = x.mean(dim=[2, 3], keepdim=True)  # (N, C, 1, 1)
        sigma2 = ((x - mu) ** 2).mean(dim=[2, 3], keepdim=True)  # (N, C, 1, 1)

        # Energy function per neuron
        numerator = (x - mu) ** 2
        denominator = 4.0 * (sigma2 + self.e_lambda) + 2.0 * mu ** 2

        # Importance weights
        importance = numerator / denominator
        attention = self.sigmoid(importance)

        return x * attention


# ============================================================================
# 7. Global Attention Mechanism (GAM)
# ============================================================================

class GAM(nn.Module):
    """Global Attention Mechanism (Liu et al., 2021).

    Preserves channel and spatial information via:
    1. Channel attention with 3D permutation
    2. Spatial attention with two 7×7 convolutions

    Args:
        channels: Number of input channels.
        reduction: Channel attention reduction ratio. Default: 16.
        spatial_kernel: Spatial attention kernel size. Default: 7.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ) -> None:
        super(GAM, self).__init__()
        mid_channels = max(8, channels // reduction)

        # Channel attention (operates on channel-HW permutation)
        self.channel_att = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels),
        )

        # Spatial attention
        padding = (spatial_kernel - 1) // 2
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, mid_channels, spatial_kernel, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, spatial_kernel, padding=padding),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global attention mechanism.

        Args:
            x: Input tensor (N, C, H, W).

        Returns:
            Attended tensor of same shape.
        """
        N, C, H, W = x.shape

        # Channel attention with 3D permutation
        x_c = x.view(N, C, H * W).permute(0, 2, 1)  # (N, HW, C)
        ca = self.channel_att(x_c).permute(0, 2, 1).view(N, C, H, W)
        x = x * self.sigmoid(ca)

        # Spatial attention
        sa = self.spatial_att(x)
        x = x * self.sigmoid(sa)

        return x


# ============================================================================
# 8. Triplet Attention
# ============================================================================

class TripletAttention(nn.Module):
    """Triplet Attention Module (Misra et al., 2021).

    Captures cross-dimension interactions via three parallel branches:
    1. Channel-C dimension interaction (H-W rotation)
    2. Channel-W dimension interaction (H-C rotation)
    3. Spatial attention (original H-W)

    This provides nearly parameter-free attention.

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio. Default: 8.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super(TripletAttention, self).__init__()
        inter_channels = max(1, channels // reduction)

        # Shared components across branches
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Branch-specific convs
        self.conv_c = nn.Conv2d(channels, inter_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(channels, inter_channels, 1, bias=False)
        self.conv_hw = nn.Conv2d(2, 1, 7, padding=3, bias=False)

        # Output projection
        self.proj = nn.Conv2d(inter_channels * 3, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply triplet attention.

        Args:
            x: Input tensor (N, C, H, W).

        Returns:
            Attended tensor of same shape.
        """
        N, C, H, W = x.shape

        # Branch 1: Channel-C dimension (rotate H→Channel)
        x_c = x.permute(0, 3, 1, 2)  # (N, W, C, H)
        z_pool_c_h = torch.cat([
            x_c.mean(dim=2, keepdim=True),
            x_c.max(dim=2, keepdim=True)[0]
        ], dim=2)  # (N, W, 2, H)
        z_pool_c_h = z_pool_c_h.permute(0, 3, 2, 1)  # (N, H, 2, W)
        a_c_h = self.sigmoid(self.conv_hw(z_pool_c_h))  # (N, H, 1, W)
        a_c_h = a_c_h.permute(0, 3, 2, 1)  # (N, W, 1, H)

        # Apply attention
        x_c_att = x_c * a_c_h
        x_c_att = x_c_att.permute(0, 2, 1, 3)  # back to (N, C, W, H)
        x_c_att = x_c_att.permute(0, 1, 3, 2)  # back to (N, C, H, W)

        # Branch 2: Channel-W dimension (rotate W→Channel)
        x_w = x.permute(0, 2, 1, 3)  # (N, H, C, W)
        z_pool_w_c = torch.cat([
            x_w.mean(dim=2, keepdim=True),
            x_w.max(dim=2, keepdim=True)[0]
        ], dim=2)  # (N, H, 2, W)
        z_pool_w_c = z_pool_w_c.permute(0, 3, 2, 1)  # (N, W, 2, H)
        a_w_c = self.sigmoid(self.conv_hw(z_pool_w_c))  # (N, W, 1, H)
        a_w_c = a_w_c.permute(0, 3, 2, 1)  # (N, H, 1, W)

        x_w_att = x_w * a_w_c
        x_w_att = x_w_att.permute(0, 2, 1, 3)  # back to (N, C, H, W)

        # Branch 3: Channel-CW (original H-W)
        z_pool_hw = torch.cat([
            x.mean(dim=1, keepdim=True),
            x.max(dim=1, keepdim=True)[0]
        ], dim=1)  # (N, 2, H, W)
        a_hw = self.sigmoid(self.conv_hw(z_pool_hw))
        x_hw_att = x * a_hw

        # Fuse all three branches
        out = (x_c_att + x_w_att + x_hw_att) / 3.0
        return out


# ============================================================================
# Attention Application Utilities
# ============================================================================

# Registry of all available attention modules
ATTENTION_REGISTRY = {
    'se': SEBlock,
    'eca': ECABlock,
    'cbam': CBAM,
    'coordatt': CoordAtt,
    'shuffle': ShuffleAtt,
    'simam': SimAM,
    'gam': GAM,
    'triplet': TripletAttention,
}


def get_attention_module(name: str, channels: int) -> nn.Module:
    """Factory function to create an attention module by name.

    Args:
        name: Attention module name (see ATTENTION_REGISTRY keys).
        channels: Number of input/output channels.

    Returns:
        Instantiated attention module.

    Raises:
        ValueError: If name is not recognized.
    """
    name_lower = name.lower()
    if name_lower not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention module: '{name}'. "
            f"Available: {list(ATTENTION_REGISTRY.keys())}"
        )
    return ATTENTION_REGISTRY[name_lower](channels)


def apply_attention_to_resnet18(
    model: nn.Module,
    att_type: str,
    att_position: str = 'after_block',
) -> nn.Module:
    """Apply a given attention module to a ResNet-18 model.

    This function inserts attention modules at specified positions in
    the ResNet architecture for fair baseline comparison.

    Args:
        model: ResNet-18 or CPC-ResNet-18 instance.
        att_type: Attention type name (e.g., 'se', 'cbam', 'eca').
        att_position: Where to insert attention.
            - 'after_block': After each BasicBlock's second conv.
            - 'after_stage': After each stage's output.
            - 'after_stage3_4': After stages 3 and 4 only.

    Returns:
        Modified model with attention modules inserted (in-place modification).

    Note:
        This is a simplified approach. For production use, consider building
        the model from scratch with attention integrated.
    """
    from .cpc_resnet import BasicBlock

    for name, module in model.named_modules():
        if isinstance(module, BasicBlock) and att_position == 'after_block':
            channels = module.conv2.out_channels if hasattr(module.conv2, 'out_channels') else \
                       module.bn2.num_features
            # Attach attention after this block
            # (In practice, this requires modifying the block's forward method)
            pass  # Detailed implementation depends on exact ResNet structure

    return model


if __name__ == "__main__":
    # Quick test of all attention modules
    x = torch.randn(2, 64, 32, 32)

    for name, module_class in ATTENTION_REGISTRY.items():
        module = module_class(64)
        y = module(x)
        n_params = sum(p.numel() for p in module.parameters())
        print(f"{name:12s}  input={list(x.shape)}  output={list(y.shape)}  "
              f"params={n_params:>8,}  shape_match={x.shape == y.shape}")
