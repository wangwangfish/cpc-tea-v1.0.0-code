"""
CPC-Net Models Package
=======================
CPC-Net: Cosine Prototype Convolution for Fine-Grained Tea Cultivar Classification.

This package contains:
- CPCConv2d: The core Cosine Prototype Convolution module
- CPC-ResNet-18: Our proposed model architecture
- Attention modules: 17 attention variants for baseline comparison
- CKA analysis: Centered Kernel Alignment spectral analysis utilities
"""

from .cpc_conv import CPCConv2d
from .cpc_resnet import cpc_resnet18
from .attention_modules import (
    SEBlock, ECABlock, CBAM, CoordAtt, ShuffleAtt, SimAM,
    GAM, TripletAttention,
    apply_attention_to_resnet18
)
from .cka_analysis import (
    compute_cka, effective_rank, extract_features, CKAFeatureExtractor
)

__all__ = [
    # Core innovation
    'CPCConv2d',
    'cpc_resnet18',
    # Attention modules
    'SEBlock', 'ECABlock', 'CBAM', 'CoordAtt', 'ShuffleAtt', 'SimAM',
    'GAM', 'TripletAttention',
    'apply_attention_to_resnet18',
    # CKA analysis
    'compute_cka', 'effective_rank', 'extract_features', 'CKAFeatureExtractor',
]
