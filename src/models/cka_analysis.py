"""
CKA Spectral Analysis Utilities
=================================
Implements Centered Kernel Alignment (CKA) computation and effective rank
analysis for validating the CTD (Critical Training Data) Threshold Hypothesis.

The CTD hypothesis posits that there exists a critical data quantity threshold
below which deep channels cannot develop specialized semantic functions.
CKA spectral analysis measures channel specialization by analyzing the
similarity structure of feature representations.

Key Functions:
--------------
- compute_cka: Compute CKA similarity between two feature matrices
- effective_rank: Compute effective rank from singular value spectrum
- extract_features: Extract intermediate features from ResNet stages
- CKAFeatureExtractor: Hook-based feature extractor for PyTorch models

Reference:
----------
- Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019
- Roy & Vetterli, "The Effective Rank: A Measure of Effective Dimensionality", EUSIPCO 2007
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Centered Kernel Alignment (CKA)
# ============================================================================

def _gram_linear(x: torch.Tensor) -> torch.Tensor:
    """Compute linear Gram matrix: G = X @ X^T.

    Args:
        x: Feature matrix of shape (N, D).

    Returns:
        Gram matrix of shape (N, N).
    """
    return torch.mm(x, x.t())


def _gram_rbf(
    x: torch.Tensor,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Compute RBF (Gaussian) Gram matrix.

    Args:
        x: Feature matrix of shape (N, D).
        threshold: Fraction of median distance for sigma.

    Returns:
        RBF Gram matrix of shape (N, N).
    """
    # Compute pairwise squared distances
    dot_product = torch.mm(x, x.t())
    sq_norms = torch.diag(dot_product)
    sq_distances = -2.0 * dot_product + sq_norms[:, None] + sq_norms[None, :]
    sq_distances = torch.clamp(sq_distances, min=0.0)

    # Use median heuristic for sigma
    mask = torch.eye(x.size(0), device=x.device, dtype=torch.bool)
    sq_distances_no_diag = sq_distances[~mask]
    if sq_distances_no_diag.numel() > 0:
        sigma = torch.median(sq_distances_no_diag) * threshold
        sigma = torch.clamp(sigma, min=1e-8)
    else:
        sigma = torch.tensor(1.0, device=x.device)

    gram = torch.exp(-sq_distances / (2.0 * sigma))
    return gram


def _center_gram(gram: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix: H @ G @ H where H = I - (1/n)11^T.

    Args:
        gram: Gram matrix of shape (N, N).

    Returns:
        Centered Gram matrix.
    """
    n = gram.size(0)
    H = torch.eye(n, device=gram.device) - (1.0 / n) * torch.ones(
        (n, n), device=gram.device
    )
    centered = H @ gram @ H
    return centered


def _hsic(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Compute Hilbert-Schmidt Independence Criterion (HSIC).

    HSIC = (1/(n-1)^2) * trace(K @ H @ L @ H)

    Args:
        gram_x: Centered Gram matrix of X.
        gram_y: Centered Gram matrix of Y.

    Returns:
        HSIC scalar value.
    """
    n = gram_x.size(0)
    hsic_val = torch.trace(gram_x @ gram_y) / ((n - 1) ** 2)
    return hsic_val


def compute_cka(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: str = 'linear',
    threshold: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """Compute Centered Kernel Alignment between two feature representations.

    CKA measures the similarity between two sets of representations by
    comparing their centered kernel matrices.

    CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

    where K and L are the centered Gram matrices of X and Y respectively.

    Args:
        x: Feature matrix of shape (N, D1).
        y: Feature matrix of shape (N, D2).
        kernel: Kernel type — 'linear' or 'rbf'. Default: 'linear'.
        threshold: RBF kernel threshold. Default: 1.0.
        eps: Small constant for numerical stability. Default: 1e-8.

    Returns:
        CKA similarity score in [0, 1]. Higher = more similar representations.

    Example:
        >>> f1 = torch.randn(100, 256)
        >>> f2 = torch.randn(100, 512)
        >>> cka = compute_cka(f1, f2)
        >>> print(f"CKA similarity: {cka:.4f}")
    """
    # Flatten spatial dimensions if present
    if x.dim() > 2:
        x = x.reshape(x.size(0), -1)
    if y.dim() > 2:
        y = y.reshape(y.size(0), -1)

    # Ensure float
    x = x.float()
    y = y.float()

    # Compute Gram matrices
    if kernel == 'linear':
        gram_x = _gram_linear(x)
        gram_y = _gram_linear(y)
    elif kernel == 'rbf':
        gram_x = _gram_rbf(x, threshold)
        gram_y = _gram_rbf(y, threshold)
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Use 'linear' or 'rbf'.")

    # Center
    gram_x = _center_gram(gram_x)
    gram_y = _center_gram(gram_y)

    # Compute CKA
    hsic_xy = _hsic(gram_x, gram_y)
    hsic_xx = _hsic(gram_x, gram_x)
    hsic_yy = _hsic(gram_y, gram_y)

    denominator = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=eps))
    cka_val = hsic_xy / denominator

    return cka_val.item()


# ============================================================================
# Pairwise CKA Matrix
# ============================================================================

def compute_cka_matrix(
    features: Dict[str, torch.Tensor],
    kernel: str = 'linear',
) -> np.ndarray:
    """Compute pairwise CKA matrix between multiple feature sets.

    Args:
        features: Dictionary mapping layer names to feature tensors (N, *).
        kernel: Kernel type. Default: 'linear'.

    Returns:
        (L, L) CKA similarity matrix where L = len(features).
    """
    layer_names = list(features.keys())
    n_layers = len(layer_names)
    cka_matrix = np.zeros((n_layers, n_layers))

    for i in range(n_layers):
        for j in range(n_layers):
            cka_matrix[i, j] = compute_cka(
                features[layer_names[i]],
                features[layer_names[j]],
                kernel=kernel,
            )

    return cka_matrix


# ============================================================================
# Effective Rank
# ============================================================================

def effective_rank(
    feature_matrix: torch.Tensor,
    threshold: float = 0.99,
    eps: float = 1e-8,
) -> Tuple[int, float, np.ndarray]:
    """Compute the effective rank from singular value decomposition.

    The effective rank is defined as the minimum number of singular values
    needed to explain `threshold` fraction of the total variance.

    For the CTD hypothesis: a LOW effective rank indicates that channels
    are NOT developing specialized functions (they collapse to a few modes).
    A HIGH effective rank indicates diverse, specialized channel functions.

    Mathematical definition:
        eff_rank = min{k : sum(σ_i^2 for i ≤ k) / sum(σ_i^2 for all i) ≥ threshold}

    Args:
        feature_matrix: Feature matrix of shape (N, D) or (D, N).
        threshold: Variance explained threshold. Default: 0.99.
        eps: Small constant for numerical stability. Default: 1e-8.

    Returns:
        Tuple of:
        - effective_rank (int): Number of significant singular values.
        - ratio (float): eff_rank / total_rank.
        - singular_values (np.ndarray): Full sorted singular value array.

    Example:
        >>> f = torch.randn(100, 512)
        >>> eff_rank, ratio, sv = effective_rank(f)
        >>> print(f"Effective rank: {eff_rank}, Ratio: {ratio:.3f}")
    """
    # Ensure 2D
    if feature_matrix.dim() > 2:
        feature_matrix = feature_matrix.reshape(feature_matrix.size(0), -1)

    # Transpose so that we analyze channel (feature) dimension
    # We want N x D where D = channels; SVD on (N, D)
    X = feature_matrix.float().numpy()

    # Compute SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Normalize singular values to get variance explained
    s_squared = S ** 2
    total_variance = s_squared.sum()
    if total_variance < eps:
        return 0, 0.0, S

    cumulative = np.cumsum(s_squared) / total_variance

    # Find effective rank
    eff_rank = int(np.searchsorted(cumulative, threshold) + 1)
    eff_rank = min(eff_rank, len(S))

    ratio = eff_rank / len(S) if len(S) > 0 else 0.0

    return eff_rank, ratio, S


# ============================================================================
# Feature Extraction Hooks
# ============================================================================

class CKAFeatureExtractor:
    """Hook-based feature extractor for ResNet models.

    Attaches forward hooks to extract intermediate feature maps from
    specified layers for CKA spectral analysis.

    Usage:
        >>> extractor = CKAFeatureExtractor(model, stages=['layer1', 'layer2'])
        >>> with torch.no_grad():
        ...     _ = model(input_batch)
        >>> features = extractor.get_features()
        >>> extractor.clear()

    Args:
        model: PyTorch model (ResNet variant).
        stages: List of layer names to extract from.
            Default: ['layer1', 'layer2', 'layer3', 'layer4'].
    """

    def __init__(
        self,
        model: nn.Module,
        stages: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.stages = stages or ['layer1', 'layer2', 'layer3', 'layer4']
        self.features: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        self._register_hooks()

    def _hook_fn(self, name: str):
        """Create a hook function for a specific layer name.

        Args:
            name: Layer name for identification.

        Returns:
            Hook function.
        """
        def hook(module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
            self.features[name] = output.detach().cpu()

        return hook

    def _register_hooks(self) -> None:
        """Register forward hooks for specified stages."""
        for name, module in self.model.named_modules():
            for stage in self.stages:
                # Match stage names like 'layer1', 'layer2', etc.
                if name == stage or name.endswith(f'.{stage}'):
                    handle = module.register_forward_hook(self._hook_fn(name))
                    self.handles.append(handle)

    def get_features(self) -> Dict[str, torch.Tensor]:
        """Get extracted features.

        Returns:
            Dictionary mapping layer names to feature tensors.
        """
        return self.features

    def clear(self) -> None:
        """Clear stored features."""
        self.features.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __del__(self) -> None:
        """Clean up hooks on deletion."""
        self.remove_hooks()


# ============================================================================
# High-Level Analysis Functions
# ============================================================================

def extract_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    stages: Optional[List[str]] = None,
    max_batches: int = 10,
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    """Extract features from a model over multiple batches.

    Args:
        model: PyTorch model in eval mode.
        dataloader: DataLoader providing input batches.
        stages: Layer names to extract. Default: ResNet stages.
        max_batches: Maximum number of batches to process.
        device: Device for computation.

    Returns:
        Dictionary mapping layer names to concatenated feature tensors.
    """
    if stages is None:
        stages = ['layer1', 'layer2', 'layer3', 'layer4']

    model.eval()
    extractor = CKAFeatureExtractor(model, stages=stages)

    all_features = {s: [] for s in stages}

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            inputs = inputs.to(device)
            _ = model(inputs)

            for stage in stages:
                if stage in extractor.features:
                    feat = extractor.features[stage]
                    all_features[stage].append(feat)

            extractor.clear()

    extractor.remove_hooks()

    # Concatenate across batches
    result = {}
    for stage, feat_list in all_features.items():
        if feat_list:
            result[stage] = torch.cat(feat_list, dim=0)
        else:
            result[stage] = torch.tensor([])

    return result


def compute_cka_across_depths(
    features: Dict[str, torch.Tensor],
    kernel: str = 'linear',
) -> Tuple[np.ndarray, List[str]]:
    """Compute CKA similarity matrix across network depths.

    Args:
        features: Layer-wise feature dictionary from extract_features.
        kernel: Kernel type. Default: 'linear'.

    Returns:
        Tuple of (CKA matrix, ordered layer names).
    """
    # Keep layers in consistent order
    ordered_names = sorted(
        features.keys(),
        key=lambda x: (x.count('layer'), x)
    )

    return compute_cka_matrix(
        {k: features[k] for k in ordered_names},
        kernel=kernel,
    ), ordered_names


def compute_effective_rank_profile(
    features: Dict[str, torch.Tensor],
    threshold: float = 0.99,
) -> Dict[str, Dict[str, float]]:
    """Compute effective rank for each layer's features.

    Args:
        features: Layer-wise feature dictionary.
        threshold: Variance explained threshold.

    Returns:
        Dictionary mapping layer name to {eff_rank, ratio, total_rank}.
    """
    profile = {}

    for name, feat in features.items():
        # Flatten spatial dims: (N, C, H, W) → (N, C*H*W)
        if feat.dim() == 4:
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.size(1))
        else:
            feat_flat = feat

        eff_r, ratio, sv = effective_rank(feat_flat, threshold)
        profile[name] = {
            'effective_rank': eff_r,
            'ratio': ratio,
            'total_rank': len(sv),
        }

    return profile


if __name__ == "__main__":
    # Quick sanity check
    print("=== CKA Analysis Module Test ===\n")

    # Test CKA computation
    x = torch.randn(100, 256)
    y = torch.randn(100, 512)
    cka_xy = compute_cka(x, y)
    cka_xx = compute_cka(x, x)
    print(f"CKA(X, X) should ≈ 1.0: {cka_xx:.6f}")
    print(f"CKA(X, Y) (random):          {cka_xy:.6f}")

    # Test effective rank
    eff_rank, ratio, _ = effective_rank(x)
    print(f"\nEffective rank (random 256-d): {eff_rank}, ratio: {ratio:.3f}")

    # Low-rank matrix test
    low_rank = torch.randn(100, 10) @ torch.randn(10, 256)
    eff_rank_lr, ratio_lr, _ = effective_rank(low_rank)
    print(f"Effective rank (rank-10):      {eff_rank_lr}, ratio: {ratio_lr:.3f}")
