"""
multi_model.py — Multi-Model Shared Projection (MMSP).

Methods for projecting multiple models onto shared 2D coordinate systems:
- Method A: Trajectory-PCA for training checkpoints
- Method B: Anchor-Point for same-architecture model pairs
- Method C: Independent comparison for different architectures
"""

import math
import numpy as np
import torch


def trajectory_pca(checkpoints_params, k=2):
    """
    MMSP Method A: Trajectory-PCA Projection.

    Given a sequence of model parameter snapshots, compute PCA of the
    parameter differences to find the 2D plane capturing most trajectory variance.

    Args:
        checkpoints_params: list of dicts {param_name: tensor}
        k: number of PCA components

    Returns:
        (pca_directions, projected_coords, centroid)
        pca_directions: list of k direction dicts
        projected_coords: list of (x, y) tuples for each checkpoint
        centroid: dict of centroid parameters
    """
    T = len(checkpoints_params)
    param_names = list(checkpoints_params[0].keys())

    # Compute centroid
    centroid = {}
    for name in param_names:
        centroid[name] = sum(cp[name].float() for cp in checkpoints_params) / T

    # Form difference vectors and Gram matrix
    gram = np.zeros((T, T))
    diffs_flat = []
    for t in range(T):
        diff_flat = torch.cat([
            (checkpoints_params[t][name].float() - centroid[name]).flatten()
            for name in param_names
        ])
        diffs_flat.append(diff_flat.cpu())

    for i in range(T):
        for j in range(i, T):
            dot = torch.dot(diffs_flat[i], diffs_flat[j]).item()
            gram[i, j] = dot
            gram[j, i] = dot

    # Eigendecompose Gram matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx[:k]]
    eigenvectors = eigenvectors[:, idx[:k]]

    # Reconstruct d-dimensional PCA directions
    pca_directions = []
    for j in range(k):
        d_flat = torch.zeros_like(diffs_flat[0])
        for t in range(T):
            d_flat += eigenvectors[t, j] * diffs_flat[t]
        d_norm = d_flat.norm()
        if d_norm > 1e-10:
            d_flat /= d_norm

        # Convert to param dict
        direction = {}
        offset = 0
        for name in param_names:
            numel = checkpoints_params[0][name].numel()
            shape = checkpoints_params[0][name].shape
            direction[name] = d_flat[offset:offset + numel].reshape(shape)
            offset += numel
        pca_directions.append(direction)

    # Project each checkpoint
    projected_coords = []
    for t in range(T):
        coords = []
        for j in range(k):
            proj = sum(
                ((checkpoints_params[t][name].float() - centroid[name]).flatten() @
                 pca_directions[j][name].flatten()).item()
                for name in param_names
            )
            coords.append(proj)
        projected_coords.append(tuple(coords))

    # Explained variance
    total_var = max(np.sum(np.maximum(np.sort(np.linalg.eigvalsh(gram))[::-1], 0)), 1e-10)
    explained_var = [max(0, eigenvalues[j]) / total_var for j in range(k)]

    return pca_directions, projected_coords, centroid, explained_var


def anchor_point_projection(params_a, params_b, params_c=None):
    """
    MMSP Method B: Anchor-Point Projection.

    For 2 models: d1 = theta_B - theta_A, d2 = random orthogonal direction.
    For 3 models: d1 = theta_B - theta_A, d2 = Gram-Schmidt of theta_C - theta_A.

    Args:
        params_a, params_b: model parameter dicts
        params_c: optional third model parameter dict

    Returns:
        (d1, d2, midpoint): direction dicts and midpoint parameters
    """
    param_names = list(params_a.keys())

    # Direction 1: B - A
    d1 = {}
    for name in param_names:
        d1[name] = (params_b[name].float() - params_a[name].float())

    # Compute distance
    dist = math.sqrt(sum((d1[name] ** 2).sum().item() for name in d1))

    # Midpoint
    midpoint = {}
    for name in param_names:
        midpoint[name] = (params_a[name].float() + params_b[name].float()) / 2

    if params_c is not None:
        # Direction 2: Gram-Schmidt of C - A orthogonal to d1
        d2_raw = {}
        for name in param_names:
            d2_raw[name] = (params_c[name].float() - params_a[name].float())

        # Project out d1 component
        d1_flat = torch.cat([d1[n].flatten() for n in d1])
        d2_flat = torch.cat([d2_raw[n].flatten() for n in d2_raw])
        proj = (d2_flat @ d1_flat) / (d1_flat @ d1_flat + 1e-10)
        d2_flat = d2_flat - proj * d1_flat

        offset = 0
        d2 = {}
        for name in param_names:
            numel = d1[name].numel()
            d2[name] = d2_flat[offset:offset + numel].reshape(d1[name].shape)
            offset += numel
    else:
        # Random orthogonal direction
        torch.manual_seed(999)
        d2 = {name: torch.randn_like(d1[name]) for name in d1}
        d1_flat = torch.cat([d1[n].flatten() for n in d1])
        d2_flat = torch.cat([d2[n].flatten() for n in d2])
        d2_flat = d2_flat - (d2_flat @ d1_flat) / (d1_flat @ d1_flat + 1e-10) * d1_flat
        offset = 0
        for name in d2:
            numel = d2[name].numel()
            d2[name] = d2_flat[offset:offset + numel].reshape(d2[name].shape)
            offset += numel

    return d1, d2, midpoint, dist


def compute_model_distance(params_a, params_b):
    """
    Compute L2 distance between two models in parameter space.

    Args:
        params_a, params_b: model parameter dicts

    Returns:
        float: L2 distance
    """
    dist_sq = 0.0
    for name in params_a:
        if name in params_b:
            diff = params_a[name].float() - params_b[name].float()
            dist_sq += (diff ** 2).sum().item()
    return math.sqrt(dist_sq)
