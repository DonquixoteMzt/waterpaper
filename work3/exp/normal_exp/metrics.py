"""
metrics.py — Geometric feature extraction from loss surfaces.

Computes flatness, roughness, basin width, curvature ratio, and other
geometric metrics from 2D loss surface evaluations.
"""

import numpy as np
from scipy.ndimage import uniform_filter


def compute_surface_metrics(alphas, betas, surface, delta_factor=0.1):
    """
    Extract geometric features from a 2D loss surface.

    Args:
        alphas: 1D array of alpha values
        betas: 1D array of beta values
        surface: 2D array of loss values (shape: [len(betas), len(alphas)])
        delta_factor: fraction of loss range to define basin boundary

    Returns:
        dict of metric name -> value
    """
    grid_size = len(alphas)
    center_idx = grid_size // 2

    # Basic metrics
    center_loss = float(surface[center_idx, center_idx])
    loss_min = float(surface.min())
    loss_max = float(surface.max())
    loss_range = loss_max - loss_min
    loss_mean = float(surface.mean())
    loss_median = float(np.median(surface))

    # Roughness: std of residuals after smoothing
    smoothed = uniform_filter(surface, size=3)
    roughness = float(np.std(surface - smoothed))

    # Quadratic fit residuals (more sophisticated roughness)
    A, B = np.meshgrid(alphas, betas)
    # Flatten for polynomial fitting
    a_flat = A.flatten()
    b_flat = B.flatten()
    s_flat = surface.flatten()
    # Fit f(a,b) = c0 + c1*a + c2*b + c3*a^2 + c4*b^2 + c5*a*b
    design = np.column_stack([
        np.ones_like(a_flat), a_flat, b_flat,
        a_flat**2, b_flat**2, a_flat * b_flat
    ])
    coeffs, residuals, _, _ = np.linalg.lstsq(design, s_flat, rcond=None)
    quad_fit = design @ coeffs
    quad_roughness = float(np.std(s_flat - quad_fit))

    # Curvatures along axes (from quadratic fit)
    # f(a,b) ≈ c0 + c1*a + c2*b + c3*a^2 + c4*b^2 + c5*a*b
    kappa_1 = 2 * coeffs[3]  # d²f/da² = 2*c3
    kappa_2 = 2 * coeffs[4]  # d²f/db² = 2*c4
    curvature_ratio = abs(kappa_1) / (abs(kappa_2) + 1e-10)

    # Basin metrics
    delta = delta_factor * loss_range if loss_range > 0 else 0.1
    basin_mask = surface <= (center_loss + delta)
    basin_area_fraction = float(basin_mask.sum()) / basin_mask.size

    # Effective basin diameter
    grid_spacing_a = (alphas[-1] - alphas[0]) / (len(alphas) - 1) if len(alphas) > 1 else 1.0
    grid_spacing_b = (betas[-1] - betas[0]) / (len(betas) - 1) if len(betas) > 1 else 1.0
    basin_area = float(basin_mask.sum()) * grid_spacing_a * grid_spacing_b
    basin_diameter = 2 * np.sqrt(basin_area / np.pi) if basin_area > 0 else 0.0

    # Basin flatness: mean excess loss within basin
    if basin_mask.sum() > 0:
        basin_flatness = float(np.mean(surface[basin_mask] - center_loss))
    else:
        basin_flatness = 0.0

    # Asymmetry along alpha axis
    n_half = grid_size // 2
    if n_half > 0:
        left = surface[center_idx, :n_half]
        right = surface[center_idx, -n_half:][::-1]
        min_len = min(len(left), len(right))
        asymmetry_alpha = float(np.mean(np.abs(left[:min_len] - right[:min_len])))
    else:
        asymmetry_alpha = 0.0

    # Convexity ratio: fraction of grid with positive local curvature
    if grid_size >= 3:
        d2f_da2 = np.diff(surface, n=2, axis=1)
        d2f_db2 = np.diff(surface, n=2, axis=0)
        # Positive curvature means locally convex
        convex_a = (d2f_da2 > 0).sum()
        convex_b = (d2f_db2 > 0).sum()
        convexity_ratio = float((convex_a + convex_b)) / float(d2f_da2.size + d2f_db2.size)
    else:
        convexity_ratio = 0.0

    return {
        'center_loss': center_loss,
        'loss_min': loss_min,
        'loss_max': loss_max,
        'loss_range': loss_range,
        'loss_mean': loss_mean,
        'loss_median': loss_median,
        'roughness': roughness,
        'quad_roughness': quad_roughness,
        'kappa_1': float(kappa_1),
        'kappa_2': float(kappa_2),
        'curvature_ratio': float(curvature_ratio),
        'basin_area_fraction': basin_area_fraction,
        'basin_diameter': float(basin_diameter),
        'basin_flatness': basin_flatness,
        'asymmetry_alpha': asymmetry_alpha,
        'convexity_ratio': convexity_ratio,
        'grid_range': float(alphas[-1]),
    }


def format_metrics_table(metrics_dict, name=""):
    """
    Format metrics as a printable table string.

    Args:
        metrics_dict: dict from compute_surface_metrics
        name: label for this experiment

    Returns:
        formatted string
    """
    lines = [f"--- Geometric Metrics{' (' + name + ')' if name else ''} ---"]
    for key, val in metrics_dict.items():
        if isinstance(val, float):
            if abs(val) > 1e4 or (0 < abs(val) < 1e-3):
                lines.append(f"  {key:25s}: {val:.4e}")
            else:
                lines.append(f"  {key:25s}: {val:.6f}")
        else:
            lines.append(f"  {key:25s}: {val}")
    return '\n'.join(lines)
