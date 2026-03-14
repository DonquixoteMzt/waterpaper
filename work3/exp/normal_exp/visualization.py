"""
visualization.py — Plotting functions for loss landscape visualization.

Generates 2D contour plots, 3D surface plots, 1D cross-section comparisons,
PFI bar charts, and multi-panel comparison figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_2d_surface(alphas, betas, surface, title, filename,
                    vmin=None, vmax=None, model_positions=None):
    """
    Plot a 2D loss surface as contour + 3D surface.

    Args:
        alphas, betas: 1D grid arrays
        surface: 2D loss values
        title: plot title
        filename: output path
        vmin, vmax: color scale limits
        model_positions: list of (x, y, label) for overlaid points
    """
    fig = plt.figure(figsize=(16, 6))
    A, B = np.meshgrid(alphas, betas)

    if vmin is None:
        vmin = surface.min()
    if vmax is None:
        vmax = min(surface.max(), surface.min() + 3 * (np.median(surface) - surface.min() + 0.1))
    levels = np.linspace(vmin, vmax, 30)

    # Contour plot
    ax1 = fig.add_subplot(121)
    cs = ax1.contourf(A, B, surface, levels=levels, cmap='viridis')
    ax1.contour(A, B, surface, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax1, label='Loss')
    ax1.set_xlabel(r'$\alpha$ (direction 1)')
    ax1.set_ylabel(r'$\beta$ (direction 2)')
    ax1.set_title(f'{title}\n(Contour)')
    ax1.plot(0, 0, 'r*', markersize=15, label=r'$\theta^*$')
    if model_positions:
        for x, y, label in model_positions:
            ax1.plot(x, y, 'o', markersize=10, label=label)
    ax1.legend(fontsize=8)

    # 3D surface
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.plot_surface(A, B, surface, cmap='viridis', alpha=0.8, vmin=vmin, vmax=vmax)
    ax3d.set_xlabel(r'$\alpha$')
    ax3d.set_ylabel(r'$\beta$')
    ax3d.set_zlabel('Loss')
    ax3d.set_title(f'{title}\n(3D Surface)')
    ax3d.view_init(elev=30, azim=225)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_1d_comparison(curves, title, filename):
    """
    Plot multiple 1D loss curves for comparison.

    Args:
        curves: dict {label: (alphas, losses)}
        title: plot title
        filename: output path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (alphas, losses) in curves.items():
        ax.plot(alphas, losses, label=label, linewidth=2)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tier_comparison(surfaces_dict, filename, suptitle='Direction Selection Comparison'):
    """
    Plot side-by-side contour comparison of multiple tiers.

    Args:
        surfaces_dict: dict {tier_name: (alphas, betas, surface)}
        filename: output path
        suptitle: overall title
    """
    n = len(surfaces_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for idx, (name, (alphas, betas, surface)) in enumerate(surfaces_dict.items()):
        A, B = np.meshgrid(alphas, betas)
        vmin = surface.min()
        vmax = min(surface.max(), surface.min() + 2 * (np.median(surface) - surface.min() + 0.1))
        levels = np.linspace(vmin, vmax, 25)
        cs = axes[idx].contourf(A, B, surface, levels=levels, cmap='viridis')
        axes[idx].contour(A, B, surface, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
        axes[idx].set_xlabel(r'$\alpha$')
        axes[idx].set_ylabel(r'$\beta$')
        axes[idx].set_title(name)
        axes[idx].plot(0, 0, 'r*', markersize=12)
        axes[idx].set_aspect('equal')
        plt.colorbar(cs, ax=axes[idx], shrink=0.8)

    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pfi_comparison(pfi_results, filename):
    """
    Plot PFI comparison bar chart across tiers.

    Args:
        pfi_results: dict {tier_name: {PFI_S, PFI_C, ...}}
        filename: output path
    """
    tiers = list(pfi_results.keys())
    pfi_s_vals = [pfi_results[t].get('PFI_S', 0) or 0 for t in tiers]
    pfi_c_vals = [pfi_results[t].get('PFI_C', 0) or 0 for t in tiers]
    colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(tiers)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(tiers, pfi_s_vals, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('PFI-S (Spectral Coverage)')
    ax1.set_title('Projection Faithfulness Index — Spectral Coverage')
    for i, v in enumerate(pfi_s_vals):
        ax1.text(i, v + max(pfi_s_vals) * 0.02, f'{v:.2e}', ha='center', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(tiers, pfi_c_vals, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('PFI-C (Curvature Capture)')
    ax2.set_title('Projection Faithfulness Index — Curvature Capture')
    for i, v in enumerate(pfi_c_vals):
        ax2.text(i, v + max(pfi_c_vals) * 0.02, f'{v:.2e}', ha='center', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tadn_invariance(curves, deviations, correlations, filename):
    """
    Plot TADN vs Layer Norm invariance test results.

    Args:
        curves: dict {label: (alphas, losses)}
        deviations: dict {method: (alphas, abs_deviations)}
        correlations: dict {method: correlation_value}
        filename: output path
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))

    # Loss curves
    for label, (alphas, losses) in curves.items():
        ax1.plot(alphas, losses, label=label, linewidth=2)
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves: Original vs Rescaled')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Deviations
    for method, (alphas, devs) in deviations.items():
        ax2.plot(alphas, devs, label=method, linewidth=2)
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel('|Loss Deviation|')
    ax2.set_title('Deviation: Original vs Rescaled')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Correlation bars
    methods = list(correlations.keys())
    corrs = [correlations[m] for m in methods]
    bar_colors = ['blue' if 'TADN' in m else 'red' for m in methods]
    ax3.bar(methods, corrs, color=bar_colors, alpha=0.7)
    ax3.set_ylabel('Pearson Correlation')
    ax3.set_title('Scale Invariance Correlation')
    ax3.set_ylim(min(corrs) - 0.05, 1.005)
    for i, v in enumerate(corrs):
        ax3.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca_convergence(pca_results, filename):
    """
    Plot gradient PCA convergence analysis.

    Args:
        pca_results: dict {N: {eigenvalues, explained_ratios, subspace_angle}}
        filename: output path
    """
    checkpoints = sorted(pca_results.keys())
    angles = [pca_results[c].get('subspace_angle_from_prev') for c in checkpoints]
    ev_ratios = [pca_results[c]['explained_ratios'] for c in checkpoints]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subspace angle
    valid_angles = [(c, a) for c, a in zip(checkpoints, angles) if a is not None]
    if valid_angles:
        ax1.plot([c for c, _ in valid_angles], [a for _, a in valid_angles],
                 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of gradient samples (N)')
        ax1.set_ylabel('Subspace angle (degrees)')
        ax1.set_title('PCA Subspace Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='5-degree threshold')
        ax1.legend()

    # Explained variance
    ax2.plot(checkpoints, [r[0] for r in ev_ratios], 'b-o', label='PC1', linewidth=2)
    ax2.plot(checkpoints, [r[1] for r in ev_ratios], 'r-s', label='PC2', linewidth=2)
    ax2.set_xlabel('Number of gradient samples (N)')
    ax2.set_ylabel('Explained variance ratio')
    ax2.set_title('Explained Variance vs. Sample Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(metrics_list, labels, filename):
    """
    Plot bar chart comparison of metrics across experiments.

    Args:
        metrics_list: list of metric dicts
        labels: list of labels
        filename: output path
    """
    keys = ['loss_range', 'roughness', 'basin_diameter', 'curvature_ratio', 'convexity_ratio']
    n_metrics = len(keys)
    n_exps = len(metrics_list)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_exps)
    for k_idx, key in enumerate(keys):
        vals = [m.get(key, 0) for m in metrics_list]
        axes[k_idx].bar(x, vals, color=['#3498db', '#2ecc71', '#e74c3c'][:n_exps], alpha=0.8)
        axes[k_idx].set_xticks(x)
        axes[k_idx].set_xticklabels(labels, rotation=15, fontsize=8)
        axes[k_idx].set_title(key.replace('_', ' ').title())
        axes[k_idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
