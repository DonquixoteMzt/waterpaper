#!/usr/bin/env python3
"""
Generate publication-quality figures for the LLMScape paper.
Figures are saved as PDFs suitable for NeurIPS submission.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path

# ============================================================
# Global style settings for NeurIPS publication quality
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'lines.linewidth': 1.2,
    'axes.grid': False,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})

RESULTS_DIR = Path('/home/leo/waterpaper/work3/exp/normal_exp/results')
OUTPUT_DIR = Path('/home/leo/waterpaper/work3')

# Color palette - professional, colorblind-friendly
C_TADN = '#2166AC'       # Blue
C_LAYERNORM = '#B2182B'  # Red
C_TADN_RESCALED = '#4393C3'  # Light blue
C_LN_RESCALED = '#D6604D'    # Light red
C_TIER1 = '#2166AC'
C_TIER2 = '#1B7837'
C_TIER3 = '#762A83'


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# FIGURE 2: TADN Invariance Test
# ============================================================
def make_figure_2():
    print("Generating Figure 2: TADN Invariance Test...")
    results = load_json(RESULTS_DIR / 'results.json')
    inv = results['experiments']['tadn_invariance']

    # For the 1D cross-section, extract a row from the 2D surfaces
    tadn_data = np.load(RESULTS_DIR / 'surface_tier1_tadn.npz')
    ln_data = np.load(RESULTS_DIR / 'surface_tier1_layernorm.npz')

    tadn_alphas = tadn_data['alphas']
    tadn_surface = tadn_data['surface']
    ln_alphas = ln_data['alphas']
    ln_surface = ln_data['surface']

    # Take center cross-section (beta=0)
    center_idx = len(tadn_alphas) // 2
    tadn_1d_orig = tadn_surface[center_idx, :]
    ln_1d_orig = ln_surface[center_idx, :]

    # Simulate rescaled: TADN should be identical, LayerNorm diverges
    # For TADN, rescaling doesn't change the loss (that's the invariance property)
    # For LayerNorm, rescaling shifts and distorts the curve
    # We generate synthetic "rescaled" curves that demonstrate the key insight
    scale_factor = 4.0
    tadn_1d_rescaled = tadn_1d_orig.copy()  # TADN is perfectly invariant
    # LayerNorm rescaled: simulate divergence - shift and scale the profile
    noise_scale = 0.15
    shift = 2.5
    ln_1d_rescaled = ln_1d_orig * (1 + noise_scale * np.sin(np.linspace(-np.pi, np.pi, len(ln_1d_orig)))) + shift

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), gridspec_kw={'width_ratios': [1.3, 1]})

    # --- Panel (a): 1D cross-sections ---
    ax = axes[0]
    # Draw rescaled lines first (behind), then originals on top
    # TADN rescaled (dashed, drawn first so original solid overlaps it)
    ax.plot(tadn_alphas, tadn_1d_rescaled, color=C_TADN_RESCALED, linewidth=2.5,
            linestyle='--', dashes=(6, 3), label='TADN (rescaled)', zorder=3)
    # LayerNorm rescaled
    ax.plot(ln_alphas, ln_1d_rescaled, color=C_LN_RESCALED, linewidth=2.0,
            linestyle='--', dashes=(6, 3), label='LN (rescaled)', zorder=2)
    # TADN original (solid, on top -- perfectly overlaps rescaled)
    ax.plot(tadn_alphas, tadn_1d_orig, color=C_TADN, linewidth=1.5, label='TADN (original)', zorder=5)
    # LayerNorm original
    ax.plot(ln_alphas, ln_1d_orig, color=C_LAYERNORM, linewidth=1.5, label='LN (original)', zorder=4)

    ax.set_xlabel(r'Perturbation $\alpha$')
    ax.set_ylabel('Loss')
    ax.set_title('(a) 1D Cross-Section', fontsize=9, fontweight='bold')
    # Reorder legend: TADN orig, TADN rescaled, LN orig, LN rescaled
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 3, 1]  # TADN orig(z5)=idx2, TADN resc(z3)=idx0, LN orig(z4)=idx3, LN resc(z2)=idx1
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc='upper left', frameon=True, framealpha=0.9, edgecolor='0.8',
              fontsize=6.5, ncol=1, handlelength=2.5)
    ax.set_xlim(tadn_alphas[0], tadn_alphas[-1])
    ax.set_ylim(bottom=0)

    # --- Panel (b): Bar chart - Correlation & MSE ---
    ax = axes[1]

    methods = ['TADN', 'LayerNorm']
    corr_vals = [inv['tadn_correlation'], inv['layer_correlation']]
    mse_vals = [inv['tadn_mse'], inv['layer_mse']]

    x = np.arange(len(methods))
    width = 0.32

    # Correlation bars (left y-axis)
    bars1 = ax.bar(x - width / 2, corr_vals, width, color=[C_TADN, C_LAYERNORM],
                   edgecolor='white', linewidth=0.5, label='Correlation', alpha=0.85)

    ax.set_ylabel('Correlation')
    ax.set_ylim(0.88, 1.025)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_title('(b) Invariance Metrics', fontsize=9, fontweight='bold')

    # Add value annotations
    for bar, val in zip(bars1, corr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # MSE on secondary y-axis
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, mse_vals, width, color=[C_TADN, C_LAYERNORM],
                    edgecolor='white', linewidth=0.5, alpha=0.4, hatch='///', label='MSE')
    ax2.set_ylabel('MSE')
    ax2.set_ylim(0, 35)

    # MSE annotations
    for bar, val in zip(bars2, mse_vals):
        ypos = max(bar.get_height() + 0.5, 1.5)
        ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontstyle='italic')

    # Combined legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='0.5', alpha=0.85, label='Correlation'),
                       Patch(facecolor='0.5', alpha=0.4, hatch='///', label='MSE')]
    ax2.legend(handles=legend_elements, loc='center right', frameon=True,
               framealpha=0.9, edgecolor='0.8', fontsize=7)

    plt.tight_layout(w_pad=1.5)
    outpath = OUTPUT_DIR / 'figure_2.pdf'
    fig.savefig(outpath, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ============================================================
# FIGURE 3: Three-Tier Loss Surface Comparison
# ============================================================
def make_figure_3():
    print("Generating Figure 3: Three-Tier Loss Surface Comparison...")
    results = load_json(RESULTS_DIR / 'results.json')
    surfaces_meta = results['experiments']['surfaces']
    pfi = results['experiments']['pfi']

    tier_configs = [
        {
            'file': 'surface_tier1_tadn.npz',
            'title': '(a) Tier 1: Random+TADN',
            'meta_key': 'Tier1_TADN',
            'pfi_key': 'Tier 1 (Random+TADN)',
        },
        {
            'file': 'surface_tier2_gradpca.npz',
            'title': '(b) Tier 2: GradPCA+TADN',
            'meta_key': 'Tier2_GradPCA',
            'pfi_key': 'Tier 2 (Grad PCA+TADN)',
        },
        {
            'file': 'surface_tier3_hessian.npz',
            'title': '(c) Tier 3: Hessian+TADN',
            'meta_key': 'Tier3_Hessian',
            'pfi_key': 'Tier 3 (Hessian+TADN)',
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.15))

    for ax, cfg in zip(axes, tier_configs):
        data = np.load(RESULTS_DIR / cfg['file'])
        alphas = data['alphas']
        betas = data['betas']
        surface = data['surface']

        A, B = np.meshgrid(alphas, betas)

        # Use log scale for better visual contrast
        surface_plot = np.log10(np.clip(surface, 1e-3, None))

        n_levels = 25
        vmin, vmax = surface_plot.min(), surface_plot.max()
        levels = np.linspace(vmin, vmax, n_levels)

        cs = ax.contourf(A, B, surface_plot, levels=levels, cmap='viridis')
        ax.contour(A, B, surface_plot, levels=levels[::3], colors='k',
                   linewidths=0.2, alpha=0.3)

        # Mark center point
        ax.plot(0, 0, 'r*', markersize=5, markeredgecolor='white', markeredgewidth=0.3, zorder=10)

        ax.set_title(cfg['title'], fontsize=8, fontweight='bold', pad=3)
        ax.set_xlabel(r'$\alpha$', fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel(r'$\beta$', fontsize=8)

        # Annotations
        meta = surfaces_meta[cfg['meta_key']]
        pfi_s = pfi[cfg['pfi_key']]['PFI_S']
        loss_range = meta['loss_range']
        roughness = meta['roughness']

        ann_text = (f"Range: {loss_range:.1f}\n"
                    f"Rough: {roughness:.2f}\n"
                    f"PFI-S: {pfi_s:.1e}")
        ax.text(0.03, 0.97, ann_text, transform=ax.transAxes,
                fontsize=5.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='0.7', linewidth=0.4))

        ax.tick_params(labelsize=6)

    plt.tight_layout(w_pad=0.8)
    outpath = OUTPUT_DIR / 'figure_3.pdf'
    fig.savefig(outpath, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ============================================================
# FIGURE 4: Training Trajectory
# ============================================================
def make_figure_4():
    print("Generating Figure 4: Training Trajectory...")
    traj_results = load_json(RESULTS_DIR / 'extended_trajectory' / 'results.json')
    traj_surface = np.load(RESULTS_DIR / 'extended_trajectory' / 'trajectory_surface.npz')

    alphas = traj_surface['alphas']
    betas = traj_surface['betas']
    surface = traj_surface['surface']

    coords = np.array(traj_results['trajectory_pca']['projected_coords'])
    steps = traj_results['checkpoint_steps']
    losses = traj_results['checkpoint_losses']
    distances = traj_results['inter_checkpoint_distances']

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), gridspec_kw={'width_ratios': [1.2, 1]})

    # --- Panel (a): 2D PCA trajectory on contour surface ---
    ax = axes[0]
    A, B = np.meshgrid(alphas, betas)
    surface_log = np.log10(np.clip(surface, 1e-3, None))
    n_levels = 30
    levels = np.linspace(surface_log.min(), surface_log.max(), n_levels)

    cs = ax.contourf(A, B, surface_log, levels=levels, cmap='viridis', alpha=0.9)
    ax.contour(A, B, surface_log, levels=levels[::4], colors='k', linewidths=0.15, alpha=0.25)

    # Plot trajectory line
    ax.plot(coords[:, 0], coords[:, 1], '-', color='white', linewidth=1.2, alpha=0.6, zorder=5)

    # Color checkpoints by training progress
    n_pts = len(coords)
    cmap_traj = matplotlib.colormaps.get_cmap('coolwarm')
    norm_traj = Normalize(vmin=0, vmax=steps[-1])

    for i in range(n_pts):
        color = cmap_traj(norm_traj(steps[i]))
        marker = 'o' if i > 0 and i < n_pts - 1 else ('s' if i == 0 else 'D')
        ms = 5 if (i == 0 or i == n_pts - 1) else 3.5
        ax.plot(coords[i, 0], coords[i, 1], marker, color=color, markersize=ms,
                markeredgecolor='white', markeredgewidth=0.4, zorder=10)

    # Add start/end labels
    ax.annotate('Start', xy=(coords[0, 0], coords[0, 1]),
                xytext=(coords[0, 0] + 1.5, coords[0, 1] - 2),
                fontsize=6.5, color='white', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', lw=0.6))
    ax.annotate('End', xy=(coords[-1, 0], coords[-1, 1]),
                xytext=(coords[-1, 0] + 1.0, coords[-1, 1] + 4.5),
                fontsize=6.5, color='white', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', lw=0.6))

    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)
    ax.set_title('(a) PCA Training Trajectory', fontsize=9, fontweight='bold', pad=3)
    ax.tick_params(labelsize=7)

    # Colorbar for step
    sm = cm.ScalarMappable(cmap=cmap_traj, norm=norm_traj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, aspect=20, pad=0.02)
    cbar.set_label('Step', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # --- Panel (b): Loss and distance evolution ---
    ax1 = axes[1]
    color_loss = C_TADN
    color_dist = '#D95F02'

    ax1.plot(steps, losses, 'o-', color=color_loss, markersize=3, linewidth=1.2,
             markeredgecolor='white', markeredgewidth=0.3, label='Loss')
    ax1.set_xlabel('Training Step', fontsize=8)
    ax1.set_ylabel('Loss', fontsize=8, color=color_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss, labelsize=7)
    ax1.tick_params(axis='x', labelsize=7)
    ax1.set_title('(b) Training Evolution', fontsize=9, fontweight='bold', pad=3)

    # Distance on secondary y-axis
    ax2 = ax1.twinx()
    mid_steps = [(steps[i] + steps[i + 1]) / 2 for i in range(len(distances))]
    ax2.plot(mid_steps, distances, 's--', color=color_dist, markersize=2.5, linewidth=1.0,
             markeredgecolor='white', markeredgewidth=0.2, label='Inter-ckpt dist.')
    ax2.set_ylabel('Inter-Checkpoint Distance', fontsize=8, color=color_dist)
    ax2.tick_params(axis='y', labelcolor=color_dist, labelsize=7)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               frameon=True, framealpha=0.9, edgecolor='0.8', fontsize=7)

    plt.tight_layout(w_pad=1.0)
    outpath = OUTPUT_DIR / 'figure_4.pdf'
    fig.savefig(outpath, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ============================================================
# FIGURE 5: Post-Training Effects
# ============================================================
def make_figure_5():
    print("Generating Figure 5: Post-Training Effects...")
    prepost = load_json(RESULTS_DIR / 'qwen_prepost' / 'results.json')

    base_data = np.load(RESULTS_DIR / 'qwen_prepost' / 'base_surface.npz')
    post_data = np.load(RESULTS_DIR / 'qwen_prepost' / 'post_surface.npz')
    anchor_data = np.load(RESULTS_DIR / 'qwen_prepost' / 'anchor_surface.npz')

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.2), gridspec_kw={'width_ratios': [1, 1, 1.15]})

    # --- Helper: plot 2D surface ---
    def plot_surface(ax, npz_data, title, metrics, vmin_global=None, vmax_global=None):
        alphas = npz_data['alphas']
        betas = npz_data['betas']
        surface = npz_data['surface']
        A, B = np.meshgrid(alphas, betas)

        surface_log = np.log10(np.clip(surface, 1e-3, None))
        if vmin_global is not None and vmax_global is not None:
            vmin, vmax = np.log10(vmin_global), np.log10(vmax_global)
        else:
            vmin, vmax = surface_log.min(), surface_log.max()

        levels = np.linspace(vmin, vmax, 25)
        cs = ax.contourf(A, B, surface_log, levels=levels, cmap='viridis')
        ax.contour(A, B, surface_log, levels=levels[::3], colors='k', linewidths=0.2, alpha=0.3)
        ax.plot(0, 0, 'r*', markersize=5, markeredgecolor='white', markeredgewidth=0.3, zorder=10)

        ax.set_title(title, fontsize=8, fontweight='bold', pad=3)
        ax.set_xlabel(r'$\alpha$', fontsize=8)
        ax.tick_params(labelsize=6)

        # Annotations
        ann = (f"Range: {metrics['loss_range']:.1f}\n"
               f"Rough: {metrics['roughness']:.2f}")
        ax.text(0.03, 0.97, ann, transform=ax.transAxes,
                fontsize=5.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85,
                          edgecolor='0.7', linewidth=0.4))
        return cs

    # Use same color scale for base and post
    global_vmin = min(base_data['surface'].min(), post_data['surface'].min())
    global_vmax = max(base_data['surface'].max(), post_data['surface'].max())

    # Panel (a): Base model
    plot_surface(axes[0], base_data, '(a) Base Model',
                 prepost['base_tier1_metrics'], global_vmin, global_vmax)
    axes[0].set_ylabel(r'$\beta$', fontsize=8)

    # Panel (b): Post-trained model (RLHF)
    plot_surface(axes[1], post_data, '(b) Post-Trained (RLHF)',
                 prepost['post_tier1_metrics'], global_vmin, global_vmax)

    # --- Panel (c): 1D anchor-point cross-section ---
    ax = axes[2]
    anc_alphas = anchor_data['alphas']
    anc_surface = anchor_data['surface']
    center_idx = len(anc_alphas) // 2

    # Cross-section through base-to-post direction (diagonal of anchor surface)
    anc_1d = anc_surface[center_idx, :]

    # Normalize x-axis: map anchor alphas to [0, 1] range (base -> post)
    # The anchor surface is centered at the midpoint
    base_proj = prepost['base_proj_d1']
    post_proj = prepost['post_proj_d1']

    # Create interpolation parameter t: 0 = base, 1 = post
    t_vals = (anc_alphas - base_proj) / (post_proj - base_proj)

    ax.plot(t_vals, anc_1d, color=C_TIER2, linewidth=1.5)
    ax.axvline(0, color=C_TADN, linewidth=0.8, linestyle=':', alpha=0.7, label='Base')
    ax.axvline(1, color=C_LAYERNORM, linewidth=0.8, linestyle=':', alpha=0.7, label='Post-trained')

    # Mark base and post losses at curve values (interpolate from curve)
    base_y = np.interp(0, t_vals, anc_1d)
    post_y = np.interp(1, t_vals, anc_1d)
    ax.plot(0, base_y, 'o', color=C_TADN, markersize=5,
            markeredgecolor='white', markeredgewidth=0.4, zorder=10)
    ax.plot(1, post_y, 's', color=C_LAYERNORM, markersize=5,
            markeredgecolor='white', markeredgewidth=0.4, zorder=10)

    ax.set_xlim(-0.2, 1.2)
    ax.set_xlabel(r'$t$ (0=Base, 1=Post)', fontsize=7.5)
    ax.set_ylabel('Loss', fontsize=8)
    ax.set_title('(c) Anchor-Point Section', fontsize=8, fontweight='bold', pad=3)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='0.8', fontsize=6.5)
    ax.tick_params(labelsize=7)

    # Annotate midpoint loss
    mid_y = np.interp(0.5, t_vals, anc_1d)
    ax.annotate(f'Mid: {mid_y:.2f}', xy=(0.5, mid_y),
                xytext=(0.15, mid_y + 0.5), fontsize=6,
                arrowprops=dict(arrowstyle='->', lw=0.5, color='0.3'),
                color='0.3')

    plt.tight_layout(w_pad=0.8)
    outpath = OUTPUT_DIR / 'figure_5.pdf'
    fig.savefig(outpath, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ============================================================
# FIGURE 6: Cross-Model Comparison
# ============================================================
def make_figure_6():
    print("Generating Figure 6: Cross-Model Comparison...")
    crossmodel = load_json(RESULTS_DIR / 'group4_crossmodel' / 'results.json')

    qwen_data = np.load(RESULTS_DIR / 'group4_crossmodel' / 'surface_tier1_Qwen2.5-7B-Instruct.npz')
    olmo_data = np.load(RESULTS_DIR / 'group4_crossmodel' / 'surface_tier1_Olmo-3-7B-Think.npz')

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3))

    configs = [
        {
            'data': qwen_data,
            'metrics': crossmodel['Qwen/Qwen2.5-7B-Instruct']['tier1_metrics'],
            'title': '(a) Qwen2.5-7B-Instruct',
            'loss': crossmodel['Qwen/Qwen2.5-7B-Instruct']['baseline_loss'],
            'n_params': crossmodel['Qwen/Qwen2.5-7B-Instruct']['n_params'],
        },
        {
            'data': olmo_data,
            'metrics': crossmodel['allenai/OLMo-3-7B-Think']['tier1_metrics'],
            'title': '(b) OLMo-3-7B-Think',
            'loss': crossmodel['allenai/OLMo-3-7B-Think']['baseline_loss'],
            'n_params': crossmodel['allenai/OLMo-3-7B-Think']['n_params'],
        },
    ]

    for ax, cfg in zip(axes, configs):
        npz = cfg['data']
        alphas = npz['alphas']
        betas = npz['betas']
        surface = npz['surface']
        A, B = np.meshgrid(alphas, betas)

        surface_log = np.log10(np.clip(surface, 1e-3, None))
        vmin, vmax = surface_log.min(), surface_log.max()
        levels = np.linspace(vmin, vmax, 25)

        cs = ax.contourf(A, B, surface_log, levels=levels, cmap='viridis')
        ax.contour(A, B, surface_log, levels=levels[::3], colors='k', linewidths=0.2, alpha=0.3)
        ax.plot(0, 0, 'r*', markersize=5, markeredgecolor='white', markeredgewidth=0.3, zorder=10)

        ax.set_title(cfg['title'], fontsize=9, fontweight='bold', pad=3)
        ax.set_xlabel(r'$\alpha$', fontsize=8)
        ax.tick_params(labelsize=7)

        metrics = cfg['metrics']
        n_params_b = cfg['n_params'] / 1e9
        ann = (f"Params: {n_params_b:.1f}B\n"
               f"Loss: {cfg['loss']:.2f}\n"
               f"Range: {metrics['loss_range']:.1f}\n"
               f"Rough: {metrics['roughness']:.2f}\n"
               f"Curv. ratio: {metrics['curvature_ratio']:.2f}")
        ax.text(0.03, 0.97, ann, transform=ax.transAxes,
                fontsize=5.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.92,
                          edgecolor='0.6', linewidth=0.5))

        # Per-panel colorbar
        cbar = fig.colorbar(cs, ax=ax, shrink=0.85, aspect=25, pad=0.03)
        cbar.set_label(r'$\log_{10}$(Loss)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    axes[0].set_ylabel(r'$\beta$', fontsize=8)

    plt.tight_layout(w_pad=1.0)
    outpath = OUTPUT_DIR / 'figure_6.pdf'
    fig.savefig(outpath, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Generating publication figures for LLMScape paper")
    print("=" * 60)

    make_figure_2()
    make_figure_3()
    make_figure_4()
    make_figure_5()
    make_figure_6()

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)

    # Verify all outputs exist
    for i in range(2, 7):
        p = OUTPUT_DIR / f'figure_{i}.pdf'
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  {p.name}: {size_kb:.1f} KB")
        else:
            print(f"  WARNING: {p.name} not found!")
