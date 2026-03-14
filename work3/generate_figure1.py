#!/usr/bin/env python3
"""
Generate Method Overview Figure (Figure 1) for the LLMScape paper.
Horizontal pipeline diagram showing the full framework.
Publication-quality for NeurIPS.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_rounded_box(ax, x, y, w, h, facecolor, edgecolor='#333333',
                     linewidth=1.2, alpha=1.0, zorder=2,
                     boxstyle="round,pad=0.02,rounding_size=0.03"):
    """Draw a rounded rectangle at (x, y) with width w and height h."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=boxstyle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        transform=ax.transData,
        mutation_scale=1.0,
    )
    ax.add_patch(box)
    return box


def draw_arrow(ax, xy_start, xy_end, color='#555555', lw=1.5,
               arrowstyle='->,head_width=4,head_length=3',
               zorder=3, connectionstyle="arc3,rad=0.0"):
    """Draw a fancy arrow between two points."""
    arrow = FancyArrowPatch(
        xy_start, xy_end,
        arrowstyle=arrowstyle,
        color=color,
        linewidth=lw,
        zorder=zorder,
        connectionstyle=connectionstyle,
        mutation_scale=1.0,
    )
    ax.add_patch(arrow)
    return arrow


def draw_mini_contour_inset(fig, ax, cx, cy, size_x=0.065, size_y=0.065):
    """Draw a small schematic contour plot using an inset axes, properly clipped."""
    # Convert data coords to figure fraction for the inset
    # We need the display coords -> figure fraction
    inv = fig.transFigure.inverted()

    # Get the bounding box in display coords
    p1 = ax.transData.transform((cx - size_x, cy - size_y))
    p2 = ax.transData.transform((cx + size_x, cy + size_y))
    # Convert to figure fraction
    fp1 = inv.transform(p1)
    fp2 = inv.transform(p2)

    inset_ax = fig.add_axes([fp1[0], fp1[1], fp2[0] - fp1[0], fp2[1] - fp1[1]])
    inset_ax.set_xlim(-1, 1)
    inset_ax.set_ylim(-1, 1)
    inset_ax.axis('off')

    n = 80
    x_grid = np.linspace(-1, 1, n)
    y_grid = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x_grid, y_grid)
    # Asymmetric bowl
    Z = 0.5 * X**2 + 1.1 * Y**2 + 0.25 * X * Y + 0.1 * np.sin(3 * X) * np.cos(2 * Y)

    inset_ax.contour(
        X, Y, Z, levels=6,
        colors=['#2166ac', '#4393c3', '#92c5de', '#f4a582', '#d6604d', '#b2182b'],
        linewidths=0.6,
    )
    # Star at minimum
    inset_ax.plot(0.05, -0.05, '*', color='#b2182b', markersize=4, zorder=6)

    # Thin border around the mini plot
    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#aaaaaa')
        spine.set_linewidth(0.5)

    return inset_ax


def main():
    # ---- Figure setup ----
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('auto')
    ax.axis('off')

    # ---- Color palette ----
    col_input  = '#dce9f5'   # light steel blue
    col_tadn   = '#c6e2f0'   # light blue
    col_shids  = '#d4edda'   # light green
    col_pfi    = '#fef3cd'   # light yellow-orange
    col_grid   = '#fde0d0'   # light salmon
    col_output = '#e8daef'   # light purple
    col_mmsp   = '#f5f5f5'   # very light gray
    col_edge   = '#444444'
    col_header = '#2c3e50'   # dark header text
    col_body   = '#34495e'   # body text
    col_step   = '#7f8c8d'   # step label color

    # ---- Font sizes ----
    fs_header = 6.5
    fs_body   = 5.0
    fs_step   = 4.5
    fs_small  = 4.2

    # ---- Main pipeline boxes: positions ----
    main_y = 0.38
    main_h = 0.48

    boxes = [
        (0.010, main_y, 0.105, main_h, col_input),    # 0: Input
        (0.128, main_y, 0.160, main_h, col_tadn),     # 1: TADN
        (0.301, main_y, 0.160, main_h, col_shids),    # 2: SHIDS
        (0.474, main_y, 0.155, main_h, col_pfi),      # 3: PFI
        (0.642, main_y, 0.155, main_h, col_grid),     # 4: Grid Eval
        (0.810, main_y, 0.180, main_h, col_output),   # 5: Output
    ]

    for (bx, by, bw, bh, bc) in boxes:
        draw_rounded_box(ax, bx, by, bw, bh, facecolor=bc, edgecolor=col_edge,
                         linewidth=1.0)

    # ---- Arrows between main boxes ----
    arrow_y = main_y + main_h / 2
    for i in range(len(boxes) - 1):
        b1, b2 = boxes[i], boxes[i + 1]
        x_start = b1[0] + b1[2] + 0.003
        x_end = b2[0] - 0.003
        draw_arrow(ax, (x_start, arrow_y), (x_end, arrow_y),
                   color='#555555', lw=1.3,
                   arrowstyle='->,head_width=3.5,head_length=2.5')

    # ===== BOX 0: Input =====
    bx, by, bw, bh = boxes[0][:4]
    cx = bx + bw / 2
    ax.text(cx, by + bh - 0.07, 'Input', fontsize=fs_header, fontweight='bold',
            ha='center', va='top', color=col_header, zorder=5)
    ax.text(cx, by + bh / 2 - 0.02, 'Model $\\theta^*$\n+ Data $\\mathcal{D}$',
            fontsize=fs_body, ha='center', va='center', color=col_body, zorder=5,
            linespacing=1.4)

    # ===== BOX 1: TADN =====
    bx, by, bw, bh = boxes[1][:4]
    cx = bx + bw / 2
    ax.text(cx, by + bh - 0.04, 'Step 1: TADN', fontsize=fs_step, fontweight='bold',
            ha='center', va='top', color=col_step, zorder=5, fontstyle='italic')
    ax.text(cx, by + bh - 0.11, 'Direction\nNormalization', fontsize=fs_header,
            fontweight='bold', ha='center', va='top', color=col_header, zorder=5,
            linespacing=1.25)

    sub_items_1 = ['Per-head Attention', 'Per-neuron FFN', 'Per-token Embedding']
    for i, txt in enumerate(sub_items_1):
        yy = by + bh - 0.27 - i * 0.075
        ax.plot(bx + 0.015, yy, 's', color='#5b9bd5', markersize=2, zorder=5)
        ax.text(bx + 0.025, yy, txt, fontsize=fs_small, ha='left', va='center',
                color=col_body, zorder=5)

    # ===== BOX 2: SHIDS =====
    bx, by, bw, bh = boxes[2][:4]
    cx = bx + bw / 2
    ax.text(cx, by + bh - 0.04, 'Step 2: SHIDS', fontsize=fs_step, fontweight='bold',
            ha='center', va='top', color=col_step, zorder=5, fontstyle='italic')
    ax.text(cx, by + bh - 0.11, 'Direction\nSelection', fontsize=fs_header,
            fontweight='bold', ha='center', va='top', color=col_header, zorder=5,
            linespacing=1.25)

    tier_colors = ['#a8d5a2', '#7ec87e', '#4caf50']
    tier_labels = ['Tier 1: Random', 'Tier 2: Grad-PCA', 'Tier 3: Hessian Eigvec']
    for i, (tc, tl) in enumerate(zip(tier_colors, tier_labels)):
        ty = by + bh - 0.28 - i * 0.075
        tw = bw - 0.020
        th = 0.065
        draw_rounded_box(ax, bx + 0.010, ty - th / 2, tw, th,
                         facecolor=tc, edgecolor='#388e3c', linewidth=0.6, alpha=0.7,
                         boxstyle="round,pad=0.005,rounding_size=0.015")
        ax.text(bx + bw / 2, ty, tl, fontsize=fs_small, ha='center', va='center',
                color='#1b5e20', fontweight='semibold', zorder=5)

    # ===== BOX 3: PFI =====
    bx, by, bw, bh = boxes[3][:4]
    cx = bx + bw / 2
    ax.text(cx, by + bh - 0.04, 'Step 3: PFI', fontsize=fs_step, fontweight='bold',
            ha='center', va='top', color=col_step, zorder=5, fontstyle='italic')
    ax.text(cx, by + bh - 0.11, 'Faithfulness\nAssessment', fontsize=fs_header,
            fontweight='bold', ha='center', va='top', color=col_header, zorder=5,
            linespacing=1.25)

    pfi_items = [
        ('PFI-S:', 'Spectral\nCoverage'),
        ('PFI-C:', 'Curvature\nCapture'),
    ]
    for i, (label, desc) in enumerate(pfi_items):
        yy = by + bh - 0.30 - i * 0.11
        ax.text(bx + 0.012, yy + 0.01, label, fontsize=fs_small, ha='left',
                va='center', color='#e67e22', fontweight='bold', zorder=5)
        ax.text(bx + 0.058, yy + 0.01, desc, fontsize=fs_small - 0.3, ha='left',
                va='center', color=col_body, zorder=5, linespacing=1.1)

    # ===== BOX 4: Grid Eval =====
    bx, by, bw, bh = boxes[4][:4]
    cx = bx + bw / 2
    ax.text(cx, by + bh - 0.04, 'Step 4: Grid Eval', fontsize=fs_step,
            fontweight='bold', ha='center', va='top', color=col_step, zorder=5,
            fontstyle='italic')
    ax.text(cx, by + bh - 0.11, 'Loss Surface\nEvaluation', fontsize=fs_header,
            fontweight='bold', ha='center', va='top', color=col_header, zorder=5,
            linespacing=1.25)

    sub_items_4 = ['Curvature-aware\nscale selection', 'Exact parameter\nrestoration']
    for i, txt in enumerate(sub_items_4):
        yy = by + bh - 0.30 - i * 0.11
        ax.plot(bx + 0.012, yy + 0.01, 'o', color='#e74c3c', markersize=2, zorder=5)
        ax.text(bx + 0.024, yy + 0.01, txt, fontsize=fs_small - 0.3, ha='left',
                va='center', color=col_body, zorder=5, linespacing=1.1)

    # ===== BOX 5: Output =====
    bx, by, bw, bh = boxes[5][:4]
    cx = bx + bw / 2
    ax.text(cx, by + bh - 0.04, 'Output', fontsize=fs_header, fontweight='bold',
            ha='center', va='top', color=col_header, zorder=5)
    ax.text(cx, by + bh - 0.14, '2D Loss\nLandscape +\nGeometric\nMetrics',
            fontsize=fs_body, ha='center', va='top', color=col_body, zorder=5,
            linespacing=1.25)

    # Mini contour plot as a proper inset (clipped)
    draw_mini_contour_inset(fig, ax, cx, by + 0.13, size_x=0.055, size_y=0.055)

    # ===== MMSP secondary path =====
    mmsp_y = 0.10
    mmsp_h = 0.20
    mmsp_x = boxes[1][0]
    mmsp_w = boxes[4][0] + boxes[4][2] - mmsp_x

    draw_rounded_box(ax, mmsp_x, mmsp_y, mmsp_w, mmsp_h,
                     facecolor=col_mmsp, edgecolor='#999999', linewidth=0.8,
                     boxstyle="round,pad=0.01,rounding_size=0.02")

    # MMSP label
    ax.text(mmsp_x + 0.008, mmsp_y + mmsp_h / 2 + 0.02, 'MMSP',
            fontsize=fs_step + 0.5, fontweight='bold', ha='left', va='center',
            color='#666666', zorder=5, fontstyle='italic')
    ax.text(mmsp_x + 0.008, mmsp_y + mmsp_h / 2 - 0.04, '(Multi-Model)',
            fontsize=fs_small - 0.3, ha='left', va='center',
            color='#888888', zorder=5)

    # Three MMSP modes
    mmsp_modes = ['Trajectory-PCA', 'Anchor-Point', 'Independent']
    mmsp_mode_colors = ['#b3cde3', '#ccebc5', '#fbb4ae']
    mode_start_x = mmsp_x + 0.11
    mode_spacing = 0.155
    for i, (mode, mc) in enumerate(zip(mmsp_modes, mmsp_mode_colors)):
        mx = mode_start_x + i * mode_spacing
        my = mmsp_y + 0.03
        mw = 0.13
        mh = 0.14
        draw_rounded_box(ax, mx, my, mw, mh, facecolor=mc, edgecolor='#888888',
                         linewidth=0.6,
                         boxstyle="round,pad=0.005,rounding_size=0.012")
        ax.text(mx + mw / 2, my + mh / 2, mode, fontsize=fs_small, ha='center',
                va='center', color='#333333', fontweight='semibold', zorder=5)

    # Dashed arrow from MMSP to output box -- route with an L-shaped path
    # Go right from MMSP end, then up into the Output box bottom
    mmsp_arrow_y = mmsp_y + mmsp_h / 2
    x_mmsp_end = mmsp_x + mmsp_w
    out_bx, out_by = boxes[5][0], boxes[5][1]
    out_cx = out_bx + boxes[5][2] / 2
    # Horizontal dashed segment from MMSP end to below Output box center
    ax.plot([x_mmsp_end + 0.003, out_cx], [mmsp_arrow_y, mmsp_arrow_y],
            '--', color='#888888', linewidth=1.0, zorder=3, dashes=(4, 2.5))
    # Vertical dashed segment going up to the Output box bottom
    ax.plot([out_cx, out_cx], [mmsp_arrow_y, out_by - 0.015],
            '--', color='#888888', linewidth=1.0, zorder=3, dashes=(4, 2.5))
    # Arrowhead pointing up into the Output box
    draw_arrow(ax, (out_cx, out_by - 0.018), (out_cx, out_by + 0.005),
               color='#888888', lw=1.0,
               arrowstyle='->,head_width=3,head_length=2.5', zorder=3)

    # Downward arrows from Step 1 and Step 2 into MMSP region
    # Arrow goes FROM bottom of main box DOWN TO top of MMSP box
    for idx in [1, 2]:
        bx_s, by_s, bw_s, bh_s = boxes[idx][:4]
        cx_s = bx_s + bw_s / 2
        draw_arrow(ax, (cx_s, by_s - 0.003), (cx_s, mmsp_y + mmsp_h + 0.008),
                   color='#aaaaaa', lw=0.9,
                   arrowstyle='->,head_width=2.5,head_length=2')

    # (Figure caption in the paper will provide the title)

    # ---- Save ----
    fig.savefig('/home/leo/waterpaper/work3/figure_1.pdf',
                dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Figure saved to /home/leo/waterpaper/work3/figure_1.pdf")


if __name__ == '__main__':
    main()
