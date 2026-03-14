"""
run_qwen_prepost.py — Qwen3-0.6B Pre/Post-Training Comparison (MMSP Method B).

Compares Qwen3-0.6B-Base (pre-trained) and Qwen3-0.6B (post-trained)
using Anchor-Point Projection to visualize post-training effects.

Usage:
    python run_qwen_prepost.py --gpu 2
"""

import os
import sys
import json
import time
import gc
import math
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from normalization import get_normalization_units, apply_tadn
from direction_selection import generate_random_direction, orthogonalize_directions
from grid_evaluation import evaluate_loss, evaluate_2d_surface, evaluate_1d_curve
from metrics import compute_surface_metrics, format_metrics_table
from multi_model import anchor_point_projection, compute_model_distance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ChunkDS(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        return {'input_ids': self.chunks[idx],
                'attention_mask': torch.ones(len(self.chunks[idx]), dtype=torch.long)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--grid-size', type=int, default=31)
    parser.add_argument('--max-eval-batches', type=int, default=5)
    parser.add_argument('--seq-len', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/qwen_prepost'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Qwen3-0.6B Pre/Post-Training Comparison (MMSP Method B)")
    print("=" * 70)
    t_start = time.time()

    base_name = "Qwen/Qwen3-0.6B-Base"
    post_name = "Qwen/Qwen3-0.6B"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- Step 1: Load both models ---
    print("\nStep 1: Loading models...")
    print(f"  Loading {base_name}...")
    model_base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    )
    model_base.eval()

    print(f"  Loading {post_name}...")
    model_post = AutoModelForCausalLM.from_pretrained(
        post_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    )
    model_post.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_heads = getattr(model_base.config, 'num_attention_heads', None)
    head_dim = getattr(model_base.config, 'head_dim', None)
    if head_dim is None and num_heads:
        hidden_size = getattr(model_base.config, 'hidden_size', None)
        if hidden_size:
            head_dim = hidden_size // num_heads
    n_params = sum(p.numel() for p in model_base.parameters())
    print(f"  Architecture: {n_params:,} params, num_heads={num_heads}, head_dim={head_dim}")

    # --- Step 2: Extract parameters ---
    print("\nStep 2: Extracting parameters...")
    params_base = {name: param.data.cpu().clone() for name, param in model_base.named_parameters()}
    params_post = {name: param.data.cpu().clone() for name, param in model_post.named_parameters()}

    # --- Step 3: Compute Anchor-Point Projection ---
    print("\nStep 3: Computing Anchor-Point Projection...")
    d1, d2, midpoint, dist = anchor_point_projection(params_base, params_post)
    print(f"  Parameter distance (Base to Post): {dist:.4f}")

    # Compute the projection of each model onto the d1 axis
    # Base model is at -dist/2 on d1 axis, Post model is at +dist/2
    d1_norm = math.sqrt(sum((d1[n].float() ** 2).sum().item() for n in d1))
    base_proj_d1 = -dist / (2 * d1_norm) if d1_norm > 0 else 0
    post_proj_d1 = dist / (2 * d1_norm) if d1_norm > 0 else 0
    print(f"  d1 norm: {d1_norm:.4f}")
    print(f"  Base position on d1: {base_proj_d1:.6f}")
    print(f"  Post position on d1: {post_proj_d1:.6f}")

    # --- Step 4: Apply TADN normalization ---
    print("\nStep 4: Applying TADN normalization...")
    units = get_normalization_units(model_base)
    d1_tadn = apply_tadn(d1, model_base, units, num_heads, head_dim)
    d2_tadn = apply_tadn(d2, model_base, units, num_heads, head_dim)

    # Recompute projections after TADN
    d1_tadn_norm = math.sqrt(sum((d1_tadn[n].float() ** 2).sum().item() for n in d1_tadn))
    # The direction between models: project params_base - midpoint and params_post - midpoint
    base_proj_tadn = sum(
        ((params_base[n].float() - midpoint[n].float()).flatten() @
         d1_tadn[n].float().flatten()).item()
        for n in d1_tadn
    ) / (d1_tadn_norm ** 2) if d1_tadn_norm > 0 else 0
    post_proj_tadn = sum(
        ((params_post[n].float() - midpoint[n].float()).flatten() @
         d1_tadn[n].float().flatten()).item()
        for n in d1_tadn
    ) / (d1_tadn_norm ** 2) if d1_tadn_norm > 0 else 0

    print(f"  After TADN: Base at d1={base_proj_tadn:.6f}, Post at d1={post_proj_tadn:.6f}")

    # --- Step 5: Prepare evaluation data ---
    print("\nStep 5: Preparing evaluation data...")
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in dataset['text'] if len(t.strip()) > 50]
    all_tokens = tokenizer(
        '\n'.join(texts[:200]), return_tensors='pt', truncation=False
    )['input_ids'][0]

    chunks = []
    for i in range(0, len(all_tokens) - args.seq_len, args.seq_len):
        chunks.append(all_tokens[i:i + args.seq_len])
    n_eval = min(50, len(chunks))
    eval_loader = DataLoader(ChunkDS(chunks[:n_eval]), batch_size=4, shuffle=False)

    # --- Step 6: Evaluate baseline losses ---
    print("\nStep 6: Evaluating baseline losses...")
    model_base.to(device)
    loss_base = evaluate_loss(model_base, eval_loader, device, max_batches=args.max_eval_batches)
    print(f"  Base model loss: {loss_base:.4f}")
    model_base.cpu()
    torch.cuda.empty_cache()

    model_post.to(device)
    loss_post = evaluate_loss(model_post, eval_loader, device, max_batches=args.max_eval_batches)
    print(f"  Post-trained model loss: {loss_post:.4f}")
    model_post.cpu()
    torch.cuda.empty_cache()

    # --- Step 7: Evaluate 2D surface from midpoint ---
    print("\nStep 7: Evaluating 2D loss surface (midpoint-centered)...")

    # Create a model at the midpoint
    mid_model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    )
    mid_model.eval()
    for name, param in mid_model.named_parameters():
        if name in midpoint:
            param.data.copy_(midpoint[name].to(param.dtype))
    mid_model.to(device)

    loss_mid = evaluate_loss(mid_model, eval_loader, device, max_batches=args.max_eval_batches)
    print(f"  Midpoint loss: {loss_mid:.4f}")

    # Grid range: make sure both models are visible
    grid_extent = max(abs(base_proj_tadn), abs(post_proj_tadn)) * 1.5
    grid_extent = max(grid_extent, 0.01)  # Minimum range
    print(f"  Grid range: [-{grid_extent:.6f}, {grid_extent:.6f}]")

    alphas, betas, surface = evaluate_2d_surface(
        mid_model, d1_tadn, d2_tadn, eval_loader, device,
        grid_range=(-grid_extent, grid_extent),
        grid_size=args.grid_size,
        max_batches=args.max_eval_batches,
    )

    # --- Step 8: Also evaluate 1D cross-section along d1 (Base → Post) ---
    print("\nStep 8: Evaluating 1D cross-section (Base to Post)...")
    alphas_1d, losses_1d = evaluate_1d_curve(
        mid_model, d1_tadn, eval_loader, device,
        alpha_range=(-grid_extent * 1.5, grid_extent * 1.5),
        n_points=51,
        max_batches=args.max_eval_batches,
    )

    # --- Step 9: Evaluate surfaces independently for each model (Tier 1) ---
    print("\nStep 9: Evaluating independent Tier 1 surfaces...")

    # Base model Tier 1
    model_base.to(device)
    raw_d1 = generate_random_direction(model_base, seed=42)
    raw_d2 = generate_random_direction(model_base, seed=123)
    raw_d2 = orthogonalize_directions(raw_d1, raw_d2)
    units_base = get_normalization_units(model_base)
    tier1_d1_base = apply_tadn(raw_d1, model_base, units_base, num_heads, head_dim)
    tier1_d2_base = apply_tadn(raw_d2, model_base, units_base, num_heads, head_dim)

    a_base, b_base, s_base = evaluate_2d_surface(
        model_base, tier1_d1_base, tier1_d2_base, eval_loader, device,
        grid_range=(-1.0, 1.0), grid_size=args.grid_size,
        max_batches=args.max_eval_batches,
    )
    metrics_base = compute_surface_metrics(a_base, b_base, s_base)
    print(format_metrics_table(metrics_base, "Qwen3-0.6B-Base (Tier 1)"))
    model_base.cpu()
    torch.cuda.empty_cache()

    # Post model Tier 1
    model_post.to(device)
    raw_d1_p = generate_random_direction(model_post, seed=42)
    raw_d2_p = generate_random_direction(model_post, seed=123)
    raw_d2_p = orthogonalize_directions(raw_d1_p, raw_d2_p)
    units_post = get_normalization_units(model_post)
    tier1_d1_post = apply_tadn(raw_d1_p, model_post, units_post, num_heads, head_dim)
    tier1_d2_post = apply_tadn(raw_d2_p, model_post, units_post, num_heads, head_dim)

    a_post, b_post, s_post = evaluate_2d_surface(
        model_post, tier1_d1_post, tier1_d2_post, eval_loader, device,
        grid_range=(-1.0, 1.0), grid_size=args.grid_size,
        max_batches=args.max_eval_batches,
    )
    metrics_post = compute_surface_metrics(a_post, b_post, s_post)
    print(format_metrics_table(metrics_post, "Qwen3-0.6B (Tier 1)"))
    model_post.cpu()
    torch.cuda.empty_cache()

    # Surface from midpoint
    metrics_mid = compute_surface_metrics(alphas, betas, surface)

    # --- Step 10: Visualizations ---
    print("\nStep 10: Creating visualizations...")

    # 10a: Anchor-Point surface with model positions
    fig = plt.figure(figsize=(18, 7))
    ax1 = fig.add_subplot(121)
    A, B = np.meshgrid(alphas, betas)
    vmin = surface.min()
    vmax = min(surface.max(), surface.min() + 3 * (np.median(surface) - surface.min() + 0.1))
    levels = np.linspace(vmin, vmax, 30)
    cs = ax1.contourf(A, B, surface, levels=levels, cmap='viridis')
    ax1.contour(A, B, surface, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax1, label='Loss')

    # Mark model positions
    ax1.plot(base_proj_tadn, 0, 'r*', markersize=15, zorder=5, label=f'Base (loss={loss_base:.3f})')
    ax1.plot(post_proj_tadn, 0, 'b*', markersize=15, zorder=5, label=f'Post (loss={loss_post:.3f})')
    ax1.plot(0, 0, 'g*', markersize=12, zorder=5, label=f'Midpoint (loss={loss_mid:.3f})')

    ax1.set_xlabel(r'$d_1$ (Base $\rightarrow$ Post direction)')
    ax1.set_ylabel(r'$d_2$ (orthogonal)')
    ax1.set_title('Qwen3-0.6B Pre/Post-Training\n(Anchor-Point, MMSP Method B)')
    ax1.legend(fontsize=8)

    # 3D
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.plot_surface(A, B, surface, cmap='viridis', alpha=0.7, vmin=vmin, vmax=vmax)
    ax3d.set_xlabel('d1')
    ax3d.set_ylabel('d2')
    ax3d.set_zlabel('Loss')
    ax3d.set_title('3D Surface')
    ax3d.view_init(elev=30, azim=225)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anchor_point_surface.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 10b: 1D cross-section
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas_1d, losses_1d, 'b-', linewidth=2, label='Loss along Base→Post')
    ax.axvline(x=base_proj_tadn, color='r', linestyle='--', label=f'Base (α={base_proj_tadn:.4f})')
    ax.axvline(x=post_proj_tadn, color='g', linestyle='--', label=f'Post (α={post_proj_tadn:.4f})')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Midpoint')
    ax.set_xlabel(r'$\alpha$ along $d_1$ (Base $\rightarrow$ Post)')
    ax.set_ylabel('Loss')
    ax.set_title('1D Loss Profile: Base to Post-Trained')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anchor_point_1d.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 10c: Side-by-side comparison (Tier 1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for idx, (a, b, s, title, m) in enumerate([
        (a_base, b_base, s_base, 'Qwen3-0.6B-Base', metrics_base),
        (a_post, b_post, s_post, 'Qwen3-0.6B (Post)', metrics_post),
        (alphas, betas, surface, 'Anchor-Point View', metrics_mid),
    ]):
        A_g, B_g = np.meshgrid(a, b)
        vmin_s = s.min()
        vmax_s = min(s.max(), s.min() + 3 * (np.median(s) - s.min() + 0.1))
        levels_s = np.linspace(vmin_s, vmax_s, 25)
        cs_s = axes[idx].contourf(A_g, B_g, s, levels=levels_s, cmap='viridis')
        axes[idx].contour(A_g, B_g, s, levels=levels_s, colors='white', linewidths=0.3, alpha=0.5)
        plt.colorbar(cs_s, ax=axes[idx], shrink=0.8)
        axes[idx].set_xlabel(r'$\alpha$')
        axes[idx].set_ylabel(r'$\beta$')
        axes[idx].set_title(f'{title}\nrange={m["loss_range"]:.2f}, rough={m["roughness"]:.3f}')
        axes[idx].plot(0, 0, 'r*', markersize=12)
    plt.suptitle('Pre vs Post-Training: Loss Landscape Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prepost_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Save results ---
    total_time = time.time() - t_start
    results = {
        'experiment': 'Qwen3-0.6B Pre/Post-Training (MMSP Method B)',
        'base_model': base_name,
        'post_model': post_name,
        'n_params': n_params,
        'parameter_distance': dist,
        'base_loss': loss_base,
        'post_loss': loss_post,
        'midpoint_loss': loss_mid,
        'base_proj_d1': base_proj_tadn,
        'post_proj_d1': post_proj_tadn,
        'grid_range': float(grid_extent),
        'grid_size': args.grid_size,
        'anchor_surface_metrics': metrics_mid,
        'base_tier1_metrics': metrics_base,
        'post_tier1_metrics': metrics_post,
        'total_time_seconds': total_time,
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    np.savez(os.path.join(output_dir, 'anchor_surface.npz'),
             alphas=alphas, betas=betas, surface=surface)
    np.savez(os.path.join(output_dir, 'base_surface.npz'),
             alphas=a_base, betas=b_base, surface=s_base)
    np.savez(os.path.join(output_dir, 'post_surface.npz'),
             alphas=a_post, betas=b_post, surface=s_post)

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    print("Qwen3-0.6B Pre/Post-Training Experiment Complete!")
    print("=" * 70)

    print("\n--- SUMMARY ---")
    print(f"Base loss: {loss_base:.4f}, Post loss: {loss_post:.4f}")
    print(f"Midpoint loss: {loss_mid:.4f}")
    print(f"Parameter distance: {dist:.4f}")
    print(f"Base Tier1: range={metrics_base['loss_range']:.2f}, roughness={metrics_base['roughness']:.4f}")
    print(f"Post Tier1: range={metrics_post['loss_range']:.2f}, roughness={metrics_post['roughness']:.4f}")
    print(f"Anchor: range={metrics_mid['loss_range']:.2f}, roughness={metrics_mid['roughness']:.4f}")


if __name__ == '__main__':
    main()
