"""
run_tinyllama_full.py — Complete TinyLlama Pre-Training Trajectory Analysis.

Downloads TinyLlama checkpoint series (7 checkpoints covering 105B-3T tokens)
and runs:
  1. Trajectory-PCA (MMSP Method A) to visualize training evolution
  2. Independent Tier 1 surfaces for selected checkpoints
  3. Per-checkpoint geometric metric extraction
  4. Inter-checkpoint distance analysis

Usage:
    python run_tinyllama_full.py --gpu 0
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
from direction_selection import generate_tier1_directions
from grid_evaluation import evaluate_loss, evaluate_2d_surface
from metrics import compute_surface_metrics, format_metrics_table
from multi_model import trajectory_pca

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
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--grid-size', type=int, default=31)
    parser.add_argument('--max-eval-batches', type=int, default=5)
    parser.add_argument('--seq-len', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/tinyllama_trajectory'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("TinyLlama Pre-Training Trajectory Analysis (Complete)")
    print("=" * 70)
    t_start = time.time()

    # Full TinyLlama checkpoint series (ordered by training step)
    checkpoints = [
        ("TinyLlama/TinyLlama-1.1B-step-50K-105b", "50K", 50, 105),
        ("TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b", "240K", 240, 503),
        ("TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T", "480K", 480, 1000),
        ("TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T", "715K", 715, 1500),
        ("TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T", "955K", 955, 2000),
        ("TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T", "1195K", 1195, 2500),
        ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "1431K", 1431, 3000),
    ]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- Step 1: Load all checkpoints' parameters ---
    print("\nStep 1: Loading checkpoint parameters...")
    all_params = []
    checkpoint_labels = []
    ref_model = None
    ref_tokenizer = None

    for ckpt_name, label, steps_k, tokens_b in checkpoints:
        print(f"  Loading {ckpt_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="eager",
        )
        model.eval()
        params = {name: param.data.cpu().clone() for name, param in model.named_parameters()}
        all_params.append(params)
        checkpoint_labels.append(f"{label} steps ({tokens_b}B tokens)")
        if ref_model is None:
            ref_model = model
            ref_tokenizer = AutoTokenizer.from_pretrained(ckpt_name, trust_remote_code=True)
            if ref_tokenizer.pad_token is None:
                ref_tokenizer.pad_token = ref_tokenizer.eos_token
        else:
            del model
        gc.collect()

    n_params = sum(p.numel() for p in all_params[0].values())
    print(f"  Loaded {len(all_params)} checkpoints, {n_params:,} parameters each")

    # --- Step 2: Compute inter-checkpoint distances ---
    print("\nStep 2: Computing inter-checkpoint distances...")
    distances = []
    for i in range(1, len(all_params)):
        dist = math.sqrt(sum(
            ((all_params[i][n].float() - all_params[i-1][n].float()) ** 2).sum().item()
            for n in all_params[0]
        ))
        distances.append(dist)
        print(f"  {checkpoint_labels[i-1]} -> {checkpoint_labels[i]}: L2={dist:.2f}")

    total_dist = math.sqrt(sum(
        ((all_params[-1][n].float() - all_params[0][n].float()) ** 2).sum().item()
        for n in all_params[0]
    ))
    print(f"  Total distance (first -> last): {total_dist:.2f}")

    # --- Step 3: Run Trajectory-PCA ---
    print("\nStep 3: Computing Trajectory-PCA...")
    pca_directions, projected_coords, centroid, explained_var = trajectory_pca(all_params, k=2)
    print(f"  Explained variance: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}")
    print(f"  Total explained: {sum(explained_var):.4f}")

    print("  Projected coordinates:")
    for i, (x, y) in enumerate(projected_coords):
        print(f"    {checkpoint_labels[i]}: ({x:.4f}, {y:.4f})")

    # --- Step 4: Apply TADN to PCA directions ---
    print("\nStep 4: Applying TADN normalization...")
    last_model = AutoModelForCausalLM.from_pretrained(
        checkpoints[-1][0], torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    )
    last_model.eval()
    num_heads = getattr(last_model.config, 'num_attention_heads', None)
    head_dim = getattr(last_model.config, 'head_dim', None)
    if head_dim is None and num_heads:
        hidden_size = getattr(last_model.config, 'hidden_size', None)
        if hidden_size:
            head_dim = hidden_size // num_heads
    print(f"  Architecture: num_heads={num_heads}, head_dim={head_dim}")

    units = get_normalization_units(last_model)
    pca_d1_tadn = apply_tadn(pca_directions[0], last_model, units, num_heads, head_dim)
    pca_d2_tadn = apply_tadn(pca_directions[1], last_model, units, num_heads, head_dim)
    del last_model
    gc.collect()

    # --- Step 5: Prepare evaluation data ---
    print("\nStep 5: Preparing evaluation data...")
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in dataset['text'] if len(t.strip()) > 50]
    all_tokens = ref_tokenizer(
        '\n'.join(texts[:200]), return_tensors='pt', truncation=False
    )['input_ids'][0]

    chunks = []
    for i in range(0, len(all_tokens) - args.seq_len, args.seq_len):
        chunks.append(all_tokens[i:i + args.seq_len])
    n_eval = min(50, len(chunks))
    eval_loader = DataLoader(ChunkDS(chunks[:n_eval]), batch_size=4, shuffle=False)

    # --- Step 6: Build centroid model and evaluate trajectory surface ---
    print("\nStep 6: Building centroid model and evaluating trajectory surface...")
    centroid_model = AutoModelForCausalLM.from_pretrained(
        checkpoints[-1][0], torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    )
    centroid_model.eval()
    for name, param in centroid_model.named_parameters():
        if name in centroid:
            param.data.copy_(centroid[name].to(param.dtype))
    centroid_model.to(device)

    # Grid range from projected coordinates
    all_x = [c[0] for c in projected_coords]
    all_y = [c[1] for c in projected_coords]
    margin = 0.3
    x_range = max(abs(min(all_x)), abs(max(all_x))) * (1 + margin)
    y_range = max(abs(min(all_y)), abs(max(all_y))) * (1 + margin)
    grid_range_val = max(x_range, y_range)
    print(f"  Grid range: [-{grid_range_val:.4f}, {grid_range_val:.4f}]")

    alphas, betas, surface = evaluate_2d_surface(
        centroid_model, pca_d1_tadn, pca_d2_tadn, eval_loader, device,
        grid_range=(-grid_range_val, grid_range_val),
        grid_size=args.grid_size,
        max_batches=args.max_eval_batches,
    )
    trajectory_metrics = compute_surface_metrics(alphas, betas, surface)
    print(format_metrics_table(trajectory_metrics, "Trajectory-PCA Surface"))

    # --- Step 7: Evaluate loss at each checkpoint position ---
    print("\nStep 7: Evaluating loss at each checkpoint position...")
    checkpoint_losses = []
    for i, (ckpt_name, label, steps_k, tokens_b) in enumerate(checkpoints):
        for name, param in centroid_model.named_parameters():
            if name in all_params[i]:
                param.data.copy_(all_params[i][name].to(param.dtype).to(param.device))
        loss = evaluate_loss(centroid_model, eval_loader, device, max_batches=args.max_eval_batches)
        checkpoint_losses.append(loss)
        print(f"  {checkpoint_labels[i]}: loss={loss:.4f}")

    # Restore centroid
    for name, param in centroid_model.named_parameters():
        if name in centroid:
            param.data.copy_(centroid[name].to(param.dtype).to(param.device))

    # --- Step 8: Independent Tier 1 surfaces for selected checkpoints ---
    print("\nStep 8: Computing independent Tier 1 surfaces for selected checkpoints...")
    selected_indices = [0, 2, 4, 6]  # 50K, 480K, 955K, 1431K
    tier1_results = {}

    for idx in selected_indices:
        ckpt_name, label, steps_k, tokens_b = checkpoints[idx]
        print(f"\n  --- Tier 1 for {label} ({tokens_b}B tokens) ---")

        # Load checkpoint into the model
        for name, param in centroid_model.named_parameters():
            if name in all_params[idx]:
                param.data.copy_(all_params[idx][name].to(param.dtype).to(param.device))

        # Generate random directions and apply TADN
        centroid_model.cpu()
        torch.cuda.empty_cache()
        d1_raw, d2_raw = generate_tier1_directions(centroid_model, seed1=42, seed2=123)
        ckpt_units = get_normalization_units(centroid_model)
        d1_t1 = apply_tadn(d1_raw, centroid_model, ckpt_units, num_heads, head_dim)
        d2_t1 = apply_tadn(d2_raw, centroid_model, ckpt_units, num_heads, head_dim)
        del d1_raw, d2_raw
        gc.collect()
        centroid_model.to(device)

        a_t1, b_t1, s_t1 = evaluate_2d_surface(
            centroid_model, d1_t1, d2_t1, eval_loader, device,
            grid_range=(-1.0, 1.0), grid_size=args.grid_size,
            max_batches=args.max_eval_batches,
        )
        m_t1 = compute_surface_metrics(a_t1, b_t1, s_t1)
        tier1_results[label] = {
            'tokens_b': tokens_b,
            'steps_k': steps_k,
            'loss': checkpoint_losses[idx],
            'metrics': m_t1,
        }
        print(f"    loss_range={m_t1['loss_range']:.2f}, roughness={m_t1['roughness']:.4f}, "
              f"basin_diameter={m_t1['basin_diameter']:.4f}, curvature_ratio={m_t1['curvature_ratio']:.4f}")

        np.savez(os.path.join(output_dir, f'surface_tier1_{label}.npz'),
                 alphas=a_t1, betas=b_t1, surface=s_t1)
        del d1_t1, d2_t1
        gc.collect()

    # --- Step 9: Visualization ---
    print("\nStep 9: Creating visualizations...")

    # 9a: Contour plot with training trajectory
    fig = plt.figure(figsize=(18, 7))
    ax1 = fig.add_subplot(121)
    A, B = np.meshgrid(alphas, betas)
    vmin = surface.min()
    vmax = min(surface.max(), surface.min() + 3 * (np.median(surface) - surface.min() + 0.1))
    levels = np.linspace(vmin, vmax, 30)
    cs = ax1.contourf(A, B, surface, levels=levels, cmap='viridis')
    ax1.contour(A, B, surface, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax1, label='Loss')

    xs = [c[0] for c in projected_coords]
    ys = [c[1] for c in projected_coords]
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(projected_coords)))
    ax1.plot(xs, ys, '-', color='white', linewidth=1.5, alpha=0.6, zorder=4)
    for i, (x, y) in enumerate(projected_coords):
        ax1.scatter(x, y, color=colors[i], s=80, edgecolors='white', linewidth=1, zorder=5)
        short_label = checkpoints[i][1]
        ax1.annotate(short_label, (x, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=7, color='white', fontweight='bold', zorder=6)

    ax1.set_xlabel(f'PC1 (explained var: {explained_var[0]:.1%})')
    ax1.set_ylabel(f'PC2 (explained var: {explained_var[1]:.1%})')
    ax1.set_title('TinyLlama-1.1B Pre-Training Trajectory\n(Trajectory-PCA, MMSP Method A)')

    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.plot_surface(A, B, surface, cmap='viridis', alpha=0.7, vmin=vmin, vmax=vmax)
    zs = []
    for (x, y) in projected_coords:
        i_x = np.argmin(np.abs(alphas - x))
        i_y = np.argmin(np.abs(betas - y))
        zs.append(surface[i_y, i_x])
    ax3d.plot(xs, ys, zs, 'r-o', markersize=5, linewidth=2, zorder=5)
    ax3d.set_xlabel('PC1'); ax3d.set_ylabel('PC2'); ax3d.set_zlabel('Loss')
    ax3d.set_title('3D Training Trajectory')
    ax3d.view_init(elev=30, azim=225)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tinyllama_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 9b: Training evolution (loss + distance + landscape metrics)
    steps_k_list = [c[2] for c in checkpoints]
    tokens_b_list = [c[3] for c in checkpoints]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss evolution
    axes[0, 0].plot(tokens_b_list, checkpoint_losses, 'b-o', markersize=6, linewidth=2)
    axes[0, 0].set_xlabel('Training Tokens (B)')
    axes[0, 0].set_ylabel('WikiText-2 Loss')
    axes[0, 0].set_title('Loss Evolution During Pre-Training')
    axes[0, 0].grid(True, alpha=0.3)

    # Inter-checkpoint distances
    axes[0, 1].plot(tokens_b_list[1:], distances, 'g-s', markersize=6, linewidth=2)
    axes[0, 1].set_xlabel('Training Tokens (B)')
    axes[0, 1].set_ylabel('Inter-Checkpoint L2 Distance')
    axes[0, 1].set_title('Parameter Movement Between Checkpoints')
    axes[0, 1].grid(True, alpha=0.3)

    # PCA space distance from centroid
    dists_from_centroid = [math.sqrt(x**2 + y**2) for x, y in projected_coords]
    axes[1, 0].plot(tokens_b_list, dists_from_centroid, 'm-^', markersize=6, linewidth=2)
    axes[1, 0].set_xlabel('Training Tokens (B)')
    axes[1, 0].set_ylabel('Distance from Centroid')
    axes[1, 0].set_title('Distance from Centroid in PCA Space')
    axes[1, 0].grid(True, alpha=0.3)

    # Landscape metrics evolution
    selected_tokens = [checkpoints[i][3] for i in selected_indices]
    selected_loss_ranges = [tier1_results[checkpoints[i][1]]['metrics']['loss_range'] for i in selected_indices]
    selected_roughness = [tier1_results[checkpoints[i][1]]['metrics']['roughness'] for i in selected_indices]
    selected_basin = [tier1_results[checkpoints[i][1]]['metrics']['basin_diameter'] for i in selected_indices]

    ax2 = axes[1, 1]
    ax2.plot(selected_tokens, selected_loss_ranges, 'r-o', markersize=6, linewidth=2, label='Loss Range')
    ax2.set_xlabel('Training Tokens (B)')
    ax2.set_ylabel('Loss Range', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Landscape Metrics Evolution')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(selected_tokens, selected_basin, 'c-s', markersize=6, linewidth=2, label='Basin Diameter')
    ax2_twin.set_ylabel('Basin Diameter', color='c')
    ax2_twin.tick_params(axis='y', labelcolor='c')

    fig.suptitle('TinyLlama-1.1B Pre-Training Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tinyllama_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 9c: Side-by-side Tier 1 surfaces for selected checkpoints
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(6*len(selected_indices), 5))
    for plot_idx, ckpt_idx in enumerate(selected_indices):
        label = checkpoints[ckpt_idx][1]
        tokens_b = checkpoints[ckpt_idx][3]
        data = np.load(os.path.join(output_dir, f'surface_tier1_{label}.npz'))
        a, b, s = data['alphas'], data['betas'], data['surface']
        A_g, B_g = np.meshgrid(a, b)
        vmin_s = s.min()
        vmax_s = min(s.max(), s.min() + 3 * (np.median(s) - s.min() + 0.1))
        levels_s = np.linspace(vmin_s, vmax_s, 25)
        cs = axes[plot_idx].contourf(A_g, B_g, s, levels=levels_s, cmap='viridis')
        axes[plot_idx].contour(A_g, B_g, s, levels=levels_s, colors='white', linewidths=0.3, alpha=0.5)
        plt.colorbar(cs, ax=axes[plot_idx], shrink=0.8)
        m = tier1_results[label]['metrics']
        axes[plot_idx].set_title(f'{label} ({tokens_b}B)\nloss={checkpoint_losses[ckpt_idx]:.3f}, range={m["loss_range"]:.1f}')
        axes[plot_idx].set_xlabel(r'$\alpha$')
        axes[plot_idx].set_ylabel(r'$\beta$')
    plt.suptitle('Tier 1 Landscape Evolution During TinyLlama Pre-Training', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tinyllama_tier1_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Step 10: Save results ---
    total_time = time.time() - t_start
    results = {
        'experiment': 'TinyLlama-1.1B Pre-Training Trajectory (Complete)',
        'checkpoints': [c[0] for c in checkpoints],
        'checkpoint_labels': checkpoint_labels,
        'steps_k': [c[2] for c in checkpoints],
        'tokens_b': [c[3] for c in checkpoints],
        'projected_coords': projected_coords,
        'explained_variance': explained_var,
        'checkpoint_losses': checkpoint_losses,
        'inter_checkpoint_distances': distances,
        'total_distance_first_to_last': total_dist,
        'trajectory_surface_metrics': trajectory_metrics,
        'tier1_results': tier1_results,
        'grid_size': args.grid_size,
        'grid_range': float(grid_range_val),
        'n_params': n_params,
        'total_time_seconds': total_time,
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    np.savez(os.path.join(output_dir, 'trajectory_surface.npz'),
             alphas=alphas, betas=betas, surface=surface)

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    print("TinyLlama Full Pre-Training Trajectory Experiment Complete!")
    print("=" * 70)

    # Summary
    print("\n--- SUMMARY ---")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Parameters: {n_params:,}")
    print(f"Explained variance: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}, total={sum(explained_var):.4f}")
    print(f"\nPre-training trajectory:")
    for i, label in enumerate(checkpoint_labels):
        x, y = projected_coords[i]
        print(f"  {label}: coord=({x:.2f},{y:.2f}), loss={checkpoint_losses[i]:.4f}")
    print(f"\nInter-checkpoint distances:")
    for i, d in enumerate(distances):
        print(f"  {checkpoint_labels[i]} -> {checkpoint_labels[i+1]}: {d:.2f}")
    print(f"  Total (first->last): {total_dist:.2f}")
    print(f"\nTier 1 landscape metrics:")
    for label, data in tier1_results.items():
        m = data['metrics']
        print(f"  {label}: loss_range={m['loss_range']:.2f}, roughness={m['roughness']:.4f}, "
              f"basin_diameter={m['basin_diameter']:.4f}")


if __name__ == '__main__':
    main()
