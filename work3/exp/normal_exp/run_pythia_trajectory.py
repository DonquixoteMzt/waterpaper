"""
run_pythia_trajectory.py — Pythia-1b Pre-Training Trajectory Visualization (MMSP Method A).

Downloads Pythia-1b checkpoint series from EleutherAI (using git revisions)
and runs Trajectory-PCA to visualize training evolution on a shared 2D plane.

Usage:
    python run_pythia_trajectory.py --gpu 0
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

from normalization import get_normalization_units, apply_tadn
from direction_selection import generate_random_direction, orthogonalize_directions
from grid_evaluation import evaluate_loss, evaluate_2d_surface
from metrics import compute_surface_metrics, format_metrics_table
from multi_model import trajectory_pca

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_pythia_checkpoint(revision, device='cpu'):
    """Load a Pythia-1b checkpoint at a specific training step."""
    from transformers import AutoModelForCausalLM
    model_name = "EleutherAI/pythia-1b"
    print(f"  Loading {model_name} revision={revision}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    params = {name: param.data.cpu().clone() for name, param in model.named_parameters()}
    return model, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--grid-size', type=int, default=31)
    parser.add_argument('--max-eval-batches', type=int, default=5)
    parser.add_argument('--seq-len', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/pythia_trajectory'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Pythia-1b Pre-Training Trajectory Visualization (MMSP Method A)")
    print("=" * 70)
    t_start = time.time()

    # Pythia-1b checkpoint series (ordered by training step)
    # Pythia-1b was trained on The Pile for 143K steps (~300B tokens)
    checkpoints = [
        ("step1000", "1K steps (~2.1B tokens)"),
        ("step16000", "16K steps (~33.6B tokens)"),
        ("step64000", "64K steps (~134B tokens)"),
        ("step100000", "100K steps (~210B tokens)"),
        ("step143000", "143K steps (~300B tokens, final)"),
    ]

    # --- Step 1: Load all checkpoints' parameters ---
    print("\nStep 1: Loading checkpoint parameters...")
    all_params = []
    checkpoint_labels = []
    ref_model = None
    ref_tokenizer = None

    for revision, label in checkpoints:
        model, params = load_pythia_checkpoint(revision)
        all_params.append(params)
        checkpoint_labels.append(label)
        if ref_model is None:
            ref_model = model  # Keep first checkpoint model for reference
            from transformers import AutoTokenizer
            ref_tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/pythia-1b", trust_remote_code=True
            )
            if ref_tokenizer.pad_token is None:
                ref_tokenizer.pad_token = ref_tokenizer.eos_token
        else:
            del model
        gc.collect()

    n_params = sum(p.numel() for p in all_params[0].values())
    print(f"  Loaded {len(all_params)} checkpoints, {n_params:,} parameters each")

    # --- Step 2: Run Trajectory-PCA ---
    print("\nStep 2: Computing Trajectory-PCA...")
    pca_directions, projected_coords, centroid, explained_var = trajectory_pca(all_params, k=2)
    print(f"  Explained variance: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}")
    print(f"  Total explained: {sum(explained_var):.4f}")

    print("  Projected coordinates:")
    for i, (x, y) in enumerate(projected_coords):
        print(f"    {checkpoint_labels[i]}: ({x:.4f}, {y:.4f})")

    # --- Step 3: Apply TADN to PCA directions ---
    print("\nStep 3: Applying TADN normalization...")
    # Use the last checkpoint model for TADN reference
    from transformers import AutoModelForCausalLM
    tadn_model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-1b", revision=checkpoints[-1][0],
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    )
    tadn_model.eval()

    num_heads = getattr(tadn_model.config, 'num_attention_heads', None)
    head_dim = getattr(tadn_model.config, 'head_dim', None)
    if head_dim is None and num_heads:
        hidden_size = getattr(tadn_model.config, 'hidden_size', None)
        if hidden_size:
            head_dim = hidden_size // num_heads
    print(f"  Architecture: num_heads={num_heads}, head_dim={head_dim}")

    units = get_normalization_units(tadn_model)

    # Apply TADN to PCA directions
    pca_d1_tadn = apply_tadn(pca_directions[0], tadn_model, units, num_heads, head_dim)
    pca_d2_tadn = apply_tadn(pca_directions[1], tadn_model, units, num_heads, head_dim)

    del tadn_model
    gc.collect()

    # --- Step 4: Prepare evaluation data ---
    print("\nStep 4: Preparing evaluation data...")
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

    from torch.utils.data import Dataset, DataLoader

    class ChunkDS(Dataset):
        def __init__(self, chunks):
            self.chunks = chunks
        def __len__(self):
            return len(self.chunks)
        def __getitem__(self, idx):
            return {'input_ids': self.chunks[idx],
                    'attention_mask': torch.ones(len(self.chunks[idx]), dtype=torch.long)}

    eval_loader = DataLoader(ChunkDS(chunks[:n_eval]), batch_size=4, shuffle=False)

    # --- Step 5: Build centroid model and evaluate loss surface ---
    print("\nStep 5: Building centroid model and evaluating surface...")
    centroid_model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-1b", revision=checkpoints[-1][0],
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    )
    centroid_model.eval()

    # Set centroid parameters
    for name, param in centroid_model.named_parameters():
        if name in centroid:
            param.data.copy_(centroid[name].to(param.dtype))

    centroid_model.to(device)

    # Determine grid range from projected coordinates
    all_x = [c[0] for c in projected_coords]
    all_y = [c[1] for c in projected_coords]
    margin = 0.3
    x_range = max(abs(min(all_x)), abs(max(all_x))) * (1 + margin)
    y_range = max(abs(min(all_y)), abs(max(all_y))) * (1 + margin)
    grid_range_val = max(x_range, y_range)

    print(f"  Grid range: [-{grid_range_val:.4f}, {grid_range_val:.4f}]")
    print(f"  Grid size: {args.grid_size}x{args.grid_size}")

    alphas, betas, surface = evaluate_2d_surface(
        centroid_model, pca_d1_tadn, pca_d2_tadn, eval_loader, device,
        grid_range=(-grid_range_val, grid_range_val),
        grid_size=args.grid_size,
        max_batches=args.max_eval_batches,
    )

    # --- Step 6: Evaluate loss at each checkpoint ---
    print("\nStep 6: Evaluating loss at each checkpoint position...")
    checkpoint_losses = []
    for i, (revision, label) in enumerate(checkpoints):
        _, ckpt_params = load_pythia_checkpoint(revision)
        for name, param in centroid_model.named_parameters():
            if name in ckpt_params:
                param.data.copy_(ckpt_params[name].to(param.dtype).to(param.device))
        loss = evaluate_loss(centroid_model, eval_loader, device, max_batches=args.max_eval_batches)
        checkpoint_losses.append(loss)
        print(f"  {label}: loss={loss:.4f}")
        del ckpt_params
        gc.collect()

    # Restore centroid
    for name, param in centroid_model.named_parameters():
        if name in centroid:
            param.data.copy_(centroid[name].to(param.dtype).to(param.device))

    # --- Step 6b: Also evaluate Tier 1 surfaces at earliest and latest checkpoints ---
    print("\nStep 6b: Evaluating independent Tier 1 surfaces for earliest and latest checkpoints...")
    tier1_results = {}
    for idx, (revision, label) in enumerate([checkpoints[0], checkpoints[-1]]):
        tag = "early" if idx == 0 else "final"
        _, ckpt_params = load_pythia_checkpoint(revision)
        for name, param in centroid_model.named_parameters():
            if name in ckpt_params:
                param.data.copy_(ckpt_params[name].to(param.dtype).to(param.device))

        # Generate Tier 1 directions for this checkpoint
        centroid_model.cpu()
        torch.cuda.empty_cache()
        raw_d1 = generate_random_direction(centroid_model, seed=42)
        raw_d2 = generate_random_direction(centroid_model, seed=123)
        raw_d2 = orthogonalize_directions(raw_d1, raw_d2)
        units_ckpt = get_normalization_units(centroid_model)
        nh = num_heads
        hd = head_dim
        t1_d1 = apply_tadn(raw_d1, centroid_model, units_ckpt, nh, hd)
        t1_d2 = apply_tadn(raw_d2, centroid_model, units_ckpt, nh, hd)
        centroid_model.to(device)

        a_t1, b_t1, s_t1 = evaluate_2d_surface(
            centroid_model, t1_d1, t1_d2, eval_loader, device,
            grid_range=(-1.0, 1.0), grid_size=args.grid_size,
            max_batches=args.max_eval_batches,
        )
        m_t1 = compute_surface_metrics(a_t1, b_t1, s_t1)
        tier1_results[tag] = {
            'label': label,
            'revision': revision,
            'metrics': m_t1,
        }
        print(f"  {tag} ({label}): range={m_t1['loss_range']:.2f}, "
              f"roughness={m_t1['roughness']:.4f}, basin_diam={m_t1['basin_diameter']:.4f}")

        np.savez(os.path.join(output_dir, f'surface_tier1_{tag}.npz'),
                 alphas=a_t1, betas=b_t1, surface=s_t1)
        del raw_d1, raw_d2, t1_d1, t1_d2, ckpt_params
        gc.collect()

    # Restore centroid
    for name, param in centroid_model.named_parameters():
        if name in centroid:
            param.data.copy_(centroid[name].to(param.dtype).to(param.device))

    # --- Step 7: Compute surface metrics ---
    print("\nStep 7: Computing surface metrics...")
    metrics = compute_surface_metrics(alphas, betas, surface)
    print(format_metrics_table(metrics, "Pythia-1b Trajectory PCA"))

    # --- Step 8: Visualization ---
    print("\nStep 8: Creating visualizations...")

    # 8a: Contour plot with checkpoint trajectory
    fig = plt.figure(figsize=(18, 7))

    ax1 = fig.add_subplot(121)
    A, B = np.meshgrid(alphas, betas)
    vmin = surface.min()
    vmax = min(surface.max(), surface.min() + 3 * (np.median(surface) - surface.min() + 0.1))
    levels = np.linspace(vmin, vmax, 30)
    cs = ax1.contourf(A, B, surface, levels=levels, cmap='viridis')
    ax1.contour(A, B, surface, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax1, label='Loss')

    # Plot training trajectory
    xs = [c[0] for c in projected_coords]
    ys = [c[1] for c in projected_coords]
    ax1.plot(xs, ys, 'r-o', markersize=8, linewidth=2, zorder=5, label='Training trajectory')
    for i, (x, y) in enumerate(projected_coords):
        short_label = checkpoint_labels[i].split('(')[0].strip()
        ax1.annotate(short_label, (x, y), textcoords="offset points",
                     xytext=(8, 8), fontsize=7, color='white',
                     fontweight='bold', zorder=6)

    ax1.set_xlabel(f'PC1 (explained var: {explained_var[0]:.1%})')
    ax1.set_ylabel(f'PC2 (explained var: {explained_var[1]:.1%})')
    ax1.set_title('Pythia-1b Pre-Training Trajectory\n(Trajectory-PCA, MMSP Method A)')
    ax1.legend(fontsize=9)

    # Right: 3D surface with trajectory
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.plot_surface(A, B, surface, cmap='viridis', alpha=0.7, vmin=vmin, vmax=vmax)

    zs = []
    for (x, y) in projected_coords:
        i_x = np.argmin(np.abs(alphas - x))
        i_y = np.argmin(np.abs(betas - y))
        zs.append(surface[i_y, i_x])
    ax3d.plot(xs, ys, zs, 'r-o', markersize=6, linewidth=2, zorder=5)
    ax3d.set_xlabel('PC1')
    ax3d.set_ylabel('PC2')
    ax3d.set_zlabel('Loss')
    ax3d.set_title('3D Training Trajectory')
    ax3d.view_init(elev=30, azim=225)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pythia_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 8b: Loss evolution along training
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = [1, 16, 64, 100, 143]  # in K
    ax1.plot(steps, checkpoint_losses, 'b-o', markersize=8, linewidth=2)
    ax1.set_xlabel('Training Steps (K)')
    ax1.set_ylabel('WikiText-2 Loss')
    ax1.set_title('Pythia-1b Loss Evolution During Pre-Training')
    ax1.grid(True, alpha=0.3)

    # Distance from centroid in PCA space
    dists = [math.sqrt(x**2 + y**2) for x, y in projected_coords]
    ax2.plot(steps, dists, 'g-s', markersize=8, linewidth=2)
    ax2.set_xlabel('Training Steps (K)')
    ax2.set_ylabel('Distance from Centroid (PCA space)')
    ax2.set_title('Parameter Movement in PCA Space')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pythia_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 8c: Inter-checkpoint distances
    inter_dists = []
    for i in range(1, len(all_params)):
        d = 0.0
        for name in all_params[i]:
            d += ((all_params[i][name].float() - all_params[i-1][name].float()) ** 2).sum().item()
        inter_dists.append(math.sqrt(d))

    fig, ax = plt.subplots(figsize=(10, 5))
    mid_steps = [(steps[i] + steps[i+1]) / 2 for i in range(len(steps)-1)]
    ax.bar(range(len(inter_dists)),
           inter_dists,
           tick_label=[f'{checkpoint_labels[i].split("(")[0].strip()}\n→\n{checkpoint_labels[i+1].split("(")[0].strip()}'
                       for i in range(len(inter_dists))],
           color='steelblue', alpha=0.8)
    ax.set_ylabel('L2 Parameter Distance')
    ax.set_title('Inter-Checkpoint Parameter Distances (Pythia-1b)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pythia_distances.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Step 9: Save results ---
    total_time = time.time() - t_start
    results = {
        'experiment': 'Pythia-1b Pre-Training Trajectory (MMSP Method A)',
        'model': 'EleutherAI/pythia-1b',
        'checkpoints': [c[0] for c in checkpoints],
        'checkpoint_labels': checkpoint_labels,
        'projected_coords': projected_coords,
        'explained_variance': explained_var,
        'checkpoint_losses': checkpoint_losses,
        'inter_checkpoint_distances': inter_dists,
        'surface_metrics': metrics,
        'tier1_early_metrics': tier1_results.get('early', {}).get('metrics', {}),
        'tier1_final_metrics': tier1_results.get('final', {}).get('metrics', {}),
        'grid_size': args.grid_size,
        'grid_range': float(grid_range_val),
        'n_params': n_params,
        'total_time_seconds': total_time,
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    np.savez(os.path.join(output_dir, 'surface.npz'),
             alphas=alphas, betas=betas, surface=surface)

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to {output_dir}/")
    print("=" * 70)
    print("Pythia-1b Trajectory Experiment Complete!")
    print("=" * 70)

    # Print summary
    print("\n--- SUMMARY ---")
    print(f"Model: EleutherAI/pythia-1b ({n_params:,} parameters)")
    print(f"Explained variance: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}")
    print(f"Total explained: {sum(explained_var):.4f}")
    for i, label in enumerate(checkpoint_labels):
        x, y = projected_coords[i]
        print(f"  {label}: coord=({x:.4f},{y:.4f}), loss={checkpoint_losses[i]:.4f}")
    print(f"Surface metrics: loss_range={metrics['loss_range']:.4f}, "
          f"roughness={metrics['roughness']:.4f}, "
          f"basin_diameter={metrics['basin_diameter']:.4f}")
    if tier1_results:
        for tag, info in tier1_results.items():
            m = info['metrics']
            print(f"Tier 1 ({tag}): range={m['loss_range']:.2f}, "
                  f"roughness={m['roughness']:.4f}, basin={m['basin_diameter']:.4f}")
    print(f"Inter-checkpoint distances: {[f'{d:.4f}' for d in inter_dists]}")


if __name__ == '__main__':
    main()
