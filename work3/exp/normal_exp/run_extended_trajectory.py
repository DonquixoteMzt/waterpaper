"""
run_extended_trajectory.py — Extended Training Trajectory + Post-Training Analysis

Creates a comprehensive training trajectory by fine-tuning Qwen3-0.6B-Base for 500 steps,
saving 10 checkpoints. This addresses three experimental needs:

1. Training Trajectory Visualization (MMSP Method A) - analogous to pre-training trajectory
   using intermediate checkpoints
2. Post-Training Effect Analysis (MMSP Method B) - comparing base vs. trained model
3. Independent Landscape Comparison (MMSP Method C) - comparing landscape geometry
   before vs. after training

Usage:
    python run_extended_trajectory.py --gpu 0
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
from multi_model import trajectory_pca, anchor_point_projection, compute_model_distance

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
    parser.add_argument('--train-steps', type=int, default=500)
    parser.add_argument('--checkpoint-interval', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/extended_trajectory'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Extended Training Trajectory + Post-Training Analysis")
    print(f"Steps: {args.train_steps}, Checkpoints every {args.checkpoint_interval} steps")
    print("=" * 70)
    t_start = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- Step 1: Load base model ---
    print("\nStep 1: Loading Qwen3-0.6B-Base...")
    base_name = "Qwen/Qwen3-0.6B-Base"
    model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_heads = getattr(model.config, 'num_attention_heads', None)
    head_dim = getattr(model.config, 'head_dim', None)
    if head_dim is None and num_heads:
        hidden_size = getattr(model.config, 'hidden_size', None)
        if hidden_size:
            head_dim = hidden_size // num_heads
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}, heads={num_heads}, head_dim={head_dim}")

    # --- Step 2: Prepare data ---
    print("\nStep 2: Preparing data...")
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in dataset['text'] if len(t.strip()) > 50]
    all_tokens = tokenizer(
        '\n'.join(texts[:2000]), return_tensors='pt', truncation=False
    )['input_ids'][0]

    # Training chunks
    train_chunks = []
    for i in range(0, len(all_tokens) - args.seq_len, args.seq_len):
        train_chunks.append(all_tokens[i:i + args.seq_len])
    print(f"  Training chunks: {len(train_chunks)} x {args.seq_len} tokens")

    # Evaluation data (from test split)
    dataset_test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts_test = [t for t in dataset_test['text'] if len(t.strip()) > 50]
    test_tokens = tokenizer(
        '\n'.join(texts_test[:200]), return_tensors='pt', truncation=False
    )['input_ids'][0]
    eval_chunks = []
    for i in range(0, len(test_tokens) - args.seq_len, args.seq_len):
        eval_chunks.append(test_tokens[i:i + args.seq_len])
    eval_chunks = eval_chunks[:50]
    eval_loader = DataLoader(ChunkDS(eval_chunks), batch_size=4, shuffle=False)
    print(f"  Eval chunks: {len(eval_chunks)} x {args.seq_len} tokens")

    # --- Step 3: Save baseline parameters ---
    print("\nStep 3: Saving baseline (step 0) parameters...")
    baseline_params = {name: param.data.cpu().clone() for name, param in model.named_parameters()}

    # Evaluate baseline loss
    model.to(device)
    baseline_loss = evaluate_loss(model, eval_loader, device, max_batches=args.max_eval_batches)
    print(f"  Baseline loss: {baseline_loss:.4f}")

    # --- Step 4: Fine-tune with checkpoint saving ---
    print(f"\nStep 4: Fine-tuning for {args.train_steps} steps (lr={args.lr})...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    train_loader = DataLoader(ChunkDS(train_chunks), batch_size=2, shuffle=True)
    train_iter = iter(train_loader)

    checkpoints = []  # List of (step, params, eval_loss)
    # Save step-0 checkpoint
    checkpoints.append((0, baseline_params, baseline_loss))
    training_losses = []

    for step in range(1, args.train_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch['input_ids'].to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        training_losses.append(loss.item())

        if step % 50 == 0:
            print(f"    Step {step}/{args.train_steps}, train_loss={loss.item():.4f}")

        if step % args.checkpoint_interval == 0:
            # Save checkpoint
            model.eval()
            with torch.no_grad():
                eval_loss = evaluate_loss(model, eval_loader, device, max_batches=args.max_eval_batches)
            ckpt_params = {name: param.data.cpu().clone() for name, param in model.named_parameters()}
            checkpoints.append((step, ckpt_params, eval_loss))
            print(f"    Checkpoint at step {step}: eval_loss={eval_loss:.4f}")
            model.train()

    model.eval()

    # Final evaluation
    with torch.no_grad():
        final_loss = evaluate_loss(model, eval_loader, device, max_batches=args.max_eval_batches)
    if checkpoints[-1][0] != args.train_steps:
        final_params = {name: param.data.cpu().clone() for name, param in model.named_parameters()}
        checkpoints.append((args.train_steps, final_params, final_loss))

    print(f"\n  Training complete. Final loss: {final_loss:.4f}")
    print(f"  Total checkpoints: {len(checkpoints)}")

    # --- Step 5: Trajectory-PCA (MMSP Method A) ---
    print("\nStep 5: Computing Trajectory-PCA...")
    all_ckpt_params = [ckpt[1] for ckpt in checkpoints]
    pca_directions, projected_coords, centroid, explained_var = trajectory_pca(all_ckpt_params, k=2)
    print(f"  Explained variance: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}")
    print(f"  Total explained: {sum(explained_var):.4f}")

    ckpt_labels = [f"Step {ckpt[0]}" for ckpt in checkpoints]
    for i, (step, _, loss) in enumerate(checkpoints):
        x, y = projected_coords[i]
        print(f"    Step {step}: ({x:.4f}, {y:.4f}), loss={loss:.4f}")

    # Apply TADN to PCA directions
    model.cpu()
    torch.cuda.empty_cache()
    units = get_normalization_units(model)
    pca_d1_tadn = apply_tadn(pca_directions[0], model, units, num_heads, head_dim)
    pca_d2_tadn = apply_tadn(pca_directions[1], model, units, num_heads, head_dim)

    # Evaluate trajectory surface
    # Set centroid parameters on model
    for name, param in model.named_parameters():
        if name in centroid:
            param.data.copy_(centroid[name].to(param.dtype))
    model.to(device)

    all_x = [c[0] for c in projected_coords]
    all_y = [c[1] for c in projected_coords]
    margin = 0.3
    grid_range_val = max(
        max(abs(min(all_x)), abs(max(all_x))),
        max(abs(min(all_y)), abs(max(all_y)))
    ) * (1 + margin)

    print(f"  Grid range: [-{grid_range_val:.4f}, {grid_range_val:.4f}]")
    alphas_traj, betas_traj, surface_traj = evaluate_2d_surface(
        model, pca_d1_tadn, pca_d2_tadn, eval_loader, device,
        grid_range=(-grid_range_val, grid_range_val),
        grid_size=args.grid_size,
        max_batches=args.max_eval_batches,
    )
    metrics_traj = compute_surface_metrics(alphas_traj, betas_traj, surface_traj)
    print(format_metrics_table(metrics_traj, "Trajectory PCA Surface"))

    # --- Step 6: Anchor-Point Projection (MMSP Method B) ---
    print("\nStep 6: Computing Anchor-Point Projection (base vs final)...")
    model.cpu()
    torch.cuda.empty_cache()

    final_params = checkpoints[-1][1]
    d1_anchor, d2_anchor, midpoint, dist = anchor_point_projection(baseline_params, final_params)
    print(f"  Parameter distance (base → final): {dist:.4f}")

    # Apply TADN
    d1_tadn = apply_tadn(d1_anchor, model, units, num_heads, head_dim)
    d2_tadn = apply_tadn(d2_anchor, model, units, num_heads, head_dim)

    # Set midpoint
    for name, param in model.named_parameters():
        if name in midpoint:
            param.data.copy_(midpoint[name].to(param.dtype))
    model.to(device)

    # Compute model projections on d1 axis
    d1_tadn_norm = math.sqrt(sum((d1_tadn[n].float() ** 2).sum().item() for n in d1_tadn))
    base_proj = sum(
        ((baseline_params[n].float() - midpoint[n].float()).flatten() @
         d1_tadn[n].float().flatten()).item()
        for n in d1_tadn
    ) / (d1_tadn_norm ** 2) if d1_tadn_norm > 0 else 0
    final_proj = sum(
        ((final_params[n].float() - midpoint[n].float()).flatten() @
         d1_tadn[n].float().flatten()).item()
        for n in d1_tadn
    ) / (d1_tadn_norm ** 2) if d1_tadn_norm > 0 else 0
    print(f"  Base projection: {base_proj:.6f}, Final projection: {final_proj:.6f}")

    # Evaluate midpoint loss
    midpoint_loss = evaluate_loss(model, eval_loader, device, max_batches=args.max_eval_batches)
    print(f"  Midpoint loss: {midpoint_loss:.4f}")

    grid_extent = max(abs(base_proj), abs(final_proj)) * 1.5
    grid_extent = max(grid_extent, 0.01)

    alphas_anchor, betas_anchor, surface_anchor = evaluate_2d_surface(
        model, d1_tadn, d2_tadn, eval_loader, device,
        grid_range=(-grid_extent, grid_extent),
        grid_size=args.grid_size,
        max_batches=args.max_eval_batches,
    )
    metrics_anchor = compute_surface_metrics(alphas_anchor, betas_anchor, surface_anchor)

    # 1D cross-section
    alphas_1d, losses_1d = evaluate_1d_curve(
        model, d1_tadn, eval_loader, device,
        alpha_range=(-grid_extent * 1.5, grid_extent * 1.5),
        n_points=51,
        max_batches=args.max_eval_batches,
    )

    # --- Step 7: Independent Tier 1 surfaces (MMSP Method C) ---
    print("\nStep 7: Independent Tier 1 surfaces for base and final models...")

    tier1_results = {}
    for tag, ckpt_params, ckpt_loss in [("base", baseline_params, baseline_loss),
                                          ("final", final_params, final_loss)]:
        # Set params
        model.cpu()
        torch.cuda.empty_cache()
        for name, param in model.named_parameters():
            if name in ckpt_params:
                param.data.copy_(ckpt_params[name].to(param.dtype))

        raw_d1 = generate_random_direction(model, seed=42)
        raw_d2 = generate_random_direction(model, seed=123)
        raw_d2 = orthogonalize_directions(raw_d1, raw_d2)
        units_ckpt = get_normalization_units(model)
        t1_d1 = apply_tadn(raw_d1, model, units_ckpt, num_heads, head_dim)
        t1_d2 = apply_tadn(raw_d2, model, units_ckpt, num_heads, head_dim)
        model.to(device)

        a_t1, b_t1, s_t1 = evaluate_2d_surface(
            model, t1_d1, t1_d2, eval_loader, device,
            grid_range=(-1.0, 1.0), grid_size=args.grid_size,
            max_batches=args.max_eval_batches,
        )
        m_t1 = compute_surface_metrics(a_t1, b_t1, s_t1)
        tier1_results[tag] = {
            'loss': ckpt_loss,
            'metrics': m_t1,
        }
        print(f"  {tag}: loss={ckpt_loss:.4f}, range={m_t1['loss_range']:.2f}, "
              f"roughness={m_t1['roughness']:.4f}, basin_diam={m_t1['basin_diameter']:.4f}")

        np.savez(os.path.join(output_dir, f'surface_tier1_{tag}.npz'),
                 alphas=a_t1, betas=b_t1, surface=s_t1)
        del raw_d1, raw_d2, t1_d1, t1_d2
        gc.collect()

    # --- Step 8: Inter-checkpoint distances ---
    print("\nStep 8: Computing inter-checkpoint distances...")
    inter_dists = []
    for i in range(1, len(checkpoints)):
        d = 0.0
        for name in checkpoints[i][1]:
            d += ((checkpoints[i][1][name].float() - checkpoints[i-1][1][name].float()) ** 2).sum().item()
        inter_dists.append(math.sqrt(d))
        print(f"  Step {checkpoints[i-1][0]} → {checkpoints[i][0]}: dist={inter_dists[-1]:.4f}")

    # --- Step 9: Visualizations ---
    print("\nStep 9: Creating visualizations...")

    # 9a: Trajectory contour with training path
    fig = plt.figure(figsize=(18, 7))
    ax1 = fig.add_subplot(121)
    A, B = np.meshgrid(alphas_traj, betas_traj)
    vmin = surface_traj.min()
    vmax = min(surface_traj.max(), surface_traj.min() + 3 * (np.median(surface_traj) - surface_traj.min() + 0.1))
    levels = np.linspace(vmin, vmax, 30)
    cs = ax1.contourf(A, B, surface_traj, levels=levels, cmap='viridis')
    ax1.contour(A, B, surface_traj, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax1, label='Loss')

    xs = [c[0] for c in projected_coords]
    ys = [c[1] for c in projected_coords]
    ax1.plot(xs, ys, 'r-o', markersize=8, linewidth=2, zorder=5, label='Training trajectory')
    for i, (x, y) in enumerate(projected_coords):
        ax1.annotate(f'Step {checkpoints[i][0]}', (x, y), textcoords="offset points",
                     xytext=(8, 8), fontsize=7, color='white', fontweight='bold', zorder=6)
    ax1.set_xlabel(f'PC1 (explained var: {explained_var[0]:.1%})')
    ax1.set_ylabel(f'PC2 (explained var: {explained_var[1]:.1%})')
    ax1.set_title(f'Training Trajectory ({args.train_steps} steps)\n(Trajectory-PCA, MMSP Method A)')
    ax1.legend(fontsize=9)

    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.plot_surface(A, B, surface_traj, cmap='viridis', alpha=0.7, vmin=vmin, vmax=vmax)
    zs = []
    for (x, y) in projected_coords:
        i_x = np.argmin(np.abs(alphas_traj - x))
        i_y = np.argmin(np.abs(betas_traj - y))
        zs.append(surface_traj[i_y, i_x])
    ax3d.plot(xs, ys, zs, 'r-o', markersize=6, linewidth=2, zorder=5)
    ax3d.set_xlabel('PC1'); ax3d.set_ylabel('PC2'); ax3d.set_zlabel('Loss')
    ax3d.set_title('3D Training Trajectory')
    ax3d.view_init(elev=30, azim=225)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_pca.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 9b: Loss evolution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    steps_list = [ckpt[0] for ckpt in checkpoints]
    losses_list = [ckpt[2] for ckpt in checkpoints]
    axes[0].plot(steps_list, losses_list, 'b-o', markersize=6, linewidth=2)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('WikiText-2 Eval Loss')
    axes[0].set_title('Loss Evolution During Training')
    axes[0].grid(True, alpha=0.3)

    # Distance from centroid
    dists_from_centroid = [math.sqrt(x**2 + y**2) for x, y in projected_coords]
    axes[1].plot(steps_list, dists_from_centroid, 'g-s', markersize=6, linewidth=2)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Distance from Centroid (PCA space)')
    axes[1].set_title('Parameter Movement')
    axes[1].grid(True, alpha=0.3)

    # Inter-checkpoint distances
    mid_steps = [(checkpoints[i][0] + checkpoints[i+1][0]) / 2 for i in range(len(inter_dists))]
    axes[2].bar(range(len(inter_dists)), inter_dists, color='steelblue', alpha=0.8)
    axes[2].set_xticks(range(len(inter_dists)))
    axes[2].set_xticklabels([f'{checkpoints[i][0]}→{checkpoints[i+1][0]}'
                              for i in range(len(inter_dists))], rotation=45, fontsize=8)
    axes[2].set_ylabel('L2 Distance')
    axes[2].set_title('Inter-Checkpoint Distances')
    axes[2].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 9c: Anchor-point surface with 1D cross-section
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    A_a, B_a = np.meshgrid(alphas_anchor, betas_anchor)
    vmin_a = surface_anchor.min()
    vmax_a = min(surface_anchor.max(), surface_anchor.min() + 3 * (np.median(surface_anchor) - surface_anchor.min() + 0.1))
    levels_a = np.linspace(vmin_a, vmax_a, 25)
    cs = ax1.contourf(A_a, B_a, surface_anchor, levels=levels_a, cmap='viridis')
    ax1.contour(A_a, B_a, surface_anchor, levels=levels_a, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax1, label='Loss')
    ax1.plot(base_proj, 0, 'r*', markersize=15, zorder=5, label=f'Base (loss={baseline_loss:.3f})')
    ax1.plot(final_proj, 0, 'b*', markersize=15, zorder=5, label=f'Final (loss={final_loss:.3f})')
    ax1.set_xlabel(r'$d_1$ (Base $\rightarrow$ Final)')
    ax1.set_ylabel(r'$d_2$ (orthogonal)')
    ax1.set_title('Post-Training Effect\n(Anchor-Point, MMSP Method B)')
    ax1.legend(fontsize=8)

    ax2.plot(alphas_1d, losses_1d, 'b-', linewidth=2, label='Loss along Base→Final')
    ax2.axvline(x=base_proj, color='r', linestyle='--', label=f'Base (α={base_proj:.4f})')
    ax2.axvline(x=final_proj, color='g', linestyle='--', label=f'Final (α={final_proj:.4f})')
    ax2.set_xlabel(r'$\alpha$ along $d_1$')
    ax2.set_ylabel('Loss')
    ax2.set_title('1D Loss Profile: Base → Final')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anchor_point.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 9d: Side-by-side Tier 1 comparison (base vs final)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (tag, title) in enumerate([("base", "Base Model (Step 0)"),
                                          ("final", f"Trained (Step {args.train_steps})")]):
        data = np.load(os.path.join(output_dir, f'surface_tier1_{tag}.npz'))
        a, b, s = data['alphas'], data['betas'], data['surface']
        A_g, B_g = np.meshgrid(a, b)
        vmin_s = s.min()
        vmax_s = min(s.max(), s.min() + 3 * (np.median(s) - s.min() + 0.1))
        levels_s = np.linspace(vmin_s, vmax_s, 25)
        cs_s = axes[idx].contourf(A_g, B_g, s, levels=levels_s, cmap='viridis')
        axes[idx].contour(A_g, B_g, s, levels=levels_s, colors='white', linewidths=0.3, alpha=0.5)
        plt.colorbar(cs_s, ax=axes[idx], shrink=0.8)
        m = tier1_results[tag]['metrics']
        axes[idx].set_title(f'{title}\nrange={m["loss_range"]:.1f}, rough={m["roughness"]:.3f}')
        axes[idx].set_xlabel(r'$\alpha$'); axes[idx].set_ylabel(r'$\beta$')
        axes[idx].plot(0, 0, 'r*', markersize=12)
    plt.suptitle('Base vs Trained: Independent Landscape Comparison (Method C)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'base_vs_trained.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Step 10: Save results ---
    total_time = time.time() - t_start

    results = {
        'experiment': 'Extended Training Trajectory + Post-Training Analysis',
        'model': base_name,
        'n_params': n_params,
        'train_steps': args.train_steps,
        'checkpoint_interval': args.checkpoint_interval,
        'learning_rate': args.lr,
        'n_checkpoints': len(checkpoints),
        'checkpoint_steps': [ckpt[0] for ckpt in checkpoints],
        'checkpoint_losses': [ckpt[2] for ckpt in checkpoints],
        'inter_checkpoint_distances': inter_dists,
        'baseline_loss': baseline_loss,
        'final_loss': final_loss,
        'loss_improvement': baseline_loss - final_loss,
        'loss_improvement_pct': (baseline_loss - final_loss) / baseline_loss * 100,
        'parameter_distance': dist,
        # Trajectory PCA
        'trajectory_pca': {
            'projected_coords': projected_coords,
            'explained_variance': explained_var,
            'total_explained': sum(explained_var),
            'surface_metrics': metrics_traj,
            'grid_range': float(grid_range_val),
        },
        # Anchor-Point
        'anchor_point': {
            'base_proj': base_proj,
            'final_proj': final_proj,
            'midpoint_loss': midpoint_loss,
            'surface_metrics': metrics_anchor,
            'grid_range': float(grid_extent),
        },
        # Independent surfaces
        'tier1_base': tier1_results['base'],
        'tier1_final': tier1_results['final'],
        # Training stats
        'training_losses_final_50': training_losses[-50:],
        'total_time_seconds': total_time,
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    np.savez(os.path.join(output_dir, 'trajectory_surface.npz'),
             alphas=alphas_traj, betas=betas_traj, surface=surface_traj)
    np.savez(os.path.join(output_dir, 'anchor_surface.npz'),
             alphas=alphas_anchor, betas=betas_anchor, surface=surface_anchor)

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    print("Extended Trajectory Experiment Complete!")
    print("=" * 70)

    print("\n--- SUMMARY ---")
    print(f"Model: {base_name} ({n_params:,} parameters)")
    print(f"Training: {args.train_steps} steps, lr={args.lr}")
    print(f"Loss: {baseline_loss:.4f} → {final_loss:.4f} ({results['loss_improvement_pct']:.1f}% improvement)")
    print(f"Parameter distance: {dist:.4f}")
    print(f"\nTrajectory-PCA:")
    print(f"  Explained variance: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}, Total={sum(explained_var):.4f}")
    print(f"  Surface: range={metrics_traj['loss_range']:.2f}, roughness={metrics_traj['roughness']:.4f}")
    print(f"\nAnchor-Point:")
    print(f"  Base proj={base_proj:.6f}, Final proj={final_proj:.6f}")
    print(f"  Midpoint loss={midpoint_loss:.4f}")
    print(f"  Surface: range={metrics_anchor['loss_range']:.2f}, roughness={metrics_anchor['roughness']:.4f}")
    print(f"\nIndependent Tier 1 surfaces:")
    for tag in ['base', 'final']:
        m = tier1_results[tag]['metrics']
        print(f"  {tag}: loss={tier1_results[tag]['loss']:.4f}, range={m['loss_range']:.2f}, "
              f"roughness={m['roughness']:.4f}, basin={m['basin_diameter']:.4f}")


if __name__ == '__main__':
    main()
