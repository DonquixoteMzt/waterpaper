"""
run_full_ablations.py — Complete ablation studies for LLMScape.

Runs all three incomplete ablation studies:
1. PCA sample size sweep (N=10,25,50,100,150,200)
2. Grid resolution sweep (11,21,31,51)
3. Evaluation data size sweep (2,5,10,20,50 chunks)

Usage:
    python run_full_ablations.py --gpu 3
"""

import os
import sys
import json
import time
import gc
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from normalization import get_normalization_units, apply_tadn
from direction_selection import (
    generate_random_direction, generate_tier1_directions,
    orthogonalize_directions, gradient_pca_with_convergence,
)
from grid_evaluation import evaluate_loss, evaluate_2d_surface
from metrics import compute_surface_metrics, format_metrics_table

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
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--seq-len', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/ablations'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Complete Ablation Studies for LLMScape")
    print("=" * 70)
    t_start = time.time()

    model_name = "Qwen/Qwen3-0.6B-Base"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    num_heads = getattr(model.config, 'num_attention_heads', None)
    head_dim = getattr(model.config, 'head_dim', None)
    if head_dim is None and num_heads:
        hidden_size = getattr(model.config, 'hidden_size', None)
        if hidden_size:
            head_dim = hidden_size // num_heads
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}, heads={num_heads}, head_dim={head_dim}")

    # Prepare data
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in dataset['text'] if len(t.strip()) > 50]
    all_tokens = tokenizer(
        '\n'.join(texts[:200]), return_tensors='pt', truncation=False
    )['input_ids'][0]

    chunks = []
    for i in range(0, len(all_tokens) - args.seq_len, args.seq_len):
        chunks.append(all_tokens[i:i + args.seq_len])
    print(f"  Total chunks: {len(chunks)}")

    # Full eval loader
    full_eval_loader = DataLoader(ChunkDS(chunks[:50]), batch_size=4, shuffle=False)
    baseline_loss = evaluate_loss(model, full_eval_loader, device, max_batches=5)
    print(f"  Baseline loss: {baseline_loss:.4f}")

    # Generate base directions (same as main experiments)
    units = get_normalization_units(model)
    d1_raw, d2_raw = generate_tier1_directions(model, seed1=42, seed2=123)
    tier1_d1 = apply_tadn(d1_raw, model, units, num_heads, head_dim)
    tier1_d2 = apply_tadn(d2_raw, model, units, num_heads, head_dim)

    results = {}

    # ====================================================================
    # ABLATION 1: PCA Sample Size Sweep
    # ====================================================================
    print("\n" + "=" * 70)
    print("ABLATION 1: PCA Sample Size Sweep")
    print("=" * 70)

    pca_n_values = [10, 25, 50, 100, 150, 200]
    grad_loader = DataLoader(ChunkDS(chunks[:200]), batch_size=1, shuffle=False)
    pca_sweep_results = {}

    # First collect all 200 gradients
    print("Collecting gradients for PCA sweep...")
    pca_results_full, _ = gradient_pca_with_convergence(
        model, grad_loader, device,
        n_max=200,
        checkpoints=pca_n_values,
        k=2,
        convergence_threshold_deg=1.0,  # Very strict to avoid early stopping
    )

    # Evaluate surfaces for each N
    for N in pca_n_values:
        if N not in pca_results_full:
            continue
        print(f"\n--- N={N}: Evaluating 2D surface ---")
        # Get PCA directions for this N
        d_flat_list = pca_results_full[N]['directions_flat']
        pca_d1 = {}
        pca_d2 = {}
        offset = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                numel = param.numel()
                pca_d1[name] = d_flat_list[0][offset:offset + numel].reshape(param.shape).to(param.device)
                pca_d2[name] = d_flat_list[1][offset:offset + numel].reshape(param.shape).to(param.device)
                offset += numel

        # Apply TADN
        pca_d1_tadn = apply_tadn(pca_d1, model, units, num_heads, head_dim)
        pca_d2_tadn = apply_tadn(pca_d2, model, units, num_heads, head_dim)

        # Evaluate 21×21 surface
        a, b, s = evaluate_2d_surface(
            model, pca_d1_tadn, pca_d2_tadn, full_eval_loader, device,
            grid_range=(-1.0, 1.0), grid_size=21, max_batches=5,
        )
        m = compute_surface_metrics(a, b, s)

        pca_sweep_results[N] = {
            'explained_ratios': pca_results_full[N]['explained_ratios'],
            'subspace_angle': pca_results_full[N].get('subspace_angle_from_prev'),
            'loss_range': m['loss_range'],
            'roughness': m['roughness'],
            'basin_diameter': m['basin_diameter'],
            'curvature_ratio': m['curvature_ratio'],
            'convexity_ratio': m['convexity_ratio'],
        }
        print(f"  N={N}: loss_range={m['loss_range']:.2f}, roughness={m['roughness']:.4f}, "
              f"explained=[{pca_results_full[N]['explained_ratios'][0]:.4f}, {pca_results_full[N]['explained_ratios'][1]:.4f}]")

    results['ablation1_pca_sample_size'] = pca_sweep_results

    # Plot PCA sample size results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    Ns = sorted(pca_sweep_results.keys())
    axes[0,0].plot(Ns, [pca_sweep_results[n]['loss_range'] for n in Ns], 'b-o', linewidth=2)
    axes[0,0].set_xlabel('N (gradient samples)')
    axes[0,0].set_ylabel('Loss Range')
    axes[0,0].set_title('Loss Range vs PCA Sample Size')
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(Ns, [pca_sweep_results[n]['roughness'] for n in Ns], 'r-s', linewidth=2)
    axes[0,1].set_xlabel('N (gradient samples)')
    axes[0,1].set_ylabel('Roughness')
    axes[0,1].set_title('Roughness vs PCA Sample Size')
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(Ns, [pca_sweep_results[n]['explained_ratios'][0] for n in Ns], 'g-^', linewidth=2, label='PC1')
    axes[1,0].plot(Ns, [pca_sweep_results[n]['explained_ratios'][1] for n in Ns], 'g-v', linewidth=2, label='PC2')
    axes[1,0].set_xlabel('N (gradient samples)')
    axes[1,0].set_ylabel('Explained Variance Ratio')
    axes[1,0].set_title('Explained Variance vs N')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    angles = [pca_sweep_results[n]['subspace_angle'] for n in Ns if pca_sweep_results[n]['subspace_angle'] is not None]
    angle_ns = [n for n in Ns if pca_sweep_results[n]['subspace_angle'] is not None]
    if angles:
        axes[1,1].plot(angle_ns, angles, 'm-D', linewidth=2)
        axes[1,1].axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='5° threshold')
        axes[1,1].set_xlabel('N (gradient samples)')
        axes[1,1].set_ylabel('Subspace Angle (degrees)')
        axes[1,1].set_title('PCA Subspace Convergence')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

    plt.suptitle('Ablation 1: PCA Sample Size Sweep', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation1_pca_sweep.png'), dpi=150, bbox_inches='tight')
    plt.close()

    del pca_results_full
    gc.collect()

    # ====================================================================
    # ABLATION 2: Grid Resolution Sweep
    # ====================================================================
    print("\n" + "=" * 70)
    print("ABLATION 2: Grid Resolution Sweep")
    print("=" * 70)

    grid_sizes = [11, 21, 31, 51]
    grid_sweep_results = {}

    for gs in grid_sizes:
        print(f"\n--- Grid {gs}x{gs} ---")
        a, b, s = evaluate_2d_surface(
            model, tier1_d1, tier1_d2, full_eval_loader, device,
            grid_range=(-1.0, 1.0), grid_size=gs, max_batches=5,
        )
        m = compute_surface_metrics(a, b, s)
        grid_sweep_results[gs] = {
            'loss_range': m['loss_range'],
            'roughness': m['roughness'],
            'basin_diameter': m['basin_diameter'],
            'curvature_ratio': m['curvature_ratio'],
            'convexity_ratio': m['convexity_ratio'],
            'center_loss': m['center_loss'],
        }
        print(f"  {gs}x{gs}: loss_range={m['loss_range']:.2f}, roughness={m['roughness']:.4f}, "
              f"basin_diam={m['basin_diameter']:.4f}")

    results['ablation2_grid_resolution'] = grid_sweep_results

    # Plot grid resolution results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    gs_list = sorted(grid_sweep_results.keys())
    gs_labels = [f'{g}×{g}' for g in gs_list]

    axes[0].bar(gs_labels, [grid_sweep_results[g]['loss_range'] for g in gs_list],
                color='#3498db', alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Loss Range')
    axes[0].set_title('Loss Range vs Grid Resolution')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(gs_labels, [grid_sweep_results[g]['roughness'] for g in gs_list],
                color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Roughness')
    axes[1].set_title('Roughness vs Grid Resolution')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(gs_labels, [grid_sweep_results[g]['basin_diameter'] for g in gs_list],
                color='#2ecc71', alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('Basin Diameter')
    axes[2].set_title('Basin Diameter vs Grid Resolution')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Ablation 2: Grid Resolution Sweep', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation2_grid_resolution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ====================================================================
    # ABLATION 3: Evaluation Data Size Sweep
    # ====================================================================
    print("\n" + "=" * 70)
    print("ABLATION 3: Evaluation Data Size Sweep")
    print("=" * 70)

    chunk_counts = [2, 5, 10, 20, 50]
    data_size_results = {}

    for nc in chunk_counts:
        print(f"\n--- {nc} chunks ---")
        sub_loader = DataLoader(ChunkDS(chunks[:nc]), batch_size=min(4, nc), shuffle=False)
        a, b, s = evaluate_2d_surface(
            model, tier1_d1, tier1_d2, sub_loader, device,
            grid_range=(-1.0, 1.0), grid_size=21, max_batches=None,  # Use all batches
        )
        m = compute_surface_metrics(a, b, s)
        center_loss = evaluate_loss(model, sub_loader, device, max_batches=None)
        data_size_results[nc] = {
            'loss_range': m['loss_range'],
            'roughness': m['roughness'],
            'basin_diameter': m['basin_diameter'],
            'curvature_ratio': m['curvature_ratio'],
            'convexity_ratio': m['convexity_ratio'],
            'center_loss': center_loss,
        }
        print(f"  {nc} chunks: loss_range={m['loss_range']:.2f}, roughness={m['roughness']:.4f}, "
              f"center_loss={center_loss:.4f}")

    results['ablation3_eval_data_size'] = data_size_results

    # Plot evaluation data size results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    nc_list = sorted(data_size_results.keys())

    axes[0].plot(nc_list, [data_size_results[n]['loss_range'] for n in nc_list],
                'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Evaluation Chunks')
    axes[0].set_ylabel('Loss Range')
    axes[0].set_title('Loss Range vs Eval Data Size')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(nc_list, [data_size_results[n]['roughness'] for n in nc_list],
                'r-s', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Evaluation Chunks')
    axes[1].set_ylabel('Roughness')
    axes[1].set_title('Roughness vs Eval Data Size')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(nc_list, [data_size_results[n]['basin_diameter'] for n in nc_list],
                'g-^', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Evaluation Chunks')
    axes[2].set_ylabel('Basin Diameter')
    axes[2].set_title('Basin Diameter vs Eval Data Size')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Ablation 3: Evaluation Data Size Sweep', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation3_data_size.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Save all results ---
    total_time = time.time() - t_start
    results['total_time_seconds'] = total_time
    results['baseline_loss'] = baseline_loss
    results['n_params'] = n_params

    with open(os.path.join(output_dir, 'all_ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to {output_dir}/")
    print("=" * 70)
    print("All Ablation Studies Complete!")
    print("=" * 70)

    # Print summary tables
    print("\n--- ABLATION 1: PCA Sample Size ---")
    print(f"{'N':>5} | {'Loss Range':>12} | {'Roughness':>10} | {'Basin D.':>10} | {'Expl. Var':>12}")
    for n in sorted(pca_sweep_results.keys()):
        r = pca_sweep_results[n]
        ev = f"{r['explained_ratios'][0]:.4f},{r['explained_ratios'][1]:.4f}"
        print(f"{n:>5} | {r['loss_range']:>12.2f} | {r['roughness']:>10.4f} | {r['basin_diameter']:>10.4f} | {ev}")

    print("\n--- ABLATION 2: Grid Resolution ---")
    print(f"{'Grid':>7} | {'Loss Range':>12} | {'Roughness':>10} | {'Basin D.':>10}")
    for g in sorted(grid_sweep_results.keys()):
        r = grid_sweep_results[g]
        print(f"{g:>3}x{g:<3} | {r['loss_range']:>12.2f} | {r['roughness']:>10.4f} | {r['basin_diameter']:>10.4f}")

    print("\n--- ABLATION 3: Eval Data Size ---")
    print(f"{'Chunks':>7} | {'Loss Range':>12} | {'Roughness':>10} | {'Basin D.':>10} | {'Center Loss':>12}")
    for n in sorted(data_size_results.keys()):
        r = data_size_results[n]
        print(f"{n:>7} | {r['loss_range']:>12.2f} | {r['roughness']:>10.4f} | {r['basin_diameter']:>10.4f} | {r['center_loss']:>12.4f}")


if __name__ == '__main__':
    main()
