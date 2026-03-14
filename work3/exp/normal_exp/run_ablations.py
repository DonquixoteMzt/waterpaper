"""
Ablation Studies for LLMScape.

1. TADN granularity ablation
2. Direction selection depth ablation (PCA sample size, power iteration count)
3. Grid resolution ablation
4. Evaluation data size ablation
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import sys
import json
import time
import gc
import math
import argparse

import numpy as np
import torch

from data_loader import prepare_data, prepare_custom_data, TokenChunkDataset
from normalization import get_normalization_units, apply_tadn, apply_layer_normalization
from direction_selection import (
    generate_tier1_directions, generate_random_direction,
    gradient_pca_with_convergence, orthogonalize_directions,
)
from grid_evaluation import evaluate_loss, evaluate_2d_surface
from metrics import compute_surface_metrics


def apply_block_normalization(direction, model, epsilon=1e-8):
    """Normalize per transformer block (all params in one block as one unit)."""
    normalized = {}
    # Group parameters by block
    blocks = {}
    for name, param in model.named_parameters():
        if name not in direction:
            continue
        # Identify block
        parts = name.split('.')
        block_key = None
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                block_key = '.'.join(parts[:i+2])
                break
        if block_key is None:
            block_key = 'other'
        if block_key not in blocks:
            blocks[block_key] = []
        blocks[block_key].append(name)

    for block_key, names in blocks.items():
        p_norm_sq = sum(model.get_parameter(n).data.detach().cpu().float().norm().item() ** 2
                        for n in names)
        d_norm_sq = sum(direction[n].detach().cpu().clone().float().norm().item() ** 2
                        for n in names)
        p_norm = math.sqrt(p_norm_sq)
        d_norm = math.sqrt(d_norm_sq)
        scale = p_norm / d_norm if d_norm > epsilon and p_norm > epsilon else 1.0
        for name in names:
            d = direction[name].detach().cpu().clone().float()
            normalized[name] = (d * scale).to(direction[name].dtype).to(direction[name].device)
    return normalized


def apply_global_normalization(direction, model, epsilon=1e-8):
    """Normalize all parameters as one unit (global normalization)."""
    p_norm_sq = sum(p.data.detach().cpu().float().norm().item() ** 2
                    for p in model.parameters() if p.requires_grad)
    d_norm_sq = sum(direction[n].detach().cpu().clone().float().norm().item() ** 2
                    for n in direction)
    p_norm = math.sqrt(p_norm_sq)
    d_norm = math.sqrt(d_norm_sq)
    scale = p_norm / d_norm if d_norm > epsilon and p_norm > epsilon else 1.0
    normalized = {}
    for name in direction:
        d = direction[name].detach().cpu().clone().float()
        normalized[name] = (d * scale).to(direction[name].dtype).to(direction[name].device)
    return normalized


def run_tadn_granularity_ablation(model, eval_loader, device, units, num_heads, head_dim,
                                   grid_size=31, max_eval_batches=5, output_dir='.'):
    """Ablation 1: TADN granularity."""
    print("\n" + "=" * 60)
    print("ABLATION 1: TADN Granularity")
    print("=" * 60)

    d1_raw, d2_raw = generate_tier1_directions(model, seed1=42, seed2=123)

    norm_methods = {
        'TADN-full': lambda d: apply_tadn(d, model, units, num_heads, head_dim),
        'TADN-layer': lambda d: apply_layer_normalization(d, model),
        'TADN-block': lambda d: apply_block_normalization(d, model),
        'TADN-global': lambda d: apply_global_normalization(d, model),
    }

    results = {}
    for name, norm_fn in norm_methods.items():
        print(f"\n  --- {name} ---")
        d1 = norm_fn(d1_raw)
        d2 = norm_fn(d2_raw)

        alphas, betas, surface = evaluate_2d_surface(
            model, d1, d2, eval_loader, device,
            grid_range=(-1.0, 1.0), grid_size=grid_size,
            max_batches=max_eval_batches,
        )
        metrics = compute_surface_metrics(alphas, betas, surface)
        results[name] = metrics
        print(f"  {name}: loss_range={metrics['loss_range']:.2f}, roughness={metrics['roughness']:.4f}, "
              f"basin_diam={metrics['basin_diameter']:.4f}")

    # Save
    with open(os.path.join(output_dir, 'ablation1_tadn_granularity.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def run_direction_depth_ablation(model, data, device, units, num_heads, head_dim,
                                  grid_size=31, max_eval_batches=5, output_dir='.'):
    """Ablation 2: Direction selection depth."""
    print("\n" + "=" * 60)
    print("ABLATION 2: Direction Selection Depth")
    print("=" * 60)

    # PCA with different N values
    pca_ns = [10, 25, 50, 100, 150, 200]
    pca_results = {}

    for n in pca_ns:
        if n > len(data['grad_loader'].dataset):
            print(f"  Skipping N={n} (not enough data)")
            continue
        print(f"\n  --- Gradient PCA N={n} ---")
        try:
            pca_res, pca_dirs = gradient_pca_with_convergence(
                model, data['grad_loader'], device,
                n_max=n, checkpoints=[n], k=2,
            )
            d1 = apply_tadn({k: v.cpu() for k, v in pca_dirs[0].items()},
                             model, units, num_heads, head_dim)
            d2 = apply_tadn({k: v.cpu() for k, v in pca_dirs[1].items()},
                             model, units, num_heads, head_dim)

            alphas, betas, surface = evaluate_2d_surface(
                model, d1, d2, data['eval_loader'], device,
                grid_range=(-1.0, 1.0), grid_size=grid_size,
                max_batches=max_eval_batches,
            )
            metrics = compute_surface_metrics(alphas, betas, surface)

            # Get explained variance
            final_n = max(pca_res.keys())
            ev_ratios = pca_res[final_n]['explained_ratios']

            pca_results[str(n)] = {
                'metrics': metrics,
                'explained_variance': ev_ratios,
            }
            print(f"  N={n}: loss_range={metrics['loss_range']:.2f}, "
                  f"ev_ratio={ev_ratios[0]:.4f}")
        except Exception as e:
            print(f"  Error at N={n}: {e}")

    with open(os.path.join(output_dir, 'ablation2_direction_depth.json'), 'w') as f:
        json.dump(pca_results, f, indent=2)
    return pca_results


def run_grid_resolution_ablation(model, eval_loader, device, units, num_heads, head_dim,
                                   max_eval_batches=5, output_dir='.'):
    """Ablation 3: Grid resolution."""
    print("\n" + "=" * 60)
    print("ABLATION 3: Grid Resolution")
    print("=" * 60)

    d1_raw, d2_raw = generate_tier1_directions(model, seed1=42, seed2=123)
    d1 = apply_tadn(d1_raw, model, units, num_heads, head_dim)
    d2 = apply_tadn(d2_raw, model, units, num_heads, head_dim)

    resolutions = [11, 21, 31, 51]
    results = {}

    for gs in resolutions:
        print(f"\n  --- Grid {gs}x{gs} ---")
        alphas, betas, surface = evaluate_2d_surface(
            model, d1, d2, eval_loader, device,
            grid_range=(-1.0, 1.0), grid_size=gs,
            max_batches=max_eval_batches,
        )
        metrics = compute_surface_metrics(alphas, betas, surface)
        results[str(gs)] = metrics
        print(f"  {gs}x{gs}: loss_range={metrics['loss_range']:.2f}, "
              f"roughness={metrics['roughness']:.4f}")

    with open(os.path.join(output_dir, 'ablation3_grid_resolution.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def run_data_size_ablation(model, tokenizer, device, units, num_heads, head_dim,
                            grid_size=21, max_eval_batches=None, output_dir='.'):
    """Ablation 4: Evaluation data size."""
    print("\n" + "=" * 60)
    print("ABLATION 4: Evaluation Data Size")
    print("=" * 60)

    from datasets import load_dataset
    from torch.utils.data import DataLoader

    # Load full dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in ds['text'] if len(t.strip()) > 50]
    all_tokens = tokenizer('\n'.join(texts[:200]), return_tensors='pt', truncation=False)['input_ids'][0]

    seq_len = 256
    all_chunks = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        all_chunks.append(all_tokens[i:i + seq_len])

    # Same directions for all
    d1_raw, d2_raw = generate_tier1_directions(model, seed1=42, seed2=123)
    d1 = apply_tadn(d1_raw, model, units, num_heads, head_dim)
    d2 = apply_tadn(d2_raw, model, units, num_heads, head_dim)

    # Different data sizes (number of chunks)
    chunk_counts = [2, 5, 10, 20, 50]
    results = {}

    for nc in chunk_counts:
        if nc > len(all_chunks):
            continue
        print(f"\n  --- {nc} chunks ({nc * seq_len} tokens) ---")
        subset = all_chunks[:nc]
        dataset = TokenChunkDataset(subset)
        loader = DataLoader(dataset, batch_size=min(4, nc), shuffle=False)

        alphas, betas, surface = evaluate_2d_surface(
            model, d1, d2, loader, device,
            grid_range=(-1.0, 1.0), grid_size=grid_size,
        )
        metrics = compute_surface_metrics(alphas, betas, surface)
        results[str(nc)] = {
            'n_tokens': nc * seq_len,
            'metrics': metrics,
        }
        print(f"  {nc} chunks: loss_range={metrics['loss_range']:.2f}, "
              f"roughness={metrics['roughness']:.4f}")

    with open(os.path.join(output_dir, 'ablation4_data_size.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--grid-size', type=int, default=31)
    parser.add_argument('--max-eval-batches', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    output_dir = 'results/ablations'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("ABLATION STUDIES")
    print("=" * 70)
    t_start = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", local_files_only=True,
    ).to(device)
    model.eval()

    units = get_normalization_units(model)
    cfg = model.config
    num_heads = getattr(cfg, 'num_attention_heads', None)
    head_dim = getattr(cfg, 'head_dim', None)
    if head_dim is None and num_heads:
        head_dim = getattr(cfg, 'hidden_size', 0) // num_heads

    # Prepare data
    config = {
        'data': {
            'dataset': 'wikitext', 'dataset_config': 'wikitext-2-raw-v1',
            'split': 'test', 'seq_len': 256, 'n_eval_chunks': 50,
            'eval_batch_size': 4, 'max_eval_batches': args.max_eval_batches,
            'n_grad_batches': 200, 'grad_batch_size': 1,
        },
        'direction': {'tier3': {'hvp_max_batches': 3, 'hvp_batch_size': 1}},
    }
    data = prepare_data(tokenizer, config)

    all_results = {}

    # Ablation 1: TADN Granularity
    all_results['tadn_granularity'] = run_tadn_granularity_ablation(
        model, data['eval_loader'], device, units, num_heads, head_dim,
        grid_size=args.grid_size, max_eval_batches=args.max_eval_batches,
        output_dir=output_dir,
    )

    # Ablation 2: Direction Selection Depth
    all_results['direction_depth'] = run_direction_depth_ablation(
        model, data, device, units, num_heads, head_dim,
        grid_size=args.grid_size, max_eval_batches=args.max_eval_batches,
        output_dir=output_dir,
    )

    # Ablation 3: Grid Resolution
    all_results['grid_resolution'] = run_grid_resolution_ablation(
        model, data['eval_loader'], device, units, num_heads, head_dim,
        max_eval_batches=args.max_eval_batches, output_dir=output_dir,
    )

    # Ablation 4: Data Size
    all_results['data_size'] = run_data_size_ablation(
        model, tokenizer, device, units, num_heads, head_dim,
        grid_size=21, output_dir=output_dir,
    )

    # Save all
    with open(os.path.join(output_dir, 'all_ablations.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate summary plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # TADN granularity plot
    if all_results['tadn_granularity']:
        methods = list(all_results['tadn_granularity'].keys())
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        keys = ['loss_range', 'roughness', 'basin_diameter', 'curvature_ratio']
        for idx, key in enumerate(keys):
            vals = [all_results['tadn_granularity'][m].get(key, 0) for m in methods]
            axes[idx].bar(range(len(methods)), vals, alpha=0.8,
                          color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
            axes[idx].set_xticks(range(len(methods)))
            axes[idx].set_xticklabels(methods, rotation=20, fontsize=8)
            axes[idx].set_title(key.replace('_', ' ').title())
            axes[idx].grid(True, alpha=0.3, axis='y')
        plt.suptitle('Ablation: Normalization Granularity', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ablation1_plot.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Direction depth plot
    if all_results['direction_depth']:
        ns = sorted([int(k) for k in all_results['direction_depth'].keys()])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        loss_ranges = [all_results['direction_depth'][str(n)]['metrics']['loss_range'] for n in ns]
        ev_ratios = [all_results['direction_depth'][str(n)]['explained_variance'][0] for n in ns]
        ax1.plot(ns, loss_ranges, 'b-o', linewidth=2)
        ax1.set_xlabel('Number of gradient samples (N)')
        ax1.set_ylabel('Loss Range')
        ax1.set_title('Loss Range vs. PCA Sample Size')
        ax1.grid(True, alpha=0.3)
        ax2.plot(ns, ev_ratios, 'r-s', linewidth=2)
        ax2.set_xlabel('Number of gradient samples (N)')
        ax2.set_ylabel('Explained Variance (PC1)')
        ax2.set_title('Explained Variance vs. PCA Sample Size')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ablation2_plot.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Grid resolution plot
    if all_results['grid_resolution']:
        gs_vals = sorted([int(k) for k in all_results['grid_resolution'].keys()])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        lr = [all_results['grid_resolution'][str(g)]['loss_range'] for g in gs_vals]
        rg = [all_results['grid_resolution'][str(g)]['roughness'] for g in gs_vals]
        ax1.plot(gs_vals, lr, 'b-o', linewidth=2)
        ax1.set_xlabel('Grid Size')
        ax1.set_ylabel('Loss Range')
        ax1.set_title('Loss Range vs. Grid Resolution')
        ax1.grid(True, alpha=0.3)
        ax2.plot(gs_vals, rg, 'r-s', linewidth=2)
        ax2.set_xlabel('Grid Size')
        ax2.set_ylabel('Roughness')
        ax2.set_title('Roughness vs. Grid Resolution')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ablation3_plot.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Data size plot
    if all_results['data_size']:
        ncs = sorted([int(k) for k in all_results['data_size'].keys()])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        tokens = [all_results['data_size'][str(n)]['n_tokens'] for n in ncs]
        lr = [all_results['data_size'][str(n)]['metrics']['loss_range'] for n in ncs]
        rg = [all_results['data_size'][str(n)]['metrics']['roughness'] for n in ncs]
        ax1.plot(tokens, lr, 'b-o', linewidth=2)
        ax1.set_xlabel('Number of Tokens')
        ax1.set_ylabel('Loss Range')
        ax1.set_title('Loss Range vs. Eval Data Size')
        ax1.grid(True, alpha=0.3)
        ax2.plot(tokens, rg, 'r-s', linewidth=2)
        ax2.set_xlabel('Number of Tokens')
        ax2.set_ylabel('Roughness')
        ax2.set_title('Roughness vs. Eval Data Size')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ablation4_plot.png'), dpi=150, bbox_inches='tight')
        plt.close()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"ALL ABLATIONS COMPLETE ({elapsed:.1f}s = {elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
