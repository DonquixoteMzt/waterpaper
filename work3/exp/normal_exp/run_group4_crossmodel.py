"""
Group 4: Cross-Model Comparison (7B models).

Compares Qwen2.5-7B-Instruct vs OLMo-3-7B-Think using
MMSP Method C (Independent Landscape Comparison).
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

from data_loader import prepare_data
from normalization import get_normalization_units, apply_tadn
from direction_selection import generate_tier1_directions, gradient_pca_with_convergence
from grid_evaluation import evaluate_loss, evaluate_2d_surface
from metrics import compute_surface_metrics
from visualization import plot_2d_surface


def run_single_model(model_name, device, output_dir, grid_size=31, max_eval_batches=3):
    """Run independent landscape analysis for one model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"  Analyzing: {model_name}")
    print(f"{'='*60}")
    t0 = time.time()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", local_files_only=True,
    ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Prepare data
    config = {
        'data': {
            'dataset': 'wikitext', 'dataset_config': 'wikitext-2-raw-v1',
            'split': 'test', 'seq_len': 256, 'n_eval_chunks': 30,
            'eval_batch_size': 2, 'max_eval_batches': max_eval_batches,
            'n_grad_batches': 50, 'grad_batch_size': 1,
        },
        'direction': {'tier3': {'hvp_max_batches': 2, 'hvp_batch_size': 1}},
    }
    data = prepare_data(tokenizer, config)

    baseline_loss = evaluate_loss(model, data['eval_loader'], device, max_eval_batches)
    print(f"  Baseline loss: {baseline_loss:.4f}")

    units = get_normalization_units(model)
    cfg = model.config
    num_heads = getattr(cfg, 'num_attention_heads', None)
    head_dim = getattr(cfg, 'head_dim', None)
    if head_dim is None and num_heads:
        head_dim = getattr(cfg, 'hidden_size', 0) // num_heads

    results = {'model': model_name, 'n_params': n_params, 'baseline_loss': baseline_loss}

    # For 7B models, generate directions on CPU to avoid OOM
    print(f"  Computing Tier 1 directions (on CPU)...")
    model.cpu()
    torch.cuda.empty_cache()
    d1_raw, d2_raw = generate_tier1_directions(model, seed1=42, seed2=123)
    d1_t1 = apply_tadn(d1_raw, model, units, num_heads, head_dim)
    d2_t1 = apply_tadn(d2_raw, model, units, num_heads, head_dim)
    # Free raw directions
    del d1_raw, d2_raw
    gc.collect()
    model.to(device)

    print(f"  Evaluating Tier 1 surface ({grid_size}x{grid_size})...")
    alphas1, betas1, surface1 = evaluate_2d_surface(
        model, d1_t1, d2_t1, data['eval_loader'], device,
        grid_range=(-1.0, 1.0), grid_size=grid_size,
        max_batches=max_eval_batches,
    )
    metrics_t1 = compute_surface_metrics(alphas1, betas1, surface1)
    results['tier1_metrics'] = metrics_t1

    short_name = model_name.split('/')[-1]
    plot_2d_surface(alphas1, betas1, surface1, f'{short_name} (Random+TADN)',
                    os.path.join(output_dir, f'surface_tier1_{short_name}.png'))

    # Tier 2: Gradient PCA + TADN (using reduced batch count for 7B)
    print(f"  Computing Tier 2 directions (Gradient PCA)...")
    pca_results, pca_dirs = gradient_pca_with_convergence(
        model, data['grad_loader'], device,
        n_max=30, checkpoints=[10, 20, 30], k=2,
    )
    model.cpu()
    torch.cuda.empty_cache()
    d1_t2 = apply_tadn({k: v.cpu() for k, v in pca_dirs[0].items()},
                        model, units, num_heads, head_dim)
    d2_t2 = apply_tadn({k: v.cpu() for k, v in pca_dirs[1].items()},
                        model, units, num_heads, head_dim)
    model.to(device)

    print(f"  Evaluating Tier 2 surface ({grid_size}x{grid_size})...")
    alphas2, betas2, surface2 = evaluate_2d_surface(
        model, d1_t2, d2_t2, data['eval_loader'], device,
        grid_range=(-1.0, 1.0), grid_size=grid_size,
        max_batches=max_eval_batches,
    )
    metrics_t2 = compute_surface_metrics(alphas2, betas2, surface2)
    results['tier2_metrics'] = metrics_t2

    plot_2d_surface(alphas2, betas2, surface2, f'{short_name} (GradPCA+TADN)',
                    os.path.join(output_dir, f'surface_tier2_{short_name}.png'))

    # Save surfaces
    np.savez(os.path.join(output_dir, f'surface_tier1_{short_name}.npz'),
             alphas=alphas1, betas=betas1, surface=surface1)
    np.savez(os.path.join(output_dir, f'surface_tier2_{short_name}.npz'),
             alphas=alphas2, betas=betas2, surface=surface2)

    elapsed = time.time() - t0
    results['time'] = elapsed
    print(f"  Done: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Clean up
    del model, d1_raw, d2_raw, d1_t1, d2_t1, d1_t2, d2_t2
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--model', type=str, default=None, help='Run single model')
    parser.add_argument('--grid-size', type=int, default=31)
    parser.add_argument('--max-eval-batches', type=int, default=3)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    output_dir = 'results/group4_crossmodel'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GROUP 4: Cross-Model Comparison (7B Models)")
    print("=" * 70)
    t_start = time.time()

    models = []
    if args.model:
        models = [args.model]
    else:
        models = ["Qwen/Qwen2.5-7B-Instruct", "allenai/Olmo-3-7B-Think"]

    all_results = {}
    for model_name in models:
        result = run_single_model(
            model_name, device, output_dir,
            grid_size=args.grid_size,
            max_eval_batches=args.max_eval_batches,
        )
        all_results[model_name] = result

    # Generate comparison plots if both models done
    if len(all_results) == 2:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        model_names = list(all_results.keys())
        short_names = [n.split('/')[-1] for n in model_names]

        # Metrics comparison bar chart
        metrics_keys = ['loss_range', 'roughness', 'basin_diameter', 'curvature_ratio',
                        'convexity_ratio', 'basin_flatness']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, key in enumerate(metrics_keys):
            vals_t1 = [all_results[m]['tier1_metrics'].get(key, 0) for m in model_names]
            vals_t2 = [all_results[m]['tier2_metrics'].get(key, 0) for m in model_names]
            x = np.arange(len(model_names))
            width = 0.35
            axes[idx].bar(x - width/2, vals_t1, width, label='Tier 1', alpha=0.8)
            axes[idx].bar(x + width/2, vals_t2, width, label='Tier 2', alpha=0.8)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(short_names, fontsize=9)
            axes[idx].set_title(key.replace('_', ' ').title())
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Cross-Model Comparison: Geometric Metrics', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'crossmodel_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"GROUP 4 COMPLETE ({elapsed:.1f}s = {elapsed/60:.1f} min)")
    print(f"{'='*70}")
    for model_name, r in all_results.items():
        short = model_name.split('/')[-1]
        t1 = r['tier1_metrics']
        t2 = r['tier2_metrics']
        print(f"  {short}:")
        print(f"    Loss: {r['baseline_loss']:.4f}")
        print(f"    Tier1: range={t1['loss_range']:.2f}, basin={t1['basin_diameter']:.4f}")
        print(f"    Tier2: range={t2['loss_range']:.2f}, basin={t2['basin_diameter']:.4f}")


if __name__ == '__main__':
    main()
