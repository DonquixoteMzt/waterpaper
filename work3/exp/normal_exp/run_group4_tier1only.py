"""
Group 4: Cross-Model Comparison (7B models) - Tier 1 Only.

Compares Qwen2.5-7B-Instruct vs OLMo-3-7B-Think using
MMSP Method C (Independent Landscape Comparison).
Tier 1 only to avoid OOM on gradient PCA for 7B models.
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import sys
import json
import time
import gc
import argparse

import numpy as np
import torch

from data_loader import prepare_data
from normalization import get_normalization_units, apply_tadn
from direction_selection import generate_tier1_directions
from grid_evaluation import evaluate_loss, evaluate_2d_surface
from metrics import compute_surface_metrics
from visualization import plot_2d_surface


def run_single_model(model_name, device, output_dir, grid_size=21, max_eval_batches=3):
    """Run Tier 1 landscape analysis for one 7B model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"  Analyzing: {model_name}")
    print(f"{'='*60}")
    t0 = time.time()

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

    # Generate directions on CPU to avoid OOM
    print(f"  Computing Tier 1 directions (on CPU)...")
    model.cpu()
    torch.cuda.empty_cache()
    d1_raw, d2_raw = generate_tier1_directions(model, seed1=42, seed2=123)
    d1 = apply_tadn(d1_raw, model, units, num_heads, head_dim)
    d2 = apply_tadn(d2_raw, model, units, num_heads, head_dim)
    del d1_raw, d2_raw
    gc.collect()
    model.to(device)

    print(f"  Evaluating Tier 1 surface ({grid_size}x{grid_size})...")
    alphas, betas, surface = evaluate_2d_surface(
        model, d1, d2, data['eval_loader'], device,
        grid_range=(-1.0, 1.0), grid_size=grid_size,
        max_batches=max_eval_batches,
    )
    metrics = compute_surface_metrics(alphas, betas, surface)
    results['tier1_metrics'] = metrics

    short_name = model_name.split('/')[-1]
    plot_2d_surface(alphas, betas, surface, f'{short_name} (Random+TADN)',
                    os.path.join(output_dir, f'surface_tier1_{short_name}.png'))
    np.savez(os.path.join(output_dir, f'surface_tier1_{short_name}.npz'),
             alphas=alphas, betas=betas, surface=surface)

    elapsed = time.time() - t0
    results['time'] = elapsed
    print(f"  Done: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    del model, d1, d2
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--grid-size', type=int, default=21)
    parser.add_argument('--max-eval-batches', type=int, default=3)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    output_dir = 'results/group4_crossmodel'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GROUP 4: Cross-Model Comparison (7B Models) - Tier 1 Only")
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

    # Comparison plot
    if len(all_results) >= 2:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        model_names = list(all_results.keys())
        short_names = [n.split('/')[-1] for n in model_names]

        metrics_keys = ['loss_range', 'roughness', 'basin_diameter', 'curvature_ratio',
                        'convexity_ratio', 'basin_flatness']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, key in enumerate(metrics_keys):
            vals = [all_results[m]['tier1_metrics'].get(key, 0) for m in model_names]
            x = np.arange(len(model_names))
            axes[idx].bar(x, vals, alpha=0.8, color=['#3498db', '#e74c3c'])
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(short_names, fontsize=9)
            axes[idx].set_title(key.replace('_', ' ').title())
            axes[idx].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Cross-Model Comparison: Tier 1 Geometric Metrics', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'crossmodel_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"GROUP 4 COMPLETE ({elapsed:.1f}s = {elapsed/60:.1f} min)")
    print(f"{'='*70}")
    for model_name, r in all_results.items():
        short = model_name.split('/')[-1]
        t1 = r['tier1_metrics']
        print(f"  {short}:")
        print(f"    Loss: {r['baseline_loss']:.4f}")
        print(f"    Tier1: range={t1['loss_range']:.2f}, basin={t1['basin_diameter']:.4f}")


if __name__ == '__main__':
    main()
