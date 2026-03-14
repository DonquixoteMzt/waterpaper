"""
run_olmo3_base.py — OLMo-3-1025-7B (Base) Cross-Model Comparison.

Runs Tier 1 landscape analysis for allenai/Olmo-3-1025-7B (base model)
to provide a fair comparison with Qwen2.5-7B-Instruct, replacing the
OLMo-3-7B-Think variant which had distribution mismatch on WikiText-2.

Usage:
    python run_olmo3_base.py --gpu 3
"""

import os
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
from metrics import compute_surface_metrics, format_metrics_table
from visualization import plot_2d_surface

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_model_tier1(model_name, device, output_dir, grid_size=21, max_eval_batches=3):
    """Run Tier 1 landscape analysis for a single model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"  Analyzing: {model_name}")
    print(f"{'='*60}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
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
    print(f"  Architecture: num_heads={num_heads}, head_dim={head_dim}")

    results = {
        'model': model_name,
        'n_params': n_params,
        'baseline_loss': baseline_loss,
        'num_heads': num_heads,
        'head_dim': head_dim,
    }

    # Generate directions on CPU to avoid OOM for 7B model
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
    print(format_metrics_table(metrics, f"{model_name.split('/')[-1]} Tier 1"))

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
    parser.add_argument('--grid-size', type=int, default=21)
    parser.add_argument('--max-eval-batches', type=int, default=3)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    output_dir = 'results/olmo3_base'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("OLMo-3-1025-7B (Base) Cross-Model Comparison")
    print("=" * 70)
    t_start = time.time()

    # Run OLMo-3-1025-7B base model
    result = run_model_tier1(
        'allenai/Olmo-3-1025-7B', device, output_dir,
        grid_size=args.grid_size,
        max_eval_batches=args.max_eval_batches,
    )

    # Load existing Qwen2.5-7B-Instruct results for comparison
    qwen_results_path = 'results/group4_crossmodel/results.json'
    qwen_results = None
    if os.path.exists(qwen_results_path):
        with open(qwen_results_path) as f:
            old_results = json.load(f)
        for key in old_results:
            if 'Qwen2.5-7B' in key:
                qwen_results = old_results[key]
                print(f"\n  Loaded existing Qwen2.5-7B-Instruct results")
                break

    # Load existing OLMo-3-7B-Think results for reference
    think_results = None
    if os.path.exists(qwen_results_path):
        with open(qwen_results_path) as f:
            old_results = json.load(f)
        for key in old_results:
            if 'Think' in key:
                think_results = old_results[key]
                print(f"  Loaded existing OLMo-3-7B-Think results for reference")
                break

    # Generate comparison table
    all_models = {'allenai/Olmo-3-1025-7B (Base)': result}
    if qwen_results:
        all_models['Qwen/Qwen2.5-7B-Instruct'] = qwen_results
    if think_results:
        all_models['allenai/Olmo-3-7B-Think'] = think_results

    if len(all_models) >= 2:
        print(f"\n{'='*70}")
        print("Cross-Model Comparison Summary")
        print(f"{'='*70}")
        print(f"{'Model':<35} {'Loss':>8} {'Range':>8} {'Rough':>8} {'Basin':>8} {'Curv.R':>8} {'Convex':>8}")
        print("-" * 95)
        for name, r in all_models.items():
            short = name.split('/')[-1]
            t1 = r['tier1_metrics']
            print(f"{short:<35} {r['baseline_loss']:>8.4f} {t1['loss_range']:>8.2f} "
                  f"{t1['roughness']:>8.4f} {t1['basin_diameter']:>8.4f} "
                  f"{t1['curvature_ratio']:>8.4f} {t1['convexity_ratio']:>8.4f}")

        # Create comparison plot
        model_names = list(all_models.keys())
        short_names = [n.split('/')[-1] for n in model_names]
        colors = ['#3498db', '#e74c3c', '#95a5a6'][:len(model_names)]

        metrics_keys = ['loss_range', 'roughness', 'basin_diameter', 'curvature_ratio',
                        'convexity_ratio', 'basin_flatness']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, key in enumerate(metrics_keys):
            vals = [all_models[m]['tier1_metrics'].get(key, 0) for m in model_names]
            x = np.arange(len(model_names))
            axes[idx].bar(x, vals, alpha=0.8, color=colors)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(short_names, fontsize=8, rotation=15)
            axes[idx].set_title(key.replace('_', ' ').title())
            axes[idx].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Cross-Model Comparison: 7B Models (Tier 1)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'crossmodel_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # Save complete results
    save_results = {
        'olmo3_base': result,
        'qwen25_7b': qwen_results,
        'olmo3_think_reference': think_results,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)
    print("OLMo-3-1025-7B Experiment Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
