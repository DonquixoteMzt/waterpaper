"""
Group 3: Post-Training Effect on Loss Landscape.

Compares Qwen2.5-7B-Instruct (post-trained) vs independent landscape analysis.
Uses base model Qwen3-0.6B-Base with different evaluation regimes
to demonstrate how alignment/instruction tuning changes the loss landscape.

Also compares independent landscapes of Qwen3-0.6B-Base using
standard evaluation vs instruction-formatted prompts.
"""

import os
import sys
import json
import time
import gc
import math
import argparse

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import numpy as np
import torch

from data_loader import prepare_data, prepare_custom_data
from normalization import get_normalization_units, apply_tadn
from direction_selection import generate_tier1_directions, gradient_pca_with_convergence
from grid_evaluation import evaluate_loss, evaluate_2d_surface
from metrics import compute_surface_metrics
from visualization import plot_2d_surface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--grid-size', type=int, default=31)
    parser.add_argument('--max-eval-batches', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    output_dir = 'results/group3_posttraining'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GROUP 3: Post-Training Effect Analysis")
    print("=" * 70)
    t_start = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ===============================================================
    # Exp 3a: Qwen3-0.6B-Base independent landscape (Tier 1 + Tier 2)
    # ===============================================================
    model_name = "Qwen/Qwen3-0.6B-Base"
    print(f"\n--- Loading {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", local_files_only=True,
    ).to(device)
    model.eval()

    config = {
        'data': {
            'dataset': 'wikitext', 'dataset_config': 'wikitext-2-raw-v1',
            'split': 'test', 'seq_len': 256, 'n_eval_chunks': 50,
            'eval_batch_size': 4, 'max_eval_batches': args.max_eval_batches,
            'n_grad_batches': 100, 'grad_batch_size': 1,
        },
        'direction': {'tier3': {'hvp_max_batches': 3, 'hvp_batch_size': 1}},
    }
    data = prepare_data(tokenizer, config)

    units = get_normalization_units(model)
    cfg = model.config
    num_heads = getattr(cfg, 'num_attention_heads', None)
    head_dim = getattr(cfg, 'head_dim', None)
    if head_dim is None and num_heads:
        head_dim = getattr(cfg, 'hidden_size', 0) // num_heads

    base_loss = evaluate_loss(model, data['eval_loader'], device, args.max_eval_batches)
    print(f"  Base loss: {base_loss:.4f}")

    # Tier 1 (Random + TADN)
    print("\n--- Tier 1: Random + TADN for base model ---")
    d1_raw, d2_raw = generate_tier1_directions(model, seed1=42, seed2=123)
    d1_t1 = apply_tadn(d1_raw, model, units, num_heads, head_dim)
    d2_t1 = apply_tadn(d2_raw, model, units, num_heads, head_dim)

    alphas_b1, betas_b1, surface_b1 = evaluate_2d_surface(
        model, d1_t1, d2_t1, data['eval_loader'], device,
        grid_range=(-1.0, 1.0), grid_size=args.grid_size,
        max_batches=args.max_eval_batches,
    )
    metrics_b1 = compute_surface_metrics(alphas_b1, betas_b1, surface_b1)
    plot_2d_surface(alphas_b1, betas_b1, surface_b1,
                    'Qwen3-0.6B-Base (Random+TADN)',
                    os.path.join(output_dir, 'surface_base_tier1.png'))

    # Tier 2 (Gradient PCA + TADN)
    print("\n--- Tier 2: Gradient PCA + TADN for base model ---")
    pca_results, pca_dirs = gradient_pca_with_convergence(
        model, data['grad_loader'], device,
        n_max=100, checkpoints=[20, 50, 100], k=2,
    )
    d1_pca = apply_tadn({k: v.cpu() for k, v in pca_dirs[0].items()},
                         model, units, num_heads, head_dim)
    d2_pca = apply_tadn({k: v.cpu() for k, v in pca_dirs[1].items()},
                         model, units, num_heads, head_dim)

    alphas_b2, betas_b2, surface_b2 = evaluate_2d_surface(
        model, d1_pca, d2_pca, data['eval_loader'], device,
        grid_range=(-1.0, 1.0), grid_size=args.grid_size,
        max_batches=args.max_eval_batches,
    )
    metrics_b2 = compute_surface_metrics(alphas_b2, betas_b2, surface_b2)
    plot_2d_surface(alphas_b2, betas_b2, surface_b2,
                    'Qwen3-0.6B-Base (GradPCA+TADN)',
                    os.path.join(output_dir, 'surface_base_tier2.png'))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ===============================================================
    # Exp 3b: Qwen2.5-7B-Instruct landscape (post-trained model)
    # ===============================================================
    instruct_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\n--- Loading {instruct_name} ---")
    tokenizer_i = AutoTokenizer.from_pretrained(instruct_name, trust_remote_code=True, local_files_only=True)
    if tokenizer_i.pad_token is None:
        tokenizer_i.pad_token = tokenizer_i.eos_token

    model_i = AutoModelForCausalLM.from_pretrained(
        instruct_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", local_files_only=True,
    ).to(device)
    model_i.eval()

    config_i = {
        'data': {
            'dataset': 'wikitext', 'dataset_config': 'wikitext-2-raw-v1',
            'split': 'test', 'seq_len': 256, 'n_eval_chunks': 30,
            'eval_batch_size': 2, 'max_eval_batches': 3,
            'n_grad_batches': 50, 'grad_batch_size': 1,
        },
        'direction': {'tier3': {'hvp_max_batches': 2, 'hvp_batch_size': 1}},
    }
    data_i = prepare_data(tokenizer_i, config_i)

    instruct_loss = evaluate_loss(model_i, data_i['eval_loader'], device, 3)
    print(f"  Instruct loss: {instruct_loss:.4f}")

    units_i = get_normalization_units(model_i)
    cfg_i = model_i.config
    num_heads_i = getattr(cfg_i, 'num_attention_heads', None)
    head_dim_i = getattr(cfg_i, 'head_dim', None)
    if head_dim_i is None and num_heads_i:
        head_dim_i = getattr(cfg_i, 'hidden_size', 0) // num_heads_i

    # Tier 1 for instruct model
    print("\n--- Tier 1: Random + TADN for Instruct model ---")
    d1_raw_i, d2_raw_i = generate_tier1_directions(model_i, seed1=42, seed2=123)
    d1_i = apply_tadn(d1_raw_i, model_i, units_i, num_heads_i, head_dim_i)
    d2_i = apply_tadn(d2_raw_i, model_i, units_i, num_heads_i, head_dim_i)

    alphas_i, betas_i, surface_i = evaluate_2d_surface(
        model_i, d1_i, d2_i, data_i['eval_loader'], device,
        grid_range=(-1.0, 1.0), grid_size=args.grid_size,
        max_batches=3,
    )
    metrics_i = compute_surface_metrics(alphas_i, betas_i, surface_i)
    plot_2d_surface(alphas_i, betas_i, surface_i,
                    'Qwen2.5-7B-Instruct (Random+TADN)',
                    os.path.join(output_dir, 'surface_instruct_tier1.png'))

    # Tier 2 for instruct model
    print("\n--- Tier 2: Gradient PCA for Instruct model ---")
    pca_results_i, pca_dirs_i = gradient_pca_with_convergence(
        model_i, data_i['grad_loader'], device,
        n_max=50, checkpoints=[10, 25, 50], k=2,
    )
    d1_pca_i = apply_tadn({k: v.cpu() for k, v in pca_dirs_i[0].items()},
                            model_i, units_i, num_heads_i, head_dim_i)
    d2_pca_i = apply_tadn({k: v.cpu() for k, v in pca_dirs_i[1].items()},
                            model_i, units_i, num_heads_i, head_dim_i)

    alphas_i2, betas_i2, surface_i2 = evaluate_2d_surface(
        model_i, d1_pca_i, d2_pca_i, data_i['eval_loader'], device,
        grid_range=(-1.0, 1.0), grid_size=args.grid_size,
        max_batches=3,
    )
    metrics_i2 = compute_surface_metrics(alphas_i2, betas_i2, surface_i2)
    plot_2d_surface(alphas_i2, betas_i2, surface_i2,
                    'Qwen2.5-7B-Instruct (GradPCA+TADN)',
                    os.path.join(output_dir, 'surface_instruct_tier2.png'))

    # Side-by-side comparison plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    surfaces_data = [
        (alphas_b1, betas_b1, surface_b1, 'Qwen3-0.6B-Base\n(Random+TADN)'),
        (alphas_b2, betas_b2, surface_b2, 'Qwen3-0.6B-Base\n(GradPCA+TADN)'),
        (alphas_i, betas_i, surface_i, 'Qwen2.5-7B-Instruct\n(Random+TADN)'),
        (alphas_i2, betas_i2, surface_i2, 'Qwen2.5-7B-Instruct\n(GradPCA+TADN)'),
    ]

    for idx, (a, b, s, title) in enumerate(surfaces_data):
        ax = axes[idx // 2][idx % 2]
        A, B = np.meshgrid(a, b)
        vmin = s.min()
        vmax = min(s.max(), s.min() + 3 * (np.median(s) - s.min() + 0.1))
        levels = np.linspace(vmin, vmax, 25)
        cs = ax.contourf(A, B, s, levels=levels, cmap='viridis')
        ax.contour(A, B, s, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
        plt.colorbar(cs, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\beta$')
        ax.plot(0, 0, 'r*', markersize=12)

    plt.suptitle('Post-Training Effect: Base vs Instruction-Tuned', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results = {
        'base_model': model_name,
        'instruct_model': instruct_name,
        'base_loss': base_loss,
        'instruct_loss': instruct_loss,
        'base_tier1_metrics': metrics_b1,
        'base_tier2_metrics': metrics_b2,
        'instruct_tier1_metrics': metrics_i,
        'instruct_tier2_metrics': metrics_i2,
        'total_time': time.time() - t_start,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    np.savez(os.path.join(output_dir, 'surface_base_tier1.npz'),
             alphas=alphas_b1, betas=betas_b1, surface=surface_b1)
    np.savez(os.path.join(output_dir, 'surface_instruct_tier1.npz'),
             alphas=alphas_i, betas=betas_i, surface=surface_i)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"GROUP 3 COMPLETE ({elapsed:.1f}s = {elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"Base loss: {base_loss:.4f}, Instruct loss: {instruct_loss:.4f}")
    print(f"Base T1: range={metrics_b1['loss_range']:.2f}, basin={metrics_b1['basin_diameter']:.4f}")
    print(f"Base T2: range={metrics_b2['loss_range']:.2f}, basin={metrics_b2['basin_diameter']:.4f}")
    print(f"Inst T1: range={metrics_i['loss_range']:.2f}, basin={metrics_i['basin_diameter']:.4f}")
    print(f"Inst T2: range={metrics_i2['loss_range']:.2f}, basin={metrics_i2['basin_diameter']:.4f}")


if __name__ == '__main__':
    main()
