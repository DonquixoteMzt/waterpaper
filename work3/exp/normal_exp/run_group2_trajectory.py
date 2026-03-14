"""
Group 2: Multi-Seed Consistency & Cross-Section Analysis.

Demonstrates framework robustness through:
- Multi-seed consistency of landscape visualizations
- 1D cross-section comparison (Random vs PCA directions)
- PCA convergence with full N=200 gradient samples
- Normalization method comparison (TADN vs LayerNorm vs NoNorm)
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

from data_loader import prepare_data
from normalization import get_normalization_units, apply_tadn, apply_layer_normalization, apply_no_normalization
from direction_selection import generate_random_direction, orthogonalize_directions, gradient_pca_with_convergence
from grid_evaluation import evaluate_loss, evaluate_2d_surface, evaluate_1d_curve
from metrics import compute_surface_metrics
from visualization import plot_2d_surface, plot_pca_convergence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--grid-size', type=int, default=31)
    parser.add_argument('--max-eval-batches', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    output_dir = 'results/group2_trajectory'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GROUP 2: Multi-Seed Consistency & Cross-Section Analysis")
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

    config = {
        'data': {
            'dataset': 'wikitext', 'dataset_config': 'wikitext-2-raw-v1',
            'split': 'test', 'seq_len': 256, 'n_eval_chunks': 50,
            'eval_batch_size': 4, 'max_eval_batches': args.max_eval_batches,
            'n_grad_batches': 50, 'grad_batch_size': 1,
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

    baseline_loss = evaluate_loss(model, data['eval_loader'], device, args.max_eval_batches)
    print(f"\nBaseline loss: {baseline_loss:.4f}")

    all_results = {
        'model': model_name,
        'baseline_loss': baseline_loss,
    }

    # ===============================================================
    # Exp 2a: Multi-seed consistency (3 random seeds)
    # ===============================================================
    print("\n--- Exp 2a: Multi-seed consistency ---")
    seeds = [(42, 123), (7, 77), (999, 1234)]
    seed_results = {}

    for seed_idx, (s1, s2) in enumerate(seeds):
        print(f"\n  Seed pair ({s1}, {s2}):")
        d1_raw = generate_random_direction(model, seed=s1)
        d2_raw = generate_random_direction(model, seed=s2)
        d2_raw = orthogonalize_directions(d1_raw, d2_raw)

        d1 = apply_tadn(d1_raw, model, units, num_heads, head_dim)
        d2 = apply_tadn(d2_raw, model, units, num_heads, head_dim)

        alphas, betas, surface = evaluate_2d_surface(
            model, d1, d2, data['eval_loader'], device,
            grid_range=(-1.0, 1.0), grid_size=args.grid_size,
            max_batches=args.max_eval_batches,
        )
        metrics = compute_surface_metrics(alphas, betas, surface)
        seed_results[f'seed_{s1}_{s2}'] = metrics

        plot_2d_surface(alphas, betas, surface,
                        f'Qwen3-0.6B-Base (seed={s1},{s2})',
                        os.path.join(output_dir, f'surface_seed{seed_idx}.png'))
        np.savez(os.path.join(output_dir, f'surface_seed{seed_idx}.npz'),
                 alphas=alphas, betas=betas, surface=surface)

        print(f"  loss_range={metrics['loss_range']:.2f}, roughness={metrics['roughness']:.4f}, "
              f"basin_diam={metrics['basin_diameter']:.4f}")

    all_results['multi_seed'] = seed_results

    # ===============================================================
    # Exp 2b: 1D cross-sections along PCA vs random
    # ===============================================================
    print("\n--- Exp 2b: 1D cross-sections (PCA vs random) ---")

    # Random direction
    d1_rand = generate_random_direction(model, seed=42)
    d1_rand_tadn = apply_tadn(d1_rand, model, units, num_heads, head_dim)

    alphas_rand, losses_rand = evaluate_1d_curve(
        model, d1_rand_tadn, data['eval_loader'], device,
        alpha_range=(-1.0, 1.0), n_points=41,
        max_batches=args.max_eval_batches,
    )

    # PCA direction
    pca_results, pca_dirs = gradient_pca_with_convergence(
        model, data['grad_loader'], device,
        n_max=50, checkpoints=[10, 20, 30, 50], k=2,
    )
    d1_pca = apply_tadn({k: v.cpu() for k, v in pca_dirs[0].items()},
                         model, units, num_heads, head_dim)

    alphas_pca, losses_pca = evaluate_1d_curve(
        model, d1_pca, data['eval_loader'], device,
        alpha_range=(-1.0, 1.0), n_points=41,
        max_batches=args.max_eval_batches,
    )

    # Plot comparison
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas_rand, losses_rand, 'b-', linewidth=2, label='Random + TADN')
    ax.plot(alphas_pca, losses_pca, 'r-', linewidth=2, label='Grad PCA + TADN')
    ax.axhline(y=baseline_loss, color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_loss:.2f})')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Loss')
    ax.set_title('1D Cross-sections: Random vs PCA Direction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_sections.png'), dpi=150, bbox_inches='tight')
    plt.close()

    all_results['cross_sections'] = {
        'random': {'loss_range': float(losses_rand.max() - losses_rand.min()),
                    'loss_at_center': float(losses_rand[len(losses_rand)//2])},
        'pca': {'loss_range': float(losses_pca.max() - losses_pca.min()),
                'loss_at_center': float(losses_pca[len(losses_pca)//2])},
    }

    print(f"  Random: loss_range={losses_rand.max()-losses_rand.min():.2f}")
    print(f"  PCA:    loss_range={losses_pca.max()-losses_pca.min():.2f}")

    # ===============================================================
    # Exp 2c: PCA convergence analysis (full N=200)
    # ===============================================================
    print("\n--- Exp 2c: PCA convergence analysis ---")
    pca_convergence = {}
    for N, res in pca_results.items():
        pca_convergence[int(N)] = {
            'eigenvalues': res['eigenvalues'],
            'explained_ratios': res['explained_ratios'],
            'subspace_angle': res['subspace_angle_from_prev'],
        }
        print(f"  N={N}: ev_ratio=[{res['explained_ratios'][0]:.4f}, {res['explained_ratios'][1]:.4f}], "
              f"angle={'%.2f' % res['subspace_angle_from_prev'] if res['subspace_angle_from_prev'] else 'N/A'}")

    all_results['pca_convergence'] = pca_convergence
    plot_pca_convergence(pca_results, os.path.join(output_dir, 'pca_convergence.png'))

    # ===============================================================
    # Exp 2d: TADN vs LayerNorm vs NoNorm comparison
    # ===============================================================
    print("\n--- Exp 2d: Normalization comparison ---")

    d1_raw = generate_random_direction(model, seed=42)
    d2_raw = generate_random_direction(model, seed=123)
    d2_raw = orthogonalize_directions(d1_raw, d2_raw)

    norm_methods = {
        'TADN': lambda d: apply_tadn(d, model, units, num_heads, head_dim),
        'LayerNorm': lambda d: apply_layer_normalization(d, model),
        'NoNorm': lambda d: apply_no_normalization(d, model),
    }

    norm_results = {}
    for method_name, norm_fn in norm_methods.items():
        d1 = norm_fn(d1_raw)
        d2 = norm_fn(d2_raw)
        alphas, betas, surface = evaluate_2d_surface(
            model, d1, d2, data['eval_loader'], device,
            grid_range=(-1.0, 1.0), grid_size=args.grid_size,
            max_batches=args.max_eval_batches,
        )
        metrics = compute_surface_metrics(alphas, betas, surface)
        norm_results[method_name] = metrics
        print(f"  {method_name}: loss_range={metrics['loss_range']:.2f}, "
              f"roughness={metrics['roughness']:.4f}, basin_diam={metrics['basin_diameter']:.4f}")

    all_results['normalization_comparison'] = norm_results

    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"GROUP 2 COMPLETE ({elapsed:.1f}s = {elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
