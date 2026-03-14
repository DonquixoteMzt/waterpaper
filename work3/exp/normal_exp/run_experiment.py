"""
run_experiment.py — Main experiment runner for LLMScape.

Implements the full pipeline:
1. Load model and data
2. TADN invariance test (vs Layer Norm baseline)
3. Three-tier direction selection (Random, Grad PCA, Hessian)
4. PFI computation for all tiers
5. 2D loss surface evaluation and geometric metric extraction
6. Visualization and results saving

Usage:
    python run_experiment.py --config config.yaml
    python run_experiment.py --config config.yaml --gpu 0
"""

import os
import sys
import json
import time
import copy
import gc
import math
import argparse

import yaml
import numpy as np
import torch

from data_loader import prepare_data
from normalization import (
    get_normalization_units, apply_tadn, apply_layer_normalization,
    apply_no_normalization, create_rescaled_model,
)
from direction_selection import (
    generate_random_direction, generate_tier1_directions,
    orthogonalize_directions, gradient_pca_with_convergence,
    power_iteration_hessian, compute_curvature_aware_scale,
)
from pfi import compute_hutchinson_tr_h2, compute_pfi
from grid_evaluation import evaluate_loss, evaluate_2d_surface, evaluate_1d_curve
from metrics import compute_surface_metrics, format_metrics_table
from visualization import (
    plot_2d_surface, plot_1d_comparison, plot_tier_comparison,
    plot_pfi_comparison, plot_tadn_invariance, plot_pca_convergence,
    plot_metrics_comparison,
)


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config, device):
    """Load model and tokenizer from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config['model']['name']
    dtype_str = config['model'].get('dtype', 'bfloat16')
    dtype = getattr(torch, dtype_str)
    attn_impl = config['model'].get('attn_implementation', 'eager')

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to(device)
    model.eval()

    # Extract architecture info
    cfg = model.config
    num_heads = getattr(cfg, 'num_attention_heads', None)
    head_dim = getattr(cfg, 'head_dim', None)
    if head_dim is None and num_heads:
        hidden_size = getattr(cfg, 'hidden_size', None)
        if hidden_size:
            head_dim = hidden_size // num_heads
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  num_heads={num_heads}, head_dim={head_dim}, params={n_params:,}")
    return model, tokenizer, num_heads, head_dim, n_params


def run_tadn_invariance_test(model, eval_loader, device, config,
                              units, num_heads, head_dim, output_dir):
    """
    Experiment 1: TADN Scale-Invariance Test.

    Compares TADN vs Layer Norm under FFN neuron rescaling.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: TADN Scale-Invariance Test")
    print("=" * 70)

    max_batches = config['data']['max_eval_batches']
    grid_range = config['grid']['range']

    # Create rescaled model
    print("Creating rescaled model (non-uniform FFN neuron scaling)...")
    model_rescaled = create_rescaled_model(model)
    model_rescaled.eval()

    # Verify equivalence
    loss_orig = evaluate_loss(model, eval_loader, device, max_batches)
    loss_resc = evaluate_loss(model_rescaled, eval_loader, device, max_batches)
    print(f"  Original loss: {loss_orig:.6f}, Rescaled loss: {loss_resc:.6f}, Diff: {abs(loss_orig-loss_resc):.6f}")

    units_resc = get_normalization_units(model_rescaled)
    raw_d = generate_random_direction(model, seed=42)

    # Apply normalizations
    d_tadn_orig = apply_tadn(raw_d, model, units, num_heads, head_dim)
    d_tadn_resc = apply_tadn(raw_d, model_rescaled, units_resc, num_heads, head_dim)
    d_layer_orig = apply_layer_normalization(raw_d, model)
    d_layer_resc = apply_layer_normalization(raw_d, model_rescaled)

    # Move to device
    for d in [d_tadn_orig, d_tadn_resc, d_layer_orig, d_layer_resc]:
        for k in d:
            d[k] = d[k].to(device)

    # Evaluate 1D curves
    n_pts = config['tadn_test']['n_1d_points']
    alpha_rng = (-grid_range, grid_range)

    print("Evaluating 1D loss curves...")
    a1, l1 = evaluate_1d_curve(model, d_tadn_orig, eval_loader, device, alpha_rng, n_pts, max_batches)
    a2, l2 = evaluate_1d_curve(model_rescaled, d_tadn_resc, eval_loader, device, alpha_rng, n_pts, max_batches)
    a3, l3 = evaluate_1d_curve(model, d_layer_orig, eval_loader, device, alpha_rng, n_pts, max_batches)
    a4, l4 = evaluate_1d_curve(model_rescaled, d_layer_resc, eval_loader, device, alpha_rng, n_pts, max_batches)

    corr_tadn = float(np.corrcoef(l1, l2)[0, 1])
    corr_layer = float(np.corrcoef(l3, l4)[0, 1])
    mse_tadn = float(np.mean((l1 - l2) ** 2))
    mse_layer = float(np.mean((l3 - l4) ** 2))

    print(f"  TADN: corr={corr_tadn:.6f}, MSE={mse_tadn:.6f}")
    print(f"  Layer: corr={corr_layer:.6f}, MSE={mse_layer:.6f}")

    # Plot
    plot_tadn_invariance(
        curves={
            'TADN (Original)': (a1, l1), 'TADN (Rescaled)': (a2, l2),
            'Layer Norm (Original)': (a3, l3), 'Layer Norm (Rescaled)': (a4, l4),
        },
        deviations={
            'TADN': (a1, np.abs(l1 - l2)),
            'Layer Norm': (a3, np.abs(l3 - l4)),
        },
        correlations={'TADN': corr_tadn, 'Layer Norm': corr_layer},
        filename=os.path.join(output_dir, 'exp1_tadn_invariance.png'),
    )
    print(f"  Saved plot to {output_dir}/exp1_tadn_invariance.png")

    # Clean up
    del model_rescaled, d_tadn_orig, d_tadn_resc, d_layer_orig, d_layer_resc
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'loss_original': loss_orig,
        'loss_rescaled': loss_resc,
        'tadn_correlation': corr_tadn,
        'layer_correlation': corr_layer,
        'tadn_mse': mse_tadn,
        'layer_mse': mse_layer,
    }


def run_direction_selection(model, data, device, config, units, num_heads, head_dim, output_dir):
    """
    Experiments 2-3: Direction selection (Tier 1-3).

    Returns all directions and Hessian eigenvalues.
    """
    from transformers import AutoModelForCausalLM

    grid_range = config['grid']['range']
    model_name = config['model']['name']
    results = {}

    # --- Tier 1: Random directions ---
    print("\n" + "=" * 70)
    print("TIER 1: Random Directions + TADN")
    print("=" * 70)
    d1_raw, d2_raw = generate_tier1_directions(model, seed1=42, seed2=123)
    tier1_d1 = apply_tadn(d1_raw, model, units, num_heads, head_dim)
    tier1_d2 = apply_tadn(d2_raw, model, units, num_heads, head_dim)
    tier1_d1 = {k: v.to(device) for k, v in tier1_d1.items()}
    tier1_d2 = {k: v.to(device) for k, v in tier1_d2.items()}

    # --- Tier 1 Baseline: Layer Norm ---
    d1_layer = apply_layer_normalization(d1_raw, model)
    d2_layer = apply_layer_normalization(d2_raw, model)
    d1_layer = {k: v.to(device) for k, v in d1_layer.items()}
    d2_layer = {k: v.to(device) for k, v in d2_layer.items()}

    # --- Tier 2: Gradient PCA ---
    print("\n" + "=" * 70)
    print("TIER 2: Gradient Covariance PCA + TADN")
    print("=" * 70)
    tier2_cfg = config['direction']['tier2']
    pca_results, pca_directions = gradient_pca_with_convergence(
        model, data['grad_loader'], device,
        n_max=tier2_cfg['n_grad_samples'],
        checkpoints=tier2_cfg['checkpoints'],
        k=tier2_cfg['k'],
        convergence_threshold_deg=tier2_cfg['convergence_threshold_deg'],
    )

    # Plot PCA convergence
    plot_pca_convergence(pca_results, os.path.join(output_dir, 'exp2_pca_convergence.png'))
    print(f"  Saved PCA convergence plot")

    # Apply TADN to PCA directions
    pca_d0_cpu = {k: v.cpu() for k, v in pca_directions[0].items()}
    pca_d1_cpu = {k: v.cpu() for k, v in pca_directions[1].items()}
    tier2_d1 = apply_tadn(pca_d0_cpu, model, units, num_heads, head_dim)
    tier2_d2 = apply_tadn(pca_d1_cpu, model, units, num_heads, head_dim)
    tier2_d1 = {k: v.to(device) for k, v in tier2_d1.items()}
    tier2_d2 = {k: v.to(device) for k, v in tier2_d2.items()}

    results['pca_convergence'] = {
        str(c): {
            'eigenvalues': pca_results[c]['eigenvalues'],
            'explained_ratios': pca_results[c]['explained_ratios'],
            'subspace_angle': pca_results[c].get('subspace_angle_from_prev'),
        }
        for c in pca_results
    }

    # --- Tier 3: Hessian Eigenvectors ---
    print("\n" + "=" * 70)
    print("TIER 3: Hessian Eigenvector Directions")
    print("=" * 70)

    # Move bf16 model to CPU, load fp32 model
    print("Moving bf16 model to CPU for Hessian computation...")
    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    fp32_device = torch.device(f'cuda:{config["hardware"].get("fp32_gpu", 0)}')
    print(f"Loading fp32 model on GPU {fp32_device}...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True,
        attn_implementation="eager",
    ).to(fp32_device)
    model_fp32.eval()

    tier3_cfg = config['direction']['tier3']
    hessian_vecs, hessian_eigs = power_iteration_hessian(
        model_fp32, data['hvp_loader'], fp32_device,
        n_iter=tier3_cfg['n_iter'],
        n_vectors=tier3_cfg['n_vectors'],
        max_batches=tier3_cfg['hvp_max_batches'],
        convergence_tol=tier3_cfg['convergence_tol'],
    )

    l_char, tier3_range = compute_curvature_aware_scale(
        hessian_eigs, config['grid']['tier3_scale_factor']
    )
    print(f"  l_char = {l_char:.6f}, tier3_range = {tier3_range:.6f}")

    results['hessian'] = {
        'eigenvalues': hessian_eigs,
        'l_char': l_char,
        'tier3_range': tier3_range,
    }

    # Apply TADN to Hessian directions (model on CPU)
    hess_d0 = {k: v.cpu() for k, v in hessian_vecs[0].items()}
    hess_d1 = {k: v.cpu() for k, v in hessian_vecs[1].items()}
    del hessian_vecs
    tier3_d1 = apply_tadn(hess_d0, model, units, num_heads, head_dim)
    tier3_d2 = apply_tadn(hess_d1, model, units, num_heads, head_dim)
    del hess_d0, hess_d1

    # --- PFI computation ---
    # Move ALL directions to CPU to free GPU memory for HVP computation
    print("\nMoving all directions to CPU for PFI computation...")
    tier1_d1 = {k: v.cpu() for k, v in tier1_d1.items()}
    tier1_d2 = {k: v.cpu() for k, v in tier1_d2.items()}
    d1_layer = {k: v.cpu() for k, v in d1_layer.items()}
    d2_layer = {k: v.cpu() for k, v in d2_layer.items()}
    tier2_d1 = {k: v.cpu() for k, v in tier2_d1.items()}
    tier2_d2 = {k: v.cpu() for k, v in tier2_d2.items()}
    # tier3_d1/d2 already on CPU from apply_tadn
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "=" * 70)
    print("PFI Computation for All Tiers")
    print("=" * 70)

    pfi_cfg = config['pfi']
    print("Computing tr(H^2) via Hutchinson estimator...")
    tr_h2, tr_h2_std = compute_hutchinson_tr_h2(
        model_fp32, data['hvp_loader'], fp32_device,
        n_hutchinson=pfi_cfg['n_hutchinson'],
        max_batches=pfi_cfg['hvp_max_batches'],
    )

    pfi_results = {}
    for tier_name, d1, d2 in [
        ('Tier 1 (Random+TADN)', tier1_d1, tier1_d2),
        ('Tier 1 (Layer Norm)', d1_layer, d2_layer),
        ('Tier 2 (Grad PCA+TADN)', tier2_d1, tier2_d2),
        ('Tier 3 (Hessian+TADN)', tier3_d1, tier3_d2),
    ]:
        print(f"\n--- {tier_name} ---")
        pfi = compute_pfi(
            model_fp32, data['hvp_loader'], fp32_device, d1, d2,
            lambda_max=hessian_eigs[0], tr_h2=tr_h2, tr_h2_std=tr_h2_std,
            max_batches=pfi_cfg['hvp_max_batches'],
        )
        pfi_results[tier_name] = pfi

    plot_pfi_comparison(pfi_results, os.path.join(output_dir, 'exp4_pfi_comparison.png'))
    print(f"  Saved PFI comparison plot")

    results['pfi'] = {k: {kk: vv for kk, vv in v.items()} for k, v in pfi_results.items()}

    # Clean up fp32 model and restore bf16 model
    del model_fp32
    torch.cuda.empty_cache()
    gc.collect()
    print("Moving bf16 model back to GPU...")
    model.to(device)
    model.eval()

    # Keep directions on CPU — grid_evaluation handles device transfer per-param
    directions = {
        'tier1_tadn': (tier1_d1, tier1_d2),
        'tier1_layer': (d1_layer, d2_layer),
        'tier2': (tier2_d1, tier2_d2),
        'tier3': (tier3_d1, tier3_d2),
    }

    return results, directions, tier3_range


def run_surface_evaluation(model, eval_loader, device, config, directions,
                            tier3_range, output_dir):
    """
    Experiment 5: 2D loss surface evaluation for all tiers.
    """
    print("\n" + "=" * 70)
    print("2D LOSS SURFACE EVALUATION")
    print("=" * 70)

    grid_size = config['grid']['size']
    grid_range = config['grid']['range']
    max_batches = config['data']['max_eval_batches']

    surfaces = {}
    all_metrics = {}

    eval_configs = [
        ('Tier1_TADN', directions['tier1_tadn'], grid_range),
        ('Tier1_LayerNorm', directions['tier1_layer'], grid_range),
        ('Tier2_GradPCA', directions['tier2'], grid_range),
        ('Tier3_Hessian', directions['tier3'], tier3_range),
    ]

    for name, (d1, d2), grange in eval_configs:
        print(f"\n--- {name} (range={grange:.4f}, grid={grid_size}x{grid_size}) ---")
        alphas, betas, surface = evaluate_2d_surface(
            model, d1, d2, eval_loader, device,
            grid_range=(-grange, grange),
            grid_size=grid_size,
            max_batches=max_batches,
        )
        surfaces[name] = (alphas, betas, surface)

        # Plot individual surface
        plot_2d_surface(alphas, betas, surface, name,
                        os.path.join(output_dir, f'surface_{name.lower()}.png'))

        # Compute metrics
        mets = compute_surface_metrics(alphas, betas, surface)
        all_metrics[name] = mets
        print(format_metrics_table(mets, name))

    # Side-by-side comparison (TADN tiers only)
    tadn_surfaces = {k: v for k, v in surfaces.items() if 'TADN' in k or 'Tier2' in k or 'Tier3' in k}
    if len(tadn_surfaces) > 1:
        plot_tier_comparison(tadn_surfaces,
                             os.path.join(output_dir, 'tier_comparison.png'),
                             'LLMScape: Direction Selection Comparison')

    # Metrics comparison plot
    if len(all_metrics) > 1:
        plot_metrics_comparison(
            list(all_metrics.values()),
            list(all_metrics.keys()),
            os.path.join(output_dir, 'metrics_comparison.png'),
        )

    return surfaces, all_metrics


def main():
    parser = argparse.ArgumentParser(description='LLMScape Experiment Runner')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--gpu', type=int, default=None, help='Override GPU index')
    parser.add_argument('--skip-tadn-test', action='store_true', help='Skip TADN invariance test')
    parser.add_argument('--skip-tier3', action='store_true', help='Skip Tier 3 (Hessian)')
    parser.add_argument('--skip-pfi', action='store_true', help='Skip PFI computation')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.gpu is not None:
        config['hardware']['gpu'] = args.gpu
        config['hardware']['fp32_gpu'] = args.gpu

    gpu_idx = config['hardware']['gpu']
    device = torch.device(f'cuda:{gpu_idx}' if torch.cuda.is_available() else 'cpu')
    output_dir = config['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("LLMScape: Loss Landscape Visualization for LLMs")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    results_log = {'config': config, 'experiments': {}}
    t_start = time.time()

    # --- Setup ---
    model, tokenizer, num_heads, head_dim, n_params = setup_model_and_tokenizer(config, device)
    data = prepare_data(tokenizer, config)
    units = get_normalization_units(model)

    results_log['model_info'] = {
        'name': config['model']['name'],
        'n_params': n_params,
        'num_heads': num_heads,
        'head_dim': head_dim,
    }

    # Baseline loss
    baseline_loss = evaluate_loss(model, data['eval_loader'], device,
                                   config['data']['max_eval_batches'])
    print(f"\nBaseline loss: {baseline_loss:.4f}")
    results_log['baseline_loss'] = baseline_loss

    # --- Experiment 1: TADN Invariance Test ---
    if not args.skip_tadn_test:
        tadn_results = run_tadn_invariance_test(
            model, data['eval_loader'], device, config,
            units, num_heads, head_dim, output_dir,
        )
        results_log['experiments']['tadn_invariance'] = tadn_results

    # --- Experiments 2-4: Direction Selection + PFI ---
    dir_results, directions, tier3_range = run_direction_selection(
        model, data, device, config, units, num_heads, head_dim, output_dir,
    )
    results_log['experiments'].update(dir_results)

    # --- Experiment 5: Surface Evaluation ---
    surfaces, surface_metrics = run_surface_evaluation(
        model, data['eval_loader'], device, config, directions,
        tier3_range, output_dir,
    )
    results_log['experiments']['surfaces'] = {
        k: v for k, v in surface_metrics.items()
    }

    # --- Save results ---
    total_time = time.time() - t_start
    results_log['total_time_seconds'] = total_time

    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results_log, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # --- Print Summary ---
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Model: {config['model']['name']} ({n_params:,} params)")
    print(f"Baseline loss: {baseline_loss:.4f}")

    if 'tadn_invariance' in results_log['experiments']:
        ti = results_log['experiments']['tadn_invariance']
        print(f"\nTADN Invariance:")
        print(f"  TADN corr: {ti['tadn_correlation']:.6f}")
        print(f"  Layer corr: {ti['layer_correlation']:.6f}")

    if 'hessian' in results_log['experiments']:
        h = results_log['experiments']['hessian']
        print(f"\nHessian Eigenvalues:")
        print(f"  lambda_1 = {h['eigenvalues'][0]:.2f}")
        print(f"  lambda_2 = {h['eigenvalues'][1]:.2f}")
        print(f"  l_char = {h['l_char']:.6f}")

    if 'pfi' in results_log['experiments']:
        print(f"\nPFI Comparison:")
        for name, pfi in results_log['experiments']['pfi'].items():
            s = pfi.get('PFI_S')
            c = pfi.get('PFI_C')
            s_str = f"{s:.2e}" if s else "N/A"
            c_str = f"{c:.2e}" if c else "N/A"
            print(f"  {name}: PFI-S={s_str}, PFI-C={c_str}")

    print(f"\nSurface Metrics:")
    for name, m in surface_metrics.items():
        print(f"  {name}: loss_range={m['loss_range']:.2f}, "
              f"roughness={m['roughness']:.4f}, "
              f"basin_diam={m['basin_diameter']:.4f}")

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)

    # Save numpy surfaces for later use
    if config['experiment'].get('save_surfaces', True):
        for name, (alphas, betas, surface) in surfaces.items():
            np.savez(os.path.join(output_dir, f'surface_{name.lower()}.npz'),
                     alphas=alphas, betas=betas, surface=surface)

    return results_log


if __name__ == '__main__':
    main()
