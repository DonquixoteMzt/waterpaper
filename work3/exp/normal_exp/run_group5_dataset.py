"""
Group 5: Dataset Sensitivity Analysis.

Investigates how the loss landscape shape varies across different
evaluation domains using the same model and directions.
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

from data_loader import prepare_data, prepare_custom_data
from normalization import get_normalization_units, apply_tadn
from direction_selection import generate_tier1_directions
from grid_evaluation import evaluate_2d_surface
from metrics import compute_surface_metrics
from visualization import plot_2d_surface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--grid-size', type=int, default=31)
    parser.add_argument('--max-eval-batches', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    output_dir = 'results/group5_dataset'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GROUP 5: Dataset Sensitivity Analysis")
    print("=" * 70)
    t_start = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_name = "Qwen/Qwen3-0.6B-Base"
    print(f"\n--- Loading model: {model_name} ---")
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

    # Generate SAME directions for all datasets
    print("\n--- Generating directions (same for all datasets) ---")
    d1_raw, d2_raw = generate_tier1_directions(model, seed1=42, seed2=123)
    d1_tadn = apply_tadn(d1_raw, model, units, num_heads, head_dim)
    d2_tadn = apply_tadn(d2_raw, model, units, num_heads, head_dim)

    # Prepare datasets (only use locally cached ones)
    datasets_config = {
        'WikiText-2-test': {
            'loader': lambda: load_dataset('wikitext', 'wikitext-2-raw-v1', split='test'),
            'text_key': 'text',
            'filter': lambda t: len(t.strip()) > 50,
        },
        'WikiText-2-train': {
            'loader': lambda: load_dataset('wikitext', 'wikitext-2-raw-v1', split='train'),
            'text_key': 'text',
            'filter': lambda t: len(t.strip()) > 50,
        },
        'WikiText-103-test': {
            'loader': lambda: load_dataset('wikitext', 'wikitext-103-raw-v1', split='test'),
            'text_key': 'text',
            'filter': lambda t: len(t.strip()) > 50,
        },
    }

    all_results = {}

    for ds_name, ds_cfg in datasets_config.items():
        print(f"\n--- Evaluating on {ds_name} ---")
        try:
            if ds_cfg.get('streaming', False):
                ds = ds_cfg['loader']()
                texts = []
                for i, item in enumerate(ds):
                    if ds_cfg['filter'](item[ds_cfg['text_key']]):
                        texts.append(item[ds_cfg['text_key']])
                    if len(texts) >= 100:
                        break
            else:
                ds = ds_cfg['loader']()
                texts = [item[ds_cfg['text_key']] for item in ds
                         if ds_cfg['filter'](item[ds_cfg['text_key']])][:100]

            eval_loader = prepare_custom_data(tokenizer, texts, seq_len=256,
                                               batch_size=4, max_chunks=50)

            from grid_evaluation import evaluate_loss
            baseline_loss = evaluate_loss(model, eval_loader, device, args.max_eval_batches)
            print(f"  Baseline loss: {baseline_loss:.4f}")

            alphas, betas, surface = evaluate_2d_surface(
                model, d1_tadn, d2_tadn, eval_loader, device,
                grid_range=(-1.0, 1.0), grid_size=args.grid_size,
                max_batches=args.max_eval_batches,
            )
            metrics = compute_surface_metrics(alphas, betas, surface)

            plot_2d_surface(alphas, betas, surface, f'Qwen3-0.6B-Base on {ds_name}',
                            os.path.join(output_dir, f'surface_{ds_name.lower().replace("-","")}.png'))

            np.savez(os.path.join(output_dir, f'surface_{ds_name.lower().replace("-","")}.npz'),
                     alphas=alphas, betas=betas, surface=surface)

            all_results[ds_name] = {
                'baseline_loss': baseline_loss,
                'metrics': metrics,
            }
            print(f"  {ds_name}: loss_range={metrics['loss_range']:.2f}, "
                  f"roughness={metrics['roughness']:.4f}, basin_diam={metrics['basin_diameter']:.4f}")

        except Exception as e:
            print(f"  ERROR with {ds_name}: {e}")
            all_results[ds_name] = {'error': str(e)}

    # Also run WikiText-2 with Tier 2 directions for comparison
    print("\n--- WikiText-2 with Grad PCA for reference ---")
    config_wt = {
        'data': {
            'dataset': 'wikitext', 'dataset_config': 'wikitext-2-raw-v1',
            'split': 'test', 'seq_len': 256, 'n_eval_chunks': 50,
            'eval_batch_size': 4, 'max_eval_batches': args.max_eval_batches,
            'n_grad_batches': 50, 'grad_batch_size': 1,
        },
        'direction': {'tier3': {'hvp_max_batches': 3, 'hvp_batch_size': 1}},
    }
    data_wt = prepare_data(tokenizer, config_wt)
    from direction_selection import gradient_pca_with_convergence
    pca_results, pca_dirs = gradient_pca_with_convergence(
        model, data_wt['grad_loader'], device,
        n_max=50, checkpoints=[10, 25, 50], k=2,
    )
    d1_pca = apply_tadn({k: v.cpu() for k, v in pca_dirs[0].items()},
                         model, units, num_heads, head_dim)
    d2_pca = apply_tadn({k: v.cpu() for k, v in pca_dirs[1].items()},
                         model, units, num_heads, head_dim)

    # Evaluate on all datasets with PCA directions
    for ds_name in all_results:
        if 'error' in all_results[ds_name]:
            continue
        try:
            if ds_name == 'WikiText-2':
                eval_loader = data_wt['eval_loader']
            else:
                ds_info = datasets_config[ds_name]
                if ds_info.get('streaming', False):
                    ds = ds_info['loader']()
                    texts = []
                    for i, item in enumerate(ds):
                        if ds_info['filter'](item[ds_info['text_key']]):
                            texts.append(item[ds_info['text_key']])
                        if len(texts) >= 100:
                            break
                else:
                    ds = ds_info['loader']()
                    texts = [item[ds_info['text_key']] for item in ds
                             if ds_info['filter'](item[ds_info['text_key']])][:100]
                eval_loader = prepare_custom_data(tokenizer, texts, seq_len=256,
                                                   batch_size=4, max_chunks=50)

            alphas_p, betas_p, surface_p = evaluate_2d_surface(
                model, d1_pca, d2_pca, eval_loader, device,
                grid_range=(-1.0, 1.0), grid_size=args.grid_size,
                max_batches=args.max_eval_batches,
            )
            metrics_p = compute_surface_metrics(alphas_p, betas_p, surface_p)
            all_results[ds_name]['metrics_pca'] = metrics_p
            print(f"  {ds_name} (PCA): loss_range={metrics_p['loss_range']:.2f}")

        except Exception as e:
            print(f"  PCA eval error for {ds_name}: {e}")

    # Comparison plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid_ds = [ds for ds in all_results if 'error' not in all_results[ds]]
    if len(valid_ds) >= 2:
        metrics_keys = ['loss_range', 'roughness', 'basin_diameter', 'curvature_ratio', 'basin_flatness']
        fig, axes = plt.subplots(1, len(metrics_keys), figsize=(4 * len(metrics_keys), 5))

        x = np.arange(len(valid_ds))
        for idx, key in enumerate(metrics_keys):
            vals = [all_results[ds]['metrics'].get(key, 0) for ds in valid_ds]
            axes[idx].bar(x, vals, alpha=0.8, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(valid_ds)])
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(valid_ds, rotation=15, fontsize=8)
            axes[idx].set_title(key.replace('_', ' ').title())
            axes[idx].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Dataset Sensitivity: Landscape Metrics (Random+TADN)', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"GROUP 5 COMPLETE ({elapsed:.1f}s = {elapsed/60:.1f} min)")
    print(f"{'='*70}")
    for ds_name, r in all_results.items():
        if 'error' not in r:
            m = r['metrics']
            print(f"  {ds_name}: loss={r['baseline_loss']:.4f}, "
                  f"range={m['loss_range']:.2f}, rough={m['roughness']:.4f}")


if __name__ == '__main__':
    main()
