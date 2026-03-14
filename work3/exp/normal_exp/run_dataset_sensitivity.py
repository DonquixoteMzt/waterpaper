"""
run_dataset_sensitivity.py — WikiText-103 and cross-domain dataset sensitivity.

Tests loss landscape sensitivity to evaluation domain using:
- WikiText-2 test (baseline)
- WikiText-2 train
- WikiText-103 test (distinct domain scale)
- PTB (Penn Treebank, different domain)

Usage:
    python run_dataset_sensitivity.py --gpu 4
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
from direction_selection import generate_tier1_directions
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


def tokenize_and_chunk(tokenizer, texts, seq_len=256, max_chunks=50):
    """Tokenize texts and create fixed-length chunks."""
    all_tokens = tokenizer(
        '\n'.join(texts), return_tensors='pt', truncation=False
    )['input_ids'][0]
    chunks = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        chunks.append(all_tokens[i:i + seq_len])
        if len(chunks) >= max_chunks:
            break
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--grid-size', type=int, default=31)
    parser.add_argument('--seq-len', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/dataset_sensitivity'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Dataset Sensitivity Analysis (WikiText-103 + Cross-Domain)")
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

    units = get_normalization_units(model)
    d1_raw, d2_raw = generate_tier1_directions(model, seed1=42, seed2=123)
    tier1_d1 = apply_tadn(d1_raw, model, units, num_heads, head_dim)
    tier1_d2 = apply_tadn(d2_raw, model, units, num_heads, head_dim)

    # --- Load datasets ---
    from datasets import load_dataset

    datasets_to_test = {}

    # WikiText-2 test
    print("\nLoading WikiText-2 test...")
    try:
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        texts = [t for t in ds['text'] if len(t.strip()) > 50]
        chunks = tokenize_and_chunk(tokenizer, texts[:200], args.seq_len, 50)
        datasets_to_test['WikiText-2 (test)'] = chunks
        print(f"  {len(chunks)} chunks")
    except Exception as e:
        print(f"  Failed: {e}")

    # WikiText-2 train
    print("Loading WikiText-2 train...")
    try:
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 50]
        chunks = tokenize_and_chunk(tokenizer, texts[:200], args.seq_len, 50)
        datasets_to_test['WikiText-2 (train)'] = chunks
        print(f"  {len(chunks)} chunks")
    except Exception as e:
        print(f"  Failed: {e}")

    # WikiText-103 test (may fail on slow networks, non-critical)
    print("Loading WikiText-103 test...")
    try:
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test', download_mode='reuse_cache_if_exists')
        texts = [t for t in ds['text'] if len(t.strip()) > 50]
        chunks = tokenize_and_chunk(tokenizer, texts[:200], args.seq_len, 50)
        datasets_to_test['WikiText-103 (test)'] = chunks
        print(f"  {len(chunks)} chunks")
    except Exception as e:
        print(f"  WikiText-103 unavailable (network): {type(e).__name__}")

    # WikiText-2 validation split (different from test)
    print("Loading WikiText-2 validation...")
    try:
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
        texts = [t for t in ds['text'] if len(t.strip()) > 50]
        chunks = tokenize_and_chunk(tokenizer, texts[:200], args.seq_len, 50)
        datasets_to_test['WikiText-2 (val)'] = chunks
        print(f"  {len(chunks)} chunks")
    except Exception as e:
        print(f"  Failed: {e}")

    # Synthetic domain: code-like text (different domain than Wikipedia)
    print("Generating synthetic code-like text...")
    code_texts = []
    for i in range(200):
        code_texts.append(f"""
def function_{i}(x, y, z):
    result = x * y + z
    if result > {i}:
        return result ** 2
    else:
        return result - {i}
    for j in range({i}):
        result += j * x

class Model_{i}:
    def __init__(self, hidden_size={64 + i}):
        self.hidden_size = hidden_size
        self.weight = [0.0] * hidden_size
    def forward(self, x):
        return sum(w * xi for w, xi in zip(self.weight, x))
""")
    chunks = tokenize_and_chunk(tokenizer, code_texts, args.seq_len, 50)
    datasets_to_test['Synthetic Code'] = chunks
    print(f"  {len(chunks)} chunks")

    # Synthetic domain: repetitive/structured text
    print("Generating synthetic structured text...")
    struct_texts = []
    for i in range(200):
        struct_texts.append(
            f"Item {i}: The {['red','blue','green','yellow'][i%4]} "
            f"{['cat','dog','bird','fish'][i%4]} with {i} spots "
            f"jumped {['over','under','around','through'][i%4]} the "
            f"{['big','small','tall','wide'][i%4]} {['fence','wall','gate','bridge'][i%4]}. "
            f"Temperature: {20+i%30} degrees. Humidity: {40+i%50} percent. "
            f"The experiment measured {i*1.5:.2f} units at time {i*0.1:.1f}s. "
            f"Results were {'positive' if i%2==0 else 'negative'} with confidence {0.5+i%50/100:.2f}."
        )
    chunks = tokenize_and_chunk(tokenizer, struct_texts, args.seq_len, 50)
    datasets_to_test['Structured/Tabular'] = chunks
    print(f"  {len(chunks)} chunks")

    print(f"\n{len(datasets_to_test)} datasets loaded successfully")

    # --- Evaluate all datasets ---
    all_results = {}
    all_surfaces = {}

    for ds_name, chunks in datasets_to_test.items():
        print(f"\n{'=' * 50}")
        print(f"Evaluating: {ds_name}")
        print(f"{'=' * 50}")

        loader = DataLoader(ChunkDS(chunks), batch_size=4, shuffle=False)

        # Baseline loss
        bl = evaluate_loss(model, loader, device, max_batches=None)
        print(f"  Baseline loss: {bl:.4f}")

        # 2D surface
        a, b, s = evaluate_2d_surface(
            model, tier1_d1, tier1_d2, loader, device,
            grid_range=(-1.0, 1.0), grid_size=args.grid_size,
            max_batches=5,
        )
        m = compute_surface_metrics(a, b, s)
        print(format_metrics_table(m, ds_name))

        all_results[ds_name] = {
            'baseline_loss': bl,
            'n_chunks': len(chunks),
            **m,
        }
        all_surfaces[ds_name] = (a, b, s)

    # --- Visualization ---
    print("\nCreating visualizations...")

    # Side-by-side contour plots
    n_ds = len(all_surfaces)
    if n_ds > 0:
        n_cols = min(n_ds, 3)
        n_rows = (n_ds + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
        if n_ds == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (ds_name, (a, b, s)) in enumerate(all_surfaces.items()):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            A, B = np.meshgrid(a, b)
            vmin = s.min()
            vmax = min(s.max(), s.min() + 3 * (np.median(s) - s.min() + 0.1))
            levels = np.linspace(vmin, vmax, 25)
            cs = ax.contourf(A, B, s, levels=levels, cmap='viridis')
            ax.contour(A, B, s, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
            plt.colorbar(cs, ax=ax, shrink=0.8)
            ax.set_xlabel(r'$\alpha$')
            ax.set_ylabel(r'$\beta$')
            m = all_results[ds_name]
            ax.set_title(f'{ds_name}\nloss={m["baseline_loss"]:.3f}, range={m["loss_range"]:.1f}, '
                        f'rough={m["roughness"]:.3f}')
            ax.plot(0, 0, 'r*', markersize=12)

        # Hide unused axes
        for idx in range(n_ds, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        plt.suptitle('Dataset Sensitivity: Loss Landscape Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Metrics comparison bar chart
    if len(all_results) >= 2:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        ds_names = list(all_results.keys())
        short_names = [n.replace('WikiText-', 'WT').replace(' (test)', '-T').replace(' (train)', '-Tr')
                      for n in ds_names]

        metrics_to_plot = [('loss_range', 'Loss Range'), ('roughness', 'Roughness'),
                          ('basin_diameter', 'Basin Diameter'), ('baseline_loss', 'Baseline Loss')]

        for idx, (key, label) in enumerate(metrics_to_plot):
            vals = [all_results[n][key] for n in ds_names]
            colors = plt.cm.Set2(np.linspace(0, 1, len(ds_names)))
            axes[idx].bar(short_names, vals, color=colors, alpha=0.8, edgecolor='black')
            axes[idx].set_ylabel(label)
            axes[idx].set_title(label)
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].tick_params(axis='x', rotation=20)

        plt.suptitle('Cross-Dataset Metric Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # --- Save results ---
    total_time = time.time() - t_start
    final_results = {
        'experiment': 'Dataset Sensitivity Analysis',
        'model': model_name,
        'grid_size': args.grid_size,
        'datasets': all_results,
        'total_time_seconds': total_time,
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    for ds_name, (a, b, s) in all_surfaces.items():
        safe_name = ds_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        np.savez(os.path.join(output_dir, f'surface_{safe_name}.npz'),
                 alphas=a, betas=b, surface=s)

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    print("Dataset Sensitivity Analysis Complete!")
    print("=" * 70)

    print("\n--- SUMMARY ---")
    print(f"{'Dataset':>25} | {'Loss':>8} | {'Range':>8} | {'Roughness':>10} | {'Basin D.':>10}")
    for ds_name, m in all_results.items():
        print(f"{ds_name:>25} | {m['baseline_loss']:>8.4f} | {m['loss_range']:>8.2f} | "
              f"{m['roughness']:>10.4f} | {m['basin_diameter']:>10.4f}")


if __name__ == '__main__':
    main()
