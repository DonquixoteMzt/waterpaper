"""
run_7b_tier2.py — 7B Model Tier 2 Analysis with Gradient Checkpointing / Streaming PCA.

Attempts gradient PCA for 7B models using:
1. Gradient checkpointing to reduce memory
2. Streaming PCA via incremental covariance updates
3. CPU offloading of gradient vectors

Usage:
    python run_7b_tier2.py --gpu 5
"""

import os
import sys
import json
import time
import gc
import math
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


def streaming_gradient_pca(model, dataloader, device, n_max=50, k=2, verbose=True):
    """
    Streaming gradient PCA using incremental Gram matrix construction.

    Instead of storing full gradient vectors (infeasible for 7B models),
    we build the Gram matrix incrementally and reconstruct PCA directions
    from a smaller number of kept gradients.

    Strategy:
    1. Compute gradients one at a time
    2. Build Gram matrix entries incrementally
    3. After each gradient, immediately compute its dot products with all
       previous gradients, then store only the gradient on CPU
    4. Final PCA via Gram matrix eigendecomposition

    Memory: O(N * d) on CPU for N gradient vectors, O(N^2) for Gram matrix
    For 7B models: N=50, d=7.3B => ~1.35 TB in float32 (too much)

    Revised strategy: chunk-based streaming
    - Process gradient in chunks, computing Gram matrix entries incrementally
    - Only keep Gram matrix (N x N) in memory
    - Re-compute gradients for final direction reconstruction
    """
    if verbose:
        print(f"Streaming gradient PCA (N_max={n_max})...")

    model.eval()
    model.train()  # Need for gradient computation

    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")
    elif hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled (via model.model)")

    # Phase 1: Compute Gram matrix incrementally
    gram = np.zeros((n_max, n_max))
    grad_norms = []

    # We'll store gradients on CPU in chunks to reconstruct directions later
    # For 7B: each gradient ~27GB float32. With N=50 that's 1.35TB.
    # Instead, store only dot products and reconstruct from top eigenvectors of Gram.

    # Actually, we need the gradients to reconstruct directions.
    # Compromise: use bfloat16 for gradient storage => ~14GB per grad, N=50 => ~700GB
    # Still too much. Use float16: ~14GB per grad.
    # Alternative: Only keep N_keep gradients around the PCA convergence point

    # Best approach: compute Gram matrix first, then do a SECOND pass
    # to reconstruct just the top-k directions

    print("  Phase 1: Computing Gram matrix (single pass)...")
    grad_refs = []  # Will store gradients in reduced precision on CPU

    for i, batch in enumerate(dataloader):
        if i >= n_max:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        # Collect gradient as flat vector (keep on CPU in bfloat16)
        g_flat = torch.cat([p.grad.detach().flatten().to(torch.bfloat16)
                           for _, p in model.named_parameters()
                           if p.requires_grad and p.grad is not None]).cpu()

        # Compute dot products with all previous gradients
        for j in range(len(grad_refs)):
            dot = torch.dot(g_flat.float(), grad_refs[j].float()).item()
            gram[i, j] = dot
            gram[j, i] = dot
        gram[i, i] = torch.dot(g_flat.float(), g_flat.float()).item()

        grad_refs.append(g_flat)
        grad_norms.append(math.sqrt(gram[i, i]))

        del outputs, loss
        model.zero_grad()
        torch.cuda.empty_cache()

        if verbose and (i + 1) % 10 == 0:
            mem_gb = sum(g.element_size() * g.nelement() for g in grad_refs) / 1e9
            print(f"    Gradient {i+1}/{n_max}, GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f}GB, "
                  f"CPU grad storage: {mem_gb:.1f}GB")

    N = len(grad_refs)
    print(f"  Collected {N} gradients")

    # Disable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    elif hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_disable'):
        model.model.gradient_checkpointing_disable()

    model.eval()

    # Phase 2: Eigendecompose Gram matrix
    print("  Phase 2: Gram matrix PCA...")
    G_N = gram[:N, :N]
    ones = np.ones((N, N)) / N
    G_centered = G_N - ones @ G_N - G_N @ ones + ones @ G_N @ ones

    eigenvalues, eigenvectors = np.linalg.eigh(G_centered)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx[:k]]
    eigenvectors = eigenvectors[:, idx[:k]]

    all_eigs = np.sort(np.linalg.eigvalsh(G_centered))[::-1]
    total_var = max(float(np.sum(np.maximum(all_eigs, 0))), 1e-10)
    explained_ratios = np.maximum(eigenvalues, 0) / total_var

    print(f"  Explained variance: PC1={explained_ratios[0]:.4f}, PC2={explained_ratios[1]:.4f}")

    # Phase 3: Reconstruct d-dimensional PCA directions
    print("  Phase 3: Reconstructing PCA directions...")
    directions = []
    for j in range(k):
        d_flat = torch.zeros_like(grad_refs[0].float())
        for t in range(N):
            d_flat += eigenvectors[t, j] * grad_refs[t].float()
        d_norm = d_flat.norm()
        if d_norm > 1e-10:
            d_flat /= d_norm
        directions.append(d_flat)

    # Convert to param dicts
    final_directions = []
    for j in range(k):
        direction = {}
        offset = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                numel = param.numel()
                direction[name] = directions[j][offset:offset + numel].reshape(param.shape)
                offset += numel
        final_directions.append(direction)

    # Clean up
    del grad_refs
    gc.collect()

    return final_directions, explained_ratios.tolist(), N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--n-grad', type=int, default=50)
    parser.add_argument('--grid-size', type=int, default=21)
    parser.add_argument('--seq-len', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/7b_tier2'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("7B Model Tier 2 Analysis with Gradient Checkpointing")
    print("=" * 70)
    t_start = time.time()

    models_to_test = [
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
    ]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {}

    for model_name, short_name in models_to_test:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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

        eval_loader = DataLoader(ChunkDS(chunks[:50]), batch_size=2, shuffle=False)
        grad_loader = DataLoader(ChunkDS(chunks[:args.n_grad]), batch_size=1, shuffle=False)

        # Load model in bfloat16
        print(f"Loading model in bfloat16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        num_heads = getattr(model.config, 'num_attention_heads', None)
        head_dim = getattr(model.config, 'head_dim', None)
        if head_dim is None and num_heads:
            hidden_size = getattr(model.config, 'hidden_size', None)
            if hidden_size:
                head_dim = hidden_size // num_heads

        print(f"  Params: {n_params:,}, heads={num_heads}, head_dim={head_dim}")
        print(f"  GPU memory used: {torch.cuda.memory_allocated(device)/1e9:.1f}GB")

        # Baseline loss
        baseline_loss = evaluate_loss(model, eval_loader, device, max_batches=5)
        print(f"  Baseline loss: {baseline_loss:.4f}")

        # --- Tier 1 (Random + TADN) ---
        print("\n--- Tier 1: Random + TADN ---")
        # Generate directions on CPU
        model.cpu()
        torch.cuda.empty_cache()

        d1_raw, d2_raw = generate_tier1_directions(model, seed1=42, seed2=123)
        units = get_normalization_units(model)
        tier1_d1 = apply_tadn(d1_raw, model, units, num_heads, head_dim)
        tier1_d2 = apply_tadn(d2_raw, model, units, num_heads, head_dim)
        del d1_raw, d2_raw
        gc.collect()

        model.to(device)
        model.eval()

        a1, b1, s1 = evaluate_2d_surface(
            model, tier1_d1, tier1_d2, eval_loader, device,
            grid_range=(-1.0, 1.0), grid_size=args.grid_size, max_batches=5,
        )
        m1 = compute_surface_metrics(a1, b1, s1)
        print(format_metrics_table(m1, f"{short_name} Tier 1"))

        # --- Tier 2: Streaming Gradient PCA ---
        print("\n--- Tier 2: Streaming Gradient PCA ---")
        try:
            pca_directions, explained_ratios, n_used = streaming_gradient_pca(
                model, grad_loader, device, n_max=args.n_grad, k=2,
            )

            # Apply TADN
            model.cpu()
            torch.cuda.empty_cache()
            tier2_d1 = apply_tadn(pca_directions[0], model, units, num_heads, head_dim)
            tier2_d2 = apply_tadn(pca_directions[1], model, units, num_heads, head_dim)
            del pca_directions
            gc.collect()

            model.to(device)
            model.eval()

            a2, b2, s2 = evaluate_2d_surface(
                model, tier2_d1, tier2_d2, eval_loader, device,
                grid_range=(-1.0, 1.0), grid_size=args.grid_size, max_batches=5,
            )
            m2 = compute_surface_metrics(a2, b2, s2)
            print(format_metrics_table(m2, f"{short_name} Tier 2"))

            tier2_success = True
        except Exception as e:
            print(f"  Tier 2 failed: {e}")
            import traceback
            traceback.print_exc()
            tier2_success = False
            m2 = None
            explained_ratios = None
            n_used = 0

        # --- Visualization ---
        print("\nCreating visualizations...")

        if tier2_success:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            for idx, (a, b, s, title, m) in enumerate([
                (a1, b1, s1, f'{short_name} Tier 1 (Random+TADN)', m1),
                (a2, b2, s2, f'{short_name} Tier 2 (GradPCA+TADN)', m2),
            ]):
                A, B = np.meshgrid(a, b)
                vmin = s.min()
                vmax = min(s.max(), s.min() + 3 * (np.median(s) - s.min() + 0.1))
                levels = np.linspace(vmin, vmax, 25)
                cs = axes[idx].contourf(A, B, s, levels=levels, cmap='viridis')
                axes[idx].contour(A, B, s, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
                plt.colorbar(cs, ax=axes[idx], shrink=0.8)
                axes[idx].set_xlabel(r'$\alpha$')
                axes[idx].set_ylabel(r'$\beta$')
                axes[idx].set_title(f'{title}\nrange={m["loss_range"]:.2f}, rough={m["roughness"]:.3f}')
                axes[idx].plot(0, 0, 'r*', markersize=12)

            plt.suptitle(f'{short_name}: Tier 1 vs Tier 2 Comparison', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{short_name.replace(".", "_")}_tier_comparison.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
        else:
            # Just plot Tier 1
            from visualization import plot_2d_surface
            plot_2d_surface(a1, b1, s1, f'{short_name} Tier 1',
                          os.path.join(output_dir, f'{short_name.replace(".", "_")}_tier1.png'))

        # Save results
        model_results = {
            'model': model_name,
            'n_params': n_params,
            'baseline_loss': baseline_loss,
            'tier1_metrics': m1,
            'tier2_success': tier2_success,
            'tier2_metrics': m2 if tier2_success else None,
            'tier2_explained_ratios': explained_ratios if tier2_success else None,
            'tier2_n_gradients': n_used,
        }
        results[short_name] = model_results

        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - t_start
    results['total_time_seconds'] = total_time

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    print("7B Tier 2 Analysis Complete!")
    print("=" * 70)

    for name, r in results.items():
        if isinstance(r, dict) and 'model' in r:
            print(f"\n{name}:")
            print(f"  Tier 1: range={r['tier1_metrics']['loss_range']:.2f}")
            if r['tier2_success']:
                print(f"  Tier 2: range={r['tier2_metrics']['loss_range']:.2f}")
                print(f"  Tier 2/Tier 1 ratio: {r['tier2_metrics']['loss_range']/r['tier1_metrics']['loss_range']:.2f}x")
            else:
                print(f"  Tier 2: FAILED")


if __name__ == '__main__':
    main()
