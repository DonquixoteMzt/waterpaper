"""
LLMScape Proof-of-Concept: Core Innovation Verification

This script verifies the three core technical contributions:
1. TADN (Transformer-Adapted Direction Normalization)
2. SHIDS (Direction selection: random vs gradient covariance PCA)
3. Basic 2D loss landscape visualization

Uses Qwen3-0.6B-Base as the test model on a subset of WikiText.
"""

import os
import sys
import json
import time
import copy
import gc
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# ============================================================
# 1. TADN: Transformer-Adapted Direction Normalization
# ============================================================

def get_normalization_units(model):
    """
    Partition model parameters into TADN normalization units.
    Each unit is a semantically meaningful group matching the transformer structure.

    Returns a dict: param_name -> list of (unit_name, slice_info) describing how
    to decompose that parameter into normalization units.
    """
    units = {}  # param_name -> list of (unit_id, dim, slice_indices)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        shape = param.shape

        if 'embed_tokens' in name or 'wte' in name:
            # Token embedding: normalize per row (per token)
            units[name] = [('row', 0)]  # normalize along dim 0 (each row is a unit)
        elif 'lm_head' in name:
            # LM head: normalize per column
            units[name] = [('col', 1)]  # normalize along dim 1
        elif any(k in name for k in ['q_proj', 'k_proj', 'v_proj']):
            # QKV projections: normalize per attention head
            units[name] = [('head', 'qkv')]
        elif 'o_proj' in name:
            # Output projection: normalize per head slice
            units[name] = [('head', 'o')]
        elif any(k in name for k in ['up_proj', 'gate_proj']):
            # FFN up/gate: normalize per column (per neuron)
            units[name] = [('col', 1)]
        elif 'down_proj' in name:
            # FFN down: normalize per row (per neuron)
            units[name] = [('row', 0)]
        elif any(k in name for k in ['layernorm', 'rmsnorm', 'norm', 'ln_']):
            # Normalization layers: entire vector as one unit
            units[name] = [('whole', None)]
        else:
            # Fallback: treat entire parameter as one unit
            units[name] = [('whole', None)]

    return units


def apply_tadn(direction, model, units, num_heads=None, head_dim=None, epsilon=1e-8):
    """
    Apply Transformer-Adapted Direction Normalization.

    For each normalization unit, scale the corresponding direction component
    so that ||d_i|| / ||theta_i|| is constant across units.

    Args:
        direction: dict of param_name -> perturbation tensor
        model: the model (for parameter norms)
        units: normalization unit partition from get_normalization_units()
        num_heads: number of attention heads
        head_dim: dimension per head
        epsilon: threshold for near-zero parameters
    """
    # Collect all non-degenerate unit norms to compute median
    all_norms = []

    normalized_direction = {}

    for name, param in model.named_parameters():
        if name not in direction:
            continue

        d = direction[name].clone().float()  # Work in float32
        p = param.data.float()  # Work in float32

        if name not in units:
            # Fallback: whole-parameter normalization
            p_norm = p.norm().item()
            d_norm = d.norm().item()
            if p_norm > epsilon and d_norm > epsilon:
                d = d * (p_norm / d_norm)
            normalized_direction[name] = d
            continue

        unit_type, unit_info = units[name][0]

        if unit_type == 'whole':
            p_norm = p.norm().item()
            d_norm = d.norm().item()
            if p_norm > epsilon and d_norm > epsilon:
                d = d * (p_norm / d_norm)
                all_norms.append(p_norm)

        elif unit_type == 'row':
            # Normalize each row independently
            dim = unit_info if unit_info is not None else 0
            p_norms = p.norm(dim=1, keepdim=True) if dim == 0 else p.norm(dim=0, keepdim=True)
            d_norms = d.norm(dim=1, keepdim=True) if dim == 0 else d.norm(dim=0, keepdim=True)

            # Avoid division by zero
            mask = (p_norms > epsilon) & (d_norms > epsilon)
            scale = torch.ones_like(p_norms)
            scale[mask] = p_norms[mask] / d_norms[mask]

            if dim == 0:
                d = d * scale
            else:
                d = d * scale
            all_norms.extend(p_norms[p_norms > epsilon].tolist())

        elif unit_type == 'col':
            # Normalize each column independently
            p_norms = p.norm(dim=0, keepdim=True)
            d_norms = d.norm(dim=0, keepdim=True)
            mask = (p_norms > epsilon) & (d_norms > epsilon)
            scale = torch.ones_like(p_norms)
            scale[mask] = p_norms[mask] / d_norms[mask]
            d = d * scale
            all_norms.extend(p_norms[p_norms > epsilon].tolist())

        elif unit_type == 'head':
            if num_heads is not None and head_dim is not None:
                if unit_info == 'qkv':
                    # Shape: [num_heads * head_dim, d_model] or [d_model, num_heads * head_dim]
                    # Reshape to [num_heads, head_dim, d_model] or similar
                    if p.dim() == 2:
                        # Try to identify the head dimension
                        if p.shape[0] == num_heads * head_dim:
                            p_r = p.view(num_heads, head_dim, -1)
                            d_r = d.view(num_heads, head_dim, -1)
                        elif p.shape[1] == num_heads * head_dim:
                            p_r = p.view(-1, num_heads, head_dim).permute(1, 0, 2)
                            d_r = d.view(-1, num_heads, head_dim).permute(1, 0, 2)
                        else:
                            # Fallback
                            p_norm = p.norm().item()
                            d_norm = d.norm().item()
                            if p_norm > epsilon and d_norm > epsilon:
                                d = d * (p_norm / d_norm)
                            normalized_direction[name] = d
                            continue

                        # Normalize per head
                        for h in range(num_heads):
                            p_norm = p_r[h].norm().item()
                            d_norm = d_r[h].norm().item()
                            if p_norm > epsilon and d_norm > epsilon:
                                d_r[h] = d_r[h] * (p_norm / d_norm)
                                all_norms.append(p_norm)

                        # Reshape back
                        if p.shape[0] == num_heads * head_dim:
                            d = d_r.view(p.shape)
                        else:
                            d = d_r.permute(1, 0, 2).reshape(p.shape)
                    else:
                        p_norm = p.norm().item()
                        d_norm = d.norm().item()
                        if p_norm > epsilon and d_norm > epsilon:
                            d = d * (p_norm / d_norm)

                elif unit_info == 'o':
                    if p.dim() == 2 and p.shape[1] == num_heads * head_dim:
                        # O projection: [d_model, num_heads*head_dim]
                        p_r = p.view(p.shape[0], num_heads, head_dim).permute(1, 0, 2)
                        d_r = d.view(d.shape[0], num_heads, head_dim).permute(1, 0, 2)
                        for h in range(num_heads):
                            p_norm = p_r[h].norm().item()
                            d_norm = d_r[h].norm().item()
                            if p_norm > epsilon and d_norm > epsilon:
                                d_r[h] = d_r[h] * (p_norm / d_norm)
                                all_norms.append(p_norm)
                        d = d_r.permute(1, 0, 2).reshape(p.shape)
                    elif p.dim() == 2 and p.shape[0] == num_heads * head_dim:
                        p_r = p.view(num_heads, head_dim, -1)
                        d_r = d.view(num_heads, head_dim, -1)
                        for h in range(num_heads):
                            p_norm = p_r[h].norm().item()
                            d_norm = d_r[h].norm().item()
                            if p_norm > epsilon and d_norm > epsilon:
                                d_r[h] = d_r[h] * (p_norm / d_norm)
                                all_norms.append(p_norm)
                        d = d_r.view(p.shape)
                    else:
                        p_norm = p.norm().item()
                        d_norm = d.norm().item()
                        if p_norm > epsilon and d_norm > epsilon:
                            d = d * (p_norm / d_norm)
            else:
                # No head info, fallback to whole
                p_norm = p.norm().item()
                d_norm = d.norm().item()
                if p_norm > epsilon and d_norm > epsilon:
                    d = d * (p_norm / d_norm)

        normalized_direction[name] = d.to(direction[name].dtype)

    return normalized_direction


def apply_layer_normalization(direction, model, epsilon=1e-8):
    """Baseline: normalize each parameter matrix as a whole (layer-wise normalization)."""
    normalized = {}
    for name, param in model.named_parameters():
        if name not in direction:
            continue
        d = direction[name].clone()
        p_norm = param.data.norm().item()
        d_norm = d.norm().item()
        if p_norm > epsilon and d_norm > epsilon:
            d = d * (p_norm / d_norm)
        normalized[name] = d
    return normalized


def apply_no_normalization(direction, model):
    """Baseline: no normalization (raw random direction)."""
    return {name: direction[name].clone() for name in direction}


# ============================================================
# 2. Direction Generation
# ============================================================

def generate_random_direction(model, seed=42):
    """Generate a random Gaussian direction vector."""
    torch.manual_seed(seed)
    direction = {}
    for name, param in model.named_parameters():
        direction[name] = torch.randn_like(param)
    return direction


def generate_gradient_covariance_pca_directions(model, dataloader, device, n_batches=50, k=2):
    """
    Tier 2: Gradient Covariance PCA Direction Selection.

    Collect per-batch gradients and find top-k principal components
    of the gradient covariance matrix via incremental SVD.
    """
    print(f"Computing gradient covariance PCA directions ({n_batches} batches)...")

    # Collect gradients as flattened vectors
    gradients = []
    model.eval()

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        # Flatten gradient
        grad_vec = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_vec.append(param.grad.detach().cpu().flatten())
            else:
                grad_vec.append(torch.zeros(param.numel()))
        grad_vec = torch.cat(grad_vec)
        gradients.append(grad_vec)

        if (i + 1) % 10 == 0:
            print(f"  Collected {i+1}/{n_batches} gradients")

    model.zero_grad()

    # Stack and compute PCA
    G = torch.stack(gradients).float()  # [n_batches, d], cast to float32
    G = G - G.mean(dim=0, keepdim=True)  # Center

    # Since n_batches << d, compute eigendecomp of G G^T (n_batches x n_batches)
    GGt = G @ G.t()  # [n_batches, n_batches]
    eigenvalues, eigenvectors = torch.linalg.eigh(GGt)

    # Sort by descending eigenvalue
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx[:k]]
    eigenvectors = eigenvectors[:, idx[:k]]

    # Project back to parameter space: principal components = G^T @ eigenvectors / sqrt(eigenvalues)
    pca_directions_flat = G.t() @ eigenvectors  # [d, k]
    for j in range(k):
        pca_directions_flat[:, j] = pca_directions_flat[:, j] / pca_directions_flat[:, j].norm()

    # Convert back to parameter dict
    directions = []
    for j in range(k):
        direction = {}
        offset = 0
        for name, param in model.named_parameters():
            numel = param.numel()
            direction[name] = pca_directions_flat[offset:offset+numel, j].reshape(param.shape).to(param.device)
            offset += numel
        directions.append(direction)

    # Report explained variance
    total_var = eigenvalues.sum().item()
    for j in range(k):
        print(f"  PC{j+1} explained variance ratio: {eigenvalues[j].item() / (eigenvalues.sum().item() + 1e-10):.4f}")

    return directions


# ============================================================
# 3. Loss Surface Evaluation
# ============================================================

@torch.no_grad()
def evaluate_loss(model, dataloader, device, max_batches=None):
    """Evaluate average loss over the dataloader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        n_tokens = attention_mask.sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def perturb_model(model, direction, alpha):
    """Add alpha * direction to model parameters in-place."""
    for name, param in model.named_parameters():
        if name in direction:
            param.data.add_(alpha * direction[name].to(param.dtype))


def evaluate_2d_surface(model, d1, d2, dataloader, device,
                        grid_range=(-1.0, 1.0), grid_size=21, max_batches=5):
    """
    Evaluate the 2D loss surface f(alpha, beta) = L(theta* + alpha*d1 + beta*d2).
    Saves original parameters and restores exactly after each point to avoid
    bfloat16 numerical drift from repeated add/subtract.
    """
    alphas = np.linspace(grid_range[0], grid_range[1], grid_size)
    betas = np.linspace(grid_range[0], grid_range[1], grid_size)
    surface = np.zeros((grid_size, grid_size))

    total_points = grid_size * grid_size
    print(f"Evaluating {total_points} grid points ({grid_size}x{grid_size})...")

    # Save original parameters (exact copy)
    original_params = {name: param.data.clone() for name, param in model.named_parameters()}

    t0 = time.time()
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Restore to original first (exact restoration)
            for name, param in model.named_parameters():
                param.data.copy_(original_params[name])

            # Apply perturbation from original
            if alpha != 0.0:
                perturb_model(model, d1, alpha)
            if beta != 0.0:
                perturb_model(model, d2, beta)

            # Evaluate
            loss = evaluate_loss(model, dataloader, device, max_batches=max_batches)
            surface[j, i] = loss  # Note: j=row (beta), i=col (alpha)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (grid_size - i - 1)
        print(f"  Row {i+1}/{grid_size} done. Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

    # Restore original parameters
    for name, param in model.named_parameters():
        param.data.copy_(original_params[name])
    del original_params
    torch.cuda.empty_cache()

    return alphas, betas, surface


# ============================================================
# 4. Visualization
# ============================================================

def plot_2d_surface(alphas, betas, surface, title, filename, vmin=None, vmax=None):
    """Plot a 2D loss surface as a contour plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    A, B = np.meshgrid(alphas, betas)

    if vmin is None:
        vmin = surface.min()
    if vmax is None:
        vmax = min(surface.max(), surface.min() + 3 * (np.median(surface) - surface.min() + 0.1))

    # Contour plot
    levels = np.linspace(vmin, vmax, 30)
    cs = axes[0].contourf(A, B, surface, levels=levels, cmap='viridis')
    axes[0].contour(A, B, surface, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=axes[0], label='Loss')
    axes[0].set_xlabel(r'$\alpha$ (direction 1)')
    axes[0].set_ylabel(r'$\beta$ (direction 2)')
    axes[0].set_title(f'{title}\n(Contour)')
    axes[0].plot(0, 0, 'r*', markersize=15, label=r'$\theta^*$')
    axes[0].legend()

    # 3D surface plot
    ax3d = fig.add_subplot(122, projection='3d')
    axes[1].remove()
    surf = ax3d.plot_surface(A, B, surface, cmap='viridis', alpha=0.8,
                              vmin=vmin, vmax=vmax)
    ax3d.set_xlabel(r'$\alpha$')
    ax3d.set_ylabel(r'$\beta$')
    ax3d.set_zlabel('Loss')
    ax3d.set_title(f'{title}\n(3D Surface)')
    ax3d.view_init(elev=30, azim=225)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_comparison(results, filename):
    """Plot comparison of different normalization methods side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1:
        axes = [axes]

    # Find global vmin/vmax for consistent coloring
    all_surfaces = [r['surface'] for r in results]
    global_min = min(s.min() for s in all_surfaces)
    global_max = min(max(s.max() for s in all_surfaces),
                     global_min + 3 * (np.median(np.concatenate([s.flatten() for s in all_surfaces])) - global_min + 0.1))

    levels = np.linspace(global_min, global_max, 30)

    for idx, (ax, r) in enumerate(zip(axes, results)):
        A, B = np.meshgrid(r['alphas'], r['betas'])
        cs = ax.contourf(A, B, r['surface'], levels=levels, cmap='viridis')
        ax.contour(A, B, r['surface'], levels=levels, colors='white', linewidths=0.3, alpha=0.5)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\beta$')
        ax.set_title(r['title'])
        ax.plot(0, 0, 'r*', markersize=15)
        ax.set_aspect('equal')

    plt.colorbar(cs, ax=axes[-1], label='Loss')
    plt.suptitle('Normalization Method Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# ============================================================
# 5. Main PoC Experiment
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B-Base')
    parser.add_argument('--grid_size', type=int, default=21)
    parser.add_argument('--grid_range', type=float, default=1.0)
    parser.add_argument('--max_eval_batches', type=int, default=5)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_grad_batches', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_log = {
        'model': args.model_name,
        'grid_size': args.grid_size,
        'grid_range': args.grid_range,
        'experiments': []
    }

    # ---- Load model and tokenizer ----
    print(f"Loading model: {args.model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"  # For HVP compatibility
    ).to(device)
    model.eval()

    # Get model config for TADN
    config = model.config
    num_heads = getattr(config, 'num_attention_heads', None)
    head_dim = getattr(config, 'head_dim', None)
    if head_dim is None and num_heads is not None:
        hidden_size = getattr(config, 'hidden_size', None)
        if hidden_size is not None:
            head_dim = hidden_size // num_heads

    print(f"  num_heads={num_heads}, head_dim={head_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Prepare data ----
    print("Loading WikiText-2 dataset...")
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Tokenize
    texts = [t for t in dataset['text'] if len(t.strip()) > 50]
    all_tokens = tokenizer('\n'.join(texts[:200]), return_tensors='pt', truncation=False)['input_ids'][0]

    # Create fixed-length chunks
    chunks = []
    for i in range(0, len(all_tokens) - args.seq_len, args.seq_len):
        chunks.append(all_tokens[i:i+args.seq_len])

    print(f"  Created {len(chunks)} chunks of length {args.seq_len}")

    # Create dataloader
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, chunks):
            self.chunks = chunks
        def __len__(self):
            return len(self.chunks)
        def __getitem__(self, idx):
            return {'input_ids': self.chunks[idx], 'attention_mask': torch.ones(len(self.chunks[idx]))}

    eval_dataset = SimpleDataset(chunks[:50])  # Use first 50 chunks for evaluation
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    grad_dataset = SimpleDataset(chunks[:args.n_grad_batches * args.batch_size])
    grad_dataloader = torch.utils.data.DataLoader(grad_dataset, batch_size=args.batch_size, shuffle=False)

    # ---- Baseline loss ----
    print("\nComputing baseline loss...")
    baseline_loss = evaluate_loss(model, eval_dataloader, device, max_batches=args.max_eval_batches)
    print(f"  Baseline loss: {baseline_loss:.4f}")
    results_log['baseline_loss'] = baseline_loss

    # ---- Get normalization units ----
    units = get_normalization_units(model)

    # ============================================================
    # Experiment 1: Compare normalization methods (TADN vs Layer vs None)
    # ============================================================
    print("\n" + "="*60)
    print("Experiment 1: Normalization Method Comparison")
    print("="*60)

    # Generate base random directions
    raw_d1 = generate_random_direction(model, seed=42)
    raw_d2 = generate_random_direction(model, seed=123)

    # Orthogonalize d2 w.r.t. d1 (in flattened space)
    d1_flat = torch.cat([raw_d1[n].flatten() for n in raw_d1])
    d2_flat = torch.cat([raw_d2[n].flatten() for n in raw_d2])
    d2_flat = d2_flat - (d2_flat @ d1_flat) / (d1_flat @ d1_flat) * d1_flat
    # Write back
    offset = 0
    for name in raw_d2:
        numel = raw_d2[name].numel()
        raw_d2[name] = d2_flat[offset:offset+numel].reshape(raw_d2[name].shape)
        offset += numel
    del d1_flat, d2_flat

    normalization_methods = [
        ('No Normalization', apply_no_normalization),
        ('Layer Normalization', apply_layer_normalization),
        ('TADN', lambda d, m: apply_tadn(d, m, units, num_heads, head_dim)),
    ]

    comparison_results = []

    for norm_name, norm_fn in normalization_methods:
        print(f"\n--- {norm_name} ---")

        d1 = norm_fn(raw_d1, model)
        d2 = norm_fn(raw_d2, model)

        # Move directions to device
        d1 = {k: v.to(device) for k, v in d1.items()}
        d2 = {k: v.to(device) for k, v in d2.items()}

        alphas, betas, surface = evaluate_2d_surface(
            model, d1, d2, eval_dataloader, device,
            grid_range=(-args.grid_range, args.grid_range),
            grid_size=args.grid_size,
            max_batches=args.max_eval_batches
        )

        # Plot individual
        plot_2d_surface(alphas, betas, surface,
                       f'{norm_name}',
                       os.path.join(output_dir, f'surface_{norm_name.lower().replace(" ", "_")}.png'))

        comparison_results.append({
            'title': norm_name,
            'alphas': alphas,
            'betas': betas,
            'surface': surface
        })

        # Compute metrics
        center_loss = surface[args.grid_size//2, args.grid_size//2]
        loss_range = surface.max() - surface.min()

        # Basin width (where loss < center + 10% of range)
        threshold = center_loss + 0.1 * loss_range
        basin_mask = surface < threshold
        basin_fraction = basin_mask.sum() / basin_mask.size

        # Roughness
        from scipy.ndimage import uniform_filter
        smoothed = uniform_filter(surface, size=3)
        roughness = np.std(surface - smoothed)

        # Asymmetry
        mid = args.grid_size // 2
        row_asym = np.mean(np.abs(surface[mid, :] - surface[mid, ::-1]))
        col_asym = np.mean(np.abs(surface[:, mid] - surface[::-1, mid]))
        asymmetry = (row_asym + col_asym) / 2

        metrics = {
            'normalization': norm_name,
            'center_loss': float(center_loss),
            'loss_min': float(surface.min()),
            'loss_max': float(surface.max()),
            'loss_range': float(loss_range),
            'basin_fraction': float(basin_fraction),
            'roughness': float(roughness),
            'asymmetry': float(asymmetry)
        }
        results_log['experiments'].append(metrics)
        print(f"  Center loss: {center_loss:.4f}")
        print(f"  Loss range: {loss_range:.4f}")
        print(f"  Basin fraction (10%): {basin_fraction:.4f}")
        print(f"  Roughness: {roughness:.6f}")
        print(f"  Asymmetry: {asymmetry:.6f}")

        # Clean up
        del d1, d2
        torch.cuda.empty_cache()

    # Comparison plot
    plot_comparison(comparison_results, os.path.join(output_dir, 'comparison_normalization.png'))

    # ============================================================
    # Experiment 2: Direction Selection (Random vs Gradient Covariance PCA)
    # ============================================================
    print("\n" + "="*60)
    print("Experiment 2: Direction Selection Comparison")
    print("="*60)

    # Tier 2: Gradient Covariance PCA
    pca_directions = generate_gradient_covariance_pca_directions(
        model, grad_dataloader, device, n_batches=args.n_grad_batches, k=2
    )

    # Apply TADN to PCA directions
    pca_d1 = apply_tadn(pca_directions[0], model, units, num_heads, head_dim)
    pca_d2 = apply_tadn(pca_directions[1], model, units, num_heads, head_dim)
    pca_d1 = {k: v.to(device) for k, v in pca_d1.items()}
    pca_d2 = {k: v.to(device) for k, v in pca_d2.items()}

    print("\n--- Gradient Covariance PCA + TADN ---")
    alphas_pca, betas_pca, surface_pca = evaluate_2d_surface(
        model, pca_d1, pca_d2, eval_dataloader, device,
        grid_range=(-args.grid_range, args.grid_range),
        grid_size=args.grid_size,
        max_batches=args.max_eval_batches
    )

    plot_2d_surface(alphas_pca, betas_pca, surface_pca,
                   'Gradient Covariance PCA + TADN',
                   os.path.join(output_dir, 'surface_grad_pca_tadn.png'))

    # Metrics for PCA
    center_pca = surface_pca[args.grid_size//2, args.grid_size//2]
    range_pca = surface_pca.max() - surface_pca.min()

    smoothed_pca = uniform_filter(surface_pca, size=3)
    roughness_pca = np.std(surface_pca - smoothed_pca)

    pca_metrics = {
        'direction': 'Gradient Covariance PCA + TADN',
        'center_loss': float(center_pca),
        'loss_min': float(surface_pca.min()),
        'loss_max': float(surface_pca.max()),
        'loss_range': float(range_pca),
        'roughness': float(roughness_pca),
    }
    results_log['experiments'].append(pca_metrics)
    print(f"  Center loss: {center_pca:.4f}")
    print(f"  Loss range: {range_pca:.4f} (Random+TADN: {comparison_results[2]['surface'].max()-comparison_results[2]['surface'].min():.4f})")
    print(f"  Roughness: {roughness_pca:.6f}")

    # Side-by-side comparison: Random+TADN vs PCA+TADN
    direction_comparison = [
        {'title': 'Random + TADN', 'alphas': comparison_results[2]['alphas'],
         'betas': comparison_results[2]['betas'], 'surface': comparison_results[2]['surface']},
        {'title': 'Grad Cov PCA + TADN', 'alphas': alphas_pca,
         'betas': betas_pca, 'surface': surface_pca}
    ]
    plot_comparison(direction_comparison, os.path.join(output_dir, 'comparison_directions.png'))

    # ============================================================
    # Experiment 3: 1D Cross-Section Analysis
    # ============================================================
    print("\n" + "="*60)
    print("Experiment 3: 1D Cross-Section Analysis")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mid = args.grid_size // 2

    # Plot 1D slices for each normalization
    for r in comparison_results:
        axes[0].plot(r['alphas'], r['surface'][mid, :], label=r['title'], linewidth=2)
    axes[0].plot(alphas_pca, surface_pca[mid, :], label='Grad PCA + TADN', linewidth=2, linestyle='--')
    axes[0].set_xlabel(r'$\alpha$ (direction 1)')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('1D Cross-Section along Direction 1 (β=0)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for r in comparison_results:
        axes[1].plot(r['betas'], r['surface'][:, mid], label=r['title'], linewidth=2)
    axes[1].plot(betas_pca, surface_pca[:, mid], label='Grad PCA + TADN', linewidth=2, linestyle='--')
    axes[1].set_xlabel(r'$\beta$ (direction 2)')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('1D Cross-Section along Direction 2 (α=0)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_sections.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'cross_sections.png')}")

    # ============================================================
    # Save Results
    # ============================================================
    results_file = os.path.join(output_dir, 'poc_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_log, f, indent=2)
    print(f"\nResults saved to {results_file}")

    print("\n" + "="*60)
    print("Proof-of-Concept Complete!")
    print("="*60)

    return results_log


if __name__ == '__main__':
    main()
