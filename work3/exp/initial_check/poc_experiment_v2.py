"""
LLMScape Enhanced Proof-of-Concept (v2)
========================================
Addresses novelty feedback by demonstrating:
1. TADN Scale-Invariance Test — advantage over layer normalization
2. Gradient PCA Convergence Analysis — adaptive sample-size selection
3. Tier 3 Hessian Eigenvector Direction Selection — full pipeline
4. Projection Faithfulness Index (PFI) — novel metric for all tiers
5. Full 3-Tier Landscape Comparison

Model: Qwen3-0.6B-Base
Dataset: WikiText-2 (test)
"""

import os
import sys
import json
import time
import copy
import gc
import math
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
    """Partition model parameters into TADN normalization units."""
    units = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'embed_tokens' in name or 'wte' in name:
            units[name] = [('row', 0)]
        elif 'lm_head' in name:
            units[name] = [('col', 1)]
        elif any(k in name for k in ['q_proj', 'k_proj', 'v_proj']):
            units[name] = [('head', 'qkv')]
        elif 'o_proj' in name:
            units[name] = [('head', 'o')]
        elif any(k in name for k in ['up_proj', 'gate_proj']):
            # up_proj.weight: [intermediate, hidden] — each row = one neuron
            units[name] = [('row', 0)]
        elif 'down_proj' in name:
            # down_proj.weight: [hidden, intermediate] — each column = one neuron
            units[name] = [('col', 1)]
        elif any(k in name for k in ['layernorm', 'rmsnorm', 'norm', 'ln_']):
            units[name] = [('whole', None)]
        else:
            units[name] = [('whole', None)]
    return units


def apply_tadn(direction, model, units, num_heads=None, head_dim=None, epsilon=1e-8):
    """Apply Transformer-Adapted Direction Normalization."""
    normalized_direction = {}
    for name, param in model.named_parameters():
        if name not in direction:
            continue
        d = direction[name].clone().float()
        p = param.data.float()

        if name not in units:
            p_norm = p.norm().item()
            d_norm = d.norm().item()
            if p_norm > epsilon and d_norm > epsilon:
                d = d * (p_norm / d_norm)
            normalized_direction[name] = d.to(direction[name].dtype)
            continue

        unit_type, unit_info = units[name][0]

        if unit_type == 'whole':
            p_norm = p.norm().item()
            d_norm = d.norm().item()
            if p_norm > epsilon and d_norm > epsilon:
                d = d * (p_norm / d_norm)

        elif unit_type == 'row':
            p_norms = p.norm(dim=1, keepdim=True)
            d_norms = d.norm(dim=1, keepdim=True)
            mask = (p_norms > epsilon) & (d_norms > epsilon)
            scale = torch.ones_like(p_norms)
            scale[mask] = p_norms[mask] / d_norms[mask]
            d = d * scale

        elif unit_type == 'col':
            p_norms = p.norm(dim=0, keepdim=True)
            d_norms = d.norm(dim=0, keepdim=True)
            mask = (p_norms > epsilon) & (d_norms > epsilon)
            scale = torch.ones_like(p_norms)
            scale[mask] = p_norms[mask] / d_norms[mask]
            d = d * scale

        elif unit_type == 'head':
            if num_heads is not None and head_dim is not None:
                if unit_info == 'qkv':
                    if p.dim() == 2 and p.shape[0] == num_heads * head_dim:
                        p_r = p.view(num_heads, head_dim, -1)
                        d_r = d.view(num_heads, head_dim, -1)
                        for h in range(num_heads):
                            pn = p_r[h].norm().item()
                            dn = d_r[h].norm().item()
                            if pn > epsilon and dn > epsilon:
                                d_r[h] = d_r[h] * (pn / dn)
                        d = d_r.view(p.shape)
                    else:
                        p_norm = p.norm().item()
                        d_norm = d.norm().item()
                        if p_norm > epsilon and d_norm > epsilon:
                            d = d * (p_norm / d_norm)
                elif unit_info == 'o':
                    if p.dim() == 2 and p.shape[0] == num_heads * head_dim:
                        p_r = p.view(num_heads, head_dim, -1)
                        d_r = d.view(num_heads, head_dim, -1)
                        for h in range(num_heads):
                            pn = p_r[h].norm().item()
                            dn = d_r[h].norm().item()
                            if pn > epsilon and dn > epsilon:
                                d_r[h] = d_r[h] * (pn / dn)
                        d = d_r.view(p.shape)
                    else:
                        p_norm = p.norm().item()
                        d_norm = d.norm().item()
                        if p_norm > epsilon and d_norm > epsilon:
                            d = d * (p_norm / d_norm)
            else:
                p_norm = p.norm().item()
                d_norm = d.norm().item()
                if p_norm > epsilon and d_norm > epsilon:
                    d = d * (p_norm / d_norm)

        normalized_direction[name] = d.to(direction[name].dtype)
    return normalized_direction


def apply_layer_normalization(direction, model, epsilon=1e-8):
    """Baseline: normalize each parameter matrix as a whole."""
    normalized = {}
    for name, param in model.named_parameters():
        if name not in direction:
            continue
        d = direction[name].clone().float()
        p_norm = param.data.float().norm().item()
        d_norm = d.norm().item()
        if p_norm > epsilon and d_norm > epsilon:
            d = d * (p_norm / d_norm)
        normalized[name] = d.to(direction[name].dtype)
    return normalized


# ============================================================
# 2. Direction Generation
# ============================================================

def generate_random_direction(model, seed=42):
    """Generate a random Gaussian direction vector."""
    torch.manual_seed(seed)
    direction = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            direction[name] = torch.randn_like(param)
    return direction


def orthogonalize_directions(d1, d2):
    """Orthogonalize d2 w.r.t. d1 in flattened space."""
    d1_flat = torch.cat([d1[n].flatten().float() for n in d1])
    d2_flat = torch.cat([d2[n].flatten().float() for n in d2])
    d2_flat = d2_flat - (d2_flat @ d1_flat) / (d1_flat @ d1_flat + 1e-10) * d1_flat
    offset = 0
    for name in d2:
        numel = d2[name].numel()
        d2[name] = d2_flat[offset:offset + numel].reshape(d2[name].shape).to(d2[name].dtype)
        offset += numel
    return d2


# ============================================================
# 3. Loss Evaluation
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
        n_tokens = attention_mask.sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def evaluate_1d_curve(model, direction, dataloader, device,
                      alpha_range=(-1.0, 1.0), n_points=21, max_batches=5):
    """Evaluate 1D loss curve f(alpha) = L(theta* + alpha*d)."""
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    losses = np.zeros(n_points)
    original_params = {name: param.data.clone() for name, param in model.named_parameters()}

    for i, alpha in enumerate(alphas):
        for name, param in model.named_parameters():
            param.data.copy_(original_params[name])
        if alpha != 0.0:
            for name, param in model.named_parameters():
                if name in direction:
                    param.data.add_(alpha * direction[name].to(param.dtype).to(param.device))
        losses[i] = evaluate_loss(model, dataloader, device, max_batches=max_batches)

    for name, param in model.named_parameters():
        param.data.copy_(original_params[name])
    del original_params
    return alphas, losses


def evaluate_2d_surface(model, d1, d2, dataloader, device,
                        grid_range=(-1.0, 1.0), grid_size=21, max_batches=5):
    """Evaluate 2D loss surface with exact parameter restoration."""
    alphas = np.linspace(grid_range[0], grid_range[1], grid_size)
    betas = np.linspace(grid_range[0], grid_range[1], grid_size)
    surface = np.zeros((grid_size, grid_size))
    original_params = {name: param.data.clone() for name, param in model.named_parameters()}

    t0 = time.time()
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            for name, param in model.named_parameters():
                param.data.copy_(original_params[name])
            if alpha != 0.0:
                for name, param in model.named_parameters():
                    if name in d1:
                        param.data.add_(alpha * d1[name].to(param.dtype).to(param.device))
            if beta != 0.0:
                for name, param in model.named_parameters():
                    if name in d2:
                        param.data.add_(beta * d2[name].to(param.dtype).to(param.device))
            surface[j, i] = evaluate_loss(model, dataloader, device, max_batches=max_batches)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (grid_size - i - 1)
        if (i + 1) % 5 == 0:
            print(f"  Row {i+1}/{grid_size} done. Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

    for name, param in model.named_parameters():
        param.data.copy_(original_params[name])
    del original_params
    torch.cuda.empty_cache()
    return alphas, betas, surface


# ============================================================
# 4. Rescaled Model for TADN Invariance Test
# ============================================================

def create_rescaled_model(model):
    """
    Create a functionally equivalent model by non-uniformly scaling FFN neurons.
    For SwiGLU FFN: scale W_up rows by c_j and W_down columns by 1/c_j.
    This preserves FFN(x) = W_down * (swish(W_gate*x) * W_up*x) exactly.

    Uses powers of 2 as scale factors for exact bfloat16 representation.
    """
    model_rescaled = copy.deepcopy(model)

    for name, param in model_rescaled.named_parameters():
        if 'up_proj.weight' in name:
            n_neurons = param.shape[0]
            # Powers of 2: exact in bfloat16, so c * (1/c) = 1.0 exactly
            scales = torch.ones(n_neurons, device=param.device, dtype=param.dtype)
            scales[:n_neurons // 4] = 8.0       # very large
            scales[n_neurons // 4:n_neurons // 2] = 4.0
            scales[n_neurons // 2:3 * n_neurons // 4] = 0.25
            scales[3 * n_neurons // 4:] = 0.125  # very small
            param.data *= scales.unsqueeze(1)

        elif 'gate_proj.weight' in name:
            # Must also scale gate_proj to maintain SwiGLU invariance:
            # FFN(x) = W_down * (σ(W_gate*x) ⊙ W_up*x)
            # If we scale neuron j output by c_j in both gate and up,
            # then σ(c_j * gate_j) * c_j * up_j ≠ c_j * σ(gate_j) * up_j
            # So we should NOT scale gate_proj.
            # The invariance only holds for W_up and W_down.
            pass

        elif 'down_proj.weight' in name:
            n_neurons = param.shape[1]
            inv_scales = torch.ones(n_neurons, device=param.device, dtype=param.dtype)
            inv_scales[:n_neurons // 4] = 0.125      # 1/8
            inv_scales[n_neurons // 4:n_neurons // 2] = 0.25  # 1/4
            inv_scales[n_neurons // 2:3 * n_neurons // 4] = 4.0   # 1/0.25
            inv_scales[3 * n_neurons // 4:] = 8.0     # 1/0.125
            param.data *= inv_scales.unsqueeze(0)

    return model_rescaled


def verify_model_equivalence(model1, model2, dataloader, device, max_batches=3):
    """Verify two models produce the same loss."""
    loss1 = evaluate_loss(model1, dataloader, device, max_batches=max_batches)
    loss2 = evaluate_loss(model2, dataloader, device, max_batches=max_batches)
    return loss1, loss2, abs(loss1 - loss2)


# ============================================================
# 5. Hessian-Vector Product and Power Iteration
# ============================================================

def compute_hvp(model_fp32, dataloader, device, v_dict, max_batches=3):
    """
    Compute Hessian-vector product Hv using Pearlmutter trick.
    Uses float32 model for numerical stability.
    Memory-efficient: processes one batch at a time.
    """
    model_fp32.train()
    for p in model_fp32.parameters():
        p.requires_grad_(True)
    model_fp32.zero_grad()

    param_names = [n for n, _ in model_fp32.named_parameters()]
    params = list(model_fp32.parameters())

    # Accumulate HVP across batches for numerical stability
    hvp_accum = {n: torch.zeros_like(p.data) for n, p in model_fp32.named_parameters()}
    total_tokens_all = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
        n_tokens = attention_mask.sum().item()

        # Forward
        outputs = model_fp32(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # First backward with create_graph
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)

        # Compute g·v
        gv = sum((g * v_dict[name].to(device).float()).sum()
                 for g, name in zip(grads, param_names)
                 if g is not None and name in v_dict)

        # Second backward: Hv for this batch
        hvp_tensors = torch.autograd.grad(gv, params, allow_unused=True)

        for name, hvp in zip(param_names, hvp_tensors):
            if hvp is not None:
                hvp_accum[name] += hvp.detach() * n_tokens

        total_tokens_all += n_tokens

        # Free memory between batches
        del outputs, loss, grads, gv, hvp_tensors
        model_fp32.zero_grad()
        torch.cuda.empty_cache()

    # Average over total tokens
    hvp_dict = {n: hvp_accum[n] / total_tokens_all for n in hvp_accum}

    model_fp32.zero_grad()
    model_fp32.eval()
    return hvp_dict


def power_iteration_hessian(model_fp32, dataloader, device,
                            n_iter=30, n_vectors=2, max_batches=3, verbose=True):
    """Compute top-k Hessian eigenvectors via power iteration."""
    vectors = []
    eigenvalues = []

    for j in range(n_vectors):
        if verbose:
            print(f"  Computing eigenvector {j+1}/{n_vectors}...")

        # Initialize
        torch.manual_seed(1000 + j)
        v = {n: torch.randn_like(p).float()
             for n, p in model_fp32.named_parameters() if p.requires_grad}
        v_norm = math.sqrt(sum((v[n] ** 2).sum().item() for n in v))
        for n in v:
            v[n] /= v_norm

        lam = 0.0
        for t in range(n_iter):
            hv = compute_hvp(model_fp32, dataloader, device, v, max_batches)

            # Deflate
            for i in range(j):
                proj = sum((hv[n] * vectors[i][n].to(device)).sum().item() for n in hv)
                for n in hv:
                    hv[n] -= proj * vectors[i][n].to(device)

            # Eigenvalue
            lam = sum((v[n].to(device) * hv[n]).sum().item() for n in hv)

            # Normalize
            hv_norm = math.sqrt(sum((hv[n] ** 2).sum().item() for n in hv))
            if hv_norm < 1e-10:
                break
            for n in hv:
                hv[n] /= hv_norm

            # Convergence check
            cos_sim = abs(sum((v[n].to(device) * hv[n]).sum().item() for n in hv))
            if verbose and (t + 1) % 10 == 0:
                print(f"    Iter {t+1}: lambda={lam:.4f}, cos_sim={cos_sim:.6f}")
            if cos_sim > 0.9999 and t > 5:
                if verbose:
                    print(f"    Converged at iter {t+1}")
                break

            v = {n: hv[n].cpu() for n in hv}

        vectors.append({n: hv[n].cpu() for n in hv})
        eigenvalues.append(lam)
        if verbose:
            print(f"  lambda_{j+1} = {eigenvalues[j]:.6f}")

    return vectors, eigenvalues


# ============================================================
# 6. Gradient Covariance PCA with Convergence Analysis
# ============================================================

def gradient_pca_with_convergence(model, dataloader, device,
                                  n_max=100, checkpoints=None, k=2):
    """
    Compute gradient covariance PCA with convergence analysis.
    Returns PCA directions at each checkpoint and subspace angles.
    """
    if checkpoints is None:
        checkpoints = [10, 20, 50, n_max]
    checkpoints = [c for c in checkpoints if c <= n_max]

    print(f"Computing gradient PCA convergence (N_max={n_max})...")
    model.eval()

    # Collect gradients and build Gram matrix incrementally
    grad_flat_list = []  # Store on CPU in float16
    gram_matrix = np.zeros((n_max, n_max))

    param_names = [n for n, p in model.named_parameters() if p.requires_grad]

    for i, batch in enumerate(dataloader):
        if i >= n_max:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        # Flatten gradient
        g_flat = torch.cat([p.grad.detach().flatten().float()
                            for n, p in model.named_parameters()
                            if p.requires_grad and p.grad is not None])

        # Update Gram matrix
        g_cpu = g_flat.cpu()
        for j in range(len(grad_flat_list)):
            dot = torch.dot(g_cpu, grad_flat_list[j]).item()
            gram_matrix[i, j] = dot
            gram_matrix[j, i] = dot
        gram_matrix[i, i] = torch.dot(g_cpu, g_cpu).item()

        grad_flat_list.append(g_cpu)
        del g_flat
        torch.cuda.empty_cache()

        if (i + 1) % 20 == 0:
            print(f"  Collected {i+1}/{n_max} gradients")

    model.zero_grad()

    # Compute PCA at each checkpoint
    pca_results = {}
    prev_subspace = None

    for N in checkpoints:
        if N > len(grad_flat_list):
            break

        G_N = gram_matrix[:N, :N]

        # Center the Gram matrix
        ones = np.ones((N, N)) / N
        G_centered = G_N - ones @ G_N - G_N @ ones + ones @ G_N @ ones

        eigenvalues, eigenvectors = np.linalg.eigh(G_centered)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx[:k]]
        eigenvectors = eigenvectors[:, idx[:k]]

        # Explained variance
        all_eigs = np.sort(np.linalg.eigvalsh(G_centered))[::-1]
        total_var = max(float(np.sum(np.maximum(all_eigs, 0))), 1e-10)
        explained_ratios = np.maximum(eigenvalues, 0) / total_var

        # Reconstruct d-dimensional directions
        directions = []
        for j in range(k):
            d_flat = torch.zeros_like(grad_flat_list[0])
            for t in range(N):
                d_flat += eigenvectors[t, j] * grad_flat_list[t]
            d_norm = d_flat.norm()
            if d_norm > 1e-10:
                d_flat /= d_norm
            directions.append(d_flat)

        # Compute subspace angle with previous checkpoint
        angle = None
        if prev_subspace is not None:
            # Principal angle between 2D subspaces
            U = torch.stack(directions)  # k x d
            V = torch.stack(prev_subspace)  # k x d
            M = U @ V.T  # k x k
            svd_vals = torch.linalg.svdvals(M.float())
            cos_angle = svd_vals.min().clamp(-1, 1).item()
            angle = math.degrees(math.acos(cos_angle))

        prev_subspace = directions

        pca_results[N] = {
            'eigenvalues': eigenvalues.tolist(),
            'explained_ratios': explained_ratios.tolist(),
            'subspace_angle_from_prev': angle,
            'directions_flat': directions,  # For later use
        }

        angle_str = f"{angle:.2f}deg" if angle is not None else "N/A"
        print(f"  N={N}: ev_ratio=[{explained_ratios[0]:.3f}, {explained_ratios[1]:.3f}], "
              f"angle_from_prev={angle_str}")

    # Convert flat directions to param dicts for the largest checkpoint
    final_N = max(c for c in checkpoints if c in pca_results)
    final_directions = []
    for j in range(k):
        d_flat = pca_results[final_N]['directions_flat'][j]
        direction = {}
        offset = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                numel = param.numel()
                direction[name] = d_flat[offset:offset + numel].reshape(param.shape).to(param.device)
                offset += numel
        final_directions.append(direction)

    return pca_results, final_directions


# ============================================================
# 7. Projection Faithfulness Index (PFI)
# ============================================================

def compute_hutchinson_tr_h2(model_fp32, dataloader, device, n_hutchinson=10, max_batches=3, verbose=True):
    """Estimate tr(H^2) via Hutchinson estimator. Model-level property, compute once."""
    tr_h2_estimates = []
    for k_idx in range(n_hutchinson):
        torch.manual_seed(2000 + k_idx)
        v = {n: torch.randn_like(p).float()
             for n, p in model_fp32.named_parameters() if p.requires_grad}
        hv = compute_hvp(model_fp32, dataloader, device, v, max_batches)
        est = sum((hv[n] ** 2).sum().item() for n in hv)
        tr_h2_estimates.append(est)
        del v, hv
        torch.cuda.empty_cache()
        if verbose:
            print(f"    Hutchinson {k_idx+1}/{n_hutchinson}: tr(H^2)~={np.mean(tr_h2_estimates):.4e}")

    tr_h2 = np.mean(tr_h2_estimates)
    tr_h2_std = np.std(tr_h2_estimates) / math.sqrt(n_hutchinson)
    if verbose:
        print(f"    Final tr(H^2) = {tr_h2:.4e} +/- {tr_h2_std:.4e}")
    return tr_h2, tr_h2_std


def compute_pfi(model_fp32, dataloader, device, d1, d2,
                lambda_max=None, tr_h2=None, tr_h2_std=None,
                max_batches=3, verbose=True):
    """
    Compute the Projection Faithfulness Index (PFI).

    PFI-S (Spectral Coverage) = (||Hd1||^2 + ||Hd2||^2) / tr(H^2)
    PFI-C (Curvature Capture) = max(|d_i^T H d_i|) / |lambda_max|

    tr_h2 should be precomputed via compute_hutchinson_tr_h2 (model property, shared across tiers).
    """
    if verbose:
        print("  Computing PFI...")

    # Normalize d1, d2 to unit vectors
    d1_norm = math.sqrt(sum((d1[n].float() ** 2).sum().item() for n in d1))
    d2_norm = math.sqrt(sum((d2[n].float() ** 2).sum().item() for n in d2))
    d1_unit = {n: d1[n].float() / d1_norm for n in d1}
    d2_unit = {n: d2[n].float() / d2_norm for n in d2}

    # Compute Hd1 and Hd2
    hd1 = compute_hvp(model_fp32, dataloader, device, d1_unit, max_batches)
    hd2 = compute_hvp(model_fp32, dataloader, device, d2_unit, max_batches)

    # ||Hd||^2 and d^T H d
    hd1_sq = sum((hd1[n] ** 2).sum().item() for n in hd1)
    hd2_sq = sum((hd2[n] ** 2).sum().item() for n in hd2)
    d1Hd1 = sum((d1_unit[n].to(device) * hd1[n]).sum().item() for n in hd1)
    d2Hd2 = sum((d2_unit[n].to(device) * hd2[n]).sum().item() for n in hd2)

    del hd1, hd2, d1_unit, d2_unit
    torch.cuda.empty_cache()

    # PFI-S
    pfi_s = (hd1_sq + hd2_sq) / (tr_h2 + 1e-10) if tr_h2 is not None else None

    # PFI-C (if lambda_max provided)
    pfi_c = None
    if lambda_max is not None and abs(lambda_max) > 1e-10:
        pfi_c = max(abs(d1Hd1), abs(d2Hd2)) / abs(lambda_max)

    results = {
        'PFI_S': pfi_s,
        'PFI_C': pfi_c,
        'Hd1_sq': hd1_sq,
        'Hd2_sq': hd2_sq,
        'd1_curvature': d1Hd1,
        'd2_curvature': d2Hd2,
        'tr_H2': tr_h2,
        'tr_H2_std': tr_h2_std,
    }

    if verbose:
        if pfi_s is not None:
            print(f"    PFI-S = {pfi_s:.6f}")
        if pfi_c is not None:
            print(f"    PFI-C = {pfi_c:.6f}")
        print(f"    d1^T H d1 = {d1Hd1:.4e}, d2^T H d2 = {d2Hd2:.4e}")

    return results


# ============================================================
# 8. Visualization Functions
# ============================================================

def plot_2d_surface(alphas, betas, surface, title, filename, vmin=None, vmax=None):
    """Plot a 2D loss surface as contour + 3D."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    A, B = np.meshgrid(alphas, betas)
    if vmin is None:
        vmin = surface.min()
    if vmax is None:
        vmax = min(surface.max(), surface.min() + 3 * (np.median(surface) - surface.min() + 0.1))
    levels = np.linspace(vmin, vmax, 30)

    cs = axes[0].contourf(A, B, surface, levels=levels, cmap='viridis')
    axes[0].contour(A, B, surface, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=axes[0], label='Loss')
    axes[0].set_xlabel(r'$\alpha$ (direction 1)')
    axes[0].set_ylabel(r'$\beta$ (direction 2)')
    axes[0].set_title(f'{title}\n(Contour)')
    axes[0].plot(0, 0, 'r*', markersize=15, label=r'$\theta^*$')
    axes[0].legend()

    ax3d = fig.add_subplot(122, projection='3d')
    axes[1].remove()
    ax3d.plot_surface(A, B, surface, cmap='viridis', alpha=0.8, vmin=vmin, vmax=vmax)
    ax3d.set_xlabel(r'$\alpha$')
    ax3d.set_ylabel(r'$\beta$')
    ax3d.set_zlabel('Loss')
    ax3d.set_title(f'{title}\n(3D Surface)')
    ax3d.view_init(elev=30, azim=225)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_1d_comparison(curves, title, filename):
    """Plot multiple 1D loss curves for comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (alphas, losses) in curves.items():
        ax.plot(alphas, losses, label=label, linewidth=2)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# ============================================================
# 9. Main Experiment
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
    parser.add_argument('--n_grad_batches', type=int, default=100)
    parser.add_argument('--hvp_batches', type=int, default=3)
    parser.add_argument('--power_iter', type=int, default=30)
    parser.add_argument('--n_hutchinson', type=int, default=10)
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
        'experiments': {}
    }

    # ---- Load model and tokenizer ----
    print(f"Loading model: {args.model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load bfloat16 model for grid evaluation
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager"
    ).to(device)
    model.eval()

    config = model.config
    num_heads = getattr(config, 'num_attention_heads', None)
    head_dim = getattr(config, 'head_dim', None)
    if head_dim is None and num_heads:
        hidden_size = getattr(config, 'hidden_size', None)
        if hidden_size:
            head_dim = hidden_size // num_heads
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  num_heads={num_heads}, head_dim={head_dim}, params={n_params:,}")

    # ---- Prepare data ----
    print("Loading WikiText-2 dataset...")
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in dataset['text'] if len(t.strip()) > 50]
    all_tokens = tokenizer('\n'.join(texts[:200]), return_tensors='pt', truncation=False)['input_ids'][0]

    chunks = []
    for i in range(0, len(all_tokens) - args.seq_len, args.seq_len):
        chunks.append(all_tokens[i:i + args.seq_len])
    print(f"  Created {len(chunks)} chunks of length {args.seq_len}")

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, chunks):
            self.chunks = chunks
        def __len__(self):
            return len(self.chunks)
        def __getitem__(self, idx):
            return {'input_ids': self.chunks[idx],
                    'attention_mask': torch.ones(len(self.chunks[idx]))}

    eval_dataset = SimpleDataset(chunks[:50])
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    grad_dataset = SimpleDataset(chunks[:args.n_grad_batches])
    grad_dataloader = torch.utils.data.DataLoader(grad_dataset, batch_size=1, shuffle=False)

    # Normalization units
    units = get_normalization_units(model)

    # Baseline loss
    print("\nComputing baseline loss...")
    baseline_loss = evaluate_loss(model, eval_dataloader, device, max_batches=args.max_eval_batches)
    print(f"  Baseline loss: {baseline_loss:.4f}")
    results_log['baseline_loss'] = baseline_loss

    # ===========================================================
    # EXPERIMENT 1: TADN Scale-Invariance Test
    # ===========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: TADN Scale-Invariance Advantage")
    print("=" * 70)

    # Create rescaled model (functionally equivalent)
    print("Creating rescaled model (non-uniform FFN neuron scaling)...")
    model_rescaled = create_rescaled_model(model)
    model_rescaled.eval()

    # Verify equivalence
    loss_orig, loss_resc, loss_diff = verify_model_equivalence(
        model, model_rescaled, eval_dataloader, device, max_batches=args.max_eval_batches)
    print(f"  Original loss: {loss_orig:.6f}")
    print(f"  Rescaled loss: {loss_resc:.6f}")
    print(f"  Difference: {loss_diff:.6f}")
    results_log['experiments']['tadn_invariance'] = {
        'loss_original': loss_orig,
        'loss_rescaled': loss_resc,
        'loss_diff': loss_diff,
    }

    # Get normalization units for rescaled model
    units_rescaled = get_normalization_units(model_rescaled)

    # Generate same random direction for both models
    raw_d = generate_random_direction(model, seed=42)

    # Apply TADN to both models
    d_tadn_orig = apply_tadn(raw_d, model, units, num_heads, head_dim)
    d_tadn_resc = apply_tadn(raw_d, model_rescaled, units_rescaled, num_heads, head_dim)

    # Apply Layer Normalization to both models
    d_layer_orig = apply_layer_normalization(raw_d, model)
    d_layer_resc = apply_layer_normalization(raw_d, model_rescaled)

    # Move to device
    d_tadn_orig = {k: v.to(device) for k, v in d_tadn_orig.items()}
    d_tadn_resc = {k: v.to(device) for k, v in d_tadn_resc.items()}
    d_layer_orig = {k: v.to(device) for k, v in d_layer_orig.items()}
    d_layer_resc = {k: v.to(device) for k, v in d_layer_resc.items()}

    # Evaluate 1D loss curves
    print("\nEvaluating 1D loss curves...")
    n_pts = 31
    alpha_rng = (-args.grid_range, args.grid_range)

    a1, l1 = evaluate_1d_curve(model, d_tadn_orig, eval_dataloader, device,
                                alpha_rng, n_pts, args.max_eval_batches)
    print(f"  TADN Original: done")
    a2, l2 = evaluate_1d_curve(model_rescaled, d_tadn_resc, eval_dataloader, device,
                                alpha_rng, n_pts, args.max_eval_batches)
    print(f"  TADN Rescaled: done")
    a3, l3 = evaluate_1d_curve(model, d_layer_orig, eval_dataloader, device,
                                alpha_rng, n_pts, args.max_eval_batches)
    print(f"  Layer Norm Original: done")
    a4, l4 = evaluate_1d_curve(model_rescaled, d_layer_resc, eval_dataloader, device,
                                alpha_rng, n_pts, args.max_eval_batches)
    print(f"  Layer Norm Rescaled: done")

    # Compute correlations
    corr_tadn = np.corrcoef(l1, l2)[0, 1]
    corr_layer = np.corrcoef(l3, l4)[0, 1]
    mse_tadn = np.mean((l1 - l2) ** 2)
    mse_layer = np.mean((l3 - l4) ** 2)
    max_dev_tadn = np.max(np.abs(l1 - l2))
    max_dev_layer = np.max(np.abs(l3 - l4))

    print(f"\n  TADN: corr={corr_tadn:.6f}, MSE={mse_tadn:.6f}, max_dev={max_dev_tadn:.6f}")
    print(f"  Layer: corr={corr_layer:.6f}, MSE={mse_layer:.6f}, max_dev={max_dev_layer:.6f}")

    results_log['experiments']['tadn_invariance'].update({
        'tadn_correlation': corr_tadn,
        'layer_correlation': corr_layer,
        'tadn_mse': mse_tadn,
        'layer_mse': mse_layer,
        'tadn_max_deviation': max_dev_tadn,
        'layer_max_deviation': max_dev_layer,
    })

    # Plot
    plot_1d_comparison({
        'TADN (Original)': (a1, l1),
        'TADN (Rescaled)': (a2, l2),
        'Layer Norm (Original)': (a3, l3),
        'Layer Norm (Rescaled)': (a4, l4),
    }, 'TADN vs Layer Normalization: Scale Invariance Test\n'
       f'TADN corr={corr_tadn:.4f}, Layer corr={corr_layer:.4f}',
       os.path.join(output_dir, 'exp1_tadn_invariance.png'))

    # Also plot deviations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(a1, np.abs(l1 - l2), 'b-o', label=f'TADN (max={max_dev_tadn:.4f})', markersize=3)
    ax1.plot(a3, np.abs(l3 - l4), 'r-s', label=f'Layer Norm (max={max_dev_layer:.4f})', markersize=3)
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel('|Loss deviation|')
    ax1.set_title('Deviation between Original and Rescaled Model')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(['TADN', 'Layer Norm'], [corr_tadn, corr_layer], color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Pearson Correlation')
    ax2.set_title('Correlation of Loss Curves')
    ax2.set_ylim(min(corr_layer, corr_tadn) - 0.05, 1.005)
    for i, v in enumerate([corr_tadn, corr_layer]):
        ax2.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp1_tadn_deviation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Clean up rescaled model
    del model_rescaled, d_tadn_orig, d_tadn_resc, d_layer_orig, d_layer_resc
    torch.cuda.empty_cache()
    gc.collect()

    # ===========================================================
    # EXPERIMENT 2: Gradient PCA Convergence Analysis
    # ===========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Gradient PCA Convergence Analysis")
    print("=" * 70)

    pca_results, pca_directions = gradient_pca_with_convergence(
        model, grad_dataloader, device,
        n_max=min(args.n_grad_batches, len(chunks)),
        checkpoints=[10, 20, 30, 50, 75, 100],
        k=2
    )

    # Plot convergence
    checkpoints_list = sorted(pca_results.keys())
    angles = [pca_results[c]['subspace_angle_from_prev'] for c in checkpoints_list]
    ev_ratios = [pca_results[c]['explained_ratios'] for c in checkpoints_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subspace angle convergence
    valid_angles = [(c, a) for c, a in zip(checkpoints_list, angles) if a is not None]
    if valid_angles:
        ax1.plot([c for c, _ in valid_angles], [a for _, a in valid_angles],
                 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of gradient samples (N)')
        ax1.set_ylabel('Subspace angle (degrees)')
        ax1.set_title('PCA Subspace Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='5-degree threshold')
        ax1.legend()

    # Explained variance ratios
    ax2.plot(checkpoints_list, [r[0] for r in ev_ratios], 'b-o', label='PC1', linewidth=2)
    ax2.plot(checkpoints_list, [r[1] for r in ev_ratios], 'r-s', label='PC2', linewidth=2)
    ax2.set_xlabel('Number of gradient samples (N)')
    ax2.set_ylabel('Explained variance ratio')
    ax2.set_title('Explained Variance vs. Sample Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_pca_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'exp2_pca_convergence.png')}")

    results_log['experiments']['pca_convergence'] = {
        str(c): {
            'eigenvalues': pca_results[c]['eigenvalues'],
            'explained_ratios': pca_results[c]['explained_ratios'],
            'subspace_angle': pca_results[c]['subspace_angle_from_prev'],
        }
        for c in pca_results
    }

    # ===========================================================
    # EXPERIMENT 3: Tier 3 — Hessian Eigenvectors via Power Iteration
    # ===========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Tier 3 — Hessian Eigenvector Directions")
    print("=" * 70)

    # Move bf16 model to CPU to free GPU memory for fp32 HVP
    print("Moving bf16 model to CPU to free GPU memory...")
    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # Create float32 model for HVP computation
    print("Creating float32 model for HVP computation...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, trust_remote_code=True,
        attn_implementation="eager"
    ).to(device)
    model_fp32.eval()

    hvp_batch_size = 1  # Minimal batch for HVP to save memory
    hvp_dataloader = torch.utils.data.DataLoader(
        SimpleDataset(chunks[:args.hvp_batches]),
        batch_size=hvp_batch_size, shuffle=False
    )

    hessian_vecs, hessian_eigs = power_iteration_hessian(
        model_fp32, hvp_dataloader, device,
        n_iter=args.power_iter, n_vectors=2, max_batches=args.hvp_batches
    )

    results_log['experiments']['hessian_eigenvectors'] = {
        'lambda_1': hessian_eigs[0],
        'lambda_2': hessian_eigs[1],
        'lambda_ratio': hessian_eigs[0] / (hessian_eigs[1] + 1e-10),
    }

    # Curvature-aware scale: characteristic length
    lambda_max = max(abs(hessian_eigs[0]), abs(hessian_eigs[1]))
    l_char = 1.0 / math.sqrt(lambda_max) if lambda_max > 1e-10 else 1.0
    print(f"  Characteristic length: l_char = {l_char:.6f}")
    print(f"  Recommended grid range: [{-3*l_char:.4f}, {3*l_char:.4f}]")
    results_log['experiments']['hessian_eigenvectors']['l_char'] = l_char

    # ===========================================================
    # EXPERIMENT 4: PFI Computation for All 3 Tiers
    # ===========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Projection Faithfulness Index (PFI)")
    print("=" * 70)

    # Model is on CPU after Exp 3. TADN uses model params for norms, so ensure
    # directions are on CPU for TADN, then move to GPU for PFI (which uses model_fp32).

    # Tier 1: Random + TADN (model on CPU generates CPU directions)
    raw_d1 = generate_random_direction(model, seed=42)
    raw_d2 = generate_random_direction(model, seed=123)
    raw_d2 = orthogonalize_directions(raw_d1, raw_d2)
    tier1_d1 = apply_tadn(raw_d1, model, units, num_heads, head_dim)
    tier1_d2 = apply_tadn(raw_d2, model, units, num_heads, head_dim)
    tier1_d1 = {k: v.to(device) for k, v in tier1_d1.items()}
    tier1_d2 = {k: v.to(device) for k, v in tier1_d2.items()}

    # Tier 2: Gradient PCA + TADN (move PCA directions to CPU first)
    pca_d0_cpu = {k: v.cpu() for k, v in pca_directions[0].items()}
    pca_d1_cpu = {k: v.cpu() for k, v in pca_directions[1].items()}
    tier2_d1 = apply_tadn(pca_d0_cpu, model, units, num_heads, head_dim)
    tier2_d2 = apply_tadn(pca_d1_cpu, model, units, num_heads, head_dim)
    tier2_d1 = {k: v.to(device) for k, v in tier2_d1.items()}
    tier2_d2 = {k: v.to(device) for k, v in tier2_d2.items()}

    # Tier 3: Hessian eigenvectors + TADN (already on CPU from power iteration)
    hess_d0_cpu = {k: v.cpu() for k, v in hessian_vecs[0].items()}
    hess_d1_cpu = {k: v.cpu() for k, v in hessian_vecs[1].items()}
    tier3_d1 = apply_tadn(hess_d0_cpu, model, units, num_heads, head_dim)
    tier3_d2 = apply_tadn(hess_d1_cpu, model, units, num_heads, head_dim)
    tier3_d1 = {k: v.to(device) for k, v in tier3_d1.items()}
    tier3_d2 = {k: v.to(device) for k, v in tier3_d2.items()}

    # Compute tr(H^2) ONCE (model property, shared across all tiers)
    print("\n--- Computing tr(H^2) via Hutchinson (shared across tiers) ---")
    tr_h2, tr_h2_std = compute_hutchinson_tr_h2(
        model_fp32, hvp_dataloader, device,
        n_hutchinson=args.n_hutchinson, max_batches=args.hvp_batches)

    # Compute PFI for each tier (only 2 HVPs per tier now)
    print("\n--- Tier 1: Random + TADN ---")
    pfi_tier1 = compute_pfi(model_fp32, hvp_dataloader, device,
                            tier1_d1, tier1_d2,
                            lambda_max=hessian_eigs[0],
                            tr_h2=tr_h2, tr_h2_std=tr_h2_std,
                            max_batches=args.hvp_batches)

    print("\n--- Tier 2: Gradient PCA + TADN ---")
    pfi_tier2 = compute_pfi(model_fp32, hvp_dataloader, device,
                            tier2_d1, tier2_d2,
                            lambda_max=hessian_eigs[0],
                            tr_h2=tr_h2, tr_h2_std=tr_h2_std,
                            max_batches=args.hvp_batches)

    print("\n--- Tier 3: Hessian Eigvec + TADN ---")
    pfi_tier3 = compute_pfi(model_fp32, hvp_dataloader, device,
                            tier3_d1, tier3_d2,
                            lambda_max=hessian_eigs[0],
                            tr_h2=tr_h2, tr_h2_std=tr_h2_std,
                            max_batches=args.hvp_batches)

    results_log['experiments']['pfi'] = {
        'tier1': pfi_tier1,
        'tier2': pfi_tier2,
        'tier3': pfi_tier3,
    }

    # PFI comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    tiers = ['Tier 1\n(Random)', 'Tier 2\n(Grad PCA)', 'Tier 3\n(Hessian)']
    pfi_s_vals = [pfi_tier1['PFI_S'], pfi_tier2['PFI_S'], pfi_tier3['PFI_S']]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    ax1.bar(tiers, pfi_s_vals, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('PFI-S (Spectral Coverage)')
    ax1.set_title('Projection Faithfulness Index — Spectral Coverage')
    for i, v in enumerate(pfi_s_vals):
        ax1.text(i, v + max(pfi_s_vals) * 0.02, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    pfi_c_vals = [pfi_tier1.get('PFI_C', 0) or 0,
                  pfi_tier2.get('PFI_C', 0) or 0,
                  pfi_tier3.get('PFI_C', 0) or 0]
    ax2.bar(tiers, pfi_c_vals, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('PFI-C (Curvature Capture)')
    ax2.set_title('Projection Faithfulness Index — Curvature Capture')
    for i, v in enumerate(pfi_c_vals):
        ax2.text(i, v + max(pfi_c_vals) * 0.02, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp4_pfi_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'exp4_pfi_comparison.png')}")

    # Clean up fp32 model and move bf16 model back to GPU
    del model_fp32
    torch.cuda.empty_cache()
    gc.collect()
    print("Moving bf16 model back to GPU...")
    model.to(device)
    model.eval()

    # ===========================================================
    # EXPERIMENT 5: Full 3-Tier Landscape Comparison
    # ===========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Full 3-Tier Landscape Comparison")
    print("=" * 70)

    # Use curvature-aware scale for Tier 3
    tier3_range = min(3 * l_char, args.grid_range)
    print(f"  Tier 3 grid range: [{-tier3_range:.4f}, {tier3_range:.4f}] (curvature-aware)")

    surfaces = {}
    for tier_name, d1, d2, grange in [
        ('Tier 1: Random+TADN', tier1_d1, tier1_d2, args.grid_range),
        ('Tier 2: Grad PCA+TADN', tier2_d1, tier2_d2, args.grid_range),
        ('Tier 3: Hessian+TADN', tier3_d1, tier3_d2, tier3_range),
    ]:
        print(f"\n--- {tier_name} (range={grange:.4f}) ---")
        alphas, betas, surface = evaluate_2d_surface(
            model, d1, d2, eval_dataloader, device,
            grid_range=(-grange, grange),
            grid_size=args.grid_size,
            max_batches=args.max_eval_batches
        )
        surfaces[tier_name] = (alphas, betas, surface)

        plot_2d_surface(alphas, betas, surface, tier_name,
                        os.path.join(output_dir, f'exp5_surface_{tier_name.split(":")[0].strip().lower().replace(" ","_")}.png'))

        from scipy.ndimage import uniform_filter
        center = surface[args.grid_size // 2, args.grid_size // 2]
        loss_range = surface.max() - surface.min()
        smoothed = uniform_filter(surface, size=3)
        roughness = np.std(surface - smoothed)

        metrics = {
            'center_loss': float(center),
            'loss_range': float(loss_range),
            'roughness': float(roughness),
            'loss_min': float(surface.min()),
            'loss_max': float(surface.max()),
            'grid_range': grange,
        }
        results_log['experiments'][f'surface_{tier_name.split(":")[0].strip().lower().replace(" ","_")}'] = metrics
        print(f"  center_loss={center:.4f}, loss_range={loss_range:.4f}, roughness={roughness:.6f}")

    # Side-by-side comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for idx, (tier_name, (alphas, betas, surface)) in enumerate(surfaces.items()):
        A, B = np.meshgrid(alphas, betas)
        vmin = surface.min()
        vmax = min(surface.max(), surface.min() + 2 * (np.median(surface) - surface.min() + 0.1))
        levels = np.linspace(vmin, vmax, 25)
        cs = axes[idx].contourf(A, B, surface, levels=levels, cmap='viridis')
        axes[idx].contour(A, B, surface, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
        axes[idx].set_xlabel(r'$\alpha$')
        axes[idx].set_ylabel(r'$\beta$')
        axes[idx].set_title(tier_name)
        axes[idx].plot(0, 0, 'r*', markersize=15)
        axes[idx].set_aspect('equal')
        plt.colorbar(cs, ax=axes[idx], shrink=0.8)

    plt.suptitle('3-Tier Direction Selection Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp5_3tier_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'exp5_3tier_comparison.png')}")

    # ===========================================================
    # Save all results
    # ===========================================================
    results_file = os.path.join(output_dir, 'poc_results_v2.json')
    with open(results_file, 'w') as f:
        json.dump(results_log, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nExp 1 — TADN Scale Invariance:")
    print(f"  TADN correlation: {corr_tadn:.6f}")
    print(f"  Layer Norm correlation: {corr_layer:.6f}")
    print(f"  TADN advantage: {(corr_tadn - corr_layer):.6f}")
    print(f"\nExp 2 — PCA Convergence:")
    for c in sorted(pca_results.keys()):
        a = pca_results[c]['subspace_angle_from_prev']
        print(f"  N={c}: angle={a:.2f}deg" if a else f"  N={c}: (baseline)")
    print(f"\nExp 3 — Hessian Eigenvalues:")
    print(f"  lambda_1 = {hessian_eigs[0]:.6f}")
    print(f"  lambda_2 = {hessian_eigs[1]:.6f}")
    print(f"  l_char = {l_char:.6f}")
    print(f"\nExp 4 — PFI Comparison:")
    print(f"  Tier 1 PFI-S: {pfi_tier1['PFI_S']:.6f}")
    print(f"  Tier 2 PFI-S: {pfi_tier2['PFI_S']:.6f}")
    print(f"  Tier 3 PFI-S: {pfi_tier3['PFI_S']:.6f}")
    if pfi_tier1.get('PFI_C') is not None:
        print(f"  Tier 1 PFI-C: {pfi_tier1['PFI_C']:.6f}")
        print(f"  Tier 2 PFI-C: {pfi_tier2['PFI_C']:.6f}")
        print(f"  Tier 3 PFI-C: {pfi_tier3['PFI_C']:.6f}")

    print("\n" + "=" * 70)
    print("Enhanced PoC Complete!")
    print("=" * 70)
    return results_log


if __name__ == '__main__':
    main()
