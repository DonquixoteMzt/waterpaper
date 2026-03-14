"""
direction_selection.py — Scalable Hessian-Informed Direction Selection (SHIDS).

Implements three tiers of direction selection:
- Tier 1: Random Gaussian directions (+ TADN normalization)
- Tier 2: Gradient Covariance PCA with adaptive convergence monitoring
- Tier 3: Power iteration for top Hessian eigenvectors via HVP
"""

import math
import time
import gc
import numpy as np
import torch


# ============================================================
# Tier 1: Random Directions
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
    """Orthogonalize d2 w.r.t. d1 using Gram-Schmidt in flattened space."""
    d1_flat = torch.cat([d1[n].flatten().float() for n in d1])
    d2_flat = torch.cat([d2[n].flatten().float() for n in d2])
    d2_flat = d2_flat - (d2_flat @ d1_flat) / (d1_flat @ d1_flat + 1e-10) * d1_flat
    offset = 0
    for name in d2:
        numel = d2[name].numel()
        d2[name] = d2_flat[offset:offset + numel].reshape(d2[name].shape).to(d2[name].dtype)
        offset += numel
    return d2


def generate_tier1_directions(model, seed1=42, seed2=123):
    """
    Generate Tier 1 directions: two orthogonal random Gaussian directions.

    Returns:
        (d1, d2): two direction dicts
    """
    d1 = generate_random_direction(model, seed=seed1)
    d2 = generate_random_direction(model, seed=seed2)
    d2 = orthogonalize_directions(d1, d2)
    return d1, d2


# ============================================================
# Tier 2: Gradient Covariance PCA with Convergence Monitoring
# ============================================================

def gradient_pca_with_convergence(model, dataloader, device,
                                   n_max=200, checkpoints=None, k=2,
                                   convergence_threshold_deg=5.0,
                                   verbose=True):
    """
    Compute gradient covariance PCA directions with convergence monitoring.

    Collects per-batch gradients and finds top-k principal components of
    the gradient covariance matrix (which approximate top Hessian eigenvectors
    per Gur-Ari et al., 2018). Monitors subspace angle convergence.

    Args:
        model: the model (bfloat16 or float32)
        dataloader: gradient data loader
        device: torch device
        n_max: maximum number of gradient batches
        checkpoints: list of N values to check convergence
        k: number of PCA directions
        convergence_threshold_deg: stop if subspace angle < threshold
        verbose: print progress

    Returns:
        (pca_results, final_directions)
        pca_results: dict {N: {eigenvalues, explained_ratios, subspace_angle}}
        final_directions: list of k direction dicts (param_name -> tensor)
    """
    if checkpoints is None:
        checkpoints = [10, 20, 50, 100, 150, 200]
    checkpoints = sorted([c for c in checkpoints if c <= n_max])

    if verbose:
        print(f"Computing gradient PCA convergence (N_max={n_max})...")
    model.eval()

    grad_flat_list = []
    gram_matrix = np.zeros((n_max, n_max))

    for i, batch in enumerate(dataloader):
        if i >= n_max:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        g_flat = torch.cat([p.grad.detach().flatten().float()
                            for _, p in model.named_parameters()
                            if p.requires_grad and p.grad is not None])

        g_cpu = g_flat.cpu()
        for j in range(len(grad_flat_list)):
            dot = torch.dot(g_cpu, grad_flat_list[j]).item()
            gram_matrix[i, j] = dot
            gram_matrix[j, i] = dot
        gram_matrix[i, i] = torch.dot(g_cpu, g_cpu).item()

        grad_flat_list.append(g_cpu)
        del g_flat, outputs, loss
        torch.cuda.empty_cache()

        if verbose and (i + 1) % 50 == 0:
            print(f"  Collected {i+1}/{n_max} gradients")

    model.zero_grad()

    # Compute PCA at each checkpoint
    pca_results = {}
    prev_subspace = None
    converged = False

    for N in checkpoints:
        if N > len(grad_flat_list):
            break

        G_N = gram_matrix[:N, :N]
        ones = np.ones((N, N)) / N
        G_centered = G_N - ones @ G_N - G_N @ ones + ones @ G_N @ ones

        eigenvalues, eigenvectors = np.linalg.eigh(G_centered)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx[:k]]
        eigenvectors = eigenvectors[:, idx[:k]]

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

        # Subspace angle with previous
        angle = None
        if prev_subspace is not None:
            U = torch.stack(directions)
            V = torch.stack(prev_subspace)
            M = U @ V.T
            svd_vals = torch.linalg.svdvals(M.float())
            cos_angle = svd_vals.min().clamp(-1, 1).item()
            angle = math.degrees(math.acos(cos_angle))

        prev_subspace = directions

        pca_results[N] = {
            'eigenvalues': eigenvalues.tolist(),
            'explained_ratios': explained_ratios.tolist(),
            'subspace_angle_from_prev': angle,
            'directions_flat': directions,
        }

        if verbose:
            angle_str = f"{angle:.2f}deg" if angle is not None else "N/A"
            print(f"  N={N}: ev_ratio=[{explained_ratios[0]:.3f}, {explained_ratios[1]:.3f}], "
                  f"angle={angle_str}")

        if angle is not None and angle < convergence_threshold_deg:
            if verbose:
                print(f"  Converged at N={N} (angle={angle:.2f}° < {convergence_threshold_deg}°)")
            converged = True
            break

    # Convert final directions to param dicts
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
# Tier 3: Power Iteration with Hessian-Vector Products
# ============================================================

def compute_hvp(model_fp32, dataloader, device, v_dict, max_batches=3):
    """
    Compute Hessian-vector product Hv using the Pearlmutter trick.

    Uses float32 for numerical stability. Processes batches sequentially
    to manage memory.

    Args:
        model_fp32: float32 model
        dataloader: data loader
        device: torch device
        v_dict: direction vector dict
        max_batches: max batches for HVP estimation

    Returns:
        dict {param_name: tensor} — the HVP result
    """
    model_fp32.train()
    for p in model_fp32.parameters():
        p.requires_grad_(True)
    model_fp32.zero_grad()

    param_names = [n for n, _ in model_fp32.named_parameters()]
    params = list(model_fp32.parameters())

    hvp_accum = {n: torch.zeros_like(p.data) for n, p in model_fp32.named_parameters()}
    total_tokens = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
        n_tokens = attention_mask.sum().item()

        outputs = model_fp32(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)

        gv = sum((g * v_dict[name].to(device).float()).sum()
                 for g, name in zip(grads, param_names)
                 if g is not None and name in v_dict)

        hvp_tensors = torch.autograd.grad(gv, params, allow_unused=True)

        for name, hvp in zip(param_names, hvp_tensors):
            if hvp is not None:
                hvp_accum[name] += hvp.detach() * n_tokens

        total_tokens += n_tokens

        del outputs, loss, grads, gv, hvp_tensors
        model_fp32.zero_grad()
        torch.cuda.empty_cache()

    hvp_dict = {n: hvp_accum[n] / total_tokens for n in hvp_accum}
    model_fp32.zero_grad()
    model_fp32.eval()
    return hvp_dict


def power_iteration_hessian(model_fp32, dataloader, device,
                             n_iter=30, n_vectors=2, max_batches=3,
                             convergence_tol=0.9999, verbose=True):
    """
    Compute top-k Hessian eigenvectors via power iteration with HVP.

    Args:
        model_fp32: float32 model with eager attention
        dataloader: data loader for HVP
        device: torch device
        n_iter: max power iterations per eigenvector
        n_vectors: number of eigenvectors to compute
        max_batches: batches per HVP
        convergence_tol: cosine similarity threshold for convergence
        verbose: print progress

    Returns:
        (vectors, eigenvalues)
        vectors: list of direction dicts
        eigenvalues: list of floats
    """
    vectors = []
    eigenvalues = []

    for j in range(n_vectors):
        if verbose:
            print(f"  Computing eigenvector {j+1}/{n_vectors}...")
            t0 = time.time()

        torch.manual_seed(1000 + j)
        v = {n: torch.randn_like(p).float()
             for n, p in model_fp32.named_parameters() if p.requires_grad}
        v_norm = math.sqrt(sum((v[n] ** 2).sum().item() for n in v))
        for n in v:
            v[n] /= v_norm

        lam = 0.0
        n_iters_done = 0
        for t in range(n_iter):
            hv = compute_hvp(model_fp32, dataloader, device, v, max_batches)

            # Deflate against previously found eigenvectors
            for i in range(j):
                proj = sum((hv[n] * vectors[i][n].to(device)).sum().item() for n in hv)
                for n in hv:
                    hv[n] -= proj * vectors[i][n].to(device)

            lam = sum((v[n].to(device) * hv[n]).sum().item() for n in hv)

            hv_norm = math.sqrt(sum((hv[n] ** 2).sum().item() for n in hv))
            if hv_norm < 1e-10:
                break
            for n in hv:
                hv[n] /= hv_norm

            cos_sim = abs(sum((v[n].to(device) * hv[n]).sum().item() for n in hv))
            n_iters_done = t + 1
            if verbose and (t + 1) % 5 == 0:
                print(f"    Iter {t+1}: lambda={lam:.2f}, cos_sim={cos_sim:.6f}")
            if cos_sim > convergence_tol and t > 3:
                if verbose:
                    print(f"    Converged at iter {t+1}")
                break

            v = {n: hv[n].cpu() for n in hv}

        vectors.append({n: hv[n].cpu() for n in hv})
        eigenvalues.append(lam)
        if verbose:
            elapsed = time.time() - t0
            print(f"  lambda_{j+1} = {eigenvalues[j]:.2f} ({n_iters_done} iters, {elapsed:.1f}s)")

    return vectors, eigenvalues


def compute_curvature_aware_scale(eigenvalues, scale_factor=3.0):
    """
    Compute curvature-aware grid scale from Hessian eigenvalues.

    l_char = 1 / sqrt(|lambda_max|)
    grid_range = scale_factor * l_char

    Args:
        eigenvalues: list of Hessian eigenvalues
        scale_factor: multiplier for characteristic length (default 3.0)

    Returns:
        (l_char, grid_range)
    """
    lambda_max = max(abs(e) for e in eigenvalues) if eigenvalues else 1.0
    l_char = 1.0 / math.sqrt(lambda_max) if lambda_max > 1e-10 else 1.0
    grid_range = scale_factor * l_char
    return l_char, grid_range
