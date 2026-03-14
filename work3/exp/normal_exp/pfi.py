"""
pfi.py — Projection Faithfulness Index (PFI).

Quantifies how faithfully a 2D projection represents the true
high-dimensional loss landscape geometry.

PFI-S (Spectral Coverage): fraction of Hessian spectral energy captured
PFI-C (Curvature Capture): alignment with maximum curvature direction
"""

import math
import numpy as np
import torch

from direction_selection import compute_hvp


def compute_hutchinson_tr_h2(model_fp32, dataloader, device,
                              n_hutchinson=10, max_batches=3, verbose=True):
    """
    Estimate tr(H^2) via Hutchinson's trace estimator.

    tr(H^2) = E[||Hv||^2] where v ~ N(0, I)

    This is a model-level property (independent of projection directions),
    so it should be computed once and shared across all tiers.

    Args:
        model_fp32: float32 model
        dataloader: data loader for HVP
        device: torch device
        n_hutchinson: number of random vectors for estimation
        max_batches: batches per HVP
        verbose: print progress

    Returns:
        (tr_h2, tr_h2_std): estimated tr(H^2) and standard error
    """
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
    Compute the Projection Faithfulness Index for a pair of directions.

    PFI-S = (||Hd1||^2 + ||Hd2||^2) / tr(H^2)
    PFI-C = max(|d1^T H d1|, |d2^T H d2|) / |lambda_max|

    Args:
        model_fp32: float32 model
        dataloader: data loader for HVP
        device: torch device
        d1, d2: direction dicts
        lambda_max: largest Hessian eigenvalue (from Tier 3)
        tr_h2: precomputed tr(H^2)
        tr_h2_std: standard error of tr(H^2) estimate
        max_batches: batches per HVP
        verbose: print progress

    Returns:
        dict with PFI-S, PFI-C, and intermediate quantities
    """
    if verbose:
        print("  Computing PFI...")

    # Normalize to unit vectors
    d1_norm = math.sqrt(sum((d1[n].float() ** 2).sum().item() for n in d1))
    d2_norm = math.sqrt(sum((d2[n].float() ** 2).sum().item() for n in d2))
    d1_unit = {n: d1[n].float() / d1_norm for n in d1}
    d2_unit = {n: d2[n].float() / d2_norm for n in d2}

    # Compute Hd1 and Hd2 (2 HVPs)
    hd1 = compute_hvp(model_fp32, dataloader, device, d1_unit, max_batches)
    hd2 = compute_hvp(model_fp32, dataloader, device, d2_unit, max_batches)

    # ||Hd||^2 and d^T H d
    hd1_sq = sum((hd1[n] ** 2).sum().item() for n in hd1)
    hd2_sq = sum((hd2[n] ** 2).sum().item() for n in hd2)
    d1Hd1 = sum((d1_unit[n].to(device) * hd1[n]).sum().item() for n in hd1)
    d2Hd2 = sum((d2_unit[n].to(device) * hd2[n]).sum().item() for n in hd2)

    del hd1, hd2, d1_unit, d2_unit
    torch.cuda.empty_cache()

    pfi_s = (hd1_sq + hd2_sq) / (tr_h2 + 1e-10) if tr_h2 is not None else None

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
            print(f"    PFI-S = {pfi_s:.6e}")
        if pfi_c is not None:
            print(f"    PFI-C = {pfi_c:.6e}")

    return results
