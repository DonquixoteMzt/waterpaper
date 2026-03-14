"""
grid_evaluation.py — Loss surface evaluation on 2D grids.

Implements exact parameter restoration for numerically correct
grid evaluation at each (alpha, beta) point.
"""

import time
import numpy as np
import torch


@torch.no_grad()
def evaluate_loss(model, dataloader, device, max_batches=None):
    """
    Evaluate average cross-entropy loss over the dataloader.

    Args:
        model: the model
        dataloader: evaluation data loader
        device: torch device
        max_batches: max number of batches (None = all)

    Returns:
        float: average loss
    """
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


def evaluate_2d_surface(model, d1, d2, dataloader, device,
                        grid_range=(-1.0, 1.0), grid_size=51,
                        max_batches=5, verbose=True):
    """
    Evaluate 2D loss surface f(alpha, beta) = L(theta* + alpha*d1 + beta*d2).

    Uses exact parameter restoration at each grid point to avoid
    bfloat16 rounding error accumulation.

    Args:
        model: the model (bfloat16 for efficiency)
        d1, d2: direction dicts
        dataloader: evaluation data loader
        device: torch device
        grid_range: (min, max) for alpha and beta
        grid_size: number of grid points per axis
        max_batches: batches per loss evaluation
        verbose: print progress

    Returns:
        (alphas, betas, surface): 1D arrays and 2D loss matrix
    """
    if isinstance(grid_range, (int, float)):
        grid_range = (-abs(grid_range), abs(grid_range))

    alphas = np.linspace(grid_range[0], grid_range[1], grid_size)
    betas = np.linspace(grid_range[0], grid_range[1], grid_size)
    surface = np.zeros((grid_size, grid_size))

    # Save original parameters (exact restoration)
    original_params = {name: param.data.clone() for name, param in model.named_parameters()}

    t0 = time.time()
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Exact restore
            for name, param in model.named_parameters():
                param.data.copy_(original_params[name])
            # Apply perturbation
            if alpha != 0.0:
                for name, param in model.named_parameters():
                    if name in d1:
                        param.data.add_(alpha * d1[name].to(param.dtype).to(param.device))
            if beta != 0.0:
                for name, param in model.named_parameters():
                    if name in d2:
                        param.data.add_(beta * d2[name].to(param.dtype).to(param.device))
            surface[j, i] = evaluate_loss(model, dataloader, device, max_batches=max_batches)

        if verbose:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (grid_size - i - 1)
            if (i + 1) % max(1, grid_size // 10) == 0 or i == 0:
                print(f"  Row {i+1}/{grid_size} done. Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

    # Restore original parameters
    for name, param in model.named_parameters():
        param.data.copy_(original_params[name])
    del original_params
    torch.cuda.empty_cache()

    return alphas, betas, surface


def evaluate_1d_curve(model, direction, dataloader, device,
                      alpha_range=(-1.0, 1.0), n_points=31,
                      max_batches=5):
    """
    Evaluate 1D loss curve f(alpha) = L(theta* + alpha*d).

    Args:
        model: the model
        direction: direction dict
        dataloader: evaluation data loader
        device: torch device
        alpha_range: (min, max)
        n_points: number of evaluation points
        max_batches: batches per evaluation

    Returns:
        (alphas, losses): 1D arrays
    """
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
