"""
normalization.py — Transformer-Adapted Direction Normalization (TADN).

Implements per-component normalization for transformer architectures:
- Per-head for attention projections (Q, K, V, O)
- Per-neuron for FFN layers (up_proj rows, down_proj columns)
- Per-token for embeddings
- Whole-vector for layer norms

Also implements baseline normalization methods for comparison.
"""

import torch
import math


def get_normalization_units(model):
    """
    Partition model parameters into TADN normalization units.

    Returns:
        dict mapping parameter name -> list of (unit_type, unit_info) tuples
        unit_type: 'row', 'col', 'head', 'whole'
        unit_info: axis index or 'qkv'/'o' for attention heads
    """
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
            # up_proj.weight: [intermediate_size, hidden_size] — each row = one neuron
            units[name] = [('row', 0)]
        elif 'down_proj' in name:
            # down_proj.weight: [hidden_size, intermediate_size] — each column = one neuron
            units[name] = [('col', 1)]
        elif any(k in name for k in ['layernorm', 'rmsnorm', 'norm', 'ln_']):
            units[name] = [('whole', None)]
        else:
            units[name] = [('whole', None)]
    return units


def apply_tadn(direction, model, units, num_heads=None, head_dim=None, epsilon=1e-8):
    """
    Apply Transformer-Adapted Direction Normalization (TADN).

    For each normalization unit i: d_i = (d_i / ||d_i||) * ||theta_i||

    Args:
        direction: dict {param_name: tensor} — raw direction
        model: the model (for parameter norms)
        units: normalization unit partition from get_normalization_units()
        num_heads: number of attention heads
        head_dim: dimension per head
        epsilon: threshold for near-zero norms

    Returns:
        dict {param_name: tensor} — TADN-normalized direction
    """
    normalized = {}
    for name, param in model.named_parameters():
        if name not in direction:
            continue
        # Ensure both tensors are on CPU in float32 for normalization
        d = direction[name].detach().cpu().clone().float()
        p = param.data.detach().cpu().float()

        if name not in units:
            p_norm = p.norm().item()
            d_norm = d.norm().item()
            if p_norm > epsilon and d_norm > epsilon:
                d = d * (p_norm / d_norm)
            normalized[name] = d.to(direction[name].dtype).to(direction[name].device)
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

        normalized[name] = d.to(direction[name].dtype).to(direction[name].device)
    return normalized


def apply_layer_normalization(direction, model, epsilon=1e-8):
    """
    Baseline: normalize each parameter matrix as a single unit.

    This is the standard layer-level normalization (adapted from Li et al., 2018
    CNN filter normalization). Each weight matrix is treated as one normalization
    unit, regardless of internal structure.

    Args:
        direction: dict {param_name: tensor}
        model: the model
        epsilon: threshold for near-zero norms

    Returns:
        dict {param_name: tensor} — layer-normalized direction
    """
    normalized = {}
    for name, param in model.named_parameters():
        if name not in direction:
            continue
        d = direction[name].detach().cpu().clone().float()
        p_norm = param.data.detach().cpu().float().norm().item()
        d_norm = d.norm().item()
        if p_norm > epsilon and d_norm > epsilon:
            d = d * (p_norm / d_norm)
        normalized[name] = d.to(direction[name].dtype).to(direction[name].device)
    return normalized


def apply_no_normalization(direction, model):
    """
    No normalization baseline: return raw direction as-is.

    Args:
        direction: dict {param_name: tensor}
        model: the model (unused, for API consistency)

    Returns:
        dict {param_name: tensor} — unnormalized direction
    """
    return {name: direction[name].clone() for name in direction}


def create_rescaled_model(model):
    """
    Create a functionally equivalent model by non-uniformly scaling FFN neurons.

    For SwiGLU FFN: scale W_up rows by c_j and W_down columns by 1/c_j.
    This preserves FFN(x) = W_down * (swish(W_gate*x) * W_up*x) exactly.
    Uses powers of 2 as scale factors for exact bfloat16 representation.

    Args:
        model: original model

    Returns:
        rescaled model (deep copy with modified weights)
    """
    import copy
    model_rescaled = copy.deepcopy(model)

    for name, param in model_rescaled.named_parameters():
        if 'up_proj.weight' in name:
            n_neurons = param.shape[0]
            scales = torch.ones(n_neurons, device=param.device, dtype=param.dtype)
            scales[:n_neurons // 4] = 8.0
            scales[n_neurons // 4:n_neurons // 2] = 4.0
            scales[n_neurons // 2:3 * n_neurons // 4] = 0.25
            scales[3 * n_neurons // 4:] = 0.125
            param.data *= scales.unsqueeze(1)
        elif 'down_proj.weight' in name:
            n_neurons = param.shape[1]
            inv_scales = torch.ones(n_neurons, device=param.device, dtype=param.dtype)
            inv_scales[:n_neurons // 4] = 0.125
            inv_scales[n_neurons // 4:n_neurons // 2] = 0.25
            inv_scales[n_neurons // 2:3 * n_neurons // 4] = 4.0
            inv_scales[3 * n_neurons // 4:] = 8.0
            param.data *= inv_scales.unsqueeze(0)

    return model_rescaled
