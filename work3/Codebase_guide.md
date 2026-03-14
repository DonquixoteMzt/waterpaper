# Codebase Guide: LLMScape — Loss Landscape Visualization for Large Language Models

## Overview

All method-critical source files reside in `exp/normal_exp/`. The codebase implements the full LLMScape pipeline described in **Methodology.md**: TADN (Section 3), SHIDS (Section 4), PFI (Section 5), MMSP (Section 6), and the Efficient Pipeline (Section 7).

---

## 1. File Index

| File | Purpose | Methodology Reference |
|------|---------|----------------------|
| `run_experiment.py` | Main orchestration: loads model, runs all experiments, saves results | Section 10 (Full Pipeline) |
| `data_loader.py` | Dataset loading, tokenization, DataLoader creation | Section 7.2 (Representative Data Subset) |
| `normalization.py` | TADN and baseline normalization methods | Section 3 (TADN) |
| `direction_selection.py` | SHIDS Tiers 1–3: Random, Gradient PCA, Hessian power iteration | Section 4 (SHIDS) |
| `pfi.py` | Projection Faithfulness Index computation | Section 5 (PFI) |
| `grid_evaluation.py` | 2D/1D loss surface evaluation with exact parameter restoration | Section 7.3–7.4 (Mixed-Precision, In-Place Perturbation) |
| `multi_model.py` | Multi-Model Shared Projection methods A, B, C | Section 6 (MMSP) |
| `metrics.py` | Geometric feature extraction from loss surfaces | Section 8 (Feature Extraction) |
| `visualization.py` | Plotting functions for all figures | — |
| `config.yaml` | Full experiment configuration (production settings) | — |
| `config_verify.yaml` | Reduced settings for fast verification runs | — |

---

## 2. Detailed File Documentation

### 2.1 `run_experiment.py` (539 lines)

**Purpose:** Main entry point. Orchestrates the full LLMScape pipeline (Methodology §10).

| Lines | Function/Block | Description | Methodology Link |
|-------|---------------|-------------|-----------------|
| 50–53 | `load_config()` | Load YAML configuration | — |
| 56–87 | `setup_model_and_tokenizer()` | Load HF model (bf16, eager attn), extract `num_heads`, `head_dim` | §7.3 |
| 90–173 | `run_tadn_invariance_test()` | **Experiment 1**: Create rescaled model, compare TADN vs Layer Norm 1D curves, compute correlation | §3.4 (Proposition 3.1) |
| 176–346 | `run_direction_selection()` | **Experiments 2–4**: Tier 1 random + TADN, Tier 1 + Layer Norm baseline, Tier 2 gradient PCA + TADN, Tier 3 Hessian + TADN, PFI computation for all tiers | §4.2, §5.4 |
| 238–264 | (within above) | Move bf16 model to CPU, load fp32 model for HVP | §7.3 (float32 for HVP) |
| 285–296 | (within above) | Move all directions to CPU to free GPU memory for PFI HVP computation | §7.3 (Memory management) |
| 349–406 | `run_surface_evaluation()` | **Experiment 5**: 2D loss surfaces for all tier/normalization combos, metric extraction, plotting | §10 (Step 5–7) |
| 409–538 | `main()` | CLI parsing, pipeline orchestration, results saving, summary printing | §10 |

**Key design decisions:**
- bf16 model and fp32 model cannot coexist on a single 40GB GPU → sequential swap (line 244–254)
- All directions kept on CPU during PFI to avoid OOM (line 285–296)
- `grid_evaluation.py` handles device transfer per-parameter, so directions can stay on CPU

---

### 2.2 `normalization.py` (213 lines)

**Purpose:** Implements TADN (Methodology §3) and baseline normalization methods.

| Lines | Function | Description | Methodology Link |
|-------|---------|-------------|-----------------|
| 17–48 | `get_normalization_units(model)` | Partition parameters into normalization units based on layer type: per-head (Q/K/V/O), per-row (embedding, up_proj, gate_proj), per-column (lm_head, down_proj), whole-vector (norms) | §3.2 (Definition 3.1) |
| 51–131 | `apply_tadn(direction, model, units, num_heads, head_dim, epsilon)` | Apply TADN: for each unit $i$, scale $\hat{d}_i = (d_i / \|d_i\|) \cdot \|\theta_i^*\|$. Handles row/col/head/whole unit types. All computation on CPU in float32 for correctness. | §3.3 (Algorithm 1) |
| 86–91 | (within above) | `'whole'` unit type: single norm for entire parameter | §3.2 (RMSNorm row) |
| 92–98 | (within above) | `'row'` unit type: per-row norms (e.g., embedding tokens, FFN up_proj neurons) | §3.2 (FFN Up/Gate row) |
| 100–106 | (within above) | `'col'` unit type: per-column norms (e.g., lm_head, FFN down_proj neurons) | §3.2 (FFN Down row, LM Head column) |
| 108–128 | (within above) | `'head'` unit type: reshape to (num_heads, head_dim, -1), normalize per head | §3.2 (Q/K/V/O per-head) |
| 134–160 | `apply_layer_normalization(direction, model, epsilon)` | Baseline: normalize each full weight matrix as one unit | §3.1 (Li et al., 2018 adapted) |
| 163–174 | `apply_no_normalization(direction, model)` | No normalization baseline (raw direction) | — |
| 177–212 | `create_rescaled_model(model)` | Deep-copy model with non-uniform FFN neuron scaling (powers of 2: 8, 4, 0.25, 0.125). Up_proj rows × $c_j$, down_proj columns × $1/c_j$. Used for TADN invariance test. | §3.4 (Proposition 3.1 proof setup) |

**Critical implementation detail:** Lines 73–74 and 130 use `.detach().cpu().clone().float()` for both direction and parameter tensors, then convert back to original dtype/device. This avoids device mismatch errors when model is on CPU (during Hessian phase) but directions may have been on GPU.

---

### 2.3 `direction_selection.py` (368 lines)

**Purpose:** Implements SHIDS three-tier direction selection (Methodology §4).

| Lines | Function | Description | Methodology Link |
|-------|---------|-------------|-----------------|
| 21–28 | `generate_random_direction(model, seed)` | Sample $\mathbf{d} \sim \mathcal{N}(0, I)$ per parameter | §4.2 Tier 1, Step 1 |
| 31–41 | `orthogonalize_directions(d1, d2)` | Gram-Schmidt in flattened space: $d_2 \leftarrow d_2 - \text{proj}_{d_1} d_2$ | §4.2 Tier 1, Step 2 |
| 44–54 | `generate_tier1_directions(model, seed1, seed2)` | Generate two orthogonal random directions | §4.2 Tier 1 |
| 61–206 | `gradient_pca_with_convergence(model, dataloader, device, n_max, checkpoints, k, convergence_threshold_deg)` | **Tier 2**: Collect N per-batch gradients, build N×N Gram matrix $G_{ij} = g_i \cdot g_j$, center, eigendecompose, reconstruct d-dimensional PCA directions, monitor subspace angle convergence | §4.2 Tier 2 (Algorithm 2) |
| 96–126 | (within above) | Gradient collection loop with incremental Gram matrix construction | §4.2 Tier 2, Steps 2a–2b |
| 134–190 | (within above) | PCA at each checkpoint N, subspace angle computation via SVD, convergence check against threshold | §4.2 Tier 2, Steps 2c–2h |
| 192–206 | (within above) | Convert final PCA directions from flat vectors to parameter dicts | §4.2 Tier 2, Step 3 |
| 213–272 | `compute_hvp(model_fp32, dataloader, device, v_dict, max_batches)` | **Pearlmutter trick HVP**: Forward → `create_graph=True` backward → dot product $g \cdot v$ → second backward = $Hv$. Token-weighted averaging across batches. | §4.2 Tier 3 (HVP Computation) |
| 251 | (within above) | `create_graph=True` — enables double backward for HVP | §4.2 Tier 3 |
| 253–255 | (within above) | $s = g \cdot v$ (scalar dot product); `.to(device).float()` handles CPU directions | §4.2 Tier 3 |
| 275–347 | `power_iteration_hessian(model_fp32, dataloader, device, n_iter, n_vectors, max_batches, convergence_tol)` | **Tier 3**: Power iteration for top-k Hessian eigenvectors. Init random → HVP → deflate → normalize → check convergence (cosine similarity > tol). | §4.2 Tier 3 (Algorithm 3) |
| 313–337 | (within above) | Main power iteration loop: HVP, deflation against previous eigenvectors, eigenvalue estimate $\lambda = v^T Hv$, convergence via cosine similarity | §4.2 Tier 3, Steps 1d.i–1d.v |
| 350–367 | `compute_curvature_aware_scale(eigenvalues, scale_factor)` | $\ell_{\text{char}} = 1/\sqrt{|\lambda_{\max}|}$, grid range $= k \cdot \ell_{\text{char}}$ | §7.1 (Algorithm 5) |

---

### 2.4 `pfi.py` (128 lines)

**Purpose:** Implements Projection Faithfulness Index (Methodology §5).

| Lines | Function | Description | Methodology Link |
|-------|---------|-------------|-----------------|
| 18–56 | `compute_hutchinson_tr_h2(model_fp32, dataloader, device, n_hutchinson, max_batches)` | Estimate $\text{tr}(H^2) = \mathbb{E}[\|Hv\|^2]$ via Hutchinson's trace estimator with $m$ random Gaussian vectors. Returns (mean, standard error). **Shared across all tiers.** | §5.4 (Denominator), Algorithm 4, Step 4–5 |
| 59–127 | `compute_pfi(model_fp32, dataloader, device, d1, d2, lambda_max, tr_h2, tr_h2_std, max_batches)` | Compute PFI-S and PFI-C for a direction pair. Normalizes directions to unit vectors, computes 2 HVPs ($H\hat{d}_1$, $H\hat{d}_2$), then: PFI-S $= (\|Hd_1\|^2 + \|Hd_2\|^2) / \text{tr}(H^2)$, PFI-C $= \max(|d_1^T H d_1|, |d_2^T H d_2|) / |\lambda_1|$ | §5.2 (Definitions 5.1–5.2), §5.4 (Algorithm 4) |
| 86–89 | (within above) | Normalize to unit vectors before HVP | Algorithm 4, implicit |
| 92–93 | (within above) | 2 HVP calls for numerator | Algorithm 4, Step 1 |
| 96–99 | (within above) | Compute $\|Hd\|^2$ and $d^T H d$ (curvatures along axes) | Algorithm 4, Steps 2–3 |
| 104 | (within above) | PFI-S = numerator / tr(H²) | Definition 5.1 |
| 108 | (within above) | PFI-C = max curvature / λ_max | Definition 5.2 |

---

### 2.5 `grid_evaluation.py` (141 lines)

**Purpose:** 2D and 1D loss surface evaluation with exact parameter restoration (Methodology §7.3–7.4).

| Lines | Function | Description | Methodology Link |
|-------|---------|-------------|-----------------|
| 13–39 | `evaluate_loss(model, dataloader, device, max_batches)` | Compute average cross-entropy loss: $L = -\frac{1}{T}\sum_t \log p_\theta(x_t)$, token-weighted across batches | §2.1 |
| 42–103 | `evaluate_2d_surface(model, d1, d2, dataloader, device, grid_range, grid_size, max_batches)` | Evaluate $f(\alpha, \beta) = L(\theta^* + \alpha d_1 + \beta d_2)$ on G×G grid. **Exact parameter restoration** at each grid point (line 72: save, line 78–79: restore). Direction cast to param dtype/device inline (line 84, 88). | §7.3–7.4, §10 Step 5 |
| 72 | (within above) | `original_params = {name: param.data.clone() ...}` — save exact copy | §7.3 (Critical: exact restoration) |
| 78–79 | (within above) | `param.data.copy_(original_params[name])` — restore before each perturbation | §7.3 |
| 84, 88 | (within above) | `.to(param.dtype).to(param.device)` — handles CPU directions with GPU model | §7.3 (Direction vectors cast to model dtype) |
| 106–140 | `evaluate_1d_curve(model, direction, dataloader, device, alpha_range, n_points, max_batches)` | 1D cross-section $f(\alpha) = L(\theta^* + \alpha d)$. Same exact restoration pattern. | §10 |

---

### 2.6 `multi_model.py` (181 lines)

**Purpose:** Implements Multi-Model Shared Projection methods (Methodology §6).

| Lines | Function | Description | Methodology Link |
|-------|---------|-------------|-----------------|
| 15–99 | `trajectory_pca(checkpoints_params, k)` | **Method A**: Compute centroid $\bar{\theta}$, form difference vectors $\theta_t - \bar{\theta}$, build Gram matrix, PCA via eigendecomposition, project checkpoints, return explained variance | §6.1 (Method A), §4.3 |
| 36–38 | (within above) | Centroid: $\bar{\theta} = \frac{1}{T+1}\sum_t \theta_t$ | §4.3, Step 2 |
| 43–54 | (within above) | Gram matrix $\Delta^T \Delta$ for efficient PCA when $T \ll d$ | §4.3, Step 3 |
| 62–80 | (within above) | Reconstruct d-dimensional directions from Gram matrix eigenvectors | §4.3, Step 4 |
| 82–93 | (within above) | Project each checkpoint: $(x_t, y_t) = (\langle \theta_t - \bar{\theta}, p_1\rangle, ...)$ | §6.1, Step 1 |
| 102–162 | `anchor_point_projection(params_a, params_b, params_c)` | **Method B**: $d_1 = \theta_B - \theta_A$. For 3 models: Gram-Schmidt of $\theta_C - \theta_A$. For 2 models: random orthogonal direction. Returns midpoint. | §6.2 |
| 119–121 | (within above) | Direction between models: $d_1 = \theta_B - \theta_A$ | §6.2, Step 1 |
| 131–148 | (within above) | Three-model Gram-Schmidt orthogonalization | §6.2, Steps 1–3 |
| 165–180 | `compute_model_distance(params_a, params_b)` | L2 distance $\|\theta_A - \theta_B\|_2$ | §6.3 |

---

### 2.7 `metrics.py` (141 lines)

**Purpose:** Geometric feature extraction from loss surfaces (Methodology §8).

| Lines | Function | Description | Methodology Link |
|-------|---------|-------------|-----------------|
| 12–117 | `compute_surface_metrics(alphas, betas, surface, delta_factor)` | Extract all geometric metrics from a 2D surface | §8.1–8.2 |
| 28–34 | (within above) | Basic stats: center_loss, min, max, range, mean, median | §8 |
| 37–38 | (within above) | Roughness: std of residuals after 3×3 uniform smoothing | §8.2 ($R$) |
| 41–53 | (within above) | Quadratic fit $f(\alpha,\beta) \approx c_0 + c_1\alpha + c_2\beta + c_3\alpha^2 + c_4\beta^2 + c_5\alpha\beta$ via least-squares | §8.2 |
| 57–59 | (within above) | Curvatures: $\kappa_1 = 2c_3$, $\kappa_2 = 2c_4$, ratio $\rho = |\kappa_1|/|\kappa_2|$ | §8.1 (Curvature ratio $\rho$) |
| 62–70 | (within above) | Basin metrics: $\delta$-sublevel set area fraction, effective diameter $w_{\text{eff}} = 2\sqrt{A/\pi}$ | §8.1 ($w_{\text{eff}}$) |
| 73–76 | (within above) | Basin flatness $\Phi$: mean excess loss within basin | §8.1 ($\Phi$) |
| 79–86 | (within above) | Asymmetry along $\alpha$ axis | §8 |
| 89–97 | (within above) | Convexity ratio: fraction of grid points with positive local second derivative | §8.2 |
| 120–140 | `format_metrics_table(metrics_dict, name)` | Pretty-print metrics for console output | — |

---

### 2.8 `data_loader.py` (126 lines)

**Purpose:** Data loading and tokenization (Methodology §7.2).

| Lines | Function/Class | Description | Methodology Link |
|-------|---------------|-------------|-----------------|
| 13–26 | `TokenChunkDataset` | PyTorch Dataset of fixed-length token chunks with attention masks | — |
| 29–95 | `prepare_data(tokenizer, config)` | Load WikiText-2, tokenize, create fixed-length chunks, return three DataLoaders: `eval_loader` (loss evaluation), `grad_loader` (gradient PCA), `hvp_loader` (HVP computation) | §7.2 |
| 44–48 | (within above) | Load dataset from HuggingFace | §7.2 |
| 59–61 | (within above) | Create non-overlapping fixed-length token chunks | §7.2 |
| 64–87 | (within above) | Three separate DataLoaders with different batch sizes (eval=4, grad=1, hvp=1) | §7.2, §7.3 |
| 98–125 | `prepare_custom_data(tokenizer, texts, seq_len, batch_size, max_chunks)` | Create DataLoader from custom text (for domain sensitivity, Experiment Group 5) | §7.2 |

---

### 2.9 `visualization.py` (272 lines)

**Purpose:** Plotting functions for all experiment figures.

| Lines | Function | Description |
|-------|---------|-------------|
| 17–64 | `plot_2d_surface()` | Contour + 3D surface plot for a single 2D loss surface |
| 67–86 | `plot_1d_comparison()` | Multiple 1D loss curves on one plot |
| 89–120 | `plot_tier_comparison()` | Side-by-side contour comparison of multiple tiers |
| 123–154 | `plot_pfi_comparison()` | PFI-S and PFI-C bar charts across tiers |
| 157–200 | `plot_tadn_invariance()` | Three-panel figure: loss curves, deviations, correlation bars (Experiment 1) |
| 203–240 | `plot_pca_convergence()` | Subspace angle + explained variance vs. N plots (Experiment 2) |
| 243–271 | `plot_metrics_comparison()` | Bar chart comparison of geometric metrics across experiments |

---

## 3. Configuration Files

### 3.1 `config.yaml` (Production)
- Model: Qwen3-0.6B-Base, bfloat16, eager attention
- Grid: 51×51, range=1.0, Tier 3 scale factor=3.0
- Tier 2: N=200 gradients, convergence at 5°, checkpoints [10,20,50,100,150,200]
- Tier 3: 30 iterations, 2 vectors, convergence tolerance 0.9999
- PFI: 10 Hutchinson samples
- Baseline: Layer Normalization

### 3.2 `config_verify.yaml` (Verification)
- Grid: 21×21, range=1.0
- Tier 2: N=100 gradients, checkpoints [10,20,50,75,100]
- Tier 3: 30 iterations, 3 HVP batches
- PFI: 5 Hutchinson samples

---

## 4. Data Flow Summary

```
main() [run_experiment.py]
  │
  ├─ setup_model_and_tokenizer() → model (bf16, GPU), tokenizer
  ├─ prepare_data() [data_loader.py] → eval_loader, grad_loader, hvp_loader
  ├─ get_normalization_units() [normalization.py] → units partition
  │
  ├─ run_tadn_invariance_test() [Experiment 1]
  │   ├─ create_rescaled_model() [normalization.py]
  │   ├─ apply_tadn() / apply_layer_normalization() for both models
  │   ├─ evaluate_1d_curve() [grid_evaluation.py] × 4
  │   └─ plot_tadn_invariance() [visualization.py]
  │
  ├─ run_direction_selection() [Experiments 2–4]
  │   ├─ generate_tier1_directions() [direction_selection.py] → Tier 1
  │   ├─ apply_tadn() / apply_layer_normalization() → normalized directions
  │   ├─ gradient_pca_with_convergence() [direction_selection.py] → Tier 2
  │   ├─ [swap bf16→CPU, load fp32→GPU]
  │   ├─ power_iteration_hessian() [direction_selection.py] → Tier 3
  │   ├─ compute_curvature_aware_scale() → tier3_range
  │   ├─ [move all directions to CPU]
  │   ├─ compute_hutchinson_tr_h2() [pfi.py] → shared tr(H²)
  │   ├─ compute_pfi() [pfi.py] × 4 tiers → PFI-S, PFI-C
  │   └─ [delete fp32, restore bf16 to GPU]
  │
  └─ run_surface_evaluation() [Experiment 5]
      ├─ evaluate_2d_surface() [grid_evaluation.py] × 4 configs
      ├─ compute_surface_metrics() [metrics.py] for each surface
      ├─ plot_2d_surface() / plot_tier_comparison() [visualization.py]
      └─ save .npz surface data
```

---

## 5. Key Implementation Notes

1. **Device management**: bf16 model and fp32 model cannot coexist on a single 40GB GPU for 0.6B+ models. The pipeline swaps: bf16→CPU, load fp32→GPU for HVP, then fp32→delete, bf16→GPU for grid evaluation.

2. **Direction storage**: All directions kept on CPU during PFI computation to maximize GPU memory for HVP double-backward. `grid_evaluation.py` casts directions to param device/dtype inline at each grid point.

3. **Exact parameter restoration**: `grid_evaluation.py` saves all original parameters once and restores exactly via `param.data.copy_()` at every grid point. This avoids bfloat16 drift from repeated add/subtract.

4. **Normalization on CPU**: `apply_tadn()` and `apply_layer_normalization()` always compute on CPU in float32 to ensure cross-device compatibility and numerical precision.

5. **Flash Attention incompatibility**: Tier 3 (Hessian) and PFI require `create_graph=True`, which is incompatible with Flash Attention. The fp32 model uses `attn_implementation="eager"`.

6. **Shared tr(H²)**: The Hutchinson estimate of tr(H²) is a model-level property, computed once and shared across all PFI computations (avoiding redundant HVPs).
