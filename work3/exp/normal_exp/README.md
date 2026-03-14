# LLMScape: Loss Landscape Visualization for Large Language Models

## Overview

This codebase implements the **LLMScape** framework for faithful, scalable loss landscape visualization and analysis of large language models. It includes:

1. **TADN** (Transformer-Adapted Direction Normalization) — per-component normalization invariant under transformer scale symmetries
2. **SHIDS** (Scalable Hessian-Informed Direction Selection) — three-tier direction selection: Random, Gradient PCA, Hessian eigenvectors
3. **PFI** (Projection Faithfulness Index) — quantitative metric for visualization faithfulness
4. **MMSP** (Multi-Model Shared Projection) — methods for comparing multiple models on shared 2D planes
5. **Efficient Pipeline** — curvature-aware scale selection, exact parameter restoration, mixed precision

## Directory Structure

```
exp/normal_exp/
├── config.yaml            # All hyperparameters and settings
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── run_experiment.py      # Main experiment runner
├── data_loader.py         # Data loading and tokenization
├── normalization.py       # TADN and baseline normalization methods
├── direction_selection.py # SHIDS: Tiers 1-3 direction selection
├── pfi.py                 # Projection Faithfulness Index computation
├── grid_evaluation.py     # 2D/1D loss surface evaluation
├── multi_model.py         # Multi-model projection methods (MMSP)
├── metrics.py             # Geometric feature extraction from surfaces
├── visualization.py       # Plotting functions
└── results/               # Output directory (created at runtime)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full experiment pipeline

```bash
cd exp/normal_exp
python run_experiment.py --config config.yaml --gpu 0
```

### 3. Configuration

Edit `config.yaml` to change:
- Model (default: Qwen3-0.6B-Base)
- Grid resolution (default: 51x51)
- Direction selection parameters
- PFI Hutchinson samples
- GPU assignment

### 4. Key command-line options

```bash
# Use a specific GPU
python run_experiment.py --config config.yaml --gpu 2

# Skip TADN invariance test (faster)
python run_experiment.py --config config.yaml --skip-tadn-test

# Skip Tier 3 Hessian computation (if OOM)
python run_experiment.py --config config.yaml --skip-tier3
```

## Output

Results are saved to `results/` directory:
- `results.json` — all quantitative results
- `exp1_tadn_invariance.png` — TADN vs Layer Norm comparison
- `exp2_pca_convergence.png` — Gradient PCA convergence analysis
- `exp4_pfi_comparison.png` — PFI across tiers
- `surface_*.png` — 2D loss landscape plots
- `surface_*.npz` — raw surface data (numpy)
- `tier_comparison.png` — side-by-side tier comparison
- `metrics_comparison.png` — geometric metrics comparison

## Hardware Requirements

- **0.6B model**: 1x A100-40GB (or equivalent with >=16GB VRAM)
- **1.1B model**: 1x A100-40GB
- **7B model**: 1-2x A100-40GB (bfloat16 for grid, fp32 for HVP)

**Important notes:**
- Tier 3 (Hessian) requires `attn_implementation="eager"` (Flash Attention incompatible with `create_graph=True`)
- fp32 model and bf16 model cannot coexist on a single 40GB GPU for 0.6B+ models
- Use `batch_size=1` for HVP computation to manage memory
