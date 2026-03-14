# Paper Outline: LLMScape — Faithful, Scalable Loss Landscape Visualization for Large Language Models

## Target Venue: NeurIPS 2025 (Main Track)
## Reference Style: Following "Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)

---

## Abstract
- Problem: Loss landscape visualization is fundamental for understanding optimization, generalization, and robustness of neural networks, but existing methods (designed for small CNNs) are inadequate for transformer-based LLMs due to scale mismatch, architecture mismatch, unfaithful random projections, and lack of multi-model comparison tools.
- Contribution: We propose **LLMScape**, a framework with five contributions: (1) TADN — provably scale-invariant normalization for transformers; (2) SHIDS — three-tier direction selection from random to Hessian eigenvectors; (3) PFI — first quantitative metric for visualization faithfulness; (4) MMSP — multi-model shared projection methods; (5) efficient pipeline with curvature-aware scale selection.
- Results: TADN achieves perfect invariance (correlation=1.000) under neuron rescaling vs. 0.918 for prior methods; PFI quantifies a 41,400× improvement from random to Hessian directions; framework scales to 7B parameters (700× prior work); RLHF alignment narrows the loss basin by 23.3%, providing geometric evidence for alignment fragility.
- Impact: First comprehensive 2D loss landscape visualization toolkit for LLMs, establishing quantitative geometric baselines for models from 0.6B to 7B parameters.

---

## 1. Introduction (~1.5 pages)
- **Opening**: Loss landscape geometry determines optimization dynamics, generalization, and robustness of neural networks. Visualizing loss landscapes has provided fundamental insights for CNNs.
- **Gap**: No existing work systematically visualizes loss landscapes of LLMs. Five key challenges: (1) filter normalization designed for CNNs, not transformers; (2) random projections provably lose curvature information; (3) no metric for visualization quality; (4) no multi-model comparison framework; (5) computational infeasibility at LLM scale.
- **Contributions** (numbered list of 5):
  1. TADN: Transformer-adapted normalization with provable invariance
  2. SHIDS: Three-tier direction selection hierarchy
  3. PFI: Projection Faithfulness Index
  4. MMSP: Multi-Model Shared Projection
  5. Comprehensive empirical study: training trajectories, post-training effects, cross-model comparison
- **Key results**: Highlight TADN invariance, PFI hierarchy, 7B scalability, RLHF basin narrowing
- **Novelty emphasis**: PFI is the first quantitative faithfulness metric; TADN is provably invariant under transformer symmetries; first 2D visualization at 7B scale

---

## 2. Related Work (~1 page)
- **2.1 Loss Landscape Visualization**: Li et al. (2018) filter normalization; Goodfellow et al. (2015) linear interpolation; Xu et al. (2024) mining-based approaches. Limitation: all demonstrated on small CNNs.
- **2.2 Hessian-Based Analysis**: Böttcher & Wheeler (2024) Hessian directions; Ghorbani et al. (2019) spectral density; Foret et al. (2021) SAM and flatness. Connection: motivates our direction selection hierarchy.
- **2.3 Loss Landscape of LLMs**: Chen et al. (2025) basin structure; Liu et al. (2023) Hessian trace and downstream performance; Kalra et al. (2026) critical sharpness at scale. Gap: scalar metrics only, no 2D visualization.
- **2.4 Mode Connectivity and Multi-Model Analysis**: Garipov et al. (2018) mode connectivity; Xia et al. (2023) training trajectories across scales. Gap: no framework for LLM multi-model comparison.

---

## 3. Method (~3.5 pages)

### 3.1 Preliminaries and Problem Formulation
- Loss landscape visualization as 2D projection: $f(\alpha, \beta) = L(\theta^* + \alpha \mathbf{d}_1 + \beta \mathbf{d}_2)$
- Four sub-problems: normalization, direction selection, faithfulness assessment, multi-model projection
- Transformer architecture notation

### 3.2 Transformer-Adapted Direction Normalization (TADN)
- **Motivation**: Scale invariance in transformers (FFN neuron rescaling, RMSNorm)
- **Definition**: Normalization unit partition (per-head attention, per-neuron FFN, per-token embedding) — Table 1
- **Algorithm 1**: TADN normalization procedure
- **Proposition 1**: Provable invariance under FFN neuron rescaling (with proof sketch)
- **Proposition 2**: Layer normalization fails under non-uniform rescaling

### 3.3 Scalable Hessian-Informed Direction Selection (SHIDS)
- **Theorem 1**: Curvature collapse under random projection (Böttcher & Wheeler, 2024)
- **Tier 1**: Random + TADN (baseline, negligible cost)
- **Tier 2**: Gradient Covariance PCA with adaptive stopping — Algorithm 2
- **Tier 3**: Power iteration for Hessian eigenvectors — Algorithm 3
- **Curvature-aware scale selection**: $\ell_{\text{char}} = 1/\sqrt{|\lambda_{\max}|}$

### 3.4 Projection Faithfulness Index (PFI)
- **Definition**: PFI-S (spectral coverage) and PFI-C (curvature capture)
- **Theorem 2**: PFI bounds and extrema (PFI-S ∈ [0,1], maximized by top Hessian eigenvectors, E[PFI-S] = 2/d for random)
- **Efficient computation**: 2+m HVPs via Hutchinson trace estimation — Algorithm 4

### 3.5 Multi-Model Shared Projection (MMSP)
- **Method A**: Trajectory-PCA for checkpoints from same training run
- **Method B**: Anchor-Point projection for same-architecture models
- **Method C**: Independent comparison for different architectures

### 3.6 Implementation Details
- Exact parameter restoration for bfloat16
- Mixed precision: bfloat16 for grid evaluation, float32 for HVPs
- Flash Attention compatibility (eager attention for HVP only)

---

## 4. Experiments (~3 pages)

### 4.1 Experimental Setup
- Models: Qwen3-0.6B-Base, Qwen3-0.6B (post-trained), Qwen2.5-7B-Instruct, OLMo-3-7B-Think
- Datasets: WikiText-2 (primary), synthetic code, structured data
- Hardware: 8× NVIDIA A100-40GB
- Metrics: Loss range, roughness, basin diameter, curvature ratio, convexity ratio, PFI-S, PFI-C

### 4.2 Method Validation
- **TADN invariance test**: Correlation=1.000 vs LayerNorm=0.918 under neuron rescaling — Figure 2
- **PFI hierarchy**: Tier 3 (1.90×10⁻⁴) >> Tier 2 (3.91×10⁻⁵) >> Tier 1 (4.58×10⁻⁹) — Table 2
- **2D surface comparison**: Three tiers reveal qualitatively different features — Figure 3
- **Multi-seed consistency**: CV < 1% for loss range — Table 3
- **Normalization ablation**: TADN-full captures 1.86× more loss range than block-level — Table 4

### 4.3 Training Trajectory Visualization
- 500-step fine-tuning with 11 checkpoints using MMSP Method A
- Trajectory-PCA captures 87.6% variance (PC1=76.4%)
- Curved trajectory with decelerating dynamics — Figure 4
- No loss barrier between base and fine-tuned models (MMSP Method B)

### 4.4 Post-Training Effect Analysis
- **Controlled fine-tuning**: Landscape geometry stable through training (roughness change < 1%)
- **Official RLHF comparison**: Basin diameter narrows 23.3%, qualitatively different from fine-tuning
- Fine-tuning: narrow valley (κ ratio 66:1); RLHF: broad modification (κ ratio 3:1) — Figure 5, Table 5

### 4.5 Cross-Model Comparison
- Qwen2.5-7B-Instruct vs OLMo-3-7B-Think using MMSP Method C
- Different training objectives produce distinct geometries — Figure 6, Table 6
- Basin diameter decreases with model size: 0.310 (0.6B) → 0.252 (7B Qwen) → 0.226 (7B OLMo)

### 4.6 Dataset Sensitivity and Robustness
- Cross-domain stability: loss range CV = 2.0% across 5 datasets
- Grid resolution independence: loss range identical at 11×11 through 51×51
- Minimum data: 512 tokens sufficient for 2% accuracy

---

## 5. Discussion (~0.5 pages)
- PFI as a standard for visualization quality assessment
- Geometric signatures of RLHF alignment: narrower basins explain fragility
- Scalability boundary: Tier 2/3 limited to ~1B on 40GB hardware; Tier 1 scales to 7B+
- Connection to flat minima theory and generalization

---

## 6. Conclusion (~0.5 pages)
- Summary: LLMScape provides the first comprehensive framework for faithful loss landscape visualization of LLMs
- Key quantitative findings: TADN invariance (1.000 vs 0.918), PFI hierarchy (41,400×), 7B scalability, RLHF basin narrowing (23.3%)
- Future work: memory-efficient Tier 2/3 for larger models, Flash Attention compatible HVP, PFI-guided direction optimization

---

## Appendix (Supplementary Material)
- A. Full proofs of Propositions and Theorems
- B. Additional experimental details and hyperparameters
- C. Additional ablation results (grid resolution, evaluation data size)
- D. Full experimental result tables

---

## Figures Plan

| Figure | Type | Content | Source |
|--------|------|---------|--------|
| Figure 1 | Method diagram | LLMScape framework overview: TADN → SHIDS → PFI → MMSP pipeline | Python (matplotlib) |
| Figure 2 | Experimental | TADN invariance: 1D cross-sections showing TADN vs LayerNorm | Python from exp data |
| Figure 3 | Experimental | 2D loss surface comparison: Tier 1 vs Tier 2 vs Tier 3 | Python from exp data |
| Figure 4 | Experimental | Training trajectory: PCA projection with checkpoint path + loss evolution | Python from exp data |
| Figure 5 | Experimental | Post-training effects: base vs RLHF anchor-point + independent surfaces | Python from exp data |
| Figure 6 | Experimental | Cross-model comparison: Qwen 7B vs OLMo 7B side-by-side | Python from exp data |

## Tables Plan

| Table | Content |
|-------|---------|
| Table 1 | TADN normalization unit partition |
| Table 2 | PFI comparison across three tiers |
| Table 3 | 2D loss surface metrics comparison across tiers and normalization methods |
| Table 4 | TADN granularity ablation results |
| Table 5 | Fine-tuning vs RLHF geometric signature comparison |
| Table 6 | Cross-model landscape metrics (0.6B, 7B Qwen, 7B OLMo) |
| Table 7 | Dataset sensitivity analysis |
