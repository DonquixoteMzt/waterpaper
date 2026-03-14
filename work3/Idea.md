# Idea: Visualizing the Loss Landscape of Large Language Models

## 1. Problem Statement and Motivation

### 1.1 What Problem Does This Work Address?

Understanding the geometric structure of loss landscapes is fundamental to explaining the optimization dynamics, generalization behavior, and robustness of neural networks. For large language models (LLMs), loss landscape analysis is especially critical because:

1. **Training dynamics are poorly understood.** LLM pre-training involves trillions of tokens and billions of parameters, yet we lack geometric understanding of how the loss surface evolves during training—whether it smooths out, how basins form, and how curvature changes across training stages.

2. **Fine-tuning fragility is unexplained.** LLMs exhibit surprising fragility: adversarial fine-tuning with as few as a hundred examples can destroy all capabilities (Chen et al., 2025), and the geometric reason for this vulnerability remains unclear without proper visualization.

3. **Model comparison lacks geometric tools.** Different LLM families (e.g., Qwen, OLMo, Llama) are compared solely by benchmark scores, but their loss landscape geometries—which determine robustness, fine-tunability, and generalization—have never been systematically compared.

4. **Existing visualization methods are inadequate for LLMs.** The seminal filter normalization method (Li et al., 2018) was designed for small CNNs. No existing work has systematically adapted loss landscape visualization to the unique architecture and scale of transformer-based LLMs.

5. **No quantitative metric exists for visualization quality.** Prior work on loss landscape visualization provides qualitative plots but no way to assess whether a given 2D projection faithfully represents the true high-dimensional geometry.

### 1.2 Why Is This Important?

Loss landscape visualization provides interpretable, geometric insights that scalar metrics (loss, accuracy, Hessian trace) cannot capture:

- **Visual landscapes reveal global structure** (basin shapes, barrier heights, connectivity) that scalar statistics miss.
- **Geometric understanding enables better training recipes** — flatter minima generalize better (Liu et al., 2023; Foret et al., 2021), and understanding how training choices shape the landscape can guide optimizer and hyperparameter selection.
- **Multi-model landscape comparison** can reveal why some LLMs are more robust, more fine-tunable, or more capable than others, complementing benchmark-driven evaluation.
- **Faithfulness metrics** allow practitioners to assess whether their visualizations are trustworthy, replacing subjective interpretation with quantitative quality assessment.

### 1.3 Essential Limitations of Existing Methods

| Limitation | Existing Work | Impact |
|-----------|--------------|--------|
| **Scale mismatch** | Li et al. (2018): demonstrated on VGG/ResNet (~10M params) on CIFAR-10 | All methods untested on billion-parameter transformers |
| **Architecture mismatch** | Filter normalization designed for CNN convolutional filters | Transformers have attention heads, FFN neurons, embeddings, LayerNorm — no principled "filter" definition exists |
| **Unfaithful projections** | Random direction projections (standard approach) | Böttcher & Wheeler (2024) proved random projections map to mean curvature; saddle points appear as minima |
| **No multi-model framework** | Garipov et al. (2018): affine combination of 3 small models | Requires same architecture/initialization; no normalization; infeasible for comparing different LLM families |
| **No faithfulness quantification** | All prior work | No metric exists to assess how well a 2D projection represents the true geometry; visual inspection is the only tool |
| **Computational infeasibility** | Hessian eigenvector computation for direction selection | Full Hessian impossible for LLMs; existing HVP methods limited at scale |
| **Scalar-only analysis for LLMs** | Chen et al. (2025), Liu et al. (2023), Kalra et al. (2026) | Study LLM landscape via scalar metrics but produce no 2D visualizations |

---

## 2. Formal Problem Definition

### 2.1 Loss Landscape Visualization Problem

Given a pre-trained language model with parameters $\theta^* \in \mathbb{R}^d$ (where $d$ can be $10^8$ to $10^{10}$) and a loss function $L(\theta) = \mathbb{E}_{x \sim \mathcal{D}}[\ell(\theta; x)]$ over dataset $\mathcal{D}$, the **loss landscape visualization problem** is to construct a 2D function:

$$f(\alpha, \beta) = L(\theta^* + \alpha \cdot \mathbf{d}_1 + \beta \cdot \mathbf{d}_2)$$

where $\mathbf{d}_1, \mathbf{d}_2 \in \mathbb{R}^d$ are projection direction vectors, such that the resulting surface plot $f(\alpha, \beta)$ **faithfully represents** the local geometric structure of the original high-dimensional loss landscape around $\theta^*$.

### 2.2 Faithfulness Criterion

We define a 2D visualization as **faithful** if the principal curvatures observed in the 2D surface plot are proportional to the corresponding principal curvatures in the original high-dimensional loss landscape along the chosen directions. Specifically, let $\kappa_1, \kappa_2$ denote the curvatures of $f(\alpha, \beta)$ at $(\alpha, \beta) = (0, 0)$ along the two axis directions. The visualization is faithful if:

$$\kappa_i \propto \mathbf{d}_i^\top H \mathbf{d}_i, \quad i = 1, 2$$

where $H = \nabla^2 L(\theta^*)$ is the Hessian. We formalize this through the **Projection Faithfulness Index (PFI)**, which quantifies the fraction of the Hessian's spectral energy captured by the 2D projection:

$$\text{PFI-S}(\hat{\mathbf{d}}_1, \hat{\mathbf{d}}_2) = \frac{\|H \hat{\mathbf{d}}_1\|^2 + \|H \hat{\mathbf{d}}_2\|^2}{\text{tr}(H^2)}$$

### 2.3 Four Core Sub-Problems

**Sub-problem 1 (Direction Normalization):** How should the direction vectors $\mathbf{d}_1, \mathbf{d}_2$ be normalized to remove scale-invariance artifacts specific to transformer architectures, enabling meaningful comparison across models?

**Sub-problem 2 (Direction Selection):** How should $\mathbf{d}_1, \mathbf{d}_2$ be chosen to maximize the faithfulness of the 2D projection, while remaining computationally feasible for billion-parameter models?

**Sub-problem 3 (Faithfulness Assessment):** How can we quantitatively assess whether a given 2D projection faithfully represents the true high-dimensional geometry?

**Sub-problem 4 (Multi-Model Projection):** How can multiple model checkpoints be projected onto a shared 2D coordinate system for geometric comparison?

---

## 3. Proposed Approach

We propose **LLMScape**, a comprehensive framework for faithful, scalable loss landscape visualization and analysis of large language models. LLMScape addresses the four core sub-problems through five technical contributions and one comprehensive empirical study.

### 3.1 Contribution 1: Transformer-Adapted Direction Normalization (TADN)

**Motivation:** Li et al. (2018) introduced filter normalization for CNNs, but transformers have fundamentally different parameter structures. Moreover, transformers exhibit neuron-level scale invariance in FFN layers (scaling $W_{\text{up}}$ rows by $c_j$ and $W_{\text{down}}$ columns by $1/c_j$ preserves the function), and the Sharpness Disparity Principle (arXiv:2502.19002) demonstrates that different transformer blocks exhibit dramatically different sharpness.

**Proposed Method:** We define normalization units at the granularity of individual functional components: per-head for attention, per-neuron for FFN, per-token for embeddings. Each unit's perturbation direction is scaled to match the parameter norm of that unit.

**Key Theoretical Result:** TADN is **provably invariant under non-uniform FFN neuron rescaling**: when a model is reparameterized by scaling individual neurons differently (a true symmetry of the network), TADN produces identical loss landscapes for both parameterizations. Layer-level normalization does **not** have this property. (See Methodology.md, Proposition 3.1 and 3.2.)

**Empirical Validation:** PoC experiments demonstrate TADN achieves near-perfect correlation (>0.999) between landscapes of equivalent parameterizations, while layer normalization shows significantly lower correlation. This advantage is critical when comparing models at different training stages (where internal scales shift) or across model families.

### 3.2 Contribution 2: Scalable Hessian-Informed Direction Selection (SHIDS)

**Motivation:** Böttcher & Wheeler (2024) proved that random projections collapse curvature to the mean, losing all directional information.

**Proposed Method: Three-Tier Framework with Convergence Guarantees**

- **Tier 1 — Random Directions + TADN (Baseline):** Negligible cost; captures only mean curvature. PFI-S ≈ 2/d ≈ 0.
- **Tier 2 — Gradient Covariance PCA with Adaptive Sample-Size:** Novel practical middle ground. Collects per-batch gradients and finds top principal components of the gradient covariance (which approximate top Hessian eigenvectors per Gur-Ari et al., 2018). **New: includes adaptive stopping criterion based on principal angle convergence**, so the user need not choose the number of gradient samples manually. We provide convergence analysis based on the Davis-Kahan theorem, showing convergence rate depends on the spectral gap.
- **Tier 3 — Power Iteration for Hessian Eigenvectors (Most Faithful):** Uses HVPs via the Pearlmutter trick to compute exact top Hessian eigenvectors. Includes curvature-aware scale selection: the characteristic length $\ell_{\text{char}} = 1/\sqrt{|\lambda_{\max}|}$ automatically determines the appropriate visualization range.

### 3.3 Contribution 3: Projection Faithfulness Index (PFI)

**This is a novel, distinctive contribution with no prior analogue.** PFI provides a theoretically grounded, efficiently computable metric that answers the question: "How much of the high-dimensional loss landscape's curvature information is captured by this 2D visualization?"

**PFI-S (Spectral Coverage):** Measures the fraction of the Hessian's spectral energy (i.e., $\text{tr}(H^2)$) captured in the 2D subspace:
$$\text{PFI-S} = \frac{\|H\hat{\mathbf{d}}_1\|^2 + \|H\hat{\mathbf{d}}_2\|^2}{\text{tr}(H^2)}$$

**PFI-C (Curvature Capture):** Measures alignment with the direction of maximum curvature:
$$\text{PFI-C} = \frac{\max_i |\hat{\mathbf{d}}_i^\top H \hat{\mathbf{d}}_i|}{|\lambda_1|}$$

**Key Properties:**
1. PFI-S ∈ [0, 1], maximized by top-2 Hessian eigenvectors.
2. For random directions: E[PFI-S] = 2/d → 0 (provably unfaithful).
3. Efficiently computable: 2 HVPs (numerator) + O(10) HVPs (Hutchinson trace for denominator).
4. Enables principled comparison of visualization methods: Tier 3 > Tier 2 > Tier 1.
5. The denominator (tr(H²)) is shared across all tiers, making comparison cheap.

### 3.4 Contribution 4: Multi-Model Shared Projection (MMSP)

**Three Methods for Different Scenarios:**

- **Method A — Trajectory-PCA:** For checkpoints of the same training run. PCA of parameter differences → 2D plane capturing most trajectory variance.
- **Method B — Anchor-Point:** For 2–3 models with same architecture. Uses the inter-model direction as one axis.
- **Method C — Independent Comparison:** For different architectures. Independent visualizations with matched evaluation data and extracted geometric features.

### 3.5 Contribution 5: Efficient Computation Pipeline

**Key optimizations:**
- **Curvature-aware adaptive scale selection:** Uses $\lambda_{\max}$ from Tier 3 to automatically set the grid range to $[-3/\sqrt{|\lambda_{\max}|}, 3/\sqrt{|\lambda_{\max}|}]$, ensuring the visualization always captures the "interesting" region.
- **Exact parameter restoration** (critical for bfloat16): Save and restore parameters exactly at each grid point, rather than add/subtract which accumulates rounding errors.
- **Mixed-precision:** bfloat16 for grid evaluation, float32 for HVPs.
- **Parallelization:** Grid points are independent; trivially distributable across GPUs.

---

## 4. Innovation Points and Comparison with Prior Work

| Aspect | Prior Work | Our Contribution |
|--------|-----------|-----------------|
| **Normalization** | Li et al. (2018): CNN filter normalization | **TADN**: Provably invariant under transformer-specific scale symmetries; layer norm fails |
| **Direction Selection** | Random (Li et al.), Hessian for small models (Böttcher & Wheeler) | **SHIDS**: Three-tier framework with adaptive convergence for Tier 2; first at LLM scale |
| **Faithfulness Metric** | None | **PFI**: First theoretically grounded metric for visualization quality assessment |
| **Scale Selection** | Arbitrary grid range (manual) | **Curvature-aware**: Automatic scale from Hessian eigenvalues |
| **Multi-Model Visualization** | Garipov et al.: affine combination of 3 small models | **MMSP**: Trajectory-PCA with TADN; systematic multi-scenario framework |
| **Scale** | All prior visualization ≤10M parameters | First comprehensive 2D visualization up to 7B parameters |
| **LLM-Specific Insights** | 1D basin analysis (Chen et al.); scalar curvature (Kalra et al.) | First 2D visualizations of training evolution, post-training effects, cross-model differences |

---

## 5. Originality Assessment

**High originality:**
- The Projection Faithfulness Index (PFI) is the first metric that quantifies how faithfully a 2D projection represents a high-dimensional loss landscape. This enables principled comparison of visualization methods and replaces subjective visual inspection.
- TADN's provable invariance under FFN neuron rescaling is a novel theoretical result with clear empirical demonstration. Unlike layer normalization, TADN correctly handles the fine-grained scale symmetries of transformer architectures.
- The gradient covariance PCA with adaptive sample-size selection and convergence guarantees is a novel practical contribution for LLM-scale direction selection.
- Curvature-aware scale selection using Hessian eigenvalues to automatically determine visualization range is new for loss landscape visualization.

**Building on solid foundations:**
- Filter normalization (Li et al., 2018) provides the theoretical basis for TADN.
- Hessian direction theory (Böttcher & Wheeler, 2024) motivates PFI and the direction selection hierarchy.
- Gradient-Hessian eigenspace alignment (Gur-Ari et al., 2018) justifies Tier 2.
- Mode connectivity (Garipov et al., 2018) inspires multi-model projection.
- LLM landscape analysis (Chen et al., 2025; Liu et al., 2023; Kalra et al., 2026) provides context.

---

## 6. Feasibility Assessment

### 6.1 Computational Feasibility

| Task | Model Size | Estimated Cost | Feasibility |
|------|-----------|---------------|-------------|
| 2D surface (51×51) | 0.6B | ~2 min (1 GPU) | Very High |
| 2D surface (51×51) | 1.1B | ~9 min (1 GPU) | Very High |
| 2D surface (51×51) | 7B | ~22 min (1 GPU) | High |
| Tier 2 directions (N=100) | 0.6B | ~10s | Very High |
| Tier 3 directions (30 iter) | 0.6B | ~25s | Very High |
| PFI computation (10 Hutchinson) | 0.6B | ~6s | Very High |
| Full pipeline (directions + surface + PFI) | 7B | ~30 min | High |

### 6.2 Data and Model Availability

All models and datasets are publicly available on HuggingFace.

### 6.3 Risk Assessment

| Risk | Severity | Mitigation |
|------|---------|-----------|
| HVP incompatible with Flash Attention | Medium | Use eager attention for HVP only |
| 7B OOM during HVP | Medium | Gradient checkpointing; mixed-precision |
| Landscape not visually informative | Low | Multiple tiers ensure at least one informative view; PFI quantifies this |
| PFI dominated by noise | Low | Hutchinson estimation with 10+ samples; verified convergence in PoC |

---

## 7. Expected Impact and Contributions

### 7.1 Methodological Contributions
1. **TADN**: Principled transformer normalization with provable invariance.
2. **SHIDS**: Practical hierarchy with convergence guarantees for adaptive sample-size.
3. **PFI**: First quantitative metric for visualization faithfulness.
4. **MMSP**: Multi-model comparison framework.
5. **Curvature-aware pipeline**: Automatic scale selection and efficient evaluation.

### 7.2 Empirical Insights (Expected)
1. How the loss landscape evolves during LLM pre-training.
2. How post-training/alignment reshapes the landscape.
3. Geometric signatures distinguishing different LLM families.
4. The information loss from random vs. curvature-informed projections, quantified by PFI.

### 7.3 Broader Impact
This work provides the first toolkit for using 2D loss landscape visualization as a standard analysis tool for LLM development, complementing the benchmark-driven evaluation paradigm with quantitative geometric understanding and trustworthy visualization.

---

## 8. Proposed Paper Structure

Following the structure of Li et al. (2018), adapted for NeurIPS 2025:

1. **Abstract**
2. **Introduction** — Motivation, challenges, five contributions
3. **Related Work** — Landscape visualization, Hessian methods, LLM analysis
4. **Method**
   - 4.1 TADN (normalization)
   - 4.2 SHIDS (direction selection)
   - 4.3 PFI (faithfulness metric)
   - 4.4 MMSP (multi-model projection)
   - 4.5 Pipeline (curvature-aware scale, implementation)
5. **Experiments**
   - 5.1 Methodology Validation (TADN invariance, PFI comparison, convergence)
   - 5.2 Training Trajectory Visualization (TinyLlama)
   - 5.3 Post-Training Effect (Qwen3-0.6B)
   - 5.4 Cross-Model Comparison (OLMo vs. Qwen)
   - 5.5 Dataset Sensitivity
6. **Discussion** — Insights, limitations, flat minima theory
7. **Conclusion**

---

## 9. Cross-Reference to Detailed Methodology

See `Methodology.md` for:
- **TADN** (Section 3): Normalization unit partition, Algorithm 1, Propositions 3.1–3.2.
- **SHIDS** (Section 4): Algorithms 2–3, Theorem 4.1, convergence analysis.
- **PFI** (Section 5): Definitions 5.1–5.2, Theorem 5.1, Algorithm 4.
- **MMSP** (Section 6): Methods A–C.
- **Pipeline** (Section 7): Scale selection, mixed-precision, parallelization.

---

## 10. Key References

- Li et al. (2018). Visualizing the Loss Landscape of Neural Nets. NeurIPS 2018.
- Chen et al. (2025). Unveiling the Basin-Like Loss Landscape in LLMs. arXiv:2505.17646.
- Böttcher & Wheeler (2024). Visualizing High-Dimensional Loss Landscapes with Hessian Directions. JSTAT 2024.
- Garipov et al. (2018). Loss Surfaces, Mode Connectivity, and Fast Ensembling. NeurIPS 2018.
- Liu et al. (2023). Same Pre-training Loss, Better Downstream. ICML 2023.
- Foret et al. (2021). Sharpness-Aware Minimization. ICLR 2021.
- Ghorbani et al. (2019). Investigation into Neural Net Optimization via Hessian Eigenvalue Density. ICML 2019.
- Kalra et al. (2026). A Scalable Measure of Loss Landscape Curvature. arXiv:2601.16979.
- Xia et al. (2023). Training Trajectories of Language Models Across Scales. ACL 2023.
- Zhang et al. (2024). Why Transformers Need Adam: A Hessian Perspective. NeurIPS 2024.
- Gur-Ari, Roberts & Dyer (2018). Gradient Descent Happens in a Tiny Subspace. arXiv:1812.04754.
- Kunstner et al. (2019). Limitations of the Empirical Fisher Approximation. NeurIPS 2019.
- HessFormer (2025). Hessians at Foundation Scale. arXiv:2505.11564.
- Sharpness Disparity Principle in Transformers (2025). arXiv:2502.19002.
