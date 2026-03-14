# Related Work: Loss Landscape Visualization and Analysis for Large Language Models

## 1. Paper Verification Summary

All candidate papers were programmatically verified using the arXiv API and Semantic Scholar API. For each paper, the title, authors, and venue were cross-checked against the API responses. The verification script is available in `verify_papers.py`, and detailed results are stored in `verification_results.json`.

| # | Paper | Venue | arXiv ID | Verification | Source | Citations |
|---|-------|-------|----------|-------------|--------|-----------|
| 1 | Visualizing the Loss Landscape of Neural Nets (Li et al., 2018) | NeurIPS 2018 | 1712.09913 | **Confirmed** | arXiv + Semantic Scholar | ~2215 |
| 2 | Unveiling the Basin-Like Loss Landscape in Large Language Models (Chen et al., 2025) | arXiv preprint | 2505.17646 | **Confirmed** | arXiv + Semantic Scholar | ~5 |
| 3 | Visualizing High-Dimensional Loss Landscapes with Hessian Directions (Böttcher & Wheeler, 2024) | JSTAT 2024 | 2208.13219 | **Confirmed** | arXiv + Semantic Scholar | ~20 |
| 4 | Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs (Garipov et al., 2018) | NeurIPS 2018 | 1802.10026 | **Confirmed** | arXiv + Semantic Scholar | ~881 |
| 5 | Same Pre-training Loss, Better Downstream (Liu et al., 2023) | ICML 2023 (Oral) | 2210.14199 | **Confirmed** | arXiv + Semantic Scholar | ~72 |
| 6 | Sharpness-Aware Minimization for Efficiently Improving Generalization (Foret et al., 2021) | ICLR 2021 | 2010.01412 | **Confirmed** | arXiv + Semantic Scholar | ~1767 |
| 7 | An Investigation into Neural Net Optimization via Hessian Eigenvalue Density (Ghorbani et al., 2019) | ICML 2019 | 1901.10159 | **Confirmed** | arXiv + Semantic Scholar | ~395 |
| 8 | Visualizing, Rethinking, and Mining the Loss Landscape of Deep Neural Networks (Xu et al., 2024) | arXiv preprint | 2405.12493 | **Confirmed** | arXiv | est. <20 |
| 9 | A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs (Kalra et al., 2026) | arXiv preprint | 2601.16979 | **Confirmed** | arXiv + Semantic Scholar | ~1 |
| 10 | Training Trajectories of Language Models Across Scales (Xia et al., 2023) | ACL 2023 | 2212.09803 | **Confirmed** | arXiv + Semantic Scholar | ~72 |

**Additional verified papers consulted but not in the top 10:**

| Paper | Venue | arXiv ID | Verification | Citations |
|-------|-------|----------|-------------|-----------|
| Dissecting Hessian (Wu et al., 2020) | arXiv | 2010.04261 | Confirmed | ~50 |
| Qualitatively Characterizing Neural Network Optimization Problems (Goodfellow et al., 2015) | ICLR 2015 | 1412.6544 | Confirmed | ~1000+ |
| Loss Landscape Degeneracy and Stagewise Development in Transformers (Hoogland et al., 2024) | TMLR 2024 | 2402.02364 | Confirmed | ~22 |

---

## 2. Ranking Methodology

Papers were ranked considering:
- **Relevance** to the core research topic: loss landscape visualization and analysis for large language models
- **Authority**: publication venue quality (NeurIPS, ICML, ICLR, ACL > journals > arXiv preprints) and citation count
- **Technical alignment**: match to the method hints in the research plan (Hessian computation, filter normalization, multi-model projection, weight normalization, etc.)
- **Recency**: newer works addressing LLM-specific challenges were given additional weight

---

## 3. In-Depth Analysis of Top 5 Papers

### 3.1 [Rank 1] Visualizing the Loss Landscape of Neural Nets (Li et al., NeurIPS 2018)

**Full Citation:** Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein. "Visualizing the Loss Landscape of Neural Nets." *Advances in Neural Information Processing Systems 31 (NeurIPS)*, 2018. arXiv:1712.09913.

**Problem Addressed:** Neural network loss functions live in extremely high-dimensional parameter spaces, making direct visualization impossible. Prior visualization methods failed to provide meaningful comparisons across different network architectures and training configurations due to scale invariance issues.

**Core Contributions:**

1. **Filter Normalization Method:** The key methodological contribution. Given network parameters θ and a random Gaussian direction vector d, they normalize each "filter" (i.e., the parameters contributing to a single feature map or neuron) in d to match the norm of the corresponding filter in θ. This removes the confounding effects of scale invariance, which Dinh et al. had exploited to build pairs of equivalent networks with different apparent sharpness. The normalization is:
   $$d_i \leftarrow \frac{d_i}{\|d_i\|} \cdot \|\theta_i\|$$
   where $d_i$ and $\theta_i$ are the i-th filter of the direction and parameters, respectively.

2. **1D and 2D Visualization:** For 1D plots, they evaluate $f(\alpha) = L(\theta^* + \alpha \cdot d)$ along a single normalized random direction. For 2D surface plots, they evaluate $f(\alpha, \beta) = L(\theta^* + \alpha \cdot d_1 + \beta \cdot d_2)$ using two orthogonal normalized random directions. They also use PCA on the optimization trajectory to select more informative projection directions.

3. **Hessian Eigenvalue Analysis:** For each point on the 2D surface, they compute the maximum and minimum eigenvalues of the Hessian to quantify convexity. They map the ratio of these eigenvalues to characterize local curvature.

**Key Findings:**
- Skip connections produce a dramatic "convexification" of the loss landscape. Without residual connections, deep networks exhibit highly chaotic, non-convex loss surfaces.
- Smaller batch sizes produce wider (flatter) minima with lower error rates, consistent with the flat minima hypothesis.
- Wider networks have smoother loss landscapes with less chaotic behavior.
- Filter normalization reveals the "true" sharpness of minimizers, which correlates well with generalization error.

**Relevance to Our Research:**
This is the foundational paper for our work. The filter normalization method is the starting point for visualization. However, several open questions remain for LLMs: (a) filter normalization was designed for CNNs—how should it be adapted for transformer architectures with attention layers, layer norms, and embedding tables? (b) The paper uses random directions or PCA of the training trajectory—but for LLMs, Hessian-based directions may be more informative (see Paper #3). (c) The paper does not address multi-model comparison on a shared 2D plane. (d) The computational cost of evaluating loss surfaces for billion-parameter models is prohibitive with naive approaches.

**Limitations:**
- The method was only demonstrated on relatively small CNNs (VGG, ResNet on CIFAR-10/ImageNet).
- Random direction projections can miss important geometric features (saddle points appear as minima).
- The relationship between 2D visualizations and the true high-dimensional geometry is not rigorously established.

---

### 3.2 [Rank 2] Unveiling the Basin-Like Loss Landscape in Large Language Models (Chen et al., 2025)

**Full Citation:** Huanran Chen, Yinpeng Dong, Zeming Wei, Yao Huang, Yichi Zhang, et al. "Unveiling the Basin-Like Loss Landscape in Large Language Models." arXiv:2505.17646, 2025.

**Problem Addressed:** Despite the rapid development of LLMs, the geometric structure of their loss landscapes remains poorly understood. Key questions include: Why does fine-tuning with adversarial data, even for just a few steps, destroy all capabilities of LLMs? Why are LLMs easily jailbroken in white-box settings?

**Core Contributions:**

1. **Basin-Like Structure Discovery:** The loss landscape of LLMs exhibits a basin-like structure—within the basin, models perform nearly identically; outside, capabilities collapse rapidly. Pre-training creates a "basic capability" basin, and subsequent alignment fine-tuning creates "specific capability" basins (e.g., safety, math, coding) nested within the basic capability basin.

2. **Most-Case vs. Worst-Case Landscape:** Two complementary perspectives are analyzed:
   - *Most-case landscape*: Measures capability degradation when parameters are perturbed along randomly sampled directions. Basin sizes are large, explaining why benign fine-tuning within the most-case basin preserves previous capabilities.
   - *Worst-case landscape*: Measures degradation along the single most adversarial direction. Basins are much smaller, and even small perturbations along worst-case directions rapidly degrade all capabilities.

3. **Theoretical Framework:** Using randomized smoothing theory, they bound performance degradation:
   - Weak guarantee: degradation ≤ $\frac{1}{\sqrt{2\pi}\sigma} \cdot \|\theta_{\text{sft}} - \theta_0\|_2$
   - Larger basin size ($\sigma$) directly translates to greater robustness against fine-tuning attacks.

4. **Gaussian-Augmented Optimizer (GO):** Proposes optimizing the expected loss over Gaussian-perturbed parameters: $L_{\text{train}}(x, \theta) = -\mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma^2 I)}[\log p(x|\theta + \epsilon)]$. Pre-training experiments on GPT2-127M show GO produces significantly enlarged basins while maintaining comparable final performance.

**Experimental Setup:**
- Models studied: Llama-3.1-8B, Qwen-2.5 (0.5B to 32B), Mistral-8B-2410
- Benchmarks: MMLU (basic proficiency), GSM8K (math reasoning), HumanEval (coding), AdvBench (safety)
- Key observation: Larger models exhibit proportionally larger basins and greater robustness.

**Relevance to Our Research:**
This is the most directly relevant paper for our LLM-focused study. It provides empirical evidence of the basin-like structure in LLM loss landscapes and establishes a framework for comparing landscape properties across models (Llama vs. Qwen vs. Mistral). The analysis of how pre-training and fine-tuning create nested basins is directly relevant to our goal of studying loss landscape evolution during training. The most-case vs. worst-case distinction highlights the importance of choosing the right projection directions for visualization—random projections capture the most-case landscape but miss worst-case fragility.

**Limitations:**
- Primarily focuses on the "basin" phenomenon rather than providing detailed visualization tools.
- The GO optimizer was only tested at small scale (GPT2-127M).
- Does not provide methods for projecting multiple models onto a shared 2D plane.
- Visualization is limited to 1D cross-sections along random or adversarial directions.

---

### 3.3 [Rank 3] Visualizing High-Dimensional Loss Landscapes with Hessian Directions (Böttcher & Wheeler, 2024)

**Full Citation:** Lucas Böttcher, Gregory Wheeler. "Visualizing High-Dimensional Loss Landscapes with Hessian Directions." *Journal of Statistical Mechanics: Theory and Experiment*, 2024(2), 023401. arXiv:2208.13219.

**Problem Addressed:** Standard approaches to loss landscape visualization use random projections to reduce the high-dimensional loss function to 2D plots. This paper investigates whether these random projections faithfully preserve the curvature properties of the original loss landscape, particularly saddle point structures.

**Core Contributions:**

1. **Theoretical Analysis of Random Projections:** By combining high-dimensional probability and differential geometry, the authors prove that the principal curvature in the expected lower-dimensional representation is proportional to the mean curvature (normalized Hessian trace) in the original loss space. Crucially, this means:
   - Saddle points in the original space are rarely correctly identified in lower-dimensional random projections.
   - Whether saddle points appear as minima, maxima, or flat regions depends on the mean curvature, not the individual principal curvatures.

2. **Hessian Direction Projections:** Instead of random directions, the authors propose projecting along the dominant Hessian eigenvectors—those associated with the largest and smallest principal curvatures. This preserves more faithful curvature information and correctly identifies saddle point structures.

3. **Efficient Computation via Hutchinson Trace Estimation:** They use the connection between expected curvature in random projections and the normalized Hessian trace to compute Hutchinson-type trace estimates without requiring Hessian-vector products as in the standard Hutchinson method.

**Experimental Validation:**
- Demonstrated on common image classifiers and function approximators with up to ~7×10⁶ parameters.
- Hessian direction projections correctly identified saddle points that random projections missed.
- Connected findings to the ongoing debate on loss landscape flatness and generalizability.

**Relevance to Our Research:**
This paper provides the theoretical justification for why we should use Hessian-based directions rather than random directions for loss landscape visualization, especially when studying LLMs. The key insight—that random projections lose saddle point information—is critical for any rigorous visualization study. However, the main challenge for applying this to LLMs is the computational cost of computing top Hessian eigenvectors for billion-parameter models. The paper's Hutchinson trace estimation approach could be adapted using modern Hessian-vector product implementations in PyTorch/JAX. The question of how to efficiently compute or approximate the dominant Hessian eigenvectors for LLMs remains open and is central to our proposed methodology.

**Limitations:**
- Only demonstrated on models up to 7M parameters, far smaller than modern LLMs.
- Does not address transformer-specific architecture considerations.
- Hessian eigenvector computation becomes prohibitively expensive for large models without further approximation.

---

### 3.4 [Rank 4] Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs (Garipov et al., NeurIPS 2018)

**Full Citation:** Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P. Vetrov, Andrew Gordon Wilson. "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs." *Advances in Neural Information Processing Systems 31 (NeurIPS)*, 2018. arXiv:1802.10026.

**Problem Addressed:** The geometric properties of deep neural network loss functions are not well understood. In particular, the relationship between different local optima (modes) found by independent training runs was unclear.

**Core Contributions:**

1. **Mode Connectivity Discovery:** The paper's central finding is that independently trained neural networks (different local optima) are connected by simple, low-loss curves in parameter space. Specifically, a polygonal chain with only one bend is sufficient to connect two optima while maintaining nearly constant training and test accuracy along the path.

2. **Curve-Finding Procedure:** They introduce a training procedure to discover these high-accuracy pathways. The curve is parameterized as a Bézier curve or a polychain in parameter space, and its parameters are optimized to minimize the expected loss along the curve.

3. **2D Loss Surface Visualization with Multiple Models:** A key contribution for our research is their method of visualizing the 2D loss surface in the plane defined by three independently trained models. Given three parameter vectors $\theta_1, \theta_2, \theta_3$, they evaluate the loss at all affine combinations $\theta = \alpha_1 \theta_1 + \alpha_2 \theta_2 + (1 - \alpha_1 - \alpha_2) \theta_3$, producing a 2D slice of the loss surface that passes through all three models.

4. **Fast Geometric Ensembling (FGE):** Inspired by mode connectivity, they propose an efficient ensembling method that explores the low-loss curves using SGD with a cyclical learning rate schedule.

**Key Results:**
- On CIFAR-10/100 and ImageNet, FGE consistently outperforms Snapshot Ensembles.
- The 2D visualizations show that modes are connected by low-loss valleys, but the straight line between two modes may cross a high-loss barrier.

**Relevance to Our Research:**
This paper is directly relevant to our goal of projecting multiple LLM checkpoints onto the same 2D loss landscape. The method of defining a 2D plane using three models as anchor points can be adapted for studying training trajectories (e.g., TinyLlama checkpoints at different training steps). The mode connectivity framework also provides theoretical context for understanding whether different LLM training configurations lead to solutions in the same basin. However, the naive affine combination approach may be problematic for LLMs due to the permutation symmetry of neurons—models with different initializations may not be directly comparable without alignment.

**Limitations:**
- The affine plane method only captures a very specific 2D slice, which may miss important features.
- Does not address the neuron permutation issue that arises when comparing independently trained models.
- Computationally expensive for LLMs: evaluating loss at each grid point requires a full forward pass through the model on a representative dataset.

---

### 3.5 [Rank 5] Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models (Liu et al., ICML 2023)

**Full Citation:** Hong Liu, Sang Michael Xie, Zhiyuan Li, Tengyu Ma. "Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models." *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR, pp. 22188–22214, 2023. arXiv:2210.14199.

**Problem Addressed:** Pre-training loss (perplexity) is the standard evaluation metric during language model development, with the assumption that lower pre-training loss leads to better downstream performance. This paper challenges this assumption.

**Core Contributions:**

1. **Pre-training Loss is Insufficient:** The paper demonstrates three scenarios where models achieve the same minimal pre-training loss but exhibit different downstream performance:
   - Continuing pre-training after validation loss has converged
   - Increasing model size
   - Changing the pre-training algorithm (e.g., SGD vs. Adam)

2. **Hessian Trace as a Superior Predictor:** Among models with identical pre-training loss (3.204–3.208), the trace of the Hessian of the pre-training loss correlates much better with downstream performance than the pre-training loss itself. As the trace of Hessian decreases (flatter landscape), downstream performance improves.

3. **Theoretical Foundation:** The authors prove that SGD with standard mini-batch noise implicitly prefers flatter minima of the pre-training loss in language models. In a synthetic language setting, they prove that among models with minimal pre-training loss, the flattest model transfers best to downstream tasks.

4. **Scale Effect:** As model size increases, the Hessian trace continues to decrease while downstream performance increases, even when pre-training loss remains nearly constant.

**Key Results:**
- Downstream SST-2 accuracy improved by 1.25% on OpenWebText when continuing training from 400K to 1400K steps (while validation loss remained flat).
- On synthetic datasets, improvements of 1.6% and 4.0% were observed.
- The correlation between Hessian trace and downstream performance was consistently stronger than the correlation with pre-training loss.

**Relevance to Our Research:**
This paper establishes a direct connection between loss landscape geometry (flatness/sharpness via Hessian trace) and model quality for language models. For our visualization study, this means:
1. Hessian-based visualization methods are not just theoretically motivated but practically informative—they can reveal differences in model quality that loss values alone cannot capture.
2. Tracking Hessian trace during pre-training (as with TinyLlama checkpoints) can reveal how the landscape flattens over training, providing a complementary view to loss surface visualization.
3. The implicit bias of optimizers toward flat regions suggests that the loss landscape geometry is not static but actively shaped by the training process.

**Limitations:**
- Hessian trace computation is expensive for large models (requires Hutchinson estimation with multiple Hessian-vector products).
- The analysis focuses on scalar summary statistics (trace) rather than full landscape visualization.
- The theoretical analysis is limited to simplified settings (SGD on language models) and may not directly apply to Adam-trained LLMs.

---

## 4. Analysis of Papers Ranked 6–10

### 4.1 [Rank 6] Sharpness-Aware Minimization for Efficiently Improving Generalization (Foret et al., ICLR 2021)

**Citation:** Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur. arXiv:2010.01412.

**Key Contribution:** Introduces SAM, which simultaneously minimizes loss value and loss sharpness via a min-max optimization problem. The connection between loss landscape geometry (flatness) and generalization is formalized through a PAC-Bayes generalization bound. SAM seeks parameters in neighborhoods with uniformly low loss, requiring only two gradient computations per iteration.

**Relevance:** SAM provides a theoretical framework linking loss landscape flatness to generalization. For our study, SAM-related concepts inform how to interpret the loss landscape visualizations—flatter regions should correspond to better-generalizing solutions. The SAM perturbation radius concept also relates to the "basin size" notion in Chen et al. (2025).

---

### 4.2 [Rank 7] An Investigation into Neural Net Optimization via Hessian Eigenvalue Density (Ghorbani et al., ICML 2019)

**Citation:** Behrooz Ghorbani, Shankar Krishnan, Ying Xiao. arXiv:1901.10159.

**Key Contribution:** Provides a systematic study of Hessian eigenvalue distributions during neural network training. Uses stochastic Lanczos quadrature to efficiently estimate the eigenvalue spectral density without computing the full Hessian. Reveals characteristic spectral patterns: a bulk of near-zero eigenvalues and a small number of outlier eigenvalues, with approximately K non-zero outliers for K-class classification.

**Relevance:** The efficient Hessian spectral density estimation methods (stochastic Lanczos quadrature) are directly applicable to our LLM loss landscape study. Understanding the spectral structure helps guide the choice of projection directions for visualization—the top eigenvectors capture the directions of greatest curvature and are thus most informative for loss landscape plots.

---

### 4.3 [Rank 8] Visualizing, Rethinking, and Mining the Loss Landscape of Deep Neural Networks (Xu et al., 2024)

**Citation:** Yichu Xu, Xin-Chun Li, Lan Li, De-Chuan Zhan. arXiv:2405.12493.

**Key Contribution:** Systematically categorizes 1D loss curve structures into v-basin, v-side, w-basin, w-peak, and vvv-basin types. Proposes mining algorithms to discover complex perturbation directions that reveal non-trivial landscape features (saddle surfaces, "wine bottle bottom" shapes). Provides theoretical insights through the Hessian matrix to explain observed phenomena.

**Relevance:** This paper extends the Li et al. (2018) framework with richer visualization categories and mining algorithms. For our LLM study, the idea of "mining" informative perturbation directions (rather than using random directions) could reveal more complex landscape features. The categorization scheme provides a vocabulary for describing different landscape geometries observed in LLM loss surfaces.

---

### 4.4 [Rank 9] A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs (Kalra et al., 2026)

**Citation:** Dayal Singh Kalra, Jean-Christophe Gagnon-Audet, Andrey Gromov, et al. arXiv:2601.16979.

**Key Contribution:** Proposes "critical sharpness" ($\lambda_c$)—the inverse of the smallest learning rate causing loss to increase—as a computationally efficient proxy for Hessian sharpness. Requires fewer than 10 forward passes given the update direction. Demonstrates progressive sharpening and Edge of Stability phenomena at scale up to 7B parameters (OLMo-2 models). Introduces "relative critical sharpness" for analyzing pre-training to fine-tuning transitions.

**Relevance:** This is the most practically relevant paper for our computational methodology. Computing full Hessian eigenvectors for LLMs is prohibitively expensive, but critical sharpness provides a scalable alternative. The demonstration at 7B scale directly validates the feasibility of loss landscape analysis for models in the size range we plan to study (Qwen3-0.6B, TinyLlama-1.1B, OLMo3-7B, Qwen2.5-7B). The relative critical sharpness concept could inform how we compare landscapes across different training stages.

---

### 4.5 [Rank 10] Training Trajectories of Language Models Across Scales (Xia et al., ACL 2023)

**Citation:** Mengzhou Xia, Mikel Artetxe, Chunting Zhou, et al. arXiv:2212.09803.

**Key Contribution:** Analyzes intermediate training checkpoints of OPT models from 125M to 175B parameters. Finds that at a given perplexity (independent of model size), a similar subset of tokens sees the most significant loss reduction, with ~9.4% showing double-descent behavior. Demonstrates that perplexity is a strong predictor of in-context learning performance, independent of model size.

**Relevance:** This paper provides methodology and motivation for studying loss landscapes along the training trajectory using intermediate checkpoints, directly applicable to our TinyLlama checkpoint analysis. The finding that training dynamics share common patterns across scales suggests that loss landscape visualizations from smaller models may transfer insights to larger ones. The per-token loss analysis could inform what data to use when evaluating loss surfaces at different training stages.

---

## 5. Thematic Summary and Research Gaps

### 5.1 Loss Landscape Visualization Methods
The field has progressed from random direction projections (Li et al., 2018) to Hessian-based projections (Böttcher & Wheeler, 2024) to mining-based approaches (Xu et al., 2024). However, **all existing methods were developed for and validated on relatively small models** (CNNs with millions of parameters). Applying these techniques to LLMs with billions of parameters requires addressing fundamental scalability challenges.

### 5.2 Weight Normalization for Visualization
Filter normalization (Li et al., 2018) is the standard approach, but it was designed for CNNs where "filters" have a clear meaning. For transformers, the notion of a "filter" must be re-interpreted for attention weights, projection matrices, layer norms, and embeddings. **No existing work has systematically studied how to normalize directions for transformer/LLM visualization.**

### 5.3 Hessian Computation at Scale
Computing full Hessian eigenvectors for LLMs is infeasible. Existing approaches include:
- Hutchinson trace estimation (Böttcher & Wheeler, 2024; Ghorbani et al., 2019)
- Hessian-vector products via autodiff (standard in PyTorch/JAX, but incompatible with Flash Attention—Kalra et al., 2026)
- Critical sharpness as a proxy (Kalra et al., 2026)

**A gap exists in developing efficient, approximate Hessian-based projection methods specifically for LLMs** that balance computational cost with visualization fidelity.

### 5.4 Multi-Model Projection
Garipov et al. (2018) project multiple models onto a 2D plane using affine combinations, but this requires the models to be in a comparable parameter space (same initialization or alignment). **No existing work has addressed projecting multiple independently trained LLMs or training checkpoints onto a shared visualization plane** with appropriate normalization and alignment.

### 5.5 Loss Landscape and LLM Capabilities
Chen et al. (2025) and Liu et al. (2023) establish that loss landscape geometry (basin size, flatness) correlates with LLM capabilities and robustness. However, **the relationship between visual landscape features and specific model capabilities (reasoning, coding, safety) remains unexplored.** Comparing landscapes across different models (e.g., OLMo3-7B vs. Qwen2.5-7B) with different training recipes could reveal how training choices shape the landscape geometry.

### 5.6 Training Dynamics via Loss Landscape
Xia et al. (2023) and Kalra et al. (2026) study training dynamics through loss or curvature metrics, but **no existing work has produced systematic loss landscape visualizations across the full pre-training trajectory of an LLM,** such as tracking how the 2D loss surface evolves as training progresses through the TinyLlama checkpoint series.

---

## 6. Key References (BibTeX-ready)

```
@inproceedings{li2018visualizing,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}

@article{chen2025unveiling,
  title={Unveiling the Basin-Like Loss Landscape in Large Language Models},
  author={Chen, Huanran and Dong, Yinpeng and Wei, Zeming and Huang, Yao and Zhang, Yichi},
  journal={arXiv preprint arXiv:2505.17646},
  year={2025}
}

@article{bottcher2024visualizing,
  title={Visualizing High-Dimensional Loss Landscapes with Hessian Directions},
  author={B{\"o}ttcher, Lucas and Wheeler, Gregory},
  journal={Journal of Statistical Mechanics: Theory and Experiment},
  volume={2024},
  number={2},
  pages={023401},
  year={2024}
}

@inproceedings{garipov2018loss,
  title={Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs},
  author={Garipov, Timur and Izmailov, Pavel and Podoprikhin, Dmitrii and Vetrov, Dmitry P. and Wilson, Andrew Gordon},
  booktitle={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}

@inproceedings{liu2023same,
  title={Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models},
  author={Liu, Hong and Xie, Sang Michael and Li, Zhiyuan and Ma, Tengyu},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  pages={22188--22214},
  year={2023},
  publisher={PMLR}
}

@inproceedings{foret2021sharpness,
  title={Sharpness-Aware Minimization for Efficiently Improving Generalization},
  author={Foret, Pierre and Kleiner, Ariel and Mobahi, Hossein and Neyshabur, Behnam},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{ghorbani2019investigation,
  title={An Investigation into Neural Net Optimization via Hessian Eigenvalue Density},
  author={Ghorbani, Behrooz and Krishnan, Shankar and Xiao, Ying},
  booktitle={International Conference on Machine Learning},
  pages={2232--2241},
  year={2019},
  publisher={PMLR}
}

@article{xu2024visualizing,
  title={Visualizing, Rethinking, and Mining the Loss Landscape of Deep Neural Networks},
  author={Xu, Yichu and Li, Xin-Chun and Li, Lan and Zhan, De-Chuan},
  journal={arXiv preprint arXiv:2405.12493},
  year={2024}
}

@article{kalra2026scalable,
  title={A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs},
  author={Kalra, Dayal Singh and Gagnon-Audet, Jean-Christophe and Gromov, Andrey and Mediratta, Ishita and Niu, Kelvin and Miller, Alexander H and Shvartsman, Michael},
  journal={arXiv preprint arXiv:2601.16979},
  year={2026}
}

@inproceedings{xia2023training,
  title={Training Trajectories of Language Models Across Scales},
  author={Xia, Mengzhou and Artetxe, Mikel and Zhou, Chunting and Lin, Xi Victoria and Pasunuru, Ramakanth and Chen, Danqi and Zettlemoyer, Luke and Stoyanov, Ves},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
  pages={13711--13738},
  year={2023}
}
```
