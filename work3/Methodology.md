# Methodology: LLMScape — Faithful, Scalable Loss Landscape Visualization for Large Language Models

## 1. Overview

We present **LLMScape**, a framework for visualizing and analyzing the loss landscapes of large language models (LLMs). LLMScape consists of five core technical contributions:

1. **Transformer-Adapted Direction Normalization (TADN)** — a principled normalization scheme that provably preserves scale-invariance properties specific to transformer architectures, enabling meaningful comparison across models with different internal scale distributions.
2. **Scalable Hessian-Informed Direction Selection (SHIDS)** — a three-tier hierarchy of direction selection methods with convergence guarantees and adaptive sample-size selection for the gradient covariance tier.
3. **Projection Faithfulness Index (PFI)** — a theoretically grounded metric that quantifies how well a 2D projection represents the true high-dimensional loss landscape geometry, enabling principled comparison of visualization methods.
4. **Multi-Model Shared Projection (MMSP)** — methods for projecting multiple models onto shared 2D coordinate systems for geometric comparison.
5. **Efficient Computation Pipeline** — engineering optimizations including curvature-aware adaptive scale selection for visualization.

---

## 2. Preliminaries and Notation

### 2.1 Loss Landscape Visualization

Consider a language model parameterized by $\theta^* \in \mathbb{R}^d$, where $d$ is the total number of parameters. Let $L(\theta) = \mathbb{E}_{x \sim \mathcal{D}}[\ell(\theta; x)]$ denote the expected loss over dataset $\mathcal{D}$, where $\ell(\theta; x) = -\log p_\theta(x)$ is the negative log-likelihood of sequence $x$ under the model.

The **2D loss surface plot** is defined by choosing two direction vectors $\mathbf{d}_1, \mathbf{d}_2 \in \mathbb{R}^d$ and evaluating:

$$f(\alpha, \beta) = L(\theta^* + \alpha \mathbf{d}_1 + \beta \mathbf{d}_2)$$

on a discrete grid $\{(\alpha_i, \beta_j)\}_{i,j=1}^{G}$ for grid resolution $G$.

### 2.2 Hessian and Curvature

The Hessian matrix $H = \nabla^2 L(\theta^*) \in \mathbb{R}^{d \times d}$ captures the local curvature of the loss landscape. Its eigendecomposition $H = \sum_{i=1}^d \lambda_i \mathbf{v}_i \mathbf{v}_i^\top$ reveals the principal curvature directions $\mathbf{v}_i$ and magnitudes $\lambda_i$.

For a 2D projection along directions $\mathbf{d}_1, \mathbf{d}_2$, the curvature of the projected surface at the origin is:

$$\kappa_{ij} = \mathbf{d}_i^\top H \mathbf{d}_j, \quad i,j \in \{1, 2\}$$

When $\mathbf{d}_1, \mathbf{d}_2$ are Hessian eigenvectors, $\kappa_{11} = \lambda_1$, $\kappa_{22} = \lambda_2$, and $\kappa_{12} = 0$, providing a faithful representation of curvature along those directions.

### 2.3 Transformer Architecture Notation

A transformer-based LLM consists of an embedding layer, $N_L$ transformer blocks, and a language model head. Each transformer block $l \in \{1, \ldots, N_L\}$ contains:

- **Multi-head attention (MHA):** $n_h$ heads, each with query, key, value projections $W_Q^{(l,h)}, W_K^{(l,h)}, W_V^{(l,h)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}$ and output projection $W_O^{(l)} \in \mathbb{R}^{(n_h \cdot d_{\text{head}}) \times d_{\text{model}}}$.
- **Feed-forward network (FFN):** Up-projection $W_{\text{up}}^{(l)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, gate projection $W_{\text{gate}}^{(l)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ (for SwiGLU architectures), and down-projection $W_{\text{down}}^{(l)} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$.
- **Layer normalization:** RMSNorm parameters $\gamma^{(l)} \in \mathbb{R}^{d_{\text{model}}}$.

The token embedding is $W_E \in \mathbb{R}^{V \times d_{\text{model}}}$ and the LM head is $W_{\text{lm}} \in \mathbb{R}^{d_{\text{model}} \times V}$ (often tied with $W_E$).

---

## 3. Transformer-Adapted Direction Normalization (TADN)

### 3.1 Motivation

Li et al. (2018) introduced filter normalization for CNNs to remove scale-invariance artifacts. The key insight is that due to the positive homogeneity of ReLU networks, one can rescale any layer's parameters by a factor $c$ and its successor by $1/c$ without changing the network function, creating an infinite family of equivalent parameterizations with arbitrarily different apparent sharpness (Dinh et al., 2017).

Transformers exhibit analogous scale invariance:
- **RMSNorm invariance:** For any layer followed by RMSNorm, scaling $W$ by $c$ does not change the output because RMSNorm divides by the root-mean-square of its input.
- **FFN neuron scaling:** In SwiGLU FFNs, $\text{FFN}(x) = W_{\text{down}} \cdot (\text{swish}(W_{\text{gate}} x) \odot W_{\text{up}} x)$. Scaling the $j$-th row of $W_{\text{up}}$ by $c_j$ and the $j$-th column of $W_{\text{down}}$ by $1/c_j$ preserves the FFN output exactly: the $j$-th hidden unit output is scaled by $c_j$, and the down-projection compensates with $1/c_j$.
- **Attention scale invariance:** In the attention mechanism, $\text{softmax}(QK^\top / \sqrt{d_k})V$, certain rescalings of Q, K, V projections can be compensated by output projection adjustments.

Zhang et al. (NeurIPS 2024) demonstrated that the Hessian spectrum exhibits significant **block-level heterogeneity** across transformer components, and the Sharpness Disparity Principle (arXiv:2502.19002) shows dramatically different sharpness characteristics across component types (Emb, QK, VO, FFN, Norm). This directly motivates per-component normalization: a single normalization strategy would fail to account for inherent scale differences.

Without proper normalization, perturbation directions may disproportionately affect certain parameters, creating misleading landscape visualizations. Moreover, **different normalization granularities lead to different invariance properties**: coarse layer-level normalization fails to preserve visualization consistency under the neuron-level rescalings that are exact symmetries of the network.

### 3.2 Normalization Unit Definition

We partition the full parameter vector $\theta \in \mathbb{R}^d$ into normalization units $\{\theta_i\}_{i=1}^{M}$ such that $\theta = [\theta_1; \theta_2; \ldots; \theta_M]$. Each unit is a semantically meaningful group of parameters corresponding to an independent functional component of the transformer.

**Definition 3.1 (Normalization Unit Partition).** For a transformer with $N_L$ layers, $n_h$ attention heads, vocabulary size $V$, model dimension $d_{\text{model}}$, and FFN dimension $d_{\text{ff}}$, the normalization unit partition $\mathcal{P}$ is:

| Component | Parameters | Unit | Count per layer | Unit dimension |
|-----------|-----------|------|-----------------|---------------|
| Token Embedding | $W_E$ | Each row $W_E[v, :] \in \mathbb{R}^{d_{\text{model}}}$ | — | $d_{\text{model}}$ |
| Q Projection | $W_Q^{(l,h)}$ | Full matrix per head | $n_h$ | $d_{\text{model}} \times d_{\text{head}}$ |
| K Projection | $W_K^{(l,h)}$ | Full matrix per head | $n_h$ | $d_{\text{model}} \times d_{\text{head}}$ |
| V Projection | $W_V^{(l,h)}$ | Full matrix per head | $n_h$ | $d_{\text{model}} \times d_{\text{head}}$ |
| O Projection | $W_O^{(l)}$ | Per-head slice $W_O^{(l)}[h] \in \mathbb{R}^{d_{\text{head}} \times d_{\text{model}}}$ | $n_h$ | $d_{\text{head}} \times d_{\text{model}}$ |
| FFN Up/Gate | $W_{\text{up}}^{(l)}$, $W_{\text{gate}}^{(l)}$ | Each column (neuron input weights) | $d_{\text{ff}}$ (each) | $d_{\text{model}}$ |
| FFN Down | $W_{\text{down}}^{(l)}$ | Each row (neuron output weights) | $d_{\text{ff}}$ | $d_{\text{model}}$ |
| RMSNorm | $\gamma^{(l)}$ | Entire vector | 1 (or 2 per layer) | $d_{\text{model}}$ |
| LM Head | $W_{\text{lm}}$ | Each column | — | $d_{\text{model}}$ |

### 3.3 Normalization Procedure

**Algorithm 1: TADN Normalization**

**Input:** Model parameters $\theta^*$, random direction $\mathbf{d} \sim \mathcal{N}(0, I_d)$, normalization unit partition $\mathcal{P}$.

**Output:** Normalized direction $\hat{\mathbf{d}}$.

1. Decompose $\theta^*$ and $\mathbf{d}$ according to partition $\mathcal{P}$: $\theta^* = [\theta_1^*; \ldots; \theta_M^*]$, $\mathbf{d} = [d_1; \ldots; d_M]$.
2. For each normalization unit $i \in \{1, \ldots, M\}$:
   - If $\|\theta_i^*\|_F > \epsilon$ (non-degenerate): $\hat{d}_i = \frac{d_i}{\|d_i\|_F} \cdot \|\theta_i^*\|_F$
   - Else (near-zero parameters): $\hat{d}_i = d_i \cdot \frac{\bar{s}}{\|d_i\|_F}$ where $\bar{s}$ is the median norm across all non-degenerate units.
3. Return $\hat{\mathbf{d}} = [\hat{d}_1; \ldots; \hat{d}_M]$.

### 3.4 Theoretical Justification

**Proposition 3.1 (Scale Invariance under FFN Neuron Rescaling).** Let $\theta$ and $\theta'$ be two parameterizations of the same network function related by non-uniform FFN neuron rescaling: for each neuron $j$ in each FFN layer, $W_{\text{up},j}' = c_j \cdot W_{\text{up},j}$ and $W_{\text{down}}'^{(:,j)} = c_j^{-1} \cdot W_{\text{down}}^{(:,j)}$ for positive scalars $c_j$. Then the TADN-normalized perturbation satisfies:

$$L(\theta' + \alpha \hat{\mathbf{d}}') = L(\theta + \alpha \hat{\mathbf{d}})$$

for all $\alpha$, where $\hat{\mathbf{d}}$ and $\hat{\mathbf{d}}'$ are TADN normalizations of the same random direction $\mathbf{d}$ with respect to $\theta$ and $\theta'$, respectively.

**Proof.** Under the rescaling, neuron $j$'s up-projection parameter changes from $W_{\text{up},j}$ to $c_j W_{\text{up},j}$, and the corresponding TADN normalization scales the direction component as:

$$\hat{d}'_{\text{up},j} = \frac{d_{\text{up},j}}{\|d_{\text{up},j}\|_F} \cdot \|c_j W_{\text{up},j}\|_F = c_j \cdot \frac{d_{\text{up},j}}{\|d_{\text{up},j}\|_F} \cdot \|W_{\text{up},j}\|_F = c_j \cdot \hat{d}_{\text{up},j}$$

Similarly, $\hat{d}'_{\text{down},j} = c_j^{-1} \cdot \hat{d}_{\text{down},j}$. The perturbed FFN output for neuron $j$:

$$\text{FFN}'_j = (c_j^{-1} W_{\text{down},j} + \alpha \cdot c_j^{-1} \hat{d}_{\text{down},j}) \cdot \text{swish}(W_{\text{gate},j} x) \cdot (c_j W_{\text{up},j} + \alpha \cdot c_j \hat{d}_{\text{up},j})x$$

$$= c_j^{-1} \cdot c_j \cdot (W_{\text{down},j} + \alpha \hat{d}_{\text{down},j}) \cdot \text{swish}(W_{\text{gate},j} x) \cdot (W_{\text{up},j} + \alpha \hat{d}_{\text{up},j})x$$

$$= (W_{\text{down},j} + \alpha \hat{d}_{\text{down},j}) \cdot \text{swish}(W_{\text{gate},j} x) \cdot (W_{\text{up},j} + \alpha \hat{d}_{\text{up},j})x = \text{FFN}_j$$

Since this holds for each neuron $j$, the total FFN output is identical, and therefore $L(\theta' + \alpha \hat{\mathbf{d}}') = L(\theta + \alpha \hat{\mathbf{d}})$. $\square$

**Proposition 3.2 (Layer Normalization Fails under Non-Uniform Rescaling).** Under the same non-uniform rescaling as Proposition 3.1 with at least two distinct scaling factors $c_j \neq c_k$, layer-level normalization (normalizing the entire $W_{\text{up}}$ matrix as one unit) does **not** preserve the loss landscape.

**Proof.** Layer normalization scales the direction for $W_{\text{up}}$ by $\|W'_{\text{up}}\|_F / \|W_{\text{up}}\|_F$, where:

$$\|W'_{\text{up}}\|_F = \sqrt{\sum_j c_j^2 \|W_{\text{up},j}\|_F^2}$$

This applies a uniform scaling factor to all neurons in the direction, regardless of their individual rescaling. The direction for neuron $j$ is scaled by the same factor $\|W'_{\text{up}}\|_F / \|W_{\text{up}}\|_F$, which differs from the required $c_j$ when the $c_j$ values are non-uniform. Therefore, the perturbed FFN output is not preserved. $\square$

**Remark.** This theoretical advantage is confirmed experimentally: when comparing an original model with a rescaled equivalent (using non-uniform $c_j \in \{0.33, 0.5, 2.0, 3.0\}$), TADN achieves near-perfect correlation (>0.999) between the loss curves, while layer normalization shows significantly lower correlation (see Initial_check.md).

---

## 4. Scalable Hessian-Informed Direction Selection (SHIDS)

### 4.1 Why Direction Selection Matters

**Theorem 4.1 (Curvature Collapse under Random Projection; Böttcher & Wheeler, 2024).** Let $H \in \mathbb{R}^{d \times d}$ be the Hessian at $\theta^*$ with eigenvalues $\lambda_1 \geq \cdots \geq \lambda_d$. For a random Gaussian direction $\mathbf{d} \sim \mathcal{N}(0, I_d)$, as $d \to \infty$:

$$\mathbf{d}^\top H \mathbf{d} / \|\mathbf{d}\|^2 \to \frac{1}{d} \text{tr}(H) = \frac{1}{d} \sum_{i=1}^d \lambda_i$$

almost surely. This means random projections collapse all curvature information into the mean curvature, and saddle points (where $\lambda_{\max} > 0 > \lambda_{\min}$) appear as local minima whenever $\text{tr}(H) > 0$.

**Corollary.** To faithfully capture curvature extremes, projection directions should be aligned with the top Hessian eigenvectors.

### 4.2 Three-Tier Direction Selection Framework

#### Tier 1: Random Directions with TADN (Baseline)

1. Sample $\mathbf{d}_1, \mathbf{d}_2 \sim \mathcal{N}(0, I_d)$.
2. Orthogonalize: $\mathbf{d}_2 \leftarrow \mathbf{d}_2 - \frac{\langle \mathbf{d}_2, \mathbf{d}_1 \rangle}{\|\mathbf{d}_1\|^2} \mathbf{d}_1$.
3. Apply TADN normalization to both.

**Cost:** Negligible (sampling + normalization).
**Faithfulness:** Captures only mean curvature; baseline for comparison. PFI-S $\approx 2/d \to 0$.

#### Tier 2: Gradient Covariance PCA with Adaptive Sample-Size (Efficient Proxy)

**Rationale:** Gur-Ari, Roberts & Dyer (2018) proved that the top eigenspaces of the Hessian $H$ and the gradient covariance matrix $\Sigma = \mathbb{E}[g g^\top]$ largely coincide. This allows approximating the top Hessian eigenvectors using only first-order information.

**Algorithm 2: Gradient Covariance PCA with Convergence Monitoring**

**Input:** Model $\theta^*$, dataset $\mathcal{D}$, initial batch count $N_0$, maximum batch count $N_{\max}$, convergence threshold $\tau_{\angle}$ (default: 5°), rank $k = 2$.

**Output:** Top-$k$ approximate Hessian eigenvectors, convergence report.

1. Set $N \leftarrow N_0$, $U_{\text{prev}} \leftarrow \emptyset$.
2. While $N \leq N_{\max}$:
   a. Collect per-batch gradients $\{g_1, \ldots, g_N\}$.
   b. Build the $N \times N$ Gram matrix $G_{ij} = g_i \cdot g_j$.
   c. Center: $\tilde{G} = (I - \frac{1}{N}\mathbf{1}\mathbf{1}^\top)G(I - \frac{1}{N}\mathbf{1}\mathbf{1}^\top)$.
   d. Eigendecompose $\tilde{G}$; take top-$k$ eigenvectors $\mathbf{e}_1, \ldots, \mathbf{e}_k$.
   e. Reconstruct $d$-dimensional directions: $\mathbf{u}_j = \sum_{i=1}^N e_{j,i} \cdot g_i$, normalize.
   f. If $U_{\text{prev}} \neq \emptyset$:
      - Compute the principal angle $\angle(U_N, U_{\text{prev}})$ between the current and previous $k$-dimensional subspaces.
      - If $\angle < \tau_{\angle}$: convergence reached; **break**.
   g. $U_{\text{prev}} \leftarrow \{u_1, \ldots, u_k\}$.
   h. $N \leftarrow \lfloor 1.5 \cdot N \rfloor$ (geometric growth).
3. Apply TADN normalization to each $\mathbf{u}_j$.
4. Return $\{\mathbf{u}_1, \ldots, \mathbf{u}_k\}$ and convergence statistics.

**Convergence Analysis.** By the Davis-Kahan perturbation theorem, the angle between the empirical top-$k$ subspace and the true top-$k$ subspace of $\Sigma$ satisfies:

$$\sin(\angle(\hat{U}_N, U^*)) \leq \frac{\|\hat{\Sigma}_N - \Sigma\|_{\text{op}}}{\sigma_k(\Sigma) - \sigma_{k+1}(\Sigma)}$$

where $\hat{\Sigma}_N = \frac{1}{N}\sum_{i=1}^N g_i g_i^\top$ is the sample covariance and $\sigma_k(\Sigma)$ is the $k$-th eigenvalue. For i.i.d. sub-Gaussian gradients, $\|\hat{\Sigma}_N - \Sigma\|_{\text{op}} = O(\sigma_1 \sqrt{d/N})$ with high probability. The convergence rate thus depends on the **spectral gap** $\sigma_k - \sigma_{k+1}$: larger gaps enable faster convergence with fewer samples.

For LLMs, where the gradient covariance has a small number of large outlier eigenvalues (Ghorbani et al., 2019), the spectral gap is typically substantial, enabling convergence with $N = 50$–$100$ mini-batches.

**Cost:** $N$ gradient computations (typically 50–100). No Hessian-vector products needed.

#### Tier 3: Power Iteration for Top Hessian Eigenvectors (Most Faithful)

**Algorithm 3: Power Iteration with Hessian-Vector Products**

**Input:** Model $\theta^*$, dataset $\mathcal{D}$, number of eigenvectors $k = 2$, convergence tolerance $\tau$.

**Output:** Top-$k$ Hessian eigenvectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ and eigenvalues $\{\lambda_1, \ldots, \lambda_k\}$.

1. For $j = 1, \ldots, k$:
   a. Initialize $\mathbf{v}_j \sim \mathcal{N}(0, I_d)$.
   b. Deflate: $\mathbf{v}_j \leftarrow \mathbf{v}_j - \sum_{i<j} \langle \mathbf{v}_j, \mathbf{v}_i \rangle \mathbf{v}_i$.
   c. Normalize: $\mathbf{v}_j \leftarrow \mathbf{v}_j / \|\mathbf{v}_j\|$.
   d. For $t = 1, \ldots, T_{\max}$ (typically 30–50):
      - Compute $\mathbf{w} = H \mathbf{v}_j$ via Hessian-vector product (Pearlmutter trick).
      - Deflate: $\mathbf{w} \leftarrow \mathbf{w} - \sum_{i<j} \langle \mathbf{w}, \mathbf{v}_i \rangle \mathbf{v}_i$.
      - Compute eigenvalue estimate: $\lambda_j = \langle \mathbf{v}_j, \mathbf{w} \rangle$.
      - Normalize: $\mathbf{v}_j \leftarrow \mathbf{w} / \|\mathbf{w}\|$.
      - If $|\langle \mathbf{v}_j^{\text{new}}, \mathbf{v}_j^{\text{old}} \rangle| > 1 - \tau$: break (converged).
   e. Apply TADN normalization to $\mathbf{v}_j$.

**HVP Computation via Pearlmutter Trick:** The HVP $H\mathbf{v} = \nabla_\theta(\nabla_\theta L(\theta) \cdot \mathbf{v})$ is computed as:
1. Forward + backward pass to get $g = \nabla_\theta L(\theta)$ with `create_graph=True`.
2. Compute $s = g \cdot \mathbf{v}$ (scalar dot product).
3. Backward pass on $s$ to get $\nabla_\theta s = H\mathbf{v}$.

**Implementation Note:** Flash Attention does not support `create_graph=True` (Kalra et al., 2026). We use standard attention (`attn_implementation="eager"`) for HVP computation and float32 precision for numerical stability.

**Cost:** $\sim 2k T_{\max}$ backward passes. For $k=2$, $T_{\max}=30$: ~120 backward passes.

**Faithfulness:** Exact principal curvature directions (up to convergence tolerance). PFI-S = $(\lambda_1^2 + \lambda_2^2) / \text{tr}(H^2)$, which is the theoretical maximum for any 2D projection.

### 4.3 Trajectory PCA Direction (for Training Trajectory Visualization)

For a sequence of checkpoints $\{\theta_0, \theta_1, \ldots, \theta_T\}$:

1. Compute the centroid: $\bar{\theta} = \frac{1}{T+1}\sum_{t=0}^{T}\theta_t$.
2. Form the difference matrix $\Delta = [\theta_0 - \bar{\theta}, \ldots, \theta_T - \bar{\theta}] \in \mathbb{R}^{d \times (T+1)}$.
3. Compute via the Gram matrix $\Delta^\top \Delta \in \mathbb{R}^{(T+1) \times (T+1)}$ (since $T \ll d$).
4. Take the top-2 left singular vectors $\mathbf{p}_1, \mathbf{p}_2$.
5. Apply TADN normalization.

---

## 5. Projection Faithfulness Index (PFI)

### 5.1 Motivation

A fundamental challenge in loss landscape visualization is assessing **how faithfully a 2D projection represents the true high-dimensional geometry**. Prior work (Li et al., 2018; Böttcher & Wheeler, 2024) has established qualitative arguments about the limitations of random projections, but no quantitative metric exists to compare different visualization approaches. We propose the Projection Faithfulness Index (PFI), a theoretically grounded metric that fills this gap.

### 5.2 Definition

**Definition 5.1 (Projection Faithfulness Index — Spectral Coverage).** Given a 2D projection along orthonormal directions $\hat{\mathbf{d}}_1, \hat{\mathbf{d}}_2$ (i.e., $\|\hat{\mathbf{d}}_i\| = 1$, $\hat{\mathbf{d}}_1 \perp \hat{\mathbf{d}}_2$) and the Hessian $H = \nabla^2 L(\theta^*)$, the **Spectral PFI** is:

$$\text{PFI-S}(\hat{\mathbf{d}}_1, \hat{\mathbf{d}}_2) = \frac{\|H \hat{\mathbf{d}}_1\|^2 + \|H \hat{\mathbf{d}}_2\|^2}{\text{tr}(H^2)} = \frac{\hat{\mathbf{d}}_1^\top H^2 \hat{\mathbf{d}}_1 + \hat{\mathbf{d}}_2^\top H^2 \hat{\mathbf{d}}_2}{\sum_{i=1}^d \lambda_i^2}$$

**Definition 5.2 (PFI — Curvature Capture).** The **Curvature PFI** measures how well the projection captures the maximum curvature direction:

$$\text{PFI-C}(\hat{\mathbf{d}}_1, \hat{\mathbf{d}}_2) = \frac{\max(|\hat{\mathbf{d}}_1^\top H \hat{\mathbf{d}}_1|, |\hat{\mathbf{d}}_2^\top H \hat{\mathbf{d}}_2|)}{|\lambda_1|}$$

where $\lambda_1$ is the eigenvalue of $H$ with largest absolute value.

### 5.3 Theoretical Properties

**Theorem 5.1 (PFI Bounds and Extrema).**

(i) $\text{PFI-S} \in [0, 1]$ for any orthonormal pair $\hat{\mathbf{d}}_1, \hat{\mathbf{d}}_2$.

(ii) $\text{PFI-S}$ is maximized when $\hat{\mathbf{d}}_1, \hat{\mathbf{d}}_2$ are the eigenvectors of $H^2$ (equivalently, of $H$) corresponding to the two largest eigenvalues $\lambda_1^2, \lambda_2^2$, achieving:

$$\text{PFI-S}_{\max} = \frac{\lambda_1^2 + \lambda_2^2}{\sum_{i=1}^d \lambda_i^2}$$

(iii) For random orthonormal directions in $\mathbb{R}^d$ (drawn uniformly from the Stiefel manifold), as $d \to \infty$:

$$\mathbb{E}[\text{PFI-S}] = \frac{2}{d} \to 0$$

**Proof.**

(i) Since $H^2$ is positive semi-definite, $\hat{\mathbf{d}}_i^\top H^2 \hat{\mathbf{d}}_i \geq 0$, so $\text{PFI-S} \geq 0$. Also, $\hat{\mathbf{d}}_i^\top H^2 \hat{\mathbf{d}}_i \leq \lambda_{\max}(H^2) = \lambda_1^2$ for unit vectors, and $\text{tr}(H^2) \geq \lambda_1^2 + \lambda_2^2$, so $\text{PFI-S} \leq (\lambda_1^2 + \lambda_2^2) / (\lambda_1^2 + \lambda_2^2) = 1$ when exactly two eigenvalues are non-zero, and $\text{PFI-S} \leq 1$ in general.

(ii) By the Rayleigh quotient characterization, $\sum_{i=1}^2 \hat{\mathbf{d}}_i^\top H^2 \hat{\mathbf{d}}_i$ is maximized over 2-dimensional subspaces by the top-2 eigenvectors of $H^2$, which are the eigenvectors of $H$ with the two largest eigenvalues in absolute value.

(iii) For a random unit vector $\hat{\mathbf{d}} \sim \text{Uniform}(S^{d-1})$: $\mathbb{E}[\hat{\mathbf{d}}^\top H^2 \hat{\mathbf{d}}] = \frac{1}{d}\text{tr}(H^2)$. For two orthogonal random unit vectors: $\mathbb{E}[\text{PFI-S}] = \frac{2}{d}\text{tr}(H^2) / \text{tr}(H^2) = \frac{2}{d}$. $\square$

**Corollary 5.1 (PFI as a Quality Metric).** For LLMs with $d \sim 10^8$–$10^{10}$, random projections achieve PFI-S $\sim 10^{-8}$–$10^{-10}$. The PFI of gradient PCA or Hessian eigenvector projections quantifies how much improvement over random is achieved. A higher PFI indicates a more faithful visualization.

**Proposition 5.1 (PFI Significance for LLM Hessians).** For neural network Hessians with the characteristic "bulk + outlier" spectral structure (Ghorbani et al., 2019)—a bulk of $d - K$ near-zero eigenvalues and $K$ outlier eigenvalues—the maximum PFI-S satisfies:

$$\text{PFI-S}_{\max} \approx \frac{\lambda_1^2 + \lambda_2^2}{\sum_{j=1}^K \lambda_j^2} \approx \frac{2}{K} \text{ (when outliers are of similar magnitude)}$$

Since $K \ll d$ for neural networks (typically $K \sim \text{number of classes}$ for classification, or $K \sim 10$–$100$ for LLMs), the maximum PFI-S is of order $O(1/K)$, which is far above the random baseline of $2/d$, demonstrating that meaningful faithful projections exist.

### 5.4 Efficient Computation

PFI can be computed efficiently using Hessian-vector products and Hutchinson trace estimation:

**Numerator:** $\|H\hat{\mathbf{d}}_i\|^2$ requires one HVP per direction (2 HVPs total).

**Denominator:** $\text{tr}(H^2) = \mathbb{E}_{\mathbf{v} \sim \mathcal{N}(0, I)}[\|H\mathbf{v}\|^2]$, estimated via Hutchinson's estimator with $m$ random vectors:

$$\widehat{\text{tr}(H^2)} = \frac{1}{m}\sum_{j=1}^m \|H\mathbf{v}_j\|^2$$

requiring $m$ additional HVPs. With $m = 10$, the relative standard error is typically <5%.

**Total cost for PFI of one tier:** $2 + m$ HVPs. **Crucially, the denominator $\text{tr}(H^2)$ is shared across all tiers**, so comparing PFI across tiers requires only 2 additional HVPs per tier beyond the shared $m$ HVPs.

**Algorithm 4: PFI Computation**

**Input:** Model, directions $\hat{\mathbf{d}}_1, \hat{\mathbf{d}}_2$, $\lambda_1$ (from Tier 3, if available), $m$ Hutchinson samples.

**Output:** PFI-S, PFI-C.

1. Compute $\mathbf{h}_i = H \hat{\mathbf{d}}_i$ for $i = 1, 2$ (2 HVPs).
2. Numerator: $\text{num} = \|\mathbf{h}_1\|^2 + \|\mathbf{h}_2\|^2$.
3. Also: $\kappa_i = \hat{\mathbf{d}}_i^\top \mathbf{h}_i$ (curvatures along projection axes).
4. For $j = 1, \ldots, m$: sample $\mathbf{v}_j \sim \mathcal{N}(0, I)$, compute $\|H\mathbf{v}_j\|^2$.
5. $\text{tr}(H^2) \approx \frac{1}{m}\sum_j \|H\mathbf{v}_j\|^2$.
6. $\text{PFI-S} = \text{num} / \text{tr}(H^2)$.
7. $\text{PFI-C} = \max(|\kappa_1|, |\kappa_2|) / |\lambda_1|$.

---

## 6. Multi-Model Shared Projection (MMSP)

### 6.1 Method A: Trajectory-PCA Projection

**Applicable scenario:** Checkpoints from the same training run (same architecture, same initialization).

Given checkpoints $\{\theta_0, \theta_1, \ldots, \theta_T\}$ and the trajectory PCA directions $\mathbf{p}_1, \mathbf{p}_2$:

1. Project each checkpoint: $(x_t, y_t) = (\langle \theta_t - \bar{\theta}, \mathbf{p}_1 \rangle, \langle \theta_t - \bar{\theta}, \mathbf{p}_2 \rangle)$.
2. Define the grid range to encompass all projected points with margin.
3. Evaluate $L(\bar{\theta} + \alpha \mathbf{p}_1 + \beta \mathbf{p}_2)$ on the grid.
4. Overlay the projected checkpoint positions on the surface plot.

### 6.2 Method B: Anchor-Point Projection

**Applicable scenario:** 2–3 models with the same architecture (e.g., base vs. fine-tuned).

**For two models $\theta_A, \theta_B$:**

1. First axis: $\mathbf{d}_1 = \theta_B - \theta_A$ (direction between models).
2. Apply TADN normalization to $\mathbf{d}_1$, then normalize to unit norm.
3. Second axis: TADN-normalized random direction orthogonalized against $\hat{\mathbf{d}}_1$, or top Hessian eigenvector orthogonal to $\hat{\mathbf{d}}_1$.
4. Evaluate on a grid centered at the midpoint.

**For three models $\theta_A, \theta_B, \theta_C$:**
1. $\mathbf{e}_1 = \theta_B - \theta_A$.
2. $\mathbf{e}_2 = (\theta_C - \theta_A) - \text{proj}_{\mathbf{e}_1}(\theta_C - \theta_A)$ (Gram-Schmidt).
3. Apply TADN and normalize.

### 6.3 Method C: Independent Landscape Comparison

**Applicable scenario:** Models with different architectures.

1. Generate independent 2D landscapes for each model using the same direction selection tier and evaluation dataset.
2. Extract comparable geometric features:
   - **Basin width $w_{\text{eff}}$:** Effective diameter of the $\delta$-sublevel set.
   - **Basin depth $D$:** $\max f - f(0,0)$.
   - **Curvature at minimum:** $\kappa_1, \kappa_2$ along projection axes.
   - **Roughness $R$:** Standard deviation of residuals after quadratic fit.
   - **Asymmetry $A$:** Mean directional deviation.
3. Present side-by-side visualizations with matched color scales.

**Connection to Mode Connectivity:** Recent work on Generalized Linear Mode Connectivity (arXiv:2506.22712) shows that independently trained transformers can be connected by low-loss paths after accounting for parameter space symmetries. For models from the same training run (Method A), the landscape between checkpoints should be smooth; for independently trained models (Method C), barriers may exist, and our independent comparison approach correctly avoids naive parameter-space interpolation.

---

## 7. Efficient Computation Pipeline

### 7.1 Curvature-Aware Adaptive Scale Selection

A key practical issue is choosing the appropriate scale range $[\alpha_{\min}, \alpha_{\max}]$ for the visualization grid. Too large a range shows only steep walls; too small shows only noise.

**Algorithm 5: Curvature-Aware Scale Selection**

1. Compute $\lambda_{\max}$ from Tier 3 (or estimate via a few power iterations).
2. Set the characteristic curvature length: $\ell_{\text{char}} = 1 / \sqrt{|\lambda_{\max}|}$.
3. Use grid range $[-k \cdot \ell_{\text{char}}, k \cdot \ell_{\text{char}}]$ with $k = 3$ (default).

This ensures:
- The visualization covers approximately 3 standard deviations of the local quadratic approximation.
- Different models and tiers are visualized at scales proportional to their curvature.
- The interesting region (basin boundary, curvature transition) is always visible.

When $\lambda_{\max}$ is not available (Tier 1 only), we use a heuristic: start with a coarse grid at $[-1, 1]$, check if the boundary loss is >10× the center loss (grid too large) or <1.1× (grid too small), and adjust.

### 7.2 Representative Data Subset Selection

For loss surface evaluation, we select a fixed subset $\mathcal{D}_{\text{eval}}$ of $n_{\text{eval}}$ tokens (default: 4,096 tokens):

1. Sample $n_{\text{candidate}}$ sequences from the full dataset.
2. Evaluate the loss on each sequence.
3. Select sequences that span the loss distribution (stratified sampling by loss quantile).
4. Verify correlation with full-dataset loss for random perturbations ($r > 0.99$).

### 7.3 Mixed-Precision Evaluation

- Forward passes: `torch.bfloat16` with `torch.autocast`.
- Direction vectors: stored in `float32` for normalization, cast to model dtype for perturbation.
- **Critical: exact parameter restoration.** Rather than add-then-subtract (which accumulates bfloat16 rounding errors), we save original parameters and restore exactly at each grid point.
- HVP computation: `float32` throughout for numerical stability.

### 7.4 Parallelization and In-Place Perturbation

Grid points are independent and trivially parallelizable. For in-place perturbation:

```python
def evaluate_at_grid_point(model, alpha, beta, d1, d2, data, original_params):
    # Exact restoration from saved parameters
    for name, param in model.named_parameters():
        param.data.copy_(original_params[name])
    # Apply perturbation
    for name, param in model.named_parameters():
        if name in d1:
            param.data.add_(alpha * d1[name] + beta * d2[name])
    # Evaluate
    with torch.no_grad():
        loss = compute_loss(model, data)
    return loss.item()
```

### 7.5 Computational Cost Estimates

| Operation | 0.6B (1 GPU) | 1.1B (1 GPU) | 7B (8 GPUs) |
|-----------|:---:|:---:|:---:|
| Single forward pass | ~0.05s | ~0.1s | ~0.5s |
| 21×21 grid | ~22s | ~44s | ~3.7min |
| 51×51 grid | ~2.2min | ~4.3min | ~21.7min |
| Tier 2 (N=100 grads) | ~10s | ~20s | ~100s |
| Tier 3 (50 HVP iters) | ~25s | ~50s | ~250s |
| PFI computation (10 Hutchinson) | ~6s | ~12s | ~60s |

---

## 8. Loss Landscape Geometric Feature Extraction

### 8.1 Flatness Metrics

1. **Local curvature ratio:** $\rho = \kappa_1 / \kappa_2$ (anisotropy of curvature).
2. **Basin flatness index:** $\Phi = \frac{1}{|\mathcal{B}_\delta|} \sum_{(\alpha, \beta) \in \mathcal{B}_\delta} (f(\alpha, \beta) - f(0,0))$.
3. **Effective basin diameter:** $w_{\text{eff}} = 2\sqrt{|\mathcal{B}_\delta| / \pi}$.

### 8.2 Surface Complexity Metrics

1. **Roughness:** $R = \text{std}(f - q)$ where $q$ is the best-fit quadratic.
2. **Convexity ratio:** Fraction of grid points with positive-definite local Hessian.

---

## 9. Feasibility Analysis

### 9.1 Memory Requirements

| Model Size | Weights (bf16) | 2 Directions (f32) | Peak Memory |
|:---:|:---:|:---:|:---:|
| 0.6B | ~1.2 GB | ~4.8 GB | ~8 GB |
| 1.1B | ~2.2 GB | ~8.8 GB | ~14 GB |
| 7B | ~14 GB | ~56 GB | ~80 GB |

For 7B, directions can use bfloat16 (halving memory) or be loaded chunk-by-chunk.

### 9.2 Flash Attention Compatibility

- **Grid evaluation:** Use Flash Attention for speed.
- **HVP (Tier 3):** Switch to eager attention (no `create_graph` support in Flash).
- **Gradient computation (Tier 2):** Flash Attention compatible (standard backward).

---

## 10. Summary of Algorithmic Pipeline

**Full LLMScape Pipeline:**

**Input:** Model(s) $\theta^*$, evaluation dataset $\mathcal{D}$, direction selection tier, grid resolution $G$.

1. **Data preparation:** Select representative subset (Section 7.2).
2. **Direction selection:** Apply chosen tier (Section 4.2) to obtain $\mathbf{d}_1, \mathbf{d}_2$.
3. **Normalization:** Apply TADN (Section 3.3).
4. **Scale selection:** Determine grid range via curvature-aware method (Section 7.1) or user specification.
5. **Grid evaluation:** For each $(\alpha_i, \beta_j)$, compute $f(\alpha_i, \beta_j)$ with exact parameter restoration (Section 7.3).
6. **Visualization:** Generate 2D surface plot.
7. **Feature extraction:** Compute geometric metrics (Section 8).
8. **PFI assessment:** Compute PFI to quantify projection quality (Section 5).
9. **Multi-model overlay (if applicable):** Project checkpoints or anchor points (Section 6).

**Output:** 2D loss landscape plot(s), geometric feature summary, PFI score, projected model positions (if applicable).
