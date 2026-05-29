---
layout: post
title: "Robust Linear Regression"
date: 2021-06-06 10:00:00
tag:
- Statistics
- Regression
- Robust Estimation
projects: true
blog: true
published: false
author: YingZhang
description: A review of robust regression methods from M-estimation to modern penalized approaches, connecting penalty functions to classical robust estimators through the She and Owen framework.
fontsize: 23pt
---

{% include mathjax_support.html %}

OLS minimizes $\sum r_i^2$, which gives every observation quadratic influence on $\hat{\beta}$. A single extreme point can arbitrarily distort the fit. Robust regression replaces or modifies this objective to limit the influence of any single observation.

The challenge has two distinct faces:

- **Vertical outliers**: extreme $y$ values at normal $x$ locations. These produce large residuals and are easy to spot under a clean fit.
- **Leverage points**: extreme $x$ values where $y$ may not follow the clean pattern. These are dangerous because they pull the fitted line toward themselves, making their own residuals small. The outlier masks itself.

Different methods handle these two cases differently. This post traces the evolution of robust regression: from classical M-estimation (which handles vertical outliers), through high-breakdown methods (which handle leverage points), to modern penalized and stability-based approaches (which handle both in high dimensions). Along the way, we build up penalty functions as reusable building blocks and show how She and Owen (2011) unified them under a single framework.

## Classical Robust Regression

### M-Estimation

M-estimation (Huber, 1964) replaces the squared loss with a loss function $\rho$ that grows more slowly in the tails:

$$\hat{\beta}_M = \arg\min_\beta \sum_{i=1}^n \rho\left(\frac{r_i}{\hat{\sigma}}\right), \quad r_i = y_i - x_i^T\beta$$

where $\hat{\sigma}$ is a robust scale estimate (e.g., median absolute deviation of residuals). The derivative $\psi = \rho'$ defines how each residual contributes to the estimating equation $\sum \psi(r_i/\hat{\sigma}) \, x_i = 0$.

**Huber loss** is the most common choice:

$$\rho_{\text{Huber}}(r) = \begin{cases} \frac{1}{2}r^2 & \lvert r \rvert \leq c \\ c\lvert r \rvert - \frac{1}{2}c^2 & \lvert r \rvert > c \end{cases}$$

Quadratic for small residuals (like OLS), linear for large residuals (like LAD). The constant $c$ controls the transition, with $c = 1.345$ giving 95% efficiency at the normal model.

**Tukey bisquare** is more aggressive:

$$\rho_{\text{bisquare}}(r) = \begin{cases} \frac{c^2}{6}\left[1 - \left(1 - \frac{r^2}{c^2}\right)^3\right] & \lvert r \rvert \leq c \\ \frac{c^2}{6} & \lvert r \rvert > c \end{cases}$$

The loss function is bounded: observations with $\lvert r \rvert > c$ contribute a constant to the objective, so their influence on $\hat{\beta}$ drops to zero. This is called a **redescending** $\psi$-function.

<figure style="text-align: center;">
  <img src="/assets/images/robust_loss_functions.png" alt="Comparison of squared loss, Huber loss, and Tukey bisquare loss" style="max-width: 90%;">
  <figcaption style="margin-top: 1.5em; font-size: 0.9em; color: #555;">Squared loss grows without bound, Huber loss grows linearly beyond c, and Tukey bisquare flattens completely.</figcaption>
</figure>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">IRLS algorithm and derivation</span></summary>

<p>The first-order condition for M-estimation is:</p>

$$\sum_{i=1}^n \psi\left(\frac{r_i}{\hat{\sigma}}\right) x_i = 0$$

<p>Define weights $w_i = \psi(r_i/\hat{\sigma}) / (r_i/\hat{\sigma})$. Then the estimating equation becomes:</p>

$$\sum_{i=1}^n w_i r_i x_i = 0 \quad \Longleftrightarrow \quad X^T W (y - X\beta) = 0$$

<p>which is the normal equation for weighted least squares with weight matrix $W = \text{diag}(w_1, \ldots, w_n)$. This suggests an iterative algorithm:</p>

<ol>
<li>Start with $\hat{\beta}^{(0)} = \hat{\beta}_{\text{OLS}}$.</li>
<li>Compute residuals $r_i^{(t)} = y_i - x_i^T \hat{\beta}^{(t)}$.</li>
<li>Compute weights $w_i^{(t)} = \psi(r_i^{(t)}/\hat{\sigma}) / (r_i^{(t)}/\hat{\sigma})$.</li>
<li>Solve weighted least squares: $\hat{\beta}^{(t+1)} = (X^T W^{(t)} X)^{-1} X^T W^{(t)} y$.</li>
<li>Repeat steps 2-4 until convergence.</li>
</ol>

<p>For Huber loss, the weights are $w_i = \min(1, c/\lvert r_i/\hat{\sigma} \rvert)$: full weight for small residuals, downweighted for large ones. For Tukey bisquare, $w_i = (1 - (r_i/c\hat{\sigma})^2)^2$ for $\lvert r_i \rvert \leq c\hat{\sigma}$ and $w_i = 0$ otherwise: extreme residuals get zero weight entirely.</p>

<p>Convergence is guaranteed for convex $\rho$ (Huber) since each IRLS step reduces the objective. For non-convex $\rho$ (bisquare), convergence to a local minimum depends on initialization.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Strengths**: simple, fast (IRLS converges in a few iterations), 95% efficient at normal model.

**Limitations**: cannot handle leverage points. The residual-based reweighting is blind to outliers that mask themselves by distorting the fit.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why M-estimation fails for leverage points</span></summary>

<p>M-estimation starts from OLS and iteratively reweights based on residual size. A leverage point (extreme $x$, anomalous $y$) pulls the OLS fit toward itself, producing a <em>small</em> residual for that point under the corrupted fit. IRLS sees a small residual, assigns full weight, and stays stuck.</p>

<p><strong>Concrete example.</strong> Fit $y = x$ to 20 clean points from $x \in [0, 5]$, then add one point at $(100, 0)$. OLS tilts the line from slope $\approx 1$ toward the leverage point. Under the tilted fit, the leverage point has a modest residual (the line passes somewhat near it), while many clean points now have larger residuals. IRLS downweights the <em>clean</em> points and keeps the leverage point at full weight, reinforcing the corruption.</p>

<p>The root cause: M-estimation only examines residuals, not the design space. A point with extreme $x$ and small residual looks fine from the residual perspective alone. To detect leverage points, you need either a method that looks at the $x$-space directly or a high-breakdown initialization that is not corrupted to begin with.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### High-Breakdown Estimators

M-estimation fails for leverage points because it starts from OLS, which is already corrupted. The solution: find an initial estimate that does not give any single point too much influence. This requires **high-breakdown** methods that resist up to 50% contamination.

#### S-Estimation

S-estimation (Rousseeuw and Yohai, 1984) minimizes a robust version of the standard deviation of the residuals:

$$\hat{\beta}_S = \arg\min_\beta \hat{\sigma}(r_1(\beta), \ldots, r_n(\beta))$$

This is conceptually the same goal as OLS (find $\beta$ that makes residuals small), but with a robust measure of "small." OLS minimizes $\sum r_i^2$, which is equivalent to minimizing the standard deviation. $\hat{\sigma}$ here is the **M-scale**, which replaces the standard deviation to resist outliers. The standard deviation solves $\frac{1}{n}\sum_{i=1}^n (r_i / \sigma)^2 = 1$, i.e., $\rho(t) = t^2$. But $t^2$ is unbounded, so a single outlier with a huge residual can inflate $\sigma$ arbitrarily. The M-scale replaces $t^2$ with a bounded function $\rho$ (typically Tukey bisquare, which rises from 0 to a maximum of 1 and stays there):

$$\frac{1}{n}\sum_{i=1}^n \rho\left(\frac{r_i}{\hat{\sigma}}\right) = \delta$$

Each observation contributes at most $\rho_{\max} = 1$ to the left-hand side, regardless of how large its residual is. This is the source of robustness: a single outlier can change the average by at most $1/n$.

**How to solve for $\hat{\sigma}$.** The left-hand side $g(\sigma) = \frac{1}{n}\sum \rho(r_i / \sigma)$ is monotonically decreasing in $\sigma$: as $\sigma$ grows, $r_i/\sigma$ shrinks toward zero, and $\rho$ returns smaller values. So there is a unique $\hat{\sigma}$ satisfying $g(\hat{\sigma}) = \delta$, and it can be found by fixed-point iteration:

$$\hat{\sigma}^{(t+1)} = \hat{\sigma}^{(t)}\sqrt{\frac{1}{n\delta}\sum_{i=1}^n \rho\left(\frac{r_i}{\hat{\sigma}^{(t)}}\right) \cdot \frac{r_i^2 / \hat{\sigma}^{(t)2}}{\rho(r_i / \hat{\sigma}^{(t)}) / (r_i / \hat{\sigma}^{(t)})^2}}$$

In practice, a simpler form is used. Define weights $w(t) = \rho(t)/t^2$ (the ratio of the bounded $\rho$ to the unbounded squared function). Then:

$$\hat{\sigma}^{(t+1)2} = \frac{1}{n\delta}\sum_{i=1}^n w\left(\frac{r_i}{\hat{\sigma}^{(t)}}\right) r_i^2$$

This is a weighted variance where the weights depend on the current scale estimate. If $\hat{\sigma}^{(t)}$ is too small, $r_i/\hat{\sigma}$ is large, $\rho$ saturates, and the right-hand side exceeds $\hat{\sigma}^{(t)2}$, pushing the estimate up. If $\hat{\sigma}^{(t)}$ is too large, $\rho$ values are small, and the estimate shrinks. Convergence is fast, typically 2-3 iterations from a reasonable starting value (e.g., the MAD of residuals).

**Why minimize the M-scale over $\beta$?** A leverage point pulling $\hat{\beta}$ toward itself reduces its own residual but inflates the residuals of many clean points. Because $\rho$ is bounded, the single small residual contributes at most 1 to the sum, while the many inflated clean residuals also each contribute near 1. The net effect: the M-scale is larger at the corrupted $\beta$ than at the true $\beta$. Minimizing the M-scale therefore finds the fit that makes the bulk of the data tightest, rejecting the corrupted solution.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Breakdown point and the FAST-S algorithm</span></summary>

<p>The <strong>breakdown point</strong> is the largest fraction of contaminated observations an estimator can tolerate before it can be driven to an arbitrary value. OLS has breakdown point $1/n$ (a single point can cause arbitrarily bad estimates). S-estimation with $\delta = 0.5$ achieves the maximum possible breakdown point of 50%.</p>

<p><strong>Why $\delta = 0.5$ gives 50% breakdown.</strong> Suppose half the observations are outliers with enormous residuals. Their $\rho(r_i/\hat{\sigma})$ values are all at the maximum (1). The clean half has small residuals, contributing $\rho$ values near 0. The average is approximately $0.5 \cdot 1 + 0.5 \cdot 0 = 0.5 = \delta$. So the defining equation is just barely satisfied. If more than 50% are outliers, the average exceeds $\delta$ for any finite $\hat{\sigma}$, and the M-scale is driven to infinity. This is the breakdown: the estimator signals that the data is too contaminated to estimate a scale.</p>

<p>The tradeoff: higher breakdown requires more aggressive bounding of $\rho$, which reduces statistical efficiency. S-estimation with 50% breakdown has only about 28% efficiency at the normal model, meaning its standard errors are nearly twice those of OLS when the data is actually clean.</p>

<p><strong>FAST-S algorithm</strong> (Salibian-Barrera and Yohai, 2006):</p>

<ol>
<li>Draw $M$ random subsets of $p+1$ observations (the minimum needed for a unique OLS fit).</li>
<li>Fit OLS on each subset to get candidate $\hat{\beta}^{(m)}$.</li>
<li>For each candidate, compute residuals $r_i = y_i - x_i^T\hat{\beta}^{(m)}$ for all $n$ observations, and run 2-3 iterations of the M-scale fixed-point iteration above. This gives a rough M-scale $\hat{\sigma}^{(m)}$ for each candidate without requiring full convergence.</li>
<li>Keep the $k$ candidates with the smallest M-scale values.</li>
<li>For the best candidates, alternate between updating $\hat{\beta}$ (via weighted least squares with weights from $\rho'$) and updating $\hat{\sigma}$ (via the fixed-point iteration) until convergence.</li>
</ol>

<p>Why random subsampling defeats leverage points: if fewer than 50% of observations are contaminated, a majority of random $p+1$-subsets will contain only clean points, giving candidates $\hat{\beta}^{(m)}$ close to the true value. Under these candidates, leverage points have large residuals and are detected.</p>

<p>The number of random subsets needed: to ensure at least one clean subset with probability $1 - \alpha$, you need $M \geq \log(\alpha) / \log(1 - (1-\epsilon)^{p+1})$ subsets, where $\epsilon$ is the contamination fraction. For $\epsilon = 0.25$, $p = 10$, $\alpha = 0.01$: $M \approx 820$.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Pros:** 50% breakdown point, handles leverage points, more efficient than LTS (28% vs 8%). **Cons:** computationally expensive (random subsampling scales exponentially with $p$), low efficiency on its own (used mainly as initialization for MM-estimation).

#### Least Trimmed Squares (LTS)

LTS (Rousseeuw, 1984) finds the subset of $h$ observations whose OLS fit has the smallest sum of squared residuals:

$$\hat{\beta}_{\text{LTS}} = \arg\min_\beta \sum_{i=1}^h r_{(i)}^2(\beta)$$

where $r_{(1)}^2 \leq r_{(2)}^2 \leq \cdots \leq r_{(n)}^2$ are the ordered squared residuals. The default $h = \lfloor n/2 \rfloor + \lfloor (p+1)/2 \rfloor$ gives 50% breakdown.

LTS is the regression analogue of the MCD estimator for covariance matrices. Both search for the tightest-fitting subset, the difference being that LTS measures fit in residual space while MCD measures fit in the full data space.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">FAST-LTS algorithm</span></summary>

<p>Exact LTS requires evaluating $\binom{n}{h}$ subsets, which is computationally infeasible. FAST-LTS (Rousseeuw and Van Driessen, 2006) uses the same concentration-step (C-step) idea as FAST-MCD:</p>

<ol>
<li>Draw a random subset of $p+1$ observations and fit OLS.</li>
<li><strong>C-step:</strong> compute residuals for all $n$ observations, select the $h$ with smallest squared residuals, refit OLS on those $h$ points.</li>
<li>Repeat step 2 until the trimmed sum of squares converges.</li>
<li>Repeat from step 1 with many random starts, keep the best.</li>
</ol>

<p>Each C-step is guaranteed to decrease (or maintain) the trimmed objective. Starting from many random subsets ensures the global optimum is found with high probability.</p>

<p><strong>LTS vs S-estimation.</strong> Both achieve 50% breakdown. LTS is simpler conceptually (just trim the worst residuals) but has lower efficiency (about 8% at the normal model without reweighting). S-estimation is more efficient (28%) because it downweights rather than hard-trims. In practice, both are used primarily as initial estimates for the second stage of MM-estimation, where efficiency is recovered.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Pros:** 50% breakdown, conceptually simple (trim worst residuals), easy to explain. **Cons:** very low efficiency (8%) without reweighting, computationally expensive (same subsampling cost as S-estimation), hard-trims rather than downweights (wastes information from borderline observations).

### MM-Estimation: Combining Breakdown and Efficiency

S-estimation and LTS provide high breakdown but low efficiency. M-estimation provides high efficiency but low breakdown. MM-estimation (Yohai, 1987) combines both through a two-stage procedure.

**Stage 1: S-estimation (find the clean fit).** Compute $\hat{\beta}_S$ and $\hat{\sigma}_S$ using S-estimation as described above. The result is highly robust (50% breakdown) but statistically inefficient (~28% efficiency).

**Stage 2: M-estimation (refine for efficiency).** Using $\hat{\beta}_S$ as the starting point and $\hat{\sigma}_S$ from Stage 1 (held fixed), run IRLS with a bisquare $\psi$-function tuned for high efficiency (e.g., 95% at the normal model). This means a wider $\rho$ than in Stage 1, so more observations receive substantial weight and statistical precision improves. Because the starting point is already close to the clean solution, leverage points have large standardized residuals $r_i / \hat{\sigma}_S$ and are correctly downweighted by $\psi$.

The scale $\hat{\sigma}_S$ must stay fixed at the Stage 1 value. If it were re-estimated in Stage 2, a leverage point could deflate the scale, inflate its own weight, and pull $\hat{\beta}$ toward itself, breaking the breakdown guarantee.

The result has both 50% breakdown (inherited from Stage 1) and ~95% efficiency at the normal model (from Stage 2's refined estimation).

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why the two stages are both necessary</span></summary>

<p><strong>Stage 1 alone is insufficient</strong> because S-estimation sacrifices efficiency for breakdown. Its standard errors are nearly twice those of OLS, meaning you lose statistical power even when the data is clean.</p>

<p><strong>Stage 2 alone is insufficient</strong> because M-estimation from OLS initialization cannot detect leverage points (the masking problem described above).</p>

<p><strong>Together they work</strong> because the initialization problem and the efficiency problem are solved separately:</p>

<ul>
<li>Stage 1 provides a starting $\hat{\beta}$ that is close to the truth, breaking the circular dependency between "clean fit" and "outlier detection."</li>
<li>Stage 2 refines this estimate using all the data (with appropriate weights), recovering the efficiency lost in Stage 1.</li>
<li>The scale $\hat{\sigma}$ is fixed at the Stage 1 value to maintain the breakdown point. If $\hat{\sigma}$ were re-estimated in Stage 2, a leverage point could deflate the scale estimate and receive undeserved weight.</li>
</ul>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Pros:** 50% breakdown + ~95% efficiency, the best of both worlds for low-dimensional robust regression. **Cons:** computationally expensive (Stage 1 requires random subsampling), does not scale well to high-dimensional settings ($p$ large), two tuning parameters (breakdown level for Stage 1, efficiency level for Stage 2).

## Penalty Functions As Building Blocks

The methods above modify the loss function or the fitting procedure. A parallel line of work uses **penalty functions** on the coefficients to enforce sparsity (variable selection) or on the residuals (outlier detection). These penalties appear throughout the rest of the post, so we define them here.

Each penalty induces a **thresholding rule**: given a noisy estimate $z$ (an OLS coefficient or a residual), the penalized estimate $\hat{\theta}$ is determined by the penalty's shape.

### LASSO ($\ell_1$ Penalty)

Tibshirani (1996):

$$P(\lvert \theta \rvert) = \lambda\lvert \theta \rvert$$

**Thresholding rule** (soft thresholding):

$$\hat{\theta} = \text{sign}(z)\max(\lvert z \rvert - \lambda, 0)$$

Small $z$ maps to exactly zero. Large $z$ is kept but shifted toward zero by $\lambda$. Every nonzero estimate carries a shrinkage bias of magnitude $\lambda$.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why LASSO has shrinkage bias and lacks the oracle property</span></summary>

<p>Consider the oracle estimator: if you knew which $\beta_j \neq 0$ in advance, you would run OLS on just those variables. The oracle achieves both correct selection and efficient estimation of the nonzero coefficients.</p>

<p>LASSO applies the same penalty $\lambda \lvert \beta_j \rvert$ to every coefficient regardless of size. For a large true $\beta_j$, the penalty still shrinks the estimate by $\lambda$, so $\hat{\beta}_j^{\text{LASSO}} \approx \beta_j - \lambda \cdot \text{sign}(\beta_j)$. This bias does not vanish even as $n \to \infty$ (unless $\lambda \to 0$, but then you lose sparsity).</p>

<p>Formally, the oracle property requires: (1) correct variable selection with probability approaching 1, and (2) asymptotic normality of the nonzero coefficient estimates at the oracle rate $\sqrt{n}(\hat{\beta}_S - \beta_S) \to N(0, \sigma^2 (X_S^T X_S)^{-1})$. LASSO satisfies (1) under irrepresentability conditions but fails (2) because of the persistent bias.</p>

<p>In practice, a common fix is <strong>adaptive LASSO</strong> (Zou, 2006): use data-dependent weights $w_j = 1/\lvert \hat{\beta}_j^{\text{init}} \rvert^\gamma$ so that large coefficients get smaller penalties. This achieves the oracle property.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Hard Threshold ($\ell_0$-like Penalty)

$$P(\lvert \theta \rvert) = \frac{\lambda^2}{2} \cdot \mathbb{1}(\theta \neq 0)$$

**Thresholding rule** (hard thresholding):

$$\hat{\theta} = z \cdot \mathbb{1}(\lvert z \rvert > \lambda)$$

Small $z$ maps to zero. Large $z$ is kept at exactly its original value with no shrinkage. But the rule is discontinuous at $\lvert z \rvert = \lambda$: a tiny perturbation in the data can cause the estimate to jump from 0 to $\lambda$.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Computational intractability and practical approximations</span></summary>

<p>The $\ell_0$ penalty counts the number of nonzero parameters, making the optimization problem:</p>

$$\min_\beta \frac{1}{2}\|y - X\beta\|^2 + \lambda^2 \|\beta\|_0$$

<p>This is NP-hard in general because it requires searching over $2^p$ possible support sets. For orthogonal designs ($X^TX = I$), the solution decouples into $p$ independent problems with the hard thresholding rule above. For general $X$, exact solutions require branch-and-bound or mixed-integer programming, which is feasible only for small $p$.</p>

<p><strong>Iterative hard thresholding (IHT)</strong> approximates the $\ell_0$ solution by repeating two steps:</p>

<ol>
<li><strong>Gradient step:</strong> take a step in the direction that reduces the least-squares loss: $\beta^{(t+1/2)} = \beta^{(t)} + \eta \, X^T(y - X\beta^{(t)})$, where $\eta$ is the step size. This is the same update as gradient descent for OLS.</li>
<li><strong>Hard threshold:</strong> set all coefficients with $\lvert \beta_j^{(t+1/2)} \rvert < \lambda$ to zero, keep the rest unchanged. This enforces sparsity by killing small coefficients after each gradient update.</li>
</ol>

<p>The algorithm keeps cycling between "move toward the least-squares solution" (step 1) and "zero out anything too small" (step 2). Under restricted isometry conditions on $X$, IHT converges to a near-optimal solution.</p>

<p><strong>Alternative: use MCP instead of $\ell_0$.</strong> The Minimax Concave Penalty (defined in the next section) with small $\gamma$ closely approximates hard thresholding but is continuous, avoiding the discontinuous jump at $\lvert z \rvert = \lambda$. This makes optimization easier because coordinate descent converges reliably, whereas IHT can oscillate near the threshold.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### SCAD (Smoothly Clipped Absolute Deviation)

Fan and Li (2001). The penalty derivative defines three regions:

$$P'(\lvert \theta \rvert) = \lambda\left[\mathbb{1}(\lvert \theta \rvert \leq \lambda) + \frac{(a\lambda - \lvert \theta \rvert)_+}{(a-1)\lambda}\mathbb{1}(\lvert \theta \rvert > \lambda)\right], \quad a = 3.7$$

- $\lvert \theta \rvert \leq \lambda$: same penalty slope as LASSO.
- $\lambda < \lvert \theta \rvert \leq a\lambda$: penalty tapers off (quadratic transition).
- $\lvert \theta \rvert > a\lambda$: penalty is constant, so large coefficients incur no additional shrinkage.

**Thresholding rule** (firm thresholding):

$$\hat{\theta} = \begin{cases} \text{sign}(z)\max(\lvert z \rvert - \lambda, 0) & \lvert z \rvert \leq 2\lambda \\ \frac{(a-1)z - \text{sign}(z)a\lambda}{a - 2} & 2\lambda < \lvert z \rvert \leq a\lambda \\ z & \lvert z \rvert > a\lambda \end{cases}$$

Soft thresholding for small coefficients (sparsity), gradual transition for moderate coefficients, and identity for large coefficients (no bias).

### MCP (Minimax Concave Penalty)

Zhang (2010):

$$P(\lvert \theta \rvert; \lambda, \gamma) = \lambda \int_0^{\lvert \theta \rvert} \left(1 - \frac{t}{\gamma\lambda}\right)_+ dt = \begin{cases} \lambda\lvert \theta \rvert - \frac{\theta^2}{2\gamma} & \lvert \theta \rvert \leq \gamma\lambda \\ \frac{\gamma\lambda^2}{2} & \lvert \theta \rvert > \gamma\lambda \end{cases}$$

**Thresholding rule**:

$$\hat{\theta} = \begin{cases} 0 & \lvert z \rvert \leq \lambda \\ \frac{\text{sign}(z)(\lvert z \rvert - \lambda)}{1 - 1/\gamma} & \lambda < \lvert z \rvert \leq \gamma\lambda \\ z & \lvert z \rvert > \gamma\lambda \end{cases}$$

The parameter $\gamma > 1$ controls the transition speed. As $\gamma \to \infty$, MCP reduces to LASSO. As $\gamma \to 1^+$, MCP approaches hard thresholding. In practice, $\gamma = 3$ is a common default. Compared to SCAD, MCP improves in two ways: (1) it starts reducing bias immediately once $\lvert z \rvert > \lambda$, whereas SCAD applies full LASSO shrinkage up to $\lvert z \rvert = 2\lambda$, and (2) it is the least nonconvex penalty that still achieves unbiasedness for large coefficients, minimizing local optima issues during optimization.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Oracle property of SCAD and MCP</span></summary>

<p>The <strong>oracle property</strong> (Fan and Li, 2001) means that with probability approaching 1 as $n \to \infty$:</p>

<ol>
<li><strong>Selection consistency:</strong> the estimated support $\hat{S} = \{j : \hat{\beta}_j \neq 0\}$ equals the true support $S_0 = \{j : \beta_j \neq 0\}$.</li>
<li><strong>Asymptotic efficiency:</strong> the nonzero estimates are asymptotically normal at the oracle rate:
$$\sqrt{n}(\hat{\beta}_{S_0} - \beta_{S_0}) \xrightarrow{d} N(0, \sigma^2 (X_{S_0}^T X_{S_0}/n)^{-1}).$$</li>
</ol>

<p>Why SCAD/MCP achieve this but LASSO does not: for large $\lvert \beta_j \rvert$, the SCAD/MCP penalty derivative is zero, meaning the estimating equation for that coefficient reduces to the OLS score equation. The penalty does not distort the estimate of large coefficients. By contrast, LASSO always has penalty derivative $\lambda$, which introduces a persistent first-order bias.</p>

<p>The conditions required for the oracle property include: (1) $\lambda \to 0$ and $\sqrt{n}\lambda \to \infty$ (the penalty vanishes but not too fast), (2) a minimum signal condition $\min_{j \in S_0} \lvert \beta_j \rvert \gg \lambda$ (true coefficients are well-separated from zero), and (3) regularity conditions on $X^TX/n$.</p>

<p><strong>Why LASSO fails property (2) specifically.</strong> Consider a true coefficient $\beta_j = 5$. LASSO estimates it as approximately $\hat{\beta}_j \approx 5 - \lambda$. If $\lambda = 1$, you get $\hat{\beta}_j \approx 4$, always biased by $\lambda$ regardless of sample size. SCAD and MCP apply zero penalty derivative for coefficients this large, so the estimate converges to 5 (the OLS estimate on the correct support), achieving the same variance as if you knew the true support from the start.</p>

<p><strong>SCAD vs MCP in practice.</strong> SCAD (Fan and Li, 2001) was the first to prove the oracle property, using a three-region penalty with default $a = 3.7$. MCP (Zhang, 2010) achieves the same oracle property with a simpler two-region penalty controlled by $\gamma$. MCP has a cleaner form: as $\gamma \to \infty$ it reduces to LASSO, as $\gamma \to 1^+$ it approaches hard thresholding, giving a single knob to interpolate between the two extremes. Both are solved via coordinate descent with similar computational cost to LASSO. In practice, the two perform similarly; MCP with $\gamma = 3$ is a common default choice.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Comparison Of Penalty Functions

| | LASSO ($\ell_1$) | Hard Threshold ($\ell_0$) | SCAD | MCP |
|---|---|---|---|---|
| **Convex** | Yes | No | No | No |
| **Shrinkage bias** | Yes, always | None | None for large coefs | None for large coefs |
| **Oracle property** | No | Yes (if solvable) | Yes | Yes |
| **Continuity** | Continuous | Discontinuous | Continuous | Continuous |
| **Computation** | Global optimum via coordinate descent | NP-hard in general | Local optima; coordinate descent or LLA | Local optima; coordinate descent |
| **Tuning parameters** | $\lambda$ | $\lambda$ | $\lambda, a$ ($a=3.7$ default) | $\lambda, \gamma$ ($\gamma=3$ default) |

<figure style="text-align: center;">
  <img src="/assets/images/thresholding_rules.png" alt="Thresholding rules of LASSO, Hard Threshold, SCAD, and MCP" style="max-width: 90%;">
  <figcaption style="font-size: 0.9em; color: #555;">Thresholding rules compared. LASSO (blue) maintains a constant gap from the identity line, the shrinkage bias. Hard thresholding (red) jumps discontinuously at $\lvert z \rvert = \lambda$. SCAD (green) and MCP (orange) smoothly merge back to the identity for large coefficients, eliminating bias.</figcaption>
</figure>

## Penalties Meet Robustness: She and Owen

She and Owen (2011) unified robust regression and penalized estimation by showing that applying a penalty to an **outlier vector** $\gamma$ is equivalent to classical M-estimation. The model:

$$y = X\beta + \gamma + \varepsilon$$

where $\gamma \in \mathbb{R}^n$ is sparse. Most entries are zero ($\gamma_i = 0$ for clean observations), and $\gamma_i \neq 0$ marks observation $i$ as an outlier with shift $\gamma_i$. The estimation problem:

$$\min_{\beta, \gamma} \frac{1}{2n}\|y - X\beta - \gamma\|^2 + \lambda \sum_{i=1}^n P(\lvert \gamma_i \rvert)$$

### Penalty-Estimator Correspondence

For fixed $\beta$, the optimal $\hat{\gamma}_i$ is determined by thresholding the residual $r_i = y_i - x_i^T\beta$. Different penalties on $\gamma$ recover different classical robust estimators:

| Penalty on $\gamma$ | Thresholding rule on residuals | Classical equivalent |
|---|---|---|
| $\ell_1$ (LASSO) | Soft threshold: shrink large residuals by $\lambda$ | Huber M-estimation |
| $\ell_0$ (hard threshold) | Keep or kill: zero out small residuals, keep large | Hard rejection / trimming |
| MCP / SCAD | Firm threshold: soft for moderate, no shrinkage for extreme | Redescending M-estimators (bisquare) |

This table is the central insight: the same penalty functions from variable selection (applied to $\beta$) become robust estimation tools when applied to $\gamma$.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Derivation: why the penalty-thresholding correspondence holds</span></summary>

<p>For fixed $\beta$, the optimization over $\gamma$ decomposes into $n$ independent scalar problems:</p>

$$\min_{\gamma_i} \frac{1}{2}(r_i - \gamma_i)^2 + \lambda P(\lvert \gamma_i \rvert)$$

<p>where $r_i = y_i - x_i^T\beta$. The solution $\hat{\gamma}_i$ is the proximal operator of $\lambda P$ applied to $r_i$.</p>

<p><strong>For $\ell_1$:</strong> $P(\lvert \gamma_i \rvert) = \lvert \gamma_i \rvert$. Taking the subgradient:</p>

$$\hat{\gamma}_i - r_i + \lambda \cdot \text{sign}(\hat{\gamma}_i) \ni 0$$

<p>For $\lvert r_i \rvert > \lambda$: $\hat{\gamma}_i = r_i - \lambda \cdot \text{sign}(r_i) = \text{sign}(r_i)(\lvert r_i \rvert - \lambda)$. For $\lvert r_i \rvert \leq \lambda$: $\hat{\gamma}_i = 0$. This is soft thresholding.</p>

<p>Now substituting back: the effective residual for estimating $\beta$ is $r_i - \hat{\gamma}_i$. For small residuals ($\lvert r_i \rvert \leq \lambda$), $\hat{\gamma}_i = 0$, so the effective residual is $r_i$ itself (quadratic loss). For large residuals ($\lvert r_i \rvert > \lambda$), $\hat{\gamma}_i = r_i - \lambda \cdot \text{sign}(r_i)$, so the effective residual is $\lambda \cdot \text{sign}(r_i)$ (constant magnitude). This is exactly Huber loss: quadratic for small residuals, linear growth (capped influence) for large ones.</p>

<p><strong>For MCP</strong> with parameters $(\lambda, \gamma)$: the penalty is $P(\lvert \gamma_i \rvert) = \lambda \lvert \gamma_i \rvert - \frac{\gamma_i^2}{2\gamma}$ for $\lvert \gamma_i \rvert \leq \gamma\lambda$, and constant $\frac{\gamma\lambda^2}{2}$ for $\lvert \gamma_i \rvert > \gamma\lambda$. The first-order condition for the scalar problem $\min_{\gamma_i} \frac{1}{2}(r_i - \gamma_i)^2 + \lambda P(\lvert \gamma_i \rvert)$ gives three regions:</p>

<ul>
<li>$\lvert r_i \rvert \leq \lambda$: the penalty dominates, so $\hat{\gamma}_i = 0$. The observation is treated as clean, and the effective residual is $r_i$ itself (quadratic loss, full influence).</li>
<li>$\lambda < \lvert r_i \rvert \leq \gamma\lambda$: the first-order condition is $(\hat{\gamma}_i - r_i) + \lambda\,\text{sign}(\hat{\gamma}_i) - \hat{\gamma}_i/\gamma = 0$, giving $\hat{\gamma}_i = \frac{\text{sign}(r_i)(\lvert r_i \rvert - \lambda)}{1 - 1/\gamma}$. The effective residual is $r_i - \hat{\gamma}_i$, which shrinks toward zero as $\lvert r_i \rvert$ grows. The observation's influence is being reduced.</li>
<li>$\lvert r_i \rvert > \gamma\lambda$: the penalty is flat (derivative is zero), so the objective is just $\frac{1}{2}(r_i - \gamma_i)^2$ plus a constant, giving $\hat{\gamma}_i = r_i$. The effective residual is $r_i - \hat{\gamma}_i = 0$. The observation has been completely absorbed by $\gamma$ and has <strong>zero influence</strong> on $\hat{\beta}$.</li>
</ul>

<p><strong>For SCAD</strong> with parameters $(\lambda, a)$, $a = 3.7$: the same three-region structure holds. For $\lvert r_i \rvert \leq 2\lambda$, it behaves like $\ell_1$ (soft thresholding on $\gamma_i$). For $2\lambda < \lvert r_i \rvert \leq a\lambda$, the penalty derivative tapers off, so $\hat{\gamma}_i = \frac{(a-1)r_i - \text{sign}(r_i)a\lambda}{a-2}$, and the effective residual shrinks. For $\lvert r_i \rvert > a\lambda$, the penalty is constant, so $\hat{\gamma}_i = r_i$ and the effective residual is zero.</p>

<p><strong>Connection to redescending M-estimators.</strong> The effective loss on observation $i$ is $L_{\text{eff}}(r_i) = \min_{\gamma_i} \frac{1}{2}(r_i - \gamma_i)^2 + \lambda P(\lvert \gamma_i \rvert)$. For MCP/SCAD, this loss is bounded: it grows quadratically for small $\lvert r_i \rvert$, slows down for moderate $\lvert r_i \rvert$, and reaches a finite maximum for large $\lvert r_i \rvert$ (since $\hat{\gamma}_i = r_i$ makes the squared term zero, leaving only the constant penalty). The derivative $\psi(r_i) = \partial L_{\text{eff}} / \partial r_i$ rises, peaks, then returns to zero. This is exactly a redescending $\psi$-function, the defining property of Tukey bisquare and similar estimators. By contrast, $\ell_1$ on $\gamma$ gives an effective loss that grows linearly forever (Huber loss), so its $\psi$-function plateaus but never redescends.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Identifiability and Algorithm

Without the penalty, the model $y = X\beta + \gamma + \varepsilon$ is underdetermined: the augmented design matrix $[X \mid I_n]$ has $p + n$ columns but only $n$ rows. The sparsity penalty resolves this by making nonzero $\gamma_i$ costly. At the optimum, $\varepsilon$ is dense and small (noise in every observation), while $\gamma$ is sparse and large (a few outlier shifts).

**Why not $\ell_2$ (ridge) on $\gamma$?** An $\ell_2$ penalty $\frac{\lambda}{2}\sum_i \gamma_i^2$ does not induce sparsity. Every $\gamma_i$ would be nonzero, and the solution is $\hat{\gamma} = \frac{1}{1+\lambda}(y - X\hat{\beta})$, which simply rescales all residuals. With $n$ free parameters in $\gamma$ absorbing a fraction of every residual, there is no clean/outlier separation, and $\beta$ is not properly identified. Identifiability in the She-Owen framework requires that the penalty forces most $\gamma_i$ to be exactly zero, so that $\beta$ is estimated from the clean observations where $\gamma_i = 0$. Only sparsity-inducing penalties ($\ell_1$, $\ell_0$, SCAD, MCP) achieve this.

**Alternating minimization:**

1. Initialize $\gamma = 0$.
2. Fix $\gamma$, solve for $\beta$: penalized regression of $(y - \gamma)$ on $X$.
3. Fix $\beta$, solve for $\gamma$: apply thresholding to residuals $r = y - X\beta$.
4. Repeat until convergence.

### Extension To High Dimensions

The full power of this framework emerges when you penalize both $\beta$ and $\gamma$:

$$\min_{\beta, \gamma} \frac{1}{2n}\|y - X\beta - \gamma\|^2 + \lambda_1 P_1(\beta) + \lambda_2 \sum_i P_2(\lvert \gamma_i \rvert)$$

This simultaneously does variable selection (via $P_1$ on $\beta$) and outlier detection (via $P_2$ on $\gamma$). For example, LASSO on $\beta$ + MCP on $\gamma$ selects variables while fully removing detected outliers.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Leverage points in the She-Owen framework</span></summary>

<p>The basic formulation with $\ell_1$ penalty on $\gamma$ alone does not solve the leverage point problem. With $\gamma = 0$ initialization, step 2 gives OLS for $\hat{\beta}$, which is already corrupted. Step 3 thresholds the residuals, but the leverage point's residual is small under the corrupted fit, so $\hat{\gamma}_i = 0$ and the algorithm stays stuck.</p>

<p><strong>Three ways to handle leverage points in this framework:</strong></p>

<p><strong>1. Joint penalization on $\beta$.</strong> The $\ell_1$ penalty on $\beta$ constrains $\hat{\beta}$ from being pulled far by a leverage point. The optimizer faces a tradeoff: chasing the leverage point costs $\lambda_1$ (because $\beta$ must move substantially) while flagging it as an outlier costs only $\lambda_2$ (one nonzero $\gamma_i$). For a single leverage point, absorbing it into $\gamma$ is cheaper. But this only works in the penalized-$\beta$ setting, not the classical $n \gg p$ case.</p>

<p><strong>2. Robust initialization.</strong> Instead of starting from $\gamma = 0$, initialize $\hat{\beta}$ with an S-estimate or LTS estimate. Then the initial residuals correctly identify leverage points, and thresholding flags them. This is MM-estimation recast in the She-Owen framework.</p>

<p><strong>3. Nonconvex penalties on $\gamma$.</strong> MCP/SCAD fully remove flagged outliers (unlike $\ell_1$ which only shrinks). Once a leverage point is detected, its influence drops to exactly zero. But detection still depends on having a good enough $\hat{\beta}$ to produce a large residual in the first place.</p>

<p>In practice, combining all three (penalized $\beta$ + robust initialization + nonconvex $\gamma$ penalty) gives the best results.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Related high-dimensional methods.** **Sparse LTS** (Alfons et al., 2013) combines LTS with LASSO, alternating between trimming outliers and selecting variables. **$\ell_1$-penalized Huber loss** (Fan et al., 2017) combines Huber loss with LASSO, solved via coordinate descent with oracle properties under weaker tail conditions than standard LASSO.

**Pros:** unifying framework (penalty choice = robustness model), simultaneous variable selection and outlier detection when both $\beta$ and $\gamma$ are penalized, flexible (swap in any penalty for either role). **Cons:** still vulnerable to leverage points without robust initialization or joint penalization on $\beta$, alternating minimization can converge to local optima with nonconvex penalties, two tuning parameters ($\lambda_1$, $\lambda_2$) when both $\beta$ and $\gamma$ are penalized.

## Stability and Algorithmic Robustness

Classical robust methods (S-estimation, LTS, MM-estimation) rely on random subsampling, which requires exponentially many draws as $p$ grows. A fundamentally different approach, based on the concept of **stability**, enables polynomial-time algorithms with optimal error guarantees in any dimension.

### Stability: Definition and Criteria

A set of $n$ points $S = \{z_1, \ldots, z_n\}$ is $(\epsilon, \delta)$**-stable** if removing any subset of at most $\epsilon n$ points changes the empirical statistics by at most $\delta$. Formally, for mean estimation:

$$\forall T \subseteq S \text{ with } \lvert T \rvert \leq \epsilon n: \quad \left\|\mu_S - \mu_{S \setminus T}\right\| \leq \delta$$

and for covariance estimation:

$$\forall T \subseteq S \text{ with } \lvert T \rvert \leq \epsilon n: \quad \left\|\Sigma_S - \Sigma_{S \setminus T}\right\|_{\text{op}} \leq \delta$$

where $\|\cdot\|_{\text{op}}$ is the spectral norm (largest eigenvalue).

**Why clean data is stable.** If $z_1, \ldots, z_n$ are i.i.d. draws from a sub-Gaussian distribution, the empirical mean and covariance concentrate around their population values. Removing a small fraction of points cannot shift them much because each point contributes $O(1/n)$. Concretely, $n$ samples from $\mathcal{N}(0, I_p)$ form an $(\epsilon, O(\epsilon\sqrt{\log(1/\epsilon)} + \sqrt{p/n}))$-stable set with high probability.

**Why contamination breaks stability.** An adversary adding $\epsilon n$ arbitrary points can inflate the empirical covariance in a specific direction. If the clean data has covariance $\approx I$, the contaminated data has top eigenvalue $\geq 1 + \Omega(\epsilon)$. This spectral gap is the detection signal.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Stability criteria and connection to classical robustness</span></summary>

<p><strong>Deterministic stability condition.</strong> Given a contaminated dataset $S' = S_{\text{good}} \cup S_{\text{bad}}$ where $\lvert S_{\text{bad}} \rvert \leq \epsilon n$, the stability condition guarantees that $S_{\text{good}}$ satisfies, for all subsets $T \subseteq S_{\text{good}}$ with $\lvert T \rvert \leq \epsilon n$:</p>

<ol>
<li><strong>Mean stability:</strong> $\|\mu_{S_{\text{good}}} - \mu_{S_{\text{good}} \setminus T}\| \leq \delta_1$</li>
<li><strong>Covariance stability:</strong> $\|\Sigma_{S_{\text{good}}} - \Sigma_{S_{\text{good}} \setminus T}\|_{\text{op}} \leq \delta_2$</li>
</ol>

<p>These conditions hold with high probability when $S_{\text{good}}$ is drawn from distributions with bounded moments. Sub-Gaussian distributions give $\delta_1 = O(\epsilon\sqrt{\log 1/\epsilon})$ and $\delta_2 = O(\epsilon \log 1/\epsilon)$. Distributions with only bounded $k$-th moments give weaker but still useful bounds: $\delta_1 = O(\epsilon^{1-1/k})$.</p>

<p><strong>Connection to classical concepts.</strong></p>

<ul>
<li><strong>Breakdown point.</strong> Stability with parameter $\epsilon$ means the estimator tolerates $\epsilon$-fraction contamination. This is exactly the breakdown point, but stability provides a constructive algorithm rather than just an existence guarantee.</li>
<li><strong>Influence function.</strong> The influence function $\text{IF}(z) = \lim_{\epsilon \to 0} (\hat{\theta}_{(1-\epsilon)P + \epsilon \delta_z} - \hat{\theta}_P)/\epsilon$ is the infinitesimal version of stability: it measures how much a single point can shift the estimate. Bounded influence $\Leftrightarrow$ stable at the population level.</li>
<li><strong>Generalized resilience</strong> (Zhu et al., 2022). A population-level generalization of stability: a distribution $P$ is $(\epsilon, \delta)$-resilient if for all distributions $Q$ with $d(P, Q) \leq \epsilon$ (under TV, Wasserstein, or other distances), the statistic of interest changes by at most $\delta$. This unifies TV and Wasserstein corruption models and shows when robust estimation is information-theoretically possible, even beyond the standard outlier contamination setting.</li>
</ul>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Spectral Filtering Algorithm

Diakonikolas et al. (2019) turned the stability condition into a concrete algorithm. The setup: $z_1, \ldots, z_n$ are i.i.d. draws from a distribution with mean $\mu$ and covariance $\Sigma_0$, but an adversary has corrupted an $\epsilon$-fraction of the samples (replacing them with arbitrary values). The goal is to estimate $\mu$ despite the contamination. If contamination breaks stability, it leaves a detectable spectral signature in the empirical covariance.

**Algorithm: FILTER for robust mean estimation**

<pre style="background:#faf6f0; padding:0.8em; border:1px solid #ddd; font-size:0.9em;">
Input: contaminated samples S = {z₁, ..., zₙ}, corruption fraction ε,
       known (or estimated) clean covariance Σ₀
Output: robust mean estimate μ̂

1. Whiten: z̃ᵢ <- Σ₀^{-1/2} zᵢ for each i = 1, ..., n
   (After whitening, the clean data has mean Σ₀^{-1/2} μ
    and covariance Σ₀^{-1/2} Σ₀ Σ₀^{-1/2} = I.
    This standardizes the scale across all directions so that
    step 2b can compare the empirical covariance against I.)
2. Repeat:
   a. Compute empirical mean μ̂ and covariance Σ̂ of current S
   b. Compute top eigenvalue λ₁ and eigenvector v₁ of (Σ̂ - I)
   c. If λ₁ ≤ O(ε log(1/ε)):
        STOP, return Σ₀^{1/2} μ̂  (undo whitening, dataset is stable)
   d. Else:
        - Project all points: pᵢ = v₁ᵀ z̃ᵢ
        - Compute scores: τᵢ = (pᵢ - median(p))²
        - Remove the point with the largest τᵢ
3. Return Σ₀^{1/2} μ̂
</pre>

In practice, $\Sigma_0$ is often unknown. Two common approaches: (1) use a robust initial estimate of $\Sigma_0$ (e.g., the MCD estimator), or (2) assume $\Sigma_0 = I$ when the data has been pre-standardized. For the sub-Gaussian case, the algorithm also works with an iteratively updated covariance estimate, re-whitening at each round.

**Why it works:**

- **Step 2c** is the stability check: if the top eigenvalue is close to 1, the covariance is close to the target, meaning no detectable contamination remains.
- **Step 2d** exploits the spectral signature: the top eigenvector $v_1$ of $(\hat{\Sigma} - I)$ points toward the direction of maximum covariance inflation. Contaminated points tend to have extreme projections along $v_1$, so they receive the highest scores $\tau_i$.
- Each iteration removes at least one contaminated point (in expectation) while removing at most one clean point, so after $O(\epsilon n)$ iterations the dataset is clean.

**Complexity:** $\tilde{O}(np^2/\epsilon)$ per iteration (dominated by the eigendecomposition), with at most $O(\epsilon n)$ iterations, giving $\tilde{O}(n^2 p^2 \epsilon)$ total. For fixed $\epsilon$, this is polynomial in $n$ and $p$.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">From robust mean to robust regression via covariance</span></summary>

<p><strong>The reduction.</strong> Linear regression $y = x^T\beta^* + \varepsilon$ with $\varepsilon$ independent of $x$ implies:</p>

$$\beta^* = \Sigma_{xx}^{-1} \Sigma_{xy}$$

<p>where $\Sigma_{xx} = \text{Cov}(x)$ and $\Sigma_{xy} = \text{Cov}(x, y)$. If we can robustly estimate the joint covariance of $(x, y)$, we can extract $\hat{\beta}$ from the robust covariance blocks.</p>

<p><strong>Algorithm for robust regression:</strong></p>

<ol>
<li>Form augmented vectors $z_i = (x_i, y_i) \in \mathbb{R}^{p+1}$.</li>
<li>Run FILTER on $\{z_1, \ldots, z_n\}$ to get robust mean $\hat{\mu}_z$ and robust covariance $\hat{\Sigma}_z$.</li>
<li>Extract blocks: $\hat{\Sigma}_{xx} = \hat{\Sigma}_z[1{:}p, \; 1{:}p]$ and $\hat{\Sigma}_{xy} = \hat{\Sigma}_z[1{:}p, \; p{+}1]$.</li>
<li>Return $\hat{\beta} = \hat{\Sigma}_{xx}^{-1} \hat{\Sigma}_{xy}$.</li>
</ol>

<p>This handles leverage points automatically: contaminated $x$-values inflate the joint covariance, and FILTER removes them before $\hat{\beta}$ is computed.</p>

<p><strong>Error guarantee.</strong> Under $\epsilon$-fraction adversarial contamination with sub-Gaussian clean data:</p>

$$\|\hat{\beta} - \beta^*\|_2 \leq O\left(\epsilon \sqrt{\log(1/\epsilon)}\right)$$

<p>which is near-optimal (the information-theoretic lower bound is $\Omega(\epsilon)$).</p>

<p><strong>Practical considerations.</strong> The algorithm is theoretically optimal but constants can be large. For moderate $n$ and $p$, MM-estimation and adaptive Huber remain more practical. The filtering approach becomes advantageous when $p$ is large (hundreds or thousands of dimensions) and the contamination is adversarial.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Pros:** polynomial-time in any dimension (unlike subsampling-based methods), information-theoretically near-optimal error rates, handles adversarial contamination with provable guarantees. **Cons:** requires known or estimable clean covariance $\Sigma_0$, assumes sub-Gaussian clean data, constants can be large (less practical than classical methods for moderate $n, p$), indirect approach to regression (extracts $\hat{\beta}$ from robust covariance rather than fitting directly).

### Covariate Filtering For Regression

Pensia et al. (2020) proposed a more practical variant: apply spectral filtering only to the covariates $x$, then run a classical robust regression estimator on the filtered data.

**Algorithm: COVARIATE-FILTER + robust regression**

<pre style="background:#faf6f0; padding:0.8em; border:1px solid #ddd; font-size:0.9em;">
Input: data {(x₁,y₁), ..., (xₙ,yₙ)}, corruption fraction ε
Output: robust regression estimate β̂

Stage 1: Covariate filtering
   1. Compute empirical covariance Σ̂ₓ of {x₁, ..., xₙ}
   2. Compute top eigenvalue λ₁ and eigenvector v₁ of Σ̂ₓ
   3. While λ₁ > threshold(ε, p, n):
      a. Project: pᵢ = v₁ᵀ xᵢ for all remaining points
      b. Compute the mean p̄ = (1/n)Σ pᵢ and variance s² = (1/n)Σ(pᵢ - p̄)²
         of the projections
      c. Compute scores: wᵢ = (pᵢ - p̄)² / s²
         (squared z-score: how far each point is from the projection
          center, measured in standard deviations along v₁)
      d. Remove the point with the largest wᵢ
      e. Recompute Σ̂ₓ, λ₁, v₁ on remaining data
   4. Let S* = remaining data after filtering

Stage 2: Robust regression on filtered data
   5. Run Huber regression (or LTS or LAD) on S*
   6. Return β̂
</pre>

**Determining the threshold.** After whitening (so clean data has covariance $I$), the empirical covariance's top eigenvalue concentrates around 1, with random fluctuations of order $O(\sqrt{p/n})$ from finite sampling. Contamination adds $O(\epsilon)$ on top. The threshold separates sampling noise from contamination signal:

$$\text{threshold} = 1 + C \cdot \max\!\left(\epsilon \log(1/\epsilon), \; \sqrt{p/n}\right)$$

for a constant $C$. If $\lambda_1$ exceeds this, the excess cannot be explained by sampling variability alone, so contamination must be present. If $\lambda_1$ is below this, any remaining contamination is too small to detect and too small to meaningfully bias the estimate. In practice, $\epsilon$ is often unknown and treated as a tuning parameter (analogous to the breakdown point in classical methods).

**Why this handles leverage points.** Stage 1 operates entirely in the $x$-space. A leverage point has an extreme $x$-value that inflates the covariance's top eigenvalue beyond the threshold. The filtering loop detects this spectral inflation and removes the responsible points. After filtering, the remaining data has well-behaved covariates, so standard robust estimators (which handle vertical outliers) suffice in Stage 2.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Error rates and comparison with joint filtering</span></summary>

<p><strong>Error guarantee.</strong> Under $\epsilon$-adversarial contamination with sub-Gaussian covariates and possibly heavy-tailed errors (only finite variance required):</p>

$$\|\hat{\beta} - \beta^*\|_2 \leq O\left(\epsilon \sqrt{\log(1/\epsilon)} + \sqrt{\frac{p}{n}}\right)$$

<p>The first term is the contamination cost (near-optimal), the second is the standard statistical rate. Huber regression in Stage 2 achieves this directly. LTS and LAD require a post-processing refinement step to match this rate.</p>

<p><strong>Heavy tails.</strong> If the covariates have only bounded $k$-th moments (not sub-Gaussian), the contamination cost degrades to $O(\epsilon^{1-1/k})$, but the algorithm still works. This is where covariate filtering improves over methods that assume sub-Gaussian covariates throughout.</p>

<p><strong>Comparison with joint filtering (Diakonikolas et al.):</strong></p>

<ul>
<li><strong>Joint filtering</strong>: works in $\mathbb{R}^{p+1}$, handles all contamination types simultaneously, extracts $\hat{\beta}$ from robust covariance. More general but $O(np^2)$ per iteration.</li>
<li><strong>Covariate filtering</strong>: works in $\mathbb{R}^p$, separates leverage detection (Stage 1) from response outlier handling (Stage 2). Allows using well-tuned classical estimators in Stage 2. Slightly cheaper and more modular.</li>
</ul>

<p>For regression specifically, covariate filtering is more practical because Stage 2 can leverage decades of robust regression software (Huber, MM-estimation) rather than requiring a custom covariance-based pipeline.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Pros:** modular (clean $x$-space first, then use any classical robust estimator), near-optimal rates, allows heavy-tailed errors (only finite variance needed), cheaper than joint filtering ($\mathbb{R}^p$ instead of $\mathbb{R}^{p+1}$). **Cons:** still requires sub-Gaussian covariates for Stage 1, requires known or estimable $\epsilon$, Stage 1 only removes leverage points (still need a robust estimator in Stage 2 for vertical outliers).

## Modern Estimators

### Adaptive Huber Regression

Sun et al. (2020) observed that classical Huber regression uses a fixed robustification parameter $c$, but the optimal $c$ should depend on $(n, p)$. Their adaptive Huber regression sets:

$$\tau \asymp \sqrt{\frac{n}{\log(np)}}$$

and solves:

$$\hat{\beta} = \arg\min_\beta \sum_{i=1}^n \ell_\tau(y_i - x_i^T\beta)$$

where $\ell_\tau$ is the Huber loss with parameter $\tau$. As $n$ grows, $\tau \to \infty$ and the estimator approaches OLS. For finite $n$, the adaptively chosen $\tau$ provides sub-Gaussian-type concentration bounds for $\hat{\beta}$ even when the errors have only finite second moments (no normality or symmetry required).

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why adaptive tuning matters and the theoretical guarantee</span></summary>

<p><strong>The problem with fixed $c$.</strong> Classical Huber regression with fixed $c = 1.345$ is calibrated for 95% efficiency under Gaussian errors at a specific sample size. But the right tradeoff between robustness and efficiency depends on how much data you have. With $n = 50$, you want aggressive robustness (small $c$) because a single outlier has high leverage. With $n = 10{,}000$, you can afford to be closer to OLS (large $c$) because each point matters less.</p>

<p><strong>The guarantee.</strong> With $\tau \asymp \sqrt{n / \log(np)}$, the adaptive Huber estimator satisfies:</p>

$$\|\hat{\beta} - \beta^*\|_2 \leq C\sqrt{\frac{p \log(np)}{n}}$$

<p>with high probability, requiring only $E[\varepsilon^2] < \infty$. This matches the sub-Gaussian rate $\sqrt{p/n}$ up to a $\sqrt{\log(np)}$ factor, without assuming sub-Gaussian tails. Under Gaussian errors, it recovers the OLS rate exactly.</p>

<p><strong>Extension to high dimensions.</strong> Adding an $\ell_1$ penalty gives penalized adaptive Huber regression:</p>

$$\hat{\beta} = \arg\min_\beta \sum_{i=1}^n \ell_\tau(y_i - x_i^T\beta) + \lambda \|\beta\|_1$$

<p>This handles simultaneous variable selection and robustness, with $\tau$ adapting to the effective dimension rather than $p$. Solved via the LAMM (Local Adaptive Majorize-Minimization) algorithm, which alternates between a quadratic majorization step and an $\ell_1$-penalized least squares step, converging at a linear rate.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Pros:** data-driven $\tau$ (no manual tuning of robustification parameter), achieves near sub-Gaussian rates with only finite second moments, extends naturally to high dimensions with $\ell_1$ penalty. **Cons:** only handles vertical outliers (heavy-tailed $y$), not leverage points, $\tau$ depends on unknown quantities ($n, p$, noise level) that must be estimated or bounded.

### Median-of-Means (MOM) Regression

Lugosi and Mendelson (2019) introduced a different approach. Instead of modifying the loss function, partition the data into $K$ blocks, compute the estimator on each block, and take the median (or a robust aggregate) across blocks.

1. Split observations randomly into $K$ blocks of size $n/K$.
2. For each block $k$, compute the empirical risk $R_k(\beta) = \frac{1}{\lvert B_k \rvert} \sum_{i \in B_k} (y_i - x_i^T\beta)^2$.
3. Find $\hat{\beta}$ that minimizes the median of $\{R_1(\beta), \ldots, R_K(\beta)\}$.

If fewer than $K/2$ blocks are contaminated, the median is determined by clean blocks, so outliers concentrated in a few blocks cannot affect the estimate.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Properties, computation, and minmax MOM</span></summary>

<p><strong>Breakdown.</strong> MOM regression tolerates up to $\lfloor K/2 \rfloor - 1$ corrupted blocks. Setting $K \approx \log(1/\delta)$ gives confidence $1-\delta$ guarantees. The tradeoff: more blocks means more robustness but less data per block (higher variance).</p>

<p><strong>Computational challenge.</strong> Minimizing the median of empirical risks is non-convex and non-smooth. The minmax MOM estimator (Lecue and Lerasle, 2020) replaces the median with a minmax formulation:</p>

$$\hat{\beta} = \arg\min_\beta \max_{k \in [K]} R_k(\beta) - \min_{k \in [K]} R_k(\beta)$$

<p>which can be solved via alternating gradient descent on $\beta$ and adversarial block selection.</p>

<p><strong>Advantages over M-estimation.</strong> MOM does not require choosing a robustification parameter ($c$ in Huber). It works under minimal moment assumptions (only finite variance). It handles both vertical outliers and leverage points if contamination is distributed across fewer than $K/2$ blocks. The price is a $\sqrt{\log K}$ factor in the convergence rate.</p>

<p><strong>Tournament procedure (Lugosi and Mendelson, 2019).</strong> A practical variant: for each pair of candidate estimators, use MOM to decide which is better. Run a tournament across candidates to find the winner. This avoids the non-convex median minimization and achieves near-optimal rates.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Pros:** no robustification parameter to tune ($K$ is the only choice), handles both outlier types if contamination is spread across fewer than $K/2$ blocks, minimal moment assumptions (only finite variance), near-optimal rates. **Cons:** computationally hard (minimizing median of risks is non-convex), less data per block means higher variance, performance depends on how contamination is distributed across blocks (concentrated contamination in a few blocks is the best case).

### Distributionally Robust Optimization (DRO)

Instead of assuming a specific contamination model, minimize worst-case expected loss over a Wasserstein ball around the empirical distribution:

$$\min_\beta \sup_{Q : W(Q, \hat{P}_n) \leq \epsilon} E_Q[\ell(y - x^T\beta)]$$

This provides finite-sample robustness guarantees without specifying what kind of outliers you face. Under squared loss with Wasserstein distance, DRO is equivalent to a specific form of regularized regression (Blanchet and Murthy, 2019), connecting robustness to regularization at a fundamental level.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">DRO-regularization equivalence</span></summary>

<p>For the Wasserstein ball of radius $\epsilon$ and squared loss, the DRO problem reduces to:</p>

$$\min_\beta E_{\hat{P}_n}[(y - x^T\beta)^2] + \epsilon \cdot R(\beta)$$

<p>where $R(\beta)$ is a regularizer determined by the Wasserstein metric's ground cost. For the $\ell_2$ ground cost, $R(\beta) = \|\beta\|_2$ (ridge-like). For the $\ell_1$ ground cost, $R(\beta) = \|\beta\|_\infty$. For the $\ell_\infty$ ground cost, $R(\beta) = \|\beta\|_1$ (LASSO).</p>

<p>This means: choosing a regularizer is implicitly choosing a robustness model. LASSO is distributionally robust against perturbations measured in the $\ell_\infty$ Wasserstein metric. Ridge is robust against $\ell_2$ perturbations. The DRO radius $\epsilon$ plays the role of the regularization parameter $\lambda$.</p>

<p>This connection gives a robustness interpretation to standard regularized regression, and conversely gives a computational recipe for DRO: just solve a regularized regression problem.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Pros:** finite-sample robustness guarantees without specifying contamination type, equivalent to regularized regression (computationally free if you already use regularization), principled connection between robustness and regularization. **Cons:** the Wasserstein ball is a worst-case model (can be overly conservative), choice of ground metric determines the regularizer (limited flexibility), does not directly target outlier detection or removal.

## Robust Covariance, Correlation, and Inference

### Robust Covariance and Correlation

Before running any regression, you may want robust correlation estimates themselves.

**Spearman's rank correlation.** Replaces values with ranks, then computes Pearson on ranks. Each observation's influence is bounded by its rank position. Good for a single pairwise correlation.

**Minimum Covariance Determinant (MCD).** Finds the subset of $h$ points whose covariance matrix has minimum determinant, giving a robust estimate of the full covariance matrix. Also provides Mahalanobis distances for flagging multivariate outliers. MCD detects outliers that are extreme only in combination, which pairwise methods like Spearman miss.

**Robust PCA.** Decomposes the data matrix as $X = L + S$ where $L$ is low-rank (clean signal) and $S$ is sparse (cell-level corruption). Unlike MCD which flags entire observations, robust PCA identifies corrupted individual cells.

**Projection-based (Stahel-Donoho).** Detects leverage points directly in the $x$-space by projecting onto random directions and measuring outlyingness. Does not use residuals, so the masking problem does not arise.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">MCD vs Robust PCA vs Spearman: when to use which</span></summary>

<p><strong>Spearman</strong>: computing a single pairwise correlation. Drop-in replacement for Pearson with no tuning.</p>

<p><strong>MCD</strong>: need a full robust covariance matrix, or need to flag multivariate outliers. Assumes the clean data is approximately elliptical (multivariate normal). Use before regression, PCA, or any method that takes a covariance matrix as input.</p>

<p><strong>Robust PCA</strong>: dimension reduction on contaminated data, or when corruption affects specific cells rather than entire observations. Assumes the clean data lives near a low-dimensional subspace (low-rank assumption on the data matrix, not the covariance matrix). Solves $\min \|L\|_* + \lambda\|S\|_1$ subject to $X = L + S$.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Robust Variance Estimation

Even with robust point estimates, inference (confidence intervals, hypothesis tests) requires reliable standard errors.

**Sandwich (HC) estimators.** Keep OLS point estimates but correct the covariance:

$$\text{Var}(\hat{\beta}) = (X^TX)^{-1} X^T \text{diag}(e_i^2) X (X^TX)^{-1}$$

Variants HC0 through HC3 (White, MacKinnon-White) differ in how they estimate $e_i^2$. These are consistent under heteroskedasticity but do not change $\hat{\beta}$ itself, so they only address variance misspecification, not outlier bias.

**Bootstrap.** Resample observation pairs $(x_i, y_i)$ or residuals, refit each time, and use the empirical distribution of $\hat{\beta}$ for standard errors. Non-parametric, works under general conditions. For robust regression, use the wild bootstrap to preserve heteroskedasticity structure.

**Scale from M/MM-estimation.** The robust regression itself produces a robust residual scale $\hat{\sigma}$. The covariance of $\hat{\beta}$ is:

$$\text{Var}(\hat{\beta}_M) = \hat{\sigma}^2 \frac{E[\psi^2]}{(E[\psi'])^2} (X^TWX)^{-1}$$

where $W$ is the IRLS weight matrix and the ratio $E[\psi^2]/(E[\psi'])^2$ is a correction factor that accounts for the non-quadratic loss.

### Conformal Prediction Under Contamination

Conformal prediction (Vovk et al., 2005) provides distribution-free prediction intervals that are valid under exchangeability. Recent extensions (Barber et al., 2023) allow conformal inference under covariate shift and contamination:

Given a new test point $x_{n+1}$, construct a prediction set $C(x_{n+1})$ such that $P(y_{n+1} \in C(x_{n+1})) \geq 1 - \alpha$ regardless of the true distribution. The prediction intervals may be wider under contamination, but they are never overconfident.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Split conformal for robust regression</span></summary>

<p><strong>Algorithm (split conformal):</strong></p>

<ol>
<li>Split data into training set $D_1$ and calibration set $D_2$.</li>
<li>Fit any robust regression on $D_1$ to get $\hat{f}$.</li>
<li>Compute residuals on $D_2$: $R_i = \lvert y_i - \hat{f}(x_i) \rvert$ for $i \in D_2$.</li>
<li>Let $q$ be the $\lceil (1-\alpha)(1 + \lvert D_2 \rvert) \rceil / \lvert D_2 \rvert$ quantile of the calibration residuals.</li>
<li>Prediction interval: $C(x_{n+1}) = [\hat{f}(x_{n+1}) - q, \; \hat{f}(x_{n+1}) + q]$.</li>
</ol>

<p><strong>Robustness benefit.</strong> If you use MM-estimation in step 2, the point predictions $\hat{f}$ are robust to training outliers. The calibration residuals in step 3 are computed on held-out data, so they reflect the model's true predictive accuracy. Even if some calibration points are outliers, conformal guarantees marginal coverage as long as the data is exchangeable.</p>

<p><strong>Weighted conformal.</strong> Under covariate shift or heterogeneous contamination, reweight the calibration residuals by likelihood ratios $w_i = p_{\text{test}}(x_i) / p_{\text{train}}(x_i)$. This corrects for distribution mismatch while maintaining coverage.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Practical Guide

### When To Use What

| Problem | Method | Why |
|---|---|---|
| Vertical outliers, low-dimensional | Huber M-estimation | Simple, fast, 95% efficient |
| Vertical + leverage outliers, low-dimensional | MM-estimation | 50% breakdown + 95% efficiency |
| Heavy-tailed errors, adaptive robustness | Adaptive Huber (Sun et al., 2020) | Data-driven $\tau$, near-optimal rates under minimal moment assumptions |
| Adversarial contamination, unknown type | MOM regression | No robustification parameter to tune, handles both outlier types |
| High-dimensional variable selection, clean data | LASSO | Convex, fast, well-understood |
| High-dimensional, need unbiased effect sizes | MCP or SCAD | Oracle property, no shrinkage for large signals |
| High-dimensional + outliers | She-Owen (LASSO on $\beta$, MCP on $\gamma$) | Simultaneous variable selection and outlier detection |
| High-dimensional + heavy tails | Penalized adaptive Huber | Robust loss + sparsity penalty, LAMM algorithm |
| High-dimensional, adversarial contamination in $x$ | Covariate filtering + Huber (Pensia et al.) | Filters leverage points spectrally, then uses classical robust regression |
| Very high-dimensional, adversarial contamination | Spectral filtering on joint $(x,y)$ (Diakonikolas et al.) | Polynomial-time, information-theoretically optimal rates |
| Distribution-free prediction intervals | Conformal + any robust base estimator | Valid coverage without distributional assumptions |
| Robust covariance / outlier detection | MCD | Full robust covariance + Mahalanobis distances |
| Dimension reduction on corrupted data | Robust PCA | Cell-level corruption, low-rank structure |
| Honest standard errors without changing estimator | Sandwich (HC) or bootstrap | Fixes inference, not point estimates |

### Key Takeaways

**The two problems are different.** Vertical outliers (extreme $y$) are easy: any method that downweights large residuals works (Huber, bisquare, LASSO on $\gamma$). Leverage points (extreme $x$) are hard: residual-based methods fail because the outlier masks itself. Always determine which type you face before choosing a method.

**The initialization trick.** The core difficulty with leverage points is circular: you need a good $\hat{\beta}$ to identify outliers (via large residuals), but you need to identify outliers to get a good $\hat{\beta}$. M-estimation with IRLS is efficient and converges fast, but if initialized from OLS, the starting $\hat{\beta}$ is already pulled by leverage points, their residuals appear small, and IRLS never corrects. The solution is always the same: break the circle with a separate, highly robust first stage.

- **MM-estimation:** Stage 1 (S-estimation) searches over random clean subsets to find a $\hat{\beta}_S$ near the truth, paying the cost of low efficiency. Stage 2 (M-estimation) refines from this clean start, recovering efficiency.
- **She-Owen:** instead of initializing $\gamma = 0$ and $\hat{\beta} = \hat{\beta}_{\text{OLS}}$, initialize $\hat{\beta}$ with an S-estimate or LTS estimate. The initial residuals then correctly reveal leverage points, and thresholding on $\gamma$ flags them.
- **Covariate filtering:** clean the $x$-space first (Stage 1), so that the downstream regression estimator (Stage 2) never sees the leverage points.

The pattern across all three: separate "finding the clean subset" (expensive, robust, low efficiency) from "efficient estimation on the clean subset" (fast, high efficiency). The first stage does the hard work of breaking the circular dependency; the second stage does the statistical work of precise estimation.

**Penalties are dual-purpose tools.** The same penalty applied to $\beta$ does variable selection; applied to $\gamma$ (residuals) does outlier detection. LASSO $\leftrightarrow$ Huber, MCP $\leftrightarrow$ bisquare. This is the She-Owen insight. Choosing a penalty is choosing both a sparsity model and an outlier model.

**Nonconvex penalties are worth the trouble.** SCAD and MCP achieve the oracle property (correct selection + unbiased estimation) while LASSO does not. The cost: local optima instead of a global optimum. In practice, coordinate descent with warm starts works well, and the bias reduction matters more than the theoretical global-optimality guarantee of LASSO.

**Stability generalizes breakdown.** Classical breakdown point tells you how many outliers an estimator can tolerate. Stability tells you the same thing, but constructively: if the data is unstable (inflated eigenvalue), here is the direction to look and the points to remove. This converts an existence result into an algorithm.

**Adapt your robustness to your data.** Fixed robustification parameters ($c = 1.345$ in Huber) are calibrated for a specific setting. Adaptive Huber scales $\tau$ with $(n, p)$. MOM avoids the parameter entirely. DRO lets the data determine the worst-case perturbation. The trend is toward methods that require less manual tuning.

**Tradeoff: breakdown point vs efficiency.** S-estimation and LTS achieve 50% breakdown but only 8-28% efficiency. OLS has 100% efficiency but $1/n$ breakdown. MM-estimation resolves this by separating the two goals into two stages, achieving 50% breakdown and 95% efficiency, at the cost of computational complexity. There is no free lunch: you cannot get both robustness and efficiency from a single-stage estimator.

**Tradeoff: convexity vs bias.** LASSO is convex (guaranteed global optimum, fast solvers) but carries permanent shrinkage bias on large coefficients. SCAD and MCP are nonconvex (local optima, need good initialization) but eliminate bias for large coefficients and achieve the oracle property. The practical question is whether the bias reduction from nonconvexity is worth the optimization difficulty, and for most problems it is.

**Tradeoff: computational scalability vs contamination model.** Classical methods (S-estimation, LTS, MM-estimation) handle arbitrary contamination but require random subsampling that scales exponentially with $p$. Stability-based methods (spectral filtering, covariate filtering) run in polynomial time in any dimension but assume sub-Gaussian clean data. For low-dimensional problems, classical methods are more practical. For high-dimensional problems with adversarial contamination, stability-based methods are the only option.

**Tradeoff: robustness to $y$-outliers vs robustness to $x$-outliers.** M-estimation (Huber, bisquare) handles vertical outliers cheaply and efficiently, but completely fails for leverage points. Adding leverage-point robustness (MM-estimation, covariate filtering) costs substantially more computation. If you know your contamination is only in $y$, M-estimation is sufficient and much cheaper. If contamination could be in $x$, you must pay the additional cost.

## References

- Huber, P. J. (1964). Robust estimation of a location parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Rousseeuw, P. J. (1984). Least median of squares regression. *Journal of the American Statistical Association*, 79(388), 871-880.
- Rousseeuw, P. J. and Yohai, V. J. (1984). Robust regression by means of S-estimators. *Lecture Notes in Statistics*, 26, 256-272.
- Yohai, V. J. (1987). High breakdown-point and high efficiency robust estimates for regression. *Annals of Statistics*, 15(2), 642-656.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
- Rousseeuw, P. J. and Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator. *Technometrics*, 41(3), 212-223.
- Fan, J. and Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360.
- Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Salibian-Barrera, M. and Yohai, V. J. (2006). A fast algorithm for S-regression estimates. *Journal of Computational and Graphical Statistics*, 15(2), 414-427.
- Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty. *Annals of Statistics*, 38(2), 894-942.
- She, Y. and Owen, A. B. (2011). Outlier detection using nonconvex penalized regression. *Journal of the American Statistical Association*, 106(494), 626-639.
- Alfons, A., Croux, C., and Gelper, S. (2013). Sparse least trimmed squares regression for analyzing high-dimensional large data sets. *Annals of Applied Statistics*, 7(1), 226-248.
- Fan, J., Li, Q., and Wang, Y. (2017). Estimation of high dimensional mean regression in the absence of symmetry and light tail assumptions. *Journal of the Royal Statistical Society: Series B*, 79(1), 247-265.
- Blanchet, J. and Murthy, K. (2019). Quantifying distributional model risk via optimal transport. *Mathematics of Operations Research*, 44(2), 565-600.
- Diakonikolas, I., Kamath, G., Kane, D., Li, J., Moitra, A., and Stewart, A. (2019). Robust estimators in high dimensions without the computational intractability. *SIAM Journal on Computing*, 48(2), 742-864.
- Lugosi, G. and Mendelson, S. (2019). Sub-Gaussian estimators of the mean of a random vector. *Annals of Statistics*, 47(2), 783-794.
- Lecue, G. and Lerasle, M. (2020). Robust machine learning by median-of-means: theory and practice. *Annals of Statistics*, 48(2), 906-931.
- Pensia, A., Jog, V., and Loh, P.-L. (2020). Robust regression with covariate filtering: Heavy tails and adversarial contamination. *Journal of the American Statistical Association*, 120(550), 2024.
- Sun, Q., Zhou, W.-X., and Fan, J. (2020). Adaptive Huber regression. *Journal of the American Statistical Association*, 115(529), 254-265.
- Zhu, B., Jiao, J., and Steinhardt, J. (2022). Generalized resilience and robust statistics. *Annals of Statistics*, 50(4), 2256-2283.
- Barber, R. F., Candes, E. J., Ramdas, A., and Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *Annals of Statistics*, 51(2), 816-845.
