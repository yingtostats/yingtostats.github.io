---
layout: post
title:  "Linear Regression Review: Geometry, Inference, and Selection"
date:   2020-07-03 21:00:00
tag:
- Statistics
- Regression
projects: true
blog: true
author: YingZhang
description: Linear regression review covering geometry, ANOVA, inference, collinearity, regularization, and feature selection.
fontsize: 23pt

---

{% include mathjax_support.html %}

Linear regression is the foundation of statistical modeling. Despite its apparent simplicity, it contains the core logic of modern inference: orthogonal projection, variance decomposition, hypothesis testing, identifiability, and regularization. This post works through the geometry of OLS and its sampling distributions, hypothesis testing from single coefficients to MANOVA, a concrete treatment of multicollinearity with numerical examples, ridge and lasso regularization, multiple testing corrections, and practical feature screening and selection.

## Model Setup

Consider the standard linear model

$$
y = X\beta + \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0,\sigma^2 I_n),
$$

where:

* $y \in \mathbb{R}^{n}$ is the response vector.
* $X \in \mathbb{R}^{n\times p}$ is the design matrix, with $p$ columns including the intercept column.
* $\beta \in \mathbb{R}^{p}$ is the coefficient vector.
* $\varepsilon$ is i.i.d. noise, $\varepsilon_{i} \sim \mathcal{N}(0,\sigma^2)$.

If $X$ has full column rank, the ordinary least squares (OLS) estimator is

$$
\hat\beta = (X^T X)^{-1}X^T y.
$$

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Derivation via likelihood and least squares</span></summary>

<p><strong>Likelihood.</strong></p>

<p>Under <span>$\varepsilon \sim \mathcal{N}(0, \sigma^2 I_n)$</span>, each observation satisfies <span>$y_i \mid x_i \sim \mathcal{N}(x_i^T\beta, \sigma^2)$</span>. The joint log-likelihood is</p>

$$
\ell(\beta, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\lVert y - X\beta \rVert_2^2.
$$

<p>For fixed <span>$\sigma^2$</span>, maximizing <span>$\ell$</span> over <span>$\beta$</span> is equivalent to minimizing the residual sum of squares</p>

$$
\mathrm{RSS}(\beta) = \lVert y - X\beta \rVert_2^2 = (y - X\beta)^T(y - X\beta).
$$

<p>So the MLE of <span>$\beta$</span> coincides with the least squares estimator.</p>

<hr>

<p><strong>Gradient and normal equations.</strong></p>

<p>Expand the RSS:</p>

$$
\mathrm{RSS}(\beta) = y^Ty - 2\beta^T X^T y + \beta^T X^T X \beta.
$$

<p>Taking the gradient with respect to <span>$\beta$</span> and setting it to zero:</p>

$$
\nabla_\beta\,\mathrm{RSS}(\beta) = -2X^Ty + 2X^TX\beta = 0.
$$

<p>This gives the normal equations:</p>

$$
X^TX\hat\beta = X^Ty.
$$

<hr>

<p><strong>Solving for <span>$\hat\beta$</span>.</strong></p>

<p>If <span>$X$</span> has full column rank, <span>$X^TX$</span> is positive definite and therefore invertible. Left-multiplying both sides by <span>$(X^TX)^{-1}$</span>:</p>

$$
\hat\beta = (X^TX)^{-1}X^Ty.
$$

<p>The second-order condition confirms this is a minimum: the Hessian <span>$\nabla^2_\beta\,\mathrm{RSS} = 2X^TX$</span> is positive definite, so RSS is strictly convex and the stationary point is the unique global minimum.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

Under the weaker assumptions $E[\varepsilon \mid X] = 0$ and $\mathrm{Var}(\varepsilon \mid X) = \sigma^2 I_n$ (no normality required), the Gauss-Markov theorem guarantees that $\hat\beta$ is BLUE (best linear unbiased estimator): among all linear unbiased estimators of $\beta$, it has the smallest variance in the sense that the difference $\mathrm{Var}(\tilde\beta) - \mathrm{Var}(\hat\beta)$ is positive semidefinite for any other linear unbiased estimator $\tilde\beta$.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Proof of the Gauss-Markov theorem</span></summary>

<p><strong>Setup.</strong></p>

<p>Any linear estimator takes the form <span>$\tilde\beta = Cy$</span> for some matrix <span>$C \in \mathbb{R}^{p \times n}$</span>. Unbiasedness requires</p>

$$
E[\tilde\beta] = CX\beta = \beta \quad \text{for all } \beta,
$$

<p>which forces <span>$CX = I_p$</span>. The OLS estimator corresponds to <span>$C_0 = (X^TX)^{-1}X^T$</span>, and it satisfies <span>$C_0 X = I_p$</span>.</p>

<hr>

<p><strong>Decompose the general estimator.</strong></p>

<p>Write <span>$C = C_0 + D$</span> where <span>$D = C - C_0$</span>. The unbiasedness constraint <span>$CX = I_p$</span> gives</p>

$$
(C_0 + D)X = I_p \implies DX = 0.
$$

<hr>

<p><strong>Variance of <span>$\tilde\beta$</span>.</strong></p>

<p>Under <span>$\mathrm{Var}(\varepsilon) = \sigma^2 I_n$</span>,</p>

$$
\mathrm{Var}(\tilde\beta) = \sigma^2 CC^T = \sigma^2(C_0 + D)(C_0 + D)^T.
$$

<p>Expanding:</p>

$$
CC^T = C_0 C_0^T + C_0 D^T + D C_0^T + DD^T.
$$

<p>Since <span>$DX = 0$</span>, we have <span>$D X (X^TX)^{-1} = 0$</span>, so <span>$D C_0^T = D X(X^TX)^{-1} = 0$</span> and similarly <span>$C_0 D^T = 0$</span>. Therefore</p>

$$
\mathrm{Var}(\tilde\beta) = \sigma^2 C_0 C_0^T + \sigma^2 DD^T = \mathrm{Var}(\hat\beta) + \sigma^2 DD^T.
$$

<hr>

<p><strong>Conclusion.</strong></p>

<p>Since <span>$DD^T$</span> is positive semidefinite for any matrix <span>$D$</span>,</p>

$$
\mathrm{Var}(\tilde\beta) - \mathrm{Var}(\hat\beta) = \sigma^2 DD^T \succeq 0.
$$

<p>Equality holds if and only if <span>$D = 0$</span>, i.e. <span>$C = C_0$</span>, meaning OLS is the unique BLUE.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

The hat matrix is

$$
H = X(X^T X)^{-1}X^T.
$$

Fitted values and residuals follow as

$$
\hat y = Hy,
\qquad
e = (I - H)y.
$$

Since $H$ and $I - H$ are complementary orthogonal projections, $\hat y \perp e$:

$$
\hat y^T e = (Hy)^T(I-H)y = y^T H(I-H)y = y^T(H - H^2)y = 0,
$$

where the last step uses idempotency $H^2 = H$.

$$
\boxed{\text{OLS = orthogonal projection of } y \text{ onto } \operatorname{col}(X).}
$$

Most classical results follow as geometric consequences.

## OLS Geometry

The regression fit $\hat y$ is the closest point to $y$ inside the feature space $\operatorname{col}(X)$. That is why residuals are orthogonal to fitted values and why sums of squares decompose cleanly.

The key orthogonality statement is

$$
X^T e = 0.
$$

Residuals are orthogonal to every column of $X$, hence orthogonal to every vector in $\operatorname{col}(X)$.

If the model includes an intercept, $\hat y - \bar y\mathbf{1} \in \operatorname{col}(X)$. Then

$$
y - \bar y\mathbf{1} = (\hat y - \bar y\mathbf{1}) + e,
$$

and the sum of squares decomposes as

$$
\mathrm{SST} = \mathrm{SSR} + \mathrm{SSE},
$$

where

$$
\mathrm{SST} = \sum_{i=1}^n (y_{i}-\bar y)^2,
\qquad
\mathrm{SSR} = \sum_{i=1}^n (\hat y_{i}-\bar y)^2,
\qquad
\mathrm{SSE} = \sum_{i=1}^n (y_{i}-\hat y_{i})^2.
$$

The coefficient of determination is

$$
R^2 = \frac{\mathrm{SSR}}{\mathrm{SST}} = 1 - \frac{\mathrm{SSE}}{\mathrm{SST}}.
$$

Regression ANOVA is the Pythagorean theorem in data space ($\mathrm{SSR} + \mathrm{SSE} = \mathrm{SST}$, because $\hat y - \bar y\mathbf{1} \perp e$).

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Normal equations, orthogonality, and the Pythagorean decomposition</span></summary>

<p><strong>Column space.</strong></p>

<p>A vector <span>$v \in \mathbb{R}^n$</span> is in <span>$\operatorname{col}(X)$</span> if and only if there exists <span>$c \in \mathbb{R}^p$</span> such that <span>$v = Xc$</span>. For example, <span>$\hat y = X\hat\beta \in \operatorname{col}(X)$</span> with <span>$c = \hat\beta$</span>, and when an intercept is included, <span>$\bar y\mathbf{1} = X(\bar y e_1) \in \operatorname{col}(X)$</span> with <span>$c = \bar y e_1$</span>, where <span>$e_1 = (1, 0, \ldots, 0)^T \in \mathbb{R}^p$</span> is the standard basis vector that picks out the intercept column of <span>$X$</span>.</p>

<hr>

<p><strong>Step 1: Normal equations.</strong></p>

<p>OLS minimizes <span>$\lVert y - X\beta \rVert_2^2$</span>. Differentiating with respect to <span>$\beta$</span> and setting to zero gives the normal equations:</p>

$$
X^T(y - X\hat\beta) = 0,
$$

<p>so <span>$X^T e = 0$</span>. The residual vector is orthogonal to every column of <span>$X$</span>.</p>

<hr>

<p><strong>Step 2: Projection interpretation.</strong></p>

<p>Because <span>$\hat y = X\hat\beta$</span> lies in <span>$\operatorname{col}(X)$</span> and <span>$e = y - \hat y$</span> is orthogonal to that space, <span>$\hat y$</span> is the orthogonal projection of <span>$y$</span> onto the feature space. This is the geometric meaning of OLS.</p>

<hr>

<p><strong>Step 3: Decomposition with an intercept.</strong></p>

<p>Write</p>

$$
y - \bar y\mathbf{1} = (\hat y - \bar y\mathbf{1}) + (y - \hat y).
$$

<p>Both <span>$\hat y$</span> and <span>$\bar y\mathbf{1}$</span> are in <span>$\operatorname{col}(X)$</span> (shown above), so <span>$\hat y - \bar y\mathbf{1} \perp e$</span>. Expand the squared norm of <span>$y - \bar y\mathbf{1}$</span> directly:</p>

$$
\lVert y - \bar y\mathbf{1} \rVert^2
= \lVert (\hat y - \bar y\mathbf{1}) + e \rVert^2
= \lVert \hat y - \bar y\mathbf{1} \rVert^2 + 2(\hat y - \bar y\mathbf{1})^T e + \lVert e \rVert^2.
$$

<p>The cross term vanishes because <span>$\hat y - \bar y\mathbf{1} \in \operatorname{col}(X)$</span> and <span>$e \perp \operatorname{col}(X)$</span>, leaving</p>

$$
\underbrace{(y - \bar y\mathbf{1})^T(y - \bar y\mathbf{1})}_{\mathrm{SST}}
=
\underbrace{(\hat y - \bar y\mathbf{1})^T(\hat y - \bar y\mathbf{1})}_{\mathrm{SSR}}
+
\underbrace{e^T e}_{\mathrm{SSE}}.
$$

<p>Expanding each inner product entry-wise: <span>$\mathrm{SST} = \sum_i(y_i - \bar y)^2$</span>, <span>$\mathrm{SSR} = \sum_i(\hat y_i - \bar y)^2$</span>, <span>$\mathrm{SSE} = \sum_i e_i^2$</span>.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

## Sampling Distributions

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Recap: Normal, $\chi^2$, $t$, and $F$ distributions and their connections</span></summary>

<p><strong>Standard normal.</strong> If <span>$Z \sim \mathcal{N}(0,1)$</span>, then <span>$Z$</span> is the building block for all four distributions.</p>

<hr>

<p><strong>Chi-squared.</strong> If <span>$Z_1, \ldots, Z_k \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$</span>, then</p>

$$
V = Z_1^2 + \cdots + Z_k^2 \sim \chi^2_k.
$$

<p>Mean <span>$k$</span>, variance <span>$2k$</span>. In regression, <span>$\mathrm{SSE}/\sigma^2 \sim \chi^2_{n-p}$</span> because the residual vector lives in an <span>$(n-p)$</span>-dimensional subspace.</p>

<hr>

<p><strong>$t$-distribution.</strong> If <span>$Z \sim \mathcal{N}(0,1)$</span> and <span>$V \sim \chi^2_k$</span> independently, then</p>

$$
T = \frac{Z}{\sqrt{V/k}} \sim t_k.
$$

<p>In regression, the coefficient $t$-statistic is</p>

$$
t_j = \frac{\hat\beta_j - \beta_j}{s\sqrt{[(X^TX)^{-1}]_{jj}}} = \frac{\mathcal{N}(0,1)}{\sqrt{\chi^2_{n-p}/(n-p)}} \sim t_{n-p}.
$$

<p>The numerator is normal (from <span>$\hat\beta_j$</span>), and the denominator involves <span>$s^2 \propto \chi^2_{n-p}$</span>, independent of <span>$\hat\beta_j$</span>.</p>

<hr>

<p><strong>$F$-distribution.</strong> If <span>$V_1 \sim \chi^2_{k_1}$</span> and <span>$V_2 \sim \chi^2_{k_2}$</span> independently, then</p>

$$
F = \frac{V_1/k_1}{V_2/k_2} \sim F_{k_1,\, k_2}.
$$

<p>In regression, the ANOVA <span>$F$</span>-statistic is a ratio of two independent chi-squared quadratic forms scaled by their degrees of freedom: <span>$\mathrm{SSR}/\sigma^2 \sim \chi^2_{p-1}$</span> and <span>$\mathrm{SSE}/\sigma^2 \sim \chi^2_{n-p}$</span>.</p>

<hr>

<p><strong>Beta distribution.</strong> If <span>$V_1 \sim \chi^2_{k_1}$</span> and <span>$V_2 \sim \chi^2_{k_2}$</span> independently, then</p>

$$
W = \frac{V_1}{V_1 + V_2} \sim \mathrm{Beta}\!\left(\frac{k_1}{2},\, \frac{k_2}{2}\right).
$$

<p>Mean <span>$k_1/(k_1+k_2)$</span>, supported on <span>$(0,1)$</span>. Under the global null, <span>$R^2 = \mathrm{SSR}/\mathrm{SST}$</span> has exactly this form with <span>$k_1 = p-1$</span> and <span>$k_2 = n-p$</span>, giving <span>$R^2 \sim \mathrm{Beta}((p-1)/2,\,(n-p)/2)$</span>. The <span>$F$</span>-statistic is a monotone function of <span>$R^2$</span>, so the Beta and <span>$F$</span> tests are equivalent.</p>

<hr>

<p><strong>Key connections.</strong></p>

<ul>
<li><span>$T^2 \sim F_{1,k}$</span>: squaring a <span>$t_k$</span> variable gives an <span>$F_{1,k}$</span> variable. This is why a single-coefficient <span>$t$</span> test and the corresponding <span>$F$</span> test with <span>$q=1$</span> are equivalent.</li>
<li><span>$\chi^2_k / k \to 1$</span> as <span>$k \to \infty$</span>, so <span>$t_k \to \mathcal{N}(0,1)$</span> for large degrees of freedom.</li>
<li><span>$k_2 \cdot F_{k_1, k_2} / k_1 \to \chi^2_{k_1}$</span> as <span>$k_2 \to \infty$</span>.</li>
</ul>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

Under the Gaussian linear model with full-rank $X$:

$$
\hat\beta \sim \mathcal{N}\!\left(\beta,\,\sigma^2(X^T X)^{-1}\right).
$$

So $\hat\beta$ is unbiased, and its variance depends on the geometry of $X$.

The fitted values $\hat y = Hy$ are a linear map of the Gaussian $y$, so

$$
\hat y \sim \mathcal{N}\!\left(X\beta,\, \sigma^2 H\right).
$$

The residuals $e = (I-H)y$ are also Gaussian:

$$
e \sim \mathcal{N}\!\left(0,\, \sigma^2(I - H)\right).
$$

Note that $e$ has a singular covariance: $I - H$ has rank $n - p$, reflecting the $p$ constraints $X^T e = 0$. In particular, the marginal variance of a single residual is $\mathrm{Var}(e_i) = \sigma^2(1-h_{ii})$: residuals at high-leverage points have smaller variance.

The residual variance estimator is

$$
s^2 = \frac{\mathrm{SSE}}{n - p},
$$

and

$$
\frac{(n-p)\,s^2}{\sigma^2} \sim \chi^2_{n-p}.
$$

Crucially, $\hat\beta$ and $s^2$ are independent under the Gaussian model.

**$R^2$ distribution.** Under the global null $H_0: \beta_1 = \cdots = \beta_{p-1} = 0$ (all slope coefficients are zero, intercept free),

$$
R^2 \sim \operatorname{Beta}\!\left(\frac{p-1}{2},\,\frac{n-p}{2}\right).
$$

Equivalently,

$$
F = \frac{R^2/(p-1)}{(1-R^2)/(n-p)} \sim F_{p-1,\,n-p}
\quad \text{under } H_0.
$$

The expected value of $R^2$ under the null is

$$
\mathbb{E}[R^2] = \frac{p-1}{n-1}.
$$

This is the key implication: even under a null model with no signal, $R^2$ grows mechanically as more predictors are added. Adjusted $R^2$ corrects for this by penalizing model complexity:

$$
R^2_{\mathrm{adj}} = 1 - (1 - R^2)\frac{n-1}{n-p}.
$$

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Derivations of the sampling distributions</span></summary>

<p><strong>Distribution of <span>$\hat\beta$</span>.</strong></p>

<p>Since <span>$\hat\beta = (X^T X)^{-1}X^T y$</span> is a linear map applied to the Gaussian vector <span>$y$</span>, it is Gaussian. Substituting <span>$y = X\beta + \varepsilon$</span> gives</p>

$$
\hat\beta = \beta + (X^T X)^{-1}X^T\varepsilon,
$$

<p>so <span>$\mathbb{E}[\hat\beta] = \beta$</span> and <span>$\operatorname{Var}(\hat\beta) = \sigma^2(X^T X)^{-1}$</span>.</p>

<hr>

<p><strong>Distribution of <span>$\mathrm{SSE}$</span>.</strong></p>

<p>The residuals are <span>$e = (I - H)\varepsilon$</span>. The matrix <span>$I - H$</span> is symmetric and idempotent with rank <span>$n - p$</span>. By the spectral theorem, any idempotent of rank <span>$k$</span> applied to a standard normal vector produces a <span>$\chi^2_k$</span> quadratic form. Therefore</p>

$$
\frac{\mathrm{SSE}}{\sigma^2} = \frac{\varepsilon^T(I-H)\varepsilon}{\sigma^2} \sim \chi^2_{n-p}.
$$

<hr>

<p><strong>Independence of <span>$\hat\beta$</span> and <span>$s^2$</span>.</strong></p>

<p><span>$\hat\beta$</span> depends on <span>$Hy$</span> and <span>$s^2$</span> depends on <span>$(I-H)y$</span>. Since <span>$H$</span> and <span>$I-H$</span> are orthogonal projections, <span>$Hy$</span> and <span>$(I-H)y$</span> are uncorrelated Gaussian vectors and therefore independent.</p>

<hr>

<p><strong>Distribution of <span>$R^2$</span> under the global null.</strong></p>

<p>Under <span>$H_0$</span>, <span>$\mathrm{SSR}/\sigma^2 \sim \chi^2_{p-1}$</span> and <span>$\mathrm{SSE}/\sigma^2 \sim \chi^2_{n-p}$</span>, independently. Their ratio is an <span>$F_{p-1,n-p}$</span> variable. Since <span>$R^2 = \mathrm{SSR}/\mathrm{SST}$</span> and <span>$\mathrm{SST} = \mathrm{SSR} + \mathrm{SSE}$</span>, the transformation</p>

$$
R^2 = \frac{V_1/(p-1)}{V_1/(p-1) + V_2/(n-p)},
\quad V_1 \sim \chi^2_{p-1},\; V_2 \sim \chi^2_{n-p},
$$

<p>yields the Beta distribution with parameters <span>$((p-1)/2, (n-p)/2)$</span>. Its mean is <span>$(p-1)/(n-1)$</span>, confirming that <span>$R^2$</span> inflates mechanically with more predictors.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

## Influence Diagnostics

An observation can affect the fit in two distinct ways: through its **residual** (how far its response is from the fitted line) and through its **leverage** (how extreme its predictor values are). Cook's distance combines both.

**Leverage.** The leverage of observation $i$ is the $i$-th diagonal entry of the hat matrix:

$$
h_{ii} = x_i^T(X^TX)^{-1}x_i, \qquad 0 \le h_{ii} \le 1, \qquad \sum_i h_{ii} = p.
$$

High leverage means $x_i$ (the $i$-th row of $X$, i.e. the vector of predictor values for observation $i$) is far from the center of the predictor space, so it has strong pull on the fitted line. A common threshold is $h_{ii} > 2p/n$.

**Studentized residuals.** The raw residual

$$
e_i = y_i - \hat{y}_i
$$

has variance $\sigma^2(1-h_{ii})$, so residuals at high-leverage points are automatically smaller. To compare residuals on a common scale, define the internally studentized residual

$$
r_i = \frac{e_i}{s\sqrt{1-h_{ii}}},
$$

and the externally studentized residual (delete-one version)

$$
t_i = \frac{e_i}{s_{(i)}\sqrt{1-h_{ii}}},
$$

where $s_{(i)}$ is the residual standard error from the model with observation $i$ deleted. Under the model, $t_i \sim t_{n-p-1}$.

**Cook's distance.** Cook's $D_i$ measures the total shift in all fitted values when observation $i$ is removed:

$$
D_i = \frac{\sum_{j=1}^n (\hat{y}_j - \hat{y}_{j(i)})^2}{p\,s^2},
$$

where $\hat{y}_{j(i)}$ is the fitted value from the model with observation $i$ deleted.

**Three equivalent decompositions.** Via the Sherman-Morrison-Woodbury update (see details below), the numerator simplifies to give:

$$
D_i
= \underbrace{\frac{e_i^2}{p\,s^2\,(1-h_{ii})^2} \cdot h_{ii}}_{\text{squared residual} \times \text{leverage}}
= \underbrace{\frac{r_i^2}{p}\cdot\frac{h_{ii}}{1-h_{ii}}}_{\text{outlyingness} \times \text{leverage ratio}}
= \underbrace{\frac{(\hat\beta - \hat\beta_{(i)})^T(X^TX)(\hat\beta - \hat\beta_{(i)})}{p\,s^2}}_{\text{observed influence} / \text{expected influence}}.
$$

**Distribution and thresholds.** Cook's $D_i$ is not exactly $F$-distributed, but it is interpreted by comparing $D_i$ to quantiles of $F_{p,\,n-p}$. The observed-over-expected framing makes the threshold $D_i > 1$ natural: it means observation $i$ shifts $\hat\beta$ more than the median random perturbation would. A more sensitive threshold in small samples is $D_i > 4/n$.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Derivation of the Cook's distance decompositions</span></summary>

<p><strong>Sherman-Morrison-Woodbury update.</strong></p>

<p>When observation $i$ is deleted, the leave-one-out OLS estimate satisfies</p>

$$
\hat\beta_{(i)} = \hat\beta - \frac{(X^TX)^{-1}x_i\,e_i}{1 - h_{ii}}.
$$

<p>This follows from the rank-1 update formula for matrix inverses: removing row $x_i^T$ from $X$ changes $(X^TX)^{-1}$ by a rank-1 correction, and the residual at the deleted point from the full model is $e_i$.</p>

<hr>

<p><strong>Numerator of $D_i$.</strong></p>

<p>The change in fitted values is $X(\hat\beta - \hat\beta_{(i)}) = X(X^TX)^{-1}x_i e_i/(1-h_{ii})$. So</p>

$$
\sum_j(\hat{y}_j - \hat{y}_{j(i)})^2
= \lVert X(\hat\beta-\hat\beta_{(i)})\rVert^2
= \frac{e_i^2}{(1-h_{ii})^2}\,x_i^T(X^TX)^{-1}X^TX(X^TX)^{-1}x_i
= \frac{e_i^2\,h_{ii}}{(1-h_{ii})^2}.
$$

<hr>

<p><strong>Connection to $r_i$.</strong></p>

<p>Since $r_i = e_i/(s\sqrt{1-h_{ii}})$, we have $e_i^2/s^2 = r_i^2(1-h_{ii})$. Substituting:</p>

$$
D_i = \frac{e_i^2 h_{ii}}{p\,s^2(1-h_{ii})^2} = \frac{r_i^2(1-h_{ii})\,h_{ii}}{p(1-h_{ii})^2} = \frac{r_i^2}{p}\cdot\frac{h_{ii}}{1-h_{ii}}.
$$

<hr>

<p><strong>Connection to coefficient shift.</strong></p>

<p>The third form follows directly from substituting the Sherman-Morrison-Woodbury update into the numerator $(\hat\beta-\hat\beta_{(i)})^T(X^TX)(\hat\beta-\hat\beta_{(i)})$ and noting $x_i^T(X^TX)^{-1}x_i = h_{ii}$.</p>

<hr>

<p><strong>Distribution.</strong></p>

<p>Under the model, $r_i^2/(1-r_i^2/(n-p-1)) \sim F_{1,n-p-1}$ (via the connection between internally and externally studentized residuals). Cook's $D_i = r_i^2 h_{ii}/(p(1-h_{ii}))$ is a product of a random variable related to $\chi^2_1$ and a fixed leverage term $h_{ii}$, so it does not follow a standard named distribution. Comparison to $F_{p,n-p}$ quantiles is interpretive: $D_i$ corresponds to approximately the $F_{p,n-p}$ percentile for the amount by which the observation shifts $\hat\beta$ relative to sampling noise.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

## Hypothesis Testing

### Single Coefficient Test

To test $H_0: \beta_{j} = 0$, use the $t$ statistic

$$
t_{j} = \frac{\hat\beta_{j}}{s\sqrt{[(X^T X)^{-1}]_{jj}}}
\sim t_{n-p}
\quad \text{under } H_0.
$$

This is the coefficient estimate divided by its standard error.

### General Linear Hypothesis Test

For a general linear hypothesis

$$
H_0: R\beta = r,
$$

with $R$ of rank $q$, the $F$ statistic is

$$
F = \frac{(R\hat\beta - r)^T \bigl[R(X^T X)^{-1}R^T\bigr]^{-1}(R\hat\beta - r) / q}{s^2}
\sim F_{q,\,n-p}
\quad \text{under } H_0.
$$

This framework covers testing several coefficients simultaneously, comparing nested models, and testing whether a newly added feature improves fit.

### ANOVA and the Regression F-Test

For the global null that all $p-1$ slope coefficients are zero,

$$
F = \frac{\mathrm{SSR}/(p-1)}{\mathrm{SSE}/(n-p)} \sim F_{p-1,\,n-p}
\quad \text{under } H_0.
$$

This is the regression ANOVA $F$-test.

### New Feature Test

Let the reduced model have $p_R$ columns with residual sum of squares $\mathrm{SSE}_R$, and the full model have $p_F$ columns with $\mathrm{SSE}_F$, where $q = p_F - p_R$ is the number of additional predictors. Then

$$
F = \frac{(\mathrm{SSE}_{R} - \mathrm{SSE}_{F})/q}{\mathrm{SSE}_{F}/(n - p_{F})} \sim F_{q,\,n-p_{F}}
\quad \text{under } H_0.
$$

Large values indicate that the new features reduced residual error more than expected under noise alone.

### MANOVA

MANOVA is the multivariate-response analogue of ANOVA and regression. Instead of a scalar response, each observation has a response vector. The question is whether groups or predictors shift the joint mean vector.

Common test statistics include Wilks' Lambda, Pillai's Trace, Hotelling-Lawley Trace, and Roy's Largest Root. The guiding idea remains: compare explained variation to unexplained variation, now in matrix form. Each statistic summarizes a different aspect of the eigenvalues of the matrix $\mathrm{SSR}\cdot\mathrm{SSE}^{-1}$.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Connections among the $t$ test, regression $F$ test, and new-feature test</span></summary>

<p><strong>One restriction.</strong> When <span>$q = 1$</span>, the general <span>$F$</span> statistic satisfies</p>

$$
F = t^2.
$$

<p>So the single-coefficient <span>$t$</span> test and the one-restriction <span>$F$</span> test are mathematically equivalent.</p>

<hr>

<p><strong>Regression $F$ as a special case.</strong> The global regression <span>$F$</span> test is the general linear hypothesis test with <span>$R = [0\mid I_{p-1}]$</span> (dropping the intercept row) and <span>$r = 0$</span>. Its numerator sum of squares is exactly <span>$\mathrm{SSR}$</span>.</p>

<hr>

<p><strong>Nested models as a general linear hypothesis.</strong> The reduced-vs-full test asks whether the coefficients of the additional <span>$q$</span> predictors are jointly zero. That is a linear hypothesis <span>$R\beta = 0$</span> of rank <span>$q$</span>, so the general formula applies directly.</p>

<hr>

<p><strong>ANOVA as regression.</strong> Classical one-way ANOVA is a regression with group indicator columns. Its <span>$F$</span> test is therefore a special regression <span>$F$</span> test with group dummies as predictors.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

## Multicollinearity

Multicollinearity refers to near-linear dependence among predictors. It does not bias $\hat\beta$ but inflates its variance, destabilizes individual coefficient estimates, and makes interpretation unreliable. Prediction can remain good even when coefficients become erratic.

**Simple regression baseline.** Start with one predictor $x_{1}$ and standardized variables (mean 0, variance 1):

$$
y = \beta_0 + \beta_1 x_1 + \varepsilon.
$$

Let $\rho_{1y} = \mathrm{cor}(x_{1}, y)$. Then:

* $\hat\beta_{1} = \rho_{1y}$ (for standardized variables).
* $\mathrm{Var}(\hat\beta_{1}) = \sigma^2/n$.
* $t_{1} = \rho_{1y}\sqrt{n}/\hat\sigma$.
* $R^2 = \rho_{1y}^2$.

**Two predictors.** Now add $x_{2}$ with $\mathrm{cor}(x_{1}, y) = \rho_{1y}$, $\mathrm{cor}(x_{2}, y) = \rho_{2y}$, and $\mathrm{cor}(x_{1}, x_{2}) = \rho_{12}$. For standardized variables, the OLS estimates are

$$
\hat\beta_{1} = \frac{\rho_{1y} - \rho_{12}\rho_{2y}}{1 - \rho_{12}^2},
\qquad
\hat\beta_{2} = \frac{\rho_{2y} - \rho_{12}\rho_{1y}}{1 - \rho_{12}^2}.
$$

The variance of $\hat\beta_{1}$ becomes

$$
\mathrm{Var}(\hat\beta_{1}) = \frac{\sigma^2}{n}\cdot\frac{1}{1 - \rho_{12}^2} = \mathrm{VIF} \cdot \mathrm{Var}_{\mathrm{simple}}(\hat\beta_{1}),
\quad \mathrm{VIF} = \frac{1}{1-\rho_{12}^2}.
$$

The $t$ statistic for $\hat\beta_{1}$ is

$$
t_{1} = \frac{\rho_{1y} - \rho_{12}\rho_{2y}}{\sqrt{1-\rho_{12}^2}} \cdot \frac{\sqrt{n}}{\hat\sigma}.
$$

The $R^2$ for the two-predictor model is

$$
R^2 = \frac{\rho_{1y}^2 + \rho_{2y}^2 - 2\rho_{1y}\rho_{2y}\rho_{12}}{1 - \rho_{12}^2}.
$$

The marginal gain from adding $x_{2}$ (partial $R^2$ of $x_{2}$ given $x_{1}$) is

$$
\Delta R^2 = \frac{(\rho_{2y} - \rho_{1y}\rho_{12})^2}{1 - \rho_{12}^2}.
$$

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why $\hat\beta_1$ changes when you add $x_2$: partial effects, multicollinearity, and sign flips</span></summary>

<p><strong>What changes and why.</strong> In the simple model $y \sim x_1$, $\hat\beta_1$ captures all association between $x_1$ and $y$, including anything shared with $x_2$. In the multiple model $y \sim x_1 + x_2$, $\hat\beta_1$ measures the effect of $x_1$ <em>holding $x_2$ fixed</em>. It uses only the variation in $x_1$ that is orthogonal to (not explained by) $x_2$.</p>

<p><strong>Formal view.</strong> The multiple regression coefficient can be written as:</p>

$$\hat\beta_1^{(x_1 + x_2)} = \frac{\mathrm{Cov}(y, \tilde{x}_1)}{\mathrm{Var}(\tilde{x}_1)},$$

<p>where $\tilde{x}_1$ is the residual of $x_1$ after regressing out $x_2$ (i.e., the part of $x_1$ that $x_2$ cannot explain). The numerator is the covariance of $y$ with this residual; the denominator is the remaining variance of $x_1$.</p>

<p><strong>What happens as $\rho_{12}$ increases.</strong> When $x_1$ and $x_2$ are highly correlated, most of $x_1$'s variation is explained by $x_2$, so $\mathrm{Var}(\tilde{x}_1) \to 0$. With a tiny denominator:</p>

<ul>
<li>The estimate becomes <strong>noisy</strong> (high variance, wide confidence intervals).</li>
<li>The coefficient can <strong>shrink toward zero</strong> because the numerator also shrinks (less unique covariance with $y$).</li>
<li>The coefficient can <strong>flip sign</strong> when $\rho_{12}$ exceeds $\rho_{2y}/\rho_{1y}$, because the numerator $\rho_{1y} - \rho_{12}\rho_{2y}$ in the standardized formula can change sign even though both $\rho_{1y} > 0$ and $\rho_{2y} > 0$.</li>
</ul>

<p><strong>Omitted variable bias vs multicollinearity.</strong> These are opposite sides of the same coin:</p>

<ul>
<li><strong>Simple model</strong> ($y \sim x_1$ only): $\hat\beta_1$ is stable and precise, but <em>biased</em> if $x_2$ is a true predictor that is correlated with $x_1$. The bias is $\rho_{12} \cdot \beta_2$, which pushes $\hat\beta_1$ away from its true value.</li>
<li><strong>Multiple model</strong> ($y \sim x_1 + x_2$): $\hat\beta_1$ is <em>unbiased</em> (if the model is correct), but has high variance and can be unstable when $\rho_{12}$ is large.</li>
</ul>

<p>Adding $x_2$ trades bias for variance. In the simple model you get a wrong but stable answer; in the multiple model you get the right answer on average but any single estimate may be far off.</p>

<p><strong>Extreme case: perfect collinearity.</strong> If $x_1 = x_2$ exactly, $\mathrm{Var}(\tilde{x}_1) = 0$ and the denominator is zero. The design matrix $X^\top X$ is singular, and OLS cannot produce estimates at all. The model is not identifiable because infinitely many combinations of $\beta_1$ and $\beta_2$ produce the same predictions.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Two-predictor formula derivation via $2\times 2$ matrix inversion</span></summary>

<p>For standardized predictors <span>$(x_1, x_2)$</span> with unit variance and correlation <span>$\rho_{12}$</span>, the matrix <span>$X^T X / n$</span> (ignoring the intercept block) is</p>

$$
\Sigma = \begin{pmatrix} 1 & \rho_{12} \\ \rho_{12} & 1 \end{pmatrix}.
$$

<p>Its inverse is</p>

$$
\Sigma^{-1} = \frac{1}{1-\rho_{12}^2}\begin{pmatrix} 1 & -\rho_{12} \\ -\rho_{12} & 1 \end{pmatrix}.
$$

<p>The vector <span>$X^T y / n$</span> equals <span>$(\rho_{1y}, \rho_{2y})^T$</span> (the sample correlations with <span>$y$</span>). So</p>

$$
\begin{pmatrix}\hat\beta_1 \\ \hat\beta_2\end{pmatrix}
= \Sigma^{-1}\begin{pmatrix}\rho_{1y}\\\rho_{2y}\end{pmatrix}
= \frac{1}{1-\rho_{12}^2}\begin{pmatrix}\rho_{1y} - \rho_{12}\rho_{2y} \\ \rho_{2y} - \rho_{12}\rho_{1y}\end{pmatrix}.
$$

<p>The diagonal entry of <span>$\Sigma^{-1}$</span> is <span>$1/(1-\rho_{12}^2)$</span>, so</p>

$$
\mathrm{Var}(\hat\beta_1) = \frac{\sigma^2}{n}\cdot\frac{1}{1-\rho_{12}^2}.
$$

<p>For <span>$R^2$</span>, use <span>$R^2 = \hat\beta^T \Sigma \hat\beta$</span> for standardized <span>$y$</span>. Expanding</p>

$$
R^2 = \frac{1}{(1-\rho_{12}^2)^2}(\rho_{1y}-\rho_{12}\rho_{2y},\; \rho_{2y}-\rho_{12}\rho_{1y})\begin{pmatrix}1&\rho_{12}\\\rho_{12}&1\end{pmatrix}\begin{pmatrix}\rho_{1y}-\rho_{12}\rho_{2y}\\\rho_{2y}-\rho_{12}\rho_{1y}\end{pmatrix}.
$$

<p>After expanding the quadratic form, this simplifies to <span>$(\rho_{1y}^2+\rho_{2y}^2-2\rho_{1y}\rho_{2y}\rho_{12})/(1-\rho_{12}^2)$</span>.</p>

<p>The partial <span>$R^2$</span> of <span>$x_2$</span> given <span>$x_1$</span> equals the squared partial correlation of <span>$x_2$</span> with <span>$y$</span> after removing the effect of <span>$x_1$</span>, which works out to <span>$(\rho_{2y}-\rho_{1y}\rho_{12})^2/(1-\rho_{12}^2)$</span>.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

**Numerical example.** Set $\rho_{1y} = 0.6$ and $\rho_{2y} = 0.4$ (both predictors positively correlated with $y$). As the inter-predictor correlation $\rho_{12}$ increases:

| $\rho_{12}$ | $\hat\beta_{1}$ | $\hat\beta_{2}$ | VIF | $R^2$ |
|-------------|-----------------|-----------------|-----|-------|
| 0.0 | 0.600 | 0.400 | 1.00 | 0.520 |
| 0.5 | 0.587 | 0.100 | 1.33 | 0.507 |
| 0.8 | 0.556 | $-0.111$ | 2.78 | 0.493 |
| 0.9 | 0.811 | $-0.329$ | 5.26 | 0.487 |

Note that $\hat\beta_{2}$ flips sign at $\rho_{12} > \rho_{2y}/\rho_{1y} = 0.4/0.6 \approx 0.67$. At $\rho_{12} = 0.8$ and $\rho_{12} = 0.9$, $x_{2}$ has a negative estimated coefficient even though $\mathrm{cor}(x_{2}, y) = 0.4 > 0$.

**Key takeaways.** Coefficient signs can flip even when both predictors are positively correlated with $y$, once $\rho_{12} > \rho_{2y}/\rho_{1y}$. The numerator of $t_{1}$ is $\rho_{1y} - \rho_{12}\rho_{2y}$: as $\rho_{12} \to 1$, this approaches $\rho_{1y} - \rho_{2y} = 0.2 > 0$ but the denominator $\sqrt{1-\rho_{12}^2} \to 0$, so variance diverges and standard errors explode even while the point estimate stays nonzero. Prediction, measured by $R^2$, degrades only slowly as $\rho_{12}$ increases, meaning the model still fits the data while individual coefficients become unreliable. This is the central danger of multicollinearity: the model can look good globally while telling a misleading story about each predictor.

**VIF thresholds.** $\mathrm{VIF} = 1/(1-\rho_{12}^2)$. VIF exceeds 5 when $\rho_{12} > 0.89$, and exceeds 10 when $\rho_{12} > 0.95$. A commonly used rule of thumb: VIF $> 10$ signals serious collinearity.

### Diagnosis

* Large standard errors despite a strong overall $R^2$.
* Unstable coefficient estimates across bootstrap refits or data subsets.
* Large VIF values for individual predictors.
* Large condition number of $X^T X$.
* Predictors that tell nearly the same substantive story.

### Remedies

* Drop or combine redundant features using domain knowledge.
* Ridge regression when prediction stability is the primary goal.
* Lasso when sparsity and interpretability are needed.
* Principal components regression (PCR) when the collinearity has a low-dimensional structure.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Choosing a remedy: prediction vs interpretation, and when each method makes sense</span></summary>

<p>All remedies address the same underlying problem: there is not enough independent variation in $x_1$ and $x_2$ to identify both effects cleanly. The right choice depends on whether you care about prediction performance or interpretable coefficients.</p>

<p><strong>1. Regularization: Ridge or Elastic Net (default for prediction).</strong> Ridge (L2 penalty) shrinks correlated coefficients toward each other, sharing the weight. It does not require choosing which variable to keep and handles many correlated features simultaneously. Elastic Net combines L1 and L2 and can zero out some coefficients while shrinking others. Use when the goal is prediction and you do not need clean interpretation of individual coefficients. This is usually the best practical first choice for prediction tasks.</p>

<p><strong>2. Drop one feature (simplest, often strong).</strong> Pick the predictor with higher correlation with $y$ or better out-of-sample performance and discard the other. Use when the features are near-duplicates (e.g., two measures of the same quantity), when interpretability matters, or when you want a simple model. Surprisingly effective in practice because the dropped variable's signal is mostly captured by the retained one.</p>

<p><strong>3. Residualization (orthogonalization).</strong> Construct the residual $x_2^{\perp} = x_2 - \hat{x}_2(x_1)$ (the part of $x_2$ not explained by $x_1$), then regress $y \sim x_1 + x_2^{\perp}$. Now $x_1$ carries the "baseline effect" and $x_2^{\perp}$ carries the "incremental effect beyond $x_1$." Use when you care about causal or incremental interpretation (common in economics and causal inference). The downside is that it is asymmetric: residualizing $x_2$ on $x_1$ gives different results than residualizing $x_1$ on $x_2$, so you must choose which variable is the baseline based on domain knowledge.</p>

<p><strong>4. Combine into a single variable.</strong> Form a weighted average $z = \alpha x_1 + (1-\alpha) x_2$ (e.g., $\alpha = 0.5$ for equal weighting, or proportional to each variable's correlation with $y$). Use when both variables measure the same latent concept (e.g., two noisy measures of income, two correlated risk scores). Reduces the problem from two noisy estimates to one stable estimate of the underlying factor.</p>

<p><strong>5. Ratio $x_2 / x_1$.</strong> Use only when theory supports a ratio interpretation: price per unit, rate of return, efficiency (output/input). Do not use ratios solely to "fix" correlation; this introduces spurious nonlinearity and can distort inference if the ratio has no substantive meaning.</p>

<p><strong>6. PCA / latent factor.</strong> Extract the first principal component of $(x_1, x_2)$, which captures the shared variation. Regress $y$ on this component (or the top $K$ components for many predictors). Use when there are many correlated variables and you want a low-dimensional representation. The tradeoff is reduced interpretability: the principal component is a linear combination, not a single observable feature.</p>

<p><strong>Practical decision tree:</strong></p>

<ul>
<li><strong>Goal is prediction:</strong> Ridge or Elastic Net (first choice). PCA if there are many correlated features.</li>
<li><strong>Goal is interpretation, features measure the same thing:</strong> combine (average or PCA).</li>
<li><strong>Goal is interpretation, want incremental effect:</strong> residualize.</li>
<li><strong>Goal is interpretation, features are redundant:</strong> drop one.</li>
<li><strong>Theory says ratio is meaningful:</strong> use the ratio.</li>
<li><strong>Unsure:</strong> fit the full model, check VIF and coefficient stability across cross-validation folds, then try Ridge and drop-one, and compare out-of-sample performance.</li>
</ul>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Principal components regression.** Multicollinearity means the columns of $X$ are nearly linearly dependent, so $X^TX$ has small eigenvalues and OLS amplifies the corresponding directions. PCR addresses this by rotating to the principal component basis via the SVD of $X$, retaining only the $K$ directions of highest variance, and regressing $y$ on those $K$ components. The low-variance directions that drive coefficient instability are discarded entirely.

The PCR estimator in the original predictor space is

$$
\hat\beta^{\mathrm{PCR}} = V_K \hat\alpha_K,
\qquad
\hat\alpha_K = (Z_K^T Z_K)^{-1} Z_K^T y = D_K^{-1} U_K^T y,
$$

where $Z_K = U_K D_K$ are the first $K$ principal components and $V_K$ maps back to the original coordinates. Choosing $K < p$ makes the estimator biased but substantially reduces variance in the unstable directions.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Theoretical basis of PCA and PCR</span></summary>

<p><strong>Definition via variance maximization.</strong></p>

<p>The first principal component direction <span>$v_1 \in \mathbb{R}^p$</span> is the unit vector that maximizes the variance of the projected scores:</p>

$$
v_1 = \arg\max_{\lVert v \rVert = 1} \mathrm{Var}(Xv) = \arg\max_{\lVert v \rVert = 1} \frac{v^T X^T X v}{n-1}.
$$

<p>The solution is the eigenvector of <span>$X^TX$</span> corresponding to its largest eigenvalue. Each subsequent direction <span>$v_k$</span> maximizes variance subject to being orthogonal to all previous directions:</p>

$$
v_k = \arg\max_{\lVert v \rVert = 1,\; v \perp v_1, \ldots, v_{k-1}} \frac{v^T X^T X v}{n-1}.
$$

<p>This gives the full set of PC directions <span>$v_1, \ldots, v_p$</span>, ordered from most to least variance. Orthogonality means each direction captures variation not already explained by the previous ones.</p>

<hr>

<p><strong>SVD of the design matrix.</strong></p>

<p>Any matrix <span>$X \in \mathbb{R}^{n \times p}$</span> admits a singular value decomposition</p>

$$
X = U D V^T,
$$

<p>where:</p>

<ul>
<li><span>$U \in \mathbb{R}^{n \times p}$</span> has <strong>orthonormal columns</strong>: <span>$U^T U = I_p$</span>, meaning each column <span>$u_k$</span> is a unit vector and distinct columns are perpendicular (<span>$u_j^T u_k = 0$</span> for <span>$j \ne k$</span>). These are the left singular vectors, one for each observation direction in the rotated space.</li>
<li><span>$D = \mathrm{diag}(d_1, \ldots, d_p)$</span> with <span>$d_1 \ge d_2 \ge \cdots \ge d_p \ge 0$</span> the singular values, measuring how much variance each direction captures.</li>
<li><span>$V \in \mathbb{R}^{p \times p}$</span> is orthogonal (<span>$V^T V = V V^T = I_p$</span>): its columns <span>$v_1, \ldots, v_p$</span> are the <strong>principal component directions</strong> in predictor space: unit vectors pointing along the axes of maximum variance. They are exactly the eigenvectors of <span>$X^TX$</span>, since <span>$X^TX = VD^2V^T$</span>.</li>
</ul>

<hr>

<p><strong>Principal components.</strong></p>

<p>The principal components are the projections of the data onto each direction <span>$v_k$</span>:</p>

$$
Z = XV = UD.
$$

<p>The <span>$k$</span>-th column <span>$z_k = Xv_k = d_k u_k$</span> is the score of each observation along the <span>$k$</span>-th PC direction. Since <span>$U$</span> has orthonormal columns, <span>$Z^TZ = D^TU^TUD = D^2$</span>, so the columns of <span>$Z$</span> are orthogonal with variance <span>$d_k^2/(n-1)$</span>. Directions with small <span>$d_k$</span> are nearly constant across observations, which is exactly what near-collinear combinations in <span>$X$</span> look like.</p>

<hr>

<p><strong>PCR estimator.</strong></p>

<p>Retain the first <span>$K$</span> components. Since <span>$Z_K^T Z_K = D_K^2$</span> is diagonal,</p>

$$
\hat\alpha_K = (Z_K^T Z_K)^{-1} Z_K^T y = D_K^{-2} Z_K^T y = D_K^{-1} U_K^T y.
$$

<p>Mapping back to the original space: <span>$\hat\beta^{\mathrm{PCR}} = V_K \hat\alpha_K$</span>. In the SVD basis, OLS would divide by each <span>$d_k$</span>; PCR simply sets the last <span>$p - K$</span> components to zero rather than dividing by near-zero values.</p>

<hr>

<p><strong>Bias-variance tradeoff.</strong></p>

<p>Let <span>$\gamma = V^T \beta$</span> be the true coefficient in the principal component basis. PCR discards components <span>$K+1, \ldots, p$</span>, introducing bias</p>

$$
\mathrm{Bias}(\hat\beta^{\mathrm{PCR}}) = -\sum_{k=K+1}^{p} \gamma_k v_k,
$$

<p>where <span>$v_k$</span> is the <span>$k$</span>-th column of <span>$V$</span>. If the true signal is concentrated in the high-variance directions (a common assumption), this bias is small. The variance of each retained component <span>$\hat\alpha_k$</span> is <span>$\sigma^2 / d_k^2$</span>; by dropping directions with small <span>$d_k$</span>, PCR avoids the exploding variance that makes OLS unreliable under collinearity.</p>

<hr>

<p><strong>Connection to ridge.</strong></p>

<p><em>Step 1: Express ridge in the SVD basis.</em></p>

<p>Substitute <span>$X = UDV^T$</span> into the ridge closed form <span>$\hat\beta^{\mathrm{ridge}} = (X^TX + \lambda I)^{-1}X^Ty$</span>:</p>

$$
X^TX + \lambda I = VD^2V^T + \lambda VV^T = V(D^2 + \lambda I)V^T.
$$

<p>Inverting (using <span>$V^{-1} = V^T$</span> since <span>$V$</span> is orthogonal):</p>

$$
(X^TX + \lambda I)^{-1} = V(D^2 + \lambda I)^{-1}V^T.
$$

<p>And <span>$X^Ty = VDU^Ty$</span>. So</p>

$$
\hat\beta^{\mathrm{ridge}} = V(D^2 + \lambda I)^{-1}V^T \cdot VDU^Ty = V(D^2 + \lambda I)^{-1}DU^Ty.
$$

<hr>

<p><em>Step 2: Compare to OLS in the SVD basis.</em></p>

<p>OLS satisfies <span>$\hat\beta^{\mathrm{OLS}} = (X^TX)^{-1}X^Ty = VD^{-2}V^T \cdot VDU^Ty = VD^{-1}U^Ty$</span>. Writing both in the rotated basis <span>$\tilde\beta = V^T\beta$</span>:</p>

<ul>
<li><strong>OLS:</strong> <span>$\tilde\beta_k^{\mathrm{OLS}} = \dfrac{1}{d_k}(U^Ty)_k$</span></li>
<li><strong>Ridge:</strong> <span>$\tilde\beta_k^{\mathrm{ridge}} = \dfrac{d_k}{d_k^2 + \lambda}(U^Ty)_k = \dfrac{d_k^2}{d_k^2+\lambda} \cdot \dfrac{1}{d_k}(U^Ty)_k$</span></li>
</ul>

<p>So ridge multiplies the OLS solution along each PC direction by the shrinkage factor</p>

$$
s_k(\lambda) = \frac{d_k^2}{d_k^2 + \lambda} \in (0, 1).
$$

<p>When <span>$d_k$</span> is large (high-variance direction), <span>$s_k \approx 1$</span> and ridge barely shrinks. When <span>$d_k \approx 0$</span> (near-collinear direction), <span>$s_k \approx 0$</span> and ridge strongly shrinks.</p>

<hr>

<p><em>Step 3: Compare shrinkage profiles of ridge and PCR.</em></p>

<p>PCR applies a hard 0/1 threshold: direction <span>$k$</span> is kept at full weight if <span>$k \le K$</span> and zeroed otherwise. Ridge applies a smooth shrinkage that depends on <span>$d_k^2$</span>:</p>

$$
\text{PCR: } s_k = \mathbf{1}[k \le K], \qquad \text{Ridge: } s_k = \frac{d_k^2}{d_k^2 + \lambda}.
$$

<p>Both suppress the small-<span>$d_k$</span> directions that cause instability under collinearity. The difference is that ridge never fully zeros a direction; it is a continuous relaxation of PCR's hard truncation. As <span>$\lambda \to \infty$</span>, all <span>$s_k \to 0$</span>; as <span>$\lambda \to 0$</span>, all <span>$s_k \to 1$</span> and ridge recovers OLS.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

## Ridge and Lasso

When collinearity is severe or the number of predictors is large relative to $n$, OLS becomes unreliable. Ridge and lasso regularization trade a small amount of bias for a large reduction in variance.

**Ridge regression** solves

$$
\hat\beta^{\mathrm{ridge}} = \arg\min_\beta \left\{\lVert y - X\beta \rVert_2^2 + \lambda\lVert\beta\rVert_2^2\right\}.
$$

The closed form is

$$
\hat\beta^{\mathrm{ridge}} = (X^T X + \lambda I)^{-1}X^T y.
$$

Adding $\lambda I$ makes the matrix always invertible, directly addressing the multicollinearity problem. Ridge shrinks all coefficients continuously toward zero and is most effective when many predictors carry partial signal.

**Lasso** solves

$$
\hat\beta^{\mathrm{lasso}} = \arg\min_\beta \left\{\lVert y - X\beta \rVert_2^2 + \lambda\lVert\beta\rVert_1\right\}.
$$

Lasso both shrinks and can set coefficients exactly to zero, producing sparse models. It has no closed form in general and is solved via coordinate descent or related algorithms. Under a single predictor, however, an exact closed form exists and reveals the soft-thresholding mechanism behind sparsity.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Lasso analytical solution under one predictor</span></summary>

<p><strong>Step 1: Reduce the objective.</strong></p>

<p>With a single standardized predictor <span>$x \in \mathbb{R}^n$</span> (so <span>$x^Tx = n$</span>) and response <span>$y$</span>, the lasso objective is</p>

$$
\min_{\beta} \left\{ \sum_{i=1}^n (y_i - x_i \beta)^2 + \lambda |\beta| \right\}.
$$

<p>Expand the sum of squares:</p>

$$
\sum_i (y_i - x_i\beta)^2 = y^Ty - 2\beta x^Ty + \beta^2 x^Tx = y^Ty - 2\beta x^Ty + n\beta^2.
$$

<p>Let <span>$\hat\beta_{\mathrm{OLS}} = x^Ty/n$</span>. Complete the square in <span>$\beta$</span>:</p>

$$
n\beta^2 - 2\beta x^Ty = n\left(\beta^2 - \frac{2x^Ty}{n}\beta\right) = n(\beta - \hat\beta_{\mathrm{OLS}})^2 - n\hat\beta_{\mathrm{OLS}}^2.
$$

<p>The term <span>$y^Ty - n\hat\beta_{\mathrm{OLS}}^2$</span> does not depend on <span>$\beta$</span>, so minimizing the lasso objective is equivalent to</p>

$$
\min_{\beta}\; f(\beta) = n(\beta - \hat\beta_{\mathrm{OLS}})^2 + \lambda|\beta|.
$$

<hr>

<p><strong>Step 2: Subgradient of the objective.</strong></p>

<p>The function <span>$f(\beta)$</span> is convex but not differentiable at <span>$\beta = 0$</span> due to the <span>$|\beta|$</span> term. Its subgradient is</p>

$$
\partial f(\beta) =
\begin{cases}
2n(\beta - \hat\beta_{\mathrm{OLS}}) + \lambda & \text{if } \beta > 0, \\
2n(\beta - \hat\beta_{\mathrm{OLS}}) - \lambda & \text{if } \beta < 0, \\
-2n\hat\beta_{\mathrm{OLS}} + \lambda[-1,1] & \text{if } \beta = 0,
\end{cases}
$$

<p>where <span>$\lambda[-1,1]$</span> at <span>$\beta=0$</span> uses the fact that the subdifferential of <span>$|\beta|$</span> at zero is the interval <span>$[-1,1]$</span>. The optimality condition is <span>$0 \in \partial f(\beta)$</span>.</p>

<hr>

<p><strong>Step 3: Solve each case.</strong></p>

<p><em>Case 1: <span>$\beta > 0$</span>.</em> Setting the gradient to zero:</p>

$$
2n(\beta - \hat\beta_{\mathrm{OLS}}) + \lambda = 0
\implies
\beta = \hat\beta_{\mathrm{OLS}} - \frac{\lambda}{2n}.
$$

<p>For this to satisfy <span>$\beta > 0$</span> we need <span>$\hat\beta_{\mathrm{OLS}} > \lambda/(2n)$</span>.</p>

<p><em>Case 2: <span>$\beta < 0$</span>.</em> Setting the gradient to zero:</p>

$$
2n(\beta - \hat\beta_{\mathrm{OLS}}) - \lambda = 0
\implies
\beta = \hat\beta_{\mathrm{OLS}} + \frac{\lambda}{2n}.
$$

<p>For this to satisfy <span>$\beta < 0$</span> we need <span>$\hat\beta_{\mathrm{OLS}} < -\lambda/(2n)$</span>.</p>

<p><em>Case 3: <span>$\beta = 0$</span>.</em> The optimality condition <span>$0 \in \partial f(0)$</span> requires</p>

$$
0 \in -2n\hat\beta_{\mathrm{OLS}} + \lambda[-1,1]
\iff
2n\hat\beta_{\mathrm{OLS}} \in \lambda[-1,1]
\iff
|\hat\beta_{\mathrm{OLS}}| \le \frac{\lambda}{2n}.
$$

<hr>

<p><strong>Step 4: Combine into the soft-thresholding operator.</strong></p>

<p>Collecting the three cases:</p>

$$
\hat\beta_{\mathrm{lasso}} = \mathcal{S}_{\lambda/(2n)}(\hat\beta_{\mathrm{OLS}}),
\qquad
\mathcal{S}_\tau(z) = \mathrm{sign}(z)\,(|z| - \tau)_+,
$$

<p>where <span>$(u)_+ = \max(u, 0)$</span>. Explicitly:</p>

$$
\hat\beta_{\mathrm{lasso}} =
\begin{cases}
\hat\beta_{\mathrm{OLS}} - \dfrac{\lambda}{2n} & \text{if } \hat\beta_{\mathrm{OLS}} > \dfrac{\lambda}{2n}, \\[8pt]
0 & \text{if } |\hat\beta_{\mathrm{OLS}}| \le \dfrac{\lambda}{2n}, \\[8pt]
\hat\beta_{\mathrm{OLS}} + \dfrac{\lambda}{2n} & \text{if } \hat\beta_{\mathrm{OLS}} < -\dfrac{\lambda}{2n}.
\end{cases}
$$

<p>The OLS estimate is shrunk toward zero by <span>$\lambda/(2n)$</span> on each side, and set exactly to zero when it falls within the dead zone <span>$[-\lambda/(2n),\, \lambda/(2n)]$</span>. This is the origin of exact sparsity: unlike ridge, which only scales <span>$\hat\beta_{\mathrm{OLS}}$</span> by a factor strictly less than one, lasso subtracts a fixed amount and clips at zero.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

### Choosing Between Ridge and Lasso

* Use ridge when predictors are strongly correlated and prediction stability is the primary goal.
* Use lasso when automatic sparsity and interpretability matter: lasso selects features as part of fitting.
* Use elastic net when you want both grouped shrinkage under correlation and sparsity:

$$
\arg\min_\beta \left\{\lVert y - X\beta \rVert_2^2 + \lambda_1\lVert\beta\rVert_1 + \lambda_2\lVert\beta\rVert_2^2\right\}.
$$

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Geometric intuition: $\ell_2$ ball, $\ell_1$ diamond, and sparsity at corners</span></summary>

<p>Both ridge and lasso can be framed as constrained optimization: minimize the OLS objective subject to a constraint on the coefficient norm.</p>

<ul>
<li><strong>Ridge</strong> imposes <span>$\lVert\beta\rVert_2^2 \le t$</span>. The feasible set is a smooth sphere in <span>$\mathbb{R}^p$</span>. The OLS objective ellipsoid first touches the sphere at a smooth boundary point, so no coefficient is forced exactly to zero.</li>
<li><strong>Lasso</strong> imposes <span>$\lVert\beta\rVert_1 \le t$</span>. The feasible set is a diamond (hypercube rotated 45 degrees) with corners on the coordinate axes. The OLS ellipsoid tends to first contact the diamond at a corner, where one or more coordinates are zero. That is why lasso produces exact sparsity.</li>
</ul>

<p><strong>Bias-variance tradeoff.</strong> As <span>$\lambda$</span> increases, both estimators become more biased but have lower variance. The optimal <span>$\lambda$</span> is chosen by cross-validation to minimize prediction error, balancing these two forces.</p>

<p><strong>Ridge under multicollinearity.</strong> In singular-value terms, OLS amplifies small singular values of <span>$X$</span>, which arise from near-collinear columns. Ridge replaces each singular value <span>$d_j$</span> of <span>$X$</span> by <span>$d_j^2/(d_j^2 + \lambda)$</span> in the shrinkage factor, damping the unstable directions. Lasso does not have such a clean SVD interpretation but still regularizes the same directions via the <span>$\ell_1$</span> penalty.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

## Multiple Testing

### Single Hypothesis

A single linear restriction on $\beta$, such as $H_0: \beta_{j} = 0$ or $H_0: \beta_{j} = 1$, is handled by a $t$ test (or equivalently, by the $F$ test with $q = 1$). The nominal $p$-value from a single test controls the type I error at the declared level, assuming the test is pre-specified.

### Multiple Hypotheses

When testing several restrictions simultaneously (such as the joint null that a set of coefficients are all zero), use the joint $F$ test rather than running separate $t$ tests. If $q$ separate $t$ tests are run and significance is declared whenever any one passes, the probability of at least one false positive exceeds the nominal level $\alpha$.

### Multiple Testing Adjustment

When testing $m$ hypotheses one by one, two error rates are relevant. The **family-wise error rate (FWER)** is the probability of at least one false rejection: $\mathrm{FWER} = P(\text{any false rejection})$. The **false discovery rate (FDR)** is the expected proportion of rejections that are false: $\mathrm{FDR} = E[V/R]$, where $V$ is the number of false rejections and $R$ is the total number of rejections ($R > 0$). FDR $\le$ FWER, so FDR control is less stringent but more powerful.

* **Bonferroni.** Reject hypothesis $k$ when its $p$-value satisfies $p_k < \alpha/m$. Controls FWER at level $\alpha$ exactly, but is conservative when tests are positively correlated.
* **Holm.** A step-down procedure that uniformly improves on Bonferroni while still controlling FWER.
* **Benjamini-Hochberg (BH).** Controls FDR at level $\alpha$ under independence (and positive dependence). Substantially more powerful than FWER procedures in high-dimensional settings.

<br>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Holm and Benjamini-Hochberg procedures</span></summary>

<p><strong>Holm (step-down).</strong></p>

<ol>
<li>Sort the <span>$m$</span> p-values in ascending order: <span>$p_{(1)} \le p_{(2)} \le \cdots \le p_{(m)}$</span>, with corresponding hypotheses <span>$H_{(1)}, \ldots, H_{(m)}$</span>.</li>
<li>Find the smallest index <span>$k$</span> such that <span>$p_{(k)} > \alpha/(m - k + 1)$</span>.</li>
<li>Reject <span>$H_{(1)}, \ldots, H_{(k-1)}$</span>; retain <span>$H_{(k)}, \ldots, H_{(m)}$</span>.</li>
</ol>

<p>Holm always rejects at least as many hypotheses as Bonferroni, since the threshold <span>$\alpha/(m-k+1)$</span> is at least <span>$\alpha/m$</span>.</p>

<hr>

<p><strong>Benjamini-Hochberg (step-up).</strong></p>

<ol>
<li>Sort p-values: <span>$p_{(1)} \le \cdots \le p_{(m)}$</span>.</li>
<li>Find the largest index <span>$k$</span> such that <span>$p_{(k)} \le k\alpha/m$</span>.</li>
<li>Reject <span>$H_{(1)}, \ldots, H_{(k)}$</span>.</li>
</ol>

<p>The threshold <span>$k\alpha/m$</span> is a line through the origin in the p-value rank plot. BH rejects all hypotheses whose sorted p-value falls below this line up to the last crossing. Under independence, <span>$\mathrm{FDR} \le \alpha$</span>. Under arbitrary positive dependence (PRDS condition), the same bound holds.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

## FDR Control

### Why FDR

In high-dimensional settings (genomics, neuroimaging, large-scale feature selection), the number of hypotheses $m$ can be in the thousands or millions. FWER control at level $\alpha$ requires each individual test to pass a threshold of order $\alpha/m$, which becomes so stringent that almost nothing is discovered. FDR control accepts that some false discoveries will occur, but keeps their expected proportion bounded.

Formally, let $V$ be the number of false rejections and $R$ the total number of rejections. The FDR is

$$
\mathrm{FDR} = E\!\left[\frac{V}{R}\,\mathbf{1}(R > 0)\right].
$$

When all null hypotheses are true, FDR $=$ FWER. Otherwise FDR $\le$ FWER, so FDR control is strictly less conservative. The BH procedure controls FDR at level $\alpha$ and is the standard baseline.

### Knockoffs

The **knockoff filter** (Barber and Candès, 2015; model-X extension by Candès et al., 2018) provides exact finite-sample FDR control for variable selection without relying on asymptotic approximations or $p$-values.

**Idea.** Construct for each feature $x_j$ a synthetic "knockoff" copy $\tilde x_j$ that mimics the correlation structure of $x_j$ with all other features, but is conditionally independent of the response $y$ given the real features. Any feature that appears important relative to its own knockoff is a genuine signal; features that are interchangeable with their knockoffs are noise.

**Knockoff construction.** For the fixed-X setting with design matrix $X$, knockoffs $\tilde X$ satisfy

$$
\tilde X^T \tilde X = X^T X,
\qquad
\tilde X^T X = X^T X - \mathrm{diag}(s),
$$

for some $s \ge 0$ chosen to make $\tilde x_j$ as different from $x_j$ as possible while respecting the correlation constraint. The augmented matrix $[X\;\tilde X]$ is constructed explicitly via QR or SDP.

**Feature statistics.** Fit a model (e.g., lasso) on the augmented design $[X\;\tilde X]$ to obtain importance scores $Z_j$ for each original feature and $\tilde Z_j$ for each knockoff. Define the antisymmetric statistic

$$
W_j = f(Z_j, \tilde Z_j),
$$

where $f$ is any function satisfying $f(z, \tilde z) = -f(\tilde z, z)$, for example $W_j = Z_j - \tilde Z_j$ or $W_j = \max(Z_j, \tilde Z_j)\cdot \mathrm{sign}(Z_j - \tilde Z_j)$.

**Knockoff filter.** For a target FDR level $q$, choose the threshold

$$
T = \min\left\{t > 0 : \frac{1 + |\{j : W_j \le -t\}|}{|\{j : W_j \ge t\}|} \le q\right\},
$$

and reject $\{j : W_j \ge T\}$. The numerator $1 + |\{j: W_j \le -t\}|$ estimates the number of false discoveries (using knockoffs as nulls); the denominator counts total rejections.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why the knockoff filter controls FDR</span></summary>

<p><strong>Symmetry argument.</strong></p>

<p>For a null feature <span>$j$</span> (one where <span>$x_j \perp y \mid X_{-j}$</span>), the knockoff construction guarantees that <span>$x_j$</span> and <span>$\tilde x_j$</span> are exchangeable in the joint distribution with <span>$y$</span>. Therefore <span>$W_j$</span> is symmetric around zero: <span>$P(W_j \ge t) = P(W_j \le -t)$</span> for any <span>$t$</span>.</p>

<p><strong>FDP estimate.</strong></p>

<p>At threshold <span>$t$</span>, the number of false discoveries is <span>$V(t) = |\{\text{null } j : W_j \ge t\}|$</span>. By the symmetry of null <span>$W_j$</span>, the count of null features with <span>$W_j \le -t$</span> has the same distribution as <span>$V(t)$</span>. The total count <span>$|\{j : W_j \le -t\}|$</span> (including non-nulls, which are rare on the negative side) overestimates <span>$V(t)$</span>, so</p>

$$
\widehat{\mathrm{FDP}}(t) = \frac{1 + |\{j : W_j \le -t\}|}{|\{j : W_j \ge t\}|}
$$

<p>is a conservative estimate of the true false discovery proportion. Choosing the smallest <span>$t$</span> where this ratio is at most <span>$q$</span> controls FDR at level <span>$q$</span>.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

### Data Splitting and Multiple Splitting

**Data splitting** separates selection from inference by using independent data for each step, avoiding the post-selection inference problem where standard $p$-values are invalid after a data-driven model selection.

**Single split.** Randomly partition the $n$ observations into two halves $\mathcal{D}_1$ and $\mathcal{D}_2$.

1. On $\mathcal{D}_1$: run lasso (or any selector) to obtain a selected set $\hat S$.
2. On $\mathcal{D}_2$: fit OLS restricted to $\hat S$ and compute standard $p$-values for each selected coefficient.

Because $\mathcal{D}_2$ was not used for selection, the $p$-values from step 2 are valid conditionally on $\hat S$. Apply BH to those $p$-values to control FDR over the selected set.

**Limitation.** A single split is unstable: different splits yield different selected sets and different $p$-values.

**Multiple splitting** (Meinshausen, Meier, and Bühlmann, 2009) repeats the procedure $B$ times with independent random splits, producing $B$ sets of $p$-values $\{p_j^{(b)}\}$.

For each feature $j$, aggregate across splits:

$$
p_j^{\mathrm{agg}} = \min\!\left(1,\; q_\gamma\!\left\{\frac{p_j^{(b)}}{\gamma} : b = 1, \ldots, B\right\}\right),
$$

where $q_\gamma$ is the $\gamma$-quantile across splits (typically $\gamma = 0.05$). The division by $\gamma$ corrects for the quantile selection. Applying BH to $\{p_j^{\mathrm{agg}}\}$ controls FDR while averaging out the instability of any single split.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why multiple splitting controls FDR</span></summary>

<p><strong>Single split validity.</strong> On a single split, the second-half <span>$p$</span>-values are independent of the selection step, so under the null <span>$H_{0,j}$</span>, <span>$p_j^{(b)} \sim \mathrm{Uniform}(0,1)$</span> conditional on <span>$j \in \hat S^{(b)}$</span>. This makes BH valid on each individual split.</p>

<p><strong>Aggregation.</strong> Taking the <span>$\gamma$</span>-quantile across splits concentrates on the splits where feature <span>$j$</span> is most significant. The correction factor <span>$1/\gamma$</span> compensates for this selection across splits: by a standard quantile inequality, <span>$P(q_\gamma\{p_j^{(b)}/\gamma\} \le \alpha) \le \alpha$</span> under the null, so the aggregated values are still valid <span>$p$</span>-values. Applying BH to the aggregated values therefore controls FDR.</p>

<p><strong>Why it outperforms a single split.</strong> Averaging over many splits stabilizes the selected set and the <span>$p$</span>-values: features that appear important in most splits get low aggregated <span>$p$</span>-values, while features selected only in rare splits do not. This reduces variance without inflating the false discovery rate.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

## Feature Screening and Selection

Screening and selection are related but distinct steps. Screening is a fast first-pass elimination of obviously irrelevant variables. Selection is the final determination of which variables enter the model.

### Feature Screening

Screening methods are intentionally coarse. Their goal is to reduce dimension before heavier modeling:

* Marginal correlation screening: keep features with $|\mathrm{cor}(x_{j}, y)|$ above a threshold.
* Univariate $t$ tests: retain features with small univariate $p$-values.
* Variance filtering: drop features with near-zero variance across samples.
* Domain-based exclusion: remove features known to be irrelevant.

In high dimensions (many more predictors than observations), sure independence screening and related sure-screening results guarantee that truly important features survive under regularity conditions.

### Feature Selection

Selection methods aim to build the final predictor set:

* **Forward selection.** Start from the intercept-only model, add the most significant feature at each step.
* **Backward elimination.** Start from the full model, remove the least significant feature at each step.
* **Stepwise regression.** Combine forward and backward moves.
* **Best subset selection.** Exhaustive search over all $2^p$ subsets; feasible only for small $p$.
* **Lasso and elastic net.** Shrinkage-based selection embedded in the objective; the solution path traces selected features as $\lambda$ varies.
* **Stability selection.** Run lasso on bootstrap subsamples, keep features selected in most replicates.

**Information criteria.** Rather than testing, model complexity can be penalized directly via:

$$
\mathrm{AIC} = -2\ell(\hat\theta) + 2p,
\qquad
\mathrm{BIC} = -2\ell(\hat\theta) + p\log n,
$$

where $\ell(\hat\theta)$ is the maximized log-likelihood and $p$ is the number of parameters. BIC penalizes complexity more heavily and is consistent (selects the true model as $n \to \infty$ if it is in the candidate set); AIC targets predictive accuracy and tends to select slightly larger models. For the Gaussian linear model, AIC and BIC reduce to penalized versions of RSS.

### Practical Warnings

* Post-selection $p$-values are too optimistic: the data was used to choose the model, so standard inference no longer applies.
* Stepwise procedures can overfit badly in moderate dimensions.
* Correlated features make selected sets unstable: small data perturbations can swap one correlated predictor for another.
* Predictive performance must be assessed out of sample, not on the training data used for selection.

### Practical Workflow

1. Standardize features when penalty-based methods will be used, so all predictors are on the same scale.
2. Screen when the dimension is large or many variables are clearly irrelevant, to reduce computation and noise.
3. Apply cross-validated lasso or elastic net for simultaneous selection and shrinkage.
4. Refit the selected model by OLS on the training fold (or full data), then examine residuals and VIFs.
5. Separate the prediction goal from the interpretation goal: a good predictive model need not have interpretable individual coefficients.

## Stability

A model is **stable** if small perturbations of the data (resampling observations, adding noise, or slightly changing the predictor set) yield similar outputs: similar selected features, similar coefficient estimates, similar predictions. Stability is a prerequisite for reliable inference and generalization. An unstable model may fit the training data well while producing wildly different answers on nearby datasets.

### Sources of Instability

* **Multicollinearity.** When two predictors are highly correlated, their OLS coefficients have large variance and opposite signs can appear depending on the sample. The selected set under lasso can swap between the two across resamples.
* **Near-singularity.** Small eigenvalues of $X^TX$ amplify noise into large coefficient variance, as shown in the ridge/PCR connection.
* **High dimension relative to $n$.** When $p$ is close to $n$, OLS overfits and small perturbations change fitted values substantially.
* **Stepwise and greedy procedures.** Sequential selection methods can take very different paths on slightly different datasets, especially when multiple predictors are competitive.

### Bootstrap Stability Diagnostics

Run the selection procedure on $B$ bootstrap resamples of the data. For each feature $j$, compute the **selection frequency**

$$
\hat\pi_j = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}[j \in \hat S^{(b)}].
$$

Features with $\hat\pi_j$ close to 1 are stably selected; features with $\hat\pi_j$ near 0.5 are on the selection boundary and should not be trusted individually. Similarly, track the bootstrap distribution of $\hat\beta_j^{(b)}$ to assess coefficient stability: wide bootstrap intervals under collinearity confirm that the point estimate is unreliable even when it appears significant.

### Stability Selection

Stability selection (Meinshausen and Bühlmann, 2010) formalizes bootstrap stability into a procedure with error control. Run lasso on $B$ subsamples of size $n/2$ (without replacement) across a grid of $\lambda$ values. The stable selected set is

$$
\hat S^{\mathrm{stable}} = \left\{j : \sup_\lambda \hat\pi_j(\lambda) \ge \pi_{\mathrm{thr}}\right\},
$$

where $\hat\pi_j(\lambda)$ is the selection frequency at regularization level $\lambda$ and $\pi_{\mathrm{thr}} \in (0.5, 1)$ is a threshold (typically 0.6–0.9). The expected number of falsely selected variables satisfies

$$
E[|\hat S^{\mathrm{stable}} \cap H_0|] \le \frac{q^2}{(2\pi_{\mathrm{thr}} - 1)\,|\hat S(\lambda)_{\max}|},
$$

where $q$ is the average model size and $H_0$ is the set of null features. This bound holds without assumptions on the design matrix or the noise distribution.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why subsampling of size $n/2$ and why the error bound holds</span></summary>

<p><strong>Why $n/2$.</strong> Using half-samples ensures two things: (1) each subsample is large enough to run lasso reliably, and (2) the selection events on two independent half-samples from the same draw are approximately independent. This independence is the key structural assumption in the error bound derivation.</p>

<hr>

<p><strong>Error bound sketch.</strong> Let <span>$\hat\pi_j$</span> be the selection frequency for a null feature <span>$j$</span>. On any single subsample, <span>$P(j \in \hat S) \le q / p$</span> (a feature enters roughly in proportion to its share of the average model size). Since subsamples are approximately independent, <span>$\hat\pi_j \approx \mathrm{Binomial}(B, q/p) / B$</span>. For <span>$\hat\pi_j \ge \pi_{\mathrm{thr}}$</span> with <span>$\pi_{\mathrm{thr}} > 1/2$</span>, the probability decays geometrically in <span>$\pi_{\mathrm{thr}}$</span>, and summing over null features gives the stated bound on expected false selections.</p>

<hr>

<p><strong>Comparison to data splitting.</strong> Data splitting uses one held-out half for inference after selection on the other half. Stability selection uses many subsamples and summarizes by selection frequency, never requiring a held-out set for inference. Both exploit the independence between the two halves; stability selection averages out the variance of a single split at the cost of not producing individual p-values.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

### Prediction Stability

Stability of the selected set does not guarantee stability of predictions. For a new point $x_*$, the bootstrap variance of the predicted value

$$
\mathrm{Var}_{\mathrm{boot}}(\hat y_*) = \frac{1}{B}\sum_{b=1}^B \left(\hat y_*^{(b)} - \bar{\hat y}_*\right)^2
$$

measures how much the fitted model varies across resamples at $x_*$. High prediction variance at a test point signals that the model is extrapolating into a region where the data provide little constraint, independent of whether the selected features are stable.

## Practical Checklist

1. Write down the model equation and identify the scientific question before fitting anything.
2. Check rank of $X$, whether an intercept is included, and whether $p$ accounts for all columns.
3. Understand the geometry: projection, residuals, and orthogonality are the foundation of everything else.
4. Use ANOVA and nested-model $F$ tests to compare models rather than scanning coefficient tables.
5. Check standard errors and VIF values before interpreting any individual coefficient.
6. Under multicollinearity, coefficients can be individually unreliable even when the joint fit is good.
7. Use ridge or lasso when instability, dimension, or collinearity is a real concern.
8. Apply multiple testing corrections whenever more than one hypothesis is evaluated, and validate predictive claims out of sample.
