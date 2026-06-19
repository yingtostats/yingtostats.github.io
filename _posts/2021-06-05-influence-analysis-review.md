---
layout: post
title: "Influence Analysis Through the Value of Information Framework"
date: 2021-06-05 10:00:00
tag:
- Statistics
- Regression
- Decision Theory
projects: true
blog: true
published: true
author: YingZhang
description: A unified view of influence analysis using the Value of Information framework, connecting classical diagnostics, variance-based sensitivity analysis, and high-dimensional influence measures.
fontsize: 23pt
---

{% include mathjax_support.html %}

Influence analysis asks: how much does a specific piece of information affect our model or decision? The "piece of information" can be a training data point, an input feature, a model parameter, or new data. Different fields have developed different approaches to answer this question, each from a distinct intellectual tradition:

- **Gradient-based (Influence Functions)**: from robust statistics and optimization. Uses gradients and Hessians of the loss to measure how the model changes under small data perturbations. Originally developed for classical statistics (Cook's distance, leverage), extended to deep learning by Koh and Liang (2017). Also the basis of feature attribution methods like saliency maps and integrated gradients.
- **Game-theoretic (Shapley Values)**: from cooperative game theory. Measures each player's fair share of the total payoff by averaging marginal contributions across all possible coalitions. Applied to data points (Data Shapley) and to features (SHAP) as separate instantiations of the same Shapley value formula.
- **Decision-theoretic (Value of Information)**: from Bayesian decision theory. Measures the expected reduction in loss from learning a piece of information. Connects to classical test statistics (F-test), observation diagnostics (Cook's distance), and variance-based sensitivity analysis (Sobol indices) as special cases.

All three frameworks can evaluate influence from both the **feature** and **data** perspectives. This post develops the Value of Information framework in detail, then connects it to influence functions and Data Shapley, showing how classical and modern methods relate under a common lens.

The central object is the **Expected Value of Information Ratio (EVOIR)**, which measures the realized influence of a piece of information relative to its expected influence. Under squared loss, EVOIR reduces to familiar quantities: the F-statistic for parameter groups, Cook's distance for observations, and variance-based sensitivity indices for features.

## The Value of Information Framework

Let $\theta$ denote the unknown parameters, $Y$ the observed data, and $\phi$ a piece of information we want to evaluate. Here $\phi$ can be a subset of $\theta$, a function of $\theta$, or a random variable conditional on $\theta$ (such as new data $Y_{\text{new}}$).

Given a loss function $L(a, \theta)$ and an action $a$, define:

- $a_{Y}$: the optimal action given data $Y$ alone.
- $a_{Y,\phi}$: the optimal action given both $Y$ and $\phi$.

### Retrospective Value of Information

The **retrospective value of information** (RVOI) measures the observed reduction in expected loss from learning $\phi$, after $\phi$ is revealed:

$$
\text{RVOI} = E\{L(a_Y, \theta) - L(a_{Y,\phi}, \theta) \mid \phi, Y\}.
$$

RVOI answers: "Now that we know $\phi$, how much did it actually help?" It is always non-negative because $a_{Y,\phi}$ optimizes the loss with strictly more information than $a_{Y}$.

### Prospective Value of Information

The **prospective value of information** (PVOI) is the expected RVOI before $\phi$ is observed:

$$
\text{PVOI} = E[\text{RVOI} \mid Y] = E\{L(a_Y, \theta) - L(a_{Y,\phi}, \theta) \mid Y\}.
$$

PVOI answers: "How much do we expect $\phi$ to help, on average before we observed it?" In health economics, PVOI is called the Value of Partial Perfect Information (for parameters) or the Value of Sample Information (for new data). It measures the expected influence of new information.

### Expected Value of Information Ratio

The **EVOIR** is the ratio of realized to expected influence:

$$
\text{EVOIR}(\phi \mid Y) = \frac{\text{RVOI}}{\text{PVOI}} = \frac{E\{L(a_Y, \theta) - L(a_{Y,\phi}, \theta) \mid \phi, Y\}}{E\{L(a_Y, \theta) - L(a_{Y,\phi}, \theta) \mid Y\}}.
$$

The expected value of EVOIR is 1 (by the law of total expectation). Values much larger than 1 indicate that $\phi$ was surprisingly influential; values near 0 indicate it had little effect. This ratio normalizes influence to a common scale, making it comparable across different types of information.

## EVOIR Under Squared Loss

Assume

$$
Y = f(X) + \epsilon, \quad E[\epsilon] = 0,
$$

where $f(X)$ is parameterized by $\theta$. Under squared loss

$$
L(\hat{f}, f) = (\hat{f}(X) - f(X))^\top (\hat{f}(X) - f(X)),
$$

the Bayes estimator is the posterior mean:

$$
\hat{f}_Y(X) = E[f(X) \mid Y], \quad \hat{f}_{Y,\phi}(X) = E[f(X) \mid Y, \phi].
$$

By the law of total expectation, $\hat{f}_{Y}(X) = E[\hat{f}_{Y,\phi}(X) \mid Y]$.

Under squared loss, RVOI, PVOI, and EVOIR simplify to clean geometric quantities:

$$
\text{RVOI} = (\hat{f}_Y(X) - \hat{f}_{Y,\phi}(X))^\top (\hat{f}_Y(X) - \hat{f}_{Y,\phi}(X)),
$$

$$
\text{PVOI} = \text{tr}(\text{Var}(\hat{f}_{Y,\phi}(X) \mid Y)),
$$

$$
\text{EVOIR} = \frac{(\hat{f}_Y(X) - \hat{f}_{Y,\phi}(X))^\top (\hat{f}_Y(X) - \hat{f}_{Y,\phi}(X))}{\text{tr}(\text{Var}(\hat{f}_{Y,\phi}(X) \mid Y))}.
$$

RVOI is the squared distance between the original and updated predictions. PVOI is the total posterior variance of the updated predictions. EVOIR is the ratio of realized squared change to expected squared change.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Derivation of RVOI, PVOI, and EVOIR under squared loss</span></summary>

<p><strong>RVOI derivation.</strong> Starting from the definition:</p>

$$\text{RVOI} = E\{L(\hat{f}_Y, f) - L(\hat{f}_{Y,\phi}, f) \mid Y, \phi\}.$$

<p>Expanding the squared loss terms:</p>

$$= E\{(\hat{f}_Y - f)^\top(\hat{f}_Y - f) - (\hat{f}_{Y,\phi} - f)^\top(\hat{f}_{Y,\phi} - f) \mid Y, \phi\}.$$

<p>Expanding and using the fact that $E[f(X) \mid Y, \phi] = \hat{f}_{Y,\phi}(X)$:</p>

$$= \hat{f}_Y^\top \hat{f}_Y - 2\hat{f}_Y^\top \hat{f}_{Y,\phi} - \hat{f}_{Y,\phi}^\top \hat{f}_{Y,\phi} + 2\hat{f}_{Y,\phi}^\top \hat{f}_{Y,\phi}$$

$$= \hat{f}_Y^\top \hat{f}_Y - 2\hat{f}_Y^\top \hat{f}_{Y,\phi} + \hat{f}_{Y,\phi}^\top \hat{f}_{Y,\phi}$$

$$= (\hat{f}_Y - \hat{f}_{Y,\phi})^\top (\hat{f}_Y - \hat{f}_{Y,\phi}).$$

<p>RVOI is the squared Euclidean distance between the original and updated Bayes estimates.</p>

<p><strong>PVOI derivation.</strong> Taking the expectation of RVOI over $\phi \mid Y$:</p>

$$\text{PVOI} = E\{(\hat{f}_Y - \hat{f}_{Y,\phi})^\top (\hat{f}_Y - \hat{f}_{Y,\phi}) \mid Y\}.$$

<p>Since $\hat{f}_{Y} = E[\hat{f}_{Y,\phi} \mid Y]$, this is the expected squared deviation of $\hat{f}_{Y,\phi}$ from its mean, which is the trace of its covariance:</p>

$$= \text{tr}(E\{(\hat{f}_{Y,\phi} - \hat{f}_Y)(\hat{f}_{Y,\phi} - \hat{f}_Y)^\top \mid Y\}) = \text{tr}(\text{Var}(\hat{f}_{Y,\phi}(X) \mid Y)).$$

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Linear Regression Example

In linear regression, $f(X) = X\beta$ with the Bayesian model:

$$
Y \mid \beta, \sigma^2 \sim N(X\beta, \sigma^2 I_n), \quad \pi(\beta, \sigma^2) \propto \sigma^{-2}.
$$

The posterior is:

$$
\beta \mid \sigma^2, Y \sim N(\hat{\beta}_Y, \sigma^2 (X^\top X)^{-1}), \quad \sigma^2 \mid Y \sim \chi^{-2}(n-p, S_n^2),
$$

where $\hat{\beta}_{Y} = (X^\top X)^{-1} X^\top Y$ and $S_{n}^2 = (Y - X\hat{\beta}_{Y})^\top (Y - X\hat{\beta}_{Y}) / (n-p)$.

### Case A: Learning a Parameter Group

Let $\phi = \beta_{g}$, a subvector of $\beta$ with $p_{2}$ components. Partition:

$$
\beta \mid \sigma^2, Y \sim N\left(\begin{pmatrix} \hat{\beta}_{Y,-g} \\ \hat{\beta}_{Y,g} \end{pmatrix}, \sigma^2 \begin{pmatrix} A & B \\ B^\top & C \end{pmatrix}\right).
$$

The EVOIR for learning $\beta_{g}$ is:

$$
\text{EVOIR} = \frac{n-p-2}{n-p} \cdot F,
$$

where $F$ is the classical F-statistic for testing $\beta_{g}$:

$$
F = \frac{(\hat{\beta}_{Y,g} - \beta_g)^\top C^{-1} (\hat{\beta}_{Y,g} - \beta_g) / p_2}{S_n^2} \sim F_{p_2, n-p}.
$$

The F-statistic, one of the most widely used quantities in regression, is (up to a known scale factor) the EVOIR for a parameter group under squared loss. Large F means $\beta_{g}$ was more influential than expected; small F means learning $\beta_{g}$ changed the predictions less than expected.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Derivation: EVOIR equals scaled F-statistic</span></summary>

<p>The conditional distribution of $\beta_{-g}$ given $\beta_{g}$ is:</p>

$$\beta_{-g} \mid Y, \beta_g, \sigma^2 \sim N(\hat{\beta}_{Y,-g} + BC^{-1}(\beta_g - \hat{\beta}_{Y,g}),\; \sigma^2(A - BC^{-1}B^\top)).$$

<p>The Bayes estimates are $\hat{\beta}_{Y}$ (given $Y$ alone) and</p>

$$\hat{\beta}_{Y,\beta_g} = \begin{pmatrix} \hat{\beta}_{Y,-g} + BC^{-1}(\beta_g - \hat{\beta}_{Y,g}) \\ \beta_g \end{pmatrix}$$

<p>(given $Y$ and $\beta_{g}$). Computing RVOI:</p>

$$\text{RVOI} = (X\hat{\beta}_Y - X\hat{\beta}_{Y,\beta_g})^\top (X\hat{\beta}_Y - X\hat{\beta}_{Y,\beta_g}).$$

<p>After simplification using the partitioned structure of $X^\top X$:</p>

$$\text{RVOI} = (\hat{\beta}_{Y,g} - \beta_g)^\top C^{-1} (\hat{\beta}_{Y,g} - \beta_g).$$

<p>For PVOI, we take the expectation over $\beta_{g} \mid Y$:</p>

$$\text{PVOI} = E\{(\hat{\beta}_{Y,g} - \beta_g)^\top C^{-1} (\hat{\beta}_{Y,g} - \beta_g) \mid Y\}.$$

<p>Since $\beta_{g} \mid \sigma^2, Y \sim N(\hat{\beta}_{Y,g}, \sigma^2 C)$, the quadratic form $(\hat{\beta}_{Y,g} - \beta_{g})^\top C^{-1} (\hat{\beta}_{Y,g} - \beta_{g}) / \sigma^2 \sim \chi^2_{p_{2}}$ conditional on $\sigma^2$. Taking the double expectation:</p>

$$\text{PVOI} = E\{E\{(\hat{\beta}_{Y,g} - \beta_g)^\top C^{-1} (\hat{\beta}_{Y,g} - \beta_g) \mid Y, \sigma^2\} \mid Y\} = E\{p_2 \sigma^2 \mid Y\} = p_2 \frac{n-p}{n-p-2} S_n^2.$$

<p>Therefore:</p>

$$\text{EVOIR} = \frac{(\hat{\beta}_{Y,g} - \beta_g)^\top C^{-1} (\hat{\beta}_{Y,g} - \beta_g)}{p_2 \frac{n-p}{n-p-2} S_n^2} = \frac{n-p-2}{n-p} \cdot \frac{(\hat{\beta}_{Y,g} - \beta_g)^\top C^{-1} (\hat{\beta}_{Y,g} - \beta_g) / p_2}{S_n^2} = \frac{n-p-2}{n-p} \cdot F.$$

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Case B: Learning New Data

Let $\phi = Y_{\text{new}}$, a vector of $m$ new observations. Let $X$ denote the combined design matrix for both the original $n$ and new $m$ samples (an $(n+m) \times p$ matrix).

$$
\text{RVOI} = \sum_{i=1}^{n+m} (x_i^\top \hat{\beta}_Y - x_i^\top \hat{\beta}_{Y, Y_{\text{new}}})^2,
$$

$$
\text{PVOI} = \text{tr}(\text{Var}(X\hat{\beta}_{Y,Y_{\text{new}}} \mid Y)),
$$

$$
\text{EVOIR} = \frac{\sum_{i=1}^{n+m} (x_i^\top \hat{\beta}_Y - x_i^\top \hat{\beta}_{Y,Y_{\text{new}}})^2}{\text{tr}(\text{Var}(X\hat{\beta}_{Y,Y_{\text{new}}} \mid Y))}.
$$

This measures how much new data changes the predictions across all observation points, relative to the expected change. It connects to the Value of Sample Information in health economics.

### Case C: Influence of a Single Observation (Cook's Distance)

Let $\phi = (x_{k}, y_{k})$, the $k$-th observation. The RVOI for removing observation $k$ measures how much the fitted values change:

$$
\text{RVOI}_k = (X\hat{\beta}_Y - X\hat{\beta}_{Y}^{(-k)})^\top (X\hat{\beta}_Y - X\hat{\beta}_{Y}^{(-k)}),
$$

where $\hat{\beta}_{Y}^{(-k)}$ is the OLS estimate with observation $k$ deleted. Cook's distance is this quantity scaled by the variance:

$$
D_k = \frac{(\hat{\beta}_Y - \hat{\beta}_Y^{(-k)})^\top (X^\top X) (\hat{\beta}_Y - \hat{\beta}_Y^{(-k)})}{p \, S_n^2} = \frac{(X\hat{\beta}_Y - X\hat{\beta}_Y^{(-k)})^\top (X\hat{\beta}_Y - X\hat{\beta}_Y^{(-k)})}{p \, S_n^2}.
$$

So Cook's distance is a scaled RVOI:

$$
D_k = \frac{\text{RVOI}_k}{p \, S_n^2}.
$$

The numerator is the realized squared change in predictions (RVOI). The denominator $p \, S_{n}^2$ is proportional to the expected prediction variance, which plays the same normalizing role as PVOI. Large $D_{k}$ means observation $k$ had an outsized influence on the fitted values.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Cook's distance closed form and connection to leverage and residuals</span></summary>

<p>Using the Sherman-Morrison-Woodbury formula for rank-1 updates, the leave-one-out estimate has a closed form:</p>

$$\hat{\beta}_Y^{(-k)} = \hat{\beta}_Y - \frac{(X^\top X)^{-1} x_k e_k}{1 - h_{kk}},$$

<p>where $e_{k} = y_{k} - x_{k}^\top \hat{\beta}_{Y}$ is the ordinary residual and $h_{kk} = x_{k}^\top (X^\top X)^{-1} x_{k}$ is the leverage (the $k$-th diagonal element of the hat matrix $H = X(X^\top X)^{-1}X^\top$). Substituting into Cook's distance:</p>

$$D_k = \frac{e_k^2}{p \, S_n^2} \cdot \frac{h_{kk}}{(1 - h_{kk})^2}.$$

<p>This reveals that Cook's distance is the product of two factors:</p>

<ul>
<li><strong>Residual magnitude:</strong> $e_{k}^2 / (p \, S_{n}^2)$ measures how poorly the model fits observation $k$. Large residuals indicate the point is an outlier in the $y$-direction.</li>
<li><strong>Leverage:</strong> $h_{kk} / (1 - h_{kk})^2$ measures how extreme observation $k$'s position is in the feature space. Points far from the center of the data have high leverage and exert more pull on the fitted line.</li>
</ul>

<p>A point needs both a large residual <em>and</em> high leverage to be truly influential. An outlier in $y$ with low leverage barely moves the fit. A high-leverage point that falls on the regression line has a small residual and low Cook's distance despite its extreme position.</p>

<p><strong>Rule of thumb.</strong> $D_{k} > 4/n$ or $D_{k} > 1$ are common thresholds for flagging influential observations, though these are guidelines rather than formal tests.</p>

<p><strong>Connection to EVOIR.</strong> Cook's distance normalizes the RVOI by $p \, S_{n}^2$ (a variance estimate). Compare with the F-statistic case where EVOIR = $\frac{n-p-2}{n-p} \cdot F$: both are ratios of a realized squared change to a variance scale. Cook's distance is the observation-level analogue of the F-statistic: F measures parameter-group influence, $D_{k}$ measures observation influence, and both are scaled RVOIs under the VoI framework.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">How to choose the Cook's distance threshold in practice</span></summary>

<p>Two common rules of thumb exist: $D_{k} > 1$ (conservative, flags only extreme cases) and $D_{k} > 4/n$ (liberal, flags roughly the most influential observations in moderate-sized samples). Neither is a formal significance test. Which to use depends on your goal.</p>

<p><strong>Start with $4/n$, then inspect.</strong> The $4/n$ rule corresponds roughly to the 50th percentile of the $F_{p, n-p}$ distribution, so it flags points that shift the fit by more than the "median expected shift." Use it as a screening step, not a deletion criterion. Plot $D_{k}$ against observation index and look for points that separate from the bulk, rather than relying on a fixed cutoff.</p>

<p><strong>Leverage deserves its own check.</strong> Cook's distance conflates leverage ($h_{kk}$) and residual size. A point can have moderate $D_{k}$ because of high leverage with a small residual (it is pulling the line toward itself). Check leverage separately: the standard threshold is $h_{kk} > 2p/n$. In R, <code>hatvalues(model)</code> returns leverages; in Python, <code>OLSResults.get_influence().hat_matrix_diag</code> does the same. Half-normal plots of leverages (R: <code>faraway::halfnorm(hatvalues(model))</code>) are more informative than a fixed cutoff because they reveal the distribution's shape and make outliers visually obvious.</p>

<p><strong>Software.</strong> In R: <code>cooks.distance(model)</code> or <code>plot(model, which=4)</code> for the index plot. In Python: <code>OLSResults.get_influence().cooks_distance</code> returns both $D_{k}$ values and p-values based on the $F_{p, n-p}$ distribution. Always examine flagged points substantively (measurement error, different subpopulation, data entry mistake) before deciding whether to remove, downweight, or keep them.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Properties of PVOI

The prospective value of information satisfies four fundamental properties under two assumptions: (A1) Bayes estimators are used as decision rules, and (A2) the Bayes estimator is unique under the given loss function.

**Non-negativity.** For any information $I$, $\text{PVOI}(I \mid Y) \geq 0$.

**Uniqueness.** Under A2, $\text{PVOI}(I \mid Y) = 0$ if and only if $I$ is empty (contains no information).

**Monotonicity.** If $I_{2} \subset I_{1}$, then $\text{PVOI}(I_{2} \mid Y) \leq \text{PVOI}(I_{1} \mid Y)$. More information is always at least as valuable.

**Additivity.** If $I_{2} \subset I_{1}$ and $I_{1-2} = I_{1} \setminus I_{2}$, then

$$
\text{PVOI}(I_1 \mid Y) = \text{PVOI}(I_2 \mid Y) + E_{I_2 \mid Y}\, \text{PVOI}(I_{1-2} \mid I_2, Y).
$$

The value of the full information set equals the value of a subset plus the expected conditional value of the remainder. This decomposition is symmetric: you can also write it as

$$
\text{PVOI}(I_1 \mid Y) = \text{PVOI}(I_{1-2} \mid Y) + E_{I_{1-2} \mid Y}\, \text{PVOI}(I_2 \mid I_{1-2}, Y).
$$

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Proofs of PVOI properties</span></summary>

<p><strong>Non-negativity.</strong> Since $d_{Y,I}$ is a Bayes estimator, by definition $E_{\theta\lvert Y,I} L(d_{Y,I}, \theta) \leq E_{\theta\rvertY,I} L(d_{Y}, \theta)$ for all $I$. Hence</p>

$$\text{PVOI}(I \mid Y) = E_{I|Y}\, E_{\theta|Y,I}\{L(d_Y, \theta) - L(d_{Y,I}, \theta)\} \geq 0.$$

<p><strong>Uniqueness.</strong> If $\text{PVOI}(I \mid Y) = 0$, then by non-negativity, $E_{\theta\lvert Y,I}\{L(d_{Y}, \theta) - L(d_{Y,I}, \theta)\} = 0$ for almost every $I \mid Y$. By uniqueness of the Bayes estimator (A2), $d_{Y,I} = d_{Y}$ for almost every $I \mid Y$, which means $I$ provides no information.</p>

<p><strong>Monotonicity.</strong> For $I_{2} \subset I_{1}$ with $I_{1-2} = I_{1} \setminus I_{2}$:</p>

$$\text{PVOI}(I_1|Y) - \text{PVOI}(I_2|Y) = E_{\theta,I_1|Y}\{L(d_{Y,I_2}, \theta) - L(d_{Y,I_1}, \theta)\}.$$

<p>Since $I_{2} \subset I_{1}$, we can write $d_{Y,I_{1}} = d_{Y,I_{2},I_{1-2}}$ and condition on $I_{2}$:</p>

$$= E_{I_2|Y}\, E_{\theta,I_{1-2}|Y,I_2}\{L(d_{Y,I_2}, \theta) - L(d_{Y,I_2,I_{1-2}}, \theta)\} = E_{I_2|Y}\, \text{PVOI}(I_{1-2} \mid Y, I_2) \geq 0.$$

<p><strong>Additivity.</strong> The monotonicity proof directly gives the additivity result:</p>

$$\text{PVOI}(I_1|Y) = \text{PVOI}(I_2|Y) + E_{I_2|Y}\, \text{PVOI}(I_{1-2} \mid I_2, Y).$$

<p>By symmetry (swapping the roles of $I_{2}$ and $I_{1-2}$), the same argument gives:</p>

$$\text{PVOI}(I_1|Y) = \text{PVOI}(I_{1-2}|Y) + E_{I_{1-2}|Y}\, \text{PVOI}(I_2 \mid I_{1-2}, Y).$$

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### RVOI Additivity

There is a corresponding additivity result for retrospective value. If $I_{2} \subset I_{1}$ and $I_{1-2} = I_{1} \setminus I_{2}$:

$$
E_{I_{1-2}|Y,I_2}\, \text{RVOI}(I_1 \mid Y, I_1) = \text{RVOI}(I_2 \mid Y, I_2) + \text{PVOI}(I_{1-2} \mid Y, I_2).
$$

The expected retrospective value of the full set, given a revealed subset, equals the retrospective value of that subset plus the prospective value of the remainder. This bridges the retrospective and prospective perspectives.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Proof of RVOI additivity</span></summary>

<p>Starting from</p>

$$E_{I_{1-2}|Y,I_2}\, \text{RVOI}(I_1|Y,I_1) - \text{PVOI}(I_{1-2}|Y,I_2),$$

<p>expanding both terms:</p>

$$= E_{\theta,I_{1-2}|Y,I_2}\{L(d_Y,\theta) - L(d_{Y,I_1},\theta)\} - E_{\theta,I_{1-2}|Y,I_2}\{L(d_{Y,I_2},\theta) - L(d_{Y,I_2,I_{1-2}},\theta)\}.$$

<p>Since $d_{Y,I_{1}} = d_{Y,I_{2},I_{1-2}}$, the terms involving $d_{Y,I_{1}}$ cancel:</p>

$$= E_{\theta,I_{1-2}|Y,I_2}\{L(d_Y,\theta) - L(d_{Y,I_2},\theta)\} = E_{\theta|Y,I_2}\{L(d_Y,\theta) - L(d_{Y,I_2},\theta)\} = \text{RVOI}(I_2|Y,I_2).$$

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Connection 1: Variance-Based Sensitivity Analysis

Variance-based sensitivity analysis (Sobol indices) asks: how much of the output variance is attributable to each input variable? The VoI framework provides a decision-theoretic interpretation of these indices.

### Sobol Indices

The first-order variance (main effect) of input $X_{i}$ is

$$
V_i = \text{Var}_{X_i}[E_{X_{\sim i}}(y \mid X_i)],
$$

and the total-effect variance is

$$
V_{T_i} = E_{X_{\sim i}}\, \text{Var}_{X_i}[y \mid X_{\sim i}].
$$

Their normalized versions are the first-order and total-effect sensitivity indices:

$$
S_i = \frac{V_i}{\text{Var}[y]}, \quad S_{T_i} = \frac{V_{T_i}}{\text{Var}[y]}.
$$

$S_{i}$ measures the main effect of $X_{i}$ alone. $S_{T_{i}}$ measures the total contribution of $X_{i}$ including all interactions with other variables.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Foundations: law of total variance and ANOVA-like decomposition</span></summary>

<p>Variance-based sensitivity analysis rests on two results. The first is the <strong>law of total variance</strong>:</p>

$$\text{Var}[y] = \text{Var}[E[y \mid X_i]] + E[\text{Var}[y \mid X_i]].$$

<p>The first term is $V_{i}$ (variance explained by $X_{i}$); the second is the residual variance.</p>

<p>The second is the <strong>high-dimensional model representation (HDMR)</strong>: any function $y = \eta(x_{1}, \ldots, x_{n})$ can be decomposed as</p>

$$y = E[y] + \sum_i z_i(x_i) + \sum_{i < j} z_{ij}(x_i, x_j) + \cdots,$$

<p>where</p>

$$z_i(x_i) = E[y \mid x_i] - E[y], \quad z_{ij}(x_i, x_j) = E[y \mid x_i, x_j] - E[y \mid x_i] - E[y \mid x_j] + E[y].$$

<p>This gives $2^n - 1$ terms, which is why the total-effect index $S_{T_{i}}$ is useful: it summarizes all interactions involving $X_{i}$ without enumerating them.</p>

<p><strong>Special case.</strong> When $y = a + bx_{1} + cx_{2}$ and $x_{1}, x_{2}$ are independent and centered:</p>

$$S_i = \frac{b^2 \text{Var}[x_1]}{\text{Var}[y]} = \rho_{y,x_1}^2 = S_{T_i}.$$

<p>The first-order and total-effect indices coincide because there are no interactions in a linear model with independent inputs.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### VoI Interpretation of Sensitivity Indices

Under squared loss with decision $\hat{d}$, the Sobol indices have direct VoI interpretations:

**First-order index as PVOI:**

$$
\text{PVOI}(X_i \mid \text{Prior}) = E_{X_i}(d_{\text{prior}} - d_{X_i})^2 = \text{Var}_{X_i}[E_{X_{\sim i}}[d_X \mid X_i]] \approx V_i.
$$

**Total-effect index as conditional PVOI:**

$$
E_{X_{\sim i}}\, \text{PVOI}(X_i \mid X_{\sim i}) = E_{X_{\sim i}}[\text{Var}_{X_i}[d_X \mid X_{\sim i}]] \approx V_{T_i}.
$$

The first-order Sobol index measures how much learning $X_{i}$ alone reduces expected loss. The total-effect index measures the average reduction from learning $X_{i}$ after all other variables are already known. The VoI framework generalizes these beyond squared loss to any loss function, which is useful when the parameter of interest is highly skewed or multimodal.

## Connection 2: Estimation Stability With Cross-Validation (ESCV)

The ESCV criterion for choosing a regularization parameter $\lambda$ is

$$
\text{ESCV}(\lambda) = \frac{\hat{\text{var}}(\hat{Y}_{[\lambda]})}{||\bar{\hat{Y}}_{[\lambda]}||_2^2} = \frac{\frac{1}{V}\sum_{k=1}^{V} ||\hat{Y}_{[-k,\lambda]} - \bar{\hat{Y}}_{[\lambda]}||_2^2}{||\bar{\hat{Y}}_{[\lambda]}||_2^2},
$$

where $\hat{Y}_{[-k,\lambda]}$ is the prediction from the model trained without fold $k$, and $\bar{\hat{Y}}_{[\lambda]} = \frac{1}{V}\sum_{k=1}^{V} \hat{Y}_{[-k,\lambda]}$.

In the VoI framework, each fold removal is an information perturbation. The numerator is an averaged RVOI:

$$
||\hat{Y}_{[-k,\lambda]} - \bar{\hat{Y}}_{[\lambda]}||_2^2 \approx \text{RVOI}((\mathbf{X}, \mathbf{y})_{[k,\lambda]} \mid (\mathbf{X}, \mathbf{y})_{[\lambda]}),
$$

and the variance estimate connects RVOI to PVOI:

$$
\hat{\text{var}}(\hat{Y}_{[\lambda]}) \approx E_{(\mathbf{X},\mathbf{y})_{[-k,\lambda]}}\, \text{PVOI}((\mathbf{X},\mathbf{y})_{[k,\lambda]} \mid (\mathbf{X},\mathbf{y})_{[-k,\lambda]}).
$$

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Derivation: ESCV variance as expected PVOI</span></summary>

<p>Starting from the variance estimate:</p>

$$\hat{\text{var}}(\hat{Y}_{[\lambda]}) = \frac{1}{V}\sum_{k=1}^{V} ||\hat{Y}_{[-k,\lambda]} - \bar{\hat{Y}}_{[\lambda]}||_2^2 \approx E_{(\mathbf{X},\mathbf{y})_{[\lambda]}}\, \text{RVOI}((\mathbf{X},\mathbf{y})_{[k,\lambda]} \mid (\mathbf{X},\mathbf{y})_{[\lambda]}).$$

<p>By the law of total expectation, conditioning on the leave-one-fold-out data:</p>

$$E_{(\mathbf{X},\mathbf{y})_{[\lambda]}}\, \text{RVOI} = E_{(\mathbf{X},\mathbf{y})_{[-k,\lambda]}}\, E_{(\mathbf{X},\mathbf{y})_{[k,\lambda]} | (\mathbf{X},\mathbf{y})_{[-k,\lambda]}}\, \text{RVOI} = E_{(\mathbf{X},\mathbf{y})_{[-k,\lambda]}}\, \text{PVOI}((\mathbf{X},\mathbf{y})_{[k,\lambda]} \mid (\mathbf{X},\mathbf{y})_{[-k,\lambda]}).$$

<p>The ESCV variance is the expected prospective value of each fold's data, averaged across leave-one-out configurations. A large ESCV indicates that the model's predictions are sensitive to which data points are included, and the regularization parameter $\lambda$ should be chosen to reduce this sensitivity.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

ESCV selects $\lambda$ by minimizing prediction instability. Viewed through VoI, it chooses the regularization that minimizes the expected influence of any single fold on the predictions. Stable models have low PVOI for each fold; unstable models have high PVOI.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">How to choose the regularization parameter with ESCV in practice</span></summary>

<p><strong>When to prefer ESCV over standard CV.</strong> Standard cross-validation (minimizing prediction error) can select models that are accurate on average but unstable: small changes in the training set produce large swings in predictions. ESCV penalizes this instability. Use ESCV when prediction stability matters more than raw predictive accuracy, for example in scientific inference (where you want robust coefficient estimates) or in high-dimensional settings where the CV error curve is flat and multiple $\lambda$ values give similar accuracy.</p>

<p><strong>Number of folds.</strong> ESCV uses the same fold structure as standard $V$-fold CV. $V = 5$ or $V = 10$ are typical defaults. Larger $V$ gives a less biased variance estimate but increases computation. Leave-one-out ($V = n$) is the most stable estimator of ESCV but costs $n$ model fits.</p>

<p><strong>Key difference from standard CV.</strong> In standard CV, each model predicts only its held-out fold, producing a single $n$-vector of out-of-sample predictions. In ESCV, each model predicts <em>all $n$ points</em>. With $V = 5$ folds, you get 5 full $n$-dimensional prediction vectors $\hat{Y}_{[-1,\lambda]}, \dots, \hat{Y}_{[-5,\lambda]}$, and ESCV measures pointwise variance across these 5 vectors. Some predictions are in-sample (the point was in training) and some are out-of-sample (the point was held out), but that is fine because ESCV measures <em>stability</em> (do different training sets give similar predictions?), not prediction accuracy.</p>

<p><strong>Implementation.</strong> ESCV is not built into standard packages, but it is straightforward to compute from any CV loop. Fit the model on each of the $V$ training folds, predict all $n$ points from each model, compute their pointwise variance, and normalize by the squared norm of their mean. In R, <code>cv.glmnet(..., keep=TRUE)$fit.preval</code> stores the full prediction matrix (all $n$ points for each fold's model). In Python, fit each fold's model manually and call <code>predict</code> on the full dataset; <code>sklearn.model_selection.cross_val_predict</code> only returns out-of-fold predictions, so it does not directly give the ESCV prediction matrix. Select the $\lambda$ that minimizes ESCV, or use the "one-standard-error" heuristic applied to the ESCV curve instead of the CV error curve.</p>

<p><strong>Combining stability with accuracy.</strong> Pure ESCV can over-regularize (very stable but poor predictions). Two practical approaches:</p>

<ol>
<li><strong>ESCV as tiebreaker (recommended).</strong> Find all $\lambda$ within one standard error of the minimum CV error, then among those pick the $\lambda$ with minimum ESCV. This is the stability-aware version of the classic 1-SE rule, requires no extra tuning, and works best when the CV error curve is flat (common in high-dimensional settings) where multiple $\lambda$ values give similar accuracy and stability should decide.</li>
<li><strong>Combined criterion.</strong> Minimize $\text{CV}_{\text{error}}(\lambda) + \alpha \cdot \text{ESCV}(\lambda)$, where $\alpha$ controls the accuracy-stability tradeoff. Set $\alpha$ by normalizing both terms to the same scale (e.g., divide each by its range across the $\lambda$ grid) and starting with $\alpha = 1$ (equal weight). Increase $\alpha$ when stability matters more (scientific inference, regulatory settings); decrease when prediction accuracy dominates (forecasting, competitions).</li>
</ol>

<p><strong>ESCV is model-agnostic.</strong> The formula uses only prediction vectors, not model internals. It works for any model with a tuning parameter: LASSO, ridge, random forest, gradient boosting, neural networks. It also doubles as an outlier detector: the per-point contribution $\text{ESCV}_{i} = \frac{1}{V}\sum_{k=1}^V (\hat{f}_{-k}(x_{i}) - \bar{f}(x_{i}))^2$ flags points that drive instability. Points with high $\text{ESCV}_{i}$ are the ones where removing a fold changes the prediction most, which is exactly the PVOI interpretation. This avoids masking because each fold's model excludes that fold's points.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Connection 3: High-Dimensional Influence Measure

In high-dimensional linear models, the classical influence measure for observation $k$ is

$$
D_k = \frac{1}{p} \sum_{j=1}^{p} (\hat{\rho}_j - \hat{\rho}_j^{(k)})^2,
$$

where $\hat{\rho}_{j}$ is the marginal correlation between feature $j$ and $y$, and $\hat{\rho}_{j}^{(k)}$ is the same quantity with observation $k$ removed.

In the VoI framework, this is exactly an RVOI when the decision is the vector of marginal correlations under squared loss:

$$
\text{RVOI}((\mathbf{X}, \mathbf{y})_{[k]} \mid (\mathbf{X}, \mathbf{y})) = \frac{1}{p} \sum_{j=1}^{p} (\hat{\rho}_j - \hat{\rho}_j^{(k)})^2 = D_k.
$$

Under regularity conditions, when there is no influential point and $\min\{n, p\} \to \infty$:

$$
n^2 D_k \to \chi^2(1).
$$

This provides a formal test: compute $n^2 D_{k}$ for each observation and obtain a p-value for the hypothesis that observation $k$ is not influential.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">How to choose the threshold for the high-dimensional influence test in practice</span></summary>

<p><strong>Use the $\chi^2(1)$ p-value with a multiple testing correction.</strong> Compute $n^2 D_{k}$ for each observation and compare to $\chi^2_{1}$ quantiles. With $n$ simultaneous tests, apply a Bonferroni correction (reject if p-value $< \alpha / n$) or, less conservatively, control the false discovery rate with the Benjamini-Hochberg procedure. At $\alpha = 0.05$ with Bonferroni, the effective threshold on $n^2 D_{k}$ is $\chi^2_{1}(1 - 0.05/n)$, which is approximately $(\log n)^2$ for large $n$.</p>

<p><strong>Practical considerations.</strong> The asymptotic $\chi^2(1)$ result requires $\min\{n, p\} \to \infty$ and assumes no truly influential point exists under the null. In finite samples, the approximation can be liberal (too many false positives) when $n/p$ is close to 1. Simulate from the fitted model to calibrate the null distribution if you are in this regime. Compute $D_{k}$ for all observations and plot the sorted $n^2 D_{k}$ values against $\chi^2(1)$ quantiles (a Q-Q plot). Points that deviate sharply from the diagonal are candidate influential observations.</p>

<p><strong>Software.</strong> No standard R or Python package implements this specific test. The computation itself is simple: for each observation $k$, recompute all $p$ marginal correlations $\hat{\rho}_{j}^{(k)}$ by the leave-one-out update formula $\hat{\rho}_{j}^{(k)} = (n \hat{\rho}_{j} - x_{kj} y_{k} / s_{xj} s_{y}) / (n-1)$ (with appropriate rescaling), then sum the squared differences. Vectorized over $k$, this costs $\mathcal{O}(np)$ total, which is feasible even when $p$ is large.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Connection 4: Gradient-Based Influence (Influence Functions and Beyond)

Gradient-based methods use the derivatives of the loss function to measure influence. Because modern frameworks (PyTorch, JAX) compute gradients via automatic differentiation, these methods apply to any differentiable model. The same gradient machinery supports both **data influence** and **feature importance**.

### Data Influence: The Influence Function

The influence of training point $z$ on the loss at test point $z_{\text{test}}$ is:

$$
\text{IF}(z, z_{\text{test}}) = -\nabla_\theta L(z_{\text{test}}, \hat{\theta})^\top H_{\hat{\theta}}^{-1} \nabla_\theta L(z, \hat{\theta}),
$$

where $\hat{\theta}$ is the trained model parameters, $H_{\hat{\theta}} = \frac{1}{n} \sum_{i=1}^{n} \nabla_\theta^2 L(z_{i}, \hat{\theta})$ is the Hessian of the training loss, and $\nabla_\theta L(z, \hat{\theta})$ is the gradient of the loss at point $z$.

The intuition: $\nabla_\theta L(z, \hat{\theta})$ is the direction each training point pushes the parameters. $H_{\hat{\theta}}^{-1}$ rescales this direction by the local curvature. The dot product with the test gradient measures how much this parameter change affects the test prediction. Large positive IF means removing $z$ would increase the test loss (the training point was helpful); large negative means it would decrease the test loss (the point was harmful).

### Feature Importance: Input Gradients

The same gradient tool measures feature importance. The gradient of the loss (or prediction) with respect to the input features tells you which features the model is most sensitive to:

$$
\text{Feature importance of } x_j = \frac{\partial L(z_{\text{test}}, \hat{\theta})}{\partial x_j}.
$$

This is the basis of several feature attribution methods:

- **Saliency maps**: the raw input gradient $\nabla_{x} L$. Shows which input dimensions the loss is locally sensitive to.
- **Integrated gradients**: averages the gradient along a path from a baseline input to the actual input, satisfying a completeness axiom (attributions sum to the prediction difference).
- **GradientSHAP**: uses gradient information to approximate Shapley values for features, bridging gradient-based and game-theoretic methods.

The gradient perspective unifies data and feature influence: gradients with respect to **parameters** (scaled by training-point gradients) give data influence; gradients with respect to **inputs** give feature importance. Both are available from the same backpropagation pass.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Derivation, computational tricks, and limitations of influence functions</span></summary>

<p><strong>Where the formula comes from.</strong> Consider upweighting training point $z$ by a small amount $\epsilon$. The perturbed parameters are:</p>

$$\hat{\theta}_{\epsilon,z} = \arg\min_\theta \frac{1}{n} \sum_{i=1}^{n} L(z_i, \theta) + \epsilon \, L(z, \theta).$$

<p>By a first-order Taylor expansion around $\epsilon = 0$:</p>

$$\hat{\theta}_{\epsilon,z} \approx \hat{\theta} - \epsilon \, H_{\hat{\theta}}^{-1} \nabla_\theta L(z, \hat{\theta}).$$

<p>The change in test loss is:</p>

$$L(z_{\text{test}}, \hat{\theta}_{\epsilon,z}) - L(z_{\text{test}}, \hat{\theta}) \approx -\epsilon \, \nabla_\theta L(z_{\text{test}}, \hat{\theta})^\top H_{\hat{\theta}}^{-1} \nabla_\theta L(z, \hat{\theta}).$$

<p>Dividing by $\epsilon$ gives the influence function.</p>

<p><strong>Computational challenge.</strong> The Hessian $H_{\hat{\theta}}$ is a $p \times p$ matrix. For models with millions or billions of parameters, explicitly forming and inverting it is impossible ($\mathcal{O}(p^3)$ cost). Instead, the key quantity $s_{\text{test}} = H_{\hat{\theta}}^{-1} \nabla_\theta L(z_{\text{test}}, \hat{\theta})$ is computed by solving the linear system $H_{\hat{\theta}} \mathbf{x} = \nabla_\theta L(z_{\text{test}}, \hat{\theta})$ using conjugate gradient (CG) iterations. Each CG step requires one Hessian-vector product $H_{\hat{\theta}} \mathbf{v}$, computable via automatic differentiation in $\mathcal{O}(p)$ time without forming $H$ explicitly.</p>

<p><strong>Assumptions and limitations.</strong> The influence function assumes: (1) local convexity around $\hat{\theta}$ (the Hessian is positive definite), (2) the perturbation $\epsilon$ is small enough for the Taylor approximation to hold, and (3) the model is trained to convergence. Deep learning models are non-convex globally, but the local curvature around a converged $\hat{\theta}$ is often well-behaved enough for the approximation to be useful.</p>

<p><strong>Scalability concerns.</strong> Even with HVP tricks, influence functions remain expensive for very large models (GPT-scale). Follow-up methods include SGD-based influence tracing (retrace the training trajectory) and representer points (decompose predictions into linear combinations of training point activations).</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Connection 5: Shapley Values (Data and Feature)

The Shapley value from cooperative game theory provides a principled way to allocate credit among players. In influence analysis, the "players" can be either **data points** or **features**, and the "game" is the model's performance. The same formula applies to both; only the definition of players and utility changes.

### The Shapley Value Formula

Given a set of players $D = \{1, \ldots, n\}$ and a utility function $U$, the Shapley value of player $i$ is:

$$
\phi_i(U, D) = \sum_{S \subseteq D \setminus \{i\}} \frac{|S|! \, (|D| - |S| - 1)!}{|D|!} [U(S \cup \{i\}) - U(S)].
$$

This averages the marginal contribution of player $i$ over all possible subsets $S$. The combinatorial weighting ensures each subset size is weighted equally. The Shapley value is the unique allocation satisfying four axioms:

- **Efficiency.** $\sum_{i=1}^{n} \phi_{i} = U(D) - U(\emptyset)$. Total credit equals total utility.
- **Symmetry.** Equal contributors receive equal credit.
- **Null player.** Non-contributors receive zero credit.
- **Linearity.** For combined utilities $U = U_{1} + U_{2}$, credits add: $\phi_{i}(U) = \phi_{i}(U_{1}) + \phi_{i}(U_{2})$.

### Data Shapley: Players Are Training Points

In Data Shapley, each player is a training sample $z_{i}$. The utility $U(S)$ is the model's performance (e.g., validation accuracy) when trained on subset $S$. Data Shapley answers: "How much does each training point contribute to the model's ability to make good predictions?"

Applications include identifying mislabeled data (negative Shapley value), data pricing (compensate data providers proportionally), and data selection (keep only the high-value training points).

### Feature Shapley (SHAP): Players Are Input Features

In SHAP, each player is an input feature $x_{j}$. The utility $U(S)$ is the model's prediction when only features in $S$ are observed (features outside $S$ are marginalized out or set to baseline values). Feature Shapley answers: "How much does each feature contribute to this specific prediction?"

SHAP provides local, per-prediction explanations. For a prediction $f(x) = 3.5$, SHAP might say: $x_{1}$ contributed +1.2, $x_{2}$ contributed -0.5, $x_{3}$ contributed +0.3, etc., and these sum to $f(x) - E[f(x)]$ (by the efficiency axiom).

### The Same Formula, Different Games

| | Data Shapley | Feature Shapley (SHAP) |
|---|---|---|
| **Players** | Training samples $z_{1}, \ldots, z_{n}$ | Input features $x_{1}, \ldots, x_{d}$ |
| **Utility** | Model performance on validation set | Model prediction $f(x)$ |
| **Question** | Which training points helped the model? | Which features drove this prediction? |
| **Scope** | Global (across training process) | Local (per prediction) |
| **Cost** | Expensive ($2^n$ subsets of data, retrain each) | Moderate ($2^d$ subsets of features, one model) |

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Computation, approximations, and comparison with influence functions</span></summary>

<p><strong>Game-theoretic intuition.</strong> Think of each player joining a coalition one at a time, in a random order. The Shapley value is the average marginal contribution across all $n!$ orderings. This is equivalent to the subset formula but sometimes easier to reason about.</p>

<p><strong>Computational challenge.</strong> The exact formula requires evaluating $U$ on $2^n$ subsets (Data Shapley) or $2^d$ subsets (SHAP), which is infeasible for large $n$ or $d$. Approximation strategies include:</p>

<ul>
<li><strong>Monte Carlo permutation sampling.</strong> Sample random orderings and average marginal contributions. Works for both data and feature Shapley.</li>
<li><strong>KernelSHAP.</strong> Approximates feature Shapley values using a weighted linear regression on binary coalition vectors. Efficient for moderate $d$.</li>
<li><strong>TreeSHAP.</strong> Exact Shapley values for tree-based models in $\mathcal{O}(TLD^2)$ time, where $T$ is the number of trees, $L$ is the maximum number of leaves, and $D$ is the maximum depth. The key insight is that a decision tree already partitions the feature space into regions, so the combinatorial sum over feature subsets can be computed by a single recursive pass through the tree. For a given input $x$, TreeSHAP pushes $x$ down the tree while tracking, at each internal node, two quantities: (1) the proportion of feature subsets $S$ that include the split feature (in which case the split is applied using $x$'s value), and (2) the proportion that exclude it (in which case both branches are followed, weighted by the training data proportion going left vs right). At each leaf, the algorithm has accumulated the exact weight that each feature subset assigns to that leaf's prediction. These weights are aggregated into Shapley values by a polynomial-time recursion that avoids enumerating all $2^d$ subsets explicitly. For an ensemble of $T$ trees (e.g., XGBoost, LightGBM, random forest), the Shapley values from each tree are simply averaged (by the linearity axiom). This makes TreeSHAP fast enough for production use on large ensembles with hundreds of trees.</li>
<li><strong>GradientSHAP.</strong> Uses model gradients to approximate feature Shapley values, bridging gradient-based and game-theoretic methods.</li>
<li><strong>KNN-Shapley.</strong> Exact data Shapley values for KNN models in $\mathcal{O}(n \log n)$ time.</li>
</ul>

<p><strong>Data Shapley vs influence functions.</strong> Both measure data-point influence, but from different perspectives:</p>

<ul>
<li><strong>Influence functions</strong> measure local, infinitesimal perturbation: "if I slightly upweight $z$, how does the test loss change?" This is a first-order approximation around the current model.</li>
<li><strong>Data Shapley</strong> measures global contribution: "across all possible training subsets, what is $z$'s average marginal contribution?" This considers the effect of $z$ at all possible model states.</li>
</ul>

<p>Influence functions are cheaper (one Hessian-vector solve) but rely on local approximations. Data Shapley is more principled (axiomatic guarantees) but requires many model retrains. For large-scale deep learning, neither is cheap, but influence functions are more practical.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Comparison Of The Three Frameworks

All three frameworks can measure influence from both the data and feature perspectives. The following table compares them as general-purpose influence analysis tools.

| | Gradient-Based (IF Family) | Game-Theoretic (Shapley Family) | Decision-Theoretic (VoI) |
|---|---|---|---|
| **Tradition** | Robust statistics, optimization | Cooperative game theory | Bayesian decision theory |
| **Core question** | How does the loss surface change when I perturb the input? | What is each player's fair share of the total payoff? | How much does learning this information reduce expected loss? |
| **Data influence** | Influence function (Hessian-scaled gradient) | Data Shapley (marginal contribution across subsets) | RVOI / EVOIR for observations (Cook's distance as special case) |
| **Feature influence** | Input gradients, saliency maps, integrated gradients | Feature Shapley (SHAP) | PVOI for features (Sobol indices as special case) |
| **Parameter influence** | Fisher information, pruning criteria | Not standard | EVOIR for parameter groups (F-statistic as special case) |
| **Scope** | Local (first-order Taylor around current model) | Global (all possible subsets) | Both local (RVOI) and global (PVOI) |


Each framework has a natural strength. Gradient-based methods are the most practical for modern deep learning because gradients are cheap to compute via backpropagation. Game-theoretic methods provide the strongest fairness guarantees (the Shapley axioms). Decision-theoretic methods provide the most general mathematical framework, unifying classical statistics and modern influence analysis under a single loss-reduction criterion.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">How the three frameworks relate to each other</span></summary>

<p><strong>Gradient-based and decision-theoretic.</strong> Cook's distance is fundamentally a decision-theoretic quantity: it is the exact RVOI (squared prediction change from removing an observation) under squared loss, computed via the closed-form Sherman-Morrison update of OLS. No gradient approximation is needed. The influence function is a gradient-based method that <em>approximates</em> this same quantity for general models via a first-order Taylor expansion. For linear regression, the approximation happens to be exact because the loss is quadratic (the Taylor expansion has no higher-order terms). So Cook's distance belongs natively to the VoI framework, and the influence function recovers it as a special case only in the linear setting. For non-linear models, RVOI is the exact quantity and the influence function is the practical approximation.</p>

<p><strong>Game-theoretic and decision-theoretic.</strong> The PVOI additivity property ($\text{PVOI}(I_{1}\lvert Y) = \text{PVOI}(I_{2}\rvertY) + E_{I_{2}\lvert Y}\text{PVOI}(I_{1-2}\rvertI_{2},Y)$) mirrors the Shapley value's marginal contribution structure. The Shapley value can be viewed as a specific way of averaging conditional PVOIs across all possible conditioning sets, with combinatorial weights ensuring fairness. Both ask "how much does this information contribute?" but VoI uses a Bayesian criterion while Shapley uses axiomatic game theory.</p>

<p><strong>Gradient-based and game-theoretic.</strong> GradientSHAP (a variant of SHAP) uses gradients to approximate Shapley values for features, blending the two traditions. For data influence, influence functions give a local approximation of what Data Shapley measures globally: the effect of a single data point on the model.</p>

<p><strong>Sobol indices bridge all three.</strong> First-order Sobol indices equal PVOI for features under squared loss (decision-theoretic), can be estimated via gradient-based methods, and relate to the ANOVA decomposition that underlies balanced Shapley values. They sit at the intersection of all three frameworks.</p>

<p><strong>The unified template.</strong> All influence methods fit a common structure:</p>

<ol>
<li>Define what information $\phi$ you care about (parameter, feature, data point, new sample).</li>
<li>Define a measure of model quality (loss, utility, variance).</li>
<li>Compare the model with and without $\phi$.</li>
<li>Aggregate across uncertainty in $\phi$ (expectation, averaging over subsets, or conditioning).</li>
</ol>

<p>The three frameworks differ in step 4: gradient-based methods use local Taylor expansion, Shapley values average over all coalitions, and VoI takes a Bayesian expectation. The choice depends on whether you need computational efficiency (gradients), axiomatic fairness (Shapley), or decision-theoretic coherence (VoI).</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Key Takeaways

**All influence methods ask the same question.** "How much does this piece of information matter?" The piece can be a data point, a feature, a parameter group, or a future observation. The three frameworks (gradient-based, game-theoretic, decision-theoretic) differ in how they formalize "matter": local perturbation of the loss surface, fair allocation of total payoff, or expected reduction in decision loss.

**Tradeoff: computational cost vs theoretical guarantees.** Influence functions cost one Hessian-vector solve and give a local, first-order approximation. Data Shapley requires $2^n$ model retrains but provides axiomatic fairness guarantees (efficiency, symmetry, null player). VoI gives the most general framework but requires Bayesian integration over unknowns. In practice, choose by model scale: influence functions for deep learning, SHAP for tree ensembles, VoI when you need decision-theoretic coherence.

**Tradeoff: local vs global scope.** Influence functions and RVOI are local: they measure what happens when you perturb one point near the current model. Shapley values and PVOI are global: they average over all possible subsets or all possible realizations. Local methods are cheaper but can miss interactions. Global methods capture the full picture but are combinatorially expensive.

**Cook's distance is a special case, not a separate idea.** Under squared loss in linear regression, Cook's distance equals the exact RVOI (loss change from removing one observation), computed via the Sherman-Morrison formula. The influence function recovers the same quantity via a first-order Taylor expansion that happens to be exact for quadratic loss. Recognizing Cook's distance as VoI connects classical regression diagnostics to the modern influence literature.

**Retrospective vs prospective is the key distinction.** RVOI measures influence after the information is revealed ("how much did it actually help?"). PVOI measures expected influence before it is revealed ("how much do we expect it to help?"). RVOI is for diagnostics (was this data point influential?). PVOI is for experimental design (should we collect this feature?). The EVOIR ratio normalizes the two to a common scale.

**Tradeoff: data influence vs feature influence.** Both ask "what matters?" but at different levels. Data influence (influence functions, Data Shapley) tells you which training points shaped the model. Feature influence (SHAP, Sobol indices, input gradients) tells you which inputs drive a specific prediction. The same gradient machinery supports both: gradients with respect to parameters give data influence, gradients with respect to inputs give feature importance.

**Shapley's four axioms are the price of fairness.** The combinatorial $2^n$ cost of Shapley values is not an accident. It is the cost of satisfying efficiency (credits sum to total), symmetry (equal contributors get equal credit), null player (non-contributors get zero), and linearity (credits add across games). No cheaper method satisfies all four. TreeSHAP exploits tree structure to avoid the exponential cost, which is why SHAP is practical for tree models but expensive for deep networks.

