---
layout: post
title: "Optimization for Machine Learning"
date: 2025-09-06 00:00:00
tag:
- Statistics
- Machine Learning
- Deep Learning
blog: true
author: YingZhang
description: Convergence rates, first-order gradient methods, momentum, adaptive methods (AdamW, Muon), proximal methods, second-order methods, and practical tricks for deep learning and LLM training.
fontsize: 20pt
---

{% include mathjax_support.html %}

Optimization in machine learning is the problem of finding parameters $\theta \in \mathbb{R}^p$ that minimize a loss function:

$$\min_\theta\; L(\theta) = \frac{1}{n}\sum_{i=1}^n \ell(\theta;\, x_i, y_i).$$

Here $\ell(\theta; x_{i}, y_{i})$ is the **per-sample loss** (e.g., cross-entropy $-\log p_\theta(y_{i} \mid x_{i})$, or squared error $\frac{1}{2}(f_\theta(x_{i}) - y_{i})^2$), and $L(\theta)$ is its average over the training set. Note: $\ell$ here is the loss, not the log-likelihood, though for maximum likelihood estimation the two coincide up to sign ($\ell = -\log p_\theta$). $n$ can be enormous in modern settings (billions of tokens for LLM pretraining), and $L$ is typically nonconvex. This post covers convergence rates, first-order methods, momentum, adaptive methods (including Muon), proximal methods, second-order methods, and practical engineering tricks for deep learning and LLM training.

<div style="border:1px solid #c9b39a; border-radius:6px; background:#fcf9f3; padding:0.7em 1.15em; margin:1.7em 0; font-size:0.88em; line-height:1.7;">
<p style="margin:0 0 0.55em; font-weight:700; color:#8b5a2b; text-transform:uppercase; letter-spacing:0.05em; font-size:0.8em;">Quick index — jump to a method</p>
<p style="margin:0.3em 0;"><a href="#convergence-rates"><strong>Convergence rates</strong></a>: <a href="#r-rate-global-rate-of-decay">R-rate (global)</a> · <a href="#q-rate-local-ratio">Q-rate (local)</a></p>
<p style="margin:0.3em 0;"><a href="#first-order-gradient-methods"><strong>First-order gradient methods</strong></a>: <a href="#gradient-descent">Gradient descent</a> · <a href="#stochastic-gradient-descent">SGD</a></p>
<p style="margin:0.3em 0;"><a href="#momentum-methods"><strong>Momentum methods</strong></a>: <a href="#heavy-ball-classical-momentum">Heavy ball</a> · <a href="#nesterov-accelerated-gradient">Nesterov</a></p>
<p style="margin:0.3em 0;"><a href="#adaptive-learning-rate-methods"><strong>Adaptive learning rate</strong></a>: <a href="#adagrad">AdaGrad</a> · <a href="#rmsprop">RMSprop</a> · <a href="#adam">Adam</a> · <a href="#adamw">AdamW</a> · <a href="#muon">Muon</a></p>
<p style="margin:0.3em 0;"><a href="#learning-rate-schedules"><strong>Learning rate schedules</strong></a></p>
<p style="margin:0.3em 0;"><a href="#proximal-and-constrained-optimization"><strong>Proximal and constrained optimization</strong></a>: <a href="#subgradients">Subgradients</a> · <a href="#optimality-conditions-and-kkt">KKT conditions</a> · <a href="#primal-and-dual-problems">Primal and dual</a> · <a href="#proximal-operator">Proximal operator</a> · <a href="#proximal-gradient-ista">ISTA</a> · <a href="#fista">FISTA</a> · <a href="#admm">ADMM</a></p>
<p style="margin:0.3em 0;"><a href="#second-order-methods"><strong>Second-order methods</strong></a></p>
<p style="margin:0.3em 0;"><a href="#loss-landscape-and-generalization"><strong>Loss landscape and generalization</strong></a></p>
<p style="margin:0.3em 0;"><a href="#practical-tricks-for-deep-learning-and-llm-training"><strong>Practical tricks (DL / LLM)</strong></a>: <a href="#gradient-clipping">Gradient clipping</a> · <a href="#mixed-precision-training">Mixed precision</a> · <a href="#gradient-accumulation">Gradient accumulation</a> · <a href="#batch-size-and-learning-rate-scaling">Batch size and LR scaling</a> · <a href="#parameter-initialization">Initialization</a> · <a href="#gradient-checkpointing">Checkpointing</a> · <a href="#optimizer-state-sharding-zero">ZeRO sharding</a> · <a href="#pre-norm-vs-post-norm">Pre- vs post-norm</a></p>
<p style="margin:0.3em 0;"><a href="#key-takeaways"><strong>Key takeaways</strong></a> · <a href="#comparison"><strong>Comparison table</strong></a></p>
</div>

## Convergence Rates

Two classification systems are commonly used and should not be confused.

**R-rate (global rate)** asks: after $k$ steps total, how small is the error? It describes the overall trajectory of

$$\|\theta_k - \theta^*\|$$

as a function of $k$, giving a global envelope like

$$O(1/k) \quad \text{or} \quad O(a^k).$$

Named "R" for *root* because it is formally defined via

$$\limsup_{k\to\infty} \|\theta_k - \theta^*\|^{1/k}.$$

**Q-rate (local ratio)** asks: how much does step $k+1$ improve over step $k$? It looks only at consecutive pairs:

$$\frac{\|\theta_{k+1} - \theta^*\|}{\|\theta_k - \theta^*\|^q},$$

and classifies convergence by the order $q$. Named "Q" for *quotient*.

The two are related but distinct. Q-linear (constant ratio less than 1 at every step) implies R-linear (geometric global decay). But R-linear does not imply Q-linear: the step-by-step ratio could spike occasionally yet still yield geometric decay on average. Q-rate is more informative near a solution; R-rate is the right tool for comparing algorithms globally.

### R-rate: Global Rate of Decay

The R-rate describes how fast $\lVert \theta_{k} - \theta^{\ast}\rVert$ shrinks as a function of $k$.

**Sublinear.** The error decays polynomially:

$$\|\theta_k - \theta^*\| = O(1/k^p), \quad p > 0.$$

The key quantity is the **fraction of remaining error removed** at each step, $1 - \text{ratio}$, where

$$\text{ratio}_k = \frac{\|\theta_{k+1} - \theta^*\|}{\|\theta_k - \theta^*\|}.$$

For $\lVert \theta_{k} - \theta^{\ast}\rVert \approx 1/k$:

$$\text{ratio}_k \approx \frac{1/(k+1)}{1/k} = \frac{k}{k+1}, \qquad 1 - \text{ratio}_k = \frac{1}{k+1} \to 0.$$

The ratio itself approaches 1, but that is not the problem (the sequence still converges because ratio $< 1$). The problem is that $1 - \text{ratio}_{k} \to 0$: the **fraction of error removed per step shrinks to zero**. At step 10 you remove $\approx 9\%$ of remaining error; at step 1000 you remove $\approx 0.1\%$. Later steps are increasingly ineffective relative to what remains.

**Linear.** The error decays geometrically:

$$\|\theta_k - \theta^*\| = O(a^k), \quad a \in (0,1).$$

For $\lVert \theta_{k} - \theta^{\ast}\rVert \approx a^k$:

$$\text{ratio}_k \approx \frac{a^{k+1}}{a^k} = a, \qquad 1 - \text{ratio}_k = 1 - a > 0.$$

The fraction of error removed per step is the **constant** $1-a$, regardless of $k$. Every step is equally effective relative to the current error. This is why linear convergence is far preferable to sublinear for optimization in practice.

Gradient descent on $\mu$-strongly convex $L$ achieves linear convergence with $a = 1 - \eta\mu$.

**Superlinear.** The ratio itself goes to 0:

$$\text{ratio}_k = \frac{\|\theta_{k+1} - \theta^*\|}{\|\theta_k - \theta^*\|} \to 0.$$

The fraction of error removed per step approaches 1: convergence **accelerates** as $k$ grows. Near a solution, Newton's method achieves Q-quadratic convergence (see below): if you have 4 correct digits today, you have 8 tomorrow.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Numerical illustration: sublinear vs linear vs quadratic</span></summary>

<p>Suppose the error at step $k$ follows three different rules. Starting from error $= 1$:</p>

<table>
<thead><tr><th>Step $k$</th><th>Sublinear $1/k$</th><th>Linear $0.5^k$</th><th>Quadratic $2^{-2^k}$</th></tr></thead>
<tbody>
<tr><td>1</td><td>1.000</td><td>0.500</td><td>0.500</td></tr>
<tr><td>2</td><td>0.500</td><td>0.250</td><td>0.250</td></tr>
<tr><td>4</td><td>0.250</td><td>0.063</td><td>0.004</td></tr>
<tr><td>8</td><td>0.125</td><td>0.004</td><td>$\approx 10^{-77}$</td></tr>
<tr><td>16</td><td>0.063</td><td>$\approx 10^{-5}$</td><td>machine zero</td></tr>
</tbody>
</table>

<p>Sublinear halves the error only when $k$ doubles: you need 16 steps to get from $1/8$ to $1/16$, while linear needs just 1 more step. Quadratic reaches machine precision in under 10 steps.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Q-rate: Local Ratio

The Q-rate (quotient rate) classifies convergence by the order $q$ in

$$\|\theta_{k+1} - \theta^*\| \le C\|\theta_k - \theta^*\|^q.$$

**Q-linear ($q=1$, $C < 1$):** each step multiplies the error by at most $C$, equivalent to R-linear. The bound is

$$\|\theta_{k+1}-\theta^*\| \le C\|\theta_k - \theta^*\|, \qquad \text{so} \qquad \|\theta_k - \theta^*\| \le C^k\|\theta_0-\theta^*\|.$$

**Q-quadratic ($q=2$):** the error at step $k+1$ is bounded by $C$ times the square of the error at step $k$:

$$\|\theta_{k+1}-\theta^*\| \le C\|\theta_k-\theta^*\|^2.$$

If the current error is $10^{-m}$ ($m$ correct digits), then the next error is at most $C \cdot 10^{-2m}$: the number of correct digits doubles each step. Newton's method achieves this near a solution.

Q-linear implies R-linear. Q-quadratic implies R-superlinear. Sublinear convergence ($O(1/k^p)$) has no clean Q-rate: the ratio $\to 1$, so the Q framework does not apply directly.

**Summary:**

| Rate | $\lVert \theta_{k} - \theta^{\ast}\rVert$ | $\text{ratio}_{k} \to$ | Fraction removed/step |
|---|---|---|---|
| Sublinear | $O(1/k^p)$ | $1$ | $\to 0$ (shrinks) |
| Linear | $O(a^k)$ | $a \in (0,1)$ | $1-a$ (constant) |
| Superlinear | faster than $a^k$ | $0$ | $\to 1$ (grows) |
| Q-quadratic | $C\lVert e_{k}\rVert^2$ | $0$ | digits double/step |

The methods in this post are organized into six themes:

1. **First-order gradient methods.** Gradient descent and its stochastic variant form the foundation. The tradeoff is per-step cost ($O(n)$ vs $O(1)$) against convergence rate ($O(1/k)$ vs $O(1/\sqrt{k})$).

2. **Momentum methods.** Heavy ball and Nesterov acceleration add memory of past gradients, reducing the effective condition number from $\kappa$ to $\sqrt{\kappa}$ and achieving the optimal $O(1/k^2)$ rate for convex problems.

3. **Adaptive learning rate methods.** AdaGrad, RMSprop, Adam, AdamW, and Muon maintain per-parameter or per-matrix scaling. AdamW is the standard for LLM training; Muon is a recent alternative that uses matrix orthogonalization instead of element-wise scaling.

4. **Proximal and constrained optimization.** Subgradients, KKT conditions, and proximal operators extend optimization to non-smooth and constrained problems. FISTA achieves the optimal $O(1/k^2)$ rate for composite objectives like the lasso.

5. **Second-order methods.** Newton's method, conjugate gradient, L-BFGS, and the natural gradient exploit curvature information for faster convergence, at the cost of $O(p^2)$ or more memory.

6. **Loss landscape and generalization.** SAM targets flat minima that generalize better, at $2\times$ the compute cost per step.

## First-Order Gradient Methods

These methods use only the gradient (or a stochastic estimate of it) to update parameters. They are the simplest and most scalable family of optimizers.

### Gradient Descent

The update rule is

$$\theta_{k+1} = \theta_k - \eta \nabla L(\theta_k),$$

where $\eta > 0$ is the learning rate (step size).

**Convergence.** Under $L$-smoothness ($\lVert \nabla L(\theta) - \nabla L(\phi)\rVert \le L\lVert \theta - \phi\rVert$) and convexity, gradient descent with $\eta = 1/L$ satisfies

$$L(\theta_k) - L(\theta^*) \le \frac{\|\theta_0 - \theta^*\|^2}{2\eta k}.$$

This is $O(1/k)$. For $\mu$-strongly convex $L$, the rate improves to linear:

$$\|\theta_k - \theta^*\|^2 \le (1 - \eta\mu)^k \|\theta_0 - \theta^*\|^2.$$

**Pros.** Simple to implement; clean convergence theory; exact gradient means fixed learning rate suffices; linear convergence for strongly convex problems.

**Cons.** Requires one full pass over all $n$ data points per step; impractical for large datasets; single global learning rate for all parameters; sensitive to $L$-smoothness constant for step size tuning.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Derivation of the O(1/k) bound</span></summary>

<p>$L$-smoothness gives the descent lemma: for any $\theta, \phi$,</p>

$$L(\phi) \le L(\theta) + \nabla L(\theta)^T(\phi - \theta) + \frac{L}{2}\|\phi - \theta\|^2.$$

<p>Apply with $\phi = \theta_{k+1} = \theta_{k} - \eta\nabla L(\theta_{k})$ and $\eta = 1/L$:</p>

$$L(\theta_{k+1}) \le L(\theta_k) - \frac{1}{2L}\|\nabla L(\theta_k)\|^2.$$

<p>Convexity gives $L(\theta_{k}) - L(\theta^{\ast}) \le \nabla L(\theta_{k})^T(\theta_{k} - \theta^{\ast})$, so by Cauchy-Schwarz:</p>

$$\|\nabla L(\theta_k)\|^2 \ge \frac{(L(\theta_k)-L(\theta^*))^2}{\|\theta_k - \theta^*\|^2}.$$

<p>Combining and telescoping over $k$ steps yields $L(\theta_{k}) - L(\theta^{\ast}) \le \lVert \theta_{0} - \theta^{\ast}\rVert^2 / (2k/L)$.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Stochastic Gradient Descent

Full-batch gradient descent requires one pass over all $n$ data points per step. For large $n$ (e.g., billions of tokens in LLM pretraining), this is prohibitive. SGD replaces the full gradient with a noisy estimate from a single sample (or mini-batch of size $B$):

$$\theta_{k+1} = \theta_k - \eta_k \nabla \ell(\theta_k;\, x_{i_k}, y_{i_k}).$$

The gradient estimate is unbiased: $E[\nabla \ell(\theta_{k}; x_{i_{k}})] = \nabla L(\theta_{k})$. The noise variance is $\sigma^2 = E[\lVert \nabla \ell - \nabla L\rVert^2]$.

**Convergence with decaying learning rate.** Under convexity and $\eta_{k} = c/\sqrt{k}$:

$$E[L(\theta_k)] - L(\theta^*) = \tilde{O}(1/\sqrt{k}).$$

This is slower than full-batch $O(1/k)$, but each iteration costs $O(1)$ vs $O(n)$, so SGD reaches a given accuracy in far fewer data touches when $n$ is large.

**Fixed vs decaying learning rate.** Whether a fixed $\eta$ converges depends on whether there is noise.

**Full-batch GD with fixed learning rate:** converges. The gradient is exact so the noise term is zero. The recursion is purely contractive and $\eta = 1/L$ gives $O(1/k)$. No decay needed.

**SGD with fixed learning rate:** does not converge to the optimum. A random sample has gradient variance

$$\sigma^2 = E\|\nabla\ell(\theta^*; x_i, y_i) - \nabla L(\theta^*)\|^2 > 0$$

even at the optimum, because a single sample never gives exactly $\nabla L(\theta^{\ast}) = 0$. The noise floor stabilizes at

$$E[L(\theta_k)] - L(\theta^*) \approx \frac{\eta \sigma^2}{2\mu} > 0,$$

so the iterates settle into a neighborhood of the optimum rather than converging to it. Decaying $\eta_{k} \to 0$ kills the noise floor and recovers convergence at $O(1/\sqrt{k})$.

In LLM training, a warmup + cosine decay schedule is used for this reason: the large early $\eta$ makes fast initial progress, and the small late $\eta$ shrinks the noise floor so the optimizer settles into a flat minimum rather than bouncing around it.

**Mini-batch SGD.** Using a batch of size $B$ reduces the gradient variance by $1/B$ (if samples are independent), improving the constant in the bound but not the $O(1/\sqrt{k})$ rate. In practice $B \in [32, 4096]$ balances variance reduction against hardware parallelism.

**Pros.** $O(1)$ cost per step; scales to arbitrarily large datasets; gradient noise helps escape saddle points and sharp minima; foundation of all modern deep learning optimization.

**Cons.** Slower $O(1/\sqrt{k})$ rate vs $O(1/k)$ for full-batch GD; requires decaying learning rate to converge; high variance with small batches; no per-parameter adaptivity (same $\eta$ for all weights).

<details>
<summary><span style="color: saddlebrown; font-style: italic;">One-step recursion: the three-term structure</span></summary>

<p>Expand the squared distance to $\theta^{\ast}$ after one SGD step, where $g_{k} = \nabla\ell(\theta_{k}; x_{i_{k}})$:</p>

$$E\|\theta_{k+1} - \theta^*\|^2 = E\|\theta_k - \eta_k g_k - \theta^*\|^2$$
$$= \underbrace{E\|\theta_k - \theta^*\|^2}_{\text{current distance}} \;-\; \underbrace{2\eta_k E[g_k^T(\theta_k - \theta^*)]}_{\text{contraction}} \;+\; \underbrace{\eta_k^2 E\|g_k\|^2}_{\text{noise injection}}.$$

<p><strong>Why the contraction term is negative.</strong> The full contraction term is $-2\eta_{k} E[g_{k}^T(\theta_{k} - \theta^{\ast})]$. It is negative when $E[g_{k}^T(\theta_{k} - \theta^{\ast})] > 0$, i.e., when the gradient points in the same direction as $(\theta_{k} - \theta^{\ast})$. Two steps:</p>

<p><strong>Step 1: Replace $g_{k}$ with $\nabla L(\theta_{k})$ via unbiasedness.</strong> Since $g_{k}$ is an unbiased estimator of the full gradient conditional on $\theta_{k}$:</p>

$$E[g_k^T(\theta_k - \theta^*)] = \nabla L(\theta_k)^T(\theta_k - \theta^*).$$

<p><strong>Step 2: Show $\nabla L(\theta_{k})^T(\theta_{k} - \theta^{\ast}) \ge L(\theta_{k}) - L(\theta^{\ast}) > 0$ by convexity.</strong> Convexity of $L$ means the function lies above every tangent hyperplane. At $\theta^{\ast}$:</p>

$$L(\theta^*) \ge L(\theta_k) + \nabla L(\theta_k)^T(\theta^* - \theta_k).$$

<p>Rearranging:</p>

$$\nabla L(\theta_k)^T(\theta_k - \theta^*) \ge L(\theta_k) - L(\theta^*) > 0,$$

<p>where the last inequality holds because $\theta_{k} \ne \theta^{\ast}$ (if $\theta_{k} = \theta^{\ast}$ we are done). Geometrically, the gradient $\nabla L(\theta_{k})$ always points away from the optimum $\theta^{\ast}$: it makes a positive angle with the vector $(\theta_{k} - \theta^{\ast})$ pointing from $\theta^{\ast}$ to the current iterate. So the contraction term $-2\eta_{k} \times \text{positive}$ is always negative, always pulling $\theta_{k}$ toward $\theta^{\ast}$.</p>

<p>The noise term is always positive (bounded by $\eta_{k}^2 G^2$ under the bounded gradient assumption $E\lVert g_{k}\rVert^2 \le G^2$). It injects error back in regardless of how close $\theta_{k}$ is to $\theta^{\ast}$.</p>

<p>Applying the bounds on both terms gives the key recursion:</p>

$$E\|\theta_{k+1} - \theta^*\|^2 \le E\|\theta_k - \theta^*\|^2 - 2\eta_k(E[L(\theta_k)] - L(\theta^*)) + \eta_k^2 G^2.$$

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why a fixed learning rate cannot converge</span></summary>

<p>With $\eta_{k} = \eta$ (constant), rearrange the key recursion to isolate the excess loss:</p>

$$2\eta(E[L(\theta_k)] - L(\theta^*)) \le E\|\theta_k - \theta^*\|^2 - E\|\theta_{k+1} - \theta^*\|^2 + \eta^2 G^2.$$

<p>As $\theta_{k} \to \theta^{\ast}$, the left side $\to 0$ and the telescoping terms $\to 0$, but the noise floor $\eta^2 G^2$ remains fixed. The equation balances at</p>

$$E[L(\theta_k)] - L(\theta^*) \;\approx\; \frac{\eta G^2}{2} \;>\; 0.$$

<p>No matter how long we run, the expected loss stays above $\theta^{\ast}$ by at least $\eta G^2/2$. Making $\eta$ small reduces the floor but also slows the contraction, so there is a tradeoff: smaller $\eta$ means slower progress but a tighter final neighborhood. With fixed $\eta$, we can only converge to a ball around $\theta^{\ast}$, not to $\theta^{\ast}$ itself.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">How decaying η_k = c/√k gives O(1/√k)</span></summary>

<p>Starting from the key recursion, rearrange and sum from $k=1$ to $K$. The distance terms telescope:</p>

$$2\sum_{k=1}^K \eta_k(E[L(\theta_k)] - L(\theta^*)) \le \|\theta_0 - \theta^*\|^2 + G^2\sum_{k=1}^K \eta_k^2.$$

<p>Define the weighted average iterate $\bar\theta_{K} = \frac{\sum_{k} \eta_{k} \theta_{k}}{\sum_{k} \eta_{k}}$. By convexity of $L$:</p>

$$E[L(\bar\theta_K)] - L(\theta^*) \le \frac{\|\theta_0 - \theta^*\|^2 + G^2\sum_{k=1}^K \eta_k^2}{2\sum_{k=1}^K \eta_k}.$$

<p>Now substitute $\eta_{k} = c/\sqrt{k}$ and evaluate the two sums:</p>

$$\sum_{k=1}^K \eta_k = c\sum_{k=1}^K \frac{1}{\sqrt{k}} \approx 2c\sqrt{K}, \qquad \sum_{k=1}^K \eta_k^2 = c^2\sum_{k=1}^K \frac{1}{k} \approx c^2\log K.$$

<p>Substituting:</p>

$$E[L(\bar\theta_K)] - L(\theta^*) \le \frac{\|\theta_0-\theta^*\|^2 + G^2 c^2\log K}{4c\sqrt{K}} = O\!\left(\frac{\log K}{\sqrt{K}}\right).$$

<p>This is sometimes written loosely as $O(1/\sqrt{K})$, which is an abuse of notation: $\log K / \sqrt{K}$ is not strictly $O(1/\sqrt{K})$ because $\log K \to \infty$. The formal notation is $\tilde{O}(1/\sqrt{K})$, meaning $O(1/\sqrt{K})$ up to logarithmic factors. Since $\log K$ grows far slower than any positive power of $K$, the $\log K$ factor is negligible in practice and the rate is effectively $1/\sqrt{K}$.</p>

<p>The key insight: $\eta_{k} \to 0$ kills the noise floor ($\sum \eta_{k}^2$ grows only as $\log K$), while $\sum \eta_{k} \sim \sqrt{K}$ grows fast enough that averaging still makes progress. Any faster decay (e.g., $\eta_{k} = c/k$) would make $\sum \eta_{k}$ grow too slowly, losing progress. Any slower decay (fixed $\eta$) leaves the noise floor nonzero.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why full-batch GD achieves O(1/k), and why mini-batch doesn't improve the rate</span></summary>

<p><strong>Full-batch GD.</strong> With the exact gradient, there is no noise: $G^2 = 0$. The recursion becomes</p>

$$L(\theta_{k+1}) \le L(\theta_k) - \frac{1}{2L}\|\nabla L(\theta_k)\|^2.$$

<p>With a fixed $\eta = 1/L$, the noise floor is zero and no decay is needed. Telescoping gives $L(\theta_{K}) - L(\theta^{\ast}) \le O(1/K)$. Full-batch can use a fixed learning rate and achieves a faster $O(1/K)$ rate precisely because the noise term is absent.</p>

<p><strong>Mini-batch SGD.</strong> With batch size $B$, samples are averaged so the gradient variance drops from $\sigma^2$ to $\sigma^2/B$. In the bound, $G^2$ is replaced by $G^2/B$:</p>

$$E[L(\bar\theta_K)] - L(\theta^*) \le \frac{\|\theta_0-\theta^*\|^2 + (G^2/B)\sum_k \eta_k^2}{2\sum_k \eta_k}.$$

<p>The numerator shrinks by $1/B$, but the structure is identical. With $\eta_{k} = c/\sqrt{k}$, the bound is still $O(1/\sqrt{K})$, just with a smaller constant $1/B$. The rate is unchanged because the noise term ($\sim \log K$) still dominates the numerator for large $K$, regardless of $B$. Only at $B = n$ (full batch) does the noise vanish entirely, eliminating the $\sum \eta_{k}^2$ term and recovering $O(1/K)$.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Momentum Methods

Gradient descent uses only the current gradient and has no memory of past steps. This causes two problems: slow progress along shallow directions, and oscillation across steep directions. Momentum methods accumulate a velocity vector that builds up speed in consistent directions and dampens oscillation in alternating directions.

**The ravine intuition.** Consider a loss surface shaped like an elongated valley: steep sides, gentle slope along the bottom. Plain GD follows the gradient at each step, oscillating left-right across the narrow steep dimension while making slow progress along the gentle bottom. Momentum helps because the left-right gradient components alternate in sign and cancel in the velocity $v_{k}$, while the consistent downhill components accumulate. The result is faster progress along the bottom and damped oscillations across it.

### Heavy Ball (Classical Momentum)

$$v_{k+1} = \beta v_k - \eta \nabla L(\theta_k), \qquad \theta_{k+1} = \theta_k + v_{k+1}.$$

The velocity $v_{k}$ is a weighted sum of all past gradients with exponentially decaying weights $\beta^j$ for gradient $j$ steps ago. In a direction where gradients consistently point the same way, $v_{k}$ grows proportionally to $1/(1-\beta)$ times the gradient magnitude. In a direction where gradients oscillate (as in the ravine sides), the positive and negative contributions cancel and $v_{k}$ stays small.

**Convergence.** For strongly convex quadratics, heavy ball with optimal $\beta = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2$ achieves linear convergence with condition number $\sqrt{\kappa}$ instead of $\kappa$, where $\kappa = L/\mu$.

**Pros.** Reduces effective condition number from $\kappa$ to $\sqrt{\kappa}$; simple one-line addition to GD; intuitive physical analogy (ball rolling with momentum).

**Cons.** Convergence guarantee only for strongly convex quadratics; can oscillate or diverge with poor $\beta$ choice; no theoretical guarantees for nonconvex objectives; superseded by Nesterov in theory.

### Nesterov Accelerated Gradient

Nesterov's insight: compute the gradient at a lookahead point, not the current iterate.

$$\theta_{k+1} = y_k - \eta \nabla L(y_k), \qquad y_{k+1} = \theta_{k+1} + \frac{k}{k+3}(\theta_{k+1} - \theta_k).$$

**Convergence.** For convex $L$:

$$L(\theta_k) - L(\theta^*) \le \frac{2\|\theta_0 - \theta^*\|^2}{\eta(k+1)^2} = O(1/k^2).$$

This is optimal for first-order methods on smooth convex functions; no gradient-based algorithm can do better in the worst case.

**Pros.** Optimal $O(1/k^2)$ rate for convex problems; widely used as the backbone of FISTA for composite objectives; momentum coefficient is theoretically derived, not a free hyperparameter.

**Cons.** Full-batch required for theoretical guarantees; no convergence improvement over SGD in the stochastic setting; momentum coefficient needs adjustment for nonconvex problems.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why lookahead gives O(1/k²): the estimate sequence argument</span></summary>

<p>Define the momentum coefficient $\lambda_{k+1}^2 = \lambda_{k+1} + \lambda_{k}^2$ (with $\lambda_{0} = 0$, so $\lambda_{k} \approx k/2$). Nesterov constructs an <em>estimate sequence</em> $\phi_{k}(\theta)$, a sequence of lower bounds on $L$ that tighten at rate $1/\lambda_{k}^2$. The lookahead point $y_{k}$ is chosen so that the iterate $\theta_{k+1}$ maintains the invariant $L(\theta_{k}) \le \phi_{k}^{\ast}$ (minimum of $\phi_{k}$). Tightening of $\phi_{k}^{\ast}$ propagates to $L(\theta_{k})$, giving the $O(1/k^2)$ bound. The key is that the momentum coefficient $k/(k+3)$ is not arbitrary; it is derived from the $\lambda_{k}$ recurrence to maintain this invariant exactly.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Adaptive Learning Rate Methods

A single global learning rate is suboptimal when parameters have very different gradient magnitudes. Adaptive methods maintain per-parameter learning rates, scaling down for parameters with large historical gradients and scaling up for sparse ones.

### AdaGrad

$G_{k} \in \mathbb{R}^p$ accumulates the sum of squared gradients for each parameter separately, and each parameter is updated by dividing by its own $\sqrt{G_{k,j}}$:

$$G_{k,j} = G_{k-1,j} + (\nabla_j L(\theta_k))^2, \qquad \theta_{k+1,j} = \theta_{k,j} - \frac{\eta}{\sqrt{G_{k,j} + \epsilon}} \cdot \nabla_j L(\theta_k).$$

**Why dividing by $\sqrt{G_{k}}$ cancels out gradient magnitude.** The effective update for parameter $j$ is $\frac{\eta}{\sqrt{G_{k,j}}} \cdot \nabla_{j} L(\theta_{k})$. Consider two cases:

- **Dense parameter** (gradient magnitude approximately $M > 0$ at every step): after $k$ steps, $G_{k,j} \approx kM^2$, so $\sqrt{G_{k,j}} \approx \sqrt{k}\,M$, and the effective update is $\frac{\eta}{\sqrt{k}\,M} \cdot M = \frac{\eta}{\sqrt{k}}$. The gradient magnitude $M$ cancels exactly, so large-gradient parameters are automatically given a smaller learning rate.
- **Sparse parameter** (gradient magnitude $M > 0$ only $s \ll k$ times, zero otherwise): $G_{k,j} \approx sM^2$, so the effective update is $\frac{\eta}{\sqrt{s}\,M} \cdot M = \frac{\eta}{\sqrt{s}}$. Since $s \ll k$, this is much larger than for the dense parameter, so rare parameters keep a large effective learning rate.

In both cases the gradient magnitude cancels out of the update. The current gradient only provides the **direction**; the **step size** is determined entirely by the history of past gradients accumulated in $G_{k,j}$. This is equivalent to normalizing each update by the root-mean-square (RMS) of past gradients: frequently updated parameters get shrinking steps, rarely updated parameters keep large steps.

**Limitation.** $G_{k,j}$ only ever grows, so $\eta/\sqrt{G_{k,j}} \to 0$ for every parameter eventually, and learning stops. This makes AdaGrad poorly suited for deep networks trained for many steps.

**Pros.** Automatic per-parameter learning rates; excellent for sparse gradients (e.g., word embeddings where most parameters receive zero gradient per step); no learning rate tuning needed for sparse problems.

**Cons.** Effective learning rate decays to zero monotonically, so learning eventually stops; unsuitable for long training runs or dense gradient problems; superseded by RMSprop and Adam in practice.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">How to choose AdaGrad hyperparameters in practice</span></summary>

<p><strong>Initial learning rate $\eta$.</strong> AdaGrad is less sensitive to $\eta$ than plain SGD because the per-parameter scaling self-corrects. Start with $\eta = 0.01$ for dense models or $\eta = 0.1$ for sparse problems (e.g., word embeddings, click-through-rate models). If the loss diverges, halve $\eta$; if training is too slow, double it. Unlike Adam, there is no momentum to interact with $\eta$, so the tuning surface is simpler.</p>

<p><strong>Epsilon $\epsilon$.</strong> The default $\epsilon = 10^{-8}$ works almost universally. Increase to $\epsilon = 10^{-6}$ or $10^{-4}$ only if you observe numerical instability (NaN gradients), which can happen when some parameters receive extremely sparse updates and $G_{k,j}$ stays near zero.</p>

<p><strong>When to use AdaGrad over Adam.</strong> AdaGrad remains a strong choice when features are genuinely sparse and training is short enough that the monotonically decaying learning rate does not become a problem. Typical use cases: training word embeddings with GloVe, online learning on high-dimensional sparse features, and convex objectives where the $\tilde{O}(1/\sqrt{k})$ regret bound is directly useful. For long training runs on dense models, prefer Adam or AdamW.</p>

<p><strong>Software.</strong> <code>torch.optim.Adagrad(params, lr=0.01, eps=1e-8)</code> in PyTorch. In TensorFlow: <code>tf.keras.optimizers.Adagrad(learning_rate=0.01, epsilon=1e-8)</code>.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### RMSprop

RMSprop fixes AdaGrad's vanishing learning rate by replacing the cumulative sum $G_{k} = \sum_{j=1}^k \nabla L(\theta_{j})^2$ with an exponential moving average (EMA):

$$v_k = \rho v_{k-1} + (1-\rho)\nabla L(\theta_k)^2, \qquad \theta_{k+1} = \theta_k - \frac{\eta}{\sqrt{v_k + \epsilon}} \odot \nabla L(\theta_k),$$

with $\rho \in [0.9, 0.99]$ typically. In AdaGrad, $G_{k}$ grows without bound so $\eta/\sqrt{G_{k}} \to 0$. The EMA instead gives each squared gradient an exponentially decaying weight, so $v_{k}$ only remembers recent gradient magnitudes and stays bounded. The effective learning rate no longer decays to zero.

**Pros.** Fixes AdaGrad's vanishing learning rate; simple and robust; good empirical performance on RNNs and recurrent architectures.

**Cons.** No convergence theory for general nonconvex problems; no first moment (momentum), unlike Adam; sensitive to $\rho$ and $\epsilon$; largely replaced by Adam/AdamW in modern practice.

### Adam

Adam (Adaptive Moment Estimation) combines RMSprop's second moment with a first moment estimate (momentum). Defaults: $\beta_{1} = 0.9$, $\beta_{2} = 0.999$, $\epsilon = 10^{-8}$.

**Step 1: first moment (momentum)**

$$m_k = \beta_1 m_{k-1} + (1-\beta_1)\nabla L(\theta_k).$$

**Step 2: second moment (adaptive scaling)**

$$v_k = \beta_2 v_{k-1} + (1-\beta_2)\nabla L(\theta_k)^2.$$

**Step 3: bias correction**

$$\hat m_k = \frac{m_k}{1-\beta_1^k}, \qquad \hat v_k = \frac{v_k}{1-\beta_2^k}.$$

**Step 4: parameter update**

$$\theta_{k+1} = \theta_k - \frac{\eta}{\sqrt{\hat v_k} + \epsilon}\hat m_k.$$

The bias corrections matter early in training. At step 1, $m_{1} = (1-\beta_{1})g_{1}$, which is much smaller than $g_{1}$. Dividing by $(1-\beta_{1}^k)$ rescales it back to the true gradient magnitude.

**Pros.** Combines adaptive learning rates with momentum; fast convergence in practice; robust to learning rate choice; default optimizer for most deep learning.

**Cons.** No convergence proof for general convex problems (counterexamples exist); $L_{2}$ regularization is broken when used with Adam (use AdamW instead); can generalize worse than SGD with momentum on some vision tasks due to sharp minima.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why bias correction is needed</span></summary>

<p>Initialize $m_{0} = 0$. At step $k$:</p>

$$m_k = (1-\beta_1)\sum_{j=1}^k \beta_1^{k-j} g_j.$$

<p>Taking expectations (assuming stationary gradients $E[g_{j}] = g$):</p>

$$E[m_k] = g(1-\beta_1)\sum_{j=1}^k \beta_1^{k-j} = g(1-\beta_1^k).$$

<p>So $m_{k}$ underestimates the true gradient by a factor $(1-\beta_{1}^k)$, which is close to zero when $k$ is small and $\beta_{1}$ is close to 1. Dividing by $(1-\beta_{1}^k)$ removes this initialization bias. The same argument applies to $v_{k}$. After many steps $\beta_{1}^k \approx 0$ and the correction becomes negligible.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### AdamW

**Why weight decay is needed.** Without regularization, nothing prevents weights from growing arbitrarily large. Large weights make the model sensitive to small input changes, hurting generalization. $L_{2}$ regularization addresses this by adding a penalty $\frac{\lambda}{2}\lVert \theta\rVert^2$ to the loss, which pulls all weights toward zero. The intended effect is simple: at each step, shrink every weight uniformly by a small amount $\eta\lambda$, independently of the gradient. This is called **weight decay**.

**The problem with Adam + $L_{2}$.** The standard way to add $L_{2}$ is to include it in the loss, making the gradient $\nabla L(\theta) + \lambda\theta$. But in Adam, every gradient term (including $\lambda\theta$) gets absorbed into the moment estimates $m_{k}$ and $v_{k}$ and then adaptively scaled by $1/\sqrt{\hat v_{k,j}}$. The effective weight decay actually applied to parameter $j$ is

$$\text{effective decay}_j = \frac{\eta\lambda}{\sqrt{\hat v_{k,j}} + \epsilon},$$

which varies per parameter. Parameters with large historical gradients (large $\hat v_{k,j}$) receive small weight decay; parameters with small gradients receive large weight decay. This is the opposite of uniform shrinkage: the adaptive scaling unintentionally distorts what $L_{2}$ regularization is supposed to do.

AdamW (Loshchilov and Hutter, 2019) fixes this by decoupling weight decay from the gradient update entirely:

$$\theta_{k+1} = \theta_k - \frac{\eta}{\sqrt{\hat v_k}+\epsilon}\hat m_k - \eta\lambda\theta_k.$$

The second term applies uniform decay $\eta\lambda$ to every parameter regardless of $\hat v_{k}$. This is what $L_{2}$ regularization was supposed to do.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why the two formulations differ: the interaction between adaptive scaling and L2</span></summary>

<p>With L2 in the loss, Adam's gradient at step $k$ is $g_{k} = \nabla L(\theta_{k}) + \lambda\theta_{k}$. The second moment accumulates:</p>

$$v_{k,j} = \beta_2 v_{k-1,j} + (1-\beta_2)(\nabla_j L + \lambda\theta_{k,j})^2.$$

<p>Expanding the square: $(\nabla_{j} L)^2 + 2\lambda\theta_{k,j}\nabla_{j} L + \lambda^2\theta_{k,j}^2$. The cross term and $\lambda^2$ term mean $v_{k,j}$ is inflated differently for each parameter depending on the magnitude of $\theta_{k,j}$ relative to $\nabla_{j} L$. The resulting update</p>

$$\frac{\eta}{\sqrt{\hat v_{k,j}}+\epsilon}(\nabla_j L + \lambda\theta_{k,j})$$

<p>does not simplify to a clean gradient step plus a clean decay step. AdamW separates them: apply the Adam step to $\nabla L$ alone, then subtract $\eta\lambda\theta_{k}$ directly. The decay is now exactly $\eta\lambda$ per step, independent of the gradient history.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Pros.** Fixes Adam's broken $L_{2}$ regularization; uniform, predictable weight decay independent of gradient history; standard optimizer for LLMs and transformers; well-calibrated with $\beta_{2} = 0.95$ for long training runs.

**Cons.** Same theoretical issues as Adam (no convergence proof); requires careful tuning of $\lambda$, $\beta_{2}$, and gradient clipping threshold; slightly more hyperparameters than SGD.

**AdamW in LLM training.** AdamW is the standard optimizer for training transformers and large language models. Practical hyperparameter choices differ from the original Adam defaults:

- $\beta_{1} = 0.9$, $\beta_{2} = 0.95$ (not 0.999). The lower $\beta_{2}$ makes the second moment adapt faster to shifting gradient magnitudes during long LLM training runs. With $\beta_{2} = 0.999$, the moving average is slow to forget large gradients from early training, causing the effective learning rate to stay suppressed.
- Weight decay $\lambda = 0.1$, applied to all parameters except biases and layer norm scales.
- **Gradient clipping**: clip the global gradient norm to a threshold (typically 1.0) before the Adam update: $g_{k} \leftarrow g_{k} \cdot \min(1, c / \lVert g_{k}\rVert)$. This prevents exploding gradients from destabilizing the second moment estimates.
- **Warmup + cosine decay**: linear warmup for the first 1–2% of steps, then cosine decay to $\eta_{\min} \approx \eta_{\max}/10$.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why β₂ = 0.95 rather than 0.999 for LLMs</span></summary>

<p>The second moment estimate tracks a weighted average of past squared gradients with effective memory window $\approx 1/(1-\beta_{2})$ steps. With $\beta_{2} = 0.999$, the window is $\approx 1000$ steps; with $\beta_{2} = 0.95$, it is $\approx 20$ steps.</p>

<p>In LLM training, gradient magnitudes vary significantly across training phases: early on, the model is essentially random and gradients are large; as the model fits, gradients shrink and the loss landscape changes. A slow-moving $\hat v_{k}$ (large $\beta_{2}$) retains memory of large early gradients, keeping the effective learning rate small even when current gradients are modest. A faster-adapting $\hat v_{k}$ (smaller $\beta_{2}$) lets the learning rate recover more quickly as the training signal changes, which matters when training continues through multiple phases (e.g., pretraining, annealing, fine-tuning).</p>

<p>The tradeoff: smaller $\beta_{2}$ makes $\hat v_{k}$ noisy (high variance estimate of second moment), which can cause instability on individual parameter updates. Gradient clipping mitigates this by bounding the maximum update size.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Muon

Muon (Jordan, 2024) replaces Adam's per-element adaptive scaling with matrix-level orthogonalization. For a weight matrix $W \in \mathbb{R}^{m \times n}$, the update proceeds in three steps.

**Step 1: momentum accumulation**

$$M_k = \beta M_{k-1} + (1 - \beta) G_k,$$

where $G_k = \nabla_W L(\theta_k)$ is the gradient with respect to $W$ and $\beta = 0.95$ typically.

**Step 2: orthogonalization via Newton-Schulz iteration**

Compute the polar factor $U_k$ of $M_k$ (the nearest orthogonal matrix in spectral norm). Instead of an SVD ($O(p^3)$), Muon uses Newton-Schulz iterations:

$$X_0 = M_k / \|M_k\|_F, \qquad X_{i+1} = X_i\left(aI + bX_i^TX_i + c(X_i^TX_i)^2\right),$$

with coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$ chosen for fast convergence. After 5 iterations, $X_i$ converges to the polar factor $U_k$.

**Step 3: update**

$$W_{k+1} = W_k - \eta\, U_k.$$

The orthogonalized update $U_k$ has the property that all its singular values are 1, so the update applies equal magnitude in every singular direction of the gradient.

**Why orthogonalization helps.** Adam normalizes each scalar parameter independently: the effective update for element $(i,j)$ is $\hat{m}_{k,ij}/\sqrt{\hat{v}_{k,ij}}$, which is axis-aligned in parameter space. For a weight matrix, the natural geometry is the singular value decomposition, not the coordinate axes. Muon's polar factor normalizes along singular directions of the gradient matrix, ensuring that every direction of the weight matrix receives an update of equal spectral magnitude. This is steepest descent under the spectral norm rather than the Frobenius norm.

**Practical usage.** Muon applies only to weight matrices (2D parameters). Embeddings, biases, and normalization parameters are optimized with AdamW. A typical training loop uses Muon for all linear layer weights and AdamW for everything else.

**Hyperparameters.** Learning rate $\eta = 0.02$ (roughly $10\times$ larger than AdamW, since the update is already spectrally normalized), momentum $\beta = 0.95$, weight decay $\lambda = 0.0$ (Muon's spectral normalization provides implicit regularization), Newton-Schulz iterations $= 5$.

**Pros.** Better loss-per-step efficiency than AdamW on LLM pretraining; respects matrix geometry of weight parameters; Newton-Schulz iterations are pure matrix multiplications (GPU-friendly); lower optimizer memory (one momentum buffer vs two for Adam).

**Cons.** Only applies to 2D weight matrices (requires a separate optimizer for other parameters); less mature hyperparameter tuning guidance than AdamW; Newton-Schulz overhead adds compute per step; no convergence theory for nonconvex objectives.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why Newton-Schulz converges to the polar factor</span></summary>

<p>The polar decomposition of a matrix $M$ is $M = US$ where $U$ has orthonormal columns and $S = (M^TM)^{1/2}$ is symmetric positive semidefinite. The polar factor $U$ is the closest orthogonal matrix to $M$ in Frobenius norm.</p>

<p>Newton-Schulz iteration computes $U$ without an SVD. Starting from $X_0 = M/\|M\|_F$ (normalized so singular values lie in $(0, 1]$), each iteration applies a polynomial that pushes all singular values toward 1:</p>

$$X_{i+1} = X_i \cdot p(X_i^TX_i),$$

<p>where $p(s) = a + bs + cs^2$. If $X_i$ has singular values $\sigma_j^{(i)}$, then $X_{i+1}$ has singular values $\sigma_j^{(i+1)} = \sigma_j^{(i)} \cdot p((\sigma_j^{(i)})^2)$. The polynomial is designed so that the map $t \mapsto t \cdot p(t^2)$ is a contraction toward the fixed point $t = 1$ for all $t \in (0, 1]$. After convergence, all singular values equal 1, so $X_\infty = U$.</p>

<p>The standard Newton-Schulz polynomial is $p(s) = \frac{3}{2} - \frac{1}{2}s$ (the iteration $X_{i+1} = \frac{1}{2}X_i(3I - X_i^TX_i)$), which has quadratic convergence near the fixed point. The quintic polynomial $(a, b, c) = (3.4445, -4.7750, 2.0315)$ used in Muon converges faster per iteration at the cost of one extra matrix multiplication per step.</p>

<p>Each iteration requires two matrix multiplications ($X^TX$ and the polynomial evaluation), so the total cost for $k$ iterations is $O(km^2n)$. This is much cheaper than a full SVD when $k$ is small (5 iterations suffice in practice).</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Muon vs Adam: element-wise vs matrix-wise normalization</span></summary>

<p>Consider a weight matrix $W \in \mathbb{R}^{m \times n}$ with gradient $G$. The two optimizers apply fundamentally different normalizations:</p>

<p><strong>Adam.</strong> Maintains running means $m_{ij}$ and variances $v_{ij}$ for each element. The update for element $(i,j)$ is $\hat{m}_{ij}/\sqrt{\hat{v}_{ij}}$. This normalizes each coordinate independently, which is equivalent to preconditioning by a diagonal matrix. The normalization is axis-aligned: it does not account for correlations between different elements of $W$.</p>

<p><strong>Muon.</strong> Computes the polar factor $U$ of the (momentum-smoothed) gradient $M$. The update for the entire matrix is $U$, which has singular values all equal to 1. This normalizes along the singular directions of $M$, which captures the matrix structure of $W$. If $M = \sum_i \sigma_i u_i v_i^T$ is the SVD, Adam would scale each element of this sum differently based on coordinate-wise history, while Muon replaces all $\sigma_i$ with 1, preserving the directions $u_i, v_i$ exactly.</p>

<p>The spectral norm interpretation: Adam performs steepest descent where "distance" is measured element-wise (weighted $\ell_2$ norm with per-element weights). Muon performs steepest descent where "distance" is measured by the spectral norm (operator norm) of the update matrix. For weight matrices that transform between hidden dimensions, the spectral norm is the more natural measure of how much a parameter change affects the network's function.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Learning Rate Schedules

The learning rate $\eta$ is often the most important hyperparameter. Common schedules:

**Warmup.** Start with a small $\eta$ and increase linearly for the first $W$ steps, then switch to another schedule. Warmup stabilizes early training when parameter estimates are far from optimum and gradients are large and noisy. Used widely in transformer training.

**Cosine decay.** After warmup, anneal according to

$$\eta_k = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi k}{K}\right),$$

where $K$ is the total number of steps. This smoothly reduces $\eta$ to $\eta_{\min}$ while spending more time at intermediate rates than linear decay.

**Linear decay.** $\eta_{k} = \eta_{0}(1 - k/K)$. Simple and effective; used in many NLP fine-tuning recipes.

**Cyclic and restart schedules.** Periodically reset $\eta$ to $\eta_{\max}$ (cosine restarts). The idea is that saddle points and poor local minima may be escaped by occasional large steps. Empirically useful when the loss landscape has many nearby local minima.

## Proximal and Constrained Optimization

Many ML objectives are non-smooth (the $\ell_1$ penalty has a kink at zero) or constrained (SVM margins, norm budgets). This theme covers the foundational tools (subgradients, KKT conditions, duality) and the algorithms that build on them (proximal gradient, FISTA, ADMM).

### Subgradients

Gradient descent requires $L$ to be differentiable. But many functions in machine learning are not differentiable everywhere: $\lvert\theta\rvert$ has a kink at 0, the ReLU $\max(0,\theta)$ has a corner at 0, and the hinge loss $\max(0, 1-y\hat{y})$ has a non-smooth point at $y\hat{y}=1$.

A **subgradient** generalizes the gradient to convex functions that may not be differentiable. For a convex function $f:\mathbb{R}^p \to \mathbb{R}$, a vector $g$ is a subgradient of $f$ at $\theta$ if it defines a **supporting hyperplane** that lies below $f$ everywhere:

$$f(\phi) \ge f(\theta) + g^T(\phi - \theta), \qquad \forall\, \phi \in \mathbb{R}^p.$$

The set of all subgradients at $\theta$ is the **subdifferential**, denoted $\partial f(\theta)$. If $f$ is differentiable at $\theta$, the subdifferential is a singleton: $\partial f(\theta) = \lbrace\nabla f(\theta)\rbrace$.

**Example: $f(\theta) = \lvert\theta\rvert$ at $\theta = 0$.**

Any line through the origin with slope $g \in [-1, 1]$ lies below $\lvert\theta\rvert$. So $\partial\lvert\theta\rvert$ at $\theta = 0$ is the interval $[-1, 1]$. At $\theta > 0$, $\partial\lvert\theta\rvert = \lbrace 1 \rbrace$; at $\theta < 0$, $\partial\lvert\theta\rvert = \lbrace -1 \rbrace$. Combining:

$$\partial|\theta| = \begin{cases} \{1\} & \theta > 0 \\ [-1, 1] & \theta = 0 \\ \{-1\} & \theta < 0 \end{cases}$$

**Example: $f(\theta) = \lVert\theta\rVert_{1}$ in $\mathbb{R}^p$.**

The subdifferential separates by coordinate:

$$[\partial\lVert\theta\rVert_1]_j = \begin{cases} \{1\} & \theta_j > 0 \\ [-1, 1] & \theta_j = 0 \\ \{-1\} & \theta_j < 0 \end{cases}$$

This is exactly the set-valued "sign" function. The soft-thresholding rule for the lasso proximal operator is derived by solving $0 \in \partial f$ using this subdifferential (see the derivation in the Proximal Methods section below).

**Subgradient descent.** Replace $\nabla L$ with any $g_{k} \in \partial L(\theta_{k})$:

$$\theta_{k+1} = \theta_k - \eta_k\, g_k.$$

This converges for convex (not necessarily smooth) $f$ with a decaying step size $\eta_{k} = c/\sqrt{k}$, achieving

$$\min_{i \le k} f(\theta_i) - f(\theta^*) \le O(1/\sqrt{k}).$$

The rate is worse than gradient descent's $O(1/k)$ on smooth problems, and it is tight: $O(1/\sqrt{k})$ is the best possible for non-smooth convex optimization with first-order methods.

**Why subgradient descent is rarely used directly.** It is slow ($O(1/\sqrt{k})$), requires careful step-size tuning, and does not produce sparse solutions for $\ell_{1}$ problems (it zig-zags near kinks without landing on them). Proximal methods solve this: they handle the non-smooth part exactly via the proximal operator instead of linearizing it with a subgradient.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">How to choose the step size for subgradient descent</span></summary>

<p><strong>Standard diminishing rule: $\eta_{k} = c / \sqrt{k}$.</strong> This satisfies the two Robbins-Monro conditions ($\sum \eta_{k} = \infty$, $\sum \eta_{k}^2 < \infty$) needed for convergence. The constant $c$ controls the tradeoff between early speed and late accuracy. A common starting point is $c = 1 / \lVert g_{1}\rVert$, which normalizes the first step to unit length. If the loss oscillates wildly, reduce $c$ by a factor of 2-5; if progress is too slow, increase it.</p>

<p><strong>Constant step size $\eta_{k} = \eta$.</strong> Converges to within $O(\eta)$ of the optimum but never reaches it. Useful when you only need an approximate solution or plan to switch to a proximal method for the final refinement. Set $\eta$ to a target accuracy: if you want $f(\theta_{k}) - f^{\ast} \le \delta$, use $\eta \approx \delta / G^2$ where $G$ is a bound on the subgradient norm.</p>

<p><strong>Polyak step size: $\eta_{k} = (f(\theta_{k}) - f^{\ast}) / \lVert g_{k}\rVert^2$.</strong> If the optimal value $f^{\ast}$ is known (or a good lower bound is available, e.g., from a dual solution), this eliminates the need to tune $c$ entirely and converges at $O(1/k)$ for strongly convex problems. In practice, use a lower bound $f_{\text{best}}$ from the dual problem or a known target loss.</p>

<p><strong>Rule of thumb.</strong> If you find yourself tuning subgradient step sizes extensively, consider switching to a proximal method (ISTA/FISTA) instead. Proximal methods handle non-smoothness exactly and use a fixed step size $\eta = 1/L$ determined by the smooth part, removing the tuning problem entirely.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Subgradient calculus rules</span></summary>

<p>Subgradients obey analogues of the familiar gradient rules:</p>

<p><strong>Scaling.</strong> $\partial(\alpha f)(\theta) = \alpha\,\partial f(\theta)$ for $\alpha > 0$.</p>

<p><strong>Sum rule.</strong> $\partial(f + g)(\theta) \supseteq \partial f(\theta) + \partial g(\theta)$ (Minkowski sum). Equality holds when at least one of $f, g$ is differentiable at $\theta$, which is the common case in ML (smooth loss + non-smooth regularizer).</p>

<p><strong>Affine composition.</strong> If $h(\theta) = f(A\theta + b)$, then $\partial h(\theta) = A^T \partial f(A\theta + b)$.</p>

<p><strong>Pointwise maximum.</strong> For $f(\theta) = \max_{i} f_{i}(\theta)$ with each $f_{i}$ convex, $\partial f(\theta) = \mathrm{conv}\{\nabla f_{i}(\theta) : i \in I(\theta)\}$, where $I(\theta) = \{i : f_{i}(\theta) = f(\theta)\}$ is the set of active functions.</p>

<p>The pointwise maximum rule is especially useful: the hinge loss $\max(0, 1-y\hat{y})$ is the max of two linear functions, so its subdifferential at the kink is the convex hull of their gradients.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Optimality Conditions and KKT

For unconstrained smooth optimization, the necessary condition for a minimum is simply $\nabla f(\theta^{\ast}) = 0$. For constrained problems, the optimality condition is more involved.

**Unconstrained convex case.** If $f$ is convex (possibly non-smooth), the necessary and sufficient condition for $\theta^{\ast}$ to be a global minimizer is

$$0 \in \partial f(\theta^*).$$

This is the subgradient optimality condition. For smooth $f$, it reduces to $\nabla f(\theta^{\ast}) = 0$. For composite objectives $f = L + R$ where $L$ is smooth and $R$ is non-smooth, the condition becomes

$$0 \in \nabla L(\theta^*) + \partial R(\theta^*),$$

which is exactly the fixed-point equation that proximal gradient descent solves.

**Constrained optimization.** Consider the general problem

$$\min_\theta\; f(\theta) \quad \text{subject to} \quad g_i(\theta) \le 0,\; i=1,\dots,m, \qquad h_j(\theta) = 0,\; j=1,\dots,r.$$

The **Lagrangian** combines the objective and constraints with multipliers $\lambda_{i} \ge 0$ (for inequalities) and $\nu_{j}$ (for equalities):

$$\mathcal{L}(\theta, \lambda, \nu) = f(\theta) + \sum_{i=1}^m \lambda_i g_i(\theta) + \sum_{j=1}^r \nu_j h_j(\theta).$$

The multipliers $\lambda, \nu$ are called **dual variables**. Intuitively, $\lambda_{i}$ is the price of violating constraint $i$: at the optimum, the cost of tightening an active constraint equals the marginal improvement in the objective.

**The KKT conditions.** At a constrained optimum $\theta^{\ast}$ (under mild regularity, e.g., Slater's condition for convex problems), there exist multipliers $\lambda^{\ast}, \nu^{\ast}$ satisfying:

1. **Stationarity:** $\nabla_\theta \mathcal{L}(\theta^{\ast}, \lambda^{\ast}, \nu^{\ast}) = 0$, i.e.,

   $$\nabla f(\theta^*) + \sum_i \lambda_i^* \nabla g_i(\theta^*) + \sum_j \nu_j^* \nabla h_j(\theta^*) = 0.$$

2. **Primal feasibility:** $g_{i}(\theta^{\ast}) \le 0$ for all $i$, and $h_{j}(\theta^{\ast}) = 0$ for all $j$.

3. **Dual feasibility:** $\lambda_{i}^{\ast} \ge 0$ for all $i$.

4. **Complementary slackness:** $\lambda_{i}^{\ast} g_{i}(\theta^{\ast}) = 0$ for all $i$.

Complementary slackness says: either constraint $i$ is active ($g_{i}(\theta^{\ast}) = 0$) or its multiplier is zero ($\lambda_{i}^{\ast} = 0$). Inactive constraints do not affect the solution.

For **convex** problems ($f, g_{i}$ convex, $h_{j}$ affine), KKT conditions are both necessary and sufficient for global optimality.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Example: KKT for ridge regression (constrained form)</span></summary>

<p>Ridge regression can be written in constrained form:</p>

$$\min_\theta\; \frac{1}{2}\|y - X\theta\|^2 \quad \text{subject to} \quad \|\theta\|^2 \le t.$$

<p>This has one inequality constraint $g(\theta) = \lVert \theta\rVert^2 - t \le 0$, no equality constraints. The Lagrangian is:</p>

$$\mathcal{L}(\theta, \lambda) = \frac{1}{2}\|y - X\theta\|^2 + \lambda(\|\theta\|^2 - t).$$

<p><strong>Stationarity:</strong> $\nabla_\theta \mathcal{L} = -X^T(y-X\theta) + 2\lambda\theta = 0$, giving $\theta = (X^TX + 2\lambda I)^{-1}X^Ty$.</p>

<p>This is the ridge solution with penalty $2\lambda$. The KKT multiplier $\lambda$ plays exactly the role of the ridge penalty parameter.</p>

<p><strong>Complementary slackness:</strong> $\lambda(\lVert \theta\rVert^2 - t) = 0$. Either $\lambda = 0$ (constraint inactive, OLS solution fits inside the ball) or $\lVert \theta\rVert^2 = t$ (constraint active, solution lies on the ball boundary). As $t \to 0$ the constraint tightens and $\lambda \to \infty$; as $t \to \infty$ the constraint becomes slack and $\lambda = 0$. This is the precise sense in which the penalized form $\frac{1}{2}\lVert y-X\theta\rVert^2 + \lambda\lVert \theta\rVert^2$ and the constrained form are equivalent.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Example: KKT for SVM</span></summary>

<p>The hard-margin SVM solves</p>

$$\min_{w,b}\; \frac{1}{2}\|w\|^2 \quad \text{subject to} \quad y_i(w^Tx_i + b) \ge 1, \;\; i=1,\dots,n.$$

<p>Rewriting constraints as $g_{i}(w,b) = 1 - y_{i}(w^Tx_{i} + b) \le 0$, the Lagrangian is:</p>

$$\mathcal{L}(w,b,\alpha) = \frac{1}{2}\|w\|^2 - \sum_i \alpha_i[y_i(w^Tx_i+b) - 1].$$

<p><strong>Stationarity (w.r.t. $w$):</strong> $w = \sum_{i} \alpha_{i} y_{i} x_{i}$. The optimal hyperplane is a linear combination of training points.</p>

<p><strong>Stationarity (w.r.t. $b$):</strong> $\sum_{i} \alpha_{i} y_{i} = 0$.</p>

<p><strong>Complementary slackness:</strong> $\alpha_{i}[y_{i}(w^Tx_{i}+b) - 1] = 0$. Either $\alpha_{i} = 0$ (point is not a support vector) or $y_{i}(w^Tx_{i}+b) = 1$ (point lies exactly on the margin). Only support vectors contribute to $w$. This is why SVM solutions are sparse in the dual: most $\alpha_{i} = 0$.</p>

<p>Substituting the stationarity conditions back into $\mathcal{L}$ gives the dual problem, which is a QP in $\alpha$ and the basis of the kernel trick.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Where KKT appears in this post.** The ADMM algorithm below solves constrained problems $\min L(\theta) + R(z)$ subject to $\theta = z$ by forming the augmented Lagrangian $\mathcal{L}(\theta, z, u) = L(\theta) + R(z) + u^T(\theta - z) + \frac{\rho}{2}\lVert\theta - z\rVert^2$, where $u$ is the dual variable (Lagrange multiplier for the equality constraint). ADMM alternates between minimizing $\mathcal{L}$ over $\theta$, then $z$, then updating $u$. At convergence, the KKT conditions for the constrained problem are satisfied: stationarity gives the primal updates, and the dual update enforces feasibility.

### Primal and Dual Problems

The **primal problem** is the original constrained optimization problem you wrote down:

$$p^* = \min_\theta\; f(\theta) \quad \text{subject to} \quad g_i(\theta) \le 0,\; h_j(\theta) = 0.$$

The **dual problem** is a different optimization problem, derived from the Lagrangian, that attacks the same question from the opposite direction. Instead of minimizing over $\theta$ while respecting constraints, you maximize over the multipliers $\lambda, \nu$ while keeping $\theta$ free:

$$d^* = \max_{\lambda \ge 0,\, \nu}\; \underbrace{\min_\theta\; \mathcal{L}(\theta, \lambda, \nu)}_{q(\lambda, \nu)}.$$

The inner minimization $q(\lambda, \nu) = \min_\theta \mathcal{L}(\theta, \lambda, \nu)$ is called the **dual function**. The outer problem maximizes $q$ over the multipliers.

**What does this actually mean?** Think of it as a two-player game.

- **Primal player** (minimizer) chooses $\theta$ to make $f(\theta)$ small while satisfying constraints.
- **Dual player** (maximizer) chooses multipliers $\lambda_{i} \ge 0$ to penalize constraint violations as harshly as possible.

The Lagrangian $\mathcal{L}(\theta, \lambda, \nu) = f(\theta) + \sum_{i} \lambda_{i} g_{i}(\theta) + \sum_{j} \nu_{j} h_{j}(\theta)$ is the payoff: the primal player wants it small, the dual player wants it large. If $\theta$ violates a constraint ($g_{i}(\theta) > 0$), the dual player can increase $\lambda_{i}$ to make the payoff arbitrarily large, punishing the violation. If $\theta$ is feasible ($g_{i}(\theta) \le 0$), the dual player's best response is $\lambda_{i} = 0$ (no punishment needed), and the Lagrangian reduces to $f(\theta)$.

The dual function $q(\lambda, \nu)$ asks: "if the primal player responds optimally to these multipliers, what is the resulting payoff?" Since the primal player minimizes, $q$ gives a **lower bound** on the primal optimal value for any feasible $\lambda, \nu$:

$$q(\lambda, \nu) \le p^* \qquad \text{for all } \lambda \ge 0.$$

This is **weak duality**, and it always holds (even for nonconvex problems). The dual problem $\max q(\lambda,\nu)$ finds the **tightest** lower bound.

**Strong duality** says the bound is tight: $d^{\ast} = p^{\ast}$. For convex problems (convex $f$, convex $g_{i}$, affine $h_{j}$) with a feasibility condition (Slater's: there exists a strictly feasible point), strong duality holds. The primal and dual achieve the same optimal value, and the optimal multipliers are exactly the KKT multipliers.

The **duality gap** $p^{\ast} - d^{\ast}$ measures how tight the bound is. Under strong duality, the gap is zero. In algorithms, a small duality gap serves as a stopping criterion: if primal and dual objectives are close, you are near optimal.

**Why solve the dual instead of the primal?** Three practical reasons:

1. **The dual may be easier.** The dual function $q(\lambda,\nu)$ is always concave (even if the primal is nonconvex), so the dual is always a concave maximization problem. Sometimes the primal has $p$ variables with $n$ constraints but the dual has only $n$ variables, which is cheaper when $n \ll p$.

2. **The dual reveals structure.** The SVM dual (see KKT example above) replaces the primal over $(w,b)$ with a QP over $\alpha_{i}$. The optimal $w = \sum_{i} \alpha_{i} y_{i} x_{i}$ depends on data only through inner products $x_{i}^T x_{j}$, which is what makes the kernel trick possible: replace $x_{i}^T x_{j}$ with $K(x_{i}, x_{j})$ to operate in high-dimensional feature space without ever computing the features.

3. **Lower bounds for free.** Any dual-feasible point gives a valid lower bound on the primal. This is useful for branch-and-bound algorithms and for certifying how far a current solution is from optimal.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Worked example: primal and dual for the lasso</span></summary>

<p>The lasso in constrained form is:</p>

$$\text{(Primal)} \quad \min_\theta\; \frac{1}{2}\|y - X\theta\|^2 \quad \text{subject to} \quad \|\theta\|_1 \le t.$$

<p>Introduce a multiplier $\lambda \ge 0$ for the constraint $\lVert \theta\rVert_{1} - t \le 0$. The Lagrangian is:</p>

$$\mathcal{L}(\theta, \lambda) = \frac{1}{2}\|y - X\theta\|^2 + \lambda(\|\theta\|_1 - t).$$

<p>This is exactly the penalized form $\frac{1}{2}\lVert y - X\theta\rVert^2 + \lambda\lVert \theta\rVert_{1}$ minus the constant $\lambda t$. So finding the right $\lambda$ in the penalized form is equivalent to solving the dual of the constrained form.</p>

<p>The dual function is $q(\lambda) = \min_\theta \mathcal{L}(\theta,\lambda)$. The inner minimization is the lasso problem with penalty $\lambda$, which gives $\hat\theta(\lambda)$. Then:</p>

$$q(\lambda) = \frac{1}{2}\|y - X\hat\theta(\lambda)\|^2 + \lambda\|\hat\theta(\lambda)\|_1 - \lambda t.$$

<p>The dual problem $\max_{\lambda \ge 0} q(\lambda)$ finds the penalty parameter $\lambda^{\ast}$ such that the constraint $\lVert \hat\theta(\lambda^{\ast})\rVert_{1} = t$ is exactly satisfied. Strong duality holds (the problem is convex and Slater's condition is easily checked), so the penalized solution with $\lambda^{\ast}$ is exactly the constrained solution with budget $t$.</p>

<p>This is the precise connection between "choose penalty $\lambda$" and "choose budget $t$": they are primal-dual pairs of the same problem.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Worked example: SVM dual and the kernel trick</span></summary>

<p>The hard-margin SVM primal is (from the KKT example above):</p>

$$\text{(Primal)} \quad \min_{w,b}\; \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^Tx_i+b) \ge 1.$$

<p>From the KKT stationarity conditions we found $w = \sum_{i} \alpha_{i} y_{i} x_{i}$ and $\sum_{i} \alpha_{i} y_{i} = 0$. Substituting these back into the Lagrangian eliminates $w$ and $b$, giving:</p>

$$\text{(Dual)} \quad \max_\alpha\; \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j\, x_i^T x_j \quad \text{s.t.} \quad \alpha_i \ge 0,\; \sum_i \alpha_i y_i = 0.$$

<p><strong>Primal:</strong> $p$ variables (dimension of $w$) with $n$ constraints. <strong>Dual:</strong> $n$ variables ($\alpha_{i}$) with simple bound constraints. When $p \gg n$, the dual is much smaller.</p>

<p>The dual depends on the data only through inner products $x_{i}^Tx_{j}$. Replace these with a kernel $K(x_{i}, x_{j}) = \phi(x_{i})^T\phi(x_{j})$ and you can find the optimal separating hyperplane in an infinite-dimensional feature space $\phi$ without ever computing $\phi(x_{i})$. This is the kernel trick, and it is only possible because duality gave us a formulation in terms of inner products.</p>

<p>Strong duality holds (Slater's condition is satisfied whenever the data is separable), so $p^{\ast} = d^{\ast}$: the primal and dual achieve the same margin.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

With the foundations in place, we now turn to the algorithms that solve non-smooth and composite problems. Proximal methods handle objectives of the form $\min_\theta L(\theta) + R(\theta)$, where $L$ is smooth (gradient available) and $R$ is convex but possibly non-smooth.

### Proximal Operator

The proximal operator of $R$ at scale $\eta$ is

$$\mathrm{prox}_{\eta R}(v) = \arg\min_\theta\left\{\eta R(\theta) + \frac{1}{2}\|\theta - v\|^2\right\}.$$

It finds the point closest to $v$ that also has small $R(\theta)$. The $\frac{1}{2}\lVert \theta - v\rVert^2$ term keeps the solution near $v$; $\eta$ controls the tradeoff.

**For $L_{1}$ regularization** ($R(\theta) = \lVert \theta\rVert_{1}$), the proximal operator is soft-thresholding applied elementwise:

$$[\mathrm{prox}_{\eta\|\cdot\|_1}(v)]_j = \mathcal{S}_\eta(v_j) = \mathrm{sign}(v_j)\max(|v_j| - \eta, 0).$$

Coordinates smaller than $\eta$ in magnitude are zeroed out; larger coordinates are shrunk by $\eta$. This is exactly the lasso solution under orthonormal design.

**For $L_{2}$ regularization** ($R(\theta) = \frac{\lambda}{2}\lVert \theta\rVert^2$), the proximal operator is ridge shrinkage:

$$\mathrm{prox}_{\eta\frac{\lambda}{2}\|\cdot\|^2}(v) = \frac{v}{1+\eta\lambda}.$$

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Derivation of the soft-thresholding proximal operator</span></summary>

<p>The problem separates by coordinate: for each $j$, minimize</p>

$$f(t) = \eta|t| + \frac{1}{2}(t - v_j)^2.$$

<p>Taking the subdifferential and setting it to zero: $0 \in \eta\,\partial\lvert t\rvert + (t - v_{j})$, i.e., $v_{j} - t \in \eta\,\partial\lvert t\rvert$.</p>

<p><strong>Case 1: $t > 0$.</strong> $\partial\lvert t\rvert = \{1\}$, so $v_{j} - t = \eta$, giving $t = v_{j} - \eta$. Valid only if $v_{j} - \eta > 0$, i.e., $v_{j} > \eta$.</p>

<p><strong>Case 2: $t < 0$.</strong> $\partial\lvert t\rvert = \{-1\}$, so $v_{j} - t = -\eta$, giving $t = v_{j} + \eta$. Valid only if $v_{j} + \eta < 0$, i.e., $v_{j} < -\eta$.</p>

<p><strong>Case 3: $t = 0$.</strong> $\partial\lvert t\rvert = [-1,1]$, so need $v_{j} \in [-\eta, \eta]$. Valid when $\lvert v_{j}\rvert \le \eta$.</p>

<p>Combining: $t^{\ast} = \mathrm{sign}(v_{j})\max(\lvert v_{j}\rvert-\eta, 0)$.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Proximal Gradient (ISTA)

The proximal gradient algorithm alternates a gradient step on $L$ with a proximal step on $R$:

$$\theta_{k+1} = \mathrm{prox}_{\eta R}\!\left(\theta_k - \eta\nabla L(\theta_k)\right).$$

**Convergence.** Under $L$-smoothness of $L$ and convexity of both $L$ and $R$, with $\eta = 1/L$:

$$L(\theta_k) + R(\theta_k) - (L(\theta^*) + R(\theta^*)) \le \frac{\|\theta_0 - \theta^*\|^2}{2\eta k} = O(1/k).$$

**Pros.** Handles non-smooth regularizers exactly; $L_{1}$ gives exact sparse solutions via soft-thresholding; clean convergence guarantee; each step is cheap when the proximal operator has a closed form.

**Cons.** Requires full-batch gradients for convergence guarantee; $O(1/k)$ rate is the same as plain GD; slow without acceleration (use FISTA instead in practice).

### FISTA

Apply Nesterov acceleration to proximal gradient:

$$\theta_{k+1} = \mathrm{prox}_{\eta R}(y_k - \eta\nabla L(y_k)), \qquad y_{k+1} = \theta_{k+1} + \frac{k}{k+3}(\theta_{k+1}-\theta_k).$$

This achieves $O(1/k^2)$, optimal for composite convex problems. FISTA is the standard algorithm for lasso, group lasso, and nuclear norm minimization.

**Pros.** Optimal $O(1/k^2)$ rate for composite convex problems; drop-in replacement for ISTA; standard for lasso and sparse recovery.

**Cons.** Full-batch only; occasional restart heuristics needed in practice for nonconvex problems; momentum coefficient sensitive to problem conditioning.

### ADMM

The Alternating Direction Method of Multipliers solves constrained or consensus problems by splitting variables:

$$\min_{\theta,z}\; L(\theta) + R(z) \quad \text{subject to } \theta = z.$$

ADMM alternates: update $\theta$ (gradient step on augmented Lagrangian), update $z$ (proximal step on $R$), update dual variable $u$. ADMM is useful when $\theta$ and $z$ involve different structures (e.g., group sparsity, matrix completion).

**Pros.** Handles complex constraints and consensus problems; each subproblem often has a closed form; distributed-computing friendly (each node solves a local subproblem).

**Cons.** Linear convergence rate at best; many hyperparameters (augmented Lagrangian penalty, step sizes); convergence theory requires convexity; sensitive to penalty parameter tuning.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">How to choose the ADMM penalty parameter in practice</span></summary>

<p><strong>Augmented Lagrangian penalty $\rho$.</strong> This is the most important ADMM hyperparameter. It controls the tradeoff between primal feasibility (how fast $\theta \to z$) and dual convergence. Too small: slow enforcement of the constraint $\theta = z$, so primal residuals stay large. Too large: the $\frac{\rho}{2}\lVert \theta - z\rVert^2$ term dominates and the subproblems become ill-conditioned. Start with $\rho = 1.0$. If the primal residual $\lVert \theta_{k} - z_{k}\rVert$ is much larger than the dual residual $\lVert \rho(z_{k} - z_{k-1})\rVert$, increase $\rho$ by $2\times$; if the dual residual dominates, decrease $\rho$ by $2\times$. This is the adaptive $\rho$ scheme of Boyd et al. (2011) and is the default in most implementations.</p>

<p><strong>Stopping criteria.</strong> Monitor both primal residual $r_{k} = \lVert \theta_{k} - z_{k}\rVert$ and dual residual $s_{k} = \lVert \rho(z_{k} - z_{k-1})\rVert$. Stop when both fall below a tolerance $\epsilon^{\text{abs}} + \epsilon^{\text{rel}} \cdot \max(\lVert \theta_{k}\rVert, \lVert z_{k}\rVert)$. Typical defaults: $\epsilon^{\text{abs}} = 10^{-4}$, $\epsilon^{\text{rel}} = 10^{-3}$.</p>

<p><strong>Software.</strong> CVXPY's SCS solver uses ADMM internally with adaptive $\rho$. For custom implementations in Python, the <code>admm</code> examples in the Boyd et al. companion code (<code>stanford.edu/~boyd/admm.html</code>) provide well-tested reference implementations for lasso, consensus, and distributed problems.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Second-Order Methods

Gradient descent uses the gradient $\nabla L$ (first-order information). Newton's method uses the Hessian $\nabla^2 L$ to scale steps by local curvature:

$$\theta_{k+1} = \theta_k - [\nabla^2 L(\theta_k)]^{-1} \nabla L(\theta_k).$$

Near a solution, Newton converges quadratically. The cost is computing and inverting the $p \times p$ Hessian, which is $O(p^3)$ per step and infeasible for large neural networks ($p \sim 10^8$).

**Pros of Newton's method.** Q-quadratic convergence near a solution; handles ill-conditioned problems naturally; no learning rate tuning needed.

**Cons of Newton's method.** $O(p^3)$ per step; $O(p^2)$ memory for Hessian; requires exact Hessian (not available in stochastic setting); impractical beyond $p \sim 10^4$.

**Conjugate Gradient (CG).** For quadratic objectives $L(\theta) = \frac{1}{2}\theta^T A\theta - b^T\theta$ with $A \succ 0$, CG finds the exact solution in at most $p$ steps without forming or inverting $A$. At each step it chooses a search direction conjugate to all previous directions under $A$-inner product ($d_{i}^T A d_{j} = 0$ for $i \ne j$), which guarantees no progress is undone. Convergence rate depends on the condition number: $\lVert \theta_{k} - \theta^{\ast}\rVert_{A} \le 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k \lVert \theta_{0} - \theta^{\ast}\rVert_{A}$.

For nonlinear objectives, Nonlinear CG (Fletcher-Reeves, Polak-Ribière) generalizes the update direction using the current gradient, with periodic restarts. CG requires no matrix storage beyond the current gradient and direction, using only $O(p)$ memory.

**Hessian-free optimization** uses CG to solve the Newton system $\nabla^2 L(\theta)\,d = -\nabla L(\theta)$ without forming $\nabla^2 L$. The key: CG only needs matrix-vector products $\nabla^2 L(\theta)\,v$, which can be computed via finite differences of gradients, $(\nabla L(\theta + \epsilon v) - \nabla L(\theta))/\epsilon$, at cost $O(p)$ per product. This makes second-order methods tractable for moderately large networks.

**Pros of CG.** Exact solution for quadratics in $p$ steps; $O(p)$ memory; enables Hessian-free second-order methods via cheap $Hv$ products.

**Cons of CG.** Exact only for quadratics; requires restarts for nonlinear objectives; sensitive to preconditioning; less commonly used now that AdamW dominates in practice.

**Quasi-Newton methods (L-BFGS)** approximate the Hessian inverse using the history of gradient differences, requiring only $O(mp)$ memory for $m$ stored vectors. L-BFGS is the standard choice for small-to-medium scale problems where full-batch gradients are available (e.g., logistic regression, shallow networks).

**Pros of L-BFGS.** Superlinear convergence; handles ill-conditioning well without forming the full Hessian; $O(mp)$ memory with $m$ typically 5–20; gold standard for full-batch convex problems.

**Cons of L-BFGS.** Requires full-batch gradients (incompatible with SGD); $O(mp)$ memory and overhead still too large for $p \sim 10^8$; convergence degrades with noisy gradients.

**Natural gradient.** Replace the Euclidean gradient with the Riemannian gradient under the Fisher information metric:

$$\theta_{k+1} = \theta_k - \eta F(\theta_k)^{-1} \nabla L(\theta_k),$$

where $F(\theta) = E[\nabla \log p(y\lvert x,\theta)\nabla \log p(y\rvertx,\theta)^T]$ is the Fisher matrix. This is equivalent to steepest descent in distribution space (KL divergence) rather than parameter space. K-FAC and Shampoo are practical approximations used in large-scale training.

**Pros of natural gradient.** Invariant to reparameterization; fast convergence near a solution; theoretically optimal for probabilistic models.

**Cons of natural gradient.** Fisher matrix is $O(p^2)$ to store and $O(p^3)$ to invert; K-FAC and Shampoo approximations are complex to implement and tune; communication-heavy in distributed training.

## Loss Landscape and Generalization

Deep networks are nonconvex. The classical worry was local minima, but empirically, large networks rarely get stuck. Two more relevant phenomena:

**Saddle points.** A point where $\nabla L = 0$ but the Hessian has both positive and negative eigenvalues. In high dimensions, most critical points are saddle points (the fraction with all positive eigenvalues decays exponentially in $p$). SGD noise helps escape saddle points; gradient descent can get stuck.

**Flat regions and sharpness.** The loss can be flat (near-zero gradients) over large regions, especially early in training. Separately, the sharpness of the final minimum (the largest Hessian eigenvalue $\lambda_{\max}(\nabla^2 L(\theta^{\ast}))$) correlates strongly with generalization: flatter minima generalize better. This motivates methods that explicitly seek flat minima.

**SAM (Sharpness-Aware Minimization).** Instead of minimizing $L(\theta)$, SAM minimizes the worst-case loss in a neighborhood:

$$\min_\theta \max_{\|\epsilon\|\le\rho} L(\theta + \epsilon).$$

The inner maximization is approximated by one gradient ascent step: $\hat\epsilon = \rho \nabla L(\theta) / \lVert \nabla L(\theta)\rVert$. The outer step then follows $\nabla L(\theta + \hat\epsilon)$. SAM adds one extra forward-backward pass per step but consistently improves generalization in image and language tasks.

**Pros.** Consistently improves generalization by finding flatter minima; can be paired with any base optimizer (SGD, AdamW); single extra hyperparameter $\rho$.

**Cons.** Exactly $2\times$ compute per step (two forward-backward passes); no convergence theory for nonconvex objectives; $\rho$ requires tuning; less commonly used in LLM training where compute is the binding constraint.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">How to choose the SAM neighborhood radius in practice</span></summary>

<p><strong>Neighborhood radius $\rho$.</strong> This is SAM's only extra hyperparameter. It controls how aggressively you seek flat minima: larger $\rho$ penalizes sharper regions more heavily but makes optimization harder. Start with $\rho = 0.05$ for SGD-based SAM or $\rho = 0.01$ for AdamW-based SAM (adaptive methods already partially normalize gradient scale, so a smaller perturbation suffices). The original SAM paper (Foret et al., 2021) used $\rho = 0.05$ with SGD on image classification.</p>

<p><strong>Tuning $\rho$.</strong> Treat it like a regularization strength: increase $\rho$ if the model overfits (training loss low, validation loss high) and decrease it if underfitting or if training becomes unstable. A grid search over $\{0.01, 0.02, 0.05, 0.1, 0.2\}$ is usually sufficient. SAM is relatively insensitive to $\rho$ within a factor of 2-3 of the optimal value.</p>

<p><strong>Reducing the compute overhead.</strong> The $2\times$ compute cost can be halved with ESAM (efficient SAM), which applies the perturbation only to a random subset of parameters, or LookSAM, which reuses the perturbation direction for multiple steps. In PyTorch, the <code>timm</code> library provides a well-tested SAM implementation: wrap your base optimizer with <code>timm.optim.SAM(base_optimizer, rho=0.05)</code>.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Practical Tricks for Deep Learning and LLM Training

The theory above gives convergence guarantees under idealized assumptions. In practice, training deep networks and LLMs requires a collection of engineering tricks to stabilize training, reduce memory, and improve generalization.

### Gradient Clipping

Before each optimizer step, rescale the gradient if its global norm exceeds a threshold $c$:

$$g_k \leftarrow g_k \cdot \min\!\left(1,\; \frac{c}{\|g_k\|_2}\right).$$

This leaves small gradients unchanged and shrinks large gradients proportionally, preventing a single bad batch from destabilizing the optimizer state. In LLM training with AdamW, $c = 1.0$ is standard. Without clipping, a spike in gradient magnitude inflates $v_{k}$ in Adam, suppressing the effective learning rate for many subsequent steps.

### Mixed Precision Training (FP16 / BF16)
{:#mixed-precision-training}

Store activations and gradients in 16-bit floating point to halve memory and double throughput on modern hardware, while keeping a **master copy of weights in FP32** for accurate parameter updates.

- **FP16** (IEEE half): 5 exponent bits, 10 mantissa bits. Dynamic range $\approx [6\times10^{-5},\, 65504]$. Gradients near zero can **underflow** to exactly 0.
- **BF16** (brain float): 8 exponent bits, 7 mantissa bits. Same dynamic range as FP32 ($\approx 10^{-38}$ to $10^{38}$) but lower precision. Preferred for LLMs because gradient underflow is rarely a problem.

**Loss scaling** (needed for FP16, not BF16): multiply the loss by a large scalar $S$ before backprop so gradients are scaled up into the representable range; divide the accumulated gradients by $S$ before the optimizer step. Dynamic loss scaling starts with $S = 2^{15}$ and halves it on overflow, doubles it every 2000 steps without overflow.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why master weights in FP32 are necessary</span></summary>

<p>The AdamW update subtracts a small increment $\Delta\theta = \eta \hat m_{k} / (\sqrt{\hat v_{k}}+\epsilon)$ from $\theta$. For a typical learning rate $\eta = 10^{-4}$ and normalized gradient, $\lvert \Delta\theta_{j}\rvert$ can be $O(10^{-5})$ or smaller. In FP16, the smallest representable difference for a weight of magnitude $O(1)$ is $\approx 10^{-3}$ (limited by the 10-bit mantissa). Any update smaller than this is rounded to zero, so the weight never changes. Storing master weights in FP32 (machine epsilon $\approx 10^{-7}$) ensures small updates accumulate correctly. The FP16 copies are used only for forward and backward passes.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Gradient Accumulation

Simulate a large effective batch size $B_{\text{eff}} = B \times K$ by running $K$ forward-backward passes on mini-batches of size $B$ before each optimizer step, accumulating (summing) gradients:

$$g_{\text{acc}} \leftarrow g_{\text{acc}} + \frac{1}{K}\nabla L_{\text{mini}}(\theta), \qquad \text{update } \theta \text{ after } K \text{ steps.}$$

This is equivalent to a single step on a batch of size $B_{\text{eff}}$ when the loss is a mean over samples, at $K\times$ lower peak memory. Used in LLM training when $B_{\text{eff}}$ is in the millions of tokens but a single GPU holds only $B$ tokens.

### Batch Size and Learning Rate Scaling

When increasing batch size by factor $k$, the variance of the mini-batch gradient decreases by $1/k$, so a larger learning rate can be used. Two common rules:

- **Linear scaling rule** (Goyal et al., 2017): $\eta \leftarrow k\eta$. Keeps the expected update magnitude constant. Works well for large-batch SGD with short warmup.
- **Square-root scaling**: $\eta \leftarrow \sqrt{k}\eta$. More conservative; often preferred for Adam since the second-moment normalization already partially accounts for gradient variance.

Neither rule holds perfectly at very large batch sizes (where noise in the gradient is too small to help generalization), so diminishing returns set in beyond a critical batch size.

### Parameter Initialization

A network with poorly initialized weights will have gradients that explode or vanish before any useful learning occurs. The goal is to keep the variance of activations and gradients $O(1)$ at initialization.

**Xavier / Glorot initialization** (for tanh, sigmoid): $W \sim \mathcal{U}\!\left[-\sqrt{6/(n_{\text{in}}+n_{\text{out}})},\, \sqrt{6/(n_{\text{in}}+n_{\text{out}})}\right]$. Derived by requiring $\mathrm{Var}(\text{output}) = \mathrm{Var}(\text{input})$ for a linear activation.

**He / Kaiming initialization** (for ReLU): $W \sim \mathcal{N}(0, 2/n_{\text{in}})$. The factor of 2 compensates for ReLU zeroing half the neurons: the effective fan-in is $n_{\text{in}}/2$.

**Residual networks** require additional care: initialize the final layer of each residual block to zero so the block acts as identity at initialization. This keeps gradients well-scaled even for very deep networks (GPT uses this for the output projection of each transformer block, scaled by $1/\sqrt{N_{\text{layers}}}$).

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Variance propagation derivation for He initialization</span></summary>

<p>Consider a layer $h = W x$ where $W \in \mathbb{R}^{m\times n}$, $x \in \mathbb{R}^n$. Assume $W_{ij} \sim (0, \sigma^2)$ iid, $x_{i} \sim (0, \sigma_{x}^2)$ iid, and $W \perp x$. Then</p>

$$\mathrm{Var}(h_i) = \sum_j \mathrm{Var}(W_{ij} x_j) = n\sigma^2\sigma_x^2.$$

<p>To preserve variance ($\mathrm{Var}(h_{i}) = \sigma_{x}^2$), set $\sigma^2 = 1/n$. For ReLU, the output has roughly half its entries zeroed, halving the variance. To compensate, set $\sigma^2 = 2/n$. He initialization applies this correction.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Gradient Checkpointing

Backpropagation requires storing all intermediate activations from the forward pass to compute gradients. For a network with $L$ layers, this is $O(L)$ memory. Gradient checkpointing reduces this to $O(\sqrt{L})$ by:

1. During the forward pass, only save activations at $\sqrt{L}$ checkpoint layers; discard the rest.
2. During the backward pass, when a non-checkpointed activation is needed, recompute it from the nearest checkpoint (one additional forward pass per segment).

The tradeoff: $O(\sqrt{L})$ memory at the cost of $\approx 33\%$ extra compute. In transformer training, checkpointing is typically applied per-layer: save activations only at the boundary of each transformer block.

### Optimizer State Sharding (ZeRO)

In data-parallel training across $G$ GPUs, the naive approach replicates full optimizer state (AdamW stores $m_{k}, v_{k}, \theta$: three copies of the model) on every GPU. For a 70B parameter model in BF16, $m_{k}$ and $v_{k}$ in FP32 alone require $\approx 560$ GB.

ZeRO (Zero Redundancy Optimizer, Rajbhandari et al., 2020) partitions:
- **Stage 1**: optimizer states across $G$ GPUs. Each GPU holds $1/G$ of $m_{k}, v_{k}$.
- **Stage 2**: also partition gradients.
- **Stage 3**: also partition model parameters.

Stage 3 reduces per-GPU memory by $3G\times$ at the cost of extra all-gather communication to reconstruct parameters when needed. ZeRO is implemented in DeepSpeed and is standard for LLM pretraining.

### Pre-norm vs Post-norm

Every transformer layer applies two sublayers (attention and feed-forward) with a residual connection and LayerNorm. The question is where LayerNorm goes relative to the sublayer.

**Post-norm** (original transformer, Vaswani et al., 2017) normalizes after the residual add:

$$h \leftarrow \mathrm{LN}(h + \mathrm{Sublayer}(h))$$

**Pre-norm** (GPT-2 onward, LLaMA, PaLM) normalizes before the sublayer:

$$h \leftarrow h + \mathrm{Sublayer}(\mathrm{LN}(h))$$

The residual stream $h$ is left unnormalized in pre-norm; only the input to each sublayer is normalized.

**Why pre-norm trains more stably.** The key difference is how gradients travel backward through $L$ layers. In post-norm, every backward pass through a layer crosses a LayerNorm, which rescales by $1/\hat{\sigma}$ where $\hat{\sigma}$ is the standard deviation of the pre-norm activations. If $\hat{\sigma}$ is large (which happens as the network grows during training), this factor shrinks gradients layer by layer, causing vanishing gradients.

In pre-norm the Jacobian of layer $l$ with respect to $h_{l-1}$ is:

$$\frac{\partial h_l}{\partial h_{l-1}} = I + \frac{\partial \,\mathrm{Sublayer}(\mathrm{LN}(h_{l-1}))}{\partial h_{l-1}}$$

The identity block $I$ provides a direct gradient highway back through all $L$ layers. Even if the sublayer Jacobian is small at initialization, the gradient is:

$$\frac{\partial \mathcal{L}}{\partial h_0} = \prod_{l=1}^{L}\left(I + J_l\right)$$

This product always contains the all-identity path (value 1 in each coordinate), so the gradient cannot vanish completely regardless of depth.

<details>
<summary>Gradient norm at initialization and warmup sensitivity</summary>
<p>At random initialization the sublayer outputs are close to zero, so $J_{l} \approx 0$ and each Jacobian $\approx I$. Under post-norm, LayerNorm is applied to $h + \mathrm{Sublayer}(h) \approx h$, so the normalization statistics are well-behaved early on. But as training progresses and sublayer norms grow, the post-norm LayerNorm must rescale increasingly large residuals, which creates instability spikes if the learning rate is not carefully warmed up.</p>
<p>Under pre-norm, LayerNorm always sees the input $h$ before the sublayer adds to it, so its statistics are controlled at every training step. This makes pre-norm substantially less sensitive to learning rate warmup. Xiong et al. (2020) showed analytically that gradient norms at initialization are $O(L)$ times smaller in post-norm than in pre-norm, which is why post-norm requires warmup to prevent the optimizer from taking huge steps early on.</p>
<p>One trade-off: because the residual stream in pre-norm can grow unboundedly across layers (no normalization at layer boundaries), the representation norms do increase with depth. LLaMA addresses this with RMSNorm (a simplified LayerNorm without mean subtraction) and keeps weight norms in check via weight decay.</p>
</details>

**In practice:**
- Post-norm can achieve slightly better final perplexity with careful tuning, but is harder to train at large scale.
- Pre-norm is the default for all modern large-scale LLMs (GPT-2/3/4, LLaMA, Mistral, PaLM) because it is stable out of the box without per-run warmup tuning.
- LLaMA replaces LayerNorm with RMSNorm ($\mathrm{RMSNorm}(x) = x / \sqrt{\frac{1}{d}\sum_{i} x_{i}^2 + \epsilon}$), dropping the mean-subtraction step for efficiency while keeping the variance-normalization benefit.

## Key Takeaways

**Noise is a feature, not a bug.** SGD's gradient noise prevents convergence to sharp minima and enables escape from saddle points. The convergence cost ($O(1/\sqrt{k})$ vs $O(1/k)$) is a worthwhile tradeoff when $n$ is large, because each step costs $O(1)$ instead of $O(n)$.

**Adaptivity solves the multi-scale problem.** Parameters in a neural network have wildly different gradient magnitudes (embedding layers vs attention heads vs output projections). A single learning rate cannot serve all of them well. AdaGrad discovered per-parameter scaling; RMSprop fixed its decay; Adam added momentum. The progression is logical, not arbitrary.

**Weight decay and $L_{2}$ are not the same thing under adaptive optimizers.** With plain SGD, the two are identical: adding $\frac{\lambda}{2}\lVert\theta\rVert^2$ to the loss produces a gradient $\nabla L + \lambda\theta$, and the update becomes

$$\theta \leftarrow \theta - \eta(\nabla L + \lambda\theta) = (1 - \eta\lambda)\,\theta - \eta\,\nabla L.$$

The $(1 - \eta\lambda)$ factor shrinks every weight uniformly by $\eta\lambda$ per step. This uniform shrinkage is what "weight decay" means.

With Adam, the same $L_{2}$ in the loss produces the same gradient $\nabla L + \lambda\theta$, but now it passes through the adaptive scaling. The effective decay on parameter $j$ becomes

$$\text{effective decay}_j = \frac{\eta\lambda}{\sqrt{\hat{v}_{k,j}} + \epsilon}$$

which varies per parameter. Parameters with large historical gradients (large $\hat{v}_{k,j}$) get less decay; parameters with small gradients get more decay. The adaptive mechanism, which was designed to normalize the learning step, unintentionally distorts the regularization.

AdamW fixes this by applying the decay directly to $\theta$ after the Adam step:

$$\theta_{k+1} = \theta_k - \frac{\eta}{\sqrt{\hat{v}_k} + \epsilon}\,\hat{m}_k - \eta\lambda\,\theta_k.$$

The decay term $\eta\lambda\,\theta_{k}$ is never scaled by $\hat{v}_{k}$, so every parameter shrinks uniformly. In practice, this means: always use `weight_decay` in AdamW (not `L2` in the loss), and only apply it to weight matrices, not to biases or layer norm parameters (which should not be regularized toward zero).

**Muon trades element-wise for matrix-wise normalization.** Adam normalizes each scalar weight independently, which is axis-aligned in parameter space and ignores the matrix structure of weight tensors. Muon orthogonalizes the gradient matrix via Newton-Schulz iterations, normalizing along singular directions instead of coordinate axes. This is steepest descent under the spectral norm rather than the Euclidean norm. The practical consequence: Muon achieves better loss per step on LLM pretraining, uses less optimizer memory (one momentum buffer instead of two), and pairs with AdamW (which handles embeddings, biases, and normalization parameters that lack matrix structure).

**The learning rate schedule encodes your training strategy.** Warmup prevents instability from large early gradients. Cosine decay spends more time at intermediate rates (where learning is productive) than linear decay. The schedule is often more important than the optimizer choice itself.

**Proximal methods extend gradient descent to non-smooth problems.** When the regularizer is not differentiable ($\ell_{1}$, nuclear norm), you cannot take a gradient. The proximal operator replaces the gradient step for the non-smooth part, and the soft-thresholding rule for $\ell_{1}$ is its most important special case. FISTA accelerates this to the optimal $O(1/k^2)$ rate.

**Subgradients generalize gradients; KKT generalizes "set the derivative to zero."** For non-smooth convex functions, the optimality condition is $0 \in \partial f(\theta^{\ast})$ (a subgradient of zero must exist), which directly gives the soft-thresholding solution for the lasso. For constrained problems, KKT replaces the single stationarity condition with four conditions (stationarity, primal/dual feasibility, complementary slackness). Complementary slackness is the key insight: it says inactive constraints have zero multipliers, which is why SVM solutions are sparse (only support vectors matter) and why ridge's penalty parameter is the KKT multiplier of the norm constraint.

**The dual is a lower bound; strong duality makes it tight.** Every constrained optimization has a dual: minimize over parameters first (giving a function of multipliers), then maximize over multipliers. The dual value is always a lower bound on the primal (weak duality). For convex problems, the bound is tight (strong duality): $p^{\ast} = d^{\ast}$. The practical payoff is that sometimes the dual is easier to solve (SVM dual has $n$ variables with simple bounds vs the primal's $p$-dimensional problem with $n$ constraints), or reveals hidden structure (the SVM dual depends on data only through inner products $x_{i}^Tx_{j}$, which enables the kernel trick). The lasso's penalized form $\frac{1}{2}\lVert y - X\theta\rVert^2 + \lambda\lVert\theta\rVert_{1}$ and constrained form $\min\frac{1}{2}\lVert y-X\theta\rVert^2$ s.t. $\lVert\theta\rVert_{1} \le t$ are primal-dual pairs: choosing $\lambda$ is solving the dual of the constrained problem.

**Second-order methods buy convergence speed with memory.** Newton converges quadratically but costs $O(p^3)$. CG avoids storing the Hessian via matrix-vector products. L-BFGS approximates curvature with $O(mp)$ memory. In practice, AdamW's implicit curvature adaptation (via $v_{k}$) is "good enough" for the scale of modern deep learning, which is why second-order methods remain niche for LLM training.

**Pre-norm is the stability default.** In a transformer block, LayerNorm can be applied before the sublayer (pre-norm: $h + \text{Sublayer}(\text{LN}(h))$) or after (post-norm: $\text{LN}(h + \text{Sublayer}(h))$). Pre-norm lets the residual stream flow through a clean addition without normalization touching it, so the sublayer always receives a normalized input regardless of how large $h$ has grown. Post-norm must rescale the sum $h + \text{Sublayer}(h)$, which creates instability spikes when sublayer outputs grow during training. Xiong et al. (2020) showed gradient norms at initialization are $O(L)$ times smaller in post-norm, which is why it requires warmup. Post-norm can achieve slightly better final perplexity with careful tuning, but pre-norm is stable out of the box and is the default for all modern LLMs (GPT-2/3/4, LLaMA, Mistral). See the Pre-norm vs Post-norm section above for the full derivation.

**Tradeoff: convergence rate vs per-step cost.** Full-batch GD converges at $O(1/k)$ but costs $O(n)$ per step. SGD converges at $O(1/\sqrt{k})$ but costs $O(1)$ per step. Nesterov achieves the optimal $O(1/k^2)$ but requires full-batch. For a fixed compute budget, SGD reaches a better solution than GD when $n$ is large, despite its slower rate, because it takes far more steps in the same time. The right comparison is always accuracy per FLOP, not accuracy per iteration.

**Tradeoff: adaptivity vs generalization.** Adaptive methods (Adam, AdamW) converge faster than SGD in the early phase because per-parameter scaling handles the multi-scale problem automatically. But SGD with momentum sometimes finds flatter minima that generalize better, especially on vision tasks. The practical compromise: use AdamW for transformers and LLMs (where adaptive scaling is essential), SGD with momentum for CNNs (where flat minima matter more). Muon offers a third option: matrix-level normalization via orthogonalization, which respects weight matrix geometry better than Adam's element-wise scaling while still providing adaptivity.

**Tradeoff: memory vs curvature information.** GD stores nothing beyond the gradient ($O(p)$). Momentum adds one velocity vector ($O(p)$). Adam adds two moment vectors ($O(3p)$). Muon stores one momentum buffer ($O(2p)$), saving memory over Adam by dropping the second moment. L-BFGS stores $m$ gradient-difference pairs ($O(mp)$). Newton stores the full Hessian ($O(p^2)$). More curvature information means faster convergence and better conditioning, but at the cost of memory that could hold a larger model or batch. At LLM scale ($p \sim 10^{10}$), even Adam's $3p$ is a significant constraint, which is why optimizer state sharding (ZeRO) exists.

## Comparison

| Method | Per-step cost | Adaptive LR | Convergence (convex) | Notes |
|---|---|---|---|---|
| GD | $O(n)$ | No | $O(1/k)$ | Baseline |
| SGD | $O(1)$ | No | $O(1/\sqrt{k})$ | Noisy; needs decay schedule |
| Nesterov | $O(n)$ | No | $O(1/k^2)$ | Optimal first-order |
| AdaGrad | $O(1)$ | Yes | $\tilde{O}(1/\sqrt{k})$ | Good for sparse; LR decays to zero |
| RMSprop | $O(1)$ | Yes | Not proven | AdaGrad without LR decay |
| Adam | $O(1)$ | Yes | Not proven | Default for deep learning; L2 reg broken |
| AdamW | $O(1)$ | Yes | Same as Adam | Standard for LLMs; $\beta_2{=}0.95$, clip |
| Muon | $O(1)$ + NS iters | Matrix-level | Not proven | Weight matrices only; pair with AdamW |
| Subgradient | $O(n)$ | No | $O(1/\sqrt{k})$ | Non-smooth; no sparsity; rarely used directly |
| ISTA | $O(n)$ | No | $O(1/k)$ | Non-smooth $R$; soft-threshold for L1 |
| FISTA | $O(n)$ | No | $O(1/k^2)$ | Nesterov-accelerated proximal |
| CG | $O(p)$/step | No | Exact in $p$ steps (quadratic) | Hessian-free via $Hv$ products |
| L-BFGS | $O(mp)$ | No | Superlinear | Full-batch only; small/medium $p$ |
| SAM | $2\times$ GD | Paired with any | Same + flatter minima | Generalization-focused |
