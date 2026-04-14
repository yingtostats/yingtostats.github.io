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
description: Convergence rates, gradient descent, momentum, adaptive methods (AdamW), proximal methods, second-order methods, and practical tricks for deep learning and LLM training.
fontsize: 20pt
---

{% include mathjax_support.html %}

Optimization in machine learning is the problem of finding parameters $\theta \in \mathbb{R}^p$ that minimize a loss function:

$$\min_\theta\; L(\theta) = \frac{1}{n}\sum_{i=1}^n \ell(\theta;\, x_i, y_i).$$

Here $\ell(\theta; x_i, y_i)$ is the **per-sample loss** (e.g., cross-entropy $-\log p_\theta(y_i \mid x_i)$, or squared error $\frac{1}{2}(f_\theta(x_i) - y_i)^2$), and $L(\theta)$ is its average over the training set. Note: $\ell$ here is the loss, not the log-likelihood, though for maximum likelihood estimation the two coincide up to sign ($\ell = -\log p_\theta$). $n$ can be enormous in modern settings (billions of tokens for LLM pretraining), and $L$ is typically nonconvex. This post covers convergence rates, first-order methods, momentum, adaptive methods, proximal methods, second-order methods, and practical engineering tricks for deep learning and LLM training.

## Convergence Rates

Two classification systems are commonly used and should not be confused.

**R-rate (global rate)** asks: after $k$ steps total, how small is the error? It describes the overall trajectory of $$\|\theta_k - \theta^*\|$$ as a function of $k$, giving a global envelope like $O(1/k)$ or $O(a^k)$. Named "R" for *root* because it is formally defined via $\limsup_{k\to\infty} \|\theta_k - \theta^*\|^{1/k}$.

**Q-rate (local ratio)** asks: how much does step $k+1$ improve over step $k$? It looks only at consecutive pairs: 

$$\frac{\|\theta_{k+1} - \theta^*\|} {\|\theta_k - \theta^*\|^q},$$

 and classifies convergence by the order $q$. Named "Q" for *quotient*.

The two are related but distinct. Q-linear (constant ratio $< 1$ at every step) implies R-linear (geometric global decay). But R-linear does not imply Q-linear: the step-by-step ratio could spike occasionally yet still yield geometric decay on average. Q-rate is more informative near a solution; R-rate is the right tool for comparing algorithms globally.

### R-rate: Global Rate of Decay

The R-rate describes how fast $\|\theta_k - \theta^*\|$ shrinks as a function of $k$.

**Sublinear.** The error decays polynomially:

$$\|\theta_k - \theta^*\| = O(1/k^p), \quad p > 0.$$

The key quantity is the **fraction of remaining error removed** at each step, $1 - \text{ratio}$, where

$$\text{ratio}_k = \frac{\|\theta_{k+1} - \theta^*\|}{\|\theta_k - \theta^*\|}.$$

For $\|\theta_k - \theta^*\| \approx 1/k$:

$$\text{ratio}_k \approx \frac{1/(k+1)}{1/k} = \frac{k}{k+1}, \qquad 1 - \text{ratio}_k = \frac{1}{k+1} \to 0.$$

The ratio itself approaches 1, but that is not the problem (the sequence still converges because ratio $< 1$). The problem is that $1 - \text{ratio}_k \to 0$: the **fraction of error removed per step shrinks to zero**. At step 10 you remove $\approx 9\%$ of remaining error; at step 1000 you remove $\approx 0.1\%$. Later steps are increasingly ineffective relative to what remains.

**Linear.** The error decays geometrically:

$$\|\theta_k - \theta^*\| = O(a^k), \quad a \in (0,1).$$

For $\|\theta_k - \theta^*\| \approx a^k$:

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

| Rate | $\|\theta_k - \theta^*\|$ | $\text{ratio}_k \to$ | Fraction removed/step |
|---|---|---|---|
| Sublinear | $O(1/k^p)$ | $1$ | $\to 0$ (shrinks) |
| Linear | $O(a^k)$ | $a \in (0,1)$ | $1-a$ (constant) |
| Superlinear | faster than $a^k$ | $0$ | $\to 1$ (grows) |
| Q-quadratic | $C\|e_k\|^2$ | $0$ | digits double/step |

## Gradient Descent

The update rule is

$$\theta_{k+1} = \theta_k - \eta \nabla L(\theta_k),$$

where $\eta > 0$ is the learning rate (step size).

**Convergence.** Under $L$-smoothness ($\|\nabla L(\theta) - \nabla L(\phi)\| \le L\|\theta - \phi\|$) and convexity, gradient descent with $\eta = 1/L$ satisfies

$$L(\theta_k) - L(\theta^*) \le \frac{\|\theta_0 - \theta^*\|^2}{2\eta k}.$$

This is $O(1/k)$. For $\mu$-strongly convex $L$, the rate improves to linear:

$$\|\theta_k - \theta^*\|^2 \le (1 - \eta\mu)^k \|\theta_0 - \theta^*\|^2.$$

**Pros.** Simple to implement; clean convergence theory; exact gradient means fixed learning rate suffices; linear convergence for strongly convex problems.

**Cons.** Requires one full pass over all $n$ data points per step; impractical for large datasets; single global learning rate for all parameters; sensitive to $L$-smoothness constant for step size tuning.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Derivation of the O(1/k) bound</span></summary>

<p>$L$-smoothness gives the descent lemma: for any $\theta, \phi$,</p>

$$L(\phi) \le L(\theta) + \nabla L(\theta)^T(\phi - \theta) + \frac{L}{2}\|\phi - \theta\|^2.$$

<p>Apply with $\phi = \theta_{k+1} = \theta_k - \eta\nabla L(\theta_k)$ and $\eta = 1/L$:</p>

$$L(\theta_{k+1}) \le L(\theta_k) - \frac{1}{2L}\|\nabla L(\theta_k)\|^2.$$

<p>Convexity gives $L(\theta_k) - L(\theta^*) \le \nabla L(\theta_k)^T(\theta_k - \theta^*)$, so by Cauchy-Schwarz:</p>

$$\|\nabla L(\theta_k)\|^2 \ge \frac{(L(\theta_k)-L(\theta^*))^2}{\|\theta_k - \theta^*\|^2}.$$

<p>Combining and telescoping over $k$ steps yields $L(\theta_k) - L(\theta^*) \le \|\theta_0 - \theta^*\|^2 / (2k/L)$.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Stochastic Gradient Descent

Full-batch gradient descent requires one pass over all $n$ data points per step. For large $n$ (e.g., billions of tokens in LLM pretraining), this is prohibitive. SGD replaces the full gradient with a noisy estimate from a single sample (or mini-batch of size $B$):

$$\theta_{k+1} = \theta_k - \eta_k \nabla \ell(\theta_k;\, x_{i_k}, y_{i_k}).$$

The gradient estimate is unbiased: $E[\nabla \ell(\theta_k; x_{i_k})] = \nabla L(\theta_k)$. The noise variance is $\sigma^2 = E[\|\nabla \ell - \nabla L\|^2]$.

**Convergence with decaying learning rate.** Under convexity and $\eta_k = c/\sqrt{k}$:

$$E[L(\theta_k)] - L(\theta^*) = \tilde{O}(1/\sqrt{k}).$$

This is slower than full-batch $O(1/k)$, but each iteration costs $O(1)$ vs $O(n)$, so SGD reaches a given accuracy in far fewer data touches when $n$ is large.

**Fixed vs decaying learning rate.** Whether a fixed $\eta$ converges depends on whether there is noise.

**Full-batch GD with fixed learning rate:** converges. The gradient is exact so the noise term is zero. The recursion is purely contractive and $\eta = 1/L$ gives $O(1/k)$. No decay needed.

**SGD with fixed learning rate:** does not converge to the optimum. A random sample has gradient variance

$$\sigma^2 = E\|\nabla\ell(\theta^*; x_i, y_i) - \nabla L(\theta^*)\|^2 > 0$$

even at the optimum, because a single sample never gives exactly $\nabla L(\theta^*) = 0$. The noise floor stabilizes at

$$E[L(\theta_k)] - L(\theta^*) \approx \frac{\eta \sigma^2}{2\mu} > 0,$$

so the iterates settle into a neighborhood of the optimum rather than converging to it. Decaying $\eta_k \to 0$ kills the noise floor and recovers convergence at $O(1/\sqrt{k})$.

In LLM training, a warmup + cosine decay schedule is used for this reason: the large early $\eta$ makes fast initial progress, and the small late $\eta$ shrinks the noise floor so the optimizer settles into a flat minimum rather than bouncing around it.

**Mini-batch SGD.** Using a batch of size $B$ reduces the gradient variance by $1/B$ (if samples are independent), improving the constant in the bound but not the $O(1/\sqrt{k})$ rate. In practice $B \in [32, 4096]$ balances variance reduction against hardware parallelism.

**Pros.** $O(1)$ cost per step; scales to arbitrarily large datasets; gradient noise helps escape saddle points and sharp minima; foundation of all modern deep learning optimization.

**Cons.** Slower $O(1/\sqrt{k})$ rate vs $O(1/k)$ for full-batch GD; requires decaying learning rate to converge; high variance with small batches; no per-parameter adaptivity (same $\eta$ for all weights).

<details>
<summary><span style="color: saddlebrown; font-style: italic;">One-step recursion: the three-term structure</span></summary>

<p>Expand the squared distance to $\theta^*$ after one SGD step, where $g_k = \nabla\ell(\theta_k; x_{i_k})$:</p>

$$E\|\theta_{k+1} - \theta^*\|^2 = E\|\theta_k - \eta_k g_k - \theta^*\|^2$$
$$= \underbrace{E\|\theta_k - \theta^*\|^2}_{\text{current distance}} \;-\; \underbrace{2\eta_k E[g_k^T(\theta_k - \theta^*)]}_{\text{contraction}} \;+\; \underbrace{\eta_k^2 E\|g_k\|^2}_{\text{noise injection}}.$$

<p><strong>Why the contraction term is negative.</strong> The full contraction term is $-2\eta_k E[g_k^T(\theta_k - \theta^*)]$. It is negative when $E[g_k^T(\theta_k - \theta^*)] > 0$, i.e., when the gradient points in the same direction as $(\theta_k - \theta^*)$. Two steps:</p>

<p><strong>Step 1: Replace $g_k$ with $\nabla L(\theta_k)$ via unbiasedness.</strong> Since $g_k$ is an unbiased estimator of the full gradient conditional on $\theta_k$:</p>

$$E[g_k^T(\theta_k - \theta^*)] = \nabla L(\theta_k)^T(\theta_k - \theta^*).$$

<p><strong>Step 2: Show $\nabla L(\theta_k)^T(\theta_k - \theta^*) \ge L(\theta_k) - L(\theta^*) > 0$ by convexity.</strong> Convexity of $L$ means the function lies above every tangent hyperplane. At $\theta^*$:</p>

$$L(\theta^*) \ge L(\theta_k) + \nabla L(\theta_k)^T(\theta^* - \theta_k).$$

<p>Rearranging:</p>

$$\nabla L(\theta_k)^T(\theta_k - \theta^*) \ge L(\theta_k) - L(\theta^*) > 0,$$

<p>where the last inequality holds because $\theta_k \ne \theta^*$ (if $\theta_k = \theta^*$ we are done). Geometrically, the gradient $\nabla L(\theta_k)$ always points away from the optimum $\theta^*$: it makes a positive angle with the vector $(\theta_k - \theta^*)$ pointing from $\theta^*$ to the current iterate. So the contraction term $-2\eta_k \times \text{positive}$ is always negative, always pulling $\theta_k$ toward $\theta^*$.</p>

<p>The noise term is always positive (bounded by $\eta_k^2 G^2$ under the bounded gradient assumption $E\|g_k\|^2 \le G^2$). It injects error back in regardless of how close $\theta_k$ is to $\theta^*$.</p>

<p>Applying the bounds on both terms gives the key recursion:</p>

$$E\|\theta_{k+1} - \theta^*\|^2 \le E\|\theta_k - \theta^*\|^2 - 2\eta_k(E[L(\theta_k)] - L(\theta^*)) + \eta_k^2 G^2.$$

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why a fixed learning rate cannot converge</span></summary>

<p>With $\eta_k = \eta$ (constant), rearrange the key recursion to isolate the excess loss:</p>

$$2\eta(E[L(\theta_k)] - L(\theta^*)) \le E\|\theta_k - \theta^*\|^2 - E\|\theta_{k+1} - \theta^*\|^2 + \eta^2 G^2.$$

<p>As $\theta_k \to \theta^*$, the left side $\to 0$ and the telescoping terms $\to 0$, but the noise floor $\eta^2 G^2$ remains fixed. The equation balances at</p>

$$E[L(\theta_k)] - L(\theta^*) \;\approx\; \frac{\eta G^2}{2} \;>\; 0.$$

<p>No matter how long we run, the expected loss stays above $\theta^*$ by at least $\eta G^2/2$. Making $\eta$ small reduces the floor but also slows the contraction, so there is a tradeoff: smaller $\eta$ means slower progress but a tighter final neighborhood. With fixed $\eta$, we can only converge to a ball around $\theta^*$, not to $\theta^*$ itself.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">How decaying η_k = c/√k gives O(1/√k)</span></summary>

<p>Starting from the key recursion, rearrange and sum from $k=1$ to $K$. The distance terms telescope:</p>

$$2\sum_{k=1}^K \eta_k(E[L(\theta_k)] - L(\theta^*)) \le \|\theta_0 - \theta^*\|^2 + G^2\sum_{k=1}^K \eta_k^2.$$

<p>Define the weighted average iterate $\bar\theta_K = \frac{\sum_k \eta_k \theta_k}{\sum_k \eta_k}$. By convexity of $L$:</p>

$$E[L(\bar\theta_K)] - L(\theta^*) \le \frac{\|\theta_0 - \theta^*\|^2 + G^2\sum_{k=1}^K \eta_k^2}{2\sum_{k=1}^K \eta_k}.$$

<p>Now substitute $\eta_k = c/\sqrt{k}$ and evaluate the two sums:</p>

$$\sum_{k=1}^K \eta_k = c\sum_{k=1}^K \frac{1}{\sqrt{k}} \approx 2c\sqrt{K}, \qquad \sum_{k=1}^K \eta_k^2 = c^2\sum_{k=1}^K \frac{1}{k} \approx c^2\log K.$$

<p>Substituting:</p>

$$E[L(\bar\theta_K)] - L(\theta^*) \le \frac{\|\theta_0-\theta^*\|^2 + G^2 c^2\log K}{4c\sqrt{K}} = O\!\left(\frac{\log K}{\sqrt{K}}\right).$$

<p>This is sometimes written loosely as $O(1/\sqrt{K})$, which is an abuse of notation: $\log K / \sqrt{K}$ is not strictly $O(1/\sqrt{K})$ because $\log K \to \infty$. The formal notation is $\tilde{O}(1/\sqrt{K})$, meaning $O(1/\sqrt{K})$ up to logarithmic factors. Since $\log K$ grows far slower than any positive power of $K$, the $\log K$ factor is negligible in practice and the rate is effectively $1/\sqrt{K}$.</p>

<p>The key insight: $\eta_k \to 0$ kills the noise floor ($\sum \eta_k^2$ grows only as $\log K$), while $\sum \eta_k \sim \sqrt{K}$ grows fast enough that averaging still makes progress. Any faster decay (e.g., $\eta_k = c/k$) would make $\sum \eta_k$ grow too slowly, losing progress. Any slower decay (fixed $\eta$) leaves the noise floor nonzero.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why full-batch GD achieves O(1/k), and why mini-batch doesn't improve the rate</span></summary>

<p><strong>Full-batch GD.</strong> With the exact gradient, there is no noise: $G^2 = 0$. The recursion becomes</p>

$$L(\theta_{k+1}) \le L(\theta_k) - \frac{1}{2L}\|\nabla L(\theta_k)\|^2.$$

<p>With a fixed $\eta = 1/L$, the noise floor is zero and no decay is needed. Telescoping gives $L(\theta_K) - L(\theta^*) \le O(1/K)$. Full-batch can use a fixed learning rate and achieves a faster $O(1/K)$ rate precisely because the noise term is absent.</p>

<p><strong>Mini-batch SGD.</strong> With batch size $B$, samples are averaged so the gradient variance drops from $\sigma^2$ to $\sigma^2/B$. In the bound, $G^2$ is replaced by $G^2/B$:</p>

$$E[L(\bar\theta_K)] - L(\theta^*) \le \frac{\|\theta_0-\theta^*\|^2 + (G^2/B)\sum_k \eta_k^2}{2\sum_k \eta_k}.$$

<p>The numerator shrinks by $1/B$, but the structure is identical. With $\eta_k = c/\sqrt{k}$, the bound is still $O(1/\sqrt{K})$, just with a smaller constant $1/B$. The rate is unchanged because the noise term ($\sim \log K$) still dominates the numerator for large $K$, regardless of $B$. Only at $B = n$ (full batch) does the noise vanish entirely, eliminating the $\sum \eta_k^2$ term and recovering $O(1/K)$.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Momentum

Gradient descent uses only the current gradient and has no memory of past steps. This causes two problems: slow progress along shallow directions, and oscillation across steep directions. Momentum methods accumulate a velocity vector that builds up speed in consistent directions and dampens oscillation in alternating directions.

**The ravine intuition.** Consider a loss surface shaped like an elongated valley: steep sides, gentle slope along the bottom. Plain GD follows the gradient at each step, oscillating left-right across the narrow steep dimension while making slow progress along the gentle bottom. Momentum helps because the left-right gradient components alternate in sign and cancel in the velocity $v_k$, while the consistent downhill components accumulate. The result is faster progress along the bottom and damped oscillations across it.

### Heavy Ball (Classical Momentum)

$$v_{k+1} = \beta v_k - \eta \nabla L(\theta_k), \qquad \theta_{k+1} = \theta_k + v_{k+1}.$$

The velocity $v_k$ is a weighted sum of all past gradients with exponentially decaying weights $\beta^j$ for gradient $j$ steps ago. In a direction where gradients consistently point the same way, $v_k$ grows proportionally to $1/(1-\beta)$ times the gradient magnitude. In a direction where gradients oscillate (as in the ravine sides), the positive and negative contributions cancel and $v_k$ stays small.

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

<p>Define the momentum coefficient $\lambda_{k+1}^2 = \lambda_{k+1} + \lambda_k^2$ (with $\lambda_0 = 0$, so $\lambda_k \approx k/2$). Nesterov constructs an <em>estimate sequence</em> $\phi_k(\theta)$, a sequence of lower bounds on $L$ that tighten at rate $1/\lambda_k^2$. The lookahead point $y_k$ is chosen so that the iterate $\theta_{k+1}$ maintains the invariant $L(\theta_k) \le \phi_k^*$ (minimum of $\phi_k$). Tightening of $\phi_k^*$ propagates to $L(\theta_k)$, giving the $O(1/k^2)$ bound. The key is that the momentum coefficient $k/(k+3)$ is not arbitrary; it is derived from the $\lambda_k$ recurrence to maintain this invariant exactly.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Adaptive Learning Rate Methods

A single global learning rate is suboptimal when parameters have very different gradient magnitudes. Adaptive methods maintain per-parameter learning rates, scaling down for parameters with large historical gradients and scaling up for sparse ones.

### AdaGrad

$G_k \in \mathbb{R}^p$ accumulates the sum of squared gradients for each parameter separately, and each parameter is updated by dividing by its own $\sqrt{G_{k,j}}$:

$$G_{k,j} = G_{k-1,j} + (\nabla_j L(\theta_k))^2, \qquad \theta_{k+1,j} = \theta_{k,j} - \frac{\eta}{\sqrt{G_{k,j} + \epsilon}} \cdot \nabla_j L(\theta_k).$$

**Why dividing by $\sqrt{G_k}$ cancels out gradient magnitude.** The effective update for parameter $j$ is $\frac{\eta}{\sqrt{G_{k,j}}} \cdot \nabla_j L(\theta_k)$. Consider two cases:

- **Dense parameter** (gradient magnitude approximately $M > 0$ at every step): after $k$ steps, $G_{k,j} \approx kM^2$, so $\sqrt{G_{k,j}} \approx \sqrt{k}\,M$, and the effective update is $\frac{\eta}{\sqrt{k}\,M} \cdot M = \frac{\eta}{\sqrt{k}}$. The gradient magnitude $M$ cancels exactly, so large-gradient parameters are automatically given a smaller learning rate.
- **Sparse parameter** (gradient magnitude $M > 0$ only $s \ll k$ times, zero otherwise): $G_{k,j} \approx sM^2$, so the effective update is $\frac{\eta}{\sqrt{s}\,M} \cdot M = \frac{\eta}{\sqrt{s}}$. Since $s \ll k$, this is much larger than for the dense parameter, so rare parameters keep a large effective learning rate.

In both cases the gradient magnitude cancels out of the update. The current gradient only provides the **direction**; the **step size** is determined entirely by the history of past gradients accumulated in $G_{k,j}$. This is equivalent to normalizing each update by the root-mean-square (RMS) of past gradients: frequently updated parameters get shrinking steps, rarely updated parameters keep large steps.

**Limitation.** $G_{k,j}$ only ever grows, so $\eta/\sqrt{G_{k,j}} \to 0$ for every parameter eventually, and learning stops. This makes AdaGrad poorly suited for deep networks trained for many steps.

**Pros.** Automatic per-parameter learning rates; excellent for sparse gradients (e.g., word embeddings where most parameters receive zero gradient per step); no learning rate tuning needed for sparse problems.

**Cons.** Effective learning rate decays to zero monotonically, so learning eventually stops; unsuitable for long training runs or dense gradient problems; superseded by RMSprop and Adam in practice.

### RMSprop

RMSprop fixes AdaGrad's vanishing learning rate by replacing the cumulative sum $G_k = \sum_{j=1}^k \nabla L(\theta_j)^2$ with an exponential moving average (EMA):

$$v_k = \rho v_{k-1} + (1-\rho)\nabla L(\theta_k)^2, \qquad \theta_{k+1} = \theta_k - \frac{\eta}{\sqrt{v_k + \epsilon}} \odot \nabla L(\theta_k),$$

with $\rho \in [0.9, 0.99]$ typically. In AdaGrad, $G_k$ grows without bound so $\eta/\sqrt{G_k} \to 0$. The EMA instead gives each squared gradient an exponentially decaying weight, so $v_k$ only remembers recent gradient magnitudes and stays bounded. The effective learning rate no longer decays to zero.

**Pros.** Fixes AdaGrad's vanishing learning rate; simple and robust; good empirical performance on RNNs and recurrent architectures.

**Cons.** No convergence theory for general nonconvex problems; no first moment (momentum), unlike Adam; sensitive to $\rho$ and $\epsilon$; largely replaced by Adam/AdamW in modern practice.

### Adam

Adam (Adaptive Moment Estimation) combines RMSprop's second moment with a first moment estimate (momentum). Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Step 1: first moment (momentum)**

$$m_k = \beta_1 m_{k-1} + (1-\beta_1)\nabla L(\theta_k).$$

**Step 2: second moment (adaptive scaling)**

$$v_k = \beta_2 v_{k-1} + (1-\beta_2)\nabla L(\theta_k)^2.$$

**Step 3: bias correction**

$$\hat m_k = \frac{m_k}{1-\beta_1^k}, \qquad \hat v_k = \frac{v_k}{1-\beta_2^k}.$$

**Step 4: parameter update**

$$\theta_{k+1} = \theta_k - \frac{\eta}{\sqrt{\hat v_k} + \epsilon}\hat m_k.$$

The bias corrections matter early in training. At step 1, $m_1 = (1-\beta_1)g_1$, which is much smaller than $g_1$. Dividing by $(1-\beta_1^k)$ rescales it back to the true gradient magnitude.

**Pros.** Combines adaptive learning rates with momentum; fast convergence in practice; robust to learning rate choice; default optimizer for most deep learning.

**Cons.** No convergence proof for general convex problems (counterexamples exist); $L_2$ regularization is broken when used with Adam (use AdamW instead); can generalize worse than SGD with momentum on some vision tasks due to sharp minima.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why bias correction is needed</span></summary>

<p>Initialize $m_0 = 0$. At step $k$:</p>

$$m_k = (1-\beta_1)\sum_{j=1}^k \beta_1^{k-j} g_j.$$

<p>Taking expectations (assuming stationary gradients $E[g_j] = g$):</p>

$$E[m_k] = g(1-\beta_1)\sum_{j=1}^k \beta_1^{k-j} = g(1-\beta_1^k).$$

<p>So $m_k$ underestimates the true gradient by a factor $(1-\beta_1^k)$, which is close to zero when $k$ is small and $\beta_1$ is close to 1. Dividing by $(1-\beta_1^k)$ removes this initialization bias. The same argument applies to $v_k$. After many steps $\beta_1^k \approx 0$ and the correction becomes negligible.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### AdamW

**Why weight decay is needed.** Without regularization, nothing prevents weights from growing arbitrarily large. Large weights make the model sensitive to small input changes, hurting generalization. $L_2$ regularization addresses this by adding a penalty $\frac{\lambda}{2}\|\theta\|^2$ to the loss, which pulls all weights toward zero. The intended effect is simple: at each step, shrink every weight uniformly by a small amount $\eta\lambda$, independently of the gradient. This is called **weight decay**.

**The problem with Adam + $L_2$.** The standard way to add $L_2$ is to include it in the loss, making the gradient $\nabla L(\theta) + \lambda\theta$. But in Adam, every gradient term (including $\lambda\theta$) gets absorbed into the moment estimates $m_k$ and $v_k$ and then adaptively scaled by $1/\sqrt{\hat v_{k,j}}$. The effective weight decay actually applied to parameter $j$ is

$$\text{effective decay}_j = \frac{\eta\lambda}{\sqrt{\hat v_{k,j}} + \epsilon},$$

which varies per parameter. Parameters with large historical gradients (large $\hat v_{k,j}$) receive small weight decay; parameters with small gradients receive large weight decay. This is the opposite of uniform shrinkage: the adaptive scaling unintentionally distorts what $L_2$ regularization is supposed to do.

AdamW (Loshchilov and Hutter, 2019) fixes this by decoupling weight decay from the gradient update entirely:

$$\theta_{k+1} = \theta_k - \frac{\eta}{\sqrt{\hat v_k}+\epsilon}\hat m_k - \eta\lambda\theta_k.$$

The second term applies uniform decay $\eta\lambda$ to every parameter regardless of $\hat v_k$. This is what $L_2$ regularization was supposed to do.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why the two formulations differ: the interaction between adaptive scaling and L2</span></summary>

<p>With L2 in the loss, Adam's gradient at step $k$ is $g_k = \nabla L(\theta_k) + \lambda\theta_k$. The second moment accumulates:</p>

$$v_{k,j} = \beta_2 v_{k-1,j} + (1-\beta_2)(\nabla_j L + \lambda\theta_{k,j})^2.$$

<p>Expanding the square: $(\nabla_j L)^2 + 2\lambda\theta_{k,j}\nabla_j L + \lambda^2\theta_{k,j}^2$. The cross term and $\lambda^2$ term mean $v_{k,j}$ is inflated differently for each parameter depending on the magnitude of $\theta_{k,j}$ relative to $\nabla_j L$. The resulting update</p>

$$\frac{\eta}{\sqrt{\hat v_{k,j}}+\epsilon}(\nabla_j L + \lambda\theta_{k,j})$$

<p>does not simplify to a clean gradient step plus a clean decay step. AdamW separates them: apply the Adam step to $\nabla L$ alone, then subtract $\eta\lambda\theta_k$ directly. The decay is now exactly $\eta\lambda$ per step, independent of the gradient history.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

**Pros.** Fixes Adam's broken $L_2$ regularization; uniform, predictable weight decay independent of gradient history; standard optimizer for LLMs and transformers; well-calibrated with $\beta_2 = 0.95$ for long training runs.

**Cons.** Same theoretical issues as Adam (no convergence proof); requires careful tuning of $\lambda$, $\beta_2$, and gradient clipping threshold; slightly more hyperparameters than SGD.

**AdamW in LLM training.** AdamW is the standard optimizer for training transformers and large language models. Practical hyperparameter choices differ from the original Adam defaults:

- $\beta_1 = 0.9$, $\beta_2 = 0.95$ (not 0.999). The lower $\beta_2$ makes the second moment adapt faster to shifting gradient magnitudes during long LLM training runs. With $\beta_2 = 0.999$, the moving average is slow to forget large gradients from early training, causing the effective learning rate to stay suppressed.
- Weight decay $\lambda = 0.1$, applied to all parameters except biases and layer norm scales.
- **Gradient clipping**: clip the global gradient norm to a threshold (typically 1.0) before the Adam update: $g_k \leftarrow g_k \cdot \min(1, c / \|g_k\|)$. This prevents exploding gradients from destabilizing the second moment estimates.
- **Warmup + cosine decay**: linear warmup for the first 1–2% of steps, then cosine decay to $\eta_{\min} \approx \eta_{\max}/10$.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why β₂ = 0.95 rather than 0.999 for LLMs</span></summary>

<p>The second moment estimate tracks a weighted average of past squared gradients with effective memory window $\approx 1/(1-\beta_2)$ steps. With $\beta_2 = 0.999$, the window is $\approx 1000$ steps; with $\beta_2 = 0.95$, it is $\approx 20$ steps.</p>

<p>In LLM training, gradient magnitudes vary significantly across training phases: early on, the model is essentially random and gradients are large; as the model fits, gradients shrink and the loss landscape changes. A slow-moving $\hat v_k$ (large $\beta_2$) retains memory of large early gradients, keeping the effective learning rate small even when current gradients are modest. A faster-adapting $\hat v_k$ (smaller $\beta_2$) lets the learning rate recover more quickly as the training signal changes, which matters when training continues through multiple phases (e.g., pretraining, annealing, fine-tuning).</p>

<p>The tradeoff: smaller $\beta_2$ makes $\hat v_k$ noisy (high variance estimate of second moment), which can cause instability on individual parameter updates. Gradient clipping mitigates this by bounding the maximum update size.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Learning Rate Schedules

The learning rate $\eta$ is often the most important hyperparameter. Common schedules:

**Warmup.** Start with a small $\eta$ and increase linearly for the first $W$ steps, then switch to another schedule. Warmup stabilizes early training when parameter estimates are far from optimum and gradients are large and noisy. Used widely in transformer training.

**Cosine decay.** After warmup, anneal according to

$$\eta_k = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi k}{K}\right),$$

where $K$ is the total number of steps. This smoothly reduces $\eta$ to $\eta_{\min}$ while spending more time at intermediate rates than linear decay.

**Linear decay.** $\eta_k = \eta_0(1 - k/K)$. Simple and effective; used in many NLP fine-tuning recipes.

**Cyclic and restart schedules.** Periodically reset $\eta$ to $\eta_{\max}$ (cosine restarts). The idea is that saddle points and poor local minima may be escaped by occasional large steps. Empirically useful when the loss landscape has many nearby local minima.

## Proximal Methods

Gradient descent requires $L$ to be smooth (differentiable). Many useful regularizers ($L_1$ penalty, nuclear norm, indicator functions of constraint sets) are not smooth. Proximal methods handle objectives of the form

$$\min_\theta\; L(\theta) + R(\theta),$$

where $L$ is smooth (gradient available) and $R$ is convex but possibly non-smooth.

### Proximal Operator

The proximal operator of $R$ at scale $\eta$ is

$$\mathrm{prox}_{\eta R}(v) = \arg\min_\theta\left\{\eta R(\theta) + \frac{1}{2}\|\theta - v\|^2\right\}.$$

It finds the point closest to $v$ that also has small $R(\theta)$. The $\frac{1}{2}\|\theta - v\|^2$ term keeps the solution near $v$; $\eta$ controls the tradeoff.

**For $L_1$ regularization** ($R(\theta) = \|\theta\|_1$), the proximal operator is soft-thresholding applied elementwise:

$$[\mathrm{prox}_{\eta\|\cdot\|_1}(v)]_j = \mathcal{S}_\eta(v_j) = \mathrm{sign}(v_j)\max(|v_j| - \eta, 0).$$

Coordinates smaller than $\eta$ in magnitude are zeroed out; larger coordinates are shrunk by $\eta$. This is exactly the lasso solution under orthonormal design.

**For $L_2$ regularization** ($R(\theta) = \frac{\lambda}{2}\|\theta\|^2$), the proximal operator is ridge shrinkage:

$$\mathrm{prox}_{\eta\frac{\lambda}{2}\|\cdot\|^2}(v) = \frac{v}{1+\eta\lambda}.$$

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Derivation of the soft-thresholding proximal operator</span></summary>

<p>The problem separates by coordinate: for each $j$, minimize</p>

$$f(t) = \eta|t| + \frac{1}{2}(t - v_j)^2.$$

<p>Taking the subdifferential and setting it to zero: $0 \in \eta\,\partial|t| + (t - v_j)$, i.e., $v_j - t \in \eta\,\partial|t|$.</p>

<p><strong>Case 1: $t > 0$.</strong> $\partial|t| = \{1\}$, so $v_j - t = \eta$, giving $t = v_j - \eta$. Valid only if $v_j - \eta > 0$, i.e., $v_j > \eta$.</p>

<p><strong>Case 2: $t < 0$.</strong> $\partial|t| = \{-1\}$, so $v_j - t = -\eta$, giving $t = v_j + \eta$. Valid only if $v_j + \eta < 0$, i.e., $v_j < -\eta$.</p>

<p><strong>Case 3: $t = 0$.</strong> $\partial|t| = [-1,1]$, so need $v_j \in [-\eta, \eta]$. Valid when $|v_j| \le \eta$.</p>

<p>Combining: $t^* = \mathrm{sign}(v_j)\max(|v_j|-\eta, 0)$.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Proximal Gradient (ISTA)

The proximal gradient algorithm alternates a gradient step on $L$ with a proximal step on $R$:

$$\theta_{k+1} = \mathrm{prox}_{\eta R}\!\left(\theta_k - \eta\nabla L(\theta_k)\right).$$

**Convergence.** Under $L$-smoothness of $L$ and convexity of both $L$ and $R$, with $\eta = 1/L$:

$$L(\theta_k) + R(\theta_k) - (L(\theta^*) + R(\theta^*)) \le \frac{\|\theta_0 - \theta^*\|^2}{2\eta k} = O(1/k).$$

**Pros.** Handles non-smooth regularizers exactly; $L_1$ gives exact sparse solutions via soft-thresholding; clean convergence guarantee; each step is cheap when the proximal operator has a closed form.

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

## Second-Order Methods

Gradient descent uses the gradient $\nabla L$ (first-order information). Newton's method uses the Hessian $\nabla^2 L$ to scale steps by local curvature:

$$\theta_{k+1} = \theta_k - [\nabla^2 L(\theta_k)]^{-1} \nabla L(\theta_k).$$

Near a solution, Newton converges quadratically. The cost is computing and inverting the $p \times p$ Hessian, which is $O(p^3)$ per step and infeasible for large neural networks ($p \sim 10^8$).

**Pros of Newton's method.** Q-quadratic convergence near a solution; handles ill-conditioned problems naturally; no learning rate tuning needed.

**Cons of Newton's method.** $O(p^3)$ per step; $O(p^2)$ memory for Hessian; requires exact Hessian (not available in stochastic setting); impractical beyond $p \sim 10^4$.

**Conjugate Gradient (CG).** For quadratic objectives $L(\theta) = \frac{1}{2}\theta^T A\theta - b^T\theta$ with $A \succ 0$, CG finds the exact solution in at most $p$ steps without forming or inverting $A$. At each step it chooses a search direction conjugate to all previous directions under $A$-inner product ($d_i^T A d_j = 0$ for $i \ne j$), which guarantees no progress is undone. Convergence rate depends on the condition number: $\|\theta_k - \theta^*\|_A \le 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k \|\theta_0 - \theta^*\|_A$.

For nonlinear objectives, Nonlinear CG (Fletcher-Reeves, Polak-Ribière) generalizes the update direction using the current gradient, with periodic restarts. CG requires no matrix storage beyond the current gradient and direction, using only $O(p)$ memory.

**Hessian-free optimization** uses CG to solve the Newton system $\nabla^2 L(\theta)\,d = -\nabla L(\theta)$ without forming $\nabla^2 L$. The key: CG only needs matrix-vector products $\nabla^2 L(\theta)\,v$, which can be computed via finite differences of gradients, $(\nabla L(\theta + \epsilon v) - \nabla L(\theta))/\epsilon$, at cost $O(p)$ per product. This makes second-order methods tractable for moderately large networks.

**Pros of CG.** Exact solution for quadratics in $p$ steps; $O(p)$ memory; enables Hessian-free second-order methods via cheap $Hv$ products.

**Cons of CG.** Exact only for quadratics; requires restarts for nonlinear objectives; sensitive to preconditioning; less commonly used now that AdamW dominates in practice.

**Quasi-Newton methods (L-BFGS)** approximate the Hessian inverse using the history of gradient differences, requiring only $O(mp)$ memory for $m$ stored vectors. L-BFGS is the standard choice for small-to-medium scale problems where full-batch gradients are available (e.g., logistic regression, shallow networks).

**Pros of L-BFGS.** Superlinear convergence; handles ill-conditioning well without forming the full Hessian; $O(mp)$ memory with $m$ typically 5–20; gold standard for full-batch convex problems.

**Cons of L-BFGS.** Requires full-batch gradients (incompatible with SGD); $O(mp)$ memory and overhead still too large for $p \sim 10^8$; convergence degrades with noisy gradients.

**Natural gradient.** Replace the Euclidean gradient with the Riemannian gradient under the Fisher information metric:

$$\theta_{k+1} = \theta_k - \eta F(\theta_k)^{-1} \nabla L(\theta_k),$$

where $F(\theta) = E[\nabla \log p(y|x,\theta)\nabla \log p(y|x,\theta)^T]$ is the Fisher matrix. This is equivalent to steepest descent in distribution space (KL divergence) rather than parameter space. K-FAC and Shampoo are practical approximations used in large-scale training.

**Pros of natural gradient.** Invariant to reparameterization; fast convergence near a solution; theoretically optimal for probabilistic models.

**Cons of natural gradient.** Fisher matrix is $O(p^2)$ to store and $O(p^3)$ to invert; K-FAC and Shampoo approximations are complex to implement and tune; communication-heavy in distributed training.

## Loss Landscape in Deep Learning

Deep networks are nonconvex. The classical worry was local minima, but empirically, large networks rarely get stuck. Two more relevant phenomena:

**Saddle points.** A point where $\nabla L = 0$ but the Hessian has both positive and negative eigenvalues. In high dimensions, most critical points are saddle points (the fraction with all positive eigenvalues decays exponentially in $p$). SGD noise helps escape saddle points; gradient descent can get stuck.

**Flat regions and sharpness.** The loss can be flat (near-zero gradients) over large regions, especially early in training. Separately, the sharpness of the final minimum (the largest Hessian eigenvalue $\lambda_{\max}(\nabla^2 L(\theta^*))$) correlates strongly with generalization: flatter minima generalize better. This motivates methods that explicitly seek flat minima.

**SAM (Sharpness-Aware Minimization).** Instead of minimizing $L(\theta)$, SAM minimizes the worst-case loss in a neighborhood:

$$\min_\theta \max_{\|\epsilon\|\le\rho} L(\theta + \epsilon).$$

The inner maximization is approximated by one gradient ascent step: $\hat\epsilon = \rho \nabla L(\theta) / \|\nabla L(\theta)\|$. The outer step then follows $\nabla L(\theta + \hat\epsilon)$. SAM adds one extra forward-backward pass per step but consistently improves generalization in image and language tasks.

**Pros.** Consistently improves generalization by finding flatter minima; can be paired with any base optimizer (SGD, AdamW); single extra hyperparameter $\rho$.

**Cons.** Exactly $2\times$ compute per step (two forward-backward passes); no convergence theory for nonconvex objectives; $\rho$ requires tuning; less commonly used in LLM training where compute is the binding constraint.

## Practical Tricks for Deep Learning and LLM Training

The theory above gives convergence guarantees under idealized assumptions. In practice, training deep networks and LLMs requires a collection of engineering tricks to stabilize training, reduce memory, and improve generalization.

### Gradient Clipping

Before each optimizer step, rescale the gradient if its global norm exceeds a threshold $c$:

$$g_k \leftarrow g_k \cdot \min\!\left(1,\; \frac{c}{\|g_k\|_2}\right).$$

This leaves small gradients unchanged and shrinks large gradients proportionally, preventing a single bad batch from destabilizing the optimizer state. In LLM training with AdamW, $c = 1.0$ is standard. Without clipping, a spike in gradient magnitude inflates $v_k$ in Adam, suppressing the effective learning rate for many subsequent steps.

### Mixed Precision Training (FP16 / BF16)

Store activations and gradients in 16-bit floating point to halve memory and double throughput on modern hardware, while keeping a **master copy of weights in FP32** for accurate parameter updates.

- **FP16** (IEEE half): 5 exponent bits, 10 mantissa bits. Dynamic range $\approx [6\times10^{-5},\, 65504]$. Gradients near zero can **underflow** to exactly 0.
- **BF16** (brain float): 8 exponent bits, 7 mantissa bits. Same dynamic range as FP32 ($\approx 10^{-38}$ to $10^{38}$) but lower precision. Preferred for LLMs because gradient underflow is rarely a problem.

**Loss scaling** (needed for FP16, not BF16): multiply the loss by a large scalar $S$ before backprop so gradients are scaled up into the representable range; divide the accumulated gradients by $S$ before the optimizer step. Dynamic loss scaling starts with $S = 2^{15}$ and halves it on overflow, doubles it every 2000 steps without overflow.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why master weights in FP32 are necessary</span></summary>

<p>The AdamW update subtracts a small increment $\Delta\theta = \eta \hat m_k / (\sqrt{\hat v_k}+\epsilon)$ from $\theta$. For a typical learning rate $\eta = 10^{-4}$ and normalized gradient, $|\Delta\theta_j|$ can be $O(10^{-5})$ or smaller. In FP16, the smallest representable difference for a weight of magnitude $O(1)$ is $\approx 10^{-3}$ (limited by the 10-bit mantissa). Any update smaller than this is rounded to zero, so the weight never changes. Storing master weights in FP32 (machine epsilon $\approx 10^{-7}$) ensures small updates accumulate correctly. The FP16 copies are used only for forward and backward passes.</p>

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

<p>Consider a layer $h = W x$ where $W \in \mathbb{R}^{m\times n}$, $x \in \mathbb{R}^n$. Assume $W_{ij} \sim (0, \sigma^2)$ iid, $x_i \sim (0, \sigma_x^2)$ iid, and $W \perp x$. Then</p>

$$\mathrm{Var}(h_i) = \sum_j \mathrm{Var}(W_{ij} x_j) = n\sigma^2\sigma_x^2.$$

<p>To preserve variance ($\mathrm{Var}(h_i) = \sigma_x^2$), set $\sigma^2 = 1/n$. For ReLU, the output has roughly half its entries zeroed, halving the variance. To compensate, set $\sigma^2 = 2/n$. He initialization applies this correction.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Gradient Checkpointing

Backpropagation requires storing all intermediate activations from the forward pass to compute gradients. For a network with $L$ layers, this is $O(L)$ memory. Gradient checkpointing reduces this to $O(\sqrt{L})$ by:

1. During the forward pass, only save activations at $\sqrt{L}$ checkpoint layers; discard the rest.
2. During the backward pass, when a non-checkpointed activation is needed, recompute it from the nearest checkpoint (one additional forward pass per segment).

The tradeoff: $O(\sqrt{L})$ memory at the cost of $\approx 33\%$ extra compute. In transformer training, checkpointing is typically applied per-layer: save activations only at the boundary of each transformer block.

### Optimizer State Sharding (ZeRO)

In data-parallel training across $G$ GPUs, the naive approach replicates full optimizer state (AdamW stores $m_k, v_k, \theta$: three copies of the model) on every GPU. For a 70B parameter model in BF16, $m_k$ and $v_k$ in FP32 alone require $\approx 560$ GB.

ZeRO (Zero Redundancy Optimizer, Rajbhandari et al., 2020) partitions:
- **Stage 1**: optimizer states across $G$ GPUs. Each GPU holds $1/G$ of $m_k, v_k$.
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
<p>At random initialization the sublayer outputs are close to zero, so $J_l \approx 0$ and each Jacobian $\approx I$. Under post-norm, LayerNorm is applied to $h + \mathrm{Sublayer}(h) \approx h$, so the normalization statistics are well-behaved early on. But as training progresses and sublayer norms grow, the post-norm LayerNorm must rescale increasingly large residuals, which creates instability spikes if the learning rate is not carefully warmed up.</p>
<p>Under pre-norm, LayerNorm always sees the input $h$ before the sublayer adds to it, so its statistics are controlled at every training step. This makes pre-norm substantially less sensitive to learning rate warmup. Xiong et al. (2020) showed analytically that gradient norms at initialization are $O(L)$ times smaller in post-norm than in pre-norm, which is why post-norm requires warmup to prevent the optimizer from taking huge steps early on.</p>
<p>One trade-off: because the residual stream in pre-norm can grow unboundedly across layers (no normalization at layer boundaries), the representation norms do increase with depth. LLaMA addresses this with RMSNorm (a simplified LayerNorm without mean subtraction) and keeps weight norms in check via weight decay.</p>
</details>

**In practice:**
- Post-norm can achieve slightly better final perplexity with careful tuning, but is harder to train at large scale.
- Pre-norm is the default for all modern large-scale LLMs (GPT-2/3/4, LLaMA, Mistral, PaLM) because it is stable out of the box without per-run warmup tuning.
- LLaMA replaces LayerNorm with RMSNorm ($\mathrm{RMSNorm}(x) = x / \sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}$), dropping the mean-subtraction step for efficiency while keeping the variance-normalization benefit.

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
| ISTA | $O(n)$ | No | $O(1/k)$ | Non-smooth $R$; soft-threshold for L1 |
| FISTA | $O(n)$ | No | $O(1/k^2)$ | Nesterov-accelerated proximal |
| CG | $O(p)$/step | No | Exact in $p$ steps (quadratic) | Hessian-free via $Hv$ products |
| L-BFGS | $O(mp)$ | No | Superlinear | Full-batch only; small/medium $p$ |
| SAM | $2\times$ GD | Paired with any | Same + flatter minima | Generalization-focused |
