---
layout: post
title:  "Computational Bayesian Inference"
date:   2016-10-28 21:00:00
tag:
- Statistics
- Bayesian
projects: true
blog: true
author: YingZhang
description: Hierarchical Model and Bayesian Computation
fontsize: 23pt

---

{% include mathjax_support.html %}

## Introduction to Bayesian Framework

### Prior, Likelihood, and Posterior

* Bayes' rule:
$$
p(\theta \mid y) = \frac{p(\theta)p(y \mid \theta)}{p(y)}
$$
* Prior: $\theta \sim p(\theta)$, which encodes existing knowledge or beliefs.
* Likelihood: $y \mid \theta \sim p(y \mid \theta)$, which describes data generation under parameter $\theta$.
* Posterior: $\theta \mid y \sim p(\theta \mid y)$, which updates beliefs after seeing data.
* Up to a normalizing constant:
$$
p(\theta \mid y) \propto p(y \mid \theta)p(\theta)
$$


## Computational Bayesian Inference

The main practical challenge in Bayesian statistics is usually computational, not conceptual.
Even when the model is clearly specified, posterior computation may be hard.

### Intractable Normalizing Constant

For many models, we only know the posterior up to proportionality:

$$
\pi(\theta)=p(\theta\mid y)=\frac{\tilde{\pi}(\theta)}{Z},
\qquad
\tilde{\pi}(\theta)=p(y\mid\theta)p(\theta),
\qquad
Z=\int \tilde{\pi}(\theta)\,d\theta
$$

$Z$ (also called the partition function) is often intractable in high dimensions.
The key idea is to work with $\tilde{\pi}(\theta)$ directly and estimate expectations without explicitly evaluating $Z$.

### Transition Kernel and Invariance

For MCMC, the basic object is the transition kernel

$$
K(\theta, A)=\Pr(\theta^{(t+1)}\in A\mid\theta^{(t)}=\theta),
$$

where $A$ is any measurable set in parameter space.
For continuous state spaces, we often write $K(\theta,\theta')$ as a transition density.

Why this matters: the kernel is the algorithm.
Once $K$ is chosen (MH, Gibbs, HMC/NUTS), it determines how fast the chain explores and whether it converges to the correct posterior.

A distribution $\pi$ is called invariant (or stationary) for $K$ if

$$
\pi(A)=\int K(\theta, A)\,\pi(\theta)\,d\theta,\quad \forall A.
$$

Intuition: if the current state is distributed as $\pi$, then after one transition it is still distributed as $\pi$.

The goal is to design $K$ so that, combined with ergodicity conditions, the chain forgets its starting point and converges to $\pi$ from any initial distribution (see the expand block below for the full argument).

$$
\boxed{\text{Invariance} + \text{Ergodicity} \;\Longrightarrow\; \mathcal{L}_0 K^t \to \pi \text{ from any random start.}}
$$

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why is invariance the key to posterior sampling?</span></summary>


<p><strong>Step 1: Build $K$ so $\pi$ is invariant (fixed point).</strong></p>

<p>MCMC constructs a kernel $K$ that preserves $\pi$.
One sufficient condition is <em>detailed balance</em> (reversibility):</p>

$$\pi(\theta)\,K(\theta,\theta') = \pi(\theta')\,K(\theta',\theta).$$

<p>Integrating over $\theta$ on both sides gives invariance directly: $\pi K = \pi$.</p>

<hr>

<p><strong>Step 2: How the distribution evolves over iterations.</strong></p>

<p>Let $\mathcal{L}(\theta^{(t)})$ denote the probability distribution of state $\theta^{(t)}$ at iteration $t$.</p>

<ul>
  <li>Initialize $\theta^{(0)} \sim \mu_0$ (often overdispersed across multiple chains).</li>
  <li>For $t = 0, 1, 2, \ldots$, draw $\theta^{(t+1)} \sim K(\theta^{(t)}, \cdot)$.</li>
</ul>

<p>This gives the one-step distribution update:</p>

$$\mathcal{L}_{t+1}(A) = \int K(\theta,A)\,\mathcal{L}_t(\theta)\,d\theta.$$

<p>In operator notation: $\mathcal{L}_{t+1} = \mathcal{L}_t K$, and by induction, $\mathcal{L}_t = \mathcal{L}_0 K^t$.</p>

<hr>

<p><strong>Step 3: The gap: invariance alone does not guarantee convergence.</strong></p>

<p>Invariance says $\pi$ is a fixed point of $K$, i.e., $\pi K = \pi$.
But $K$ could have <em>other</em> fixed points, or the chain might enter periodic cycles or get stuck in isolated regions.</p>

<p>Example: a chain that deterministically alternates between two symmetric modes satisfies invariance but never mixes.</p>

<p>So $\pi K = \pi$ does not by itself tell you that $\mathcal{L}_0 K^t \to \pi$.</p>

<hr>

<p><strong>Step 4: Ergodicity closes the gap.</strong></p>

<p>A chain is <em>ergodic</em> if it satisfies three conditions:</p>

<ol>
  <li><strong>$\psi$-irreducible</strong>: from any starting point, the chain can reach any region of positive $\pi$-measure in finite steps.</li>
  <li><strong>Aperiodic</strong>: the chain does not return to any state with a fixed period $> 1$.</li>
  <li><strong>Positive Harris recurrent</strong>: the chain returns to any region of positive measure infinitely often, with finite expected return time.</li>
</ol>

<p>Under invariance + ergodicity, for <em>any</em> initial distribution $\mu_0$:</p>

$$\|\mathcal{L}_0 K^t - \pi\|_{\mathrm{TV}} \to 0 \quad \text{as } t \to \infty,$$

<p>where total variation distance measures the largest possible difference in probability any event can have under two distributions:</p>

$$\|P - Q\|_{\mathrm{TV}} = \sup_{A} \lvert P(A) - Q(A) \rvert.$$

<p>It equals zero only when $P$ and $Q$ assign identical probabilities to every event.
So the statement above means: when running the chain long enough, the distribution of the chain state becomes indistinguishable from $\pi$, no matter where the chain started.</p>

$$\boxed{\text{Invariance} + \text{Ergodicity} \;\Longrightarrow\; \mathcal{L}_0 K^t \to \pi \text{ from any random start.}}$$

<hr>

<p><strong>Step 5: How to verify ergodicity in practice.</strong></p>

<p>The three conditions above are often hard to check analytically. Use diagnostics instead:</p>

<ul>
  <li><strong>Multiple overdispersed chains</strong>: start chains from very different initial values; if they all mix to the same target, that is evidence of ergodicity.</li>
  <li><strong>$\hat{R} < 1.01$</strong>: between-chain and within-chain variance agree, confirming chains have merged.</li>
  <li><strong>ESS large enough</strong> (e.g., $\mathrm{ESS} > 400$): low ESS suggests poor mixing, a practical warning sign.</li>
  <li><strong>Trace plots overlap</strong>: all chains should interleave well after warm-up.</li>
  <li><strong>HMC-specific</strong>: zero divergences, max tree depth not saturated, $\mathrm{E\text{-}BFMI} > 0.3$.</li>
</ul>

<hr>

<p><strong>Summary workflow.</strong></p>

<ol>
  <li>Design $K$ to satisfy invariance, typically via detailed balance.</li>
  <li>Ensure (or assume from theory) the chain is ergodic.</li>
  <li>Verify mixing empirically with the diagnostics above before trusting any posterior summaries.</li>
  <li>Use post-warmup draws as approximate samples from $\pi$.</li>
</ol>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

### Core Diagnostics: ESS, $\hat{R}$, and MCSE

Before discussing specific samplers, define the diagnostics used throughout.

#### Effective Sample Size (ESS)

There are two common ESS definitions because they come from two different estimation settings.

1. MCMC ESS (autocorrelation-based): for correlated Markov chain draws.
2. IS ESS (weight-based): for weighted i.i.d. draws from importance sampling.

For an MCMC chain $x_1,\dots,x_S$ and any test function $h(x)$ (e.g., $h(x)=x$ for the mean, $h(x)=x^2$ for the second moment) with lag-$k$ autocorrelation $\rho_k = \mathrm{Corr}(h(x_t), h(x_{t+k}))$,

$$
\mathrm{ESS}_{\mathrm{MCMC}}
\approx
\frac{S}{1+2\sum_{k=1}^{\infty}\rho_k}.
$$

Use this when samples come from MH/Gibbs/HMC/NUTS.
Interpretation: dependence reduces information relative to $S$ i.i.d. samples.

<details markdown="1">
<summary><span style="color: saddlebrown; font-style: italic;">Where does this formula come from?</span></summary>


**Goal.** Estimate $\mu = \mathbb{E}_{\pi}(h(x))$ from a chain $x_1,\dots,x_S$.

The natural estimator is the sample mean $\hat\mu = \frac{1}{S}\sum_{s=1}^S h(x_s)$.
Its variance determines how precise the estimate is.

**i.i.d. baseline.** If draws were independent with $\mathrm{Var}(h(x))=\sigma^2$, then

$$
\mathrm{Var}(\hat\mu) = \frac{\sigma^2}{S}.
$$

**Correlated case.** For a stationary chain, expand $\mathrm{Var}\left(\sum_s h(x_s)\right)$ directly:

$$
\mathrm{Var}\left(\sum_{s=1}^S h(x_s)\right)
= \sum_{s=1}^S\sum_{t=1}^S \mathrm{Cov}(h(x_s), h(x_t))
= \sum_{s=1}^S\sum_{t=1}^S \sigma^2\rho_{|s-t|}.
$$

Group terms by lag $k = \lvert s-t \rvert$. At lag $k=0$ there are $S$ pairs; at lag $k \geq 1$ there are $2(S-k)$ pairs. Therefore,

$$
\mathrm{Var}\left(\sum_{s=1}^S h(x_s)\right)
= \sigma^2\left(S + 2\sum_{k=1}^{S-1}(S-k)\rho_k\right).
$$

For large $S$, $S-k \approx S$ and autocorrelations decay to zero, giving

$$
\approx \sigma^2 S\left(1 + 2\sum_{k=1}^{\infty}\rho_k\right).
$$

Dividing by $S^2$ gives $\mathrm{Var}(\hat\mu)$:

$$
\mathrm{Var}(\hat\mu)
= \frac{\sigma^2}{S}\left(1 + 2\sum_{k=1}^{\infty}\rho_k\right).
$$

**Defining ESS.** We want the number $N$ of i.i.d. draws that would give the same variance as $S$ correlated draws:

$$
\frac{\sigma^2}{N} = \frac{\sigma^2}{S}\left(1+2\sum_{k=1}^\infty\rho_k\right)
\implies
N = \frac{S}{1+2\sum_{k=1}^\infty\rho_k}.
$$

That $N$ is $\mathrm{ESS}_{\mathrm{MCMC}}$. When $\rho_k>0$ (positive autocorrelation, typical in MCMC), the denominator exceeds 1 and $\mathrm{ESS}<S$.

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

For importance sampling with normalized weights $\bar{w}_s$,

$$
\mathrm{ESS}_{\mathrm{IS}} = \frac{1}{\sum_{s=1}^S \bar{w}_s^2},
\qquad
\bar{w}_s = \frac{w_s}{\sum_j w_j}.
$$

Use this when estimates are weighted averages under IS.
Interpretation: uneven weights reduce the effective number of contributing draws.

When to use which:

* If your estimator is based on a Markov chain average, use $\mathrm{ESS}_{\mathrm{MCMC}}$.
* If your estimator is based on weighted proposal draws, use $\mathrm{ESS}_{\mathrm{IS}}$.

#### Potential Scale Reduction Factor ($\hat{R}$)

$\hat{R}$ is a multi-chain convergence diagnostic.
Motivation: if chains from different initial values have all mixed to the same target, between-chain and within-chain variance should agree.

Suppose $m$ chains each have length $n$ after warm-up.
Let draws for chain $j$ be $\theta_{1j},\dots,\theta_{nj}$.
Define chain mean and sample variance:

$$
\bar{\theta}_{\cdot j}=\frac{1}{n}\sum_{i=1}^n \theta_{ij},
\qquad
s_j^2=\frac{1}{n-1}\sum_{i=1}^n\left(\theta_{ij}-\bar{\theta}_{\cdot j}\right)^2.
$$

The grand mean and the two variance components are:

$$
\bar{\theta}_{\cdot\cdot}=\frac{1}{m}\sum_{j=1}^m\bar{\theta}_{\cdot j},
\qquad
W=\frac{1}{m}\sum_{j=1}^m s_j^2,
\qquad
B=\frac{n}{m-1}\sum_{j=1}^m\left(\bar{\theta}_{\cdot j}-\bar{\theta}_{\cdot\cdot}\right)^2.
$$

A standard form is

$$
\hat{V} = \frac{n-1}{n}W + \frac{1}{n}B,
\qquad
\hat{R}=\sqrt{\frac{\hat{V}}{W}}.
$$

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Where does $\hat{V}=\frac{n-1}{n}W+\frac{1}{n}B$ come from?</span></summary>

Use a random-effects view for post-warmup draws:

$$
\theta_{ij}=\mu+a_j+\varepsilon_{ij},
\qquad
\operatorname{Var}(a_j)=\tau^2,
\qquad
\operatorname{Var}(\varepsilon_{ij})=\sigma^2.
$$

Then an individual draw has variance

$$
\operatorname{Var}(\theta_{ij})=\tau^2+\sigma^2.
$$

Interpretation:

<ul>
  <li>$W$ estimates within-chain variance, so $W\approx\sigma^2$.</li>
  <li>$B/n$ estimates variance of chain means $\bar\theta_{\cdot j}$, so</li>
</ul>

Since $\bar\theta_{\cdot j}=\mu+a_j+\bar\varepsilon_{\cdot j}$ and $\operatorname{Var}(\bar\varepsilon_{\cdot j})=\sigma^2/n$, the chain mean has variance $\tau^2+\sigma^2/n$, giving

$$
\frac{B}{n}\approx \tau^2+\frac{\sigma^2}{n}.
$$

Subtracting the averaging term gives

$$
\hat\tau^2\approx \frac{B}{n}-\frac{W}{n}.
$$

Estimate total variance by

$$
\hat V
=
W+\hat\tau^2
=
W+\left(\frac{B}{n}-\frac{W}{n}\right)
=
\frac{n-1}{n}W+\frac{1}{n}B.
$$

So $\hat V$ is a pooled variance estimate combining within-chain variability ($W$) and between-chain information ($B$).

Consequences:

<ul>
  <li>If chains mix well, $B\approx W$, so $\hat V\approx W$ and $\hat R\approx 1$.</li>
  <li>If chains have not mixed, $B>W$, so $\hat V>W$ and $\hat R>1$.</li>
</ul>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

How to use:

1. Run at least 4 chains with dispersed initial values.
2. Check $\hat{R}$ for each important parameter and functional.
3. Practical target: $\hat{R}\approx 1$ (often $<1.01$).
4. If large, run longer and/or reparameterize.

#### Monte Carlo Standard Error (MCSE)

MCSE quantifies given the simulation how precisely $\hat\mu_f$ estimates $\mathbb{E}_{\pi}f(\theta).$ 

It combines two sources: posterior spread (large $\mathrm{Var}_\pi(f)$ makes any estimate harder to pin down) and simulation efficiency (small ESS inflates error). Concretely,

$$
\mathrm{MCSE}(\hat\mu_f) \approx \sqrt{\frac{\widehat{\mathrm{Var}}(f)}{\mathrm{ESS}}}.
$$

Large MCSE means either the posterior variance of $f(\theta)$ is large, or ESS is small (chain is inefficient). Small MCSE means the simulation has enough effective draws to estimate $\hat\mu_f$ reliably.

**Why ESS appears here.** For i.i.d. draws, $\mathrm{MCSE} = \sigma_f/\sqrt{S}$. For a correlated chain, autocorrelation reduces the effective information to $\mathrm{ESS} \leq S$, so replace $S$ with $\mathrm{ESS}$.

Practical estimation:

1. Compute $\widehat{\mathrm{Var}}(f) = \frac{1}{S-1}\sum_{s=1}^S\bigl(f(\theta^{(s)})-\hat\mu_f\bigr)^2$.
2. Compute $\widehat{\mathrm{ESS}}$ from estimated lag autocorrelations $\hat\rho_k$ of $f(\theta^{(1)}),\dots,f(\theta^{(S)})$:
$$
\widehat{\mathrm{ESS}} = \frac{S}{1 + 2\sum_{k=1}^{K}\hat\rho_k},
$$
where the sum is truncated at the first $K$ such that $\hat\rho_K$ drops below a noise threshold.
3. Plug into $\mathrm{MCSE} \approx \sqrt{\widehat{\mathrm{Var}}(f)/\widehat{\mathrm{ESS}}}$.
4. **Reporting rule**: MCSE should be small relative to posterior SD $\sqrt{\widehat{\mathrm{Var}}(f)}$, or relative to decision-relevant effect sizes.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Split-bulk-ESS: making ESS robust to non-stationarity and non-normality</span></summary>

<p>The posterior $\pi(\theta)$ is always a fixed distribution, stationary by definition. What may not be stationary is the Markov chain used to sample it. After discarding warm-up, you assume the chain has converged and is sampling from $\pi$. But a chain can still be drifting in practice, e.g., stuck in one mode and slowly moving toward another. The chain's distribution at draw 100 may then differ from draw 5000, even though warm-up was discarded. In that situation the plain ESS formula is wrong: it assumes stationarity and computes autocorrelations as if all draws come from the same distribution. A drifting chain has artificially low autocorrelations (adjacent draws look similar, but far-apart draws look different for the wrong reason), so ESS is overestimated. Split-bulk-ESS catches this by comparing the two halves of each chain. ESS is meaningful only when the chain has converged; split-$\hat{R}$ is the check that tells you whether that assumption holds.</p>

<p>Concretely, the plain ESS formula can fail in two ways:</p>
<ol>
  <li>A drifting chain overestimates ESS because early and late draws come from different parts of the space.</li>
  <li>Heavy-tailed draws inflate variance estimates and distort autocorrelation estimates.</li>
</ol>
<p>Split-bulk-ESS fixes both.</p>

<p><strong>Step 1: Split.</strong> Cut each of the $m$ chains of length $S$ into two halves of length $n = S/2$, giving $2m$ half-chains. This exposes within-chain non-stationarity: if a chain drifts, its two halves look like they come from different distributions and the between-half $\hat{R}$ exceeds 1.</p>

<p><strong>Step 2: Rank-normalize.</strong> Sort all $2m \cdot n = mS$ draws together and map each to a $z$-score via $z = \Phi^{-1}\!\left(\frac{r-3/8}{mS+1/4}\right)$. This removes dependence on the original scale and tail shape.</p>

<p><strong>Step 3: Estimate autocorrelations via the variogram.</strong> For each lag $k$, pool across all $2m$ half-chains to form the variogram estimate</p>

$$
\hat\gamma_k = \frac{1}{2m(n-k)}\sum_{j=1}^{2m}\sum_{i=1}^{n-k}(\tilde\theta_{i+k,j}-\tilde\theta_{i,j})^2,
$$

<p>where $\tilde\theta_{i,j}$ denotes the rank-normalized draw $i$ in half-chain $j$. The lag-$k$ autocorrelation is then</p>

$$
\hat\rho_k = 1 - \frac{\hat\gamma_k}{2\hat{V}},
$$

<p>where $\hat{V} = \frac{n-1}{n}W + \frac{1}{n}B$ is the pooled variance estimator from $\hat{R}$ (using the $2m$ half-chains). Pooling across half-chains gives more stable $\hat\rho_k$ estimates than using a single chain.</p>

<p><strong>Step 4: Bulk-ESS.</strong> Plug into the standard formula with total draws $N = mS$:</p>

$$
\mathrm{Bulk\text{-}ESS} = \frac{mS}{1 + 2\sum_{k=1}^{K}\hat\rho_k}.
$$

<p><strong>Tail-ESS</strong> is computed the same way but replaces $\tilde\theta_{i,j}$ with the indicator $\mathbf{1}[\tilde\theta_{i,j} \leq q]$ for a tail quantile $q$ (e.g., 5th or 95th percentile), measuring how well the tails are sampled.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

### Sampling from a Hard Distribution

Recall that the posterior is $\pi(\theta) = \tilde\pi(\theta)/Z$, where $\tilde\pi(\theta) = p(y\mid\theta)p(\theta)$ is the unnormalized posterior (likelihood times prior) and $Z = \int\tilde\pi(\theta)\,d\theta$ is the intractable normalizing constant. Direct sampling from $\pi(\theta)$ is impossible when $Z$ cannot be computed. MCMC sidesteps this by building a Markov chain that only requires evaluating $\tilde\pi(\theta)$, since $Z$ cancels in all acceptance ratios. After warm-up, samples from the chain approximate posterior draws.

#### 1) Metropolis-Hastings (MH)

Given current state $\theta^{(t)}$:

1. Propose $\theta' \sim q(\theta'\mid\theta^{(t)})$.
2. Accept with probability

$$
\alpha = \min\left(1,
\frac{\tilde{\pi}(\theta')\,q(\theta^{(t)}\mid\theta')}
{\tilde{\pi}(\theta^{(t)})\,q(\theta'\mid\theta^{(t)})}
\right)
$$

Notice $Z$ cancels in the ratio, which is why MH works for unnormalized targets.

The transition kernel is

$$
K(\theta,\theta')=q(\theta'\mid\theta)\alpha(\theta,\theta')
\quad (\theta'\neq\theta),
$$

with remaining mass at $\theta$ as rejection probability.

* Pros: simple and broadly applicable because it only requires evaluating $\tilde{\pi}(\theta)$ up to proportionality.
* Cons: can mix slowly in high dimensions because local random proposals struggle to cross low-density regions; proposal scale strongly controls acceptance and movement.
* What to check: acceptance rate (diagnoses proposal mismatch), trace/autocorrelation (diagnose dependence), ESS and $\hat{R}$ (already defined above).

Practical diagnostic guide:

1. Acceptance rate: if near 0, proposals are too aggressive; if near 1 with tiny moves, chain is too sticky. Tune proposal scale to balance movement and acceptance.
2. Trace plots across chains: you want overlapping, stationarity-like behavior without long trends.
3. Autocorrelation function (ACF): slowly decaying ACF indicates high dependence and low effective information.
4. $\hat{R}$ and ESS: target $\hat{R}$ close to 1 (for example $<1.01$) and sufficiently large ESS for each reported quantity.

<details>
	<summary><span style="color: saddlebrown; font-style: italic;">Why does MH have the right invariant distribution?</span></summary>


MH is designed so that its transition kernel satisfies detailed balance with respect to $\pi$:

$$
\pi(\theta)K(\theta,\theta')=\pi(\theta')K(\theta',\theta).
$$

For $\theta'\neq\theta$,

$$
K(\theta,\theta')=q(\theta'\mid\theta)\alpha(\theta,\theta'),
$$

and with

$$
\alpha(\theta,\theta')=
\min\left(1,
\frac{\tilde{\pi}(\theta')q(\theta\mid\theta')}{\tilde{\pi}(\theta)q(\theta'\mid\theta)}
\right),
$$

one can verify both sides are equal case-by-case (ratio above or below 1).
Detailed balance implies invariance, and invariance plus ergodicity gives convergence to $\pi$.

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

#### 2) Gibbs Sampling

When full conditionals are easy to sample, update block by block:

$$
\theta_1^{(t+1)} \sim p(\theta_1\mid\theta_2^{(t)},\dots,\theta_d^{(t)},y),
$$
$$
\dots,
$$
$$
\theta_d^{(t+1)} \sim p(\theta_d\mid\theta_1^{(t+1)},\dots,\theta_{d-1}^{(t+1)},y)
$$

* Pros: often straightforward in conjugate/hierarchical models because full conditionals have closed forms and every update is accepted.
* Cons: requires tractable full conditionals; mixing can be slow under strong posterior dependence because one-coordinate-at-a-time moves change state only incrementally.
* What to check: cross-chain agreement and $\hat{R}$ (global convergence), slow drift across blocks (poor mixing), ESS per parameter and posterior correlations (where dependence hurts efficiency).

Practical diagnostic guide:

1. Block-wise trace behavior: if one block moves while others barely move, reparameterization or blocking may be needed.
2. Pairwise posterior scatter/correlation: very high correlations often explain slow Gibbs mixing.
3. ESS by parameter: low ESS for hyperparameters is common and should be explicitly monitored.
4. Multiple starting values: chains started far apart should converge to the same marginal distributions.

<details>
	<summary><span style="color: saddlebrown; font-style: italic;">Why does Gibbs sampling target the correct distribution?</span></summary>


Let $\theta=(\theta_1,\dots,\theta_d)$ and suppose each update samples the exact full conditional.
Each coordinate update leaves $\pi(\theta)$ invariant:

$$
\pi(\theta_{-j})\,\pi(\theta_j\mid\theta_{-j})
\xrightarrow{\text{resample }\theta_j}
\pi(\theta_{-j})\,\pi(\theta_j\mid\theta_{-j}).
$$

Composing invariant kernels (for $j=1,\dots,d$) keeps $\pi$ invariant. Under standard ergodicity conditions, the Gibbs chain converges to $\pi$.

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

#### 3) Hamiltonian Monte Carlo (HMC / NUTS)

For continuous high-dimensional posteriors, gradient-based samplers can mix much faster than random-walk MH.

* HMC uses gradients of $\log \tilde{\pi}(\theta)$ to propose long-distance moves with high acceptance.
* NUTS (No-U-Turn Sampler) adaptively selects trajectory length and is often a strong default in modern Bayesian software.

A typical HMC proposal evolves Hamiltonian dynamics on

$$
H(\theta,r)=U(\theta)+K(r),
\qquad
U(\theta)=-\log \tilde{\pi}(\theta),
\qquad
K(r)=\frac{1}{2}r^TM^{-1}r.
$$

Definitions:

* $r$: auxiliary momentum variable (same dimension as $\theta$), usually drawn from $\mathcal{N}(0,M)$ each iteration.
* $M$: mass matrix (positive-definite), which rescales geometry; in practice often adapted during warm-up.

What HMC/NUTS are in one sentence:

* HMC: simulate physics-inspired trajectories using gradients, then accept/reject with MH correction.
* NUTS: an adaptive HMC variant that automatically chooses trajectory length via a no-U-turn stopping rule.

How to use HMC/NUTS in practice:

1. Choose a differentiable posterior model and parameterization.
2. Warm-up/adapt step size $\epsilon$ and mass matrix $M$.
3. Simulate trajectories with leapfrog integration.
4. Apply Metropolis correction for exactness.
5. Monitor diagnostics (divergences, tree-depth, ESS, $\hat{R}$, MCSE).

What leapfrog integration means:

It is a symplectic, reversible numerical integrator for Hamiltonian dynamics that alternates half-step momentum and full-step position updates; this makes long trajectories stable enough for MCMC and supports MH correction.

* Pros: high efficiency in smooth continuous models because gradient-informed proposals move far while preserving high acceptance, improving ESS per compute.
* Cons: needs differentiability and good geometry; poor scaling/parameterization causes unstable trajectories and biased exploration.
* What to check: divergences and energy diagnostics (numerical/geometric mismatch), tree-depth saturation (trajectory truncation), $\hat{R}$ and ESS (final sampling quality).

Practical diagnostic guide:

1. Divergences: numerical integration fails to track the Hamiltonian path well; often indicates bad geometry (funnel, strong curvature) or too-large step size.
2. Tree-depth saturation (NUTS): hitting max tree depth often means trajectories are truncated early and exploration is inefficient.
3. Energy diagnostics: check E-BFMI or energy overlap; poor behavior means momentum resampling is not exploring energy levels effectively.
4. Reparameterization check: if diagnostics remain poor, use non-centered or scaled parameterizations and rerun.

<details>
	<summary><span style="color: saddlebrown; font-style: italic;">Why does HMC target the correct posterior?</span></summary>


HMC augments the target with momentum $r$:

$$
\pi(\theta,r) \propto \exp\{-H(\theta,r)\}
= \exp\{-U(\theta)\}\exp\{-K(r)\}.
$$

Ideal Hamiltonian flow preserves both volume and total energy $H$, so it preserves $\pi(\theta,r)$.
In practice, leapfrog integration introduces small discretization error, and a Metropolis correction step with acceptance

$$
\alpha=\min\{1,\exp[-H(\theta',r')+H(\theta,r)]\}
$$

restores exact invariance of the joint target. Marginalizing out $r$ gives the desired posterior $\pi(\theta)$.

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

### Estimating Posterior Expectations

Many Bayesian summaries are expectations:

$$
\mathbb{E}_{\pi}[f(\theta)] = \int f(\theta)\pi(\theta)d\theta
$$

#### 1) Monte Carlo Estimator from Posterior Samples

If $\theta^{(1)},\dots,\theta^{(S)} \sim \pi$, then

$$
\hat{\mu}_f = \frac{1}{S}\sum_{s=1}^S f\big(\theta^{(s)}\big) \approx \mathbb{E}_{\pi}(f(\theta))
$$

Note: $\widehat{\mathrm{Var}}(f) = \frac{1}{S-1}\sum_s(f(\theta^{(s)})-\hat\mu_f)^2$ estimates the posterior variance $\mathrm{Var}_{\pi}(f(\theta))$, i.e., how spread out $f(\theta)$ is under the posterior, not the estimation error of $\hat\mu_f$ itself.
With MCMC samples, draws are correlated, so precision depends on effective sample size (ESS), not only raw sample size.

* Pros: conceptually simple because posterior expectations reduce to empirical averages of transformed draws; applies to almost any $f$.
* Cons: Monte Carlo variance can be high, and with correlated samples effective information grows slower than sample count.
* What to check: MCSE (numerical precision of estimate), ESS (effective information), and running-estimate stability (whether additional draws still shift conclusions).

Practical diagnostic guide:

1. Running mean plots for key functionals: curves should flatten as samples accumulate.
2. MCSE relative to posterior SD: ensure Monte Carlo error is small enough for decision-making.
3. Function-specific ESS: ESS should be checked for each important $f(\theta)$, not only for raw parameters.

#### 2) Importance Sampling (IS)

When sampling directly from $\pi$ is hard, draw from a proposal $q(\theta)$ and reweight:

$$
\mathbb{E}_{\pi}[f(\theta)]
=
\frac{\int f(\theta)\,\tilde{\pi}(\theta)\,d\theta}{\int \tilde{\pi}(\theta)\,d\theta}
\approx
\frac{\sum_{s=1}^{S} w_s f(\theta^{(s)})}{\sum_{s=1}^{S} w_s},
\quad \theta^{(s)}\sim q,
\quad w_s=\frac{\tilde{\pi}(\theta^{(s)})}{q(\theta^{(s)})}
$$

The self-normalized form above is common when $Z$ is unknown.

The normalized weights are

$$
\bar{w}_s = \frac{w_s}{\sum_{j=1}^S w_j},
$$

and a common weight-based ESS diagnostic is

$$
\mathrm{ESS}_{\mathrm{IS}} = \frac{1}{\sum_{s=1}^S \bar{w}_s^2}.
$$

* Pros: no Markov dependence because samples are i.i.d. from $q$; also enables post-hoc reweighting and sensitivity analysis without rerunning chains.
* Cons: unstable when $q$ undercovers target mass/tails, because a few very large weights dominate and inflate variance.
* What to check: normalized weight concentration and maximum weight share (dominance risk), $\mathrm{ESS}_{\mathrm{IS}}$ (usable sample size), and tail mismatch diagnostics between $q$ and target.

Practical diagnostic guide:

1. Weight histogram / log-weight spread: extreme right tails indicate unstable estimators.
2. Maximum normalized weight $\max_s \bar{w}_s$: large values mean one or few points dominate.
3. $\mathrm{ESS}_{\mathrm{IS}}$: if very small relative to $S$, improve proposal or use tempering/bridging.
4. Support check: verify $q(\theta)>0$ wherever target has mass; support mismatch invalidates IS.

<details>
	<summary><span style="color: saddlebrown; font-style: italic;">Why does importance sampling estimate target expectations correctly?</span></summary>


If $q(\theta)>0$ whenever $\tilde{\pi}(\theta)>0$, then

$$
\mathbb{E}_{\pi}[f(\theta)]
=
\frac{\int f(\theta)\tilde{\pi}(\theta)d\theta}{\int \tilde{\pi}(\theta)d\theta}
=
\frac{\int f(\theta)\frac{\tilde{\pi}(\theta)}{q(\theta)}q(\theta)d\theta}{\int \frac{\tilde{\pi}(\theta)}{q(\theta)}q(\theta)d\theta}.
$$

So with $\theta^{(s)}\sim q$ and $w_s=\tilde{\pi}(\theta^{(s)})/q(\theta^{(s)})$, the self-normalized estimator is the Monte Carlo plug-in for numerator and denominator.

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>

#### Monte Carlo Standard Error (MCSE)

MCSE was defined earlier; in practice always report MCSE together with posterior estimates, especially when comparing close means or effects.

### Approximate Inference

Two common approximations are variational inference and Laplace approximation.

#### Variational Inference (VI)

Fit a tractable family $q_\lambda(\theta)$ by optimizing ELBO.

Start from

$$
\log p(y)=\log \int q_\lambda(\theta)\frac{p(y,\theta)}{q_\lambda(\theta)}d\theta.
$$

Apply Jensen's inequality:

$$
\log p(y)
\ge
\mathbb{E}_{q_\lambda}\!\left[\log p(y,\theta)-\log q_\lambda(\theta)\right]
\equiv
\mathrm{ELBO}(q_\lambda).
$$

Now add and subtract the same expectation to get the standard decomposition:

$$
\log p(y)
=
\mathrm{ELBO}(q_\lambda)
+
\mathrm{KL}\big(q_\lambda(\theta)\,\|\,p(\theta\mid y)\big),
$$

with

$$
\mathrm{ELBO}(q_\lambda)
=
\mathbb{E}_{q_\lambda}[\log p(y,\theta)]
-
\mathbb{E}_{q_\lambda}[\log q_\lambda(\theta)].
$$

Since KL is nonnegative, maximizing ELBO is equivalent to minimizing
$\mathrm{KL}\big(q_\lambda(\theta)\,\|\,p(\theta\mid y)\big)$.

* Pros: fast and scalable because optimization is typically cheaper than long-run simulation.
* Cons: approximation bias can be substantial because restricted variational families may not capture dependence, skewness, or multimodality; uncertainty is often underestimated.
* What to check: sensitivity to variational family choice, posterior predictive calibration, and spot-checks against MCMC on manageable subsets.

Practical diagnostic guide:

1. Variational family sensitivity: compare mean-field vs richer families; large changes imply approximation fragility.
2. Posterior predictive checks: compare replicated-data summaries against observed data to detect misfit.
3. Calibration of uncertainty: verify interval coverage on known or simulated benchmarks when possible.
4. Local MCMC spot-check: for a reduced model/data subset, compare VI means/intervals with MCMC.

#### Laplace Approximation

Approximate posterior near mode $\hat{\theta}$ by a Gaussian based on local curvature.

Let

$$
\hat{\theta}=\arg\max_\theta \log p(\theta\mid y),
\qquad
H = -\nabla^2_{\theta}\log p(\theta\mid y)\big\vert_{\theta=\hat{\theta}}.
$$

Then

$$
p(\theta\mid y) \approx \mathcal{N}(\hat{\theta}, H^{-1}).
$$

* Pros: very fast because it uses only local curvature at a mode; useful for initialization and rough baselines.
* Cons: unreliable for multimodal, skewed, or heavy-tailed posteriors because a single local Gaussian cannot represent global shape.
* What to check: posterior-shape diagnostics, sensitivity to which mode is found, and comparisons to simulation-based estimates for key functionals.

Practical diagnostic guide:

1. Posterior-shape diagnostics: inspect skewness, kurtosis, and multimodality indicators (for example density plots or profile likelihood/posterior slices).
2. Mode sensitivity: run optimization from multiple starts; different converged modes imply one Gaussian is insufficient.
3. Curvature stability: check Hessian conditioning and positive definiteness near the mode.
4. Cross-method comparison: compare Laplace summaries with MCMC/VI for key posterior means and intervals.

### A Practical Workflow for Hard Posteriors

1. Define unnormalized target $\tilde{\pi}(\theta)$ and support constraints.
2. Reparameterize to improve geometry (for example log scale for positive parameters).
3. Choose toolkit based on structure:
   * Gibbs: tractable full conditionals.
   * HMC/NUTS: smooth continuous models.
   * MH: custom/non-differentiable settings.
   * IS: expectation estimation with a good proposal.
   * VI/Laplace: speed-first approximation.
4. Run diagnostics ($\hat{R}$, ESS, trace behavior, weight diagnostics for IS).
5. Report both posterior uncertainty and Monte Carlo uncertainty (MCSE).
6. Validate with posterior predictive checks and sensitivity analyses.



## Priors and Hierarchical Modeling

<details>
	<summary><span style="color: saddlebrown; font-style: italic;">What are the prior choices and how does the hierarchical rat-tumor example work?</span></summary>


<p><strong>Types of Prior</strong></p>

<p>The two most common practical choices in this post are weak/non-informative priors and Jeffreys-type reference priors.</p>

<p><strong>Jeffreys Prior</strong></p>

<p>Another reference prior is Jeffreys prior:</p>

$$
p(\theta) \propto \sqrt{J(\theta)}
$$

<p>where $J(\theta)$ is Fisher information:</p>

$$
J(\theta)=\mathbb{E}\left[\left(\frac{\partial}{\partial\theta}\log p(y\mid\theta)\right)^2\middle\vert\theta\right]
= -\mathbb{E}\left[\frac{\partial^2}{\partial\theta^2}\log p(y\mid\theta)\middle\vert\theta\right]
$$

<p>Key property (invariance): under a one-to-one reparameterization $\phi = h(\theta)$, Jeffreys prior remains form-invariant.</p>

<p><strong>Bayesian Hierarchical Model</strong></p>

<p><strong>Setting Hyperparameters</strong></p>

<p>Consider a classic example: tumor incidence in rat studies (Tarone, 1982).</p>

<ul>
  <li>Let $y_i$ be the number of rats with tumor in group $i$.</li>
  <li>Let $n_i$ be the total rats in group $i$.</li>
  <li>We observe rates $y_i/n_i$ across many groups.</li>
</ul>

<p>The key question is how to set the prior hyperparameters in a principled way.</p>

<p><strong>Model Initialization</strong></p>

<ul>
  <li>Let $\theta_i$ be the tumor probability for group $i$.</li>
  <li>Sampling model:</li>
</ul>

$$
y_i \mid \theta_i \sim \text{Binomial}(n_i,\theta_i)
$$

<ul>
  <li>Prior model:</li>
</ul>

$$
\theta_i \sim \text{Beta}(\alpha,\beta)
$$

<ul>
  <li>Posterior for each group:</li>
</ul>

$$
\theta_i \mid y_i \sim \text{Beta}(\alpha+y_i,\beta+n_i-y_i)
$$

<p><strong>Toy Example</strong></p>

<ul>
  <li>How should we choose $\alpha$ and $\beta$?</li>
  <li>These parameters are called hyperparameters.</li>
</ul>

<p><strong>Prior Specification</strong></p>

<p><strong>Fixed Prior Distribution</strong></p>

<p>Informative prior:</p>

<ul>
  <li>Suppose prior mean $m$ and variance $v$ are known from domain knowledge.</li>
  <li>Match moments to Beta$(\alpha,\beta)$:</li>
</ul>

$$
\alpha = m\left(\frac{m(1-m)}{v}-1\right),\qquad
\beta = (1-m)\left(\frac{m(1-m)}{v}-1\right)
$$

<p><strong>Empirical Bayes</strong></p>

<ul>
  <li>Empirical Bayes idea: estimate $\alpha,\beta$ from historical groups, then plug them into the prior.</li>
  <li>This can work well in practice, but be careful about using the same data twice.</li>
  <li>If the same data are used to both estimate hyperparameters and update the posterior, uncertainty may be underestimated.</li>
</ul>

<p><strong>Non-Informative Prior</strong></p>

<p>Do we have to use data to set the hyperparameters?</p>

<ul>
  <li>Often, scientific prior information is limited.</li>
  <li>A default choice is a weakly informative or non-informative prior, such as</li>
</ul>

$$
\theta_i \sim \text{Uniform}(0,1)=\text{Beta}(1,1)
$$

<p>with</p>

$$
y_i \mid \theta_i \sim \text{Binomial}(n_i,\theta_i)
$$

<p><strong>Hierarchical Prior</strong></p>

<p>Yes. This is exactly the hierarchical Bayesian approach.</p>

<ul>
  <li>Add one more level:</li>
</ul>

$$
y_i \mid \theta_i \sim \text{Binomial}(n_i,\theta_i)
$$
$$
\theta_i \mid \alpha,\beta \sim \text{Beta}(\alpha,\beta)
$$
$$
\alpha \sim \text{Gamma}(a_\alpha,b_\alpha),\qquad
\beta \sim \text{Gamma}(a_\beta,b_\beta)
$$

<ul>
  <li>This model propagates hyperparameter uncertainty into posterior inference.</li>
  <li>It also enables partial pooling across groups, which stabilizes noisy group-level estimates.</li>
</ul>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>

</details>
