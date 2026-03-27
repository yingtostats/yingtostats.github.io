---
layout: post
title:  "Introduction to Hierarchical Model"
date:   2016-10-28 21:00:00
tag:
- Statistics
- Bayesian
projects: true
blog: true
author: YingZhang
description: Hierarchical Model and Latent Dirichlet Allocation
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



## Types of Prior

Where can prior information come from?

* Past experience
* Historical studies
* Expert knowledge

Common categories of priors:

* Informative priors: carry substantial domain information.
* Weakly informative priors: regularize estimates without being overly restrictive.
* Non-informative (or reference) priors: aim to minimize prior influence.


### Common Types of Priors
What priors are often used in practice?

* Expert priors: elicited from subject-matter experts.
* Conjugate priors: posterior stays in the same distribution family as the prior.
* Non-informative priors: for example, uniform or Jeffreys priors.





#### Conjugate Prior

* Definition: prior and posterior belong to the same family.
* Example:
$$
	heta \sim \text{Beta}(\alpha,\beta),\qquad y \mid \theta \sim \text{Binomial}(n,\theta)
$$
Then
$$
p(\theta \mid y) \propto p(y \mid \theta)p(\theta)
\propto \theta^{\alpha+y-1}(1-\theta)^{\beta+n-y-1}
$$
so
$$
	heta \mid y \sim \text{Beta}(\alpha+y,\beta+n-y)
$$




##### Why we use Conjugate Prior?

* They simplify posterior computation and interpretation.
* They are useful for quick analytical checks and teaching examples.




#### Non-informative Prior

* A common non-informative choice is $p(\theta) \propto 1$.
* On an unbounded domain, this is an improper prior (it does not integrate to 1).
* Improper priors can still be useful if they produce a proper posterior.

Example:
$$
p(\theta) \propto 1,\qquad y \mid \theta \sim N(\theta,1)
$$
Then
$$
p(\theta \mid y) \propto p(y \mid \theta) \propto \exp\left(-\frac{(y-\theta)^2}{2}\right)
$$
so
$$
	heta \mid y \sim N(y,1)
$$
which is a proper posterior.




##### Jeffreys Prior

* Another reference prior is Jeffreys prior:
$$
p(\theta) \propto \sqrt{J(\theta)}
$$
where $J(\theta)$ is Fisher information:
$$
J(\theta)=\mathbb{E}\left[\left(\frac{\partial}{\partial\theta}\log p(y\mid\theta)\right)^2\middle\vert\theta\right]
= -\mathbb{E}\left[\frac{\partial^2}{\partial\theta^2}\log p(y\mid\theta)\middle\vert\theta\right]
$$
* Key property (invariance): under a one-to-one reparameterization $\phi = h(\theta)$, Jeffreys prior remains form-invariant.





## Bayesian Hierarchical Model

### How to set the Hyperparameters?

Consider a classic example: tumor incidence in rat studies (Tarone, 1982).

* Let $y_i$ be the number of rats with tumor in group $i$.
* Let $n_i$ be the total rats in group $i$.
* We observe rates $y_i/n_i$ across many groups.

The key question is how to set the prior hyperparameters in a principled way.


#### Model Initialization

* Let $\theta_i$ be the tumor probability for group $i$.
* Sampling model:
$$
y_i \mid \theta_i \sim \text{Binomial}(n_i,\theta_i)
$$
* Prior model:
$$
	heta_i \sim \text{Beta}(\alpha,\beta)
$$
* Posterior for each group:
$$
	heta_i \mid y_i \sim \text{Beta}(\alpha+y_i,\beta+n_i-y_i)
$$


#### Toy Example
* How should we choose $\alpha$ and $\beta$?
* These parameters are called hyperparameters.

#### How to set the priors?

##### Fixed Prior Distribution

Informative prior:

* Suppose prior mean $m$ and variance $v$ are known from domain knowledge.
* Match moments to Beta$(\alpha,\beta)$:
$$
\alpha = m\left(\frac{m(1-m)}{v}-1\right),\qquad
\beta = (1-m)\left(\frac{m(1-m)}{v}-1\right)
$$

##### Approximate estimate using Historical Data

* Empirical Bayes idea: estimate $\alpha,\beta$ from historical groups, then plug them into the prior.
* This can work well in practice, but be careful about using the same data twice.
* If the same data are used to both estimate hyperparameters and update the posterior, uncertainty may be underestimated.

##### Set the Hyperparameters without Data

Do we have to use data to set the hyperparameters?

* Often, scientific prior information is limited.
* A default choice is a weakly informative or non-informative prior, such as
$$
	heta_i \sim \text{Uniform}(0,1)=\text{Beta}(1,1)
$$
with
$$
y_i \mid \theta_i \sim \text{Binomial}(n_i,\theta_i)
$$

##### Can we regard hyperparameters in prior as random variables?

Yes. This is exactly the hierarchical Bayesian approach.

* Add one more level:
$$
y_i \mid \theta_i \sim \text{Binomial}(n_i,\theta_i)
$$
$$
	heta_i \mid \alpha,\beta \sim \text{Beta}(\alpha,\beta)
$$
$$
\alpha \sim \text{Gamma}(a_\alpha,b_\alpha),\qquad
\beta \sim \text{Gamma}(a_\beta,b_\beta)
$$
* This model propagates hyperparameter uncertainty into posterior inference.
* It also enables partial pooling across groups, which stabilizes noisy group-level estimates.

## Latent Dirichlet Allocation

* LDA is a classic hierarchical Bayesian model for text.
* It represents each document as a mixture of latent topics.

### Model Initialization

#### From Beta Distribution to Dirichlet

* Beta-Binomial is a 2-category conjugate pair.
* Dirichlet-Multinomial generalizes this to multiple categories.
* If
$$
\vec{p} \sim \text{Dir}(\vec{\alpha}),\qquad
\vec{n} \mid \vec{p} \sim \text{Multinomial}(N,\vec{p})
$$
then
$$
\vec{p} \mid \vec{n} \sim \text{Dir}(\vec{\alpha}+\vec{n})
$$

#### Notation and Assumption

* Vocabulary size: $V$.
* Number of topics: $K$.
* A document is a sequence of words $\mathbf{w}_d=(w_{d1},\ldots,w_{dN_d})$.
* A corpus contains $M$ documents.
* Bag-of-words assumption: word order is ignored within each document.

### Where is the "Latent" in LDA?

The latent variables are topic assignments $z_{dn}$ and document-level topic proportions $\theta_d$.

Generative process for LDA:

* For each topic $k$, draw topic-word distribution
$$
\beta_k \sim \text{Dirichlet}(\eta)
$$
* For each document $d$, draw topic mixture
$$
	heta_d \sim \text{Dirichlet}(\alpha)
$$
* For each word position $n$ in document $d$:
$$
z_{dn} \mid \theta_d \sim \text{Categorical}(\theta_d)
$$
$$
w_{dn} \mid z_{dn},\beta \sim \text{Categorical}(\beta_{z_{dn}})
$$

So “latent” refers to unobserved topic structure $(\theta,z)$ behind observed words $w$.


### Posterior Inference

#### Intractable Posterior

* Goal:
$$
p(\theta,\mathbf{z} \mid \mathbf{w},\alpha,\beta)
= \frac{p(\theta,\mathbf{z},\mathbf{w}\mid\alpha,\beta)}{p(\mathbf{w}\mid\alpha,\beta)}
$$
* The denominator requires summing/integrating over many latent configurations, so exact inference is generally intractable.

Common approximation methods:

* Variational inference: optimize a tractable lower bound (ELBO).
* Collapsed Gibbs sampling: sample topic assignments with some variables integrated out.
* Online/stochastic variational methods: scale LDA to large corpora.

In practice, both variational and Gibbs methods can recover coherent topic structures when hyperparameters are reasonably chosen.

## Summary

* Hierarchical Bayesian models let us model uncertainty at multiple levels.
* In the rat example, hierarchical priors enable partial pooling across groups.
* LDA is a hierarchical model where latent topic variables explain observed text.
* The main computational challenge is posterior inference, typically handled by approximation.


## Reference

- D. Blei, A. Ng, and M. Jordan. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3, 993-1022.
- K. Nigam, A. McCallum, S. Thrun, and T. Mitchell. (2000). Text classification from labeled and unlabeled documents using EM. Machine Learning, 39(2/3), 103-134.
- A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari, and D. B. Rubin. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.
