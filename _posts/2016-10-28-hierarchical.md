---
layout: post
title:  "Introduction to Hierarchical Model"
date:   2016-10-28 21:00:00
tag:
- Statistics
- Bayesian
projects: true
blog: false
author: Jannis
description: Hierarchical Model and Latent Dirichlet Allocation
fontsize: 23pt

---

{% include mathjax_support.html %}

<center> [Data Mining Center](http://www.rucdmc.net), Renmin University of China </center>

## Introduction to Bayesian Framework

### Priori, Likelihood and Posteriori


*  Bayes Formula $$p(\theta\mid y) = \frac{p(\theta)p(y\mid\theta)}{p(y)}$$
*  Prior $\theta$ $\sim$ Prior Distribution $p(\theta)$
Determined from past information or subjective assessment.
*  Observations $y\mid \theta$ $\sim$ Likelihood $p(y \mid \theta)$
Given the parameter $\theta$, the observed data $y$'s distribution.
*  Posterior $\theta \mid y$ $\sim$ Posterior Distribution $p(\theta \mid y)$
Updatad distribution of $\theta$ based on its prior and observed data.
*  $$p(\theta \mid y) = \frac{p(\theta)p(y\mid \theta)}{\int p(\theta)p(y\mid \theta) d\theta} \propto p(\theta)p(y\mid \theta)$$



## Types of Prior

What exactly is prior when we talk about it?

*  Past experience
*  Historical Research
*  Subjective Beliefs


We can define three types of priors according to the information they contain

*  Informative Priors
Prior distributions giving numerical information that is crucial to estimation of the model.
*  Non-informative Priors
Uniform or nearly so, and basically allow the information from the likelihood to be interpreted probabilistically.
*  Weakly Informative Priors
Not supplying any controversial information but are strong enough to pull the data away from inappropriate inferences that are consistent with the likelihood.


### Common Types of Priors
What kinds of prior do we usally use?

*  Experts' Prior
Prior distributions obtained via consulting experts.
*  Conjugate Priors
The prior distribution and the posterior distribution are from the same distribution family.
For example, if $\theta \sim $beta distribution then $\theta\mid y \sim beta$ distribution
*  Non-informative Priors
Uniform Prior or Jeffrey Prior





#### Conjugate Prior

*  The prior distribution and the posterior distribution are from the same distribution family.
*  Example : $$\theta \sim \text{Beta}(\alpha,\beta) \quad p(\theta) \propto \theta^{\alpha -1}(1 - \theta)^{\beta - 1} \quad \theta \in (0,1) $$
$$y\mid \theta \sim \text{Binomial}(n,\theta) \quad \quad p(y\mid \theta) \propto \theta^{y}(1 - \theta)^{n-y}$$
*  Hence we can derive the posterior $$p(\theta\mid y) \propto p(y\mid \theta)\cdot p(\theta) \propto \theta^{\alpha + y -1}(1 - \theta)^{\beta +n - y - 1}$$
Therefore, $$\theta \mid y \sim \text{Beta}(\alpha + y,\beta +n - y) \quad \theta \in (0,1) $$
*  $\theta$'s prior and posterior are both Beta distribution.




##### Why we use Conjugate Prior?

*  They simplify the computation!
We can easily derive the posterior distribution if we use conjugate prior.
*  Common Conjugate Families




#### Non-informative Prior

*  Uniform $$p(\theta) \propto 1$$
*  Example 1: $$p(\theta) = \frac{1}{2} \quad \quad \theta \in (0,2) $$
*  Example 2: $$p(\theta) \propto 1 \quad \quad \theta \in (-\infty,\infty) $$
Is that correct?
*  Prior is not a distribution! Its density cannot be integrated to 1.
We call this prior is  **improper**.
*  Improper prior can **sometimes** lead to proper posterior.
*  As long as it can lead to proper posterior, the prior can be useful.
*  Example 3: $$p(\theta) \propto 1 \quad \quad \theta \in (-\infty,\infty) $$
$$y\mid \theta \sim N(\theta, 1)$$
*  Hence we can derive the posterior $$p(\theta\mid y) \propto p(y\mid \theta)\cdot 1 = \frac{1}{\sqrt{2\pi}}e^{-\frac{(\theta - y)^2}{2}}$$
Therefore, $$\theta \mid y \sim N(y,1) \quad \theta \in (-\infty,\infty) $$
*  It's a proper posterior!




##### Jeffrey Prior

*  Do we have any other choice for non-informative prior?
*  Yes! That is Jeffrey Prior.
*  $$p(\theta) \propto [J(\theta)]^{\frac{1}{2}}$$ where $J(\theta)$ is the {\em Fisher Information} for $\theta$
$$J(\theta) = E((\frac{d logp(y\mid \theta)}{d\theta})^2\mid \theta)= - E(\frac{d^{2} logp(y\mid \theta)}{d\theta^{2}}\mid \theta)$$
*  Jeffrey's Invariance Principal:
No matter how I parametrize $\theta$, the prior density $p(\theta)$ is equivalent.
*   $$p(\theta) \propto [J(\theta)]^{\frac{1}{2}} \quad \text{Let} \quad \phi = h(\theta) \quad \text{One-to-One mapping}$$ We can prove that $$p(\phi) \propto [J(\phi)]^{\frac{1}{2}}$$





## Bayesian Hierarchical Model



### How to set the Hyperparameters?

figure missing

*  The table displays the values of $\frac{y_{i}}{n_{i}}$ : $i = 1,2,3,...,70$
 \centering{(number of rats with tumor) / (total number of rats)}
*  Tumor Incidence of rats in historical control groups and current group of rats, from Tarone (1982).


#### Model Initialization

*  Suppose $\theta$ is the probability that the rat had tumor.
*  Suppose $$y \mid \theta \sim \text{Binomial}(n,\theta)$$
*  $$\theta \sim \text{Beta}(\alpha, \beta)$$
*  Since Beta-Binomial is conjugate, so we can derive the posterior of $\theta$ easily
*  $$\theta\mid y \sim \text{Beta}(\alpha + y, \beta + n - y)$$


#### Toy Example
* How to set the $\alpha$ and $\beta$?
* We call the parameters in prior distribution *hyperparameter*.

#### How to set the priors?

##### Fixed Prior Distribution
Informative Prior
*  We knew that $\theta \sim$ Beta Distribution with known mean and variance.
*  $\theta$ vary due to differences in rats and experimental conditions.
*  Find the corresponding $\alpha$, $\beta$.
*  $\theta \sim$ Beta$(\alpha,\beta)$ as its prior distribution.

##### Approximate estimate using Historical Data

*  Use Historical Data's Mean and Variance to estimate $\alpha$ and $\beta$.
*  $$y_i \mid \theta \sim Binomial(n_i, \theta)$$
$$\theta \sim \text{Beta}(\hat{\alpha},\hat{\beta})$$
*  $\theta\mid y_1, y_2,\ldots,y_{71} \sim \text{Beta}(\hat{\alpha} + \sum_{i = 1}^{71} y_i ,\hat{\beta} + \sum_{i = 1}^{71}n_{i} -  \sum_{i = 1}^{71} y_i)$

*  Bayes Estimate $$\begin{aligned}&E\{\theta \mid y\}
&= \frac{\hat{\alpha} +  \sum_{i = 1}^{71} y_i}{\hat{\alpha} +\hat{\beta} + \sum_{i = 1}^{71}n_{i}}
&= \frac{\hat{\alpha}}{\hat{\alpha} +\hat{\beta} + \sum_{i = 1}^{71}n_{i}} +  \frac{ \sum_{i = 1}^{71} y_i}{\hat{\alpha} +\hat{\beta} + \sum_{i = 1}^{71}n_{i}}
&= (\frac{\hat{\alpha}}{\hat{\alpha} +\hat{\beta}})(\frac{\hat{\alpha} + \hat{\beta}}{\hat{\alpha} +\hat{\beta} + \sum_{i = 1}^{71}n_{i}}) + (\frac{\sum_{i = 1}^{71} y_i}{\sum_{i = 1}^{71}n_{i}})(\frac{\sum_{i = 1}^{71}n_{i}}{\hat{\alpha} +\hat{\beta} + \sum_{i = 1}^{71}n_{i}})
&=  (\frac{\sum_{i = 1}^{70} y_i}{\sum_{i = 1}^{70}n_{i}})(\frac{\hat{\alpha} + \hat{\beta}}{\hat{\alpha} +\hat{\beta} + \sum_{i = 1}^{71}n_{i}}) + (\frac{\sum_{i = 1}^{71} y_i}{\sum_{i = 1}^{71}n_{i}})(\frac{\sum_{i = 1}^{71}n_{i}}{\hat{\alpha} +\hat{\beta} + \sum_{i = 1}^{71}n_{i}})
\end{aligned}$$
*  Is that Correct?
*  NO!
*  Overestimate the precision of the posterior. (Data Used Twice)

##### Set the Hyperparameters without Data

Do we have to use data to set the hyperparameters?

*  In most cases in reality, we are not sure what how to set the priors scientifically.
*  However, the hyperparameters of the prior may not be that important.
*  If lacking information, use non-informative prior such as $Uniform(0,1) = Beta(1,1)$
*  In this case,
$$y_i \mid \theta_i \sim \text{Binomial}(n_i,\theta_i) $$
$$\theta_i \sim \text{Uniform}(0,1)$$ for $i = 1,2,...,70,71$

##### Can we regard hyperparameters in prior as random variables?

Set one more level of Hierarchical Model

Regard $\alpha$ \& $\beta$ as Random Variables

*  If we want to model the uncertainty of $\alpha$ and $\beta$,
*  We can assign a prior distributions for $\alpha$ and $\beta$ respectively.
*  Just add one more level of Hierarchical Model.
*  For example,
$$y_i \mid \theta_i \sim \text{Binomial}(n_i,\theta_i) $$
$$\theta_i \mid \alpha, \beta \sim \text{Beta}(\alpha,\beta)$$
$$\alpha \sim Gamma(1,2) \text{ , } \beta \sim Gamma(3,4)$$  for $i = 1,2,...,70,71$
*  The level of this model increased from 2 to 3.
*  This is Hierarchical Model.

## Latent Dirichlet Allocation

*  A classic example of Hierarchical Model
*  Analyze the model of Text Data

### Model Initialization

#### From Beta Distribution to Dirichlet

*  Beta-Binomial is a conjugate distribution.
*  $$f(x\mid \alpha, \beta) = \frac{1}{\text{Beta}(\alpha,\beta)}x^{\alpha - 1}(1-x)^{\beta - 1}$$ for $x \in (0,1)$
*  Dirichlet-Multinomial is a conjugate distribution
*  $$f(x_1, x_2, ... , x_n \mid \alpha_1, ...,\alpha_n) = \frac{\Gamma (\alpha_1 + ... + \alpha_n)}{\Gamma (\alpha_1)...\Gamma (\alpha_n)}x_{1}^{\alpha_1 - 1}x_{2}^{\alpha_2 - 1}...x_{n}^{\alpha_n - 1}$$
*  $x_1,x_2,...,x_{n-1} \in (0,1) , x_1+x_2+...+x_{n-1} < 1 , x_n = 1 - (x_1 + ... + x_{n-1})$
*  $$ \text{Dir}(\vec{p} \mid \vec{\alpha}) \times \text{MultiCount}(\vec{n}) =  \text{Dir}(\vec{p} \mid \vec{\alpha} + \vec{n})$$

#### Notation and Assumption

*  A *Vocabulary* indexed by $\{1,2,...,V\}$
*  A *word* is the basic unit of discrete data and is represented by a V-vector s.t. $$w^v = 1 \text{ and } w^u = 0 \text{ for } u \neq v$$
*  For example $$w_i = (0,0,1,0,0...,0)$$ If the ith *word*  matches the 3rd word in vocabulary
*  A *document* is a sequence of N words denoted by $**w** = (w_1,w_2,...,w_N)$
*  A *corpus* is a collection of M documents denoted by $**D** = \{ {** w_1**},{**w_2**},...,{**w_M**} \}$
*  There are k topics in total.
*  **Bag-of-words** Assumption (Exchangeable)

### Where is the "Latent" in LDA?
figure missing

*  $$w \mid \beta, z \sim \text{Multinomial} $$
$$z \mid \theta \sim \text{Multinomial}(\theta)$$
$$\theta \sim \text{Dirichlet}(\alpha)$$
*  So $\alpha$ and $\beta$ are the Hyperparameters in this model. \#(k + kV)

*  $$\beta = \{\beta_{ij}\}_{k \times V}$$
*  where $${\beta_{ij}} = p(w^{j} = 1 \mid z^i = 1)$$


### Posteriorl Inference

#### Intractable Posterior

*  We want to find the posterior distribution of $\theta$ and $z$
$$p(\theta,{\bf z} \mid {\bf w}, \alpha, \beta) = \frac{p(\theta,{\bf z},{\bf w}\mid \alpha \beta)}{p({\bf w} \mid \alpha, \beta)}$$
*  However, the posterior distribution is intractable. (Denominator Part)
*  How to get the posterior?


## Reference

- D. Blei, A. Ng, and M. Jordan. (2003) Latent Dirichlet Allocation, Journal of Machine Learning Research 3:993-1022.
- K. Nigam, A.McCallum, S. Thrun, and T. Mitchell (2000) Text classification from labeled and unlabeled documents using EM.,Machine Learning 39(2/3):103-134
- A. Gelman, J.B. Carlin, H.S. Stern, D.B. Dunson, A. Vehtari, and D.B. Rubin (2013),Bayesian Data Analysis,CRC Press 39(2/3):101-103
