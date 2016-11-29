---
layout: post
title:  "Optimization"
date:   2016-09-19 14:44:00
tag:
- Math
- Notes
blog: true
author: Jannis
description: Notes about optimization.
fontsize: 20pt
---

Life is NP hard. <br>


 {% include mathjax_support.html %}


## Rate of Convergence

We say <span> $\{x_{k}\}$ </span> converges to $x^{\ast}$ if and only if


 <div> $$\lim_{k \rightarrow \infty} {\parallel x_{k} - x^{\ast} \parallel}= 0$$ </div>

### Converge Sublinearly

<div>$$\lim_{k \rightarrow \infty} \frac{\parallel x_{k+1} - x^{\ast} \parallel}{\parallel x_{k} - x^{\ast} \parallel} = a$$ </div>

and <span>$a \in (0,1)$ then the $\{x_{k}\}$</span> is said to converge sublinearly.

#### Toy Example

<span>$x_{k} = 2^{-k}$</span>，<span>$x_{k} = 2^{-k}$</span> is monotonically decreasing and the minimum is <span>$x^{\ast} = 0$</span>，

hence we can derive

 <div>$$\lim_{k \rightarrow \infty} \frac{\parallel x_{k+1} - x^{\ast} \parallel}{\parallel x_{k} - x^{\ast} \parallel} = \lim_{k \rightarrow \infty} \frac{2^{-(k+1)}}{2^{-k}} = \frac{1}{2}$$ </div>

By the definition，<span>$x_{k} = 2^{-k}$</span> converges sublinearly.

### Converge Superlinearly

<div> $$\lim_{k \rightarrow \infty} \frac{\parallel x_{k+1} - x^{\ast} \parallel}{\parallel x_{k} - x^{\ast} \parallel} = 0$$ </div> then the <span>$\{x_{k}\}$</span> is said to converge superlinearly.

#### Toy Example

<span>$x_{k} = k^{-k}$</span> is monotonically decreasing, it is trivial if we convert it into <span>$x_{k} = k^{-k} = e^{ln(k^{-k})} = e^{-klnk}$</span>，the minimum is
<div>$$x^{\ast} = \lim_{k \rightarrow \infty} k^{k} =\lim_{k \rightarrow \infty} e^{ln(k^{-k})} =\lim_{k \rightarrow \infty} e^{-klnk} = 0$$</div>

Therefore

<div>$$\lim_{k \rightarrow \infty} \frac{\parallel x_{k+1} - x^{\ast} \parallel}{\parallel x_{k} - x^{\ast} \parallel} = \lim_{k \rightarrow \infty} \frac{(k+1)^{-(k+1)}}{k^{-k}} = \lim_{k \rightarrow \infty} \frac{(k+1)^{-k}}{(k)^{-k}(k+1)} $$</div>

<div>$$= \lim_{k \rightarrow \infty} \frac{1}{k+1}(\frac{k+1}{k})^{-k} = \lim_{k \rightarrow \infty} \frac{1}{k+1}(\frac{1}{k} + 1)^{-k} =  \lim_{k \rightarrow \infty} \frac{1}{(k+1)}\frac{1}{e} = 0$$</div>

By the definition，<span>$x_{k} = k^{-k}$</span> converges superlinearly.


### Converge with order q

<div> $$\lim_{k \rightarrow \infty} \frac{\parallel x_{k+1} - x^{\ast} \parallel}{\parallel x_{k} - x^{\ast} \parallel ^{q} = a$$ </div>

for any $a$ as a constant, then the <span>$\{x_{k}\}$</span> is said to converge with order q.

#### Toy Example

Given that <span>$x_{k} = a^{2^{k}}, 0 < a < 1$</span> is monotonically decreasing， its minimum is <span>$x^{\ast} = 0$</span>

Therefore,

<div>$$\lim_{k \rightarrow \infty} \frac{\parallel x_{k+1} - x^{\ast} \parallel}{\parallel x_{k} - x^{\ast} \parallel^{2}}  = \lim_{k \rightarrow \infty} \frac{a^{2^{k+1}}}{(a^{2^{k}})^{2}} = \lim_{k \rightarrow \infty} \frac{a^{2^{k+1}}}{a^{2^{k}\ast 2}} = \lim_{k \rightarrow \infty} \frac{a^{2^{k+1}}}{a^{2^{k+1}}} = 1$$</div>

By the definition，<span>$x_{k} = a^{2^{k}}, 0 < a < 1$</span> converges with order q.
