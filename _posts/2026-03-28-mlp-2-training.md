---
layout: post
title: "LLM Basics Series 2: MLP Training Essentials"
date: 2026-01-31 10:00:00
tag:
- Machine Learning
- LLM
projects: false
blog: true
author: YingZhang
coauthor_name: WenboGuo
coauthor_url: "https://henrygwb.github.io"
description: Beginner guide to loss, backpropagation, and regularization for MLPs.
fontsize: 23pt
---

{% include mathjax_support.html %}

This post continues from MLP core ideas and focuses on training: loss, backpropagation, and regularization.

## Step 1: Choose a Loss

For classification, we usually use cross-entropy. For regression, MSE is common.

Classification objective:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{ic}\log p_{ic}
$$

where $y_{ic}$ is the target indicator and $p_{ic}$ is predicted probability.

## Step 2: Backpropagation

Training alternates between:

1. Forward pass: compute predictions and loss.
2. Backward pass: compute gradients via chain rule.
3. Parameter update: optimizer step.

Generic update:

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
$$

where $\eta$ is learning rate.

Frameworks like PyTorch handle automatic differentiation, but understanding the gradient flow explains why exploding and vanishing gradients happen and why the training tricks in later sections help.

### Two-Layer MLP Walkthrough

Consider a two-layer MLP with MSE loss on a single sample:

$$
z_1 = xW_1 + b_1, \quad h = \phi(z_1), \quad \hat{y} = hW_2 + b_2, \quad \mathcal{L} = \tfrac{1}{2}\|\hat{y} - y\|^2.
$$

The forward pass stores intermediate values $z_1$, $h$, and $\hat{y}$. The backward pass then uses the chain rule in reverse order to compute gradients for each parameter.

#### Forward Pass

Starting from input $x \in \mathbb{R}^{1 \times d_{in}}$:

1. Pre-activation: $z_1 = xW_1 + b_1 \in \mathbb{R}^{1 \times d_h}$.
2. Activation: $h = \phi(z_1) \in \mathbb{R}^{1 \times d_h}$.
3. Output: $\hat{y} = hW_2 + b_2 \in \mathbb{R}^{1 \times d_{out}}$.
4. Loss: $\mathcal{L} = \tfrac{1}{2}\|\hat{y} - y\|^2$.

#### Backward Pass

Starting from the loss, we compute gradients layer by layer in reverse.

**Step 1.** Gradient of the loss with respect to the output:

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y \in \mathbb{R}^{1 \times d_{out}}.
$$

**Step 2.** Gradients for the second-layer parameters:

$$
\frac{\partial \mathcal{L}}{\partial W_2} = h^\top \frac{\partial \mathcal{L}}{\partial \hat{y}} \in \mathbb{R}^{d_h \times d_{out}}, \quad \frac{\partial \mathcal{L}}{\partial b_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}}.
$$

**Step 3.** Gradient passed back to the hidden layer:

$$
\frac{\partial \mathcal{L}}{\partial h} = \frac{\partial \mathcal{L}}{\partial \hat{y}} W_2^\top \in \mathbb{R}^{1 \times d_h}.
$$

**Step 4.** Gradient through the activation function:

$$
\frac{\partial \mathcal{L}}{\partial z_1} = \frac{\partial \mathcal{L}}{\partial h} \odot \phi'(z_1),
$$

where $\odot$ denotes element-wise multiplication and $\phi'(z_1)$ is the derivative of the activation evaluated at the pre-activation values. For ReLU, $\phi'(z) = \mathbf{1}[z > 0]$, so each element is either 0 or 1.

**Step 5.** Gradients for the first-layer parameters:

$$
\frac{\partial \mathcal{L}}{\partial W_1} = x^\top \frac{\partial \mathcal{L}}{\partial z_1} \in \mathbb{R}^{d_{in} \times d_h}, \quad \frac{\partial \mathcal{L}}{\partial b_1} = \frac{\partial \mathcal{L}}{\partial z_1}.
$$

Each step depends on the gradient from the step before it, which is why computation must proceed backward from the loss. Series 1 covers the underlying vector-Jacobian product (VJP) derivation in detail.

### Activation Derivatives

The activation derivative $\phi'(z)$ controls how gradient flows through each neuron.

| Activation | $\phi(z)$ | $\phi'(z)$ | Gradient behavior |
|---|---|---|---|
| ReLU | $\max(0, z)$ | $\mathbf{1}[z > 0]$ | Passes gradient unchanged or blocks it entirely |
| Sigmoid | $\sigma(z)$ | $\sigma(z)(1-\sigma(z))$ | Maximum 0.25 at $z=0$; shrinks gradient |
| Tanh | $\tanh(z)$ | $1 - \tanh^2(z)$ | Maximum 1 at $z=0$; shrinks gradient for large $\lvert z \rvert$ |

Sigmoid and Tanh derivatives are always less than 1. In a deep network, multiplying many such factors together makes gradients shrink exponentially toward zero. This is the vanishing gradient problem.

ReLU avoids this because its derivative is exactly 1 for positive inputs. However, if a neuron's pre-activation is always negative, the derivative is always 0 and the neuron stops learning entirely (the dying ReLU problem).

### Vanishing And Exploding Gradients

For a network with $L$ layers, the gradient at layer $l$ involves a product of $L - l$ terms:

$$
\frac{\partial \mathcal{L}}{\partial W_l} \propto \prod_{k=l}^{L-1} W_k^\top \cdot \operatorname{diag}(\phi'(z_k)).
$$

If the spectral norm of each factor is consistently less than 1, the product shrinks exponentially with depth. If consistently greater than 1, it grows exponentially.

This connects directly to the training tricks covered later in this post:

- Proper initialization (Xavier, He) keeps each factor close to norm 1 at the start of training.
- Normalization layers (BatchNorm, LayerNorm) prevent activations from drifting to extreme values where activation derivatives saturate.
- Gradient clipping caps the gradient norm to prevent occasional large values from destabilizing the optimizer step.
- Residual connections provide a direct additive path for gradients, bypassing the multiplicative chain.

## Step 3: Regularize for Better Generalization

Useful options:

- Weight decay (L2 penalty).
- Dropout.
- Early stopping.
- Data augmentation (if applicable).

Weight decay adds a penalty:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda\|W\|_2^2
$$

This discourages overly large weights.

## Step 4: Handle Categorical Inputs With Embeddings

Not every input feature should be treated as a continuous number. Many important features are categorical:

- token IDs in language models,
- item IDs in recommendation systems,
- user IDs,
- product categories,
- zip codes or other discrete codes.

If you feed these IDs directly as raw integers, the model may wrongly assume an ordering. For example, item ID 100 is not "twice" item ID 50 in any meaningful way. One-hot encoding avoids this fake ordering, but it becomes huge and sparse when the number of categories is large.

An embedding layer solves this problem by learning a dense vector for each category.

If a categorical feature has $K$ possible values, an embedding table is

$$
E \in \mathbb{R}^{K \times d_e},
$$

where $d_e$ is the embedding dimension. Looking up category $i$ returns the $i$-th row $E_i \in \mathbb{R}^{d_e}$.

So instead of feeding a huge one-hot vector, we feed a learned dense representation. Categories that play similar roles in the task can end up with similar embeddings.

### One-Hot Versus Embedding Dimension

With one-hot encoding, the input dimension for a feature with $K$ categories is $K$ (or $K-1$ if one column is dropped to avoid multicollinearity in linear models; in neural networks all $K$ columns are typically kept). With an embedding layer, the input dimension reduces to $d_e$.

In practice $d_e \ll K$. The purpose of the embedding is to compress a high-dimensional sparse representation into a low-dimensional dense one. Common heuristics for choosing $d_e$:

- $d_e = \min(50,\; \lceil (K+1)/2 \rceil)$ (fast.ai default for tabular models),
- $d_e = \lceil K^{1/4} \rceil$ (Google recommendation),
- start with a modest value like 8, 16, or 32 and tune.

For example, a feature with $K = 10{,}000$ categories would need a 10,000-dimensional one-hot vector, but an embedding dimension of $d_e = 10$ to $50$ is usually sufficient.

When $K$ is small (say 3 or 4), one-hot encoding already works well and an embedding layer adds unnecessary complexity. Setting $d_e > K$ is rare and seldom helpful.

### Embedding As A Linear Projection Without Bias

An embedding lookup is mathematically identical to multiplying a one-hot vector by a weight matrix with no bias term.

Write the one-hot vector for category $i$ as $\mathbf{x}_i \in \mathbb{R}^K$, which has a 1 at position $i$ and 0 elsewhere. A linear layer without bias computes

$$
\mathbf{h} = E^\top \mathbf{x}_i,
$$

where $E \in \mathbb{R}^{K \times d_e}$. Because $\mathbf{x}_i$ is one-hot, the matrix-vector product selects exactly the $i$-th row of $E$:

$$
E^\top \mathbf{x}_i = E_i \in \mathbb{R}^{d_e}.
$$

This is the same result as looking up row $i$ in the embedding table. The two operations produce identical outputs and learn identical parameters. The only difference is computational: an index lookup is $O(d_e)$, while a full matrix multiply is $O(K \times d_e)$. When $K$ is large, the lookup avoids the cost of multiplying by all the zeros in the one-hot vector.

No bias is needed because the one-hot vector already selects a unique row. A bias term would add the same vector to every category and could be absorbed into the embedding rows themselves, so it provides no additional expressive power.

Examples:

- In LLMs, each token ID is mapped to a token embedding before entering the transformer.
- In recommendation systems, user IDs and item IDs are often embedded and then combined with other dense features.
- In tabular models, high-cardinality categorical columns are often embedded instead of one-hot encoded.

For LLMs, the embedding dimension equals the model's hidden size $d_{model}$, which is determined by the overall parameter budget rather than the tabular heuristics above. The following table shows how vocab size, embedding dimension, and depth scale across real models:

| Model | Params | Vocab ($K$) | Embed dim ($d_e$) | Layers | $K / d_e$ |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | 151,936 | 1,024 | 28 | ~148x |
| Qwen3-1.7B | 1.7B | 151,936 | 2,048 | 28 | ~74x |
| Qwen3-4B | 4B | 151,936 | 2,560 | 36 | ~59x |
| Qwen3-8B | 8B | 151,936 | 4,096 | 36 | ~37x |
| Qwen3-32B | 32B | 151,936 | 5,120 | 64 | ~30x |
| GPT-3 | 175B | 50,257 | 12,288 | 96 | ~4x |
| Mistral 7B | 7B | 32,000 | 4,096 | 32 | ~8x |

All entries satisfy $d_e \ll K$, so embeddings are always a dimension reduction. Embedding dimension scales with model size, not vocab size: Qwen3-0.6B and Qwen3-32B share the same 152K vocab but use $d_e$ of 1,024 and 5,120 respectively.

Practical guidance:

- small cardinality categories can still use one-hot encoding,
- large cardinality categories usually benefit from embeddings,
- embedding dimension is a tunable hyperparameter (see heuristics above),
- rare categories may need an "unknown" bucket or frequency thresholding.

Embeddings turn discrete labels into trainable continuous features that an MLP can use effectively.

## Step 5: Training Tricks That Matter In Practice

After loss, backpropagation, and regularization are in place, a few engineering tricks often make the difference between unstable training and a model that learns reliably.

### Normalize Inputs And Hidden States

If one feature has scale 0.01 and another has scale 10,000, optimization becomes unnecessarily hard. Standardizing continuous inputs to mean 0 and variance 1 is a strong default.

Inside the network, normalization layers can also help:

- BatchNorm is common for standard MLPs with decent batch size,
- LayerNorm is often preferred for sequence models and transformers.

These methods stabilize activations, improve gradient flow, and often let you use larger learning rates safely.

### Use Good Initialization

Bad initialization can make gradients vanish or explode before training even starts. Initialization sets the starting values of weight matrices $W$ in each layer. Biases are typically initialized to zero.

The goal is to keep each layer's output variance roughly equal to its input variance, so that signals neither explode nor vanish as they propagate through many layers.

**Forward pass constraint.** For a single layer $y = Wx$ (ignoring bias and activation), where $W \in \mathbb{R}^{d_{out} \times d_{in}}$, a single output element is

$$
y_j = \sum_{i=1}^{d_{in}} W_{ji}\, x_i.
$$

If the $W_{ji}$ and $x_i$ are independent and zero-mean, then

$$
\mathrm{Var}(y_j) = d_{in} \cdot \mathrm{Var}(W) \cdot \mathrm{Var}(x).
$$

To keep $\mathrm{Var}(y) = \mathrm{Var}(x)$, we need $\mathrm{Var}(W) = 1/d_{in}$.

**Backward pass constraint.** During backpropagation, the gradient flowing back to each input element is

$$
\frac{\partial \mathcal{L}}{\partial x_i} = \sum_{j=1}^{d_{out}} W_{ji}\, \frac{\partial \mathcal{L}}{\partial y_j}.
$$

This is a sum of $d_{out}$ terms (gradients flow through $W^\top$). By the same variance argument:

$$
\mathrm{Var}\!\left(\frac{\partial \mathcal{L}}{\partial x_i}\right) = d_{out} \cdot \mathrm{Var}(W) \cdot \mathrm{Var}\!\left(\frac{\partial \mathcal{L}}{\partial y}\right).
$$

To keep gradient variance stable across layers, we need $\mathrm{Var}(W) = 1/d_{out}$.

The forward pass wants $1/d_{in}$ and the backward pass wants $1/d_{out}$. In general $d_{in} \neq d_{out}$, so both constraints cannot be satisfied simultaneously.

**Xavier/Glorot initialization** compromises between forward and backward by averaging the two constraints:

$$
W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{d_{in} + d_{out}}\right).
$$

This assumes a near-linear activation around zero, which Sigmoid and Tanh approximate for small inputs. It is the standard choice for Sigmoid and Tanh networks.

**He/Kaiming initialization** accounts for ReLU zeroing out roughly half the activations, so only about $d_{in}/2$ terms contribute non-zero values. To compensate, the variance is doubled:

$$
W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{d_{in}}\right).
$$

This is the standard choice for ReLU and its variants (LeakyReLU, ELU, etc.). In PyTorch this is `torch.nn.init.kaiming_normal_`.

Why not initialize to all zeros: every neuron computes the same output, so every gradient is identical and the network can never break symmetry. Why not large random values: for Sigmoid/Tanh, large pre-activations saturate the output where gradients approach zero, stalling learning. For ReLU, large weights cause activations to grow unchecked through layers, quickly reaching NaN.

### Residual Connections

Even with good initialization, very deep networks can still suffer from vanishing gradients because the signal must pass through many multiplicative factors. Residual connections address this by adding a skip path around each block.

A standard layer computes

$$
y = f(x),
$$

where $f$ is some transformation (linear layer + activation, or an entire transformer sub-block). A residual layer instead computes

$$
y = f(x) + x.
$$

The output is the transformation plus the original input. The block $f(x)$ only needs to learn the *residual*, the difference between the desired output and the input, rather than the full mapping.

**Why this helps gradient flow.** Consider a network with $L$ residual blocks:

$$
x_1 = f_1(x_0) + x_0, \quad x_2 = f_2(x_1) + x_1, \quad \dots, \quad x_L = f_L(x_{L-1}) + x_{L-1}.
$$

The gradient of the loss with respect to an early layer $x_l$ is

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \prod_{k=l+1}^{L} \left(I + \frac{\partial f_k}{\partial x_{k-1}}\right).
$$

Each factor in this product is $I + \frac{\partial f_k}{\partial x_{k-1}}$ rather than just $\frac{\partial f_k}{\partial x_{k-1}}$. The identity matrix $I$ provides a direct path for the gradient to flow unchanged. Even if $\frac{\partial f_k}{\partial x_{k-1}}$ is small, the gradient does not vanish because the $I$ term always passes it through. Without the skip connection, the product would be $\prod_k \frac{\partial f_k}{\partial x_{k-1}}$, which shrinks exponentially when each factor has norm less than 1.

**Why this helps optimization.** At initialization, if the weights are small, $f(x) \approx 0$ and each residual block approximates the identity $y \approx x$. The network starts close to a simple pass-through, and training gradually adds complexity by learning non-zero residuals. This gives the optimizer a much easier starting point compared to a plain deep network where every layer must simultaneously learn a useful transformation.

Residual connections were introduced in ResNet for image classification and are now used in virtually all deep architectures. Every transformer block uses them: the attention sub-layer and the FFN sub-layer each have a skip connection around them.

### Tune Learning Rate, Batch Size, And Schedule Together

Learning rate is usually the most sensitive hyperparameter.

- If loss diverges or becomes NaN, reduce it.
- If training barely moves, increase it moderately.
- A scheduler is almost always better than a fixed learning rate.

#### Learning Rate Schedules

A learning rate schedule changes $\eta$ over the course of training. The basic intuition: early in training, parameters are far from a good solution, so larger steps are helpful. Later, the optimizer is near a minimum and large steps overshoot, so a smaller rate gives finer convergence.

**Warmup.** Training starts with $\eta = 0$ (or very small) and linearly increases to the target learning rate $\eta_{\max}$ over a warmup period of $T_w$ steps:

$$
\eta_t = \eta_{\max} \cdot \frac{t}{T_w}, \quad t \leq T_w.
$$

Warmup helps because at initialization the model's activations, gradients, and optimizer statistics (e.g. Adam's running mean and variance estimates) are all unreliable. A large learning rate applied to noisy early gradients can push parameters into a bad region that is hard to recover from. Warmup gives the optimizer time to calibrate before taking full-sized steps. It is especially important for transformers and large models. A typical warmup length is 1--5% of total training steps.

**Constant schedule.** After warmup (or from the start), $\eta$ stays fixed. Simple, but usually suboptimal because the same step size that works early is too large late in training.

**Step decay.** $\eta$ is multiplied by a factor (e.g. 0.1) at predefined milestones:

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / S \rfloor},
$$

where $S$ is the step interval and $\gamma$ is the decay factor (commonly 0.1). This was the standard schedule in early deep learning (e.g. ResNet training drops the learning rate by 10x at epoch 30 and 60). It is simple but requires choosing the milestone epochs manually.

**Cosine decay.** $\eta$ follows a half-cosine curve from $\eta_{\max}$ down to $\eta_{\min}$ (often 0 or $0.1 \cdot \eta_{\max}$) over $T$ total steps:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi \, t}{T}\right)\right).
$$

Cosine decay is smooth, has no milestones to tune, and decays slowly at first then faster toward the end. It is the most common schedule for transformer and LLM training. GPT-3, LLaMA, and most modern LLM recipes use warmup followed by cosine decay.

**Linear decay.** $\eta$ decreases linearly from $\eta_{\max}$ to $\eta_{\min}$:

$$
\eta_t = \eta_{\max} - (\eta_{\max} - \eta_{\min}) \cdot \frac{t}{T}.
$$

Simpler than cosine and sometimes used for fine-tuning (e.g. BERT fine-tuning often uses linear decay with warmup).

**Warmup + cosine decay (the standard LLM recipe).** Combine warmup for the first $T_w$ steps with cosine decay for the remaining $T - T_w$ steps. In practice this looks like:

1. Steps $0$ to $T_w$: linear increase from 0 to $\eta_{\max}$.
2. Steps $T_w$ to $T$: cosine decay from $\eta_{\max}$ to $\eta_{\min}$.

Most modern training frameworks (PyTorch, Hugging Face Transformers) have built-in schedulers for all of these. In PyTorch: `torch.optim.lr_scheduler.CosineAnnealingLR` for cosine decay, `torch.optim.lr_scheduler.StepLR` for step decay, and `get_cosine_schedule_with_warmup` in Hugging Face for warmup + cosine.

Batch size also matters:

- small batches add more gradient noise,
- large batches are more hardware-efficient but can change optimization behavior.

In practice, learning rate, batch size, and optimizer should be tuned together rather than in isolation.

### Clip Gradients When Training Is Unstable

Gradient clipping limits the magnitude of the gradient vector before the optimizer step. The standard method is norm-based clipping:

$$
g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|}\right),
$$

where $g$ is the full gradient vector (concatenation of all parameter gradients) and $c$ is the clipping threshold. When $\|g\| \leq c$, the factor equals 1 and the gradient is unchanged. When $\|g\| > c$, the entire vector is scaled down so that its norm becomes exactly $c$. Because the operation is a scalar multiplication, the gradient direction is preserved. Only the magnitude is capped.

This is different from element-wise clipping, which clamps each component independently to $[-c, c]$ and can change the gradient direction.

In PyTorch, norm-based clipping is `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=c)`, called after `loss.backward()` and before `optimizer.step()`.

Choosing $c$:

- $c = 1.0$ is the most common default. GPT-2, GPT-3, and many transformer training recipes use this value.
- Start with $c = 1.0$. If training is still unstable, try lowering to 0.5 or 0.25. If clipping activates on nearly every step, $c$ may be too small and the effective learning rate is reduced.
- A useful diagnostic is to log the unclipped gradient norm during training. If the norm is usually around 0.5 but occasionally spikes to 50, then $c = 1.0$ will catch the spikes without affecting normal steps. Also log the fraction of steps where clipping activates. If clipping triggers on less than 1--5% of steps, $c$ is doing its job as a safety net. If it triggers on most steps, the threshold is too aggressive and effectively shrinking the learning rate every update.
- Gradient clipping interacts with learning rate: clipping with a large $c$ and a small learning rate can have the same effect as a smaller $c$ with a larger learning rate. Tune them together.

### Watch Out For Data Problems

Some failures are caused more by data than by model architecture.

- Class imbalance can make the model ignore rare classes.
- Noisy labels can cap performance even when training loss looks low.
- Poor train/validation splits can make evaluation misleading.

Common fixes include reweighting the loss, oversampling minority classes, stronger data cleaning, or better splitting strategy.

### Debug With Curves, Gradients, And Outputs

When training goes wrong, do not change hyperparameters blindly. Diagnose first by inspecting the following signals.

#### Loss Curves

Plot training loss and validation loss over epochs. The shape of these two curves is the single most informative diagnostic.

- **Training loss decreasing, validation loss flat or increasing.** This is overfitting. The model is memorizing training data instead of learning general patterns. Try stronger regularization (increase dropout, add weight decay, reduce model size), more training data, or early stopping at the epoch where validation loss was lowest.
- **Both losses high and barely decreasing.** This is underfitting. The model does not have enough capacity or the learning rate is too low. Try increasing model size (more layers or larger hidden dimension), increasing learning rate, training for more epochs, or checking that the input features actually contain signal for the target.
- **Both losses decreasing but a large gap between them.** Mild overfitting. The model is learning but generalizing poorly. Regularization or more data usually helps.
- **Training loss oscillates wildly.** Learning rate is likely too large. Reduce it by 2--10x. If using SGD, switching to Adam can also stabilize updates because Adam normalizes by the running variance of each gradient coordinate.
- **Training loss plateaus early at a high value.** Check that the loss function matches the task (cross-entropy for classification, MSE for regression). 

#### Gradient Norms

Log the global gradient norm (the norm of all parameter gradients concatenated) at each step.

- **Gradient norm is very large or spiking.** Risk of instability. This is where gradient clipping helps. If spikes are frequent, the learning rate may also be too large.
- **Gradient norm is near zero.** Vanishing gradients. Check activation functions (Sigmoid/Tanh can saturate), initialization (too large or too small), and whether the network is too deep without residual connections.
- **Gradient norm is stable but loss is not decreasing.** The optimizer may be stuck in a flat region or saddle point. Try increasing learning rate, using a learning rate warmup schedule, or switching optimizers.

#### Activations And Parameters

- **Activations collapsing to the same value across neurons.** This means the layer has lost diversity, often caused by poor initialization (all zeros) or exploding/vanishing signals. Check initialization and add normalization layers.
- **Activations or parameters becoming NaN or Inf.** Numerical instability, usually from a learning rate that is too large, missing gradient clipping, or a division by a near-zero value. Reduce learning rate and add gradient clipping as a first step.
- **Many ReLU neurons outputting exactly zero on all inputs.** Dying ReLU. Those neurons have permanently negative pre-activations and will never recover. Try LeakyReLU, reduce learning rate, or check initialization.

#### Model Outputs On Fixed Examples

Pick a small set of examples (5--10) and log the model's predictions on them periodically during training. This is more interpretable than aggregate loss because you can see whether predictions are moving in the right direction, stuck at a constant, or oscillating.

For classification, check whether predicted probabilities are always near uniform (model has not learned anything) or always near 0/1 from the start (likely a data leak or label encoding bug).

## Practical Hyperparameters to Start

For small tabular/text-feature tasks:

1. Hidden size: 128 or 256.
2. Learning rate: $10^{-3}$ with Adam/AdamW.
3. Batch size: 32 or 64.
4. Epochs: 20 to 100 with early stopping.

Then tune one factor at a time.

## Putting It All Together: Two-Layer MLP Example

The following code combines the ideas from this post into a working example: synthetic data, a two-layer MLP with dropout and He initialization, AdamW with weight decay, linear warmup + cosine decay schedule, gradient clipping, and train/validation loss logging.

```python
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Synthetic data ──────────────────────────────────────────────
# 2,000 samples, 4-class classification. Inputs are a mix of:
#   - d_cont continuous features (e.g. age, price),
#   - 3 categorical features with different cardinalities
#     (e.g. user_region, product_category, device_type).
# Labels come from a known linear combination of continuous features
# and category-specific effects, plus noise.
n_samples = 2000
d_cont = 15                              # number of continuous features
cat_cardinalities = [50, 100, 10]        # K for each categorical feature
cat_embed_dims = [8, 16, 4]              # d_e for each (chosen with K^(1/4) as a guide)
n_classes = 4

# Continuous features
X_cont = torch.randn(n_samples, d_cont)

# Categorical features: integer IDs in [0, K).
X_cat = torch.stack([
    torch.randint(0, K, (n_samples,)) for K in cat_cardinalities
], dim=1)  # shape: (n_samples, 3)

# Generate labels: continuous part + learned per-category effects + noise.
W_cont_true = torch.randn(d_cont, n_classes)
cat_effects_true = [torch.randn(K, n_classes) for K in cat_cardinalities]
logits_true = X_cont @ W_cont_true
for i, eff in enumerate(cat_effects_true):
    logits_true = logits_true + eff[X_cat[:, i]]
logits_true = logits_true + 0.5 * torch.randn(n_samples, n_classes)
y = logits_true.argmax(dim=1)

# 80/20 train-validation split
n_train = int(0.8 * n_samples)
train_ds = TensorDataset(X_cont[:n_train], X_cat[:n_train], y[:n_train])
val_ds = TensorDataset(X_cont[n_train:], X_cat[n_train:], y[n_train:])

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# ── Model ───────────────────────────────────────────────────────
class TwoLayerMLPWithEmbeddings(nn.Module):
    def __init__(self, d_cont, cat_cardinalities, cat_embed_dims,
                 d_hidden, n_classes, dropout=0.2):
        super().__init__()

        # One nn.Embedding per categorical feature. Each lookup returns
        # a d_e-dim dense vector for the given integer ID. This is
        # equivalent to multiplying a one-hot vector by a K x d_e matrix,
        # but avoids building the one-hot explicitly.
        self.embeddings = nn.ModuleList([
            nn.Embedding(K, d_e)
            for K, d_e in zip(cat_cardinalities, cat_embed_dims)
        ])

        # Total input dim after concatenating continuous features and
        # all embedding outputs.
        d_in = d_cont + sum(cat_embed_dims)

        self.layer1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        # Dropout: randomly zeros activations during training to
        # reduce overfitting. model.eval() disables it automatically.
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_hidden, n_classes)

        # He/Kaiming initialization: Var(W) = 2/d_in, which accounts
        # for ReLU zeroing out ~half the activations. Biases start at
        # zero. This keeps activation variance stable across layers.
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
        nn.init.zeros_(self.layer1.bias)
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity="relu")
        nn.init.zeros_(self.layer2.bias)
        # Embedding tables: small random init so initial category
        # representations are distinct but not too large.
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

    def forward(self, x_cont, x_cat):
        # Look up an embedding for each categorical feature.
        # x_cat[:, i] has shape (batch,); emb(...) returns (batch, d_e_i).
        embed_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        # Concatenate continuous features with all embeddings along dim=1.
        x = torch.cat([x_cont] + embed_outs, dim=1)

        x = self.layer1(x)      # linear: d_in -> d_hidden
        x = self.relu(x)        # activation
        x = self.dropout(x)     # dropout (only active during training)
        x = self.layer2(x)      # linear: d_hidden -> n_classes
        return x                # raw logits, no softmax

d_hidden = 128
model = TwoLayerMLPWithEmbeddings(
    d_cont, cat_cardinalities, cat_embed_dims,
    d_hidden, n_classes, dropout=0.2,
)

# ── Optimizer and schedule ──────────────────────────────────────
lr_max = 1e-3
weight_decay = 1e-4
n_epochs = 50
grad_clip = 1.0  # norm-based clipping threshold

# AdamW: adaptive per-parameter learning rates with decoupled weight
# decay. Weight decay (lambda = 1e-4) penalizes large weights without
# distorting Adam's adaptive step sizes.
optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr_max, weight_decay=weight_decay
)

# Linear warmup + cosine decay schedule.
# First 5% of steps: ramp lr from 0 to lr_max (gives optimizer time
# to calibrate running statistics before taking full-sized steps).
# Remaining steps: cosine decay from lr_max to 0.
# lr_lambda returns a multiplier applied to lr_max at each step.
steps_per_epoch = math.ceil(n_train / batch_size)
total_steps = n_epochs * steps_per_epoch
warmup_steps = int(0.05 * total_steps)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps           # linear warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── Training loop ───────────────────────────────────────────────
# CrossEntropyLoss expects raw logits (no softmax) and integer class labels.
# It internally applies log-softmax then negative log-likelihood.
loss_fn = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    # ---- train ----
    # Set model to training mode: enables dropout and any other
    # train-only behavior (e.g. BatchNorm running stats update).
    model.train()
    train_loss = 0.0
    n_steps = 0
    n_clipped = 0
    grad_norm_sum = 0.0

    for xb_cont, xb_cat, yb in train_loader:
        # 1. Forward pass: compute predicted logits for this batch.
        #    The model takes continuous and categorical inputs separately.
        logits = model(xb_cont, xb_cat)

        # 2. Compute loss: cross-entropy between predictions and targets.
        loss = loss_fn(logits, yb)

        # 3. Zero out gradients from the previous step.
        #    Without this, gradients would accumulate across batches.
        optimizer.zero_grad()

        # 4. Backward pass: compute gradient of loss w.r.t. every parameter.
        #    This is where the chain rule (backpropagation) runs.
        loss.backward()

        # 5. Gradient clipping: scale the entire gradient vector so its
        #    norm does not exceed grad_clip. Direction is preserved,
        #    only magnitude is capped. clip_grad_norm_ returns the
        #    original (unclipped) norm, so we capture it for diagnostics
        #    without an extra computation pass.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=grad_clip
        )
        grad_norm_sum += grad_norm.item()
        if grad_norm.item() > grad_clip:
            n_clipped += 1
        n_steps += 1

        # 6. Optimizer step: update parameters using the (clipped) gradients.
        #    AdamW applies adaptive learning rates per parameter and
        #    decoupled weight decay.
        optimizer.step()

        # 7. Scheduler step: update learning rate for the next step.
        #    With warmup + cosine decay, this is called per step (not per epoch).
        scheduler.step()

        # Accumulate loss weighted by batch size for epoch-level average.
        train_loss += loss.item() * xb_cont.size(0)

    train_loss /= n_train
    # Gradient diagnostics: average norm and fraction of steps clipped.
    # If clipping triggers on <1-5% of steps, c is a safety net.
    # If it triggers on most steps, c may be too small or lr too large.
    avg_grad_norm = grad_norm_sum / n_steps
    clip_fraction = n_clipped / n_steps

    # ---- validate ----
    # Set model to evaluation mode: disables dropout and freezes
    # BatchNorm running stats.
    model.eval()
    val_loss = 0.0
    correct = 0

    # torch.no_grad() disables gradient computation, saving memory
    # and time since we do not need gradients for validation.
    with torch.no_grad():
        for xb_cont, xb_cat, yb in val_loader:
            logits = model(xb_cont, xb_cat)
            val_loss += loss_fn(logits, yb).item() * xb_cont.size(0)
            # predicted class = index of the largest logit
            correct += (logits.argmax(dim=1) == yb).sum().item()

    val_loss /= (n_samples - n_train)
    val_acc = correct / (n_samples - n_train)

    # Log train/val loss, accuracy, learning rate, gradient norm, and
    # clip fraction every 10 epochs. In practice, log every epoch or
    # every N steps to a tool like TensorBoard or Weights & Biases.
    if (epoch + 1) % 10 == 0:
        print(
            f"epoch {epoch+1:3d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.3f} | "
            f"lr {scheduler.get_last_lr()[0]:.6f} | "
            f"grad norm {avg_grad_norm:.4f} | "
            f"clipped {clip_fraction:.1%}"
        )
```

## Tensor Shapes Reference

Tracking what each dimension represents is the most reliable way to read and debug tensor code. The rule: **label every axis by name, not just by size.**

### 2D Tensors (Tabular Data, MLP Layers)

Most MLP operations use 2D tensors where axis 0 is the batch and axis 1 is the feature dimension.

| Tensor | Shape | Meaning |
|---|---|---|
| Input features | $(N, d_{in})$ | $N$ samples, each with $d_{in}$ features |
| Weight matrix | $(d_{in}, d_h)$ | maps $d_{in}$ inputs to $d_h$ hidden units |
| Hidden activations | $(N, d_h)$ | one $d_h$-dim vector per sample |
| Output logits | $(N, C)$ | one score per class, per sample |

Matrix multiply contracts the shared axis: $(N, d_{in}) \times (d_{in}, d_h) \to (N, d_h)$.

### 3D Tensors (Sequences, Text)

Sequence models add a length axis. A batch of sentences becomes $(N, T, d)$: batch, sequence length, feature dimension.

| Tensor | Shape | Meaning |
|---|---|---|
| Token IDs | $(N, T)$ | $N$ sequences, each $T$ tokens, integer indices |
| Token embeddings | $(N, T, d_e)$ | each token ID looked up in the embedding table |
| Attention output | $(N, T, d_{model})$ | one vector per token position |

Operations that act per-token (e.g. the FFN inside a transformer) treat the $(N, T)$ axes as a flat batch and operate on the last axis. A linear layer with weight $(d_{model}, d_{ffn})$ applied to a $(N, T, d_{model})$ tensor produces $(N, T, d_{ffn})$. PyTorch broadcasts the matrix multiply over all leading dimensions automatically.

### 4D Tensors (Images)

Image data uses 4 axes: $(N, C, H, W)$, where $C$ is channels (e.g. 3 for RGB), $H$ is height, and $W$ is width.

| Tensor | Shape | Meaning |
|---|---|---|
| Batch of RGB images | $(N, 3, 224, 224)$ | $N$ images, 3 color channels, 224x224 pixels |
| After a conv layer | $(N, 64, 112, 112)$ | 64 feature maps, spatial size halved |
| After global average pool | $(N, 64)$ | one 64-dim vector per image |
| After final linear | $(N, C)$ | class logits |

A convolution kernel has shape $(C_{out}, C_{in}, k_H, k_W)$. It slides over the $(H, W)$ spatial dimensions, contracts over $C_{in}$, and produces $C_{out}$ output channels. Pooling layers reduce $H$ and $W$. Global average pooling collapses both spatial axes entirely, producing a 2D tensor $(N, C_{out})$ that can be fed into a standard linear layer.

### 5D Tensors (Video)

Video adds a time axis: $(N, C, T, H, W)$, where $T$ is the number of frames. 3D convolutions slide over both time and space. The same principle applies: label each axis, track what contracts, and verify with `print(x.shape)`.

### Reshape, View, Flatten

When dimensions need to be rearranged:

- `x.view(N, -1)` or `x.flatten(1)` collapses all axes after batch into one. A $(N, 64, 7, 7)$ tensor becomes $(N, 64 \times 7 \times 7) = (N, 3136)$. This is the standard bridge from convolutional layers to linear layers.
- `x.view(N, T, d)` restores a flattened sequence back to 3D.
- `x.permute(0, 2, 1)` swaps axes. A $(N, T, d)$ tensor becomes $(N, d, T)$, which is useful when switching between "channel-last" and "channel-first" formats.
- `x.unsqueeze(dim)` adds a size-1 axis for broadcasting. `x.squeeze(dim)` removes one.

The underlying data in memory does not change during a `view` or `reshape`. These operations only change how indices are interpreted.

### Debugging Tip

When an operation fails with a shape mismatch, print the shape of every input tensor right before the failing line. The error is almost always that two axes that should be the same size are not, or that a dimension is in the wrong position. Naming your axes makes the fix obvious.

## Why This Matters for LLMs

Transformer FFN layers are trained with exactly these principles at much larger scale. Token embeddings are just a large categorical embedding table, and many LLM stability tricks are the same ideas here repeated at scale: normalization, initialization, learning-rate schedules, clipping, and regularization. If MLP training feels clear, transformer training is conceptually easier.

## Next Post

We now move to transformers: attention, token embeddings, and why transformers replaced RNN-style sequence models.
