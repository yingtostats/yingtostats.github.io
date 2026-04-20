---
layout: post
title: "LLM Basics Series 3: Transformer Attention Intuition"
date: 2026-02-10 10:00:00
tag:
- Machine Learning
- LLM
projects: false
blog: true
author: YingZhang
coauthor_name: WenboGuo
coauthor_url: "https://henrygwb.github.io"
description: Beginner introduction to transformer attention and core building blocks.
fontsize: 23pt
---

{% include mathjax_support.html %}

This post starts the transformer section. We keep it simple: what attention does, why it works, and how one transformer block is built.

## Why Transformers Replaced RNN-Style Models

RNNs process tokens step by step, which is hard to parallelize and can struggle with long-range dependencies. Transformers process all tokens together and use attention to connect distant words.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">What "process all tokens together" means and why transformers can parallelize but RNNs cannot</span></summary>

<p><strong>RNN: sequential computation dependency.</strong> In an RNN, each hidden state is a function of the previous one:</p>

$$h_1 = f(x_1), \quad h_2 = f(x_2, h_1), \quad h_3 = f(x_3, h_2), \quad \ldots$$

<p>The GPU cannot start computing $h_3$ until $h_2$ is a finished number, because $h_3$ takes $h_2$ as input. This is a hard sequential dependency in the computation itself, not just in the data. Even during training when the full input sequence is known, the hidden states must be computed one after another.</p>

<p><strong>Transformer: parallel across all positions.</strong> Self-attention computes the relationship between every pair of tokens in one matrix operation. The queries, keys, and values are all derived from the input sequence at once:</p>

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V.$$

<p>The attention score matrix $QK^\top$ has shape $(T \times T)$, where entry $(i, j)$ is how much token $i$ attends to token $j$. All $T^2$ entries are computed simultaneously in a single matrix multiply.</p>

<p><strong>Two types of attention masks.</strong> There are two settings:</p>

<ul>
<li><strong>Bidirectional (encoder, BERT-style).</strong> The full $T \times T$ matrix. Every token attends to every other token. No masking.</li>
<li><strong>Causal/autoregressive (decoder, GPT-style).</strong> A lower-triangular mask. Row $i$ only attends to tokens $1, \ldots, i$. Future positions are set to $-\infty$ before softmax, zeroing them out. This is what LLMs use.</li>
</ul>

<p>In practice, the causal mask is implemented as an additive mask $M$ applied to the attention logits before softmax:</p>

$$\text{Attn} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

<p>where $M$ has 0 at allowed positions and $-\infty$ at blocked positions:</p>

$$M = \begin{pmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

<p>Adding $-\infty$ makes $e^{-\infty} = 0$ inside softmax, so those positions receive exactly zero attention weight. This is more numerically stable than multiplying by a binary 0/1 mask after softmax, because it guarantees the attention weights over allowed positions still sum to 1.</p>

<p><strong>The causal mask does not make computation sequential.</strong> Row 3 of the attention matrix uses keys from tokens 1, 2, 3. But it does not depend on the <em>result</em> of row 1 or row 2. All rows read from the same input vectors $k_1, k_2, \ldots, k_T$, which are all available from the start. The mask restricts what each row <em>sees</em>, but all rows are computed in one operation:</p>

<ul>
<li>Row 1: $[q_1 \cdot k_1]$</li>
<li>Row 2: $[q_2 \cdot k_1, \; q_2 \cdot k_2]$</li>
<li>Row 3: $[q_3 \cdot k_1, \; q_3 \cdot k_2, \; q_3 \cdot k_3]$</li>
<li>Row 4: $[q_4 \cdot k_1, \; q_4 \cdot k_2, \; q_4 \cdot k_3, \; q_4 \cdot k_4]$</li>
</ul>

<p>All four rows are computed simultaneously. The causal mask is a <em>data restriction</em> (cannot peek at the future), not a <em>computation dependency</em> (must wait for a previous result).</p>

<p><strong>Contrast during training.</strong></p>

<ul>
<li><strong>RNN training:</strong> still sequential. $h_3$ literally cannot be computed until $h_2$ is finished, because $h_3 = f(x_3, h_2)$.</li>
<li><strong>Transformer training:</strong> the entire $T \times T$ masked attention matrix is one batched matrix multiply. The full input sequence is known from the training data, so all positions are processed in parallel.</li>
</ul>

<p><strong>What about inference?</strong> During generation (inference), GPT-style models do produce tokens one at a time, because each new token depends on the model's own previous output. That part is inherently sequential. But training is fully parallel, which is why transformers are so much faster to train on long sequences and scale so well with GPU hardware.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<figure>
  <img src="https://arxiv.org/html/1706.03762v7/Figures/ModalNet-21.png" alt="The Transformer model architecture from Attention Is All You Need, showing the encoder (left) and decoder (right) with multi-head attention, feed-forward layers, residual connections, and layer normalization." style="width: 55%; display: block; margin: 0 auto;">
  <figcaption style="margin-top: 1.5em;">The Transformer architecture (Vaswani et al., 2017). The encoder (left) maps input tokens through stacked layers of multi-head self-attention and feed-forward networks. The decoder (right) adds masked self-attention and cross-attention to the encoder output. Each sub-layer has a residual connection and layer normalization.</figcaption>
</figure>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why transformer gradients are more stable than RNN/LSTM</span></summary>

<p><strong>The root cause in RNNs.</strong> An RNN applies the same transformation at every time step:</p>

$$h_t = f(W h_{t-1} + x_t).$$

<p>During backpropagation through time, the gradient from step $T$ back to step 0 involves a product of Jacobians:</p>

$$\frac{\partial h_T}{\partial h_0} = \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}.$$

<p>Each factor depends on $W$. If the spectral norm of $W$ is consistently less than 1, this product shrinks exponentially (vanishing gradients). If greater than 1, it grows exponentially (exploding gradients). The critical point: <strong>sequence length = depth of the computation graph</strong>. A 1,000-token sequence means multiplying through 1,000 Jacobians.</p>

<p><strong>LSTMs partially fix this.</strong> The LSTM cell state provides an additive path:</p>

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t.$$

<p>When the forget gate $f_t \approx 1$ and the input gate $i_t \approx 0$, the cell state passes through nearly unchanged, so gradients can survive longer chains. But the gates are themselves functions of $W$ and $h_{t-1}$, so the gradient path still involves multiplicative factors at every time step. For very long sequences, the product of gate values can still decay or grow, and the model must learn to hold gates open for exactly the right duration. LSTMs reduce the problem but do not eliminate it.</p>

<p><strong>What transformers change.</strong> Transformers do not propagate information sequentially across time. The entire sequence is processed in parallel through self-attention. The computation graph depth equals the number of layers (e.g., 12, 24, 96), <strong>not</strong> the sequence length (which could be 1,000 or 100,000 tokens).</p>

<p>The gradient path length is bounded and fixed, regardless of how long the input is. A 24-layer transformer processing a 10,000-token sequence still only backpropagates through 24 layers. An RNN processing the same sequence would backpropagate through 10,000 steps.</p>

<p><strong>But fixed depth alone is not enough.</strong> A plain 24-layer network without skip connections can still have vanishing or exploding gradients (the same issue from MLP training in Series 2). Transformers stay stable because of three additional design choices working together:</p>

<p><strong>(1) Residual connections.</strong> Each sub-layer computes</p>

$$x_{l+1} = x_l + \text{SubLayer}(x_l).$$

<p>The gradient through a residual block is $I + \frac{\partial \text{SubLayer}}{\partial x_l}$, where the identity $I$ provides a direct path for gradients to flow unchanged. Even if the sub-layer gradient is small, the identity term prevents vanishing. This is the same mechanism discussed in the MLP initialization section of Series 2.</p>

<p><strong>(2) Layer normalization.</strong> Applied before or after each sub-layer, LayerNorm keeps activations well-scaled and prevents values from drifting to extreme ranges where gradients saturate.</p>

<p><strong>(3) Attention as a short path.</strong> In an RNN, information from token 1 must pass through every intermediate hidden state to reach token 1,000. In a transformer, self-attention connects every token pair directly in a single layer. The gradient from any output token to any input token passes through at most $L$ layers (the network depth), regardless of how far apart they are in the sequence. There is no chain of $T$ sequential multiplications.</p>

<p><strong>Summary.</strong> The accurate statement is: transformers avoid the vanishing/exploding gradient problem because (1) computation depth is fixed and independent of sequence length (no sequential recurrence), (2) residual connections provide direct gradient paths through every layer, and (3) layer normalization keeps activations stable. It is not simply that "the depth is pre-specified"; it is that the depth does not grow with input length, and the architecture includes mechanisms that keep gradients well-behaved across the fixed number of layers.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Token Embedding

A tokenizer splits raw text into a sequence of $T$ integer token IDs. The embedding layer maps each ID to a dense vector:

$$
E = \text{EmbeddingLookup}(\text{token IDs}) \in \mathbb{R}^{T \times d_{model}},
$$

where $T$ is the sequence length and $d_{model}$ is the model's hidden dimension. As discussed in Series 2, this lookup is mathematically equivalent to multiplying a one-hot vector by the embedding weight matrix $W_e \in \mathbb{R}^{V \times d_{model}}$, where $V$ is the vocabulary size.

Typical values from real models (see the embedding table in Series 2):

| Model | Vocab $V$ | $d_{model}$ |
|---|---|---|
| GPT-2 Small | 50,257 | 768 |
| LLaMA 2 7B | 32,000 | 4,096 |
| Qwen3-8B | 151,936 | 4,096 |

The embedding table $W_e$ is a learned parameter. For Qwen3-8B, it contains $151{,}936 \times 4{,}096 \approx 622\text{M}$ parameters.

### Positional Encoding

Self-attention treats its input as a set: it computes pairwise scores between all tokens, but the operation itself is permutation-invariant. Shuffling the input rows would produce the same attention weights (just shuffled). Without position information, the model cannot distinguish "the cat sat on the mat" from "mat the on sat cat the."

Positional encoding adds position information so the model knows where each token sits in the sequence. Three methods are commonly used:

**Sinusoidal (original transformer).** Fixed sine and cosine functions at different frequencies, added to the token embeddings before attention:

$$
X = E + P, \quad P_{t, 2i} = \sin(t \cdot \omega_i), \quad P_{t, 2i+1} = \cos(t \cdot \omega_i), \quad \omega_i = 1/10000^{2i/d_{model}},
$$

where $t$ is the position index and $i$ is the dimension pair index. Each pair oscillates at a different frequency, giving each position a unique fingerprint. No learned parameters. Bounded, deterministic, and unique.

**Learned (GPT-2, BERT).** A learned embedding table $W_p \in \mathbb{R}^{T_{\max} \times d_{model}}$, added to the token embeddings the same way. Flexible, but cannot generalize beyond $T_{\max}$.

**RoPE (LLaMA, Qwen, Mistral).** Instead of adding to embeddings, RoPE *rotates* the Q and K vectors inside each attention head. For dimension pair $i$ at position $t$:

$$
\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}, \quad \theta = t \cdot \omega_i, \quad \omega_i = 1/10000^{2i/d_k}.
$$

The attention score between positions $t$ and $s$ becomes $q_t^\top R_{s-t}\, k_s$, which depends only on the relative distance $s - t$. This is the dominant method in modern LLMs. No learned parameters, strong length generalization, and content/position are cleanly separated.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Sinusoidal encoding: binary counting intuition and why both sin and cos</span></summary>

<p>The intuition comes from binary counting. In binary, the rightmost bit flips every step, the next bit flips every 2 steps, the next every 4, and so on. Each digit oscillates at a different frequency, and the combination uniquely identifies every number. Sinusoidal positional encoding does the same thing with smooth waves instead of binary flips: dimension pair $i = 0$ oscillates fastest (like the least significant bit), and later pairs oscillate progressively slower (like more significant bits).</p>

<p><strong>Why both sin and cos.</strong> Using both functions for each frequency allows the model to compute relative positions via a linear transformation. The trigonometric addition formulas give:</p>

$$\begin{pmatrix} \sin((t+k)\omega) \\ \cos((t+k)\omega) \end{pmatrix} = \begin{pmatrix} \cos(k\omega) & \sin(k\omega) \\ -\sin(k\omega) & \cos(k\omega) \end{pmatrix} \begin{pmatrix} \sin(t\omega) \\ \cos(t\omega) \end{pmatrix}.$$

<p>For any fixed offset $k$, there exists a linear transformation (a rotation matrix) that maps $P_t$ to $P_{t+k}$. The model can learn this linear map, which means it can learn "the token $k$ positions to the left" using a simple matrix multiply. If only sine were used without cosine, the model would need nonlinear operations to recover relative positions.</p>

<p>This rotation matrix is the same 2D rotation matrix that RoPE uses. Both methods share the same mathematical foundation.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">RoPE: where it operates, what each parameter means, and a worked example</span></summary>

<p><strong>Where RoPE operates.</strong> Unlike sinusoidal and learned encodings which are added at the $d_{model}$ level before attention, RoPE is applied <em>inside</em> the attention mechanism, after the Q and K projections have already split the $d_{model}$-dimensional vector into $H$ heads of $d_k$ dimensions each:</p>

<ol>
<li>Embedding: $(T, d_{model})$. Sinusoidal or learned PE would be added here.</li>
<li>Q and K projections: $(T, d_{model}) \to H$ heads of $(T, d_k)$ each, where $d_k = d_{model}/H$.</li>
<li>RoPE is applied here, to each head's $(T, d_k)$ query and key vectors independently.</li>
</ol>

<p>So RoPE operates at the per-head dimension $d_k$, not $d_{model}$.</p>

<p><strong>The three quantities in $\theta = t \cdot \omega_i$:</strong></p>

<ul>
<li>$t$ is the <strong>position index</strong> of the token in the sequence. $t = 0$ for the first token, $t = 1$ for the second, and so on. Tokens further in the sequence get rotated by a larger angle.</li>
<li>$i$ is the <strong>dimension pair index</strong> within one token's vector. $i = 0$ for dimensions $(0, 1)$, $i = 1$ for dimensions $(2, 3)$, etc. A 2D rotation takes exactly 2 numbers as input, so the $d_k$ dimensions are grouped into $d_k/2$ pairs.</li>
<li>$\omega_i = 1/10000^{2i/d_k}$ is the <strong>frequency</strong> for pair $i$. Pair $i = 0$ has $\omega_0 = 1$ (fastest, encodes fine-grained local position). The last pair has $\omega \approx 1/10000$ (slowest, encodes coarse long-range position). Position $t$ determines <em>how much</em> to rotate; frequency $\omega_i$ determines <em>how sensitive</em> that pair is to position changes.</li>
</ul>

<p><strong>Worked example.</strong> Consider $T = 3$ tokens ("I", "like", "cats") with $d_{model} = 8$ and $H = 2$ heads, so $d_k = 4$ per head.</p>

<p>Start from the token embeddings $X \in \mathbb{R}^{3 \times 8}$. The full Q projection gives $Q_{\text{full}} = XW_Q \in \mathbb{R}^{3 \times 8}$. Split the 8 columns into $H = 2$ heads of $d_k = 4$:</p>

<pre><code>Q_full:
         [  head 0 (cols 0-3)  |  head 1 (cols 4-7)  ]
"I":     [  q00, q01, q02, q03 |  q04, q05, q06, q07 ]
"like":  [  q10, q11, q12, q13 |  q14, q15, q16, q17 ]
"cats":  [  q20, q21, q22, q23 |  q24, q25, q26, q27 ]
</code></pre>

<p>For head 0, the query matrix is $(3, 4)$. RoPE groups the $d_k = 4$ dimensions into $d_k/2 = 2$ pairs: dimensions $(0, 1)$ form pair 0, and dimensions $(2, 3)$ form pair 1:</p>

<pre><code>            pair i=0       pair i=1
            dim 0, dim 1   dim 2, dim 3
t=0  "I":   (q00, q01)     (q02, q03)
t=1  "like":(q10, q11)     (q12, q13)
t=2  "cats":(q20, q21)     (q22, q23)
</code></pre>

<p>RoPE rotates each pair by angle $t \times \omega_i$:</p>

<ul>
<li>Pair $i=0$: frequency $\omega_0 = 1$ (fast). Token "cats" at $t=2$ gets rotated by $2 \times 1 = 2$ radians.</li>
<li>Pair $i=1$: frequency $\omega_1 = 1/10000^{2/4} = 0.01$ (slow). Token "cats" at $t=2$ gets rotated by $2 \times 0.01 = 0.02$ radians.</li>
</ul>

<p>Early pairs capture fine-grained position differences; later pairs capture coarse, long-range patterns. In Qwen3-8B with $d_k = 128$, there are 64 pairs spanning frequencies from fast to slow.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why the relative position property falls out from rotation matrices</span></summary>

<p>Write the rotated query and key as $R_t q_t$ and $R_s k_s$, where $R_t$ is the block-diagonal rotation matrix for position $t$. The attention score between tokens at positions $t$ and $s$ is:</p>

$$
(R_t q_t)^\top (R_s k_s) = q_t^\top R_t^\top R_s\, k_s = q_t^\top R_{s-t}\, k_s.
$$

<p>The last step uses the fact that rotation matrices satisfy $R_t^\top R_s = R_{s-t}$. The absolute positions $t$ and $s$ disappear, and only the gap $s - t$ remains. The model does not care whether "cat sat" appears at positions (3, 4) or (500, 501); it sees the same relative distance of 1 in both cases.</p>

<p>Why this matters:</p>

<ul>
<li><strong>Relative position awareness.</strong> The model naturally learns distance-based patterns rather than memorizing absolute position slots.</li>
<li><strong>Length generalization.</strong> Since attention scores depend on relative distance, a model trained on 4K-length sequences can potentially generalize to 8K or longer at inference time.</li>
<li><strong>No extra parameters.</strong> RoPE is deterministic with no learned parameters.</li>
<li><strong>Applied to Q and K only.</strong> The value vectors $V$ are not rotated, because position information only needs to affect which tokens attend to each other, not the content that gets aggregated.</li>
</ul>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Comparison Of Positional Encoding Methods

| | Sinusoidal | Learned | RoPE |
|---|---|---|---|
| Where applied | Added to embeddings before attention | Added to embeddings before attention | Applied to Q, K inside attention |
| Operates at | $d_{model}$ | $d_{model}$ | $d_k$ (per head) |
| Parameters | None (fixed) | $T_{\max} \times d_{model}$ learned | None (fixed) |
| Position type | Absolute | Absolute | Relative |
| Length generalization | Possible in theory (smooth extrapolation) but limited in practice | Cannot exceed $T_{\max}$ | Strong (only relative distance matters) |
| Used in | Original transformer | GPT-2, BERT | LLaMA, Qwen, Mistral, GPT-NeoX |

**Sinusoidal vs RoPE.** Both use the same mathematical structure: sine/cosine pairs at geometrically decreasing frequencies. The rotation matrix in the sin/cos addition formula is the same 2D rotation matrix that RoPE uses. The difference is how position is injected:

- Sinusoidal: computes a fixed position vector and *adds* it to the token embedding. The position information mixes with the content information before Q, K, V projections. All downstream layers see the summed signal, and the model must learn to separate "what the token means" from "where the token is."
- RoPE: *rotates* the Q and K vectors after projection. Position information enters only through the attention scores. The value vectors $V$ are untouched, keeping content and position cleanly separated. This also means the FFN layers (which operate on the residual stream, not on Q/K) do not see position encoding directly.

The relative-position property is the main practical advantage. Sinusoidal encoding *can* represent relative positions through a linear map (as shown above), but the model must learn to extract it. RoPE makes relative distance appear directly in the attention score formula ($q_t^\top R_{s-t} k_s$), so the structure is built in rather than learned.

**Learned vs sinusoidal.** Learned embeddings can in principle represent any position pattern, since each position gets its own free vector. In practice, for short sequences where training data covers all positions well, learned embeddings perform comparably to sinusoidal. The disadvantage is the hard ceiling at $T_{\max}$: the model has no embedding for position $T_{\max} + 1$. Sinusoidal encoding has no such limit since the function is defined for any $t$.

**Why RoPE won.** For modern LLMs that need to handle long contexts (4K to 128K+ tokens), the combination of relative position encoding, no extra parameters, clean separation of content and position, and strong length generalization made RoPE the dominant choice. Nearly all open-weight LLMs released since 2023 use RoPE.

## Self-Attention Core

From the position-encoded input $X \in \mathbb{R}^{T \times d_{model}}$, self-attention builds three matrices using learned projections:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V,
$$

where $W_Q, W_K \in \mathbb{R}^{d_{model} \times d_k}$ and $W_V \in \mathbb{R}^{d_{model} \times d_v}$. In the original transformer, $d_k = d_v = d_{model} / H$, where $H$ is the number of attention heads (explained in the next section). The resulting shapes are:

- $Q \in \mathbb{R}^{T \times d_k}$ (queries: what each token is looking for),
- $K \in \mathbb{R}^{T \times d_k}$ (keys: what each token advertises),
- $V \in \mathbb{R}^{T \times d_v}$ (values: what each token contributes if attended to).

The attention output is:

$$
\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \in \mathbb{R}^{T \times d_v}.
$$

Step by step:

1. $QK^\top \in \mathbb{R}^{T \times T}$: the raw attention score matrix. Entry $(i, j)$ measures how much token $i$ should attend to token $j$.
2. Divide by $\sqrt{d_k}$: prevents the dot products from growing too large when $d_k$ is big, which would push softmax into saturation where gradients vanish.
3. Softmax over each row: normalizes scores so that each row sums to 1, giving attention weights.
4. Multiply by $V$: each token's output is a weighted average of all value vectors, weighted by the attention scores.

Each token asks, "Which other tokens are relevant to me?" The attention weights are the answer, and the output is the weighted combination of their value vectors.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why three separate projections (Q, K, V) instead of using embeddings directly</span></summary>

<p><strong>The problem with raw embeddings.</strong> The simplest attention would compute similarity between token embeddings directly ($XX^\top$), then use that to mix them. But this forces each token's vector to serve two conflicting roles simultaneously: (1) describe what the token is <em>looking for</em>, and (2) describe what the token <em>offers</em> to others. These are different things. When you read "The cat sat on the mat", the word "sat" needs to look for a subject (who sat?) but offer an action (what happened?). One vector cannot optimally do both.</p>

<p><strong>The database analogy.</strong> Think of a key-value database. You search with a query, the system compares it against all keys to find relevant entries, and returns the matching values. In self-attention:</p>

<ul>
<li><strong>Query</strong> ($Q$): what this token is looking for. ("I need a subject.")</li>
<li><strong>Key</strong> ($K$): what this token advertises to others. ("I am a noun/agent.")</li>
<li><strong>Value</strong> ($V$): the actual content to pass along if selected. ("Here is my semantic representation.")</li>
</ul>

<p>Each is a different learned linear projection of the same embedding, so the same token produces three different vectors serving three different roles:</p>

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V.$$

<p>$W_Q$ learns to extract "what am I looking for?", $W_K$ learns to extract "what should I advertise?", and $W_V$ learns to extract "what content should I pass along?"</p>

<p><strong>Example 1: "The cat sat on the mat."</strong> Consider "sat" building its contextualized representation. Without Q/K/V separation, "sat" has one vector that must simultaneously be similar to "cat" (to find the subject) and "mat" (to find the location), but "cat" and "mat" are very different tokens. With Q/K/V separation:</p>

<ul>
<li>"sat" produces a query $q_{\text{sat}}$ encoding "I need a noun/subject."</li>
<li>"cat" produces a key $k_{\text{cat}}$ encoding "I am a noun/agent" and a value $v_{\text{cat}}$ carrying the semantic content of "cat."</li>
<li>"the" produces a key $k_{\text{the}}$ encoding "I am a determiner."</li>
</ul>

<p>The score $q_{\text{sat}} \cdot k_{\text{cat}}$ is high, so the output for "sat" receives a large weight of $v_{\text{cat}}$. Now the representation of "sat" carries information about its subject.</p>

<p><strong>Example 2: "She gave him the book because he asked for it."</strong> The pronoun "it" needs to resolve what it refers to. Its query encodes "I need a concrete noun/object." The keys for "book", "she", "him" all advertise their roles. The key for "book" (concrete object) matches the query for "it" most strongly, so "it" receives the value of "book" and its output representation carries the meaning of what "it" refers to. Meanwhile "he" might produce a query encoding "I need a person/subject", which matches the key of "she" or "him" depending on the context. Different tokens ask different questions through their queries, and the same token can be a good answer for one query but not another.</p>

<p><strong>What if you removed Q/K separation?</strong> Using $XX^\top$ for scores makes attention symmetric: "cat" attending to "sat" would have the same score as "sat" attending to "cat". But these should be different. "sat" looks for a subject (high attention to "cat"), while "cat" looks for a predicate (high attention to "sat" for a different reason). Separate $W_Q$ and $W_K$ projections break this symmetry, allowing asymmetric attention patterns.</p>

<p><strong>What if you removed V?</strong> If you use $K$ as $V$ (i.e., the output is a weighted sum of key vectors), then the information used for <em>matching</em> is the same information <em>passed along</em>. Keys and values encode fundamentally different things:</p>

<ul>
<li>Keys encode a token's <strong>role</strong>: what kind of token it is (noun, verb, subject, object, animate, etc.), so that queries can find it. Tokens with similar roles should have similar keys.</li>
<li>Values encode a token's <strong>semantic meaning</strong>: what the word actually means (cat = furry animal; mat = flat surface). Tokens with the same role can have completely different meanings.</li>
</ul>

<p>The separation matters because role similarity and meaning similarity are independent. Consider "The <strong>cat</strong> sat on the <strong>mat</strong>." Both "cat" and "mat" are nouns, so they have similar keys (both match a "looking for a noun" query). But they mean entirely different things, so they need different values.</p>

<p><strong>With separate V</strong>, "sat" attends to "cat" with weight 0.7 and "mat" with weight 0.2 (via $QK^\top$), and the output is:</p>

$$0.7 \times v_{\text{cat}}(\text{furry, animal, alive}) + 0.2 \times v_{\text{mat}}(\text{flat, fabric, floor}).$$

<p>The output carries mostly "cat" meaning. The model knows <em>which</em> noun "sat" relates to.</p>

<p><strong>Without V</strong> ($K$ used as $V$), the output is a weighted sum of keys:</p>

$$0.7 \times k_{\text{cat}}(\text{noun, subject, animate}) + 0.2 \times k_{\text{mat}}(\text{noun, object, inanimate}).$$

<p>You get a blurry mix of <em>roles</em>, not <em>meanings</em>. The model knows "sat" relates to something that is mostly a subject-noun, but it has lost <em>which</em> noun, because both "cat" and "mat" advertised similar role information as nouns. The sentences "the cat sat" and "the mat sat" would produce nearly identical outputs for "sat."</p>

<p>The core principle: role similarity (both nouns, similar keys) is exactly what makes matching work, but it is also what would make the output indistinguishable if keys were reused as values. Separate $V$ breaks this by allowing "similar keys, different values" to coexist. The matching function ($QK^\top$) and the content function ($V$) are fully decoupled.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Multi-Head Attention

Instead of running one attention operation with $d_k = d_{model}$, multi-head attention runs $H$ attention heads in parallel, each with a smaller dimension $d_k = d_{model} / H$.

For each head $h = 1, \ldots, H$:

$$
Q_h = XW_Q^{(h)}, \quad K_h = XW_K^{(h)}, \quad V_h = XW_V^{(h)},
$$

where $W_Q^{(h)}, W_K^{(h)} \in \mathbb{R}^{d_{model} \times d_k}$ and $W_V^{(h)} \in \mathbb{R}^{d_{model} \times d_v}$. Each head computes its own attention:

$$
\text{head}_h = \text{Attn}(Q_h, K_h, V_h) \in \mathbb{R}^{T \times d_v}.
$$

The $H$ head outputs are concatenated along the feature dimension and projected back to $d_{model}$:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\, W_O \in \mathbb{R}^{T \times d_{model}},
$$

where $W_O \in \mathbb{R}^{(H \cdot d_v) \times d_{model}}$ is the output projection. Since $H \cdot d_v = H \cdot (d_{model}/H) = d_{model}$, the concatenated dimension matches $d_{model}$, and the output has the same shape as the input.

Why multiple heads help: each head can learn to attend to different types of relationships. One head might focus on syntactic dependencies (subject-verb), another on coreference (pronoun-noun), another on local context (adjacent words). A single head would have to compress all these patterns into one set of attention weights.

Typical head counts from real models:

| Model | $d_{model}$ | Heads $H$ | $d_k = d_{model}/H$ |
|---|---|---|---|
| BERT-base | 768 | 12 | 64 |
| GPT-2 Small | 768 | 12 | 64 |
| GPT-3 175B | 12,288 | 96 | 128 |
| LLaMA 2 7B | 4,096 | 32 | 128 |
| Qwen3-8B | 4,096 | 32 | 128 |
| Qwen3-32B | 5,120 | 64 | 80 |

The per-head dimension $d_k$ is typically 64 or 128 across model sizes. Larger models increase the number of heads, not the per-head dimension. The reason is that each head learns to attend to one type of pattern (syntactic, coreference, local context, etc.). A larger model processing more complex text benefits from more *distinct* attention patterns, not from making each existing pattern higher-resolution. Increasing $d_k$ beyond 64-128 gives diminishing returns because a 128-dimensional dot product is already expressive enough to distinguish relevant tokens from irrelevant ones. But going from 32 heads to 64 heads lets the model track twice as many independent relationships in parallel. In terms of parameter count, both options use the same total: $H \times d_k = d_{model}$, so 64 heads at $d_k = 64$ has the same parameters as 32 heads at $d_k = 128$. There is also a lower bound on $d_k$: if the per-head dimension is too small, the attention score $q^\top k$ does not have enough capacity to distinguish relevant tokens from irrelevant ones (a rank bottleneck in the attention matrix). In practice, $d_k = 64$ or 128 is the sweet spot where each head is expressive enough, and larger models spend additional capacity on more heads rather than wider heads.

**Grouped-query attention (GQA).** In standard multi-head attention, each head has its own $K$ and $V$ projections. GQA shares $K$ and $V$ across groups of query heads. For example, Qwen3-8B has 32 query heads but only 8 KV heads, so every 4 query heads share the same keys and values. This reduces memory and computation during inference (fewer KV cache entries) with minimal quality loss. The query projections remain independent, so the model retains most of its expressive power.

## One Transformer Block

The full pipeline from raw tokens to the first transformer block is:

1. **Token embedding**: token IDs $\to$ $E \in \mathbb{R}^{T \times d_{model}}$.
2. **Positional encoding**: add sinusoidal/learned PE to get $X = E + P$, or leave $X = E$ if using RoPE (which is applied later inside attention).
3. **Feed into block 1**.

Each block then applies:

1. Multi-head self-attention (with RoPE applied to Q, K inside the attention if using RoPE).
2. Residual connection + layer norm.
3. MLP/FFN.
4. Residual connection + layer norm.

Written out for input $X \in \mathbb{R}^{T \times d_{model}}$:

$$
X' = \text{LayerNorm}(X + \text{MultiHead}(X)),
$$

$$
\text{BlockOutput} = \text{LayerNorm}(X' + \text{FFN}(X')) \in \mathbb{R}^{T \times d_{model}}.
$$

Positional encoding is only injected once. For sinusoidal or learned PE, it is added before block 1 and then carried through all subsequent blocks via the residual connections. For RoPE, the rotation is applied to Q and K inside every block's attention (each block re-applies RoPE to its own Q/K projections using the same position-dependent rotation).

The FFN is a two-layer MLP applied independently to each token position:

$$
\text{FFN}(x) = \phi(xW_1 + b_1)W_2 + b_2,
$$

where $W_1 \in \mathbb{R}^{d_{model} \times d_{ffn}}$, $W_2 \in \mathbb{R}^{d_{ffn} \times d_{model}}$, and $d_{ffn}$ is typically 4x to 8x $d_{model}$. For example, Qwen3-8B has $d_{model} = 4{,}096$ and $d_{ffn} = 12{,}288$ (3x). Modern LLMs often use SwiGLU activation (a gated variant of ReLU) instead of plain ReLU.

The block output has the same shape as the input: $\mathbb{R}^{T \times d_{model}}$. This is what allows blocks to be stacked. A full transformer repeats this block $L$ times (e.g., 36 layers for Qwen3-8B, 96 layers for GPT-3).

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Batched sequences: from $(T, d_{model})$ to $(N, T, d_{model})$</span></summary>

<p>The formulas above show a single sequence for clarity, but in practice a batch of $N$ sequences is processed together. All tensors gain a leading batch dimension:</p>

<ul>
<li><strong>Embedding lookup:</strong> $N$ sequences of $T$ token IDs $\to (N, T, d_{model})$.</li>
<li><strong>Q, K, V projections:</strong> the matrix multiply broadcasts over $N$ and $T$. $(N, T, d_{model}) \times (d_{model}, d_k) \to (N, T, d_k)$ per head.</li>
<li><strong>Attention scores:</strong> $QK^\top$ is computed per sample. $(N, T, d_k) \times (N, d_k, T) \to (N, T, T)$. Each sample gets its own $(T \times T)$ attention matrix. Sample 1's tokens never attend to sample 2's tokens.</li>
<li><strong>Causal mask:</strong> the same $(T \times T)$ mask $M$ is broadcast identically across all $N$ samples.</li>
<li><strong>FFN:</strong> applied independently to each token in each sample. $(N, T, d_{model}) \to (N, T, d_{ffn}) \to (N, T, d_{model})$.</li>
<li><strong>LayerNorm:</strong> normalizes across $d_{model}$ for each token in each sample independently. It does not mix across the batch or across positions.</li>
</ul>

<p>With multi-head attention, the full working shape is $(N, H, T, d_k)$: batch, heads, sequence length, per-head dimension. All four dimensions are handled in parallel by the GPU.</p>

<p>The training loss is averaged over both positions and the batch:</p>

$$\mathcal{L} = -\frac{1}{N \cdot T}\sum_{n=1}^{N}\sum_{t=0}^{T-1} \log P_\theta(x_{t+1}^{(n)} \mid x_{\leq t}^{(n)}).$$

<p>The batch dimension is purely for GPU parallelism and training efficiency. Each sample's attention and gradient are independent.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">LayerNorm vs BatchNorm: what they normalize and why transformers use LayerNorm</span></summary>

<p>Both normalize activations to stabilize training, but they normalize across different dimensions.</p>

<p><strong>BatchNorm</strong> normalizes across the <em>batch</em> dimension. For a batch of $N$ samples, each with $d$ features, it computes the mean and variance of each feature <em>across all samples in the batch</em>:</p>

$$\hat{x}_j = \frac{x_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}, \quad \mu_j = \frac{1}{N}\sum_{i=1}^{N} x_{ij}, \quad \sigma_j^2 = \frac{1}{N}\sum_{i=1}^{N}(x_{ij} - \mu_j)^2.$$

<p>Feature $j$ is normalized using statistics from all $N$ samples. This works well when the batch is large enough that the per-feature statistics are stable.</p>

<p><strong>LayerNorm</strong> normalizes across the <em>feature</em> dimension. In a transformer, this means <strong>per-token normalization</strong>. Each token's $d_{model}$-dimensional vector is normalized independently using its own mean and variance:</p>

$$\hat{h}_j = \frac{h_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad \mu = \frac{1}{d_{model}}\sum_{j=1}^{d_{model}} h_j, \quad \sigma^2 = \frac{1}{d_{model}}\sum_{j=1}^{d_{model}}(h_j - \mu)^2.$$

<p>For a batch with shape $(N, T, d_{model})$, there are $N \times T$ independent normalizations, each over $d_{model}$ values. No information crosses between tokens, and no information crosses between samples:</p>

<pre><code>Sample 1, Token 0: [0.5, -1.2, 0.8, ..., -0.3]  ← normalize these d_model values
Sample 1, Token 1: [1.1,  0.3, 2.1, ...,  0.7]  ← normalize these d_model values
Sample 1, Token 2: [0.2, -0.5, 0.1, ...,  1.4]  ← normalize these d_model values
...
Sample N, Token T: [...]                           ← normalize these d_model values
</code></pre>

<p>Each row is standardized to mean 0 and variance 1 on its own.</p>

<p><strong>Visually comparing the two</strong>, for a tensor of shape $(N, T, d_{model})$:</p>

<pre><code>BatchNorm: normalizes across N (and T) for each feature dimension
           ↓ down the batch axis
           [sample 1, token 1, feature j]
           [sample 2, token 1, feature j]  ← compute mean/var of feature j
           [sample 3, token 1, feature j]
           ...

LayerNorm: normalizes across d_model for each token in each sample
           → across the feature axis
           [sample 1, token 1, feature 0, feature 1, ..., feature d]
                                ← compute mean/var across all features
</code></pre>

<p><strong>Why transformers use LayerNorm:</strong></p>

<ul>
<li><strong>Variable sequence lengths.</strong> In a batch of sequences, different samples may have different lengths (padded to the same $T$). BatchNorm would compute statistics mixing real tokens with padding, corrupting the normalization. LayerNorm operates per token independently, so padding in other samples does not matter.</li>
<li><strong>Small or variable batch sizes.</strong> BatchNorm needs a reasonably large batch for stable statistics. During inference (batch size = 1) or with small batches, the estimates become noisy. LayerNorm works identically regardless of batch size, because it only looks within one sample.</li>
<li><strong>Autoregressive generation.</strong> During token-by-token generation, each new token must be normalized. LayerNorm can normalize one token using its own $d_{model}$ features. BatchNorm would need statistics from a batch of sequences at the same position, which is not naturally available during generation.</li>
<li><strong>No running statistics needed.</strong> BatchNorm maintains running mean/variance for inference (computed during training). LayerNorm computes everything on the fly from the current input, making it simpler and avoiding train/inference mismatch.</li>
</ul>

<p>Both methods also learn a per-feature scale $\gamma$ and shift $\beta$ parameter after normalization: $y_j = \gamma_j \hat{x}_j + \beta_j$. These let the model undo the normalization if needed.</p>

<p><strong>When to use which:</strong></p>

<ul>
<li><strong>BatchNorm:</strong> standard MLPs and CNNs with large, fixed-size batches (image classification, tabular data). As discussed in Series 2.</li>
<li><strong>LayerNorm:</strong> transformers, RNNs, and any setting with variable-length sequences or small batches. All modern LLMs use LayerNorm.</li>
</ul>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Encoder vs Decoder

Three architectures exist, differing in what attention can see and how the model is used.

**Encoder-only (BERT-style)**: bidirectional attention, used for representation and classification.

**Decoder-only (GPT-style)**: causal attention, used for autoregressive text generation. This is what modern LLMs use.

**Encoder-decoder (original transformer, T5)**: encoder with bidirectional attention, decoder with causal attention plus cross-attention to the encoder. Used for sequence-to-sequence tasks.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Decoder-only: causal attention and autoregressive generation</span></summary>

<p><strong>The autoregressive formulation.</strong> A decoder-only LLM models the probability of a sequence by factoring it into a chain of next-token predictions:</p>

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1}).$$

<p>At each position $t$, the model predicts the next token using only the tokens before it. This is enforced by the causal attention mask.</p>

<p><strong>How the causal mask creates $T$ training examples from one sequence.</strong> During training, the full sequence is known. A key question is: does the model run $T$ separate forward passes, one for each prefix? No. The causal mask achieves the same effect in a single forward pass.</p>

<p>Consider the input sequence [The, cat, sat, on, the, mat] with $T = 6$ tokens. The model runs one forward pass through all $L$ transformer blocks. At every attention layer, the $QK^\top$ score matrix has shape $(6, 6)$. The causal mask $M$ is added before softmax:</p>

$$\frac{QK^\top}{\sqrt{d_k}} + M, \quad M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

<p>After softmax, the attention weight matrix looks like (schematically, with $w_{ij}$ denoting the attention weight from token $i$ to token $j$):</p>

<pre><code>          The   cat   sat   on    the   mat
The   [  1.0    0     0     0     0     0  ]
cat   [  0.3   0.7    0     0     0     0  ]
sat   [  0.1   0.5   0.4    0     0     0  ]
on    [  0.05  0.15  0.6   0.2    0     0  ]
the   [  0.02  0.08  0.1   0.3   0.5    0  ]
mat   [  0.01  0.04  0.05  0.1   0.3   0.5]
</code></pre>

<p>Each row sums to 1 over its allowed positions. The zeros above the diagonal come from the $-\infty$ mask making $e^{-\infty} = 0$ in softmax. The critical property: <strong>row $t$'s output depends only on tokens $0, \ldots, t$</strong>, even though all rows are computed in the same matrix multiply. Row 2 ("sat") is computed using only "The", "cat", "sat" — exactly as if the model only saw those three tokens.</p>

<p>After all $L$ transformer blocks, the model produces an output vector $h_t \in \mathbb{R}^{d_{model}}$ for each position $t$. Each $h_t$ was built using only tokens $0, \ldots, t$ (the mask was applied at every layer). Now each position independently predicts the next token:</p>

<pre><code>Position 0: h_0 was built from [The]                    → predicts "cat"
Position 1: h_1 was built from [The, cat]               → predicts "sat"
Position 2: h_2 was built from [The, cat, sat]          → predicts "on"
Position 3: h_3 was built from [The, cat, sat, on]      → predicts "the"
Position 4: h_4 was built from [The, cat, sat, on, the] → predicts "mat"
</code></pre>

<p><strong>One forward pass, $T$ predictions.</strong> The model computes attention once with the $(T \times T)$ masked matrix, passes through all blocks once, and produces $T$ output vectors. Each output vector is equivalent to what you would get if you ran the model separately on that prefix. The causal mask guarantees this equivalence because information cannot leak from future positions.</p>

<p><strong>The output head and loss.</strong> After the final transformer block, each position's output is projected to vocabulary size $V$:</p>

$$\text{logits}_t = h_t W_{\text{head}} \in \mathbb{R}^{V}, \quad P(x_{t+1} \mid x_{\leq t}) = \text{softmax}(\text{logits}_t).$$

<p>The training loss is the average cross-entropy across all $T$ positions:</p>

$$\mathcal{L} = -\frac{1}{T}\sum_{t=0}^{T-1} \log P_\theta(x_{t+1} \mid x_{\leq t}).$$

<p>So one sequence of $T$ tokens produces $T$ training signals (one per position), all from a single forward pass. This is extremely efficient compared to running $T$ separate forward passes for each prefix.</p>

<p>The output projection $W_{\text{head}} \in \mathbb{R}^{d_{model} \times V}$ often shares weights with the token embedding table $W_e \in \mathbb{R}^{V \times d_{model}}$ (called weight tying: $W_{\text{head}} = W_e^\top$). This reduces parameters and ties the input/output representations together.</p>

<p><strong>Inference (generation).</strong> At inference time, the model generates tokens one at a time. Starting from a prompt, it runs a forward pass, takes the logits at the last position, samples or picks the highest-probability token, appends it to the sequence, and repeats. Each step is autoregressive: the output of step $t$ becomes part of the input for step $t+1$. This is inherently sequential, unlike training where all positions are computed in parallel.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Encoder-only: bidirectional attention and masked language modeling</span></summary>

<p><strong>Bidirectional attention.</strong> In an encoder-only model like BERT, there is no causal mask. Every token attends to every other token, including tokens that come <em>after</em> it in the sequence. The full $T \times T$ attention matrix is used with no entries masked out.</p>

<p>This means each token's output representation is informed by the entire sequence in both directions. For a sentence like "The bank by the river", the word "bank" can attend to "river" to disambiguate its meaning (riverbank, not financial bank), even though "river" appears later.</p>

<p><strong>Training objective: masked language modeling (MLM).</strong> Since the model sees all tokens, it cannot be trained to predict the next token (it would just look ahead and copy the answer). Instead, BERT randomly masks ~15% of input tokens (replacing them with a [MASK] token) and trains the model to predict the original token at those masked positions:</p>

<pre><code>Input:  [The, cat, [MASK], on, the, [MASK]]
Target:  predict "sat" at position 2, "mat" at position 5
</code></pre>

<p>The model must use the full bidirectional context to fill in the blanks.</p>

<p><strong>Use cases.</strong> Encoder-only models produce a rich contextual representation for each token, making them well-suited for tasks where you need to understand the full input: text classification, named entity recognition, semantic similarity, and extractive question answering. They are not designed for open-ended text generation.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Encoder-decoder: cross-attention and sequence-to-sequence tasks</span></summary>

<p><strong>Two stacks.</strong> An encoder-decoder model has two separate stacks of transformer blocks. The encoder processes the input sequence (e.g., a French sentence) with bidirectional attention. The decoder generates the output sequence (e.g., the English translation) autoregressively with causal attention.</p>

<p><strong>Cross-attention.</strong> The decoder has an additional attention layer that connects it to the encoder. In each decoder block, the structure is:</p>

<ol>
<li>Causal self-attention (decoder attends to its own previous tokens).</li>
<li>Cross-attention (decoder attends to the encoder's output).</li>
<li>FFN.</li>
</ol>

<p>In cross-attention, the queries come from the decoder and the keys and values come from the encoder:</p>

$$Q = X_{\text{decoder}} W_Q, \quad K = X_{\text{encoder}} W_K, \quad V = X_{\text{encoder}} W_V.$$

<p>This lets each decoder token ask: "Which parts of the input are relevant for generating my next word?" For translation, a decoder token generating the English word "cat" would attend strongly to the French encoder tokens for "chat."</p>

<p><strong>No causal mask on cross-attention.</strong> The decoder can attend to all encoder positions (the full input is known). The causal mask only applies to the decoder's self-attention (cannot peek at future output tokens).</p>

<p><strong>Use cases.</strong> Translation, summarization, speech-to-text, and any task where the input and output are different sequences. The original "Attention Is All You Need" transformer was an encoder-decoder model for machine translation. T5 and BART are well-known encoder-decoder models.</p>

<p><strong>Why decoder-only won for LLMs.</strong> Encoder-decoder requires a clear separation of "input" and "output", which fits translation but not open-ended conversation. Decoder-only models treat everything as one continuous sequence (prompt + response), which is simpler and more flexible. At sufficient scale, decoder-only models match or exceed encoder-decoder performance on most tasks, so the added complexity of two stacks is rarely justified for general-purpose LLMs.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Beginner Mental Model

- Attention mixes information across tokens (inter-token communication).
- FFN transforms each token representation independently (per-token computation).
- Stacking blocks alternates between these two operations, building progressively deeper language understanding.

## Next Post

We will cover practical transformer training and inference concepts: pretraining, fine-tuning, context windows, and decoding.
