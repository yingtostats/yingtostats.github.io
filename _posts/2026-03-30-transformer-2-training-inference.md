---
layout: post
title: "LLM Basics Series 4: Transformer Training and Inference"
date: 2026-02-20 10:00:00
tag:
- Machine Learning
- LLM
projects: false
blog: true
author: YingZhang
coauthor_name: WenboGuo
coauthor_url: "https://henrygwb.github.io"
description: Beginner guide to how transformers are trained and how text generation works.
fontsize: 23pt
---

{% include mathjax_support.html %}

This post turns transformer architecture into practical workflow: how models are trained and how they generate text.

## Pretraining Objective (Decoder LLM)

Most LLMs are trained by maximizing the log-likelihood of next-token prediction over a large text corpus. Given a training sequence $x = (x_1, x_2, \ldots, x_T)$, the autoregressive factorization from Series 3 gives:

$$
\max_\theta \sum_{t=1}^{T} \log p_\theta(x_t \mid x_1, \ldots, x_{t-1}).
$$

Each term asks: given all tokens before position $t$, how much probability does the model assign to the actual next token $x_t$? Summing over all positions and all sequences in the training corpus, the optimizer adjusts $\theta$ (every learnable parameter in the transformer) to make these probabilities as large as possible.

The training data is typically a massive collection of web text, books, code, and other sources (hundreds of billions to trillions of tokens). Through this objective, the model learns statistical patterns of language, factual knowledge, reasoning patterns, and coding ability, all as a byproduct of learning to predict the next token well.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">What is $p_\theta$? Parametric distribution, not a fixed multinomial</span></summary>

<p>$p_\theta$ is a <strong>parametric</strong> conditional distribution. The subscript $\theta$ denotes every learnable parameter in the entire transformer: all $W_Q, W_K, W_V, W_O$ matrices across all heads and layers, all FFN weights, the embedding table, LayerNorm parameters, and the output head.</p>

<p>The output at each position is a <strong>categorical distribution</strong> over the vocabulary of size $V$. Here is how it is constructed:</p>

<ol>
<li>The transformer processes tokens $x_{\leq t}$ through all $L$ blocks, producing $h_t \in \mathbb{R}^{d_{model}}$.</li>
<li>The output head projects to vocab size: $\text{logits}_t = h_t W_{\text{head}} \in \mathbb{R}^{V}$. This gives one raw score per vocabulary token.</li>
<li>Softmax converts logits to probabilities:</li>
</ol>

$$p_\theta(x_{t+1} = w \mid x_{\leq t}) = \frac{e^{\text{logits}_{t,w}}}{\sum_{w'=1}^{V} e^{\text{logits}_{t,w'}}}.$$

<p>For each position, this gives a probability distribution over all $V$ tokens (e.g., 151,936 for Qwen3). The probabilities sum to 1.</p>

<p><strong>This is not a fixed multinomial.</strong> A fixed multinomial would have static probabilities $p_1, \ldots, p_V$ that do not change with input. Here, the probabilities change for every context. $p_\theta(\cdot \mid \text{"The cat"})$ is a completely different distribution from $p_\theta(\cdot \mid \text{"The dog"})$, even though they share the same parameters $\theta$. The transformer is the function that maps each context to a different distribution over the vocabulary.</p>

<p>Training adjusts $\theta$ via gradient descent on the cross-entropy loss so that $p_\theta$ assigns high probability to the actual next token in the training data. Changing $\theta$ changes the distribution at every context simultaneously.</p>

<p><strong>Cross-entropy loss vs maximum likelihood.</strong> The pretraining objective above is written as maximizing log-likelihood. In practice, training frameworks minimize a loss. The two are the same objective with a sign flip:</p>

<ul>
<li>Maximum likelihood: $\max_\theta \sum_t \log p_\theta(x_t \mid x_{&lt;t})$</li>
<li>Cross-entropy loss: $\min_\theta -\sum_t \log p_\theta(x_t \mid x_{&lt;t})$</li>
</ul>

<p>The name "cross-entropy" comes from information theory. For a true distribution $q$ (one-hot: all probability mass on the actual next token $w^*$) and predicted distribution $p_\theta$, the cross-entropy is:</p>

$$H(q, p_\theta) = -\sum_{w=1}^{V} q(w) \log p_\theta(w).$$

<p>Since $q$ is one-hot, only one term survives:</p>

$$H(q, p_\theta) = -\log p_\theta(w^*).$$

<p>This is exactly the negative log-likelihood of the true token. So "cross-entropy loss" and "negative log-likelihood" are the same formula when the target is a single token. Maximizing log-likelihood, minimizing negative log-likelihood, and minimizing cross-entropy all lead to the same gradient update.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Pretraining Data And Training Setup

The training corpus for modern LLMs typically includes web crawls (Common Crawl, filtered and deduplicated), books, Wikipedia, code repositories (GitHub), scientific papers, and curated instruction datasets. Total size ranges from hundreds of billions to trillions of tokens. Data quality matters enormously: filtering out duplicates, low-quality pages, and toxic content is a large engineering effort that often determines model quality more than architecture choices.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Data processing pipeline: chunking, deduplication, quality filtering, and toxicity removal</span></summary>

<p><strong>Chunking long documents.</strong> A book or paper can be hundreds of thousands of tokens, far longer than the model's context window (e.g., 4K or 32K tokens). Long documents are split into chunks for training. Common approaches:</p>

<ul>
<li><strong>Fixed-length chunking.</strong> Split into non-overlapping segments of the context window size (e.g., every 4,096 tokens). Simple but can cut in the middle of a sentence or paragraph, losing local coherence at chunk boundaries.</li>
<li><strong>Document-aware chunking.</strong> Split at natural boundaries (chapter breaks, section headers, paragraph breaks) and then pack segments up to the context length. This preserves local coherence. If a chapter is 10K tokens and the context window is 4K, split at the nearest paragraph break to 4K.</li>
<li><strong>Packing multiple short documents.</strong> Short documents (e.g., tweets, short articles) are concatenated into one training sequence with a special separator token (e.g., <code>&lt;|endoftext|&gt;</code>) between them, up to the context length. This avoids wasting compute on padding. The causal mask does not need modification because each document's tokens are simply earlier context for the next document's tokens; the model learns that the separator resets the topic.</li>
</ul>

<p><strong>Deduplication.</strong> Web crawls contain enormous amounts of duplicate content: the same Wikipedia article appears on dozens of mirror sites, the same news article is syndicated across hundreds of outlets, boilerplate text (cookie notices, navigation menus) appears on every page of a site. Training on duplicates wastes compute and biases the model toward memorizing repeated content. Deduplication operates at multiple levels:</p>

<ul>
<li><strong>Exact deduplication.</strong> Hash each document (e.g., SHA-256 of the full text) and remove exact copies. Fast but misses near-duplicates (same article with a slightly different header).</li>
<li><strong>Near-deduplication with MinHash.</strong> Compute a set of hash signatures (MinHash) for each document based on its n-gram content. Two documents with a high fraction of matching signatures (e.g., Jaccard similarity above 0.8) are considered near-duplicates. One copy is kept, the rest are removed. This catches paraphrased or lightly edited copies. The LLaMA and Qwen technical reports describe using MinHash-based deduplication.</li>
<li><strong>Substring-level deduplication.</strong> Even after document-level deduplication, the same paragraph can appear across many different documents (e.g., a common definition that appears in multiple Wikipedia articles and textbooks). Suffix array-based methods (used in the "Deduplicating Training Data" paper by Lee et al.) find and remove repeated substrings across the entire corpus. This is more expensive but catches fine-grained repetition.</li>
</ul>

<p><strong>Quality filtering.</strong> Raw web crawls (Common Crawl) contain billions of pages, most of which are low quality: spam, SEO content, auto-generated text, navigation menus, error pages, and so on. Quality filtering keeps only pages that resemble "good" text (books, Wikipedia, well-written articles). Common methods:</p>

<ul>
<li><strong>Heuristic rules.</strong> Remove documents that are too short, have too many special characters, have very low or very high word repetition ratios, contain too many URLs, or have a very low ratio of alphabetic characters. For example, a page where more than 30% of lines end with "..." is likely auto-generated spam.</li>
<li><strong>Language detection.</strong> Use a language classifier (e.g., fastText) to keep only documents in the target language(s). This removes garbled text, mixed-language spam, and encoding errors.</li>
<li><strong>Perplexity filtering.</strong> Train a small language model (e.g., a KenLM n-gram model) on a known high-quality corpus (Wikipedia, books). Score each web page by its perplexity under this model. Pages with very high perplexity (the model finds them surprising and incoherent) are likely low quality and are removed. Pages with very low perplexity might be too repetitive or templated. A middle range is kept. This is the approach used in CCNet (the Common Crawl filtering pipeline used by LLaMA).</li>
<li><strong>Classifier-based filtering.</strong> Train a binary classifier (e.g., a small BERT or logistic regression on TF-IDF features) to distinguish "Wikipedia-like" text from random web text. Score each page and keep those above a threshold. GPT-3 used a classifier trained on WebText (Reddit-upvoted links) as positive examples and raw Common Crawl as negative examples.</li>
</ul>

<p><strong>Toxic and harmful content removal.</strong> Even after quality filtering, the corpus can contain hate speech, explicit content, personally identifiable information (PII), and other harmful material. Approaches include:</p>

<ul>
<li><strong>Keyword and regex filters.</strong> Remove documents containing known slurs, explicit terms, or patterns matching PII (phone numbers, email addresses, social security numbers). Fast but high false-positive rate (the word "kill" appears in legitimate contexts) and misses rephrased toxicity.</li>
<li><strong>Toxicity classifiers.</strong> Use a trained classifier (e.g., Perspective API, or a custom model trained on labeled toxic/non-toxic data) to score each document. Remove or downweight documents above a toxicity threshold. More robust than keyword matching but not perfect.</li>
<li><strong>PII scrubbing.</strong> Use named entity recognition and regex patterns to detect and replace names, addresses, phone numbers, and other PII with placeholder tokens. This is especially important for compliance with privacy regulations.</li>
<li><strong>Domain blocklists.</strong> Maintain lists of known harmful, adult, or spam domains and exclude all pages from those domains.</li>
</ul>

<p><strong>Data mixing and weighting.</strong> The final training corpus is a mixture of sources with intentional proportions. A typical mix might be:</p>

<ul>
<li>Web text (filtered Common Crawl): ~60-70%</li>
<li>Code (GitHub): ~5-10%</li>
<li>Books: ~5-10%</li>
<li>Wikipedia: ~3-5%</li>
<li>Scientific papers (ArXiv): ~3-5%</li>
<li>Curated/instruction data: ~1-5%</li>
</ul>

<p>Higher-quality sources (books, Wikipedia) are often upsampled (repeated multiple times) despite being smaller, because their quality-per-token is much higher than raw web text. The mixing ratios are tuned empirically and can significantly affect downstream performance.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

Training hyperparameters for large-scale pretraining follow the patterns from Series 2, but at much larger scale:

- **Optimizer**: AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.95$ (lower than the default 0.999 to adapt faster during long training runs).
- **Learning rate**: warmup for the first 0.1--1% of steps, then cosine decay to ~10% of peak LR.
- **Batch size**: large (millions of tokens per batch), often ramped up during training.
- **Gradient clipping**: max gradient norm $c = 1.0$. If the gradient vector's length exceeds $c$, it is scaled down to length $c$ (direction preserved). See Series 2 for the full formula.
- **Precision**: mixed precision training (BF16 or FP16 for forward/backward, FP32 for optimizer states).
- **Parallelism**: data parallelism + tensor parallelism + pipeline parallelism combined, as discussed in Series 1.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Tokenization: how raw text becomes token IDs</span></summary>

<p>Before any training or inference, raw text must be converted into integer token IDs that the embedding table can look up. This is done by a <strong>tokenizer</strong>, which is trained separately from the model.</p>

<p><strong>Byte-Pair Encoding (BPE)</strong> is the most common tokenization algorithm (used by GPT, LLaMA, Qwen). It works by:</p>

<ol>
<li>Start with individual characters (or bytes) as the initial vocabulary.</li>
<li>Count all adjacent pairs in the training corpus.</li>
<li>Merge the most frequent pair into a new token.</li>
<li>Repeat until the vocabulary reaches the target size (e.g., 32K, 50K, 152K tokens).</li>
</ol>

<p>For example, if "th" appears very frequently, it becomes a single token. Then "the" might be merged next. Common words like "the" become single tokens, while rare words are split into subword pieces: "unforgettable" might become ["un", "forget", "table"] or ["un", "forg", "ettable"] depending on the learned merges.</p>

<p><strong>Why tokenization matters:</strong></p>

<ul>
<li>Different models use different tokenizers. The same text produces different token counts and different token IDs across models. Qwen3 with 152K vocab tokenizes text more efficiently (fewer tokens per sentence) than GPT-2 with 50K vocab.</li>
<li>The vocabulary size $V$ determines the embedding table size ($V \times d_{model}$ parameters) and the output head size.</li>
<li>Tokenization affects context window utilization: more efficient tokenization fits more text into the same context length.</li>
</ul>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Mixed precision training: BF16, FP16, and FP32</span></summary>

<p>Modern LLM training uses mixed precision to reduce memory and increase throughput. The key idea: use lower precision (16-bit) for most computations and higher precision (32-bit) only where it matters.</p>

<p><strong>Three formats:</strong></p>

<ul>
<li><strong>FP32</strong> (32-bit float): full precision. 1 sign bit, 8 exponent bits, 23 mantissa bits. Used for optimizer states ($m_t$, $v_t$ in AdamW) and the master copy of weights.</li>
<li><strong>FP16</strong> (16-bit float): half precision. 1 sign bit, 5 exponent bits, 10 mantissa bits. Smaller range ($\pm 65504$), which can cause overflow during training.</li>
<li><strong>BF16</strong> (bfloat16): 1 sign bit, 8 exponent bits, 7 mantissa bits. Same range as FP32 (8 exponent bits) but lower precision (7 vs 23 mantissa bits). Preferred for LLM training because it avoids overflow issues.</li>
</ul>

<p><strong>The mixed precision recipe:</strong></p>

<ol>
<li>Store a master copy of weights in FP32.</li>
<li>Cast weights to BF16 for the forward and backward pass (half the memory, faster matmuls on modern GPUs).</li>
<li>Compute gradients in BF16.</li>
<li>Update the FP32 master weights using the FP32 optimizer states.</li>
</ol>

<p>This gives nearly the same training dynamics as full FP32, at roughly half the memory for model weights and significantly faster computation. The FP32 optimizer states (AdamW stores $m_t$ and $v_t$ per parameter) are the dominant memory cost.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## From Pretraining to Task Adaptation

A pretrained LLM can predict next tokens, but it does not follow instructions, answer questions helpfully, or avoid harmful outputs. Task adaptation bridges this gap. The common pipeline has three stages:

**1. Continued pretraining (optional).** Further train the base model on domain-specific text (e.g., medical literature, legal documents, a company's internal docs). This uses the same next-token prediction objective but on a targeted corpus. Useful when the target domain was underrepresented in the original pretraining data.

**2. Supervised fine-tuning (SFT).** Train the model on (instruction, response) pairs so it learns to follow instructions and produce helpful answers. The training objective is still next-token prediction, but only on the response tokens (the instruction tokens are provided as context but not included in the loss). This is covered in detail in the SFT series.

**3. Preference/RL alignment.** Use human preference data (which response is better?) to further align the model with human values: helpfulness, harmlessness, honesty. Methods include RLHF (reinforcement learning from human feedback) and DPO (direct preference optimization). Covered in the RL series.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why pretraining alone is not enough</span></summary>

<p>A pretrained model is trained to predict the next token in web text. If you give it the prompt "What is the capital of France?", it might continue with "What is the capital of Germany? What is the capital of Italy?" because in its training data, quiz questions often appear in lists. It is completing the <em>document</em>, not answering the <em>question</em>.</p>

<p>SFT teaches the model that when it sees a question, the appropriate continuation is an answer, not more questions. RL alignment then refines the style, safety, and helpfulness of those answers based on human preferences.</p>

<p>This three-stage pipeline (pretrain → SFT → alignment) is the standard recipe used by GPT-4, Claude, LLaMA-Chat, Qwen-Chat, and most production LLMs.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Combining multiple domains: multi-task mixing, model merging, LoRA, and Mixture of Experts</span></summary>

<p>When you need a model that handles multiple domains (e.g., medical <em>and</em> legal), there are several approaches to combine fine-tuned capabilities without losing either.</p>

<p><strong>Approach 1: multi-task fine-tuning.</strong> Mix all domain data together in one SFT stage, then one RL stage. You control the domain balance through data mixing ratios (e.g., 50% medical, 50% legal). The model learns both domains jointly and there is no forgetting. This is the simplest and most common approach when you have access to all data at once.</p>

<p><strong>Approach 2: sequential fine-tuning.</strong> Fine-tune one domain after another: base → SFT medical → SFT legal → RL. The problem is <strong>catastrophic forgetting</strong>: when you fine-tune on legal data, gradient updates overwrite the medical-specialized weights, and medical performance degrades. You can mitigate this by mixing in some medical data during the legal stage (experience replay), but the balance is fragile to tune.</p>

<p><strong>Approach 3: model merging (weight averaging).</strong> Fine-tune separate models from the same base, then merge their weights arithmetically:</p>

$$W_{\text{merged}} = \alpha \cdot W_{\text{medical}} + (1 - \alpha) \cdot W_{\text{legal}}.$$

<p>This works surprisingly well when both models started from the same base, because the fine-tuned weights remain in a similar region of parameter space. More advanced merging methods include TIES-Merging (only merge parameters that changed significantly, resolve sign conflicts), DARE (randomly drop small weight deltas before merging to reduce interference), and SLERP (spherical interpolation on the parameter hypersphere). Model merging is cheap (no training, just arithmetic) and popular in the open-source community. The limitation is that it is heuristic: there is no guarantee the merged model is optimal.</p>

<p><strong>Approach 4: LoRA adapters.</strong> Train lightweight low-rank adapters per domain on a shared frozen base. Each adapter modifies only ~0.1% of parameters. At inference, load the appropriate adapter for the detected domain, or merge multiple adapters into the base weights: $W = W_{\text{base}} + \Delta W_{\text{medical}} + \Delta W_{\text{legal}}$. This avoids catastrophic forgetting because the base is frozen. Each adapter is small and fast to train. The downside is that independently trained adapters may interfere when combined.</p>

<p><strong>Approach 5: Mixture of Experts (MoE).</strong> MoE is the most principled architectural approach. Rather than merging after the fact, the model is designed from the start with multiple specialist sub-networks (experts) and a learned router that selects which experts to activate for each input.</p>

<p><strong>How MoE works inside a transformer block.</strong> In a standard transformer, the FFN is a single two-layer MLP applied to every token. In an MoE transformer, the FFN is replaced by $E$ parallel expert networks (each is a separate FFN with its own weights) plus a gating router:</p>

$$\text{MoE}(x) = \sum_{i=1}^{E} g_i(x) \cdot \text{FFN}_i(x),$$

<p>where $g_i(x)$ is the gating weight for expert $i$, computed by the router:</p>

$$g(x) = \text{TopK}\!\left(\text{softmax}(x W_{\text{gate}})\right).$$

<p>The router is a simple linear layer $W_{\text{gate}} \in \mathbb{R}^{d_{model} \times E}$ that produces a score for each expert. TopK selects only the top $k$ experts (e.g., $k = 2$ out of $E = 64$) and zeros out the rest. Only the selected experts run their FFN computation, so the actual compute per token is much smaller than having all $E$ experts active.</p>

<p><strong>Concrete example: Qwen3-235B-A22B.</strong> This model has 235B total parameters but only 22B activated per token. Each transformer block has 128 expert FFNs, and the router selects 8 of them for each token. Different tokens in the same sequence can activate different experts. A medical question might route to experts that specialize in scientific reasoning, while a legal question in the same batch might route to different experts that specialize in formal language and citation patterns.</p>

<p><strong>Why MoE helps with multi-domain capability.</strong> In a dense model, every parameter is used for every token. Specializing in medical language and legal language must share the same FFN weights, creating tension. In an MoE model, different experts can specialize in different domains or capabilities. The router learns to direct medical tokens to medical-specialized experts and legal tokens to legal-specialized experts, without interference. This specialization emerges naturally during training without explicit domain labels.</p>

<p><strong>The load balancing problem.</strong> Without constraints, the router might learn to send all tokens to the same few experts (winner-take-all), leaving most experts unused. This wastes parameters and defeats the purpose. MoE training adds an auxiliary loss that penalizes imbalanced routing:</p>

$$\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} f_i \cdot p_i,$$

<p>where $f_i$ is the fraction of tokens routed to expert $i$ and $p_i$ is the average router probability assigned to expert $i$. This loss is minimized when all experts receive equal traffic. It is weighted by a small coefficient (e.g., 0.01) and added to the main next-token prediction loss.</p>

<p><strong>Trade-offs of MoE.</strong></p>

<ul>
<li><strong>Parameter efficiency.</strong> MoE can scale total parameters (and therefore knowledge capacity) without proportionally scaling compute per token. Qwen3-235B has 10x the parameters of Qwen3-32B but only ~4x the active parameters per token.</li>
<li><strong>Memory cost.</strong> All expert weights must be stored in memory even though only a few are active per token. Qwen3-235B needs enough GPU memory to hold 235B parameters, not just 22B.</li>
<li><strong>Communication overhead.</strong> In distributed training, different experts may live on different GPUs. The router must send tokens to the right GPU, which adds communication cost (all-to-all communication).</li>
<li><strong>Training stability.</strong> Router training can be unstable early on. The load balancing loss, careful initialization of the gate, and sometimes expert dropout are needed to stabilize training.</li>
</ul>

<p><strong>Is MoE the dominant approach?</strong> MoE is increasingly adopted for large-scale models (Mixtral, Qwen3-235B, GPT-4 is widely believed to be MoE, DeepSeek-V2/V3). It is the most principled way to scale knowledge capacity while keeping inference cost manageable. However, for smaller models or when adding a new domain to an existing dense model, multi-task mixing or LoRA adapters are more practical because they do not require changing the architecture. MoE is best thought of as an architectural choice made before pretraining, not a post-hoc merging technique.</p>

<p><strong>Summary.</strong></p>

<table>
<thead><tr><th>Approach</th><th>Training cost</th><th>Forgetting risk</th><th>Quality</th><th>Complexity</th></tr></thead>
<tbody>
<tr><td>Multi-task mix</td><td>One SFT + one RL</td><td>None</td><td>High</td><td>Low</td></tr>
<tr><td>Sequential</td><td>Multiple stages</td><td>High</td><td>Variable</td><td>Low</td></tr>
<tr><td>Model merging</td><td>Separate fine-tunes</td><td>None</td><td>Good</td><td>Low</td></tr>
<tr><td>LoRA adapters</td><td>Small per-domain</td><td>None</td><td>Good</td><td>Medium</td></tr>
<tr><td>MoE</td><td>Full pretraining</td><td>None</td><td>Highest</td><td>High</td></tr>
</tbody>
</table>

<p>For most cases: multi-task mixing if you can combine the data upfront, model merging or LoRA if you need to add domains to an existing model without retraining, and MoE if you are designing a large-scale system from scratch where multi-domain capacity and inference efficiency both matter.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Inference: How Text Generation Works

At inference, a decoder-only LLM generates text one token at a time. Starting from a prompt, the model:

1. Runs a forward pass through all $L$ transformer blocks.
2. Takes the logits at the **last** position: $\text{logits} = h_T W_{\text{head}} \in \mathbb{R}^V$.
3. Applies a **sampling strategy** to select the next token.
4. Appends the new token to the sequence and repeats from step 1.

The sampling strategy controls the trade-off between quality (coherence, correctness) and diversity (creativity, variety).

### Greedy Decoding

Always pick the token with the highest probability:

$$x_{t+1} = \arg\max_w \; p_\theta(w \mid x_{\leq t}).$$

This is deterministic: the same prompt always produces the same output. It tends to produce safe, repetitive text because it always takes the single most likely path.

### Temperature Sampling

Before sampling, divide the logits by a temperature parameter $\tau > 0$:

$$p_\tau(w) = \frac{e^{\text{logits}_w / \tau}}{\sum_{w'} e^{\text{logits}_{w'} / \tau}}.$$

Temperature controls the sharpness of the distribution:

- $\tau = 1.0$: the original distribution (no change).
- $\tau < 1.0$ (e.g., 0.3): sharpens the distribution. High-probability tokens get even higher probability, low-probability tokens get pushed closer to zero. Output is more focused and deterministic.
- $\tau > 1.0$ (e.g., 1.5): flattens the distribution. Probabilities become more uniform. Output is more random and creative.
- $\tau \to 0$: converges to greedy decoding.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Why temperature works: the effect on softmax</span></summary>

<p>Consider three tokens with logits $[3.0, 1.0, 0.5]$.</p>

<p>At $\tau = 1.0$: softmax gives approximately $[0.78, 0.11, 0.06]$. Token 1 dominates.</p>

<p>At $\tau = 0.5$: logits become $[6.0, 2.0, 1.0]$. Softmax gives approximately $[0.97, 0.02, 0.01]$. Token 1 almost certain.</p>

<p>At $\tau = 2.0$: logits become $[1.5, 0.5, 0.25]$. Softmax gives approximately $[0.49, 0.18, 0.14]$. Much more uniform.</p>

<p>Dividing by $\tau$ scales the logit <em>differences</em>. Small $\tau$ amplifies differences (sharp distribution), large $\tau$ compresses differences (flat distribution). The ranking of tokens does not change; only the confidence does.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

### Top-k Sampling

After computing probabilities, keep only the $k$ highest-probability tokens and zero out the rest. Renormalize so probabilities sum to 1, then sample.

For example, with $k = 3$ and probabilities $[0.4, 0.25, 0.15, 0.1, 0.05, 0.05]$:

1. Keep top 3: $[0.4, 0.25, 0.15, 0, 0, 0]$.
2. Renormalize: $[0.50, 0.31, 0.19, 0, 0, 0]$.
3. Sample from these three tokens.

This prevents the model from ever picking very unlikely tokens (which can cause incoherent output) while still allowing diversity among the top candidates.

### Top-p (Nucleus) Sampling

Instead of a fixed $k$, keep the smallest set of tokens whose cumulative probability exceeds a threshold $p$ (e.g., $p = 0.9$). This adapts to the shape of the distribution:

- When the model is confident (one token has probability 0.95), top-p with $p = 0.9$ keeps just that one token.
- When the model is uncertain (many tokens around 0.05--0.1), top-p keeps many candidates.

This is more flexible than top-k, which always keeps exactly $k$ tokens regardless of whether the model is confident or uncertain.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Top-k vs top-p: when each is better</span></summary>

<p><strong>Top-k's problem:</strong> $k$ is fixed regardless of context. At some positions the model is very confident (one token has 0.9 probability), but top-k still samples from $k$ tokens, introducing unnecessary noise. At other positions many tokens are plausible, but $k$ might cut off good candidates.</p>

<p><strong>Top-p adapts:</strong> it includes however many tokens are needed to cover probability mass $p$. Confident positions use fewer tokens, uncertain positions use more.</p>

<p><strong>In practice</strong>, top-p ($p = 0.9$ or $0.95$) is the more common default in production LLMs. Many systems combine temperature + top-p: first adjust sharpness with temperature, then truncate with top-p. Some also combine top-k and top-p together (apply both filters).</p>

<p>Typical settings for different use cases:</p>

<ul>
<li><strong>Code generation:</strong> low temperature (0.2--0.4), low top-p (0.9). Correctness matters more than creativity.</li>
<li><strong>Creative writing:</strong> higher temperature (0.7--1.0), higher top-p (0.95). Diversity and surprise are desirable.</li>
<li><strong>Factual Q&A:</strong> temperature close to 0 or greedy. Minimize hallucination.</li>
</ul>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Context Window And KV Cache

### Context Window

The context window is the maximum number of tokens the model can process in a single forward pass. It is determined by the positional encoding and the memory required for the $(T \times T)$ attention matrix.

| Model | Context window |
|---|---|
| GPT-2 | 1,024 |
| GPT-3 | 2,048 |
| GPT-4 | 8,192 / 128K |
| LLaMA 2 | 4,096 |
| Qwen3-8B | 32,768 |
| Claude | 200K |

If the input exceeds the context window, the model cannot attend to tokens beyond it. Longer context windows require more memory (the attention matrix grows as $T^2$) and more compute.

### KV Cache

During autoregressive generation, the model produces tokens one at a time. At each step, it needs the key and value vectors of all previous tokens to compute attention. Without optimization, every new token would require recomputing Q, K, V for the entire sequence from scratch.

The KV cache stores the key and value vectors from all previous steps. When generating token $t+1$:

1. Compute Q, K, V only for the new token (one position).
2. Append the new K and V to the cache.
3. Compute attention: the new Q attends to all cached K/V vectors.

This reduces each generation step from $O(T \cdot d_{model})$ recomputation to $O(d_{model})$ for the new token, plus the attention over the cached keys.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">KV cache: step-by-step example and memory cost</span></summary>

<p><strong>Step-by-step.</strong> Generating the sequence "The cat sat":</p>

<p><strong>Step 1: process prompt "The".</strong></p>

<ul>
<li>Compute $q_0, k_0, v_0$ for "The".</li>
<li>Cache: $K = [k_0]$, $V = [v_0]$.</li>
<li>Attention: $q_0$ attends to $[k_0]$. Output predicts "cat".</li>
</ul>

<p><strong>Step 2: generate "cat".</strong></p>

<ul>
<li>Compute $q_1, k_1, v_1$ for "cat" only (one token, not the full sequence).</li>
<li>Append to cache: $K = [k_0, k_1]$, $V = [v_0, v_1]$.</li>
<li>Attention: $q_1$ attends to $[k_0, k_1]$. Output predicts "sat".</li>
</ul>

<p><strong>Step 3: generate "sat".</strong></p>

<ul>
<li>Compute $q_2, k_2, v_2$ for "sat" only.</li>
<li>Append to cache: $K = [k_0, k_1, k_2]$, $V = [v_0, v_1, v_2]$.</li>
<li>Attention: $q_2$ attends to $[k_0, k_1, k_2]$. Output predicts the next token.</li>
</ul>

<p>At each step, only one new token is processed through the Q, K, V projections and FFN. The expensive part (projecting and transforming all previous tokens) is not repeated.</p>

<p><strong>Memory cost.</strong> The KV cache stores $K$ and $V$ for every layer and every head. For a model with $L$ layers, $H_{kv}$ KV heads, per-head dimension $d_k$, and sequence length $T$:</p>

$$\text{KV cache size} = 2 \times L \times H_{kv} \times T \times d_k \times \text{bytes per element}.$$

<p>For Qwen3-8B ($L = 36$, $H_{kv} = 8$ with GQA, $d_k = 128$) at $T = 32{,}768$ in BF16 (2 bytes):</p>

$$2 \times 36 \times 8 \times 32{,}768 \times 128 \times 2 \approx 4.8 \text{ GB}.$$

<p>This is per sequence. Serving many concurrent users requires multiplying by the number of active sequences, which is why KV cache memory is often the bottleneck for inference serving.</p>

<p><strong>GQA reduces KV cache.</strong> Grouped-query attention (discussed in Series 3) shares K and V across groups of query heads. Qwen3-8B has 32 query heads but only 8 KV heads, reducing the cache by 4x compared to full multi-head attention.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Why Scaling Works

Empirically, LLM performance improves predictably with three factors:

1. **More data**: more tokens in the training corpus.
2. **Larger models**: more parameters ($d_{model}$, layers, heads).
3. **More compute**: more GPU-hours of training.

Kaplan et al. (2020) and Hoffmann et al. (2022, "Chinchilla") showed that these follow power-law scaling laws: loss decreases as a smooth function of compute, data, and model size. The Chinchilla result showed that many early LLMs were undertrained: for a given compute budget, training a smaller model on more data often outperforms training a larger model on less data.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Scaling laws and compute-optimal training</span></summary>

<p><strong>The scaling law.</strong> The cross-entropy loss $L$ on held-out text follows approximately:</p>

$$L(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty,$$

<p>where $N$ is the number of parameters, $D$ is the number of training tokens, and $L_\infty$ is an irreducible loss (entropy of natural language). The exponents $\alpha_N \approx 0.076$ and $\alpha_D \approx 0.095$ mean that both axes give diminishing returns: doubling model size or data does not halve the loss.</p>

<p><strong>Compute-optimal training (Chinchilla).</strong> For a fixed compute budget $C \propto N \times D$ (FLOPs $\approx 6ND$ for a forward + backward pass), the optimal allocation is roughly:</p>

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}.$$

<p>Model size and data should scale at the same rate. The original GPT-3 (175B parameters, 300B tokens) was undertrained by this criterion. Chinchilla (70B parameters, 1.4T tokens) matched GPT-3 performance with less compute by using a better ratio.</p>

<p>Modern LLMs (LLaMA 3, Qwen3) train well beyond the Chinchilla-optimal point, because inference cost depends on model size but not training data. A smaller model trained on more data is cheaper to deploy even if training costs more.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

Efficiency techniques are essential at this scale:

- **Mixed precision** (BF16/FP16): cuts memory roughly in half and speeds up matmuls on modern GPUs.
- **Parallelism**: data + tensor + pipeline parallelism combined (see Series 1).
- **FSDP**: shards optimizer states and parameters across GPUs (see Series 1).
- **Gradient checkpointing**: trades compute for memory by recomputing activations during backward instead of storing them all.
- **Flash Attention**: fused CUDA kernel that computes attention without materializing the full $(T \times T)$ matrix, reducing memory from $O(T^2)$ to $O(T)$.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">How Flash Attention works: tiling, online softmax, and GPU memory hierarchy</span></summary>

<p><strong>The problem with standard attention.</strong> Standard attention computes the full $T \times T$ score matrix, writes it to GPU main memory (HBM), applies softmax, writes the result back, then multiplies by $V$. For $T = 32{,}768$, this matrix has about 1 billion entries (~2 GB in BF16 per head per layer). The computation is not limited by arithmetic (GPUs have plenty of FLOPS); it is limited by <strong>memory bandwidth</strong>, the time spent reading and writing this huge intermediate matrix.</p>

<p><strong>GPU memory hierarchy.</strong> A GPU has two levels of memory:</p>

<ul>
<li><strong>SRAM (on-chip):</strong> ~20 MB, ~19 TB/s bandwidth. Very fast, very small.</li>
<li><strong>HBM (off-chip):</strong> ~80 GB, ~2 TB/s bandwidth. Large, but ~10x slower.</li>
</ul>

<p>Standard attention writes the full $T \times T$ matrix to HBM and reads it back multiple times. The GPU spends most of its time waiting for data transfers, not doing math.</p>

<p><strong>Flash Attention's solution: tiling.</strong> Instead of computing the full $T \times T$ matrix at once, Flash Attention processes Q, K, V in small blocks (tiles) that fit entirely in SRAM. It never writes the full attention matrix to HBM:</p>

<pre><code>Standard:  Q, K → [full T×T matrix in HBM] → softmax → [full T×T in HBM] → ×V → output

Flash:     for each block of Q rows:
             for each block of K, V columns:
               compute scores (in SRAM)
               update running softmax (in SRAM)
               accumulate output (in SRAM)
             write final output block to HBM
</code></pre>

<p><strong>The hard part: incremental softmax.</strong> Softmax needs the entire row to normalize: $\text{softmax}(s_j) = e^{s_j} / \sum_{j'} e^{s_{j'}}$. If you process K in blocks, you do not have all $T$ scores at once. Flash Attention uses the <strong>online softmax trick</strong>: maintain a running maximum $m$ and running sum $\ell$ as each block is processed, and rescale partial results when the maximum changes.</p>

<p>For each new block $b$:</p>

<ol>
<li>Compute local scores: $s^{(b)} = q \cdot K_b^\top / \sqrt{d_k}$.</li>
<li>Update running max: $m_{\text{new}} = \max(m_{\text{old}}, \max(s^{(b)}))$.</li>
<li>Rescale old accumulator: $O \leftarrow O \cdot e^{m_{\text{old}} - m_{\text{new}}}$.</li>
<li>Accumulate: $O \leftarrow O + e^{s^{(b)} - m_{\text{new}}} \cdot V_b$.</li>
<li>Update running sum: $\ell \leftarrow \ell \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum e^{s^{(b)} - m_{\text{new}}}$.</li>
</ol>

<p>After all blocks: $O \leftarrow O / \ell$. The rescaling in step 3 is the key: when a new block has larger scores than all previous blocks, the old partial results are scaled down to match. The final result is <strong>mathematically identical</strong> to computing softmax over the full row.</p>

<p><strong>What you gain:</strong></p>

<table>
<thead><tr><th></th><th>Standard attention</th><th>Flash Attention</th></tr></thead>
<tbody>
<tr><td>Memory</td><td>$O(T^2)$ for score matrix</td><td>$O(T)$, only tiles in SRAM</td></tr>
<tr><td>HBM reads/writes</td><td>3 full $T \times T$ matrices</td><td>Only Q, K, V, and output</td></tr>
<tr><td>Speed</td><td>Bandwidth-bound</td><td>2-4x faster in practice</td></tr>
<tr><td>Output</td><td>Exact</td><td>Exact (not an approximation)</td></tr>
</tbody>
</table>

<p><strong>Flash Attention 2 and 3.</strong> Flash Attention 2 improved parallelism by distributing work across GPU thread blocks more efficiently (parallelize over the sequence length dimension, not just batch and head). Flash Attention 3 further optimizes for newer GPU architectures (Hopper). All modern LLM training and inference frameworks (PyTorch 2.0+, vLLM, TensorRT-LLM) use Flash Attention by default.</p>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Common Beginner Mistakes

- **Treating decoding hyperparameters as minor details.** Temperature, top-p, and top-k dramatically affect output quality. A model that seems incoherent at $\tau = 1.2$ might be excellent at $\tau = 0.6$.
- **Ignoring tokenization differences between models.** The same text produces different token counts across models. "Unforgettable" might be 1 token in one model and 3 in another. This affects context window utilization and cost.
- **Comparing models without fixed prompts/seeds/settings.** Sampling introduces randomness. Without fixed seeds and identical decoding parameters, differences between runs can be larger than differences between models.
- **Confusing context window with knowledge.** A 128K context window means the model can *attend to* 128K tokens in one pass, not that it *remembers* 128K tokens across conversations. Each conversation starts fresh (unless context is explicitly provided).

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Context window vs knowledge: how to make the model "remember" things</span></summary>

<p><strong>LLMs have no persistent memory across conversations.</strong> When you start a new conversation, the model knows nothing about previous conversations. Its only information comes from (1) what it learned during pretraining and fine-tuning (baked into the weights $\theta$), and (2) what you provide in the current context window.</p>

<p><strong>The system prompt.</strong> Most chat APIs allow a system prompt (or system message) that is prepended to every conversation. This is the primary way to give the model persistent instructions or knowledge within a session:</p>

<pre><code>System: You are a medical assistant specializing in cardiology. 
        Always cite clinical guidelines when answering.
        The patient's history: [... relevant details ...]

User:   What are the treatment options for atrial fibrillation?
</code></pre>

<p>The system prompt is just regular tokens in the context window. It has no special mechanism; it works because the model attends to it on every turn. If the system prompt is 2,000 tokens and the context window is 32K, you have 30K tokens left for conversation.</p>

<p><strong>Retrieval-Augmented Generation (RAG).</strong> For knowledge that does not fit in the system prompt or changes frequently, RAG retrieves relevant documents from an external database and inserts them into the prompt before the model generates:</p>

<ol>
<li>User asks a question.</li>
<li>A retrieval system (embedding similarity search, keyword search, etc.) finds the most relevant documents from a knowledge base.</li>
<li>The retrieved documents are inserted into the prompt as context.</li>
<li>The model generates an answer grounded in the provided documents.</li>
</ol>

<p>This is how most production systems handle large or dynamic knowledge bases (company docs, product catalogs, legal databases). The model does not "remember" the documents; they are provided fresh in each query's context window.</p>

<p><strong>Fine-tuning.</strong> To permanently bake domain knowledge into the model's weights, fine-tune on domain-specific data (continued pretraining or SFT). This is more expensive than RAG but makes the knowledge available without consuming context window space. The trade-off: fine-tuning is slow to update (retrain for new information), while RAG can be updated instantly by changing the document database.</p>

<p><strong>Practical guidelines:</strong></p>

<ul>
<li>For <strong>behavioral instructions</strong> (tone, format, persona): use the system prompt.</li>
<li>For <strong>specific facts or documents</strong> the model needs per query: use RAG.</li>
<li>For <strong>broad domain competence</strong> (the model should "think like a doctor"): fine-tune.</li>
<li>For <strong>conversation history</strong>: include previous turns in the context. When the conversation exceeds the context window, older turns must be truncated or summarized.</li>
</ul>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

- **Underestimating the importance of the prompt.** The same model can give vastly different quality answers depending on how the question is phrased. Prompt engineering is not a hack; it is a core skill for using LLMs effectively.

<details>
<summary><span style="color: saddlebrown; font-style: italic;">Prompt engineering: practical rules of thumb by task type</span></summary>

<p>The prompt is the only input the model sees. Small changes in phrasing can cause large changes in output quality. Some practical guidelines:</p>

<p><strong>General principles:</strong></p>

<ul>
<li><strong>Be specific.</strong> "Summarize this article" is vague. "Summarize this article in 3 bullet points, each under 20 words, focusing on the methodology" tells the model exactly what you want.</li>
<li><strong>Provide examples (few-shot).</strong> Showing 2--3 examples of the desired input-output format is often more effective than describing the format in words. The model pattern-matches from examples.</li>
<li><strong>Specify the output format.</strong> If you want JSON, say "respond in valid JSON with keys: name, age, diagnosis." If you want a table, show the header row. Ambiguous format instructions produce ambiguous output.</li>
<li><strong>Assign a role.</strong> "You are an experienced cardiologist reviewing a case" activates different knowledge patterns than "Answer this medical question." Roles prime the model toward domain-appropriate language and reasoning depth.</li>
</ul>

<p><strong>Task-specific guidelines:</strong></p>

<ul>
<li><strong>Factual Q&A.</strong> Ask the model to cite sources or say "I don't know" if uncertain. Include "Answer based only on the following context: [...]" to reduce hallucination. Use low temperature.</li>
<li><strong>Code generation.</strong> Specify the language, framework, and version. Provide function signatures or test cases as constraints. Ask the model to think step-by-step before writing code.</li>
<li><strong>Summarization.</strong> Specify length, audience, and focus. "Summarize for a technical audience in 100 words" vs "Explain to a 10-year-old in 2 sentences" produce very different outputs from the same input.</li>
<li><strong>Analysis and reasoning.</strong> Use chain-of-thought prompting: "Think step by step before giving your final answer." This significantly improves accuracy on math, logic, and multi-step reasoning tasks.</li>
<li><strong>Creative writing.</strong> Provide constraints (genre, tone, length, audience) rather than leaving it open-ended. "Write a noir detective monologue, 200 words, first person" gives better results than "Write something creative."</li>
<li><strong>Classification and extraction.</strong> Define the categories explicitly and provide one example per category. For extraction, specify the exact fields you want and their types.</li>
</ul>

<p><strong>Common prompt mistakes:</strong></p>

<ul>
<li><strong>Too vague:</strong> "Make this better." Better how? More concise? More formal? More accurate?</li>
<li><strong>Contradictory instructions:</strong> "Be concise and thorough." Pick one priority.</li>
<li><strong>Assuming prior context:</strong> "As I said earlier..." in a new conversation. The model has no memory of earlier conversations.</li>
<li><strong>Over-constraining:</strong> 20 instructions in the system prompt can cause the model to ignore some. Prioritize the most important 3--5 constraints.</li>
</ul>

<p style="margin-top: 0.9em; padding-top: 0.45em; border-top: 1px dashed #c9b39a; font-size: 0.92em; color: #8b5a2b;"><em>End of expanded note.</em></p>
</details>

## Next Post

We now move to supervised fine-tuning (SFT): how to build instruction-following models with high-quality data.
