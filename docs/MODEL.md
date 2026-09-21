# Model Architecture & Mathematical Formulation (`docs/MODEL.md`)

*Authoritative Source: [`src/model/gpt.py`](../src/model/gpt.py), [`src/model/config.py`](../src/model/config.py), [`configs/model.gpu.yaml`](../configs/model.gpu.yaml)*

---

## 1. Overview & Hyperparameters

The core `llm-engine` architecture is a causal decoder-only Transformer optimized for low-memory execution.

### Architectural Parameters (Active GPU Configuration):
- **Base Parameters**: 81,314,304 (~81.3M) at 40,000 vocab.
- **Extended Parameters**: 82,338,304 (~82.3M) at 42,000 vocab.
- **Layers ($L$)**: 16
- **Hidden Dimension ($d_{\text{model}}$)**: 512
- **Attention Heads ($H_q$)**: 8 (head dimension $d_k = 512 / 8 = 64$)
- **KV Heads ($H_{kv}$)**: 2 (Grouped-Query Attention with 4:1 query-to-KV ratio)
- **Intermediate FFN Dimension ($d_{\text{ffn}}$)**: 2048 (`ffn_multiple_of: 128`)
- **Activation**: SwiGLU
- **Normalization**: Pre-Norm Root Mean Square Normalization (RMSNorm) with $\epsilon = 10^{-5}$
- **Positional Encoding**: Rotary Position Embeddings (RoPE) with $\text{base} = 10000.0, \text{scale} = 1.0$
- **Weight Tying**: Input token embedding weights are shared with the final LM head projection matrix
- **Context Length**: Configured capacity 1024 tokens; pretraining sequences 512 tokens

---

## 2. Mathematical Formulations

### 2.1 RMSNorm (Root Mean Square Normalization)
Unlike standard LayerNorm, RMSNorm enforces scale invariance without estimating or centering around the mini-batch mean, reducing computational overhead:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

where $\gamma \in \mathbb{R}^d$ is a learnable scaling vector, and $\epsilon = 10^{-5}$. No additive bias term is used.

### 2.2 Grouped-Query Attention (GQA) & RoPE
Let $x$ be the normalized input tensor. Query, Key, and Value projections are computed as:

$$Q = x W_Q \in \mathbb{R}^{B \times H_q \times T \times d_k}$$

$$K = x W_K \in \mathbb{R}^{B \times H_{kv} \times T \times d_k}$$

$$V = x W_V \in \mathbb{R}^{B \times H_{kv} \times T \times d_k}$$

Rotary Positional Embeddings (RoPE) rotate pairs of coordinate elements in $Q$ and $K$ by frequency-dependent angles:

$$R_{\Theta, m}^d = \text{diag}\left(R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m}\right)$$

where $\theta_i = b^{-2(i-1)/d}$ with base $b = 10000.0$.

Before computing attention, each of the $H_{kv}=2$ Key and Value head tensors is repeated 4 times along the head dimension to match the $H_q=8$ Query heads. The causal scaled dot-product attention is then:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

where $M_{i,j} = -\infty$ for $j > i$ (causal mask).

### 2.3 SwiGLU Feed-Forward Network
The feed-forward network uses a gated linear unit with the SiLU (Swish) activation function:

$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$

where:
- $W_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$
- $W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$
- $W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$

---

## 3. Weight Tying & Vocabulary Extension Rules

The language modeling head shares its weight tensor directly with the token embedding table:

$$\text{LM\_Head}(h) = h W_{\text{emb}}^T \quad \text{where } W_{\text{emb}} \in \mathbb{R}^{V \times d_{\text{model}}}$$

### Append-Only Vocabulary Extension Rule:
When expanding the vocabulary from 40K to 42K:
1. The new tokens MUST be appended strictly to the end of the embedding tensor (`new_indices >= 40000`).
2. Existing token indices (0 to 39,999) must never change meaning or position.
3. Checkpoint loading must detect append-only expansion and resize the weight tensor, copying existing weights and initializing the new rows with normal distribution ($\sigma = 0.02$).
4. Verified by [`tests/test_vocabulary_compatibility.py`](../tests/test_vocabulary_compatibility.py).

---

## 4. Checkpoint State-Dict Contract

An atomic PyTorch checkpoint (`checkpoints/*/*.pt`) contains:
- `model_state_dict`: Ordered mapping of all weight tensors.
- `optimizer_state_dict`: AdamW momentum and variance buffers.
- `scheduler_state_dict`: Learning rate step counter and schedule state.
- `ema_state_dict`: Shadow weights for exponential moving average.
- `scaler_state_dict`: PyTorch AMP GradScaler scale factor.
- `epoch`, `global_step`, `best_metric`: Training progress metadata.
- `config`: Complete copy of the model and training configuration.
- `rng_state`: PyTorch, Python, and NumPy random number generator states for deterministic training resumption.


---

## 5. Opt-In Hybrid Attention and Sparse MoE Research Profiles

The default active architecture remains the dense-GQA 16-layer model above. An
optional layer cycle can now be supplied through `attention_layer_pattern`.
For example, [`configs/model.hybrid.gpu.yaml`](../configs/model.hybrid.gpu.yaml)
uses:

```yaml
attention_layer_pattern: [linear, linear, linear, dense]
```

`CausalLinearAttention` preserves the ordinary Q/K/V/output projection tensor
shapes but replaces softmax attention with an `ELU(x)+1` feature map. Prefill is
processed in bounded chunks, and autoregressive decoding stores recurrent
`key_sum` and `key_value_sum` state whose size does not grow with context.
This is a **reference research implementation**, not an implementation of the
architecture-specific Qwen linear-convolution layer.

The optional `sliding_window` attention pattern applies the same configured
local window during both full-sequence prefill and one-token KV-cache decoding.
This preserves the model's receptive field rather than allowing cached decoding
to see an unintended full prefix.

Sparse MoE layers expose a Switch-style router balance signal. The trainer only
adds it when `moe_aux_loss_weight` is non-zero, so existing dense and MoE
checkpoints retain their previous inference behavior. The trainer also records
router entropy/load diagnostics and the weighted auxiliary loss.

A planning-only 7B-class hybrid-MoE profile is provided at
[`configs/scaling/model.hybrid-moe-7b.yaml`](../configs/scaling/model.hybrid-moe-7b.yaml).
It can be sized and planned without allocating the weights; it is not a bundled
trained model.
