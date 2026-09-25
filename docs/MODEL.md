# Model Architecture & Mathematical Formulation (`docs/MODEL.md`)

*Authoritative Source: [`src/model/gpt.py`](../src/model/gpt.py), [`src/model/config.py`](../src/model/config.py), [`configs/model.gpu.yaml`](../configs/model.gpu.yaml)*

*Media-generation source: [`src/audio_generation/model.py`](../src/audio_generation/model.py), [`src/audio_generation/pipeline.py`](../src/audio_generation/pipeline.py), [`src/video_generation/model.py`](../src/video_generation/model.py), [`src/video_generation/pipeline.py`](../src/video_generation/pipeline.py)*

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

---

## 6. Image, Audio, and Video Generation Models

The image, audio, and video systems are separate checkpoint families from the
causal language model. The image stack provides pixel-space DDPM/DDIM and a
native latent-diffusion research path; audio and video use tokenizer-backed,
trainable text conditioning. Media checkpoints are never interchangeable with
language-model checkpoints.

### 6.1 Image Diffusion and VAE

The pixel-space image diffusion U-Net denoises RGB tensors directly. The native
latent-diffusion path first encodes an image $x$ with `AutoencoderKL` into a
Gaussian latent, $z=\mu+\exp(\tfrac12\log\sigma^2)\odot\xi$, then decodes it
with $\hat{x}=D(z)$. Its VAE objective is:

$$\mathcal{L}_{\mathrm{VAE}}=\lVert\hat{x}-x\rVert_1-
\tfrac{\lambda_{\mathrm{KL}}}{2}\operatorname{mean}
\left(1+\log\sigma^2-\mu^2-\exp(\log\sigma^2)\right)$$

For latent diffusion, the scaled latent $z_0'=s z_0$ is passed to the same
noise-prediction process used below, and decoded as $D(z/s)$. The bundled
`latent.production.yaml` is explicitly `planning_only: true`; it describes a
native 256×256, 4-channel latent architecture but does not claim a bundled or
ready-to-train production checkpoint.

### 6.2 Shared Audio/Video Latent-Diffusion Objective

For a media sample $x$, an autoencoder produces a latent $z_0 = E(x)$ and a
reconstruction $\hat{x} = \tanh(D(z_0))$. At a randomly sampled diffusion step
$t$, the scheduler creates a noised latent:

$$z_t = \sqrt{\bar{\alpha}_t}z_0 + \sqrt{1-\bar{\alpha}_t}\,\tilde{\epsilon}$$

where $\tilde{\epsilon}$ may include configured noise offset and input
perturbation. The denoiser predicts the original noise $\epsilon$:

$$\hat{\epsilon}_\theta = \epsilon_\theta(z_t, t, c, C)$$

Training combines noise-prediction MSE with L1 reconstruction loss:

$$\mathcal{L} = \mathbb{E}\left[w_t\lVert\hat{\epsilon}_\theta-\epsilon\rVert_2^2\right]
+ \lambda_{\mathrm{recon}}\lVert\hat{x}-x\rVert_1$$

When `min_snr_gamma` is positive, the implementation uses
$w_t=\min(\mathrm{SNR}_t,\gamma)/\mathrm{SNR}_t$, with
$\mathrm{SNR}_t=\bar{\alpha}_t/(1-\bar{\alpha}_t)$. Classifier-free guidance
is trained by dropping text conditioning with probability
`condition_dropout`; at DDIM inference it combines predictions as:

$$\hat{\epsilon}_{\mathrm{cfg}} = \hat{\epsilon}_{\mathrm{uncond}}
+ s(\hat{\epsilon}_{\mathrm{cond}}-\hat{\epsilon}_{\mathrm{uncond}})$$

where $s$ is `guidance_scale`.

### 6.3 Text Conditioning and FiLM

Given text-encoder states $H\in\mathbb{R}^{B\times S\times d}$ and attention
mask $m$, the pooled condition is:

$$c = \frac{\sum_{j=1}^{S}m_jH_j}{\max(1,\sum_{j=1}^{S}m_j)}$$

The denoisers add MLP projections of a sinusoidal timestep embedding and $c$.
In a residual block, FiLM applies the resulting embedding $e$ as a learned
per-channel scale and shift: $\mathrm{FiLM}(h,e)=\gamma(e)\odot h+\beta(e)$.

### 6.4 Audio Architecture

`AudioAutoencoder1D` maps mono waveforms $x\in\mathbb{R}^{B\times 1\times N}$
through strided 1-D convolutions to $z\in\mathbb{R}^{B\times C_z\times
\lceil N/2^k\rceil}$, where $k$ is `downsample_stages`; transposed convolutions
decode the result. `AudioDenoiser1D` is a WaveNet-like stack of dilated residual
blocks (dilations $2^{i\bmod 8}$), with optional text cross-attention.

The 4 GB development profile uses 16 kHz, four-second mono clips (64,000
samples), $k=6$, 32 latent channels, 128 denoiser channels, 10 residual blocks,
and cross-attention every third block. See
[`configs/audio_generation/local_4gb.yaml`](../configs/audio_generation/local_4gb.yaml).

### 6.5 Video Architecture

`VideoAutoencoder3D` encodes RGB clips
$x\in\mathbb{R}^{B\times3\times T\times H\times W}$ with 3-D convolutions.
Each spatial stage halves height and width; when `temporal_downsample` is true,
the first stage also halves time. Thus the latent shape is approximately:

$$B\times C_z\times\lceil T/f_t\rceil\times\lceil H/2^k\rceil\times\lceil W/2^k\rceil$$

where $f_t\in\{1,2\}$ and $k$ is `spatial_stages`. The `VideoDenoiser3D` uses
FiLM-conditioned 3-D residual blocks, temporal self-attention independently at
each latent spatial position, and memory-bounded cross-attention over flattened
video latent tokens in configurable chunks.

The 4 GB development profile is deliberately small: 8 RGB frames at 64×64 and
8 FPS, three spatial stages with temporal downsampling, 8 latent channels, 96
denoiser channels, eight residual blocks, temporal attention every second block,
and text cross-attention every fourth block. See
[`configs/video_generation/local_4gb.yaml`](../configs/video_generation/local_4gb.yaml).

Both media pipelines use a 1,000-step cosine noise schedule during training and
DDIM sampling at inference. They support text-to-media as well as initialized
audio/video or image-to-video generation; an initialized input is encoded,
noised according to `strength`, then denoised, optionally preserving masked
regions.
