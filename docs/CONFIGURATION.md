# Configuration Reference & Parameter Catalog (`docs/CONFIGURATION.md`)

*Authoritative Source: [`src/utils/config.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/utils/config.py), [`src/model/config.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/config.py)*

This document provides a reference for all configuration parameters across `configs/*.yaml` and environment variables.

---

## 1. YAML Configuration Inheritance

YAML files loaded through `src/utils/config.py:load_yaml` support single and multiple inheritance via the `extends` keyword:

```yaml
extends: defaults/training-runtime.yaml
runtime:
  tokenizer: data/tokenizer-finetuning
  output: checkpoints/finetuning/latest.pt
  best_output: checkpoints/finetuning/best.pt
```

- Parent paths are relative to the file containing `extends`.
- Inheritance merges parent mappings left-to-right; child properties override parent values.
- Lists and scalar values are replaced, not concatenated.
- Precedence: **Explicit CLI Argument > Active Child YAML > Extended Parent Defaults > Python Code Fallback**.

---

## 2. Model Architecture Parameters (`configs/model.gpu.yaml`)

| Name | Type | Default | Valid Values | Purpose | Used By | Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `vocab_size` | `int` | `40000` | Positive int | Total token vocabulary capacity | `GPTModel`, `EmbeddingLayer` | Changes weight matrix shapes; breaking if modified without migration |
| `hidden_size` | `int` | `512` | Multiples of `heads` | Dimensionality of transformer residual stream | All model layers | Primary scale factor for model capacity and VRAM usage |
| `layers` | `int` | `16` | Positive int | Number of Transformer decoder layers | `GPTModel` | Direct linear scaling of compute and memory depth |
| `heads` | `int` | `8` | Factor of `hidden_size` | Number of Query attention heads | `CausalSelfAttention` | Determines query head dimension $d_k = \text{hidden} / \text{heads}$ |
| `kv_heads` | `int` | `2` | Factor of `heads` | Number of Key and Value heads (GQA) | `CausalSelfAttention` | 4x KV cache reduction during autoregressive serving |
| `max_position` | `int` | `1024` | Positive int | Maximum sequence context length | `RotaryEmbedding`, `Generator` | Upper bound for generation and attention masks |
| `position_type` | `str` | `"rotary"` | `"rotary"`, `"learned"`, `"sinusoidal"` | Positional encoding method | `RotaryEmbedding` | RoPE allows relative position modeling |
| `rope_base` | `float` | `10000.0` | Positive float | Base frequency for rotary embeddings | `RotaryEmbedding` | Frequency scaling factor |
| `rope_scale` | `float` | `1.0` | Positive float | Scaling factor for context interpolation | `RotaryEmbedding` | Interpolation factor for context extension |
| `ffn_hidden_size` | `int` | `2048` | Positive int | Intermediate feed-forward layer width | `FeedForward` | Representation capacity |
| `ffn_multiple_of` | `int` | `128` | Positive int | Alignment multiple for GPU tensor cores | `FeedForward` | Optimizes GEMM kernel execution on CUDA |
| `ffn_activation` | `str` | `"swiglu"` | `"swiglu"`, `"geglu"`, `"gelu"`, `"relu"` | Non-linear activation function | `FeedForward` | SwiGLU offers superior convergence efficiency |
| `norm_type` | `str` | `"rms_norm"` | `"rms_norm"`, `"layer_norm"` | Layer normalization algorithm | `RMSNorm` | RMSNorm is ~15% faster and omits mean centering |
| `norm_eps` | `float` | `1e-5` | Small positive float | Epsilon for numerical stability | `RMSNorm` | Prevents division by zero in variance computation |
| `tie_word_embeddings`| `bool` | `true` | `true`, `false` | Share input and output LM head weights | `GPTModel` | Saves ~80 MB of VRAM by eliminating separate projection matrix |
| `gradient_checkpointing` | `bool` | `true` | `true`, `false` | Recompute activations in backward pass | `TransformerBlock` | **Mandatory on 4 GB VRAM**; reduces activation memory >75% |

---

## 3. Training Hyperparameters (`configs/pretraining.gpu.yaml`)

| Name | Type | Default | Valid Values | Purpose | Used By | Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `batch_size` | `int` | `2` | `1`, `2`, `4` | Per-step micro-batch size | `Trainer`, `DataLoader` | Must remain $\le 2$ on 4 GB GPUs to avoid CUDA OOM |
| `gradient_accumulation_steps` | `int` | `16` | $\ge 1$ | Accumulation steps before optimizer step | `Trainer` | Scales effective batch size ($2 \times 16 = 32$) |
| `learning_rate` | `float` | `5e-5` | Positive float | Peak learning rate | `AdamW`, `Scheduler` | Determines weight update step size |
| `weight_decay` | `float` | `0.1` | $\ge 0.0$ | Decoupled L2 regularization | `AdamW` | Excludes 1D norm gains and biases |
| `beta1` | `float` | `0.9` | $(0, 1)$ | First moment momentum coefficient | `AdamW` | Gradient direction smoothing |
| `beta2` | `float` | `0.95` | $(0, 1)$ | Second moment variance coefficient | `AdamW` | Variance adaptation rate |
| `mixed_precision` | `str` | `"fp16"` | `"fp16"`, `"bf16"`, `"none"` | AMP precision mode | `Trainer`, `GradScaler` | Halves activation memory and leverages Tensor Cores |
| `lr_schedule` | `str` | `"cosine"` | `"cosine"`, `"linear"`, `"constant"` | Learning rate decay schedule | `Scheduler` | Cosine decay with warmup provides smooth convergence |
| `warmup_ratio` | `float` | `0.05` | $[0.0, 0.5]$ | Fraction of steps spent warming up LR | `Scheduler` | Prevents early gradient explosion |
| `ema_decay` | `float` | `0.999` | $(0.9, 1.0)$ | Decay rate for exponential moving average | `EMAModel` | Generates smoother, more resilient checkpoints |
| `z_loss_coefficient` | `float` | `0.0001` | $\ge 0.0$ | Logit regularizer penalizing large $\log Z$ | `ChunkedCrossEntropyLoss` | Stabilizes logits and prevents numeric overflow |
| `early_stopping_patience` | `int` | `3` | $\ge 1$ | Validation evaluations before stopping | `Trainer` | Halts training when validation loss stops improving |

---

## 4. Serving & Inference Hyperparameters (`configs/inference.yaml`)

| Name | Type | Default | Valid Values | Purpose | Used By | Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `temperature` | `float` | `0.7` | $\ge 0.0$ | Softmax entropy scaling | `Sampler` | Lower = deterministic; Higher = creative |
| `top_p` | `float` | `0.9` | $(0.0, 1.0]$ | Nucleus probability mass threshold | `Sampler` | Restricts sampling to high-probability tokens |
| `top_k` | `int` | `50` | $\ge 0$ | Number of top candidate tokens | `Sampler` | Truncates improbable token tail |
| `repetition_penalty` | `float` | `1.1` | $\ge 1.0$ | Multiplicative logit penalty on seen tokens | `Sampler` | Eliminates degenerate generation loops |
| `max_tokens` | `int` | `256` | Positive int | Maximum generated tokens | `Generator` | Enforces generation budget |
| `max_batch_size` | `int` | `4` | Positive int | Maximum concurrent dynamic batch requests | `DynamicBatcher` | Balances throughput and VRAM |
| `batch_timeout_ms` | `int` | `10` | $\ge 0$ | Queue aggregation timeout | `DynamicBatcher` | Bounds latency during low traffic |

---

## 5. Environment Variables (`.env`)

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PORT` | `8000` | HTTP and WebSocket server listening port |
| `HOST` | `0.0.0.0` | Bind network address for serving |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DEVICE` | `auto` | Target device selection (`cuda`, `cpu`, `auto`) |
| `RATE_LIMIT_PER_MINUTE`| `60` | Maximum requests per IP/token per minute |
| `WORKSPACE_DIR` | `.` | Restricts sandboxed local tool execution boundary |
