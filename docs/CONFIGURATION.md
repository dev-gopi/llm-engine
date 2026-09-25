# Configuration Reference & Parameter Catalog (`docs/CONFIGURATION.md`)

*Authoritative Source: [`src/utils/config.py`](../src/utils/config.py), [`src/model/config.py`](../src/model/config.py)*

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
| `validation_lr_adaptation_enabled` | `bool` | `true` | `true`, `false` | Master switch for validation-driven LR adaptation | `Trainer` | When `false`, keeps best-checkpoint selection and early stopping but never changes LR |
| `validation_lr_decay_factor` | `float` or `null` | `0.5` | $(0, 1)$ or `null` | Multiplier applied after a validation plateau | `Trainer` | Lowers the remaining scheduled learning-rate curve without changing model architecture |
| `validation_lr_patience` | `int` | `1` | $\ge 1$ | Consecutive non-improving validation checks before each LR reduction | `Trainer` | Controls how quickly plateau recovery begins |
| `validation_lr_min_scale` | `float` | `0.25` | $(0, 1]$ | Floor relative to the scheduled learning rate | `Trainer` | Prevents repeated reductions from making learning ineffective |
| `validation_lr_min_steps_between_decays` | `int` | evaluation interval | $\ge 0$ | Minimum optimizer steps between validation-triggered reductions | `Trainer` | Avoids multiple LR reductions from closely spaced checks |

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

---

## 6. Image, audio, and video generation configuration

### Image diffusion

Image-generation profiles live under [`configs/diffusion/`](../configs/diffusion/).
`training.local.yaml` configures the runnable pixel-space DDPM/DDIM workflow;
`latent.production.yaml` defines a separate native VAE/latent-diffusion research
profile and is intentionally blocked by `planning_only: true` until a reviewed
dataset and run plan are supplied.

| Name | Used by | Purpose |
| :--- | :--- | :--- |
| `image_size`, `image_channels` | Image dataset, U-Net, VAE | Input/output image geometry and channels |
| `train_data`, `validation_data` | Image dataset | Root directories containing licensed training/validation images |
| `base_channels`, `channel_multipliers`, `res_blocks`, `attention_resolutions` | Pixel U-Net | Image denoiser capacity and attention placement |
| `num_classes`, `class_dropout_probability` | Optional class-conditioned DDPM | Classifier-free conditioning; omit classes for unconditional diffusion |
| `vae_base_channels`, `vae_downsample_factor`, `latent_channels`, `latent_scale` | Native latent path | VAE capacity, compression, and diffusion latent scaling |
| `kl_weight`, `vae_batch_size`, `vae_epochs`, `vae_learning_rate` | `train_vae.py` | VAE reconstruction/KL balance and optimization |
| `timesteps`, `noise_schedule`, `inference_steps`, `ddim_eta` | Image diffusion | Noise schedule and DDIM sampling quality/speed trade-off |
| `image_normalization`, `train_resize_mode`, `eval_resize_mode` | Image processor | Preprocessing; training and inference must agree on normalization |

### Audio and video

Audio and video diffusion profiles are self-contained YAML files under
[`configs/audio_generation/`](../configs/audio_generation/) and
[`configs/video_generation/`](../configs/video_generation/). They must use the
same tokenizer and architecture when resuming a checkpoint.

| Name | Used by | Valid values / constraint | Purpose |
| :--- | :--- | :--- | :--- |
| `tokenizer` | Audio and video trainers | Existing tokenizer directory | Tokenizer lineage for text conditioning |
| `train_manifest`, `validation_manifest` | Dataset loaders | JSONL records containing `audio` or `video` plus non-empty `text` | Captioned media inputs; relative paths resolve from the manifest |
| `sample_rate`, `duration_seconds` | Audio | Positive | Fixed decoded waveform length is `round(sample_rate × duration_seconds)` |
| `frames`, `height`, `width`, `fps` | Video | Positive; height/width divisible by `2^spatial_stages` | Clip geometry and MP4 output rate |
| `text_hidden_size`, `text_layers`, `text_heads`, `text_max_length` | Text conditioner | Positive; hidden size divisible by heads | Trainable tokenizer-backed text encoder |
| `autoencoder_channels`, `latent_channels` | Both | Positive | Encoder/decoder width and diffusion latent width |
| `downsample_stages` | Audio | Positive | Waveform compression factor `2^downsample_stages` |
| `spatial_stages`, `temporal_downsample` | Video | Positive; boolean | Spatial compression `2^spatial_stages` and optional first-stage 2× temporal compression |
| `model_channels`, `blocks`, `dropout` | Both | Positive widths/block count; dropout in `[0,1)` | Denoiser capacity and regularization |
| `temporal_heads`, `attention_every` | Video | `model_channels` divisible by heads | Per-spatial-location temporal self-attention |
| `cross_attention_every`, `cross_attention_heads`, `cross_attention_chunk_size` | Both; chunk size video only | Channel count divisible by cross-attention heads | Token-level text cross-attention; video chunks flattened latent tokens to bound memory |
| `timesteps`, `noise_schedule`, `beta_start`, `beta_end` | Both | Positive; schedule currently `cosine` or scheduler-supported choice | Forward noise schedule |
| `condition_dropout`, `guidance_scale` | Both | Dropout in `[0,1]`; non-negative scale | Classifier-free guidance training and sampling strength |
| `reconstruction_weight`, `min_snr_gamma`, `noise_offset`, `input_perturbation` | Both | Non-negative | Diffusion-loss stabilization and reconstruction balance |
| `inference_steps`, `ddim_eta` | Both | Steps in `[1, timesteps]`; eta non-negative | DDIM sampling quality/speed and stochasticity |
| `batch_size`, `gradient_accumulation_steps`, `mixed_precision` | Both | Positive; `none`, `fp16`, or `bf16` | Memory budget and effective batch size |
| `output`, `best_output`, `save_every_steps` | Both | Writable paths; positive interval | Rolling resume and lowest-validation-loss EMA checkpoints |

The `local_4gb.yaml` profiles use `batch_size: 1`, accumulation of 16, and FP16.
Increase frame count, spatial resolution, clip duration, or denoiser width only
after measuring peak VRAM and validating checkpoint restore.
