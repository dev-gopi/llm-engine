# ADR-0013: Compact Latent Diffusion Architecture for Consumer Hardware (`ADR-0013-compact-latent-diffusion-architecture.md`)

## Status

**Accepted**

## Context

The llm-engine project needed image generation capability to complement its text and
vision subsystems. The target deployment environment is consumer-grade hardware with
**4GB VRAM** (e.g., NVIDIA RTX 3050), which rules out full-scale diffusion models:

- **Stable Diffusion 1.5** requires ~4GB for the U-Net alone, plus VAE and text encoder,
  totaling ~7–8GB minimum.
- **Stable Diffusion XL** requires 6.5GB+ for the U-Net.
- Even with fp16, these models exceed the 4GB budget when accounting for optimizer states
  and activations during training.

Requirements:

1. **4GB VRAM budget** — full training pipeline (model + optimizer + activations) must fit.
2. **Flexible conditioning** — support unconditional, class-conditional, and text-conditional
   generation.
3. **No external dependencies** — avoid requiring CLIP or other large pretrained models at
   inference time.
4. **Latent-space operation** — generate in compressed latent space for efficiency, with a
   VAE for encoding/decoding.
5. **Fast sampling option** — support deterministic sampling with fewer steps than full DDPM.

## Decision

We implement a custom **compact latent diffusion stack** with four components:

### 1. AutoencoderKL (VAE)

- Configurable `downsample_factor`: 2, 4, or 8 (powers of 2 enforced).
- KL-regularized latent space for smooth interpolation.
- Produces `VAEOutput` dataclass with `sample`, `mean`, `logvar`, and `kl_loss`.
- Separate encode/decode paths enable latent-space operations.

### 2. SmallUNet (2-level U-Net)

- **2 encoder levels + 2 decoder levels** with skip connections (vs. 4 levels in SD).
- `base_channels=64` (vs. 320 in SD), yielding ~2.3M parameters.
- `TimeEmbedding` via sinusoidal positional encoding → Linear → SiLU → Linear.
- `ResidualBlock` with GroupNorm, time conditioning injection, and optional channel change.
- `SpatialAttention` for self-attention at the bottleneck.
- `SpatialCrossAttention` for text-context cross-attention conditioning.

### 3. Conditioning Modes (mutually exclusive)

The U-Net supports three conditioning modes, enforced to be mutually exclusive at
forward-pass time:

| Mode | Input | Mechanism |
|------|-------|-----------|
| `text_condition` | Single vector | Added to time embedding |
| `class_labels` | Integer labels | Embedding table → added to time embedding |
| `text_context` | Token sequence | `SpatialCrossAttention` in residual blocks |

**Classifier-free guidance (CFG)** is supported via:
- Random condition dropout during training (replacing condition with zeros/null).
- Guided sampling: `pred = uncond + scale * (cond - uncond)`.

### 4. DiffusionScheduler (DDPM + DDIM)

- **DDPM**: full stochastic sampling (default 1000 steps).
- **DDIM**: deterministic sampling with configurable step count (default 50 steps).
- Noise schedules: `linear` or `cosine`.
- Unified API: `add_noise()`, `step()`, `set_timesteps()`.

### 5. DiffusionTextEncoder

- Small local transformer encoder (not CLIP) that maps tokenized text to conditioning
  representations.
- Supports both vector output (pooled) for `text_condition` mode and sequence output for
  `text_context` mode.
- Avoids external model dependency.

### 6. LatentDiffusionPipeline

- Orchestrates VAE + U-Net + scheduler + text encoder into a single generation API.
- `latent_scale` default is `0.18215` (matches Stable Diffusion convention for latent
  normalization).

## Consequences

### Positive

- **Fits in 4GB VRAM** — the full training pipeline (VAE + U-Net + optimizer + activations)
  operates within the target memory budget at 128px resolution.
- **DDIM fast inference** — deterministic 50-step sampling produces results 20× faster than
  full 1000-step DDPM, making interactive use feasible.
- **No external dependencies** — the local `DiffusionTextEncoder` avoids requiring CLIP
  weights or the `transformers` library at inference time.
- **Independent stack** — the diffusion subsystem is fully independent from text model
  checkpoints, with separate configs, training scripts, and model files.
- **Flexible conditioning** — three mutually exclusive modes cover the common use cases
  (unconditional, class-conditional, text-to-image) without architectural changes.

### Negative

- **Limited quality** — the compact architecture (2 levels, 64 base channels) produces
  lower-fidelity images than full-scale models. This is an intentional trade-off for
  hardware accessibility.
- **Resolution constraints** — designed for 128px images. Higher resolutions would require
  architectural scaling that may exceed the VRAM budget.
- **Text conditioning quality** — the small local `DiffusionTextEncoder` has limited
  language understanding compared to CLIP ViT-L/14. Complex prompts may not be well
  represented.
- **Mutual exclusivity of conditioning** — cannot combine class labels with text context
  in a single forward pass. Multi-signal conditioning would require architectural changes.

## Related

- [ADR-0012: Non-Invasive Vision-Language Integration](ADR-0012-non-invasive-vision-language-integration.md)
- [src/diffusion/unet.py](../../src/diffusion/unet.py)
- [src/diffusion/vae.py](../../src/diffusion/vae.py)
- [src/diffusion/scheduler.py](../../src/diffusion/scheduler.py)
- [src/diffusion/text_encoder.py](../../src/diffusion/text_encoder.py)
- [src/diffusion/latent_pipeline.py](../../src/diffusion/latent_pipeline.py)
- [configs/diffusion/model.small.yaml](../../configs/diffusion/model.small.yaml)
- [configs/diffusion/latent.production.yaml](../../configs/diffusion/latent.production.yaml)
