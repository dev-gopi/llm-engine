# Vision, Multimodal & Diffusion Context Pack (`vision.context.md`)

> Compact reference for AI agents working on the vision, multimodal, and diffusion
> subsystems. Covers architecture, source layout, configuration, and constraints.

---

## Quick Reference

| Subsystem | Key Components |
|-----------|---------------|
| **Vision Encoder** | ViT with PatchEmbedding (Conv2d projection), 4 layers, hidden=192, 3 heads, image_size=128, patch_size=16 |
| **Classifier** | `VisionEncoder.pooled()` → Linear head (≥2 classes) |
| **Multimodal** | `VisionLanguageModel` wraps `VisionEncoder` + `MiniGPT` non-invasively via `VisionProjector` (Linear→GELU→Dropout→Linear→LayerNorm) |
| **Diffusion** | `SmallUNet` (2-level U-Net) + `DiffusionScheduler` (DDPM/DDIM, linear/cosine) |
| **Latent Diffusion** | `AutoencoderKL` + `SmallUNet` + `DiffusionTextEncoder` + `LatentDiffusionPipeline` |
| **Image Data** | Pillow-based processor (no torchvision), folder datasets, SHA-256 audit |

---

## Source Files

### Vision

- [src/vision/encoder.py](../../src/vision/encoder.py) — `VisionEncoder`, `VisionBlock`
- [src/vision/classifier.py](../../src/vision/classifier.py) — `VisionClassifier`
- [src/vision/patch_embedding.py](../../src/vision/patch_embedding.py) — `PatchEmbedding`

### Multimodal

- [src/multimodal/model.py](../../src/multimodal/model.py) — `VisionLanguageModel`
- [src/multimodal/projector.py](../../src/multimodal/projector.py) — `VisionProjector`

### Diffusion

- [src/diffusion/pipeline.py](../../src/diffusion/pipeline.py) — `DiffusionPipeline`
- [src/diffusion/latent_pipeline.py](../../src/diffusion/latent_pipeline.py) — `LatentDiffusionPipeline`
- [src/diffusion/unet.py](../../src/diffusion/unet.py) — `SmallUNet`, `TimeEmbedding`, `ResidualBlock`, `SpatialAttention`, `SpatialCrossAttention`
- [src/diffusion/vae.py](../../src/diffusion/vae.py) — `AutoencoderKL`, `VAEOutput`
- [src/diffusion/scheduler.py](../../src/diffusion/scheduler.py) — `DiffusionScheduler`
- [src/diffusion/text_encoder.py](../../src/diffusion/text_encoder.py) — `DiffusionTextEncoder`

### Image Data

- [src/image_data/dataset.py](../../src/image_data/dataset.py) — `ImageDataset`, `ImageClassificationDataset`, `discover_images`
- [src/image_data/processor.py](../../src/image_data/processor.py) — `ImageProcessor`, `load_image`, `tensor_to_image`
- [src/image_data/audit.py](../../src/image_data/audit.py) — `audit_images`, `ImageAudit`

---

## Key Configs

### Vision & Multimodal

- [configs/vision/model.small.yaml](../../configs/vision/model.small.yaml)
- [configs/vision/multimodal.yaml](../../configs/vision/multimodal.yaml)
- [configs/vision/training.local.yaml](../../configs/vision/training.local.yaml), [training.production.yaml](../../configs/vision/training.production.yaml), [training.hf-sample.yaml](../../configs/vision/training.hf-sample.yaml)

### Diffusion

- [configs/diffusion/model.small.yaml](../../configs/diffusion/model.small.yaml)
- [configs/diffusion/training.local.yaml](../../configs/diffusion/training.local.yaml), [training.production.yaml](../../configs/diffusion/training.production.yaml), [training.hf-sample.yaml](../../configs/diffusion/training.hf-sample.yaml)
- [configs/diffusion/latent.production.yaml](../../configs/diffusion/latent.production.yaml)

---

## Tests

- [tests/test_vision_models.py](../../tests/test_vision_models.py)
- [tests/test_diffusion.py](../../tests/test_diffusion.py)
- [tests/test_image_data.py](../../tests/test_image_data.py)
- [tests/test_train_vision_resources.py](../../tests/test_train_vision_resources.py)
- [tests/test_prepare_hf_image_dataset.py](../../tests/test_prepare_hf_image_dataset.py)

---

## Architecture Notes

### Multimodal Integration

- `VisionLanguageModel` is **non-invasive**: it does not modify `MiniGPT` source or checkpoint layout.
- Vision and language backbones can be independently frozen during projector training.
- Frozen modules are kept in `eval()` to disable dropout.
- Visual tokens are selected from patch positions `1..visual_tokens` (CLS excluded).
- **Loss masking**: only response tokens are supervised; prompt and visual tokens get `loss_mask=False`.

### Diffusion Stack

- The diffusion stack is **independent** from text model checkpoints.
- U-Net supports three **mutually exclusive** conditioning modes:
  - `text_condition` — single vector conditioning
  - `class_labels` — class-conditional generation
  - `text_context` — cross-attention sequence conditioning
- CFG (classifier-free guidance) implemented via condition dropout during training and guided sampling.
- VAE `downsample_factor` must be 2, 4, or 8.
- `LatentDiffusionPipeline` `latent_scale` default is `0.18215` (matches Stable Diffusion convention).

---

## Hardware Notes

- Vision defaults are designed for **RTX 3050 4GB**: 128px images, 192-dim hidden, 4 layers.
- Diffusion `base_channels=64` keeps the U-Net compact (~2.3M params).
- Image processor uses **Pillow** (no torchvision dependency).

---

## Common Patterns

### Freezing backbones for projector training

```python
model = VisionLanguageModel(vision_encoder, language_model, projector)
model.freeze_vision()    # freezes VisionEncoder, sets eval()
model.freeze_language()  # freezes MiniGPT, sets eval()
# Only projector parameters receive gradients
```

### Conditioning mode selection (U-Net)

```python
# Vector conditioning (e.g., CLIP embedding)
output = unet(x, t, text_condition=vec)

# Class-conditional
output = unet(x, t, class_labels=labels)

# Cross-attention (sequence)
output = unet(x, t, text_context=seq)

# These are MUTUALLY EXCLUSIVE — do not combine
```

### Latent diffusion pipeline

```python
pipeline = LatentDiffusionPipeline(vae, unet, scheduler, text_encoder)
images = pipeline.generate(prompt, num_steps=50, guidance_scale=7.5)
```
