# Image model extensions

Image understanding and image generation are isolated from the existing text model.
No existing language-model architecture or checkpoint is modified.

## Install image I/O

```bash
.venv/bin/pip install -e '.[images]'
```

## Vision understanding

The scratch ViT lives under `src/vision`. It converts a 128×128 image into patch
tokens. `src/multimodal` projects selected patch tokens from the vision width into
the existing `MiniGPT` embedding width. Initially train the projector while both
the vision encoder and language model are frozen.

Inspect the untrained architecture:

```bash
.venv/bin/python scripts/inspect_vision_model.py \
  --config configs/vision/model.small.yaml
```

Multimodal JSONL records should use this schema:

```json
{"image":"images/cat.jpg","prompt":"What is shown?","response":"A cat is sitting beside a window."}
```

## Diffusion generation

The scratch DDPM implementation lives under `src/diffusion`. It contains a noise
scheduler, a small conditional U-Net, a training-loss function, and a sampling
loop. Sampling random weights is only a wiring test and does not create meaningful
images:

```bash
.venv/bin/python scripts/sample_diffusion.py \
  --config configs/diffusion/model.small.yaml \
  --output outputs/generated_images/smoke-test.png
```

Useful generation requires training on normalized images. Text-conditioned
generation additionally requires paired image-caption data and learned text
conditioning. Keep vision, multimodal, diffusion, and text checkpoints in separate
directories.
