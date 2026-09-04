# Vision and diffusion models

The image stack is independent from text-model checkpoints. It includes a Vision
Transformer encoder and classifier, a multimodal projector, and a conditional
pixel-space diffusion U-Net. The repository provides local smoke-test profiles and
larger profiles labelled `production`; it does not provide pretrained weights.
Production-labelled means stronger architecture and operational defaults, not a
claim that unvalidated weights are ready to serve.

## Install image support

```bash
.venv/bin/pip install -e '.[images]'
```

Audit a dataset before training:

```bash
.venv/bin/python scripts/audit_image_dataset.py data/images/hf-mnist/train \
  --fail-on-corrupt
```

All image workflows share `ImageProcessor`. Configuration supports
`train_resize_mode` (`random_crop`, `center_crop`, or `stretch`),
`eval_resize_mode`, `image_normalization` (`minus_one_one`, `zero_one`, or
`imagenet`), `horizontal_flip_probability`, and `color_jitter`. Diffusion should
normally retain `minus_one_one`; train and inference must use the same
normalization.

## Is a tokenizer required?

- Vision classification does not use text, so it does not need a tokenizer.
- Unconditional diffusion does not use text and does not need a tokenizer.
- The vision-language adapter uses the same tokenizer as `MiniGPT`.
- Text-to-image diffusion needs both a tokenizer and a trained text encoder. A
  tokenizer alone cannot produce useful conditioning vectors; the current sample
  diffusion trainer is intentionally unconditional.

## Download the Hugging Face sample

The default sample is `ylecun/mnist`: 1,000 training images and 200 held-out
images. It is written once in class folders that both trainers can read.

```bash
.venv/bin/python scripts/prepare_hf_image_dataset.py
```

Then train both sample profiles:

```bash
.venv/bin/python scripts/train_vision.py \
  --config configs/vision/training.hf-sample.yaml

.venv/bin/python scripts/train_diffusion.py \
  --config configs/diffusion/training.hf-sample.yaml
```

Generate and classify:

```bash
.venv/bin/python scripts/sample_diffusion.py \
  --config configs/diffusion/training.hf-sample.yaml \
  --checkpoint checkpoints/diffusion/best.pt \
  --class-id 7 --guidance-scale 3 \
  --output outputs/generated_images/mnist.png

.venv/bin/python scripts/classify_image.py data/images/hf-mnist/validation/0/000003.png \
  --config configs/vision/training.hf-sample.yaml \
  --checkpoint checkpoints/vision/best.pt
```

The downloader accepts other Hugging Face image-classification datasets when
their Parquet rows contain encoded image bytes and integer labels. Pass matching
`--dataset`, `--dataset-config`, column names, split names, and ordered `--labels`.

The Hugging Face diffusion sample is class-conditioned. It learns a class
embedding plus a null class, randomly drops labels during training for
classifier-free guidance, and accepts `--class-id` during generation. Set
`sample_every_steps` to save EMA preview images while training. General
unconditional diffusion profiles remain fully supported by omitting `num_classes`.

## Vision classification

Arrange images in class folders. Folder names become stable alphabetical labels:

```text
data/images/vision/train/
├── cat/
│   ├── 001.jpg
│   └── 002.jpg
└── dog/
    ├── 001.jpg
    └── 002.jpg
```

For production training, create `data/images/vision/validation` with the same
class folders. The trainer evaluates every epoch and writes `best.pt` beside the
latest checkpoint. Training uses deterministic seeding, horizontal-flip
augmentation, gradient clipping, warmup plus cosine decay, and optional BF16/FP16.

Set `num_classes` in the selected config to the number of folders, then train:

```bash
.venv/bin/python scripts/train_vision.py \
  --config configs/vision/training.local.yaml \
  --data data/images/vision/train \
  --output checkpoints/vision/local.pt \
  --device cuda
```

Resume the same optimizer stage with `--resume checkpoints/vision/local.pt`.
Classify an image after training:

```bash
.venv/bin/python scripts/classify_image.py images/example.jpg \
  --config configs/vision/training.local.yaml \
  --checkpoint checkpoints/vision/local.pt \
  --top-k 2 --device cuda
```

The encoder supports `pool_type: cls` or `mean`. Set `strict_image_size: false`
to accept rectangular resolutions divisible by the patch size; learned positional
embeddings are bicubically interpolated.

## Vision-language adapter

`src/multimodal` maps image patch tokens into `MiniGPT`. The profile at
`configs/vision/multimodal.yaml` freezes both backbones and trains only the
projector. Expected paired records are:

```json
{"image":"images/cat.jpg","prompt":"What is shown?","response":"A cat beside a window."}
```

This adapter remains an architecture component. A caption/VQA data collator and
evaluation policy must be chosen for the target dataset before production use.

## Diffusion training

Put licensed training images anywhere beneath one directory:

```text
data/images/diffusion/train/
├── image-0001.jpg
└── nested/image-0002.png
```

The production profile also expects `data/images/diffusion/validation`. Validation
uses EMA weights and the lowest-loss checkpoint is written as `best.pt`. Diffusion
training supports mixed precision, learning-rate warmup/decay, gradient clipping,
horizontal flips, non-finite loss checks, and resumable EMA/optimizer/scaler state.

Train the local profile:

```bash
.venv/bin/python scripts/train_diffusion.py \
  --config configs/diffusion/training.local.yaml \
  --data data/images/diffusion/train \
  --output checkpoints/diffusion/local.pt \
  --device cuda
```

Resume with `--resume checkpoints/diffusion/local.pt`. Checkpoints contain model,
optimizer, step, RNG state, and the complete resolved configuration.

Generate one image with fewer DDIM steps:

```bash
.venv/bin/python scripts/sample_diffusion.py \
  --config configs/diffusion/training.local.yaml \
  --checkpoint checkpoints/diffusion/local.pt \
  --steps 50 --eta 0 \
  --output outputs/generated_images/sample.png \
  --device cuda
```

Use `--batch-size 4` to write numbered outputs. `eta: 0` is deterministic for a
fixed seed; larger values add stochasticity. The U-Net supports cosine or linear
noise schedules, bottleneck attention, residual dropout, reduced-step DDIM, and
classifier-free guidance when external conditioning vectors are supplied.

## Production checklist

- Keep train, validation, and test images disjoint and deduplicate them.
- Confirm licenses, consent, privacy, and prohibited-content handling.
- Track validation loss plus task metrics; do not select by training loss alone.
- Use mixed precision and distributed training only after a local correctness run.
- Scan checkpoints and pin configuration, code revision, and dependency versions.
- Evaluate robustness, demographic performance, memorization, and unsafe outputs.
- Serve behind authentication, request limits, input validation, and monitoring.
- Keep vision, multimodal, diffusion, and language checkpoints in separate paths.
