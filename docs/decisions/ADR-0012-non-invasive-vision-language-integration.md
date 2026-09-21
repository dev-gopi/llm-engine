# ADR-0012: Non-Invasive Vision-Language Integration (`ADR-0012-non-invasive-vision-language-integration.md`)

## Status

**Accepted**

## Context

The llm-engine project needed multimodal vision-language capabilities — the ability to
process images alongside text and generate text responses conditioned on visual input.

The existing text model (`MiniGPT`) had a stable architecture with established checkpoints,
training pipelines, and downstream users. Any integration approach had to satisfy these
constraints:

1. **Checkpoint compatibility** — existing text-only checkpoints must load and run without
   modification.
2. **Source preservation** — the `MiniGPT` class and its modules must not be modified, keeping
   the text pipeline fully independent.
3. **Independent training** — the vision and language backbones should be freezable
   independently, enabling efficient projector-only training.
4. **Clean separation** — vision components should live in their own module hierarchy
   (`src/vision/`, `src/multimodal/`) with no cross-imports into `src/model/`.

Alternative approaches considered:

- **Inline modification**: Adding image embedding logic directly into `MiniGPT.forward()`.
  Rejected because it would break checkpoint compatibility and couple vision logic into the
  text model.
- **Subclassing**: Creating a `MultimodalGPT(MiniGPT)` subclass that overrides `forward()`.
  Rejected because it tightly couples to `MiniGPT` internals and complicates checkpoint
  loading.
- **Adapter injection**: Inserting adapter layers into existing transformer blocks. Rejected
  because it modifies the model graph and requires checkpoint migration.

## Decision

We adopt a **non-invasive wrapper** pattern:

`VisionLanguageModel` wraps `VisionEncoder` + `MiniGPT` externally, connecting them through a
`VisionProjector` module (Linear → GELU → Dropout → Linear → LayerNorm) that maps vision
features into the language model's embedding space.

### Key design choices

1. **External composition** — `VisionLanguageModel` holds references to `VisionEncoder`,
   `MiniGPT`, and `VisionProjector` as submodules. It accesses `MiniGPT`'s existing public
   submodules (`tok`, `blocks`, `head`, `norm`, `rotary_emb`) without modifying their source.

2. **Embedding-level integration** — The projector output is concatenated with text token
   embeddings at the embedding level, before passing through the transformer blocks. This
   avoids any changes to block internals.

3. **`_language_forward_from_embeddings`** — A private method on `VisionLanguageModel`
   duplicates the core forward logic of `MiniGPT` (embedding → blocks → norm → head),
   operating on pre-built embedding sequences that include visual tokens.

4. **Independent freezing** — `freeze_vision()` and `freeze_language()` methods freeze
   individual backbones and set them to `eval()` mode (disabling dropout). During projector
   training, only the projector parameters receive gradients.

5. **Visual token selection** — Visual tokens are taken from patch positions `1..visual_tokens`,
   excluding the CLS token at position 0, to provide spatial information to the language model.

6. **Loss masking** — Prompt tokens and visual tokens are assigned `loss_mask=False` so that
   only response tokens contribute to the training loss.

## Consequences

### Positive

- **Full checkpoint compatibility** — `MiniGPT` checkpoints load unchanged. The
  `VisionLanguageModel` state dict cleanly separates into `vision_encoder.*`,
  `language_model.*`, and `projector.*` prefixes.
- **Projector-only training** — With both backbones frozen, training requires gradients only
  through the small projector module, enabling efficient fine-tuning on consumer hardware.
- **Clean module boundaries** — Vision code in `src/vision/`, multimodal glue in
  `src/multimodal/`, text model in `src/model/` — no cross-contamination.
- **Independent evolution** — Text and vision components can be upgraded, replaced, or
  scaled independently.

### Negative

- **Forward logic duplication** — `_language_forward_from_embeddings` replicates parts of
  `MiniGPT.forward()`. If `MiniGPT`'s forward pass changes (e.g., new normalization, attention
  mask handling), the duplication must be updated manually.
- **Position type synchronization** — Position embedding and rotary encoding handling must be
  kept in sync between `MiniGPT` and the wrapper's `_language_forward_from_embeddings`. This
  is a manual coordination point.
- **No deep fusion** — The non-invasive approach limits integration to the embedding level.
  More sophisticated cross-modal attention patterns (e.g., interleaved cross-attention within
  transformer blocks) would require a different architecture.

## Related

- [ADR-0013: Compact Latent Diffusion Architecture](ADR-0013-compact-latent-diffusion-architecture.md)
- [src/multimodal/model.py](../../src/multimodal/model.py)
- [src/multimodal/projector.py](../../src/multimodal/projector.py)
- [src/vision/encoder.py](../../src/vision/encoder.py)
- [configs/vision/multimodal.yaml](../../configs/vision/multimodal.yaml)
