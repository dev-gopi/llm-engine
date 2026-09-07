# MiniGPT core architecture review

The active `configs/model.gpu.yaml` profile is an internally consistent compact
causal decoder: 16 layers, hidden width 512, 8 query heads, 2 KV heads, head width
64, RoPE, pre-norm RMSNorm, SwiGLU with width 2048, and tied vocabulary embeddings.
The 40,000-token profile has 81,314,304 parameters. Expanding the tied vocabulary
to 42,000 adds 1,024,000 parameters. The configured context is 1,024 tokens;
current fine-tuning examples are capped at 512.

The size estimator reports approximately 0.303 GiB for FP32 weights and 0.008
GiB for one full-length BF16 KV cache. These are not training-memory estimates:
gradients, Adam state, EMA, activations, full vocabulary logits, and temporary
buffers add substantial memory. GQA reduces KV storage; checkpointing and
chunked loss exchange computation for reduced retained activation storage.

## Position handling extension

Learned, rotary, and sinusoidal positions now derive logical token positions
from a binary attention mask when explicit IDs are absent. Masked tokens do not
advance the logical position. Cached decoding uses the complete prefix mask
when supplied. If only the current-token mask is supplied, cached tokens are
assumed valid; callers must provide the complete mask for padded caches.

Sinusoidal embeddings now accept explicit `[sequence]` or `[batch, sequence]`
position IDs, including the explicit IDs used by batched decoding. Explicit IDs
take precedence over automatic positions. Parameter shapes and state-dict keys
are unchanged. Unmasked calls retain their original position behavior. Padded
calls can produce corrected, different outputs.

Tests compare padded prefill and cached decoding against unpadded sequences
for all three position mechanisms, and check sinusoidal explicit-ID/offset
agreement. Existing generation, attention, training, and export tests are also
included in validation.

## Limits of this review

This establishes code-level consistency, not that every subsystem is correct or
that the model achieves a particular accuracy. No production checkpoint was
trained or benchmarked here, and CUDA performance has not been measured.
Tokenization, corpus quality, validation coverage, and training duration remain
important to final quality. No larger context window or new learned architecture
was enabled by this change; these require separate training and evaluation.
