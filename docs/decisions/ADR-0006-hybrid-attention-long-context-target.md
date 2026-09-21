# ADR-0006: Target Long-Context Architecture: Hybrid Attention & RoPE Scaling

## Status
`PROPOSED`

## Context
Expanding context length from 1024 to 4K, 8K, and 16K with standard full attention imposes quadratic $O(N^2)$ memory and compute costs that cannot fit in 4 GB VRAM.

## Decision
We propose evolving the backbone into a **Hybrid Attention Architecture**:
- 75% of layers utilize linear/recurrent attention (sub-quadratic $O(N)$ memory and compute).
- 25% of layers retain full Grouped-Query Attention with causal masking for expressive retrieval.
- Context window expansion via **YaRN / NTK-aware RoPE scaling**.

## Alternatives Considered
- **Pure Full Attention with FlashAttention only**: FlashAttention reduces constant factors but cannot circumvent quadratic memory growth for long sequence KV caches.
- **Pure State Space Models (Mamba / RWKV)**: Eliminates KV cache entirely, but exhibits weaker in-context needle retrieval compared to hybrid models.

## Consequences
- **Positive**: Enables 8K+ context processing on consumer GPUs; retains high recall via periodic full-attention layers.
- **Negative**: Requires architectural retraining and custom layer routing.

## Related Files
- [`docs/TARGET_STATE.md`](../TARGET_STATE.md)
- [`docs/UPGRADE_ROADMAP.md`](../UPGRADE_ROADMAP.md)
- [`src/model/attention.py`](../../src/model/attention.py)

