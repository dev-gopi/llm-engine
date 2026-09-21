# ADR-0003: Adoption of Rotary Position Embeddings (RoPE)

## Status
`ACCEPTED`

## Context
Traditional absolute positional embeddings learn fixed lookup vectors up to a pre-set sequence length (e.g. 512 or 1024), degrading catastrophically when presented with sequence lengths beyond the training horizon.

## Decision
We adopted **Rotary Position Embeddings (RoPE)** (`position_type: rotary`) across Query and Key projections in attention.

## Alternatives Considered
- **Learned Absolute Embeddings (GPT-2 style)**: Static, incapable of zero-shot context length extension, requires $O(T \cdot d)$ extra parameters.
- **ALiBi**: Modifies attention bias directly; effective but less compatible with standard FlashAttention and GQA optimizations than RoPE.

## Consequences
- **Positive**: Encodes relative token distance naturally via rotation angles; zero parameter memory footprint; extensible via frequency interpolation (NTK-aware / YaRN).
- **Negative**: Attention head dimension must remain an even integer.

## Related Files
- [`src/model/positional.py`](../../src/model/positional.py)
- [`src/model/attention.py`](../../src/model/attention.py)
- [`tests/test_positional.py`](../../tests/test_positional.py)

