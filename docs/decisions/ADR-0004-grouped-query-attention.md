# ADR-0004: Adoption of Grouped-Query Attention (GQA)

## Status
`ACCEPTED`

## Context
Standard Multi-Head Attention (MHA) allocates equal numbers of Query, Key, and Value heads ($H_q = H_{kv} = 8$). During autoregressive generation, storing full KV tensors for every head consumes large amounts of VRAM, limiting batch sizes on consumer hardware.

## Decision
We adopted **Grouped-Query Attention (GQA)** with $H_q = 8$ and $H_{kv} = 2$ (a 4:1 query-to-KV ratio).

## Alternatives Considered
- **Standard Multi-Head Attention (MHA)**: Requires 4x larger KV cache memory during inference.
- **Multi-Query Attention (MQA, $H_{kv}=1$)**: Maximum memory savings, but slight degradation in model capacity and fine-grained attention patterns.

## Consequences
- **Positive**: Cuts KV cache memory footprint by 75% (from 32 KB/token to 8 KB/token across 16 layers), allowing higher concurrency during serving.
- **Negative**: Requires repeating or broadcasting Key and Value head tensors across query groups during the attention forward pass.

## Related Files
- [`src/model/attention.py`](../../src/model/attention.py)
- [`configs/model.gpu.yaml`](../../configs/model.gpu.yaml)
- [`tests/test_attention.py`](../../tests/test_attention.py)

