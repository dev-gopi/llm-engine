# Architecture Decision Records (`docs/decisions/`)

This directory contains the Architecture Decision Records (ADRs) for `llm-engine`. Each record documents an architectural decision, its context, consequences, and alternatives considered.

---

## Index of Decisions

| ADR ID | Title | Status | Date |
| :--- | :--- | :--- | :--- |
| **[ADR-0001](ADR-0001-decoder-only-transformer.md)** | Selection of Decoder-Only Causal Transformer Architecture | `ACCEPTED` | 2026-08 |
| **[ADR-0002](ADR-0002-custom-bpe-tokenizer.md)** | Custom BPE Tokenizer with Byte Fallback | `ACCEPTED` | 2026-08 |
| **[ADR-0003](ADR-0003-rotary-position-embeddings.md)** | Adoption of Rotary Position Embeddings (RoPE) | `ACCEPTED` | 2026-08 |
| **[ADR-0004](ADR-0004-grouped-query-attention.md)** | Adoption of Grouped-Query Attention (GQA) | `ACCEPTED` | 2026-08 |
| **[ADR-0005](ADR-0005-hardware-constrained-training-strategy.md)**| Training Strategy for 4 GB Consumer GPUs | `ACCEPTED` | 2026-09 |
| **[ADR-0006](ADR-0006-hybrid-attention-long-context-target.md)**| Target Long-Context Architecture: Hybrid Attention & RoPE Scaling | `PROPOSED` | 2026-09 |
| **[ADR-0007](ADR-0007-agentic-tool-calling-and-mcp.md)**| Tool Calling via Model Context Protocol (MCP) and JSON Schemas | `ACCEPTED` | 2026-09 |
| **[ADR-0008](ADR-0008-progressive-context-documentation-system.md)**| Progressive Context Documentation System for AI Coding Agents | `ACCEPTED` | 2026-09 |
| **[ADR-0009](ADR-0009-distributed-training-fsdp-and-tensor-parallelism.md)**| Distributed Training Strategy: FSDP and Tensor Parallelism | `ACCEPTED` | 2026-09 |
| **[ADR-0010](ADR-0010-model-growth-and-progressive-depth-expansion.md)**| Model Growth and Progressive Depth Expansion | `ACCEPTED` | 2026-09 |
| **[ADR-0011](ADR-0011-parameter-efficient-fine-tuning-peft-lora.md)**| Parameter-Efficient Fine-Tuning via LoRA and QLoRA | `ACCEPTED` | 2026-09 |
| **[ADR-0012](ADR-0012-non-invasive-vision-language-integration.md)**| Non-Invasive Vision-Language Integration | `ACCEPTED` | 2026-09 |
| **[ADR-0013](ADR-0013-compact-latent-diffusion-architecture.md)**| Compact Latent Diffusion Architecture for Consumer Hardware | `ACCEPTED` | 2026-09 |

---

## ADR Template
Every ADR should follow this format:
```markdown
# ADR-XXXX: Decision Title

## Status
[PROPOSED | ACCEPTED | DEPRECATED | SUPERSEDED]

## Context
Background and problem statement.

## Decision
The architectural choice made.

## Alternatives Considered
Other options evaluated and why they were rejected.

## Consequences
Positive and negative trade-offs.

## Related Files
Authoritative files affected by this decision.
```

