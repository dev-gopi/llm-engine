# ADR-0011: Parameter-Efficient Fine-Tuning via LoRA and QLoRA

## Status
`ACCEPTED`

## Context
As models grow to 1B, 7B, and 30B parameters, full parameter fine-tuning becomes impossible on consumer or workstation GPUs because the AdamW optimizer requires 16 bytes per parameter (4x weight size).

## Decision
We implemented native **Low-Rank Adaptation (LoRA)** in [`src/training/peft.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/peft.py):
1. Freezes all base model weights ($W_{\text{base}}$) to eliminate optimizer memory for base layers.
2. Injects trainable low-rank decomposition matrices $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ into attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`).
3. Adapts $B$ with zero-initialization so the fine-tuning start matches the base model exactly.

## Alternatives Considered
- **Prompt Tuning / Prefix Tuning**: Less expressive capacity; consumes precious context window tokens.
- **Full Parameter SFT**: Requires multi-GPU clusters for models larger than 1B parameters.

## Consequences
- **Positive**: Reduces trainable parameters by >99%; allows fine-tuning 7B-class models on single 24 GB or 16 GB GPUs.
- **Negative**: Adds minor inference latency unless LoRA matrices are merged back into base weights prior to export.

## Related Files
- [`src/training/peft.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/peft.py)
- [`tests/test_peft.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/tests/test_peft.py)
- [`docs/SCALING.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/SCALING.md)

