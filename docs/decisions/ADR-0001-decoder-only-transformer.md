# ADR-0001: Selection of Decoder-Only Causal Transformer Architecture

## Status
`ACCEPTED`

## Context
When designing `llm-engine`, we required a scalable neural network architecture capable of autoregressive generative tasks, instruction following, and conversational reasoning.

## Decision
We selected the **decoder-only causal Transformer** architecture with pre-layer normalization (Pre-Norm RMSNorm) and weight-tied embeddings.

## Alternatives Considered
- **Encoder-Decoder (T5 / BART)**: Good for translation and summarization, but double the parameter complexity and poor KV cache efficiency during open-ended dialogue generation.
- **Prefix-LM / Non-causal Enc**: More complex attention masks with no generative advantage over causal decoders.

## Consequences
- **Positive**: Single unified architecture for pretraining, fine-tuning, and autoregressive generation. KV cache during inference is clean and uniform.
- **Negative**: Unidirectional causal attention is slightly less efficient at pure bidirectional text embedding tasks than masked language models (BERT).

## Related Files
- [`src/model/gpt.py`](../../src/model/gpt.py)
- [`src/model/transformer_block.py`](../../src/model/transformer_block.py)
- [`configs/model.gpu.yaml`](../../configs/model.gpu.yaml)

