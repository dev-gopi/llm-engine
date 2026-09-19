# Tokenizer Specification (`docs/TOKENIZER.md`)

*Authoritative Source: [`src/tokenizer/bpe.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/tokenizer/bpe.py), [`src/tokenizer/trainer.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/tokenizer/trainer.py), [`configs/tokenizer.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/tokenizer.yaml)*

---

## 1. Algorithm & Design

The `llm-engine` tokenizer is a custom Byte-Pair Encoding (BPE) implementation featuring:
- **Regex Pre-Tokenization**: Splits words, numbers, punctuation, and whitespace into atomic segments matching GPT-style contractions.
- **UTF-8 Byte Fallback**: All 256 byte values are mapped to initial vocabulary tokens, ensuring that the tokenizer can encode any arbitrary binary or Unicode sequence without `<|unk|>` fallbacks.
- **Atomic Merge Ranks**: Merges are ordered strictly by frequency rank recorded during tokenizer training.

---

## 2. Vocabulary Sizes & Special Tokens

### 2.1 Reserved Special Tokens (Base 40,000 Vocab)

| Token String | Index | Role |
| :--- | :--- | :--- |
| `<|pad|>` | `0` | Padding token for batch alignment |
| `<|unk|>` | `1` | Unknown token fallback (rarely used due to byte fallback) |
| `<|bos|>` | `2` | Beginning of sequence |
| `<|eos|>` | `3` | End of sequence / Stop token |
| `<|mask|>` | `4` | Mask token for fill-in-the-middle or MLM |

### 2.2 Fine-Tuning Chat Extensions (42,000 Vocab)

The tokenizer extends the base 40,000 vocabulary with dedicated conversational role tokens in `data/tokenizer-finetuning/`:

| Token String | Assigned Range | Purpose |
| :--- | :--- | :--- |
| `<|system|>` | `40000+` | Delimits system instructions |
| `<|user|>` | `40000+` | Delimits user prompt turns |
| `<|assistant|>`| `40000+` | Delimits assistant response turns |
| `<|thinking|>` | `40000+` | Begins chain-of-thought reasoning block |
| `<|end|>` | `40000+` | Terminating turn token |

For an existing checkpoint tokenizer, add these delimiters with the append-only
agent-protocol extension rather than inserting them into a base tokenizer.
This preserves every existing token ID; `<|tool|>` is included alongside the
role, thinking, and end delimiters.

---

## 3. Training a New Tokenizer

To train a new BPE vocabulary on a custom corpus:

```bash
.venv/bin/python scripts/tokenize.py \
  --train \
  --data data/raw/corpus.txt \
  --vocab-size 40000 \
  --output data/tokenizer/
```

This generates:
- `vocab.json`: Token string to integer ID mapping.
- `merges.txt`: Ordered list of byte-pair merge operations.
- `metadata.json`: SHA256 checksums, training timestamps, and configuration.

---

## 4. Vocabulary Compatibility Invariants

1. **Deterministic Merges**: The order of merges in `merges.txt` must never be altered once a model checkpoint is trained.
2. **Append-Only Additions**: New special tokens or domain tokens must be added to the end of the vocabulary (`new_id >= old_vocab_size`).
3. **Validation**: Any vocabulary alteration must pass [`tests/test_vocabulary_compatibility.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/tests/test_vocabulary_compatibility.py).
