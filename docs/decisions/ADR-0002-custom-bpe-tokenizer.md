# ADR-0002: Custom BPE Tokenizer with Byte Fallback

## Status
`ACCEPTED`

## Context
Standard off-the-shelf tokenizers often introduce massive dependency footprints, opaque C++ bindings, and fixed vocabularies that cannot easily be extended with custom control tokens without breaking checkpoint compatibility.

## Decision
We implemented a **pure-Python Byte-Pair Encoding (BPE) tokenizer** with UTF-8 byte fallback and append-only vocabulary extension rules.

## Alternatives Considered
- **HuggingFace `tokenizers` library**: Extremely fast Rust backend, but adds heavy dependency chain and difficult-to-control append-only weight table resizings.
- **SentencePiece**: Requires native compilation, complex C++ dependency management, and lacks simple Python-level byte rank inspection.

## Consequences
- **Positive**: Zero external C++ or Rust dependencies; deterministic merge rank ordering; explicit append-only vocabulary extension logic tested by unit suites.
- **Negative**: Training tokenizers from scratch in Python is slower than optimized Rust tokenizers on massive multi-gigabyte corpora.

## Related Files
- [`src/tokenizer/bpe.py`](../../src/tokenizer/bpe.py)
- [`src/tokenizer/trainer.py`](../../src/tokenizer/trainer.py)
- [`tests/test_vocabulary_compatibility.py`](../../tests/test_vocabulary_compatibility.py)

