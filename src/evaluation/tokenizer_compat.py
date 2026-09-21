"""Tokenizer/checkpoint compatibility contract."""
from __future__ import annotations

def compatibility_report(tokenizer, checkpoint_metadata):
    expected=checkpoint_metadata.get("tokenizer_fingerprint")
    current=tokenizer.fingerprint
    compatible=list(getattr(tokenizer,"compatible_base_fingerprints",()))
    return {"current":current,"expected":expected,"vocab_size":tokenizer.vocab_size,"compatible":expected is None or expected==current or expected in compatible}

def assert_compatible(tokenizer, checkpoint_metadata):
    report=compatibility_report(tokenizer,checkpoint_metadata)
    if not report["compatible"]: raise ValueError("tokenizer/checkpoint compatibility gate failed")
    return report
