# WikiText-103 Raw processed corpus

Article-level JSONL derived from the local `wikitext-103-raw-v1` Parquet
shards. Empty rows are removed, top-level headings delimit documents, and the
WikiText `@-@`, `@,@`, and `@.@` artifacts are normalized.

- Source: `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`
- License: Creative Commons Attribution-ShareAlike 4.0
- Regenerate: `.venv/bin/python scripts/prepare_wikitext.py`
- Outputs: `train.jsonl`, `validation.jsonl`, and `test.jsonl`

Generated JSONL files are excluded from Git.
