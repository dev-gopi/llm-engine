# UltraChat 200k working subset

- Source: `HuggingFaceH4/ultrachat_200k`
- Source URL: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- License: MIT
- Language: English
- Purpose: supervised fine-tuning for a general conversational assistant
- Format: one JSON object per line with `id`, `bot_name`, `messages`, and `source`

The working subset is generated deterministically by `scripts/prepare_ultrachat.py`.
The validation and test sets are disjoint slices of the upstream `test_sft` split.
Raw Parquet and generated JSONL files are excluded from Git.
