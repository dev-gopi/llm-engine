# DailyDialog working subset

- Source: `ConvLab/dailydialog`
- Source URL: https://huggingface.co/datasets/ConvLab/dailydialog
- License: CC BY-NC-SA 4.0
- Language: English
- Purpose: daily-life conversational training
- Format: JSON array containing `bot_name`, user/assistant message pairs, and source metadata
- Maximum pairs: 2,000

This dataset is licensed for non-commercial use with attribution and share-alike
requirements. Review the license before using a model trained on it commercially.

The working subset is generated deterministically by `scripts/prepare_dailydialog.py`.
Run `scripts/split_dailydialog.py` afterward to create dialogue-grouped JSONL
splits. With the current seed and 10% validation ratio, it creates 1,803 train
records and 197 validation records; pairs from one source dialogue never cross
the split boundary. Raw archives and generated dataset files are excluded from Git.
