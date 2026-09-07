# Dataset formats

Processed datasets are UTF-8 JSON Lines files: one valid JSON object per line.
Keep train, validation, and test splits disjoint. Stable `id` and `source`
fields are strongly recommended for provenance and debugging.

## Pretraining text

The loader accepts `text`, `utterance`, or `content`. `text` is preferred:

```json
{"id":"wiki-0001","source":"wikitext_103","text":"Example document text."}
```

Each record should be a coherent document or passage. Do not mix validation or
test records into training. Remove duplicates, corrupted text, secrets, and
disallowed personal information before training.

## Supervised fine-tuning chat

Use ordered `messages` with supported roles. A typical record is:

```json
{"id":"chat-0001","source":"my_sft","messages":[{"role":"user","content":"বাংলাদেশের রাজধানী কী?"},{"role":"assistant","content":"বাংলাদেশের রাজধানী ঢাকা।"}]}
```

Supported roles are `system`, `user`, and `assistant`. Keep turns in dialogue
order and include a useful assistant response. During chat training, assistant
tokens are the intended response targets; malformed or empty conversations
should be removed.

## DPO preferences

Every record must contain a prompt, a preferred answer, and a rejected answer:

```json
{"id":"pref-0001","source":"human_review","prompt":"Explain RAM simply.","chosen":"RAM is short-term working memory used by running programs.","rejected":"RAM is permanent disk storage."}
```

`prompt`, `chosen`, and `rejected` must be non-empty, and the two answers must
not be identical. Keep all candidates for the same prompt in one split to avoid
train/validation leakage. Preference quality matters more than raw pair count.

## Domain evaluation

Capability evaluation accepts a category, prompt, and one or more expected
answer fragments:

```json
{"category":"english","prompt":"What is the capital city of France?","expected":["Paris"]}
```

Use separate categories for English, Bengali, Hindi, coding, mathematics,
reasoning, and chat. Evaluation examples must not occur in training data.

## Split layout

The conventional processed layout is:

```text
data/processed/<dataset>/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── dataset-manifest.yaml
```

A source dataset may lack an official test split. In that case create a
deterministic disjoint split and record the method and seed in the manifest.
Never copy identical examples into several splits merely to make files exist.

## Dataset manifest

Record at least the source name, source version or revision, split counts,
license, allowed use, preparation method, and privacy/governance review status.
Dataset licensing is independent of this repository's software licensing.

## Validation commands

Check all paths referenced by an SFT configuration:

```bash
.venv/bin/python -c "from pathlib import Path; from utils.config import load_yaml; c=load_yaml(Path('configs/finetuning.gpu.yaml')); missing=[p for p in c['train_files']+c['validation_files'] if not Path(p).is_file()]; print('Missing:', missing or 'none')"
```

Inspect a JSONL file without loading it all into memory:

```bash
head -n 2 data/processed/preferences/train.jsonl
wc -l data/processed/preferences/*.jsonl
```

Use `scripts/prepare_hf_dataset.py --help` for the supported Hugging Face
download and normalization workflow. Review each dataset card and license
before downloading or using it.

## Web and Hugging Face corpus pipeline

`scripts/prepare_corpus.py` is a configuration-driven, DataTrove-compatible
pretraining path. It streams Hugging Face datasets through the `datasets`
library or reads extracted web JSONL, then performs text extraction, quality
filtering, language selection, PII redaction, exact/near deduplication,
statistics, and JSONL sharding. Start from `configs/corpus.web.yaml`:

```bash
python -m pip install -e '.[data]'
python scripts/prepare_corpus.py --config configs/corpus.web.yaml
python scripts/build_token_shards.py 'data/processed/web-corpus/shard-*.jsonl' \
  --tokenizer data/tokenizer --output data/tokens/web-corpus
```

For WARC/Common Crawl-scale processing, use DataTrove's WARC reader with the
same source/extraction/filter/output settings; the checked-in script keeps a
portable JSONL reader for local and testable runs. Always record the dataset
card, license, revision, and filtering policy in the generated manifest.
