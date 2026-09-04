# Dataset catalog

This catalog documents every dataset used or retained by the repository,
including pretraining, supervised fine-tuning, preference optimization,
tokenizer discovery, evaluation, RAG knowledge, and optional profiles. The
authoritative paths and weights remain in `configs/`; provenance and review
state remain in each `data/processed/<name>/dataset-manifest.yaml`.

## Repository-wide source and reference index

| Local name/input | Used by | Upstream reference |
| --- | --- | --- |
| `tinystories` | V1/V2/v2 optional pretraining and tokenizer training | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |
| `wikitext_103` | V1/V2/v2 optional pretraining and tokenizer training | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) |
| `fineweb_edu` | Low-weight causal-LM SFT, tokenizer extension, and optional pretraining | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) |
| `code_pretraining` | Low-weight causal-LM SFT, tokenizer extension, and optional pretraining | [codeparrot/codeparrot-clean-valid](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid) |
| `ultrachat_200k` | V1/V2/v2 SFT and tokenizer discovery | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) |
| `v2_openassistant_en` | v2 SFT and tokenizer discovery | [agentlans/OpenAssistant-oasst](https://huggingface.co/datasets/agentlans/OpenAssistant-oasst) and original [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) |
| `helpsteer` | V1/V2/v2 SFT, tokenizer discovery, and preference derivation | [nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) |
| `openorca` | V1/V2/v2 SFT and tokenizer discovery | [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) |
| `gsm8k` | V1/V2/v2 mathematical SFT and evaluation | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| `v2_math_instruct` | v2 mathematical SFT and evaluation | [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) |
| `core_chat` | V1/V2/v2 SFT | [Local project-owned manifest](../data/processed/core_chat/dataset-manifest.yaml) |
| `code_instructions` | V1/V2/v2 coding SFT | [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) |
| `v2_code_feedback` | v2 coding SFT | [m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) |
| `general_qa` | V1/V2/v2 English SFT | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| `safety_alignment` | V1/V2/v2 safety SFT | [fwnlp/self-instruct-safety-alignment](https://huggingface.co/datasets/fwnlp/self-instruct-safety-alignment) |
| `writing_editing` | V1/V2/v2 writing SFT | [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) |
| `multilingual_bn_hi` | V1/V2/v2 Bengali SFT | [rishiraj/bengalichat](https://huggingface.co/datasets/rishiraj/bengalichat) |
| `multilingual_hi` | V1/V2/v2 Hindi SFT | [rishiraj/hindichat](https://huggingface.co/datasets/rishiraj/hindichat) |
| `tool_calling` | V1/V2/v2 structured tool-use SFT | [narrative-io/narrative-function-calling-v1](https://huggingface.co/datasets/narrative-io/narrative-function-calling-v1) |
| `emoji_chat` | V2/v2 SFT | [Local project-owned manifest](../data/processed/emoji_chat/dataset-manifest.yaml) |
| `bangla_qa` | V2/v2 Bengali SFT | [kamruzzaman-asif/bangla-instruction-dataset](https://huggingface.co/datasets/kamruzzaman-asif/bangla-instruction-dataset) (`QApair`) |
| `v2_bengali_news` | v2 Bengali SFT | [soketlabs/bhasha-sft](https://huggingface.co/datasets/soketlabs/bhasha-sft) (`aya_templated_bengali_news`) |
| `bangla_reading_qa` | V2/v2 Bengali SFT | [kamruzzaman-asif/bangla-instruction-dataset](https://huggingface.co/datasets/kamruzzaman-asif/bangla-instruction-dataset) (`RQA`) |
| `hindi_history_qa` | V2/v2 Hindi SFT | [kaifahmad/indian-history-hindi-QA-3.4k](https://huggingface.co/datasets/kaifahmad/indian-history-hindi-QA-3.4k) |
| `hinglish_chat` | V2/v2 Hinglish SFT | [DSMJ910/hinglish-instruct-10k](https://huggingface.co/datasets/DSMJ910/hinglish-instruct-10k) |
| `hindi_hinglish` | v2 Hindi/Hinglish reasoning SFT | [Subh775/formatted-hindi-hinglish-cot](https://huggingface.co/datasets/Subh775/formatted-hindi-hinglish-cot) |
| `v2_hindi_news` | v2 Hindi SFT | [soketlabs/bhasha-sft](https://huggingface.co/datasets/soketlabs/bhasha-sft) (`aya_templated_hindi_news`) |
| `code_alpaca` | V2/v2 coding SFT | [flwrlabs/code-alpaca-20k](https://huggingface.co/datasets/flwrlabs/code-alpaca-20k) |
| `preferences` | DPO and tokenizer discovery | Derived locally from [nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) |
| `recovery_sft` | Focused v2 response-quality recovery | Deterministically filtered from the governed chat, English, Bengali, Hindi, mathematics, and coding sources listed in `scripts/prepare_recovery_sft.py` |
| `dailydialog` | Optional retained conversation experiment; excluded from active profiles | [ConvLab/dailydialog](https://huggingface.co/datasets/ConvLab/dailydialog) |
| `wikipedia_en` | Tokenizer discovery, low-weight causal-LM fine-tuning, and local RAG | [English Wikipedia dumps](https://dumps.wikimedia.org/enwiki/) |
| `wikipedia_simple` | Tokenizer discovery, low-weight causal-LM fine-tuning, and local RAG | [Simple English Wikipedia dumps](https://dumps.wikimedia.org/simplewiki/) |
| `wikipedia_bn` | Tokenizer discovery, low-weight causal-LM fine-tuning, and local RAG | [Bengali Wikipedia dumps](https://dumps.wikimedia.org/bnwiki/) |
| `wikipedia_hi` | Tokenizer discovery, low-weight causal-LM fine-tuning, and local RAG | [Hindi Wikipedia dumps](https://dumps.wikimedia.org/hiwiki/) |

Packed profiles reference generated token-shard manifests built from these
same processed sources; shards are representations of a dataset, not new
upstream datasets. Evaluation profiles also reuse held-out splits or dedicated
case files and therefore do not create additional training sources.

## Active supervised fine-tuning mixture

`dataset_weights` control deterministic sampling targets for one training
epoch. They are not raw dataset-size percentages. Small capability datasets
can therefore retain a useful share without being overwhelmed by OpenOrca or
UltraChat. All weights total exactly 100%.

| Local name | Weight | Validation domain | Upstream source | Purpose | License/review status |
| --- | ---: | --- | --- | --- | --- |
| `ultrachat_200k` | 8% | Chat | `HuggingFaceH4/ultrachat_200k` | Multi-turn assistant behavior and broad instruction following | MIT reviewed; privacy unreviewed |
| `v2_openassistant_en` | 3% | Chat | `agentlans/OpenAssistant-oasst` | Human-style English conversations and multi-turn diversity | Apache-2.0 reviewed; privacy unreviewed |
| `helpsteer` | 8% | English | `nvidia/HelpSteer` | Helpful, correct, coherent response style with quality signals | Upstream license and privacy unreviewed |
| `openorca` | 8% | English | `Open-Orca/OpenOrca` | Broad English instructions, explanations, and general reasoning | Upstream license and privacy unreviewed |
| `gsm8k` | 13% | GSM8K | `openai/gsm8k` | Grade-school arithmetic word problems with worked reasoning | MIT reviewed; privacy unreviewed |
| `v2_math_instruct` | 5% | GSM8K | `TIGER-Lab/MathInstruct` | Broader mathematical fields and reasoning formats beyond GSM8K | Mixed source licenses and privacy unreviewed; Camel-Math and GSM8K-RFT excluded |
| `core_chat` | 2% | Chat | `gopi/core-chat-v1` | Preserve Gopi identity, greeting behavior, honesty, and response conventions | Project-owned and reviewed |
| `code_instructions` | 3% | Coding | `iamtarun/python_code_instructions_18k_alpaca` | Python generation, explanation, debugging, and implementation instructions | Upstream license and privacy unreviewed |
| `v2_code_feedback` | 4% | Coding | `m-a-p/CodeFeedback-Filtered-Instruction` | More complex coding requests and higher-difficulty solutions | Apache-2.0 label; upstream policy/license and privacy review incomplete |
| `general_qa` | 11% | English | `databricks/databricks-dolly-15k` | General questions, factual answers, summarization, and brainstorming | Upstream license and privacy unreviewed |
| `safety_alignment` | 2.5% | English | `fwnlp/self-instruct-safety-alignment` | Safer refusals and handling of harmful or inappropriate requests | Upstream license and privacy unreviewed |
| `writing_editing` | 1.5% | English | `HuggingFaceH4/no_robots` | Rewriting, editing, structured writing, and natural response quality | Upstream license and privacy unreviewed |
| `multilingual_bn_hi` | 5% | Bengali | `rishiraj/bengalichat` | General Bengali instruction following and conversation | Upstream license and privacy unreviewed |
| `multilingual_hi` | 4% | Hindi | `rishiraj/hindichat` | General Hindi instruction following and conversation | Upstream license and privacy unreviewed |
| `tool_calling` | 1.5% | Coding | `narrative-io/narrative-function-calling-v1` | Structured function/tool selection and argument generation | Upstream license and privacy unreviewed |
| `emoji_chat` | 0.5% | Chat | `gopi/emoji-chat-v1` | Emoji meaning and compact informal conversation | Project-owned and reviewed |
| `bangla_qa` | 2% | Bengali | `kamruzzaman-asif/bangla-instruction-dataset:QApair` | Bengali factual and instructional question answering | Apache-2.0 reviewed; privacy unreviewed |
| `v2_bengali_news` | 3% | Bengali | `soketlabs/bhasha-sft` | Bengali news summarization and concise information extraction | Mixed licenses and privacy unreviewed |
| `bangla_reading_qa` | 4% | Bengali | `kamruzzaman-asif/bangla-instruction-dataset:RQA` | Bengali reading comprehension and context-grounded answers | Apache-2.0 reviewed; privacy unreviewed |
| `hindi_history_qa` | 2% | Hindi | `kaifahmad/indian-history-hindi-QA-3.4k` | Hindi factual QA focused on Indian history | Apache-2.0 reviewed; privacy unreviewed |
| `hinglish_chat` | 1% | Hindi | `DSMJ910/hinglish-instruct-10k` | Code-switched Hindi/English conversational behavior | Apache-2.0 reviewed; privacy unreviewed |
| `hindi_hinglish` | 2% | Hindi | `Subh775/formatted-hindi-hinglish-cot` | Hindi/Hinglish step-by-step reasoning and instruction following | Upstream license and privacy unreviewed |
| `v2_hindi_news` | 3% | Hindi | `soketlabs/bhasha-sft` | Hindi news summarization and information extraction | Mixed licenses and privacy unreviewed |
| `code_alpaca` | 3% | Coding | `flwrlabs/code-alpaca-20k` | Additional programming instructions and response diversity | Apache-2.0 reviewed; privacy unreviewed |
| `fineweb_edu` | 2% | English | `HuggingFaceFW/fineweb-edu` | Low-weight educational-web causal-LM retention | ODC-By-1.0 reviewed; Common Crawl terms apply; privacy unreviewed |
| `code_pretraining` | 2% | Coding | `codeparrot/codeparrot-clean-valid` | Low-weight raw-code causal-LM retention | Mixed per-record licenses and privacy unreviewed |

### Training shares by validation domain

These totals group datasets according to `validation_domains`; they describe
the training mixture, not the independent validation aggregation weights.

| Domain | Training share | Primary capability |
| --- | ---: | --- |
| English | 31% | General QA, factual answers, educational text, writing, helpfulness, and safety |
| Chat | 11.5% | Multi-turn dialogue, identity, and informal conversation |
| Bengali | 14% | Bengali conversation, QA, reading comprehension, and news |
| Hindi | 12% | Hindi/Hinglish dialogue, facts, reasoning, and news |
| Coding | 13.5% | Code generation, raw-code retention, debugging, explanations, and tool calls |
| GSM8K | 18% | Arithmetic and broader mathematical reasoning |

Validation uses a deliberately different domain weighting: English 15%,
Bengali 14%, Hindi 12%, coding 10%, GSM8K 18%, and chat 31%. This makes chat
quality prominent in checkpoint selection without changing training sampling.

## Additional tokenizer and mixed-objective inputs

`configs/tokenizer.yaml` scans every active SFT training file. Its `extension`
section discovers up to 2,000 expensive, frequent tokens from FineWeb-Edu and
CodeParrot and writes an append-only artifact to `data/tokenizer-finetuning`.
The active fine-tuning profile uses mixed-objective inputs conservatively:
preference records train only on the chosen response,
while Wikipedia records receive ordinary causal-language-model loss.

| Input | Purpose | Used for SFT? |
| --- | --- | --- |
| `data/processed/preferences/train.jsonl` | Chosen-response SFT before later pairwise DPO | Yes, 2% |
| `data/rag/wikipedia/wikipedia-en.jsonl` | English knowledge vocabulary and named entities | Yes, 1% |
| `data/rag/wikipedia/wikipedia-simple.jsonl` | Simpler English vocabulary and phrasing | Yes, 1% |
| `data/rag/wikipedia/wikipedia-bn.jsonl` | Bengali knowledge vocabulary and script coverage | Yes, 1% |
| `data/rag/wikipedia/wikipedia-hi.jsonl` | Hindi knowledge vocabulary and Devanagari coverage | Yes, 1% |
| `data/processed/fineweb_edu/train.jsonl` | Educational-web causal-LM retention | Yes, 2% |
| `data/processed/code_pretraining/train.jsonl` | Raw-code causal-LM retention | Yes, 2% |

The remaining 92% of the mixture stays instruction/preference-oriented, preventing raw
knowledge text from dominating assistant behavior.

## Optional continued-pretraining inputs

The direct-growth workflow may skip this stage. If
`configs/pretraining.gpu.yaml` is used, its separate causal-language-
model mixture contains:

| Local name | Weight | Purpose |
| --- | ---: | --- |
| `tinystories` | 10% | Simple narrative structure, grammar, and coherent short-form text |
| `wikitext_103` | 90% | General encyclopedic language modeling and broader factual prose |

These are plain-text pretraining datasets, not assistant-response SFT data.

### Downloaded future-pretraining subsets

Two bounded plain-text datasets are stored locally for diversified training.
They are not referenced by the active 90% WikiText / 10% TinyStories
pretraining configuration. In GPU SFT they contribute ordinary causal-LM loss
at 2% each; do not use the changed sampler to resume an older SFT run.

| Local name | Train | Validation | Test | Processed size | Intended role | Governance status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `fineweb_edu` | 533,797 | 2,000 | 2,011 | about 2.5 GiB | Clean educational web pages, articles, and general prose | ODC-By-1.0 reviewed; Common Crawl terms and attribution apply; privacy review incomplete |
| `code_pretraining` | 79,822 | 1,000 | 1,000 | about 803 MiB | Raw Python source-code language modelling | Mixed per-record licenses and privacy review remain incomplete |

Files are under `data/processed/fineweb_edu/` and
`data/processed/code_pretraining/`. Their raw Parquet download caches were
deleted after conversion. Both have `dataset-manifest.yaml` provenance files.
Do not claim commercial readiness for `code_pretraining` until every retained
record's upstream license and privacy status have been reviewed.

## Governance and quality requirements

Training currently uses `dataset_governance.policy: warn` and
`commercial_use: false`. Warnings do not mean a dataset is approved. Before
publishing weights or enabling commercial use:

1. Resolve every `license_unreviewed` and `privacy_unreviewed` finding.
2. Retain source attribution and required notices.
3. Deduplicate train and validation content across overlapping sources.
4. Sample and manually inspect answers for correctness, safety, language
   quality, truncation, templated repetition, and leaked personal information.
5. Keep fixed, unseen evaluation cases outside all tokenizer-training and
   model-training inputs.

Run the machine-readable audit with:

```bash
.venv/bin/python scripts/audit_datasets.py \
  --training-config configs/finetuning.gpu.yaml \
  --stage sft
```

The audit intentionally exits unsuccessfully while required reviews remain
incomplete, even though the training profile's `warn` policy permits an
educational local run.

## JSONL quality gate (no token shards)

Use `scripts/clean_jsonl_corpus.py` followed by
`scripts/pack_jsonl_corpus.py`. Passing the automated checks does not replace
manual license, privacy, or sample-quality review.

| Check | Implementation/evidence |
| --- | --- |
| 1. Exact duplicates | Bounded canonical-content digest index; counts are recorded in `.audit.json`. |
| 2. Near-duplicates/templates | Bounded SimHash index with configurable Hamming distance. Template quality still needs sampling. |
| 3. Train/validation overlap | Clean held-out data first and pass it with `--exclude` when cleaning train. |
| 4. Broken markup/Unicode/empty passages | NFKC and control-character cleanup plus empty, length, printable, and alphanumeric filters. |
| 5. Secrets/PII/generated text | Common credentials, email, phone, IP, address, government and financial IDs are redacted. Generated-text quality needs manual review. |
| 6. Language detection | Every accepted row receives a `language` field and aggregate counts are audited. |
| 7. Length/truncation | Exact tokenizer lengths and the percentage exceeding 512 tokens are recorded. |
| 8. Packing | Short documents are combined into bounded JSONL records instead of individually padded. |
| 9. Document boundaries | Every document or long-document chunk ends in the EOS special token; the loader does not append a duplicate EOS to `prepacked` rows. |
| 10. Provenance/license/preparation | Source manifests plus `.audit.json` and `.packing.json` record provenance and preparation evidence. License/privacy approvals remain human decisions. |

Generated `data/cleaned/` outputs are reproducible local artifacts and are not
committed. The cleaned pretraining configuration must only be started after all
referenced outputs exist and their reports have been reviewed.
