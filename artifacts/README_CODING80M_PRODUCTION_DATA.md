# GopiCoder-80M — Production Dataset Acquisition & Filtering Guide

> **Goal:** build a high-quality, production-grade dataset for the ~81.8M-parameter `gopi-coder-80m` model.
>
> **Important:** no dataset or filtering pipeline can guarantee **100% factual or code accuracy**. The realistic production goal is to make every training record traceable, license-aware, deduplicated, secret/PII-cleaned, syntax-checked where possible, test-verified where tests exist, and benchmark-decontaminated. Quality should be measured with acceptance gates, not claimed as 100%.

---

## 1. Recommended total training budget

For an ~80M dense decoder model, a strong first production target is **~1.6B–2.0B training tokens** before SFT. This is roughly 20–25 tokens per parameter and is a sensible starting scale for this model size.

### Recommended pretraining mix

| Source type | Target tokens | Share | Purpose |
|---|---:|---:|---|
| High-quality permissive source code | 950M | 52.8% | Core coding ability |
| FIM transformations of accepted source code | 300M | 16.7% | Infill/editing ability |
| Code + documentation pairs | 150M | 8.3% | Natural-language-to-code alignment |
| GitHub issues/commits/notebooks | 150M | 8.3% | Debugging, change reasoning, repo work |
| Technical/educational English | 250M | 13.9% | Explanation, reasoning, API understanding |
| **Total** | **1.80B** | **100%** | |

Do not blindly download 1.8B tokens and train. Download more raw material than you need, filter aggressively, then sample the accepted pool to the target.

---

# 2. Primary code datasets

## 2.1 The Stack v2 — main code source

**Dataset:** `bigcode/the-stack-v2`

Hugging Face:
https://huggingface.co/datasets/bigcode/the-stack-v2

Recommended training IDs:
https://huggingface.co/datasets/bigcode/the-stack-v2-train-full-ids

The Stack v2 contains source-code metadata for more than 3 billion files across 600+ programming/markup languages. The current dataset card reports about 67.5 TB full data, 32.1 TB deduplicated, and roughly 900B tokens in the filtered full training set.

### Recommended use for GopiCoder-80M

Do **not** use all languages. For an 80M model, concentrate capacity.

Suggested accepted-token budget:

| Language/category | Target tokens |
|---|---:|
| Python | 190M |
| TypeScript | 110M |
| JavaScript | 100M |
| Java | 100M |
| Go | 90M |
| C++ | 85M |
| C | 55M |
| Rust | 60M |
| SQL | 45M |
| Shell/Bash | 35M |
| HTML/CSS/JSON/YAML/TOML/Dockerfile | 50M |
| Other high-quality code | 30M |
| **Total** | **950M** |

### License rule

The Stack v2 contains code under many original repository licenses. Preserve provenance and comply with the original license for every retained item. For a commercial/product training pipeline, create an explicit allowlist approved by your legal requirements.

A conservative permissive-license allowlist can start with:

- MIT
- Apache-2.0
- BSD-2-Clause
- BSD-3-Clause
- ISC
- 0BSD
- Unlicense

Do **not** assume every file in The Stack v2 is commercially interchangeable just because the dataset can be accessed.

### Important access note

The Stack v2 card states that bulk code-content access requires agreement with Software Heritage / Inria and that the released Hugging Face records may contain Software Heritage IDs rather than file contents.

---

## 2.2 CodeParrot Clean — Python-focused source

Hugging Face:
https://huggingface.co/datasets/codeparrot/codeparrot-clean

Train:
https://huggingface.co/datasets/codeparrot/codeparrot-clean-train

Validation:
https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid

The cleaned dataset contains roughly 5.36M Python files and was created with exact deduplication plus filters for line length, alphanumeric ratio, and auto-generated code.

### Recommended quantity

Use **50M–120M tokens** as a Python quality booster, but only after:

1. validating repository/license metadata,
2. re-running near-deduplication against your entire code pool,
3. secret/PII scanning,
4. removing benchmark contamination,
5. removing generated/vendor/minified files.

Do not add it on top of overlapping The Stack records without cross-dataset deduplication.

---

## 2.3 CodeSearchNet — natural-language/code pairs

Official repository:
https://github.com/github/CodeSearchNet

Hugging Face discovery:
https://huggingface.co/datasets?search=CodeSearchNet

Useful HF mirror:
https://huggingface.co/datasets/code-search-net/code_search_net

CodeSearchNet contains roughly **2 million comment/code pairs** across Python, JavaScript, Ruby, Go, Java and PHP.

### Recommended quantity

After license and quality filtering:

- **300K–800K accepted pairs**
- roughly **80M–150M tokens**
- emphasize Python, JavaScript/TypeScript-like patterns, Java, Go

Use these examples for:
- docstring → function learning
- function → explanation learning
- API semantics
- code search/retrieval-style alignment

### Important license note

The original examples come from open-source repositories with repository-specific licenses. Preserve provenance and do not assume one blanket license applies to all examples.

---

# 3. Real software-development data

## 3.1 StarCoder training data — issues, commits, notebooks

Hugging Face:
https://huggingface.co/datasets/bigcode/starcoderdata

The dataset card reports approximately:
- 783 GB code
- 54 GB GitHub Issues
- 13 GB Jupyter notebook scripts/text-code pairs
- 32 GB GitHub commits
- about 250B tokens total

### Recommended use

For an 80M model, do **not** ingest the entire collection.

Sample approximately **150M accepted tokens** from:
- `github-issues-filtered-structured`
- `git-commits-cleaned`
- `jupyter-structured-clean-dedup`

Recommended accepted mix:

| Data | Tokens |
|---|---:|
| GitHub issues | 55M |
| commits / commit messages | 55M |
| notebooks | 40M |

Use this data to teach:
- bug descriptions
- code-change intent
- commit-message semantics
- issue → implementation relationships
- natural-language + executable code context

Again, retain original provenance/license information.

---

## 3.2 GitHub public repository data via BigQuery

Google BigQuery public datasets:
https://cloud.google.com/bigquery/public-data

GitHub BigQuery tutorial:
https://codelabs.developers.google.com/codelabs/bigquery-github

This is useful when you want to create your **own curated real-data corpus** rather than relying entirely on prebuilt datasets.

### Recommended repository-selection policy

Only ingest repositories that satisfy all of the following:

- explicit recognized license
- not archived
- not a fork unless needed for provenance
- meaningful source tree
- recent or historically significant maintained project
- enough code to avoid toy/template repos
- no generated/vendor-only repository
- no malware/exploit repository unless you are intentionally building a security model and have separate safety controls

For code-agent training, real issue/PR/commit data from selected permissive repositories is often more valuable than adding another huge block of random code.

---

# 4. Technical English / reasoning data

## 4.1 FineWeb-Edu

Hugging Face:
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

FineWeb-Edu contains approximately **1.3T tokens** of educational web content and is distributed under ODC-By at the dataset level; downstream users must still consider rights in the underlying content.

### Recommended quantity

Use only **200M–300M tokens** for this 80M coding model.

Do not let general web text dominate the model.

Recommended selection topics:
- computer science
- algorithms and data structures
- operating systems
- networking
- databases
- distributed systems
- software engineering
- programming languages
- security fundamentals
- mathematics useful for programmers
- technical writing/documentation

Reject:
- SEO pages
- product spam
- scraped forum fragments
- shallow listicles
- duplicate tutorials
- low-information boilerplate
- pages dominated by navigation/legal text

---

# 5. SFT / instruction-tuning datasets

Pretraining teaches code and language. SFT teaches the model how to behave as a coding assistant.

Target **100K–180K final SFT examples**, not one million noisy examples.

---

## 5.1 OpenCoder SFT Stage 2

Hugging Face:
https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2

The dataset includes an `educational_instruct` subset of roughly **118K rows** with instruction, code, and test-case fields. The dataset card describes compiler-validated educational examples.

### Recommended quantity

Take **40K–70K examples** after your own filtering.

Prioritize samples that:
- have executable tests,
- compile,
- have unambiguous instructions,
- are not trivial duplicates,
- fit your target languages,
- have a concise correct solution.

---

## 5.2 Magicoder OSS Instruct

Official organization dataset:
https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K

Approximately **75K examples**, MIT dataset license.

### Recommended quantity

Keep **15K–30K** after compilation/test validation.

Because this data was synthetically generated, treat it as **instruction data**, not as unquestionable ground truth.

---

## 5.3 Glaive Code Assistant

Hugging Face:
https://huggingface.co/datasets/glaiveai/glaive-code-assistant

Approximately **136K rows**, Apache-2.0 dataset license.

A larger v3 also exists:
https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3

### Recommended quantity

Use **20K–40K** highly filtered samples rather than blindly ingesting everything.

The original dataset is synthetic, so verify answers through:
- compilation,
- unit tests,
- static analysis,
- reference execution where possible.

---

## 5.4 CodeFeedback Filtered Instruction

Dataset card:
https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction

The card describes a filtered collection of approximately **156K high-complexity code instructions**, selected from an initial ~287K pool.

### Recommended quantity

Use **10K–25K** instructions as prompts/seeds, preferably regenerate or verify solutions yourself.

The dataset card warns that some source data was generated by OpenAI models, so review usage terms before adopting it in a commercial pipeline.

---

# 6. Algorithm/problem-solving data

## 6.1 Codeforces dataset

Hugging Face:
https://huggingface.co/datasets/open-r1/codeforces

The dataset currently contains roughly **29K displayed rows / ~35K estimated rows** and provides problem statements, tests and related metadata.

### Recommended use

For an 80M model:
- retain **8K–15K** problems for training,
- use multiple correct solutions only after deduplication,
- prefer easier and medium problems,
- heavily cap very long competitive-programming solutions.

Most important: execute candidate solutions against tests before accepting them.

Use these examples for:
- algorithm selection
- edge-case handling
- input/output formatting
- test-based correctness

Do **not** turn the entire model into a competitive-programming specialist.

---

# 7. FIM (Fill-in-the-Middle) data

Do not download a separate low-quality FIM dataset unless necessary.

Create FIM examples from your **already accepted, licensed, cleaned code**.

Target: **300M FIM training tokens**.

Recommended transformation:

```text
<fim_prefix>
def parse_user(data):
    name = data.get("name")
<fim_suffix>
    return {"name": name, "active": active}
<fim_middle>
    active = bool(data.get("active", False))
```

### FIM generation rules

- choose split points at syntax-aware boundaries when possible,
- middle span should usually be 5–35% of the file/window,
- do not cut UTF-8 sequences,
- avoid creating examples containing only braces/imports,
- preserve language metadata,
- preserve source/provenance hash,
- generate at most a small number of FIM variants per original sample.

---

# 8. Real coding-agent trajectories

This is the most important data for making the model an **agent**, not only a code autocomplete model.

Target **20K–40K high-quality trajectories**.

The best source is your own curated collection of real, permissively licensed repositories.

A trajectory should look like:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "The pagination endpoint returns duplicate records. Find the bug and fix it."
    },
    {
      "role": "assistant",
      "tool_calls": [
        {"name": "search_code", "arguments": {"query": "pagination"}}
      ]
    },
    {
      "role": "tool",
      "content": "src/api/list_users.py:41 ..."
    },
    {
      "role": "assistant",
      "tool_calls": [
        {"name": "read_file", "arguments": {"path": "src/api/list_users.py"}}
      ]
    },
    {
      "role": "tool",
      "content": "..."
    },
    {
      "role": "assistant",
      "tool_calls": [
        {"name": "apply_patch", "arguments": {"patch": "..."}}
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {"name": "run_tests", "arguments": {"target": "tests/test_list_users.py"}}
      ]
    },
    {
      "role": "tool",
      "content": "12 passed"
    },
    {
      "role": "assistant",
      "content": "Fixed the pagination offset bug and verified the relevant tests pass."
    }
  ],
  "verification": {
    "tests_passed": true
  }
}
```

### Only accept an agent trajectory when

- repository license is allowed,
- task is reproducible,
- initial state is pinned by commit SHA,
- patch applies cleanly,
- tests fail before the fix when applicable,
- tests pass after the fix,
- no unrelated files are modified,
- no secret/token/private data appears,
- tool calls are syntactically valid,
- final explanation matches the actual patch.

This is substantially higher quality than generating thousands of fake tool calls with no executable repository.

---

# 9. Preference / DPO data

Target **20K–40K preference pairs**.

Do not create preference labels only from another LLM's opinion.

For coding, use executable evidence.

Example:

```json
{
  "prompt": "Implement parse_port(value) and reject values outside 1..65535.",
  "chosen": "def parse_port(value): ...",
  "rejected": "def parse_port(value): ...",
  "verification": {
    "chosen_tests_passed": 18,
    "chosen_tests_failed": 0,
    "rejected_tests_passed": 14,
    "rejected_tests_failed": 4
  }
}
```

Preferred ranking signals:

1. all tests pass,
2. compiles/parses,
3. no new security issue,
4. preserves API contract,
5. simpler patch,
6. static analysis clean,
7. style/clarity.

---

# 10. Datasets that must remain evaluation-only

Do not put your benchmark test cases into pretraining/SFT/DPO.

Keep evaluation data isolated and hash-block it from training.

Recommended evaluation suites include:

- HumanEval
- MBPP
- SWE-bench / SWE-bench Verified
- LiveCodeBench
- your own private repository tasks
- your own hidden unit-test set

If benchmark examples leak into training, the score is not a reliable measure of generalization.

---

# 11. Production filtering pipeline

Use this sequence.

```text
RAW DATA
   ↓
license/provenance gate
   ↓
file-type & language identification
   ↓
binary/vendor/generated/minified removal
   ↓
secret + PII scanning
   ↓
basic text/code quality filters
   ↓
parser/compiler validation
   ↓
exact deduplication
   ↓
near-duplicate detection
   ↓
cross-dataset deduplication
   ↓
benchmark decontamination
   ↓
quality scoring
   ↓
token-length/window filtering
   ↓
train/validation split by repository
   ↓
immutable manifest + hashes
   ↓
TRAINING DATA
```

---

# 12. Step-by-step filtering rules

## 12.1 Provenance and license

Every record should have at least:

```json
{
  "source": "the-stack-v2",
  "repository": "owner/repo",
  "revision": "commit-sha",
  "path": "src/file.py",
  "license_spdx": "Apache-2.0",
  "source_url": "...",
  "content_sha256": "..."
}
```

Reject when:
- source is unknown,
- license is missing or incompatible with your policy,
- repository/path cannot be traced,
- deletion/opt-out policy requires removal.

Maintain a deletion ledger so a source can be removed later and your derived shards can be rebuilt.

---

## 12.2 Remove generated/vendor/minified code

Reject common paths such as:

```text
node_modules/
vendor/
dist/
build/
target/
coverage/
.cache/
.next/
third_party/
__pycache__/
```

Reject obvious generated files:
- `.min.js`
- `.map`
- generated protobuf/OpenAPI outputs unless intentionally training them
- lockfiles as dominant training material
- vendored libraries
- machine-generated bundles

Keep small amounts of:
- Dockerfiles
- YAML
- JSON
- Terraform
- CI configuration

because coding agents need to edit these.

---

## 12.3 Size and structure rules

Reasonable default code-file filters:

```text
minimum characters:       80
maximum raw file size:    256 KB for normal sampling
maximum line length:      1,000
average line length:      < 140
alphanumeric fraction:    >= 0.20
maximum repeated-line %:  20%
```

Large files can be chunked syntactically instead of discarded.

---

## 12.4 Syntax / parser validation

Where parsers are available:

- Python → `ast.parse`
- JavaScript / TypeScript → tree-sitter / TypeScript parser
- Go → `go/parser` or `gofmt`
- Rust → parser / `cargo check` when project context is available
- Java → tree-sitter / javac when dependency context exists
- C/C++ → tree-sitter / clang syntax where practical

Suggested acceptance target:

```text
>= 98% of SFT code must parse/compile in an appropriate validation environment
100% of examples claiming "tests pass" must actually pass the recorded tests
```

Raw pretraining files may include partial snippets, so syntax requirements can be slightly looser than SFT.

---

## 12.5 Secret detection

Reject or redact any content matching real credentials:

- AWS access keys
- GitHub tokens
- private keys
- OAuth secrets
- JWT signing secrets
- database passwords
- connection strings with credentials
- `.env` secrets

Use multiple scanners, for example:
- Gitleaks
- TruffleHog
- custom high-entropy detector
- provider-specific token regexes

Do not replace secrets with realistic-looking fake credentials. Use stable placeholders such as:

```text
<REDACTED_API_KEY>
<REDACTED_PASSWORD>
```

---

## 12.6 PII filtering

Remove or redact:
- personal email addresses when not necessary for code semantics,
- phone numbers,
- private addresses,
- credentials,
- accidentally committed personal data.

Preserve legitimate documentation examples only when they clearly use placeholders/example domains.

---

## 12.7 Exact deduplication

Canonicalize carefully:

1. normalize line endings,
2. trim trailing whitespace,
3. optionally normalize repeated blank lines,
4. hash normalized content.

Use SHA-256.

Do **not** remove meaningful indentation.

---

## 12.8 Near deduplication

Use MinHash/LSH or another token-shingle method.

Recommended starting point:

```text
shingle size: 5–10 tokens
Jaccard threshold: 0.80–0.90
```

Deduplicate:
- inside each source,
- across languages when files are generated copies,
- across all datasets,
- between pretraining and SFT,
- between training and validation.

Keep the highest-quality/provenance-rich version.

---

## 12.9 Benchmark decontamination

Create normalized hashes and fuzzy fingerprints for evaluation problems.

Reject training examples that strongly overlap with:
- benchmark prompt,
- canonical solution,
- unit tests,
- distinctive problem statement.

Use:
- exact substring matching,
- token n-gram overlap,
- MinHash,
- AST fingerprints for code.

Run contamination scanning **after all datasets are merged**, not separately.

---

# 13. Quality scoring

Give every candidate a score from 0–100.

Example scoring:

| Signal | Points |
|---|---:|
| explicit approved license/provenance | 10 |
| parses/compiles | 15 |
| repository quality | 10 |
| meaningful identifiers | 8 |
| comments/docstrings useful | 7 |
| non-generated | 10 |
| non-duplicate | 10 |
| no secrets/PII | 10 |
| reasonable complexity | 5 |
| tests available and passing | 10 |
| formatting/readability | 5 |

Suggested use:

```text
90–100 : premium SFT / agent / high sampling weight
80–89  : high-quality pretraining
70–79  : normal pretraining
<70    : reject
```

For SFT:
```text
minimum score: 90
```

For agent trajectories:
```text
minimum score: 95
tests_passed must be true when executable tests exist
```

---

# 14. Repository quality signals

Positive:
- recognized license
- CI configuration
- tests directory
- meaningful README
- package metadata
- multiple substantive source files
- nontrivial history
- releases/tags
- normal dependency graph

Negative:
- generated dump
- tutorial copied thousands of times
- one-file spam repository
- token/credential dumps
- malware delivery
- obfuscated code
- dependency/vendor mirror
- minified frontend bundle
- mostly binary content

Do not use GitHub stars alone as a quality score.

---

# 15. Train/validation split

Split by **repository**, never randomly by file.

Example:

```text
train       97%
validation   2%
internal test 1%
```

Why repository-level splitting matters:

If files from the same repository appear in both train and validation, validation loss becomes artificially optimistic.

For SFT, also group related tasks/problem families before splitting.

---

# 16. Recommended SFT composition

Target **~150K final examples**:

| Category | Count |
|---|---:|
| code generation | 28K |
| debugging / bug fix | 25K |
| code explanation | 15K |
| refactoring | 12K |
| unit-test generation | 12K |
| API/library usage | 10K |
| algorithms/data structures | 12K |
| SQL/database | 6K |
| shell/DevOps/config | 6K |
| code review/security | 6K |
| multi-turn coding conversations | 8K |
| repository agent/tool trajectories | 10K |
| **Total** | **150K** |

This distribution can overlap with OpenCoder/Magicoder/Glaive after quality filtering, but the **agent trajectories should preferably be your own executable data**.

---

# 17. Language mix for SFT

A practical target:

```text
Python        32%
TypeScript    13%
JavaScript    10%
Java          10%
Go             9%
C++            8%
Rust           6%
SQL            5%
Shell          3%
C              2%
Config/other   2%
```

Do not let Python become 60–80% unless you intentionally want a Python-specialized model.

---

# 18. Data acceptance metrics for a production release

Do not say "100% accurate." Track metrics like these.

### Pretraining corpus

```text
known provenance:                  100%
license field present:             100%
secret scanner pass:               100%
exact duplicate rate after merge: <0.1%
near duplicate rate:              <2%
benchmark contamination:           0 known matches
invalid UTF-8:                     0
```

### SFT

```text
known provenance/license:             100%
schema validation:                    100%
no unresolved secrets/PII:            100%
syntax parse/compile rate:           >=98%
executable examples with tests pass: 100%
duplicate prompt rate:               <1%
benchmark contamination:              0 known matches
manual audit acceptance:             >=95% on sampled batch
```

### Agent trajectories

```text
valid tool schema:                    100%
patch applies:                        100%
tests recorded accurately:            100%
final state reproducible:              100%
unrelated destructive changes:          0
hidden credential leakage:              0
```

These are defensible quality guarantees. "100% model accuracy" is not.

---

# 19. Manual audit

Even after automated filtering, manually audit random samples.

Recommended:

```text
Every new source:
    inspect 200 samples before enabling it

Every production shard:
    inspect at least 100 random samples

Every SFT release:
    inspect 1,000 random examples

Every agent trajectory release:
    inspect all failures + 5–10% random successful trajectories
```

Record reviewer decisions.

---

# 20. Data manifest

For every dataset version create:

```yaml
dataset_name: gopi-coder-80m-pretrain-v1
created_at: 2026-09-22
tokenizer: gopi-coder-8k
sources:
  - name: the-stack-v2
    source_version: v2.2.0
    accepted_tokens: 950000000
  - name: codesearchnet
    accepted_tokens: 150000000
  - name: starcoderdata
    accepted_tokens: 150000000
  - name: fineweb-edu
    accepted_tokens: 250000000
derived:
  fim_tokens: 300000000
filters:
  license_gate: true
  secret_scan: true
  pii_scan: true
  exact_dedup: true
  near_dedup: true
  benchmark_decontamination: true
```

Also store:
- input source versions,
- filter software version,
- filter configuration hash,
- accepted/rejected counts,
- token counts by language,
- license histogram,
- duplicate statistics,
- contamination report,
- SHA-256 for generated shards.

---

# 21. Suggested production folder structure

```text
data/coding80m_production/
├── raw/
│   ├── stack_v2/
│   ├── codesearchnet/
│   ├── starcoder/
│   └── fineweb_edu/
├── staging/
│   ├── normalized/
│   ├── licensed/
│   ├── secret_clean/
│   ├── parsed/
│   └── deduped/
├── pretrain/
│   ├── code/
│   ├── docs/
│   ├── issues_commits/
│   └── fim/
├── sft/
│   ├── coding/
│   ├── debugging/
│   ├── tests/
│   └── agent/
├── dpo/
├── eval_private/
└── manifests/
```

Never edit the final accepted shards manually. Regenerate them from a versioned pipeline.

---

# 22. Recommended acquisition order

Start in this order:

### Phase A — 50M-token pilot

Use:
- The Stack v2 / CodeParrot clean
- CodeSearchNet
- small FineWeb-Edu subset

Purpose:
- tokenizer validation
- data pipeline validation
- training stability

### Phase B — 300M-token model-quality run

Add:
- more language-balanced source code
- issues/commits/notebooks
- FIM

Purpose:
- verify learning curves and coding quality

### Phase C — full 1.8B-token production pretraining

Only after the data report is clean.

Then SFT with ~150K verified examples.

Then optional DPO with 20K–40K executable preference pairs.

---

# 23. What not to do

Do not:

- download random Google Drive datasets with unknown provenance,
- train on benchmark solutions,
- ingest entire GitHub repositories without license checks,
- assume "open source" means unrestricted use,
- mix validation/test files into training,
- use millions of synthetic instructions without verification,
- train from code containing API keys/secrets,
- use only Python,
- keep generated/minified/vendor files,
- evaluate only on examples derived from the training datasets,
- claim 100% correctness from dataset filtering.

---

# 24. Best practical source stack for GopiCoder-80M

If you want the simplest high-quality production recipe:

```text
1. The Stack v2 filtered permissive code
      → ~950M tokens

2. Same accepted code converted to FIM
      → ~300M tokens

3. CodeSearchNet code/doc pairs
      → ~150M tokens

4. StarCoder GitHub issues + commits + notebooks
      → ~150M tokens

5. FineWeb-Edu technical subset
      → ~250M tokens

TOTAL
      → ~1.80B pretraining tokens
```

Then:

```text
OpenCoder SFT Stage 2
Magicoder OSS-Instruct
Glaive Code Assistant
your own verified real repository tasks

→ ~150K final SFT examples
```

Then:

```text
test-verified candidate pairs
→ 20K–40K DPO pairs
```

This is much better for an 80M model than collecting a huge uncontrolled corpus.

---

# 25. Final production principle

For a small coding model, **quality and specialization matter more than raw dataset size**.

Your 10M experiment already demonstrated this principle: a smaller, correctly trained model can answer better than a larger poorly aligned model.

For GopiCoder-80M, optimize for:

```text
known source
+ legal/provenance traceability
+ clean code
+ correct syntax
+ executable verification
+ deduplication
+ balanced languages
+ real software-development tasks
+ clean instruction tuning
+ benchmark isolation
```

—not "download everything."

