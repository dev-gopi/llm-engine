# Portable MiniGPT backend

This backend keeps the custom MiniGPT Transformer and its trained weights. It
adds a local `from_pretrained` / `save_pretrained` workflow. It does not download
third-party models or convert their weights into MiniGPT.

Export your existing checkpoint without running training:

```bash
.venv/bin/python scripts/export_pretrained.py \
  --checkpoint checkpoints/finetuning/best.pt \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --output exports/minigpt-bundle
```

The exporter uses available EMA weights by default; `--raw-weights` selects raw
model weights. It loads checkpoint contents on CPU, so optimizer state does not
consume GPU memory. CPU RAM must still accommodate the training checkpoint.
The destination must not already exist.

Use the exported bundle:

```python
from inference.pretrained import MiniGPTBackend

backend = MiniGPTBackend.from_pretrained("exports/minigpt-bundle", device="auto")
result = backend.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain Python dictionaries."},
], max_tokens=128)
print(result.text)

# Raw completion, streaming, and equal-length cohort batching:
result = backend.generate("Once upon a time", max_tokens=64)
for event in backend.stream("Once upon a time", max_tokens=64):
    print(event.token, end="", flush=True)
results = backend.generate_batch(["Hello", "World"], max_tokens=32)
```

Bundles contain `config.json`, `model.safetensors`, and `tokenizer/`. Loading
validates format version, architecture, tokenizer fingerprint, vocabulary, and
state-dict compatibility. Safetensors serialization preserves tied model weights
without including optimizer or training RNG state. Call options override saved
generation defaults. `chat` uses the existing chat template and does not persist
history; supply the complete bounded message list on each call.

This is a Python backend and export CLI; the existing HTTP server configuration
continues using its checkpoint loader. One backend instance owns mutable
generation/cache state: serialize access rather than sharing it across
uncoordinated threads. No additional training, intelligence gains, remote model
hub compatibility, or trillion-scale runtime support are implied by the bundle
format.

## Persistent chat and reviewed learning data

Persistence is opt-in, with a database path and session ID that you choose:

```python
session = backend.open_session("data/chat/private.sqlite", "my-session",
                               system_prompt="You are a helpful assistant.",
                               ttl_seconds=86400)
print(session.chat("My name is Alice.", max_tokens=64).text)
print(session.chat("What is my name?", max_tokens=64).text)
print(session.history())
```

Opening the same session ID and database after a restart restores recent
history. This supplies context; it does not update weights or guarantee correct
recall. History is trimmed to the context budget and expires after the configured
TTL. Only successful nonempty replies are saved. Use one owner per session and
serialize access to the backend; simultaneous handles/processes updating the same
session are not supported. Use different session IDs for different users.
The database stores plain text; protect it with your filesystem permissions.

Select training examples explicitly:

```python
print(session.review_pending())  # inspect all messages in the bounded snapshot
session.approve_conversation(corrected_response="Your name is Alice.")
session.export_training("data/processed/reviewed-chat/train.jsonl")
```

Approval covers the entire snapshot, including earlier assistant replies.
Review all of them before approval because the SFT loader trains on all assistant
messages. `corrected_response` replaces only the final training answer; it does
not rewrite chat history. Nothing enters the export without approval. Repeated
identical approvals are deduplicated per session. Exports refuse to overwrite
existing files. Approved data persists independently of history expiry.

The JSONL uses the existing `messages` training format. Split reviewed examples
into separate training and held-out validation sets, then add the training file
to your fine-tuning YAML (and an explicit dataset weight when using weighted
mixtures). Train to a separate output checkpoint and compare held-out metrics
and response probes before deploying it. Training is never started by chat,
approval, or export.

```python
session.forget()  # delete conversation history
session.forget(include_training_examples=True)  # also delete approved DB examples
```

Already-exported JSONL files are independent copies and are not deleted by
`forget`. Backend model bundles do not contain the chat database.

## Browser playground

The existing `/ui/` playground exposes Min P and newline-separated stop strings
for both REST and streaming generation. “Use conversation memory” sends the
existing tab session ID when enabled and independent requests when disabled.
Server retention follows the existing serving session-store configuration; the
browser transcript remains tab-local. “New chat” starts a new session and does
not delete the previous server session.

Open “Review a response for training” to copy the latest completed visible
prompt/answer, edit either field, and explicitly approve a JSONL download. Review
fields are not persisted in browser settings. The download is a standalone
user/assistant pair accepted by the training loader. It does not include hidden
server prompts, earlier conversation, tools, or attachments; add necessary
context to the reviewed prompt. This UI flow does not access `ChatSession`'s
SQLite approval table, upload training data, or start training.

## Output controls and prompt boundaries

The playground now exposes minimum tokens and repeated token n-gram blocking.
Minimum tokens delays EOS only; explicit stop strings or the context limit still
end generation. N-gram blocking and repetition penalties can reduce loops but
may damage code, quotations, or legitimate repetition. They cannot establish
factual accuracy or repair an undertrained checkpoint.

Chat serialization escapes literal `<|` sequences inside message content while
preserving template-generated role markers. The chat training encoder uses the
same escaping convention. This prevents content from creating special-token
role boundaries; it changes how literal control-token examples are encoded.
Already tokenized datasets are not rewritten. Explicit raw generation with
`allow_special_tokens=True` remains a trusted-caller interface.

Prompt-injection checks ignore invisible Unicode format characters for detection
only, closing that particular obfuscation gap without changing the original
multilingual text. These pattern checks are heuristic and can miss attacks or
reject benign text. They are not an authorization boundary: keep tool allowlists,
server authentication, and resource limits enabled independently. No output
moderation classifier or factuality verifier was added in this change.
