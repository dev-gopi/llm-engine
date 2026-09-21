# Instruction and Multi-Turn Corpus Contract

`CHAT-002` defines a review contract for instruction-tuning examples. It is a
data-admission contract, not evidence that any particular external corpus has
been reviewed.

Each record must be a JSON object with a non-empty `messages` list. A message
has a `role` of `system`, `user`, `assistant`, or `tool` and string `content`.
The canonical renderer is `inference.chat_session.format_chat_messages`.
Only assistant content and its end delimiter receive supervised loss; system,
user, and tool payloads are context only.

Before activation, reviewers must label examples for clarification, safe
refusal, format following, conversational consistency, or tool-turn handling;
record source, version, license, privacy review, and reviewer decision in the
adjacent dataset manifest. Tool payloads must be treated as untrusted data and
must not be mistaken for instructions.

`configs/evaluation.instruction_following.jsonl` is the versioned deterministic
smoke suite. It covers one case in each required category and is scored with
`evaluation.benchmarks.BenchmarkCase`; it measures protocol compliance, not
model capability or corpus quality.
