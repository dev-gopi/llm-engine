"""Extract a clean conversational subset from processed SmolTalk JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)
repository_root = str(Path(__file__).resolve().parents[1])
if repository_root not in sys.path:
    sys.path.insert(0, repository_root)

from datasets.loader import iter_records
from scripts.prepare_recovery_sft import BAD_TEXT, SYSTEM_PROMPT, _repetition_ratio, clean_text


ALLOWED_SUBSETS = {"everyday-conversations"}
IDENTITY_LEAK = re.compile(r"\b(?:open\s*assistant|chatgpt|claude)\b", re.IGNORECASE)


def clean_record(record: Mapping) -> dict | None:
    if record.get("source_subset") not in ALLOWED_SUBSETS:
        return None
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    cleaned = [{"role": "system", "content": SYSTEM_PROMPT}]
    seen_user = False
    assistant_count = 0
    for message in messages:
        if not isinstance(message, Mapping):
            return None
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            continue
        if role not in {"user", "assistant"} or not isinstance(content, str):
            return None
        content = clean_text(content)
        if not content or BAD_TEXT.search(content) or _repetition_ratio(content) > 0.45:
            return None
        if role == "assistant" and IDENTITY_LEAK.search(content):
            return None
        if role == "user":
            seen_user = True
        elif not seen_user:
            return None
        else:
            assistant_count += 1
        cleaned.append({"role": role, "content": content})
    if not seen_user or not assistant_count or cleaned[-1]["role"] != "assistant":
        return None
    digest = hashlib.sha256(
        json.dumps(cleaned, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return {
        "id": f"smoltalk-chat-{digest[:20]}",
        "domain": "chat",
        "source": "smoltalk_everyday_conversations",
        "messages": cleaned,
    }


def prepare(source: Path, destination: Path, *, excluded_ids: set[str] | None = None) -> tuple[int, set[str]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded_ids = excluded_ids or set()
    accepted: dict[str, dict] = {}
    for record in iter_records(source):
        cleaned = clean_record(record)
        if cleaned is not None and cleaned["id"] not in excluded_ids:
            accepted.setdefault(cleaned["id"], cleaned)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for key in sorted(accepted):
                json.dump(accepted[key], stream, ensure_ascii=False, separators=(",", ":"))
                stream.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return len(accepted), set(accepted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/smoltalk_chat"))
    args = parser.parse_args()
    validation_count, validation_ids = prepare(
        args.input / "validation.jsonl", args.output / "validation.jsonl"
    )
    train_count, _ = prepare(
        args.input / "train.jsonl", args.output / "train.jsonl",
        excluded_ids=validation_ids,
    )
    print(json.dumps({"train": train_count, "validation": validation_count}))


if __name__ == "__main__":
    main()
