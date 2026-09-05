"""Build a deterministic, cleaned SFT recovery dataset from selected local sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from datasets.loader import iter_records
from inference.context import format_system_prompt


BASE_SYSTEM_PROMPT = "You are Gopi, a helpful assistant. Answer clearly and briefly."
SYSTEM_PROMPT = format_system_prompt(BASE_SYSTEM_PROMPT, "plain")

DOMAIN_SOURCES = {
    "chat": ("core_chat", "helpsteer", "v2_openassistant_en"),
    # OpenOrca is deliberately excluded: its local split is 2.9M records and
    # its broad synthetic style is counterproductive for this narrow recovery.
    "english": ("general_qa",),
    "bengali": ("bangla_qa", "bangla_reading_qa", "multilingual_bn_hi"),
    "hindi": ("hindi_history_qa", "multilingual_hi", "hinglish_chat"),
    "math": ("gsm8k", "v2_math_instruct"),
    "coding": ("code_instructions", "code_alpaca", "v2_code_feedback"),
}

TRAIN_LIMITS = {
    "chat": 30_000,
    "english": 20_000,
    "bengali": 15_000,
    "hindi": 10_000,
    "math": 15_000,
    "coding": 10_000,
}

BAD_TEXT = re.compile(
    r"\{\s*(?:name|answer|response|insert[^}]*)\s*\}|"
    r"\[\s*insert[^]]*\]|lorem ipsum|=\s*=\s*=\s*actions",
    re.IGNORECASE,
)


def clean_text(value: str) -> str:
    value = value.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    value = "\n".join(line.rstrip() for line in value.splitlines()).strip()
    # Leading whitespace is meaningful in Python and fenced code examples.
    return value


def extract_pair(record: Mapping[str, Any]) -> tuple[str, str] | None:
    messages = record.get("messages")
    if isinstance(messages, list):
        last_user: str | None = None
        pairs: list[tuple[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role, content = message.get("role"), message.get("content")
            if not isinstance(content, str):
                continue
            if role == "user":
                last_user = content
            elif role == "assistant" and last_user is not None:
                pairs.append((last_user, content))
                last_user = None
        return pairs[-1] if pairs else None
    prompt = record.get("prompt") or record.get("instruction") or record.get("question")
    response = (
        record.get("chosen") or record.get("response") or record.get("output")
        or record.get("answer")
    )
    return (prompt, response) if isinstance(prompt, str) and isinstance(response, str) else None


def quality_pair(pair: tuple[str, str] | None) -> tuple[str, str] | None:
    if pair is None:
        return None
    prompt, response = map(clean_text, pair)
    if not 2 <= len(prompt) <= 2_000 or not 1 <= len(response) <= 4_000:
        return None
    if BAD_TEXT.search(prompt) or BAD_TEXT.search(response):
        return None
    if _repetition_ratio(response) > 0.45:
        return None
    if len(response.split()) > 600 or len(prompt.split()) > 300:
        return None
    return prompt, response


def record_has_quality_signals(record: Mapping[str, Any], source: str) -> bool:
    """Use available human ratings; do not invent thresholds for unrated sources."""
    if source != "helpsteer":
        return True
    scores = record.get("scores")
    if not isinstance(scores, Mapping):
        return False
    return all(
        isinstance(scores.get(name), (int, float)) and float(scores[name]) >= 3.0
        for name in ("helpfulness", "correctness", "coherence")
    )


def _repetition_ratio(text: str) -> float:
    units = [item.casefold() for item in re.findall(r"\w+|[^\w\s]", text)]
    if len(units) < 12:
        return 0.0
    trigrams = list(zip(units, units[1:], units[2:]))
    return 1.0 - len(set(trigrams)) / len(trigrams)


def prompt_key(prompt: str) -> str:
    normalized = re.sub(r"\W+", " ", prompt.casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_record(domain: str, source: str, prompt: str, response: str) -> dict[str, Any]:
    return {
        "id": hashlib.sha256(f"{domain}\0{source}\0{prompt}\0{response}".encode()).hexdigest()[:20],
        "domain": domain,
        "source": source,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def collect_records(
    paths: Iterable[tuple[str, Path]], domain: str, *, excluded_prompts: set[str], limit: int,
    tokenizer=None, max_length: int = 512,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: dict[str, tuple[str, dict[str, Any]]] = {}
    stats = {"read": 0, "rejected": 0, "duplicate": 0}
    for source, path in paths:
        if not path.is_file():
            continue
        for record in iter_records(path):
            stats["read"] += 1
            if not record_has_quality_signals(record, source):
                stats["rejected"] += 1
                continue
            pair = quality_pair(extract_pair(record))
            if pair is None:
                stats["rejected"] += 1
                continue
            prompt, response = pair
            key = prompt_key(prompt)
            if key in excluded_prompts or key in candidates:
                stats["duplicate"] += 1
                continue
            item = make_record(domain, source, prompt, response)
            rank = hashlib.sha256(item["id"].encode()).hexdigest()
            candidates[key] = (rank, item)
    selected = []
    stats["over_context"] = 0
    for _, item in sorted(candidates.values(), key=lambda item: item[0]):
        if tokenizer is not None:
            from datasets.loader import TextDataset
            identifiers, _ = TextDataset._encode_chat(item["messages"], tokenizer, True, True)
            if len(identifiers) > max_length:
                stats["over_context"] += 1
                continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected, stats


def build_dataset(data_root: Path, output: Path, *, tokenizer=None, max_length: int = 512,
                  train_limit: int | None = None, validation_limit: int = 1_000) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"system_prompt": SYSTEM_PROMPT, "domains": {},
                              "tokenizer_fingerprint": tokenizer.fingerprint if tokenizer else None,
                              "max_length": max_length if tokenizer else None}
    validation_prompts: set[str] = set()
    validations: dict[str, list[dict[str, Any]]] = {}
    validation_statistics: dict[str, dict[str, int]] = {}

    # Reserve every held-out prompt, including rows beyond the validation
    # display limit and examples rejected by quality filtering.
    held_out_prompts: set[str] = set()
    for sources in DOMAIN_SOURCES.values():
        for source in sources:
            for split in ("validation", "test"):
                path = data_root / source / f"{split}.jsonl"
                if path.is_file():
                    for record in iter_records(path):
                        pair = extract_pair(record)
                        if pair is not None:
                            held_out_prompts.add(prompt_key(pair[0]))

    # Discover the complete held-out prompt set before admitting any training
    # record, including prompts duplicated across differently named domains.
    for domain, sources in DOMAIN_SOURCES.items():
        validation_paths = [
            (source, data_root / source / "validation.jsonl") for source in sources
        ]
        validation, validation_stats = collect_records(
            validation_paths,
            domain,
            excluded_prompts=validation_prompts,
            limit=validation_limit,
            tokenizer=tokenizer, max_length=max_length,
        )
        validations[domain] = validation
        validation_statistics[domain] = validation_stats
        validation_prompts.update(prompt_key(item["messages"][1]["content"]) for item in validation)

    for domain, sources in DOMAIN_SOURCES.items():
        validation = validations[domain]
        validation_stats = validation_statistics[domain]
        training_paths = [(source, data_root / source / "train.jsonl") for source in sources]
        training, training_stats = collect_records(
            training_paths,
            domain,
            excluded_prompts=held_out_prompts,
            limit=train_limit if train_limit is not None else TRAIN_LIMITS[domain],
            tokenizer=tokenizer, max_length=max_length,
        )
        held_out_prompts.update(prompt_key(item["messages"][1]["content"]) for item in training)
        domain_dir = output / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "name": f"Gopi cleaned recovery SFT ({domain})",
            "source": "Derived from governed local SFT sources; see each record's source field",
            "version": "1",
            "license": {
                "identifier": "LicenseRef-Mixed-Upstream",
                "review_status": "unreviewed",
                "commercial_use": "unknown",
            },
            "allowed_stages": ["sft"],
            "privacy_review": "unreviewed",
        }
        import yaml
        (domain_dir / "dataset-manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
        for split, records in (("train", training), ("validation", validation)):
            with (domain_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        summary["domains"][domain] = {
            "train": len(training), "validation": len(validation),
            "train_stats": training_stats, "validation_stats": validation_stats,
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/recovery_sft"))
    parser.add_argument("--tokenizer", type=Path, help="reject complete conversations that exceed context")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-limit", type=int, help="maximum accepted examples per domain")
    parser.add_argument("--validation-limit", type=int, default=1_000)
    args = parser.parse_args()
    if args.max_length < 2 or args.validation_limit < 1 or (args.train_limit is not None and args.train_limit < 1):
        parser.error("limits must be positive and max-length at least 2")
    from tokenizer.encoder import Tokenizer
    tokenizer = Tokenizer.load(args.tokenizer) if args.tokenizer else None
    print(json.dumps(build_dataset(args.data_root, args.output, tokenizer=tokenizer,
                                  max_length=args.max_length, train_limit=args.train_limit,
                                  validation_limit=args.validation_limit), indent=2))


if __name__ == "__main__":
    main()
