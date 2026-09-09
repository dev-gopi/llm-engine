"""Create a small, deterministic correction corpus for observed SFT failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


SYSTEM = "You are Gopi, a helpful assistant. Answer clearly and briefly."


def record(identifier: str, domain: str, prompt: str, answer: str) -> dict:
    return {
        "id": identifier,
        "domain": domain,
        "source": "reviewed_synthetic_corrections_v1",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def candidates() -> list[dict]:
    rows: list[dict] = []
    facts = [
        ("France", "Paris"), ("India", "New Delhi"), ("Bangladesh", "Dhaka"),
        ("Japan", "Tokyo"), ("Italy", "Rome"), ("Germany", "Berlin"),
        ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Nepal", "Kathmandu"),
        ("Spain", "Madrid"),
    ]
    for country, capital in facts:
        rows.append(record(f"fact-{country.lower()}", "factual_qa",
                           f"What is the capital of {country}? Answer in one sentence.",
                           f"The capital of {country} is {capital}."))

    for left in range(2, 31):
        for right in range(2, 11):
            rows.append(record(f"add-{left}-{right}", "math",
                               f"What is {left} + {right}?", str(left + right)))
            if left >= right:
                rows.append(record(f"sub-{left}-{right}", "math",
                                   f"What is {left} - {right}?", str(left - right)))
    for value in range(2, 13):
        rows.append(record(f"mul-{value}", "math", f"What is {value} multiplied by 6?",
                           str(value * 6)))

    words = ["READY", "BLUE", "MANGO", "DONE", "YES", "SAFE", "START", "OK"]
    for index, word in enumerate(words):
        rows.append(record(f"exact-{index}", "instruction",
                           f"Reply with exactly this word: {word}", word))

    code = [
        ("add", "Write a Python function add(a, b) that returns their sum.",
         "```python\ndef add(a, b):\n    return a + b\n```"),
        ("subtract", "Write a Python function subtract(a, b) that returns a minus b.",
         "```python\ndef subtract(a, b):\n    return a - b\n```"),
        ("length", "Which Python built-in returns the number of items in a list?",
         "Use `len`, for example `len(items)`."),
        ("remainder", "What Python expression gives the remainder of a divided by b?",
         "`a % b`"),
        ("square", "Write a Python function square(x) that returns x squared.",
         "```python\ndef square(x):\n    return x * x\n```"),
    ]
    for name, prompt, answer in code:
        rows.append(record(f"code-{name}", "coding", prompt, answer))

    multilingual = [
        ("bn-capital", "bengali", "বাংলাদেশের রাজধানীর নাম বলো।", "বাংলাদেশের রাজধানী ঢাকা।"),
        ("bn-add", "bengali", "বাংলায় উত্তর দাও: দুই আর তিন যোগ করলে কত হয়?", "পাঁচ।"),
        ("bn-sun", "bengali", "বাংলায় বলো: সূর্য কোন দিক থেকে ওঠে?", "সূর্য পূর্ব দিক থেকে ওঠে।"),
        ("hi-capital", "hindi", "हिंदी में बताओ: भारत की राजधानी का नाम क्या है?", "भारत की राजधानी नई दिल्ली है।"),
        ("hi-add", "hindi", "हिंदी में उत्तर दो: दो और तीन जोड़ने पर कितना होता है?", "पाँच।"),
        ("hi-sun", "hindi", "हिंदी में बताओ: सूरज किस दिशा से निकलता है?", "सूरज पूर्व दिशा से निकलता है।"),
    ]
    for identifier, domain, prompt, answer in multilingual:
        rows.append(record(identifier, domain, prompt, answer))

    rows.extend([
        record("identity", "chat", "Hello! Who are you?", "I'm Gopi, a helpful AI assistant."),
        record("thanks", "chat", "Thank you for helping me.", "You're welcome!"),
        record("reasoning-apples", "reasoning",
               "I have 8 apples, buy 3 more, and give away 2. How many remain?",
               "8 + 3 - 2 = 9, so 9 apples remain."),
        record("reasoning-order", "reasoning",
               "Asha is older than Bilal, and Bilal is older than Chen. Who is youngest?",
               "Chen is the youngest."),
    ])
    return rows


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
    validation = [row for index, row in enumerate(ordered) if index % 10 == 0]
    training = [row for index, row in enumerate(ordered) if index % 10 != 0]
    return training, validation


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/refinement"))
    args = parser.parse_args()
    training, validation = split(candidates())
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "train.jsonl", training)
    write_jsonl(args.output / "validation.jsonl", validation)
    manifest = {
        "schema_version": 1,
        "name": "Gopi correctness refinement corrections",
        "source": "Deterministic reviewed templates in scripts/prepare_refinement_data.py",
        "version": "1",
        "license": {
            "identifier": "LicenseRef-Project-Generated",
            "review_status": "reviewed",
            "commercial_use": "allowed",
        },
        "allowed_stages": ["sft"],
        "privacy_review": "reviewed",
        "splits": {"train": len(training), "validation": len(validation)},
    }
    (args.output / "dataset-manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
