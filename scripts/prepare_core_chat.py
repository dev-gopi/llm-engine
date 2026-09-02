"""Create a deterministic, project-owned dataset for core assistant behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = "You are Gopi, a helpful assistant. Answer clearly and briefly."


def pair(identifier: str, prompt: str, answer: str) -> dict:
    return {
        "id": identifier,
        "bot_name": "Gopi",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "source": "gopi/core-chat-v1",
    }


def build_records() -> list[dict]:
    records: list[dict] = []
    identity_prompts = [
        "What is your name?", "Who are you?", "Introduce yourself.",
        "Hello, who are you?", "Can you tell me your name?",
    ]
    identity_answers = [
        "My name is Gopi. I am a helpful AI assistant.",
        "I am Gopi, an AI assistant here to help you.",
        "I'm Gopi. How can I help you today?",
    ]
    for prompt_index, prompt in enumerate(identity_prompts):
        for answer_index, answer in enumerate(identity_answers):
            records.append(pair(f"identity-{prompt_index}-{answer_index}", prompt, answer))

    greetings = {
        "Hi": "Hello! How can I help you?",
        "Hello": "Hello! What can I help you with today?",
        "Good morning": "Good morning! How can I help you?",
        "How are you?": "I'm ready to help. What would you like to know?",
        "Thank you": "You're welcome!",
        "Goodbye": "Goodbye! Have a great day.",
    }
    for index, (prompt, answer) in enumerate(greetings.items()):
        records.append(pair(f"greeting-{index}", prompt, answer))

    for left in range(21):
        for right in range(21):
            records.append(pair(
                f"addition-{left}-{right}",
                f"What is {left} plus {right}?",
                f"{left} plus {right} is {left + right}.",
            ))

    facts = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of India?", "The capital of India is New Delhi."),
        ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet."),
        ("How many days are in a week?", "There are seven days in a week."),
        ("What do plants need for photosynthesis?", "Plants use light, water, and carbon dioxide for photosynthesis."),
        ("Name three fruits.", "Apple, banana, and orange."),
        ("Give one tip for staying healthy.", "Exercise regularly and eat a balanced diet."),
        ("Give two ways to save water.", "Turn off taps when not in use and repair leaks promptly."),
    ]
    for index, (prompt, answer) in enumerate(facts):
        records.append(pair(f"fact-{index}", prompt, answer))

    rewrites = [
        ("Rewrite this politely: Send me the report now.", "Could you please send me the report?"),
        ("Correct this sentence: She go to school every day.", "She goes to school every day."),
        ("Rewrite this politely: Give me your notes.", "Could you please share your notes with me?"),
    ]
    for index, (prompt, answer) in enumerate(rewrites):
        records.append(pair(f"rewrite-{index}", prompt, answer))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/core_chat"))
    args = parser.parse_args()
    records = build_records()
    validation = records[::10]
    validation_ids = {record["id"] for record in validation}
    train = [record for record in records if record["id"] not in validation_ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("validation", validation)):
        destination = args.output_dir / f"{name}.jsonl"
        with destination.open("w", encoding="utf-8") as stream:
            for record in subset:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"train": len(train), "validation": len(validation)}, indent=2))


if __name__ == "__main__":
    main()
