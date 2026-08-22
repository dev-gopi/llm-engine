"""Convert DailyDialog into user/assistant pairs for conversational training."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def prepare(source: Path, destination: Path, maximum_pairs: int, bot_name: str) -> int:
    with zipfile.ZipFile(source) as archive:
        with archive.open("data/dialogues.json") as stream:
            dialogues = json.load(stream)

    pairs: list[dict[str, object]] = []
    for dialogue in dialogues:
        if dialogue.get("data_split") != "train":
            continue

        turns = dialogue.get("turns", [])
        for index in range(len(turns) - 1):
            user_turn = turns[index]
            assistant_turn = turns[index + 1]
            if user_turn.get("speaker") != "user" or assistant_turn.get("speaker") != "system":
                continue

            user_text = user_turn.get("utterance", "").strip()
            assistant_text = assistant_turn.get("utterance", "").strip()
            if not user_text or not assistant_text:
                continue

            pairs.append(
                {
                    "id": f"{dialogue['dialogue_id']}-pair-{index // 2}",
                    "bot_name": bot_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                f"You are {bot_name}, a helpful, honest, and friendly AI assistant."
                            ),
                        },
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ],
                    "domain": dialogue.get("domains", []),
                    "user_emotion": user_turn.get("emotion"),
                    "assistant_emotion": assistant_turn.get("emotion"),
                    "source": "ConvLab/dailydialog",
                }
            )
            if len(pairs) == maximum_pairs:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(pairs, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                return len(pairs)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/dailydialog/data.zip"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/dailydialog/dailydialog-conversations.json"),
    )
    parser.add_argument("--maximum-pairs", type=int, default=2_000)
    parser.add_argument("--bot-name", default="Gopi")
    args = parser.parse_args()

    count = prepare(args.source, args.output, args.maximum_pairs, args.bot_name)
    print(json.dumps({"conversation_pairs": count, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
