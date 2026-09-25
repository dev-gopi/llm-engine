"""Run fixed qualitative probes on the best fine-tuned checkpoint."""

import json
import time
from pathlib import Path

import torch

from datasets.preprocessor import format_messages
from inference.generator import Generator
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml


torch.set_num_threads(4)
output_directory = Path("reports/finetuning_response_test")
output_directory.mkdir(parents=True, exist_ok=True)
tokenizer = Tokenizer.load("data/tokenizer-finetuning")
model = MiniGPT.from_config(
    adapt_config_to_tokenizer(load_yaml("configs/model.gpu.yaml"), tokenizer),
    device="cpu",
)
checkpoint = "checkpoints/finetuning/best.pt"
payload = load_checkpoint(
    checkpoint,
    model,
    map_location="cpu",
    use_ema=True,
    restore_rng=False,
    **checkpoint_tokenizer_options(tokenizer),
)
settings = {
    "max_tokens": 96,
    "temperature": 0.2,
    "top_k": 20,
    "top_p": 0.9,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 3,
    "seed": 42,
    "allow_special_tokens": True,
}
report = {
    "checkpoint": checkpoint,
    "step": payload.get("step"),
    "ema_used": bool((payload.get("ema") or {}).get("shadow")),
    "device": "cpu",
    "retrieval": False,
    "settings": settings,
    "responses": [],
}
del payload
generator = Generator(model, tokenizer, device="cpu")
probes = [
    ("greeting", "Hello! Who are you?"),
    ("factual_question", "What is the capital of France? Answer in one sentence."),
    ("reasoning", "I have 3 apples and buy 2 more. Then I give away 1 apple. How many apples do I have left?"),
    ("instruction_following", "Reply with exactly this word: READY"),
    ("story_request", "Write a short story about a girl who helps a lost puppy."),
    ("coding", "Write a Python function add(a, b) that returns their sum."),
    ("bengali", "বাংলায় এক বাক্যে বলো: বাংলাদেশের রাজধানী কী?"),
    ("hindi", "हिंदी में एक वाक्य में बताओ: भारत की राजधानी क्या है?"),
]
for name, prompt in probes:
    rendered_prompt = format_messages(
        [
            {"role": "system", "content": "You are Gopi, a helpful assistant. Answer clearly and briefly."},
            {"role": "user", "content": prompt},
        ],
        add_generation_prompt=True,
    )
    started_at = time.monotonic()
    result = generator.generate(rendered_prompt, **settings)
    row = {
        "name": name,
        "prompt": prompt,
        "response": result.text,
        "finish_reason": result.finish_reason,
        "completion_tokens": len(result.token_ids),
        "seconds": time.monotonic() - started_at,
    }
    report["responses"].append(row)
    (output_directory / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(row, ensure_ascii=False), flush=True)
