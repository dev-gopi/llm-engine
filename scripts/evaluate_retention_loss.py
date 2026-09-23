"""Compare two checkpoints on identical complete raw-text validation records.

Token perplexities across different tokenizers are not directly comparable;
bits per UTF-8 byte measures the same text under each model's own tokenizer.
This checks language retention, not instruction-following or reasoning accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch

from local_dataset.loader import iter_records
from local_dataset.preprocessor import record_to_text
from model.gpt import MiniGPT
from model.loss import CausalLanguageModelLoss
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("baseline", "candidate"):
        parser.add_argument(f"--{role}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{role}-tokenizer", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-file", type=int, default=24)
    parser.add_argument("--scan-limit", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new path")
    if min(args.samples_per_file, args.scan_limit, args.threads) < 1 or args.max_length < 2:
        parser.error("sample, scan, and thread limits must be positive; max-length must be at least 2")
    torch.set_num_threads(args.threads)
    config = load_yaml(args.model_config)
    if args.max_length > config["max_position"]:
        parser.error("max-length exceeds model context")
    tokenizers = {role: Tokenizer.load(getattr(args, f"{role}_tokenizer")) for role in ("baseline", "candidate")}
    samples = {}
    for path in args.dataset:
        selected = []
        seen = set()
        for index, record in enumerate(iter_records(path)):
            if index >= args.scan_limit:
                break
            if "messages" in record or "token_ids" in record:
                parser.error("retention loss expects raw text records, not chat or prepacked tokens")
            text = record_to_text(record)
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            ids = {role: tok.encode(text, add_bos=True, add_eos=True) for role, tok in tokenizers.items()}
            if any(len(value) > args.max_length for value in ids.values()):
                continue
            selected.append({"line_index": index, "sha256": digest, "bytes": len(text.encode()), "ids": ids})
            if len(selected) == args.samples_per_file:
                break
        if not selected:
            parser.error(f"no complete records fit both tokenizers: {path}")
        samples[str(path)] = selected
    report = {
        "note": "Bounded development sample: first unique complete records fitting BOTH tokenizers; not a corpus-wide estimate. EOS is scored.",
        "weights": args.weights, "device": args.device, "max_length": args.max_length,
        "samples": {path: [{key: value for key, value in row.items() if key != "ids"} for row in rows]
                    for path, rows in samples.items()},
        "models": {},
    }
    device = resolve_device(args.device)
    objective = CausalLanguageModelLoss(reduction="sum", chunk_size=128)
    for role, tok in tokenizers.items():
        model = MiniGPT.from_config(adapt_config_to_tokenizer(config, tok), device="cpu")
        checkpoint = getattr(args, f"{role}_checkpoint")
        info = load_checkpoint(checkpoint, model, use_ema=args.weights == "ema", restore_rng=False,
                               **checkpoint_tokenizer_options(tok, allow_extension=False))
        model.to(device).eval()
        results = {}
        with torch.inference_mode():
            for path, rows in samples.items():
                nll = 0.0
                tokens = 0
                for row in rows:
                    inputs = torch.tensor([row["ids"][role]], device=device)
                    details = objective(model(inputs), inputs, return_details=True)
                    nll += float(details.cross_entropy)
                    tokens += details.token_count
                byte_count = sum(row["bytes"] for row in rows)
                results[path] = {"records": len(rows), "bytes": byte_count, "tokens": tokens,
                                 "cross_entropy": nll / tokens, "perplexity": math.exp(min(nll / tokens, 80)),
                                 "bits_per_byte": nll / (byte_count * math.log(2))}
        report["models"][role] = {"checkpoint": str(checkpoint), "step": info["step"],
                                   "ema_applied": info["ema_applied"], "tokenizer_fingerprint": tok.fingerprint,
                                   "domains": results}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["candidate_minus_baseline_bits_per_byte"] = {
        path: report["models"]["candidate"]["domains"][path]["bits_per_byte"]
              - report["models"]["baseline"]["domains"][path]["bits_per_byte"] for path in samples
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["candidate_minus_baseline_bits_per_byte"], indent=2))


if __name__ == "__main__":
    main()
