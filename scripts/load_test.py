"""Load-test the generation API and emit a JSON release-gate report."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

from evaluation.load_testing import LoadSample, release_gate, summarize


async def run(args) -> tuple[dict, bool]:
    semaphore = asyncio.Semaphore(args.concurrency)
    randomizer = random.Random(args.seed)
    sizes = [int(value) for value in args.prompt_words.split(",")]
    prompts = ["word " * randomizer.choice(sizes) for _ in range(args.requests)]

    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as client:
        async def one(index: int, prompt: str) -> LoadSample:
            async with semaphore:
                started = time.perf_counter()
                try:
                    timeout = args.cancel_after if args.cancel_every and index % args.cancel_every == 0 else args.timeout
                    response = await asyncio.wait_for(client.post(
                        "/v1/generate",
                        headers={"Authorization": f"Bearer {args.api_key}"} if args.api_key else {},
                        json={"prompt": prompt, "max_tokens": args.max_tokens, "seed": args.seed + index},
                    ), timeout=timeout)
                    payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    usage = payload.get("usage", {})
                    return LoadSample(time.perf_counter() - started, response.status_code,
                                      int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
                except asyncio.TimeoutError:
                    return LoadSample(time.perf_counter() - started, None, cancelled=True)
                except Exception as error:
                    return LoadSample(time.perf_counter() - started, None, error=type(error).__name__)

        started = time.perf_counter()
        samples = await asyncio.gather(*(one(index, prompt) for index, prompt in enumerate(prompts, 1)))
        report = summarize(list(samples), time.perf_counter() - started)
    passed = release_gate(report, max_p95_seconds=args.max_p95, max_failure_rate=args.max_failure_rate)
    report["release_gate_passed"] = passed
    return report, passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-words", default="16,128,512")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--cancel-every", type=int, default=0)
    parser.add_argument("--cancel-after", type=float, default=0.05)
    parser.add_argument("--max-p95", type=float, default=10)
    parser.add_argument("--max-failure-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens) < 1:
        parser.error("requests, concurrency, and max-tokens must be positive")
    report, passed = asyncio.run(run(args))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
