"""Save fixed raw-completion and chat probes for side-by-side checkpoint review."""
from __future__ import annotations

import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
sys.path[:] = [entry for entry in sys.path if str(Path(entry or '.').resolve()) != script_directory]

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone

import torch
from datasets.loader import iter_records
from datasets.preprocessor import format_messages
from inference.generator import Generator
from inference.context import format_system_prompt
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, default=Path('configs/evaluation.responses.jsonl'))
    parser.add_argument('--model-config', type=Path, default=Path('configs/model.gpu.yaml'))
    parser.add_argument('--tokenizer', type=Path, default=Path('data/tokenizer'))
    parser.add_argument('--inference-config', type=Path, default=Path('configs/inference.yaml'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--weights', choices=['model', 'ema'], default='model')
    parser.add_argument('--max-tokens', type=int, default=96)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.max_tokens < 1:
        parser.error('threads and max-tokens must be positive')
    if args.output.exists() or args.output.with_suffix('.md').exists():
        parser.error('output already exists; use a distinct name to preserve comparisons')
    if args.output.suffix != '.json':
        parser.error('output must end with .json')
    probes = list(iter_records(args.prompts))
    if not probes or any(p.get('mode') not in ('raw', 'chat') or not isinstance(p.get('prompt'), str)
                         or not p['prompt'].strip() for p in probes):
        parser.error('prompts must contain nonempty prompt strings and raw/chat modes')
    torch.set_num_threads(args.threads)
    tokenizer = Tokenizer.load(args.tokenizer)
    model_config = adapt_config_to_tokenizer(load_yaml(args.model_config), tokenizer)
    # Load the full checkpoint on CPU, then transfer only model weights to GPU.
    model = MiniGPT.from_config(model_config, device='cpu')
    payload = load_checkpoint(args.checkpoint, model, use_ema=args.weights == 'ema',
                              restore_rng=False, **checkpoint_tokenizer_options(tokenizer))
    ema_used = args.weights == 'ema' and bool((payload.get('ema') or {}).get('shadow'))
    report = {'checkpoint': str(args.checkpoint), 'step': payload.get('step'),
              'created_at': datetime.now(timezone.utc).isoformat(),
              'weights_requested': args.weights, 'ema_used': ema_used,
              'tokenizer_fingerprint': tokenizer.fingerprint,
              'prompts_sha256': hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
              'model_config': model_config, 'responses': [],
              'note': 'Qualitative development probes, not a standardized accuracy benchmark. No retrieval or tools.'}
    del payload
    device = resolve_device(args.device)
    report['device'] = str(device)
    generator = Generator(model, tokenizer, device=device)
    options = dict(max_tokens=args.max_tokens, temperature=0.2, top_k=20, top_p=0.9,
                   repetition_penalty=1.15, no_repeat_ngram_size=3, seed=args.seed,
                   allow_special_tokens=True)
    report['generation_settings'] = options
    inference_config = load_yaml(args.inference_config)
    system_prompt = format_system_prompt(
        str(inference_config.get('system_prompt', 'You are Gopi, a helpful assistant. Answer clearly and briefly.')),
        str(inference_config.get('response_format', 'plain')),
    )
    report['system_prompt'] = system_prompt
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for probe in probes:
        prompt = probe['prompt']
        if probe['mode'] == 'chat':
            prompt = format_messages([
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}], add_generation_prompt=True)
        start = time.monotonic()
        result = generator.generate(prompt, **options)
        report['responses'].append({**probe, 'rendered_prompt': prompt, 'response': result.text,
                                    'finish_reason': result.finish_reason,
                                    'completion_tokens': len(result.token_ids),
                                    'seconds': time.monotonic() - start})
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(f'{probe.get("name", probe["mode"])}: {result.text!r}', flush=True)
    lines = ['# Model response probes', '', f'Checkpoint: `{args.checkpoint}`, step {report["step"]}.', '', report['note']]
    for row in report['responses']:
        lines += ['', '## ' + row.get('name', row['mode']), '', '**Prompt**', '', '```text',
                  row['prompt'], '```', '', '**Actual response**', '', '```text', row['response'], '```',
                  '', 'Review criterion: ' + row.get('criterion', 'Coherence and relevance.'),
                  '', f'Finish reason: {row["finish_reason"]}.']
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
