"""Prepare a reproducible local pretraining pilot with held-out exclusion and lossless packing.

Defaults to a bounded, uniformly sampled pilot. Use --train-records 0 and
--validation-records 0 for full source splits. No source files are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from dataclasses import asdict
from datasets.filters import CorpusFilter
from datasets.loader import iter_records
from datasets.preprocessor import record_to_text

DOMAINS = ('tinystories', 'wikitext_103', 'fineweb_edu', 'code_pretraining')


def sample_records(path: Path, limit: int, seed: int):
    """Reservoir sample raw lines so unused large records need not be parsed."""
    if limit == 0:
        yield from iter_records(path)
        return
    rng = random.Random(seed)
    sample = []
    count = 0
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            count += 1
            if len(sample) < limit:
                sample.append((number, line))
            else:
                index = rng.randrange(count)
                if index < limit:
                    sample[index] = (number, line)
    for _, line in sorted(sample):
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f'{path}: record must be an object')
        yield record


def held_out_texts(root: Path):
    for domain in DOMAINS:
        for split in ('validation', 'test'):
            path = root / domain / f'{split}.jsonl'
            if path.is_file():
                for record in iter_records(path):
                    yield record_to_text(record)


def prepare(root: Path, output: Path, tokenizer: Path, train_records: int,
            validation_records: int, seed: int, sequence_length: int):
    if output.resolve() == root.resolve() or root.resolve() in output.resolve().parents:
        raise ValueError('output must be separate from the source dataset tree')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'output is not empty: {output}; choose a new directory')
    for domain in DOMAINS:
        for name in ('train.jsonl', 'validation.jsonl', 'dataset-manifest.yaml'):
            if not (root / domain / name).is_file():
                raise FileNotFoundError(root / domain / name)
    output.mkdir(parents=True, exist_ok=True)
    # The single training filter deduplicates across domains as well as within
    # each source. Exclusions cover entire original held-out splits, not samples.
    training_filter = CorpusFilter(excluded_texts=held_out_texts(root),
                                   max_fingerprints=5_000_000, preserve_whitespace=True,
                                   redact_pii=False)
    validation_filter = CorpusFilter(max_fingerprints=5_000_000,
                                     preserve_whitespace=True, redact_pii=False)
    summary = {'seed': seed, 'train_candidate_limit_per_domain': train_records,
               'validation_candidate_limit_per_domain': validation_records,
               'sequence_length': sequence_length, 'domains': {},
               'limitations': ['Document-level exact and SimHash near-duplicate exclusion; not exhaustive substring detection.',
                               'Existing source license/privacy review states are retained.',
                               'Quality heuristics do not verify factual correctness.']}
    for domain in DOMAINS:
        destination = output / domain
        destination.mkdir()
        shutil.copy2(root / domain / 'dataset-manifest.yaml', destination / 'dataset-manifest.yaml')
        summary['domains'][domain] = {}
        for split, limit, corpus_filter in (
            ('validation', validation_records, validation_filter),
            ('train', train_records, training_filter),
        ):
            source = root / domain / f'{split}.jsonl'
            clean = destination / f'{split}.clean.jsonl'
            digest = hashlib.sha256()
            before = asdict(corpus_filter.stats)
            count = accepted = 0
            with clean.open('w', encoding='utf-8') as stream:
                for record in sample_records(source, limit, seed):
                    count += 1
                    text = corpus_filter.apply(record_to_text(record))
                    if text is None:
                        continue
                    line = json.dumps({'text': text, 'source': domain,
                                       'source_id': record.get('id')}, ensure_ascii=False) + '\n'
                    stream.write(line)
                    digest.update(line.encode())
                    accepted += 1
            if accepted == 0:
                raise ValueError(f'no accepted records: {source}')
            packed = destination / f'{split}.packed.jsonl'
            subprocess.run([sys.executable, str(Path(__file__).with_name('pack_jsonl_corpus.py')),
                            str(clean), '--output', str(packed), '--tokenizer', str(tokenizer),
                            '--sequence-length', str(sequence_length)], check=True)
            metadata = json.loads(packed.with_suffix('.packing.json').read_text())
            stats = {k: v - before[k] for k, v in asdict(corpus_filter.stats).items()}
            summary['domains'][domain][split] = {
                'source': str(source), 'source_size_bytes': source.stat().st_size,
                'source_mtime_ns': source.stat().st_mtime_ns, 'sampled_records': count,
                'accepted_records': accepted, 'clean_sha256': digest.hexdigest(),
                'filter_stats': stats, 'packing': metadata,
            }
            (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(f'{domain}/{split}: accepted={accepted}/{count} packed={metadata["output_records"]}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/processed'))
    parser.add_argument('--output', type=Path, default=Path('data/cleaned/pretraining-pilot'))
    parser.add_argument('--tokenizer', type=Path, default=Path('data/tokenizer'))
    parser.add_argument('--train-records', type=int, default=512)
    parser.add_argument('--validation-records', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sequence-length', type=int, default=512)
    args = parser.parse_args()
    if min(args.train_records, args.validation_records) < 0 or args.sequence_length < 2:
        parser.error('record limits must be non-negative and sequence length at least 2')
    prepare(args.data_root, args.output, args.tokenizer, args.train_records,
            args.validation_records, args.seed, args.sequence_length)


if __name__ == '__main__':
    main()
