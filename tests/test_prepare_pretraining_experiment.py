import json

from scripts.prepare_pretraining_experiment import sample_records


def test_reservoir_sample_is_reproducible_and_uses_entire_file(tmp_path):
    source = tmp_path / 'source.jsonl'
    rows = [{'text': str(i)} for i in range(100)]
    source.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    first = list(sample_records(source, 10, 42))
    assert first == list(sample_records(source, 10, 42))
    assert len(first) == 10
    assert len({r['text'] for r in first}) == 10
    assert any(int(r['text']) > 50 for r in first)
    assert list(sample_records(source, 0, 42)) == rows
