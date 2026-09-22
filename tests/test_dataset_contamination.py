import json

from datasets.contamination import audit_contamination, load_documents


def write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

def test_detects_exact_and_near_duplicate_cross_split(tmp_path):
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    write(train, [{"text": "A small cat sits quietly on the green mat today."}])
    write(evaluation, [{"text": "A small cat sits quietly on the green mat today."},
                       {"text": "A small cat sits quietly on the green mat today near window."}])
    result = audit_contamination(load_documents([str(train)], "train", 100, 3),
                                 load_documents([str(evaluation)], "evaluation", 100, 3), threshold=0.75)
    assert result["exact_match_count"] == 1
    assert result["near_duplicate_count"] >= 1
    assert result["status"] == "failed"

def test_clean_splits_pass(tmp_path):
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    write(train, [{"text": "alpha beta gamma delta epsilon"}])
    write(evaluation, [{"text": "mountain river cloud forest sunrise"}])
    result = audit_contamination(load_documents([str(train)], "train", 100, 3),
                                 load_documents([str(evaluation)], "evaluation", 100, 3))
    assert result["status"] == "passed"
