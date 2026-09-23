from local_dataset.quality import quality_source_weights, score_text


def test_quality_score_is_bounded_and_penalizes_repetition():
    good = score_text("This document contains several distinct words and useful information for a model.")
    bad = score_text("spam spam spam spam spam spam spam spam spam")
    assert 0 <= good["score"] <= 1
    assert 0 <= bad["score"] <= 1
    assert good["score"] > bad["score"]

def test_quality_source_weights_are_normalized():
    weights = quality_source_weights({"a": 0.9, "b": 0.6}, {"a": 2, "b": 1})
    assert sum(weights.values()) == 1.0
    assert weights["a"] > weights["b"]
