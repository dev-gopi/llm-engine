from evaluation.hallucination import HallucinationCase, score_response


def test_unknown_requires_abstention():
    c=HallucinationCase('u','unknown','unknown fact','abstain')
    assert score_response(c,"I don't know") == 1.0
