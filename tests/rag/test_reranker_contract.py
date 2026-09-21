from dataclasses import dataclass

from rag.reranker import CallableCrossEncoderReranker, ndcg_at_k, reciprocal_rank
from rag.retriever import RetrieverPipeline


@dataclass(frozen=True)
class Doc:
    url: str
    text: str


def test_callable_cross_encoder_preserves_document_and_citation():
    docs = [Doc("a", "wrong"), Doc("b", "right answer"), Doc("c", "also wrong")]
    reranker = CallableCrossEncoderReranker(lambda pairs: [0.1, 0.9, 0.2])
    ranked = reranker.rerank("question", docs, top_k=2)
    assert [item.document.url for item in ranked] == ["b", "c"]
    assert ranked[0].citation == "b"
    assert ranked[0].rank == 1


def test_ranking_metrics_are_reproducible():
    docs = [Doc("a", "a"), Doc("b", "b"), Doc("c", "c")]
    ranked = CallableCrossEncoderReranker(lambda pairs: [0.3, 0.9, 0.2]).rerank("q", docs, top_k=3)
    relevant = {"b"}
    assert reciprocal_rank(ranked, relevant) == 1.0
    assert ndcg_at_k(ranked, relevant, 3) == 1.0


def test_retriever_requires_candidate_pool_at_least_top_k():
    class Retriever:
        def search(self, query, *, top_k):
            return [Doc("a", "x")]

    pipeline = RetrieverPipeline(Retriever())
    try:
        pipeline.search("q", candidate_k=1, top_k=2)
    except ValueError as error:
        assert "candidate_k" in str(error)
    else:
        raise AssertionError("expected candidate_k validation")

def test_reranking_quality_is_measurable_against_candidate_recall():
    candidates = [Doc("distractor", "noise"), Doc("target", "answer")]
    ranked = CallableCrossEncoderReranker(lambda pairs: [0.1, 0.9]).rerank("question", candidates, top_k=1)
    evaluation = RetrieverPipeline.evaluate(candidates, ranked, {"target"}, k=1)
    assert evaluation.recall_at_k == 1.0
    assert evaluation.reranked_recall_at_k == 1.0
    assert evaluation.mrr == 1.0
    assert evaluation.ndcg_at_k == 1.0
