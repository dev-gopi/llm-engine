from evaluation.rag_benchmark import RAGBenchmarkCase, evaluate_rag
from rag.reranker import LexicalCrossEncoderBaseline


class Doc:
    def __init__(self, identifier, text): self.id=identifier; self.text=text; self.url=identifier


def test_rag_benchmark_reports_recall_reranking_faithfulness_and_citations():
    cases=[RAGBenchmarkCase("capital france", frozenset({"fr"}), ("Paris",))]
    docs=[[Doc("fr","Paris is the capital of France."), Doc("uk","London is the capital of the United Kingdom.")]]
    report=evaluate_rag(cases,docs,LexicalCrossEncoderBaseline(),top_k=1,answers=["Paris [fr]"])
    assert report.retrieval_recall_at_k == 1.0
    assert report.reranked_recall_at_k == 1.0
    assert report.faithfulness == 1.0
    assert report.citation_correctness == 1.0
