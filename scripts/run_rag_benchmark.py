"""Run the deterministic RAG recall/reranking benchmark fixtures."""
from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import json

from evaluation.rag_benchmark import RAGBenchmarkCase, evaluate_rag
from rag.reranker import LexicalCrossEncoderBaseline


class Doc:
    def __init__(self, identifier: str, text: str):
        self.id = identifier
        self.text = text
        self.url = identifier


def main() -> None:
    cases = [RAGBenchmarkCase("capital of france", frozenset({"doc-fr"}), ("Paris",))]
    docs = [[Doc("doc-uk", "London is the capital of the United Kingdom."), Doc("doc-fr", "Paris is the capital of France.")]]
    report = evaluate_rag(cases, docs, LexicalCrossEncoderBaseline(), top_k=1, answers=["Paris [doc-fr]"])
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if report.reranked_recall_at_k < report.retrieval_recall_at_k:
        raise SystemExit("RAG reranking regressed recall on the fixture")


if __name__ == "__main__":
    main()
