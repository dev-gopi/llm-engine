from evaluation.embeddings import HashEmbeddingModel, evaluate_retrieval
from inference.rag import DocumentChunk, embedding_search


def test_embedding_recall_and_rerank():
    model=HashEmbeddingModel(64)
    result=evaluate_retrieval(model,['refund policy','python code'],['refund policy details','unrelated','python code example'],[{0},{2}])
    assert result.recall_at_1 == 1.0

def test_rag_embedding_search():
    model=HashEmbeddingModel(64)
    chunks=[DocumentChunk('a.md','refund policy',1),DocumentChunk('b.md','python code',1)]
    assert embedding_search(chunks,'refund',model,top_k=1)[0].title=='a.md'
