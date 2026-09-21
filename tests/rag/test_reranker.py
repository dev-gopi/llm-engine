from rag.reranker import LexicalCrossEncoderBaseline

def test_reranker_preserves_metadata_and_orders_relevance():
    class D:
        def __init__(self,url,text): self.url=url; self.text=text
    docs=[D("a","cats and dogs"),D("b","quantum physics"),D("c","cats are mammals")]
    ranked=LexicalCrossEncoderBaseline().rerank("cats",docs,top_k=2)
    assert [r.document.url for r in ranked] == ["a","c"]
    assert [r.rank for r in ranked] == [1,2]
