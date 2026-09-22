"""Dedicated embedding and retrieval evaluation primitives."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class EmbeddingModel(Protocol):
    dimension: int
    def encode(self, texts: Sequence[str]) -> Tensor: ...

class HashEmbeddingModel:
    """Deterministic dependency-free sentence embedding baseline."""
    def __init__(self, dimension: int = 256):
        if dimension < 8: raise ValueError("dimension must be >= 8")
        self.dimension = dimension
    def encode(self, texts: Sequence[str]) -> Tensor:
        rows=[]
        for text in texts:
            vec=torch.zeros(self.dimension, dtype=torch.float32)
            for token in str(text).casefold().split():
                h=hashlib_sha(token)
                vec[h % self.dimension] += 1.0
                vec[(h >> 8) % self.dimension] += 0.5
            norm=vec.norm()
            rows.append(vec / norm if norm else vec)
        return torch.stack(rows) if rows else torch.empty((0,self.dimension))

def hashlib_sha(value: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")

def cosine_recall_at_k(query_embeddings: Tensor, document_embeddings: Tensor, relevant: Sequence[set[int]], k: int = 5) -> float:
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2 or query_embeddings.size(1) != document_embeddings.size(1):
        raise ValueError("embedding matrices must be 2D with equal dimensions")
    if len(relevant) != query_embeddings.size(0) or k < 1: raise ValueError("invalid relevance labels or k")
    q=torch.nn.functional.normalize(query_embeddings.float(), dim=-1); d=torch.nn.functional.normalize(document_embeddings.float(), dim=-1)
    hits=0
    for i, target in enumerate(relevant):
        top=torch.matmul(q[i], d.T).topk(min(k,d.size(0))).indices.tolist()
        hits += int(bool(target.intersection(top)))
    return hits / max(len(relevant),1)

@dataclass(frozen=True)
class RetrievalBenchmark:
    recall_at_1: float
    recall_at_5: float
    rerank_recall_at_5: float
    queries: int
    embedding_dimension: int

    def to_dict(self): return self.__dict__.copy()

def evaluate_retrieval(model: EmbeddingModel, queries: Sequence[str], documents: Sequence[str], relevant: Sequence[set[int]], *, reranker=None) -> RetrievalBenchmark:
    q=model.encode(queries); d=model.encode(documents)
    r1=cosine_recall_at_k(q,d,relevant,1); r5=cosine_recall_at_k(q,d,relevant,5)
    rerank=r5
    if reranker is not None:
        hits=0
        for i, targets in enumerate(relevant):
            scores=reranker(queries[i], documents)
            order=sorted(range(len(documents)), key=lambda j: scores[j], reverse=True)[:5]
            hits += int(bool(targets.intersection(order)))
        rerank=hits/max(len(relevant),1)
    return RetrievalBenchmark(r1,r5,rerank,len(queries),model.dimension)
