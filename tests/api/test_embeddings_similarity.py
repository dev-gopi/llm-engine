import torch

from embeddings import EmbeddingService


def test_dedicated_embedding_similarity_is_deterministic_and_semantically_stable_for_contract_fixture():
    service=EmbeddingService()
    result=service.encode(["the capital of France is Paris","the capital of France is Paris","a database stores rows"])
    vectors=torch.tensor(result.embeddings)
    similarity=torch.nn.functional.cosine_similarity(vectors[0:1],vectors[1:2]).item()
    unrelated=torch.nn.functional.cosine_similarity(vectors[0:1],vectors[2:3]).item()
    assert similarity == 1.0
    assert similarity > unrelated
