"""Deterministic difficulty curriculum for reasoning SFT."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class CurriculumItem:
    id: str
    difficulty: float
    domain: str
    payload: dict

def difficulty_of(item: dict) -> float:
    if "difficulty" in item: return float(item["difficulty"])
    text=str(item.get("prompt",item.get("question","")))
    return min(1.0, 0.1 + len(text)/4000.0)

def order_curriculum(items, *, buckets=5):
    if buckets < 1: raise ValueError("buckets must be positive")
    values=[CurriculumItem(str(x.get("id",i)),difficulty_of(x),str(x.get("domain","general")),x) for i,x in enumerate(items)]
    values.sort(key=lambda x:(x.difficulty,x.domain,x.id))
    return values

def curriculum_weights(items, *, temperature=1.0):
    if temperature<=0: raise ValueError("temperature must be positive")
    ordered=order_curriculum(items)
    return {x.id: (x.difficulty+1e-6)**temperature for x in ordered}
