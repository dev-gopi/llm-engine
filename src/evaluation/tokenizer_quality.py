"""Tokenizer quality benchmark across multilingual, code, JSON, Unicode and numbers."""
from __future__ import annotations

def benchmark_tokenizer(tokenizer, samples:dict[str,list[str]]):
    out={}
    for category,texts in samples.items():
        counts=[]
        for text in texts:
            ids=tokenizer.encode(text,allowed_special="all")
            counts.append({"chars":len(text),"tokens":len(ids),"ratio":len(ids)/max(1,len(text))})
        out[category]={"samples":counts,"mean_tokens_per_char":sum(x["ratio"] for x in counts)/max(1,len(counts))}
    return out
