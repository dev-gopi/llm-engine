"""Deterministic beam-search decoder independent of the model backend."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor

@dataclass(frozen=True)
class Beam:
    tokens: tuple[int,...]
    score: float
    finished: bool=False

def beam_search(logits_fn:Callable[[Tensor],Tensor], prompt:Tensor, *, num_beams:int=4, max_new_tokens:int=32, eos_token_id:int|None=None, length_penalty:float=1.0)->tuple[int,...]:
    if prompt.ndim!=1 or num_beams<1 or max_new_tokens<1: raise ValueError("invalid beam-search arguments")
    beams=[Beam(tuple(int(x) for x in prompt.tolist()),0.0,False)]
    for _ in range(max_new_tokens):
        candidates=[]
        active=[b for b in beams if not b.finished]
        if not active: break
        for beam in active:
            ids=torch.tensor(beam.tokens,device=prompt.device,dtype=prompt.dtype).unsqueeze(0)
            logits=logits_fn(ids); logits=logits[0,-1] if logits.ndim==3 else logits[-1]
            logp=torch.log_softmax(logits.float(),dim=-1); values,indices=torch.topk(logp,num_beams)
            for value,index in zip(values.tolist(),indices.tolist()):
                token=beam.tokens+(int(index),); done=eos_token_id is not None and int(index)==eos_token_id
                candidates.append(Beam(token,beam.score+float(value),done))
        candidates.extend(b for b in beams if b.finished)
        beams=sorted(candidates,key=lambda b:b.score/(max(len(b.tokens)-len(prompt),1)**length_penalty),reverse=True)[:num_beams]
    return max(beams,key=lambda b:b.score/(max(len(b.tokens)-len(prompt),1)**length_penalty)).tokens
