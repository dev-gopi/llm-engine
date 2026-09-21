"""Governed instruction-data quality scoring and balancing."""
from __future__ import annotations
import hashlib, math, re
from dataclasses import dataclass
from collections import Counter

@dataclass(frozen=True)
class InstructionScore:
    score: float
    category: str
    flags: tuple[str,...]
    components: dict[str,float]
    fingerprint: str

def score_instruction(record: dict) -> InstructionScore:
    prompt=str(record.get("prompt", record.get("instruction", ""))).strip()
    answer=str(record.get("response", record.get("output", ""))).strip()
    category=str(record.get("category", "general")).strip() or "general"
    flags=[]
    if not prompt: flags.append("empty_prompt")
    if not answer: flags.append("empty_answer")
    if len(prompt)>16000: flags.append("prompt_too_long")
    if len(answer)>32000: flags.append("answer_too_long")
    if re.search(r"(?:\b(?:password|secret|api[_ -]?key)\b)\s*[:=]\s*\S+", prompt+"\n"+answer, re.I): flags.append("possible_secret")
    repeated = 1.0
    words=answer.lower().split()
    if words:
        counts=Counter(words); repeated=1.0-min(1.0, max(counts.values())/len(words))
    relevance=1.0 if prompt and answer else 0.0
    completeness=min(1.0, len(answer)/max(32.0, len(prompt)*0.35)) if answer else 0.0
    safety=0.0 if "possible_secret" in flags else 1.0
    score=max(0.0,min(1.0,0.35*relevance+0.25*completeness+0.20*repeated+0.20*safety))
    fp=hashlib.sha256((prompt+"\0"+answer).encode()).hexdigest()
    return InstructionScore(score,category,tuple(flags),{"relevance":relevance,"completeness":completeness,"diversity":repeated,"safety":safety},fp)

def filter_and_balance(records, *, min_score=.65, category_targets=None):
    scored=[(r,score_instruction(r)) for r in records]
    accepted=[x for x in scored if x[1].score>=min_score and not x[1].flags]
    if not category_targets: return [r for r,_ in accepted], {"accepted":len(accepted),"rejected":len(scored)-len(accepted)}
    buckets={k:[] for k in category_targets}
    for r,s in accepted:
        if s.category in buckets: buckets[s.category].append((r,s))
    selected=[]
    for cat,n in category_targets.items(): selected.extend(buckets[cat][:int(n)])
    return [r for r,_ in selected], {"accepted":len(selected),"available":{k:len(v) for k,v in buckets.items()}}
