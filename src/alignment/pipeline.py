"""Preference-data governance and controlled alignment-method comparisons."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import hashlib, re

@dataclass(frozen=True)
class PreferenceAudit:
    valid: bool
    flags: tuple[str,...]
    fingerprint: str
    quality: float

def audit_preference(record: Mapping[str,object]) -> PreferenceAudit:
    p=str(record.get("prompt","")).strip(); c=str(record.get("chosen","")).strip(); r=str(record.get("rejected","")).strip()
    flags=[]
    if not p or not c or not r: flags.append("missing_text")
    if c==r: flags.append("identical_pair")
    if len(c)<8 or len(r)<8: flags.append("short_response")
    if re.search(r"(?:api[_ -]?key|password|secret)\s*[:=]", p+c+r, re.I): flags.append("possible_secret")
    # A pair is not accepted merely because chosen is longer. Length is reported, not rewarded.
    quality=1.0 if not flags else max(0.0,1.0-.25*len(flags))
    fp=hashlib.sha256((p+"\0"+c+"\0"+r).encode()).hexdigest()
    return PreferenceAudit(not flags,tuple(flags),fp,quality)

def audit_preference_set(records):
    seen=set(); audits=[]; duplicates=0
    for r in records:
        a=audit_preference(r); audits.append(a)
        if a.fingerprint in seen: duplicates+=1
        seen.add(a.fingerprint)
    return {"records":len(audits),"valid":sum(a.valid for a in audits),"duplicates":duplicates,"flag_counts":_flags(audits)}

def _flags(audits):
    out={}
    for a in audits:
        for f in a.flags: out[f]=out.get(f,0)+1
    return out

def compare_alignment_methods(results: Mapping[str,Mapping[str,float]], *, metrics: tuple[str,...]=("capability","safety","regression")):
    # Return raw measurements and deltas only; never rank methods.
    if not results: raise ValueError("alignment results cannot be empty")
    baseline=results.get("sft") or next(iter(results.values()))
    out={"baseline":"sft" if "sft" in results else next(iter(results)),"methods":{}}
    for name,values in results.items(): out["methods"][name]={m:float(values[m]) for m in metrics if m in values}
    out["deltas"]={name:{m:float(values[m])-float(baseline[m]) for m in metrics if m in values and m in baseline} for name,values in results.items()}
    return out
