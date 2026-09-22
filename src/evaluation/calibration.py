"""Calibration metrics for confidence versus correctness."""
from __future__ import annotations


def calibration_metrics(confidences, correct, bins=10):
    if len(confidences)!=len(correct) or not confidences: raise ValueError("confidence/correctness lengths must match and be non-empty")
    if bins<1: raise ValueError("bins must be positive")
    n=len(confidences); ece=0.0; brier=0.0; rows=[]
    for i in range(bins):
        lo=i/bins; hi=(i+1)/bins; idx=[j for j,c in enumerate(confidences) if lo <= c < hi or (i==bins-1 and c==hi)]
        if not idx: continue
        acc=sum(bool(correct[j]) for j in idx)/len(idx); conf=sum(float(confidences[j]) for j in idx)/len(idx)
        ece += len(idx)/n*abs(acc-conf); rows.append({"bin":i,"count":len(idx),"accuracy":acc,"confidence":conf})
    brier=sum((float(c)-float(bool(y)))**2 for c,y in zip(confidences,correct))/n
    return {"ece":ece,"brier":brier,"bins":rows}
