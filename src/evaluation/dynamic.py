"""Generated/dynamic evaluation with deterministic anti-contamination controls."""
from __future__ import annotations

import hashlib
import random


def fixture_seed(version: str, prompt: str) -> int:
    return int(hashlib.sha256((version+"\0"+prompt).encode()).hexdigest()[:16],16)

def generate_math_fixture(version: str, index: int):
    rng=random.Random(fixture_seed(version,str(index)))
    a,b=rng.randint(2,999),rng.randint(2,999)
    return {"id":f"{version}-{index}","prompt":f"What is {a} * {b}?","answer":a*b,"generator_version":version}

def generate_logic_fixture(version: str,index:int):
    rng=random.Random(fixture_seed(version,str(index))); a,b=rng.sample(range(10,99),2)
    return {"id":f"{version}-{index}","prompt":f"If {a} > {b}, is {a} > {b}?","answer":"yes","generator_version":version}

def anti_contamination_check(fixtures, training_fingerprints):
    collisions=[f["id"] for f in fixtures if f.get("id") in training_fingerprints]
    return {"checked":len(fixtures),"collisions":collisions,"passed":not collisions}
