"""Cancellation and failure cleanup contract for inference requests."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RequestLifecycle:
    request_id:str; state:str="running"; resources:int=0
    def cancel(self): self.state="cancelled"; self.resources=0
    def timeout(self): self.state="timed_out"; self.resources=0
    def oom(self): self.state="oom"; self.resources=0
    def server_restart(self): self.state="restarted"; self.resources=0
    def complete(self): self.state="completed"; self.resources=0

def stress_summary(events):
    leaked=[e.request_id for e in events if e.state in {"cancelled","timed_out","oom","restarted"} and e.resources]
    return {"events":len(events),"leaked_resources":leaked,"passed":not leaked}
