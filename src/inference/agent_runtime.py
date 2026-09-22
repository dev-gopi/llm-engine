"""Bounded, approval-gated agent state machine."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentState(str, Enum):
    PLANNING="planning"; WAITING_APPROVAL="waiting_approval"; EXECUTING="executing"; OBSERVING="observing"; COMPLETED="completed"; FAILED="failed"

@dataclass
class AgentStep:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    approved: bool = False
    result: Any = None

class BoundedAgentRuntime:
    def __init__(self, tools: dict[str, Callable[..., Any]], *, max_steps: int = 8, require_approval: bool = True, allowed_tools: set[str] | None = None):
        if max_steps < 1: raise ValueError("max_steps must be positive")
        self.tools=dict(tools); self.max_steps=max_steps; self.require_approval=require_approval
        self.allowed_tools=set(allowed_tools) if allowed_tools is not None else set(tools)
        unknown=self.allowed_tools-set(self.tools)
        if unknown: raise ValueError(f"unknown allowed tools: {sorted(unknown)}")
        self.state=AgentState.PLANNING; self.steps=[]; self.observations=[]

    def plan(self, steps: list[AgentStep]) -> None:
        if self.state not in {AgentState.PLANNING, AgentState.FAILED}: raise RuntimeError("runtime is not accepting a plan")
        if not steps or len(steps)>self.max_steps: raise ValueError("plan exceeds bounded step limit")
        if any(step.name not in self.allowed_tools for step in steps): raise ValueError("plan contains a tool outside the allowlist")
        self.steps=list(steps); self.state=AgentState.WAITING_APPROVAL if self.require_approval else AgentState.EXECUTING

    def approve(self) -> None:
        if self.state != AgentState.WAITING_APPROVAL: raise RuntimeError("no plan is waiting for approval")
        for step in self.steps: step.approved=True
        self.state=AgentState.EXECUTING

    def reject(self, reason: str = "human approval rejected") -> None:
        if self.state != AgentState.WAITING_APPROVAL: raise RuntimeError("no plan is waiting for approval")
        self.state=AgentState.FAILED; self.observations.append({"type":"approval_rejected","reason":reason})

    def run(self) -> list[AgentStep]:
        if self.state != AgentState.EXECUTING: raise RuntimeError("runtime must be approved before execution")
        for step in self.steps:
            if self.require_approval and not step.approved: raise RuntimeError("every step requires approval")
            self.state=AgentState.EXECUTING
            try: step.result=self.tools[step.name](**step.arguments)
            except Exception as error:
                step.result={"error":str(error)}; self.observations.append({"step":step.name,"status":"error","error":str(error)}); self.state=AgentState.FAILED; return self.steps
            self.observations.append({"step":step.name,"status":"ok","result":step.result}); self.state=AgentState.OBSERVING
        self.state=AgentState.COMPLETED
        return self.steps

    def recover(self, replacement: list[AgentStep]) -> None:
        if self.state != AgentState.FAILED: raise RuntimeError("recovery is only available after failure")
        self.state=AgentState.PLANNING; self.plan(replacement)

    def snapshot(self) -> dict[str, Any]:
        return {"state":self.state.value,"max_steps":self.max_steps,"steps":[{"name":s.name,"arguments":s.arguments,"approved":s.approved,"result":s.result} for s in self.steps],"observations":list(self.observations)}
