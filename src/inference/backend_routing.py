"""Health-aware inference backend routing, failover, and tenant QoS."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import heapq
import itertools
import threading
import time
from typing import Any, Callable

class CircuitState(str, Enum):
    CLOSED="closed"; OPEN="open"; HALF_OPEN="half_open"

@dataclass
class CircuitBreaker:
    failure_threshold:int=5
    recovery_seconds:float=30.0
    success_threshold:int=2
    state:CircuitState=CircuitState.CLOSED
    failures:int=0
    half_open_successes:int=0
    opened_at:float|None=None
    def __post_init__(self):
        if self.failure_threshold<1 or self.success_threshold<1 or self.recovery_seconds<0: raise ValueError("invalid circuit-breaker policy")
    def allow(self, now:float|None=None)->bool:
        now=time.monotonic() if now is None else now
        if self.state is CircuitState.OPEN and self.opened_at is not None and now-self.opened_at>=self.recovery_seconds:
            self.state=CircuitState.HALF_OPEN; self.half_open_successes=0
        return self.state is not CircuitState.OPEN
    def success(self)->None:
        if self.state is CircuitState.HALF_OPEN:
            self.half_open_successes+=1
            if self.half_open_successes>=self.success_threshold:
                self.state=CircuitState.CLOSED; self.failures=0; self.opened_at=None
        else: self.failures=0
    def failure(self, now:float|None=None)->None:
        now=time.monotonic() if now is None else now
        self.failures+=1
        if self.state is CircuitState.HALF_OPEN or self.failures>=self.failure_threshold:
            self.state=CircuitState.OPEN; self.opened_at=now; self.half_open_successes=0

@dataclass
class BackendEndpoint:
    name:str
    generate:Callable[...,Any]
    priority:int=100
    breaker:CircuitBreaker=field(default_factory=CircuitBreaker)
    failures:int=0
    requests:int=0

class FailoverRouter:
    def __init__(self, endpoints:list[BackendEndpoint], *, retryable:tuple[type[BaseException],...]=(RuntimeError,TimeoutError,ConnectionError)):
        if not endpoints: raise ValueError("at least one backend endpoint is required")
        self.endpoints=sorted(endpoints,key=lambda e:(e.priority,e.name)); self.retryable=retryable
    def generate(self,*args,**kwargs):
        errors=[]
        for endpoint in self.endpoints:
            if not endpoint.breaker.allow(): continue
            endpoint.requests+=1
            try:
                value=endpoint.generate(*args,**kwargs); endpoint.breaker.success(); return value
            except self.retryable as exc:
                endpoint.failures+=1; endpoint.breaker.failure(); errors.append((endpoint.name,exc))
        detail="; ".join(f"{name}: {exc}" for name,exc in errors) or "all backend circuits are open"
        raise RuntimeError(f"all inference backends failed: {detail}")

class TenantQoSQueue:
    """Thread-safe weighted priority queue with per-tenant in-flight caps."""
    def __init__(self, *, default_limit:int=8):
        if default_limit<1: raise ValueError("default_limit must be positive")
        self.default_limit=default_limit; self._limits={}; self._active={}; self._heap=[]; self._seq=itertools.count(); self._lock=threading.Lock()
    def set_limit(self, tenant:str, limit:int)->None:
        if limit<1: raise ValueError("limit must be positive")
        with self._lock: self._limits[tenant]=limit
    def submit(self, tenant:str, payload:Any, *, priority:int=100, weight:float=1.0)->None:
        if not tenant or weight<=0: raise ValueError("tenant and positive weight are required")
        with self._lock: heapq.heappush(self._heap,(priority/weight,next(self._seq),tenant,payload))
    def acquire(self):
        with self._lock:
            skipped=[]; selected=None
            while self._heap:
                item=heapq.heappop(self._heap); tenant=item[2]; limit=self._limits.get(tenant,self.default_limit)
                if self._active.get(tenant,0)<limit:
                    self._active[tenant]=self._active.get(tenant,0)+1; selected=(tenant,item[3]); break
                skipped.append(item)
            for item in skipped: heapq.heappush(self._heap,item)
            return selected
    def release(self, tenant:str)->None:
        with self._lock:
            current=self._active.get(tenant,0)
            if current<1: raise RuntimeError("tenant has no acquired request")
            if current==1: self._active.pop(tenant,None)
            else: self._active[tenant]=current-1

__all__=["CircuitState","CircuitBreaker","BackendEndpoint","FailoverRouter","TenantQoSQueue"]
