"""Authenticated production reward-scoring HTTP service with batching and hot reload."""
from __future__ import annotations
import asyncio, json, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

class RewardRequest(BaseModel):
    prompt: str = Field(min_length=1)
    completion: str = Field(min_length=1)
    verifier: str = "default"
    calibration_version: str | None = None

class RewardResponse(BaseModel):
    reward: float
    verifier: str
    calibration_version: str

class RewardBatchRequest(BaseModel):
    requests: list[RewardRequest] = Field(min_length=1, max_length=256)

class RewardBatchResponse(BaseModel):
    results: list[RewardResponse]

@dataclass
class RewardServiceState:
    scorer: Callable[[str,str,str], float]
    api_key: str | None = None
    calibration_file: Path | None = None
    calibration_version: str = "default"
    _mtime: float | None = None
    def reload_calibration(self) -> None:
        if self.calibration_file is None or not self.calibration_file.exists(): return
        mtime=self.calibration_file.stat().st_mtime
        if self._mtime == mtime: return
        payload=json.loads(self.calibration_file.read_text(encoding="utf8"))
        self.calibration_version=str(payload.get("version", self.calibration_file.stem))
        self._mtime=mtime

def create_reward_app(state: RewardServiceState, *, max_batch: int = 64) -> FastAPI:
    if max_batch < 1: raise ValueError("max_batch must be positive")
    app=FastAPI(title="LLM Engine Reward Service", version="1")
    @app.get("/health")
    async def health():
        state.reload_calibration(); return {"status":"ok", "calibration_version":state.calibration_version}
    @app.post("/v1/rewards/batch", response_model=RewardBatchResponse)
    async def score_batch(request: RewardBatchRequest, authorization: str | None = Header(default=None)):
        if state.api_key is not None and authorization != f"Bearer {state.api_key}": raise HTTPException(401,"invalid credentials")
        state.reload_calibration()
        results=[]
        for item in request.requests[:max_batch]:
            if item.calibration_version and item.calibration_version != state.calibration_version:
                raise HTTPException(409, "requested calibration version is not loaded")
            try: reward=float(state.scorer(item.prompt,item.completion,item.verifier))
            except Exception as exc: raise HTTPException(422,str(exc)) from exc
            results.append(RewardResponse(reward=reward,verifier=item.verifier,calibration_version=state.calibration_version))
        return RewardBatchResponse(results=results)
    @app.post("/v1/rewards", response_model=RewardResponse)
    async def score(request: RewardRequest, authorization: str | None = Header(default=None)):
        if state.api_key is not None and authorization != f"Bearer {state.api_key}": raise HTTPException(401,"invalid credentials")
        state.reload_calibration()
        if request.calibration_version and request.calibration_version != state.calibration_version:
            raise HTTPException(409, "requested calibration version is not loaded")
        try: reward=float(state.scorer(request.prompt, request.completion, request.verifier))
        except Exception as exc: raise HTTPException(422, str(exc)) from exc
        return RewardResponse(reward=reward, verifier=request.verifier, calibration_version=state.calibration_version)
    return app
