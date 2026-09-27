"""Optional inference backend adapters with strict capability detection."""
from __future__ import annotations

from dataclasses import dataclass
import asyncio
import importlib.util
from typing import Any
from abc import ABC, abstractmethod

class BackendUnavailableError(RuntimeError): pass

@dataclass(frozen=True)
class BackendCapability:
    name:str; available:bool; reason:str|None=None; version:str|None=None

def detect_backend(name:str)->BackendCapability:
    name=name.lower().strip(); modules={"vllm":"vllm","tensorrt_llm":"tensorrt_llm","flashinfer":"flashinfer","flash_attn":"flash_attn","flashattention":"flash_attn"}
    if name not in modules: raise ValueError(f"unknown inference backend: {name}")
    module=modules[name]; spec=importlib.util.find_spec(module)
    if spec is None: return BackendCapability(name,False,f"optional dependency {module!r} is not installed")
    try:
        mod=__import__(module); return BackendCapability(name,True,version=getattr(mod,"__version__",None))
    except Exception as exc: return BackendCapability(name,False,f"dependency import failed: {exc}")

class OptionalInferenceBackend(ABC):
    name="base"
    def __init__(self,**options):
        self.options=dict(options); cap=detect_backend(self.name)
        if not cap.available: raise BackendUnavailableError(f"{self.name}: {cap.reason}")
        self.capability=cap
    @abstractmethod
    def generate(self,prompts:list[str],**kwargs): pass

class VLLMBackend(OptionalInferenceBackend):
    name="vllm"
    def __init__(self,model:str,**options):
        self.model=model; super().__init__(**options)
        from vllm import LLM
        self._engine=LLM(model=model,**options)
    def generate(self,prompts:list[str],**kwargs):
        from vllm import SamplingParams
        return self._engine.generate(prompts,SamplingParams(**kwargs))
    def chat(self,conversations:list[list[dict[str,Any]]],**kwargs):
        from vllm import SamplingParams
        chat=getattr(self._engine,"chat",None)
        if not callable(chat): raise BackendUnavailableError("installed vLLM does not expose LLM.chat")
        return chat(conversations,SamplingParams(**kwargs))

class VLLMAsyncBackend(OptionalInferenceBackend):
    """Native asynchronous vLLM token/event backend."""
    name="vllm"
    def __init__(self,model:str,**options):
        self.model=model; super().__init__(**options)
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
        except ImportError as exc: raise BackendUnavailableError("installed vLLM lacks AsyncLLMEngine") from exc
        self._engine=AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=model,**options))
    async def stream(self,prompt:str,*,request_id:str,**kwargs):
        from vllm import SamplingParams
        async for item in self._engine.generate(prompt,SamplingParams(**kwargs),request_id): yield item
    async def generate_async(self,prompt:str,*,request_id:str,**kwargs):
        last=None
        async for last in self.stream(prompt,request_id=request_id,**kwargs): pass
        return last

class TensorRTLLMBackend(OptionalInferenceBackend):
    name="tensorrt_llm"
    def __init__(self,engine:str,*,tokenizer=None,runner=None,**options):
        self.engine=engine; super().__init__(**options)
        import tensorrt_llm
        self._runtime=tensorrt_llm; self.tokenizer=tokenizer
        if runner is not None: self.runner=runner; return
        runner_cls=None
        try:
            from tensorrt_llm.runtime import ModelRunnerCpp as runner_cls
        except Exception:
            try:
                from tensorrt_llm.runtime import ModelRunner as runner_cls
            except Exception as exc: raise BackendUnavailableError("TensorRT-LLM runtime exposes no supported ModelRunner") from exc
        factory=getattr(runner_cls,"from_dir",None)
        self.runner=factory(engine,**options) if callable(factory) else runner_cls(engine,**options)
    def generate(self,prompts:list[str],**kwargs):
        if self.tokenizer is None: raise BackendUnavailableError("TensorRT-LLM generation requires a tokenizer")
        encoded=[self.tokenizer.encode(p) for p in prompts]
        generate=getattr(self.runner,"generate",None)
        if not callable(generate): raise BackendUnavailableError("TensorRT-LLM runner exposes no generate()")
        outputs=generate(encoded,**kwargs)
        # Preserve raw output when the runner already returns text/structured results.
        if isinstance(outputs,list) and outputs and all(isinstance(x,str) for x in outputs): return outputs
        decode=getattr(self.tokenizer,"decode",None)
        if not callable(decode): return outputs
        try: return [decode(x) for x in outputs]
        except Exception: return outputs

class FlashInferBackend(OptionalInferenceBackend):
    name="flashinfer"
    def __init__(self,**options): super().__init__(**options); import flashinfer; self.runtime=flashinfer
    def single_prefill(self,q,k,v,*,causal=True,**kwargs):
        fn=getattr(self.runtime,"single_prefill_with_kv_cache",None)
        if not callable(fn): raise BackendUnavailableError("flashinfer single-prefill API is unavailable")
        return fn(q,k,v,causal=causal,**kwargs)

class FlashAttentionBackend(OptionalInferenceBackend):
    name="flash_attn"
    def __init__(self,**options): super().__init__(**options); import flash_attn; self.runtime=flash_attn
    def attention(self,q,k,v,*,dropout_p:float=0.0,causal:bool=True,softmax_scale=None):
        fn=getattr(self.runtime,"flash_attn_func",None)
        if not callable(fn): raise BackendUnavailableError("flash_attn_func is unavailable")
        return fn(q,k,v,dropout_p=dropout_p,softmax_scale=softmax_scale,causal=causal)

def backend_matrix(): return {name:detect_backend(name) for name in ("vllm","tensorrt_llm","flashinfer","flash_attn")}

__all__=["BackendCapability","BackendUnavailableError","OptionalInferenceBackend","VLLMBackend","VLLMAsyncBackend","TensorRTLLMBackend","FlashInferBackend","FlashAttentionBackend","backend_matrix","detect_backend"]
