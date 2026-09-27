#!/usr/bin/env python3
"""Verify configured Omni providers without claiming unavailable capabilities."""
from __future__ import annotations
import json
from omni_platform.capabilities import build_capability_snapshot
from omni_platform.providers import ProviderRegistry
from omni_platform.speech import EnergyVAD, HuggingFaceASRProvider, HuggingFaceTTSProvider, SpeechToSpeechPipeline

registry = ProviderRegistry(); registry.register(EnergyVAD())
asr = HuggingFaceASRProvider.from_env(); tts = HuggingFaceTTSProvider.from_env()
for p in (asr, tts):
    if p is not None: registry.register(p)
if asr is not None and tts is not None: registry.register(SpeechToSpeechPipeline(asr, tts))
snapshot = build_capability_snapshot(provider_capabilities=registry.capability_set())
print(json.dumps({"providers": registry.describe(), **snapshot.as_dict()}, indent=2))
