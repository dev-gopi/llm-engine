"""Conservative prompt checks applied before tools, retrieval, or generation."""

from __future__ import annotations

import re
import unicodedata


PROMPT_INJECTION_REFUSAL = "I can't follow instructions that try to override or reveal my safety rules."
UNSAFE_REQUEST_REFUSAL = "I can't help with a request that could cause serious harm or violate someone's safety."

_LEET_TRANSLATION = str.maketrans({"@": "a", "4": "a", "3": "e", "1": "i", "!": "i", "0": "o", "5": "s", "$": "s", "7": "t"})

_INJECTION_PATTERNS = (
    r"\b(?:ignore|disregard|forget|override)\b.{0,50}\b(?:previous|prior|above|system|developer|safety)\b.{0,30}\b(?:instructions?|prompts?|rules?|message)\b",
    r"\b(?:ignore|disregard|override)\b.{0,30}\b(?:your|all|safety)\b.{0,20}\b(?:instructions?|rules?|guardrails?)\b",
    r"\b(?:reveal|show|print|repeat|leak|expose)\b.{0,50}\b(?:system|developer|hidden|initial)\b.{0,20}\b(?:prompt|instructions?|message)\b",
    r"\b(?:jailbreak|developer mode|god mode|unrestricted mode|do anything now|\bdan\b)\b",
    r"\bpretend (?:that )?you (?:have no|do not have|don't have) (?:rules|restrictions|safety)\b",
    r"\b(?:bypass|disable|remove|evade)\b.{0,40}\b(?:guardrails?|filters?|moderation|safety|restrictions?)\b",
    r"\b(?:act as|you are now)\b.{0,40}\b(?:unfiltered|unrestricted|jailbroken|without rules)\b",
)

_HARMFUL_PATTERNS = (
    # Violence, weapons, and self-harm instructions.
    r"\b(?:how (?:do|can|to)|instructions? (?:for|to)|steps? (?:for|to)|best way to)\b.{0,100}\b(?:kill|murder|poison|torture|kidnap|seriously hurt|attack|commit suicide|self harm)\b",
    r"\b(?:build|make|assemble|detonate|hide|obtain)\b.{0,60}\b(?:bomb|explosive|silencer|untraceable (?:gun|weapon)|bioweapon)\b",
    r"\b(?:encourage|glorify|help me|ways? to)\b.{0,50}\b(?:suicide|self[ -]?harm|cut myself)\b",
    # Cyber abuse and unauthorized access.
    r"\b(?:write|make|create|deploy|spread|install|hide)\b.{0,60}\b(?:malware|ransomware|keylogger|botnet|spyware|computer virus|rootkit)\b",
    r"\b(?:hack|breach|ddos|take down|gain access to)\b.{0,60}\b(?:account|server|website|network|computer|phone|wifi)\b",
    r"\b(?:steal|harvest|capture|phish for|crack|bypass)\b.{0,60}\b(?:passwords?|credentials?|authentication|bank accounts?|two[ -]?factor|2fa)\b",
    # Fraud, privacy abuse, and illicit drugs.
    r"\b(?:create|write|run|set up|help with)\b.{0,60}\b(?:phishing|scam|identity theft|fraud|counterfeit|money laundering)\b",
    r"\b(?:dox|stalk|track|locate|expose)\b.{0,50}\b(?:someone|person|address|location|private information|personal data)\b",
    r"\b(?:make|cook|synthesize|manufacture|sell|traffic)\b.{0,50}\b(?:meth|methamphetamine|cocaine|heroin|fentanyl|illegal drugs?)\b",
    # Sexual exploitation and explicit sexual requests.
    r"\b(?:sexual|pornographic|explicit)\b.{0,30}\b(?:child|minor|underage)\b",
    r"\b(?:sex(?:ual(?:ly|ity)?|ting|f|y)?|porn[a-z]*|nudes?|nsfw|erotic[a-z]*|fetish|incest|rape)\b",
    # Targeted hate or dehumanization.
    r"\b(?:write|create|spread|promote)\b.{0,50}\b(?:racist|hateful|supremacist|genocidal)\b.{0,30}\b(?:propaganda|message|content|speech)\b",
    r"\b(?:kill|eliminate|exterminate|deport) all\b.{0,40}\b(?:race|religion|ethnicity|gay|trans|immigrants?|people)\b",
    r"\b(?:are|is)\b.{0,30}\b(?:subhuman|vermin|animals that should die)\b",
)


def blocked_prompt_message(prompt: str) -> str | None:
    """Return a safe refusal for a high-confidence unsafe prompt, otherwise ``None``."""
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())
    normalized = normalized.translate(_LEET_TRANSLATION)
    if any(re.search(pattern, normalized, flags=re.DOTALL) for pattern in _INJECTION_PATTERNS):
        return PROMPT_INJECTION_REFUSAL
    if any(re.search(pattern, normalized, flags=re.DOTALL) for pattern in _HARMFUL_PATTERNS):
        return UNSAFE_REQUEST_REFUSAL
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    if len(compact) <= 16 and re.fullmatch(r"sex(?:ual(?:ly|ity)?|ting|f|y)?", compact):
        return UNSAFE_REQUEST_REFUSAL
    return None
