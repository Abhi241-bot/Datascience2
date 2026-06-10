"""Shared Groq LLM factory + robust JSON parsing for the agents.

One place to construct the orchestrator model so every agent uses the same config.
"""
from __future__ import annotations

import functools
import json
import re
from typing import Any

from src import config


@functools.lru_cache(maxsize=4)
def get_llm(temperature: float | None = None):
    """Return a cached ChatGroq instance."""
    from langchain_groq import ChatGroq

    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free: https://console.groq.com/keys)."
        )
    return ChatGroq(
        model=config.ORCHESTRATOR_MODEL,
        temperature=config.ORCHESTRATOR_TEMPERATURE if temperature is None else temperature,
        api_key=config.GROQ_API_KEY,
    )


def chat(prompt: str, temperature: float | None = None) -> str:
    """Single-shot completion; returns the text content."""
    resp = get_llm(temperature).invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


def parse_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response (handles ```json fences)."""
    text = text.strip()
    # strip code fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    # grab the first {...} or [...] block
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    candidate = m.group(1) if m else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # last resort: trailing-comma cleanup
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(cleaned)
