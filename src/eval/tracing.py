"""LangSmith tracing — every agent run becomes an inspectable trace.

LangChain/LangGraph auto-emit traces to LangSmith when the env vars are set. We
load them from .env via src.config; this module just makes the wiring explicit and
exposes a status helper for the Gradio UI.

Enable by setting in .env (free tier: https://smith.langchain.com):
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls__...
    LANGCHAIN_PROJECT=multi-agent-analyst
"""
from __future__ import annotations

import os

from src import config


def enable_langsmith() -> bool:
    """Ensure tracing env vars are exported. Returns True if tracing is active."""
    if str(config.LANGSMITH_TRACING).lower() == "true" and os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", config.LANGSMITH_PROJECT)
        return True
    return False


def status() -> str:
    if enable_langsmith():
        return f"LangSmith tracing ON → project '{config.LANGSMITH_PROJECT}'"
    return ("LangSmith tracing OFF (set LANGCHAIN_TRACING_V2=true and "
            "LANGCHAIN_API_KEY in .env to enable full agent traces)")


if __name__ == "__main__":
    print(status())
