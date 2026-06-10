"""Stateful memory — the typed state object carried across graph nodes.

LangGraph threads one state dict through every node. Annotated reducers let nodes
*append* to the reasoning log / findings instead of overwriting, so memory
accumulates across the planner -> researcher (loop) -> analyst flow.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


class Finding(TypedDict):
    """One piece of evidence gathered by a tool, with provenance for citation."""
    tool: str                 # retrieval | web_search | text_to_sql
    source: str               # filename / URL / "SQL:<query>"
    content: str              # human-readable evidence text
    raw: dict                 # the tool's full structured output


class AnalystState(TypedDict, total=False):
    # ── input ──
    question: str

    # ── planner output ──
    plan: list[str]                              # human-readable step list
    actions: list[dict]                          # [{tool, input, reason}, ...]
    is_sql_shaped: bool                          # question needs the DB / text_to_sql

    # ── researcher memory (accumulates across loops) ──
    findings: Annotated[list[Finding], operator.add]
    reasoning_log: Annotated[list[str], operator.add]   # streamed to the UI
    loop_count: int
    needs_more: bool

    # ── HITL ──
    approved: bool                               # set by the human at the checkpoint
    human_feedback: str

    # ── analyst output ──
    report: str
    citations: list[str]

    # ── control / errors ──
    error: str


def new_state(question: str) -> AnalystState:
    return {
        "question": question,
        "plan": [],
        "actions": [],
        "is_sql_shaped": False,
        "findings": [],
        "reasoning_log": [],
        "loop_count": 0,
        "needs_more": False,
        "approved": False,
        "human_feedback": "",
        "report": "",
        "citations": [],
        "error": "",
    }


def log(msg: str) -> dict:
    """Return a state-update fragment that appends one reasoning line."""
    return {"reasoning_log": [msg]}
