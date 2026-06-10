"""Analyst agent — synthesizes the final cited analytical report.

Consumes the accumulated findings and writes a structured report where every
quantitative claim is backed by the SQL result and every qualitative claim cites a
10-K source. Citations use the provenance tags attached to each finding.
"""
from __future__ import annotations

from src.agents.llm import chat
from src.memory.state import AnalystState


ANALYST_PROMPT = """\
You are the Analyst. Write a concise, well-structured analytical report answering
the question, using ONLY the evidence below. This is an analytical artifact, not a
chat reply: include a short thesis, the key findings with concrete numbers, and a
brief risk/forward-looking note where relevant.

Question: {question}

Evidence (cite by the [source] tag in brackets):
{evidence}

Requirements:
- Every numeric claim MUST cite a SQL finding, e.g. "Apple led on revenue [SQL]".
- Every qualitative claim MUST cite a filing source, e.g. "[aapl_10k.md]".
- Do NOT invent sources or numbers not present in the evidence.
- End with a "Sources" section listing the distinct [source] tags you used.

Format in Markdown with headings: ## Thesis, ## Findings, ## Risks & Outlook, ## Sources.
"""

HUMAN_FEEDBACK_NOTE = "\nThe human reviewer added this guidance — incorporate it: {fb}\n"


def _format_evidence(state: AnalystState) -> tuple[str, list[str]]:
    lines = []
    citations: list[str] = []
    for f in state.get("findings", []):
        tag = _cite_tag(f)
        if tag not in citations:
            citations.append(tag)
        lines.append(f"[{tag}] ({f['tool']}) {f['content'][:600]}")
    return "\n\n".join(lines), citations


def _cite_tag(finding) -> str:
    if finding["tool"] == "text_to_sql":
        return "SQL"
    if finding["tool"] == "web_search":
        return finding["source"] or "web"
    return finding["source"] or "corpus"


def analyst_node(state: AnalystState) -> dict:
    evidence, citations = _format_evidence(state)
    if not evidence:
        return {
            "report": "No evidence was gathered; unable to produce a report.",
            "citations": [],
            "reasoning_log": ["[analyst] no findings available."],
        }

    prompt = ANALYST_PROMPT.format(question=state["question"], evidence=evidence)
    if state.get("human_feedback"):
        prompt += HUMAN_FEEDBACK_NOTE.format(fb=state["human_feedback"])

    try:
        report = chat(prompt, temperature=0.2)
    except Exception as e:
        report = f"Report generation failed: {e}"

    return {
        "report": report,
        "citations": citations,
        "reasoning_log": [f"[analyst] synthesized report citing {len(citations)} sources."],
    }
