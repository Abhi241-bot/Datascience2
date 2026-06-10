"""Researcher agent — executes the plan's tool actions and gathers cited evidence.

Loops: after running the planned actions it asks the LLM whether the evidence is
sufficient. If not (and under the loop cap), it proposes follow-up actions and the
graph routes back here. Every finding carries provenance so the analyst can cite it.
"""
from __future__ import annotations

from src import config
from src.agents.llm import chat, parse_json
from src.memory.state import AnalystState, Finding
from src.tools.retrieval import retrieve
from src.tools.text_to_sql import text_to_sql
from src.tools.web_search import web_search


# ── tool dispatch -> normalized Findings ───────────────────────────────────────
def _run_action(action: dict) -> tuple[list[Finding], list[str]]:
    tool = action.get("tool")
    query = action.get("input", "")
    findings: list[Finding] = []
    logs: list[str] = [f"[researcher] calling {tool}: {query}"]

    try:
        if tool == "text_to_sql":
            out = text_to_sql(query)
            if out.get("error"):
                logs.append(f"[researcher] text_to_sql: {out['error']}")
            sql = out.get("sql") or ""
            rows = out.get("rows", [])
            cols = out.get("columns", [])
            preview = ", ".join(str(r) for r in rows[:10])
            content = f"SQL: {sql}\nColumns: {cols}\nRows ({out.get('row_count', 0)}): {preview}"
            findings.append(Finding(tool="text_to_sql", source=f"SQL:{sql}",
                                    content=content, raw=out))
            logs.append(f"[researcher] text_to_sql returned {out.get('row_count', 0)} rows "
                        f"(via {out.get('provenance', '?')})")

        elif tool == "retrieval":
            out = retrieve(query)
            for ch in out.get("chunks", []):
                findings.append(Finding(tool="retrieval", source=ch["source"],
                                        content=ch["text"], raw=ch))
            logs.append(f"[researcher] retrieval returned {len(out.get('chunks', []))} chunks")

        elif tool == "web_search":
            out = web_search(query)
            for r in out.get("results", []):
                findings.append(Finding(tool="web_search", source=r.get("url", ""),
                                        content=f"{r.get('title','')}: {r.get('snippet','')}",
                                        raw=r))
            logs.append(f"[researcher] web_search returned {len(out.get('results', []))} results")
        else:
            logs.append(f"[researcher] unknown tool '{tool}' — skipped")
    except Exception as e:
        logs.append(f"[researcher] {tool} failed: {e}")

    return findings, logs


def research_node(state: AnalystState) -> dict:
    actions = state.get("actions", [])
    new_findings: list[Finding] = []
    logs: list[str] = []
    for action in actions:
        f, lg = _run_action(action)
        new_findings += f
        logs += lg

    loop_count = state.get("loop_count", 0) + 1

    # reflect: is the evidence enough to answer, or do we need another pass?
    needs_more, follow_up, reflect_logs = _reflect(state, new_findings, loop_count)

    update: dict = {
        "findings": new_findings,
        "reasoning_log": logs + reflect_logs,
        "loop_count": loop_count,
        "needs_more": needs_more,
    }
    if needs_more and follow_up:
        update["actions"] = follow_up  # researcher runs these on the next loop
    return update


def _reflect(state: AnalystState, new_findings: list[Finding], loop_count: int):
    """Ask the LLM whether to gather more evidence. Returns (needs_more, actions, logs)."""
    if loop_count >= config.MAX_RESEARCH_LOOPS:
        return False, [], [f"[researcher] reached loop cap ({config.MAX_RESEARCH_LOOPS}); proceeding."]

    all_findings = state.get("findings", []) + new_findings
    evidence = "\n".join(f"- ({f['tool']}|{f['source']}) {f['content'][:200]}" for f in all_findings)
    prompt = (
        "You are the Researcher reflecting on whether the gathered evidence is "
        "sufficient to write a complete, cited answer.\n\n"
        f"Question: {state['question']}\n\nEvidence so far:\n{evidence}\n\n"
        "Return ONLY JSON: {\"sufficient\": true|false, \"missing\": \"what's missing\", "
        "\"actions\": [{\"tool\": \"text_to_sql|retrieval|web_search\", \"input\": \"...\", \"reason\": \"...\"}]}\n"
        "If sufficient, actions = []."
    )
    try:
        parsed = parse_json(chat(prompt))
        sufficient = bool(parsed.get("sufficient", True))
        follow = parsed.get("actions", []) or []
        if sufficient or not follow:
            return False, [], ["[researcher] evidence sufficient; moving to review."]
        return True, follow, [
            f"[researcher] need more: {parsed.get('missing', '')}",
            *[f"[researcher] follow-up -> {a['tool']}: {a['input']}" for a in follow],
        ]
    except Exception as e:
        return False, [], [f"[researcher] reflection failed ({e}); proceeding with current evidence."]


def route_after_research(state: AnalystState) -> str:
    """Conditional edge: loop back to research, or continue to the HITL checkpoint."""
    if state.get("needs_more") and state.get("loop_count", 0) < config.MAX_RESEARCH_LOOPS:
        return "research"
    return "human_review"
