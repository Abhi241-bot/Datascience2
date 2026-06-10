"""LangGraph workflow — the stateful multi-agent graph.

Flow:
    START -> planner -> research -(loop if needed)- > human_review -> analyst -> END

- planner: decomposes the question into tool actions.
- research: runs the tools, accumulates cited findings, reflects; conditional edge
  loops back for another pass (up to MAX_RESEARCH_LOOPS) when evidence is thin.
- human_review + interrupt_before=["analyst"]: the **human-in-the-loop checkpoint**.
  Execution pauses after the evidence is gathered and before the final report is
  written, so a human can approve or steer it.
- analyst: writes the cited report.

A MemorySaver checkpointer persists state per thread_id, which is what makes the
pause/resume HITL possible.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.analyst import analyst_node
from src.agents.planner import plan_node
from src.agents.researcher import research_node, route_after_research
from src.guardrails.validators import check_input, check_output
from src.memory.state import AnalystState, new_state


def guard_input_node(state: AnalystState) -> dict:
    """Input guardrail — runs before the planner. Rejects injection, redacts PII."""
    res = check_input(state["question"])
    if not res.allowed:
        # reject: produce a safe refusal and skip straight to the end
        return {
            "error": "; ".join(res.violations),
            "report": ("**Request blocked by input guardrail.**\n\n"
                       f"Reason: {res.violations[0]}\n\n"
                       "Please rephrase as a financial-analysis question."),
            "reasoning_log": [f"[guard:input] REJECTED ({res.category}): {res.violations}"],
        }
    update: dict = {"reasoning_log": [f"[guard:input] passed ({res.category})."]}
    if res.category == "pii":
        update["question"] = res.sanitized  # proceed on redacted text
        update["reasoning_log"] = [f"[guard:input] {res.violations[0]}"]
    return update


def route_after_input(state: AnalystState) -> str:
    return "planner" if not state.get("error") else "blocked"


def guard_output_node(state: AnalystState) -> dict:
    """Output guardrail — citations present, no hallucinated sources, read-only SQL."""
    res = check_output(state.get("report", ""), state.get("findings", []))
    update: dict = {
        "report": res.sanitized or state.get("report", ""),
        "reasoning_log": [f"[guard:output] {'passed' if res.allowed else 'violations'}: "
                          f"{res.violations or 'clean'}"],
    }
    if not res.allowed:
        update["error"] = "; ".join(res.violations)
    return update


def human_review_node(state: AnalystState) -> dict:
    """Summarize gathered evidence for the human checkpoint (before the report)."""
    findings = state.get("findings", [])
    by_tool: dict[str, int] = {}
    for f in findings:
        by_tool[f["tool"]] = by_tool.get(f["tool"], 0) + 1
    summary = ", ".join(f"{n} from {t}" for t, n in by_tool.items()) or "no evidence"
    return {
        "reasoning_log": [
            f"[checkpoint] Gathered {len(findings)} findings ({summary}).",
            "[checkpoint] Awaiting human approval before writing the final report.",
        ]
    }


def build_graph(checkpointer: MemorySaver | None = None):
    g = StateGraph(AnalystState)
    g.add_node("guard_input", guard_input_node)
    g.add_node("planner", plan_node)
    g.add_node("research", research_node)
    g.add_node("human_review", human_review_node)
    g.add_node("analyst", analyst_node)
    g.add_node("guard_output", guard_output_node)

    g.add_edge(START, "guard_input")
    # input guardrail: reject -> END, else -> planner
    g.add_conditional_edges("guard_input", route_after_input,
                            {"planner": "planner", "blocked": END})
    g.add_edge("planner", "research")
    g.add_conditional_edges(
        "research",
        route_after_research,
        {"research": "research", "human_review": "human_review"},
    )
    g.add_edge("human_review", "analyst")
    g.add_edge("analyst", "guard_output")   # output guardrail after the report
    g.add_edge("guard_output", END)

    # Pause before the analyst writes the final report -> the HITL checkpoint.
    return g.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=["analyst"],
    )


# ── convenience runner (used by tests / evals; the Gradio app streams instead) ──
def run_analysis(question: str, thread_id: str = "default",
                 auto_approve: bool = True, feedback: str = "") -> AnalystState:
    """Run the full graph, handling the HITL pause.

    auto_approve=True resumes past the checkpoint automatically (non-interactive).
    Set auto_approve=False to stop AT the checkpoint and inspect state yourself.
    """
    graph = build_graph()
    cfg = {"configurable": {"thread_id": thread_id}}

    # Phase 1: run until the interrupt before "analyst" (or END if input blocked).
    graph.invoke(new_state(question), cfg)

    snapshot = graph.get_state(cfg)
    if not snapshot.next:
        return snapshot.values  # finished already (e.g. blocked by input guardrail)
    if not auto_approve:
        return snapshot.values  # paused at the HITL checkpoint

    # Human approves (optionally with feedback), then resume to produce the report.
    graph.update_state(cfg, {"approved": True, "human_feedback": feedback})
    graph.invoke(None, cfg)  # resume from the checkpoint
    return graph.get_state(cfg).values


if __name__ == "__main__":
    import json

    q = "Which company had the highest revenue in the latest fiscal year, and what are its main risks?"
    final = run_analysis(q)
    print("\n──── REASONING LOG ────")
    print("\n".join(final.get("reasoning_log", [])))
    print("\n──── REPORT ────")
    print(final.get("report", ""))
    print("\n──── CITATIONS ────", final.get("citations", []))
