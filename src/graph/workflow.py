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
from src.memory.state import AnalystState, new_state


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
    g.add_node("planner", plan_node)
    g.add_node("research", research_node)
    g.add_node("human_review", human_review_node)
    g.add_node("analyst", analyst_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "research")
    g.add_conditional_edges(
        "research",
        route_after_research,
        {"research": "research", "human_review": "human_review"},
    )
    g.add_edge("human_review", "analyst")
    g.add_edge("analyst", END)

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

    # Phase 1: run until the interrupt before "analyst".
    graph.invoke(new_state(question), cfg)

    snapshot = graph.get_state(cfg)
    if not auto_approve:
        return snapshot.values  # paused at the checkpoint

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
