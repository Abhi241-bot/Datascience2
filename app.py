"""Gradio app — the recruiter-facing demo. Deploys to Hugging Face Spaces.

Two tabs:
  1. Analyst — ask a question, watch the agent's reasoning + tool calls stream live,
     approve at the human-in-the-loop checkpoint, then read the cited report.
  2. Evals — the latest RAGAS + DeepEval scorecards.

Run locally:  python app.py    (then open http://127.0.0.1:7860)
"""
from __future__ import annotations

import json
import uuid

import gradio as gr

from src import config
from src.bootstrap import ensure_data
from src.eval.tracing import status as langsmith_status
from src.graph.workflow import build_graph
from src.memory.state import new_state

# Rebuild the DB from the committed SQL dump + warm the corpus index if needed.
ensure_data()

# One compiled graph with a shared in-memory checkpointer; thread_id isolates runs.
GRAPH = build_graph()


# ── streaming helpers ──────────────────────────────────────────────────────────
def _render_log(state: dict) -> str:
    return "\n".join(state.get("reasoning_log", []))


def start_run(question: str, hitl: bool):
    """Stream planner+research; pause at the HITL checkpoint or auto-continue."""
    question = (question or "").strip()
    if not question:
        yield "Enter a question first.", "", gr.update(visible=False), ""
        return

    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    last = {}
    for state in GRAPH.stream(new_state(question), cfg, stream_mode="values"):
        last = state
        yield _render_log(state), "", gr.update(visible=False), thread_id

    snap = GRAPH.get_state(cfg)
    paused_before_analyst = bool(snap.next) and "analyst" in snap.next

    if paused_before_analyst and hitl:
        log = _render_log(last) + "\n\n⏸  HITL CHECKPOINT — review the gathered evidence above, then click **Approve & write report**."
        yield log, "", gr.update(visible=True), thread_id
        return

    # auto-approve (or input was blocked and the graph already ended)
    if paused_before_analyst:
        GRAPH.update_state(cfg, {"approved": True})
        for state in GRAPH.stream(None, cfg, stream_mode="values"):
            last = state
            yield _render_log(state), "", gr.update(visible=False), thread_id

    final = GRAPH.get_state(cfg).values
    yield _render_log(final), final.get("report", ""), gr.update(visible=False), thread_id


def approve_run(thread_id: str, feedback: str):
    """Resume past the HITL checkpoint and stream the analyst's report."""
    if not thread_id:
        yield "No active run to approve.", "", gr.update(visible=False)
        return
    cfg = {"configurable": {"thread_id": thread_id}}
    GRAPH.update_state(cfg, {"approved": True, "human_feedback": (feedback or "").strip()})

    for state in GRAPH.stream(None, cfg, stream_mode="values"):
        yield _render_log(state), state.get("report", ""), gr.update(visible=False)

    final = GRAPH.get_state(cfg).values
    yield _render_log(final), final.get("report", ""), gr.update(visible=False)


# ── evals tab ──────────────────────────────────────────────────────────────────
def _load_scorecard(name: str) -> dict | None:
    path = config.EVAL_RESULTS_DIR / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def evals_markdown() -> str:
    rag = _load_scorecard("ragas_scorecard.json")
    de = _load_scorecard("deepeval_scorecard.json")
    md = ["## Evaluation Scorecards\n"]

    if rag:
        md.append(f"### RAGAS  ·  engine: `{rag.get('engine','?')}`  ·  judge: `{rag.get('judge_model','?')}`  ·  {rag.get('n_questions','?')} questions\n")
        md.append("| Metric | Score | Threshold | Pass |")
        md.append("|---|---|---|---|")
        thr = rag.get("thresholds", {})
        passed = rag.get("passed", {})
        for k, v in rag.get("metrics", {}).items():
            md.append(f"| {k} | {v:.3f} | {thr.get(k,'—')} | {'✅' if passed.get(k) else '⚠️'} |")
        md.append("")
    else:
        md.append("_RAGAS scorecard not found — run `python -m src.eval.ragas_eval`._\n")

    if de:
        md.append(f"### DeepEval (G-Eval)  ·  engine: `{de.get('engine','?')}`  ·  judge: `{de.get('judge_model','?')}`  ·  {de.get('n_questions','?')} questions\n")
        md.append("| Criterion | Score |")
        md.append("|---|---|")
        for k, v in de.get("metrics", {}).items():
            md.append(f"| {k} | {v:.3f} |")
        md.append("")
    else:
        md.append("_DeepEval scorecard not found — run `python -m src.eval.deepeval_eval`._\n")

    return "\n".join(md)


# ── UI ───────────────────────────────────────────────────────────────────────
EXAMPLES = [
    "Which company had the highest revenue in its most recent fiscal year, and what are its main risks?",
    "Did Apple or Microsoft spend more on R&D in their latest fiscal year, and what does each say about its strategy?",
    "Rank the technology companies by latest revenue and explain the competitive pressures each faces.",
]

with gr.Blocks(title="Multi-Agent Analyst", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🔎 Multi-Agent Financial Analyst\n"
        "Autonomous **LangGraph** agent over **real SEC EDGAR** data: it plans, calls tools "
        "(retrieval over 10-Ks · web search · a **QLoRA-fine-tuned Text-to-SQL** model), "
        "passes guardrails, pauses at a human checkpoint, and writes a **cited report**.\n"
    )
    thread_state = gr.State("")

    with gr.Tab("Analyst"):
        with gr.Row():
            question = gr.Textbox(label="Ask the analyst", scale=4,
                                  placeholder="e.g. Which company had the highest revenue last year, and what are its risks?")
            hitl = gr.Checkbox(label="Human-in-the-loop", value=True, scale=1,
                               info="Pause for approval before the report")
        run_btn = gr.Button("Run analysis", variant="primary")
        gr.Examples(EXAMPLES, inputs=question)

        with gr.Row():
            reasoning = gr.Textbox(label="🧠 Reasoning & tool calls (live)", lines=18)
            report = gr.Markdown(value="*The cited report will appear here.*")

        with gr.Group(visible=False) as approve_box:
            feedback = gr.Textbox(label="Optional guidance for the analyst", placeholder="e.g. emphasize margins")
            approve_btn = gr.Button("✅ Approve & write report", variant="primary")

        run_btn.click(start_run, [question, hitl],
                      [reasoning, report, approve_box, thread_state])
        approve_btn.click(approve_run, [thread_state, feedback],
                          [reasoning, report, approve_box])

    with gr.Tab("Evals"):
        gr.Markdown(evals_markdown())
        gr.Markdown(f"_{langsmith_status()}_")

    gr.Markdown(
        "—\nTip: data/SQL questions route to the fine-tuned Text-to-SQL tool; "
        "qualitative questions route to 10-K retrieval. Malicious inputs and write-SQL "
        "are blocked by guardrails."
    )

if __name__ == "__main__":
    demo.launch()
