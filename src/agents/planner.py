"""Planner agent — decomposes the question into a tool-grounded plan.

Outputs an ordered list of concrete tool actions the researcher will execute. The
planner decides WHICH tools fit: retrieval (qualitative/narrative from 10-Ks),
text_to_sql (quantitative/aggregate questions over the financials DB), web_search
(recent/external context). This is what makes the system *do* something rather
than answer from a single retrieval pass.
"""
from __future__ import annotations

from src.agents.llm import chat, parse_json
from src.memory.state import AnalystState, log

TOOLS_DESC = """\
Available tools:
- "text_to_sql": answers QUANTITATIVE questions using a SQL database of real company
  financials. Tables: companies(cik, name, ticker, sector) and
  financials(cik, fiscal_year, revenue_musd, gross_profit_musd, operating_income_musd,
  net_income_musd, rnd_musd, assets_musd, liabilities_musd, cash_musd). Use for
  numbers, rankings, growth, comparisons across companies/years.
- "retrieval": searches real 10-K filing text (Business, Risk Factors, MD&A) for
  QUALITATIVE context — strategy, risks, segments, drivers. Use for "why"/"what
  risks"/"describe" questions.
- "web_search": finds RECENT or external information not in the filings/DB.
"""

PLANNER_PROMPT = """\
You are the Planner in a multi-agent financial analyst system. Decompose the user's
question into an ordered plan of tool actions that will gather the evidence needed
to write a cited analytical report.

{tools}

User question: "{question}"

Return ONLY JSON of this exact shape:
{{
  "is_sql_shaped": <true if the question needs numbers from the financials DB>,
  "plan": ["short human-readable step", ...],
  "actions": [
    {{"tool": "text_to_sql|retrieval|web_search", "input": "the precise query/question for that tool", "reason": "why"}}
  ]
}}

Rules:
- Prefer 2-4 actions. Use text_to_sql for any quantitative comparison/ranking/growth.
- Use retrieval to explain or contextualize the numbers (drivers, risks, strategy).
- Only add web_search if the question needs recent/external info beyond the filings.
- Each action's "input" must be self-contained.
"""


def plan_node(state: AnalystState) -> dict:
    question = state["question"]
    prompt = PLANNER_PROMPT.format(tools=TOOLS_DESC, question=question)
    try:
        parsed = parse_json(chat(prompt))
        actions = parsed.get("actions", []) or []
        plan = parsed.get("plan", []) or [a.get("reason", "") for a in actions]
        is_sql = bool(parsed.get("is_sql_shaped", False))
    except Exception as e:
        # robust fallback: do both retrieval + SQL so the run still produces evidence
        plan = ["Retrieve qualitative context", "Query financials DB"]
        actions = [
            {"tool": "retrieval", "input": question, "reason": "qualitative context"},
            {"tool": "text_to_sql", "input": question, "reason": "quantitative data"},
        ]
        is_sql = True
        return {
            "plan": plan, "actions": actions, "is_sql_shaped": is_sql,
            **log(f"[planner] LLM plan parse failed ({e}); using default plan."),
            "reasoning_log": [
                f"[planner] LLM plan parse failed ({e}); using default plan.",
                *[f"[planner] step: {p}" for p in plan],
            ],
        }

    lines = [f"[planner] Plan for: {question}"]
    lines += [f"[planner] step {i+1}: {p}" for i, p in enumerate(plan)]
    lines += [f"[planner] -> {a['tool']}: {a['input']}" for a in actions]
    return {"plan": plan, "actions": actions, "is_sql_shaped": is_sql, "reasoning_log": lines}
