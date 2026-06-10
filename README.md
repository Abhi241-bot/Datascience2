# Multi-Agent Analyst System

> 🚧 Under construction — built phase by phase per [`MultiAgent_Analyst_System_SPEC.md`](MultiAgent_Analyst_System_SPEC.md). The recruiter-grade README lands in the final phase.

An autonomous **multi-agent analyst**: a LangGraph workflow that plans a task, calls tools (retrieval over a financials corpus, web search, and a **Text-to-SQL tool powered by a QLoRA model fine-tuned on Spider**), and produces a **cited analytical report** — wrapped in stateful memory, a human-in-the-loop checkpoint, guardrails, and a RAGAS/DeepEval evaluation harness. Deploys to Hugging Face Spaces with a Gradio UI that streams reasoning live.

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | QLoRA fine-tune of the SQL tool (Spider) | ✅ scripts + notebook |
| 2 | Tools (retrieval, web search, text-to-SQL) | ✅ |
| 3 | LangGraph workflow + memory | ⏳ |
| 4 | Guardrails | ⏳ |
| 5 | Evals (RAGAS, DeepEval, LangSmith) | ⏳ |
| 6 | Gradio app + HF Spaces deploy | ⏳ |
| 7 | Tests & README | ⏳ |

## Stack

LangGraph · LangSmith · RAGAS · DeepEval · Chroma · Unsloth/PEFT/BitsAndBytes (QLoRA) · Groq (orchestrator) · Gradio · Python 3.11

## Quickstart (dev)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY
```

See [`finetune/README.md`](finetune/README.md) for the fine-tuning workflow.
