# Multi-Agent Analyst System with Evals, Guardrails & a Fine-Tuned SQL Tool — Build Spec

> **For the AI agent (Antigravity):** This is a build specification, not a tutorial. Build phase by phase. After each phase, stop and confirm the acceptance criteria before moving on. This project has TWO trainable parts (the QLoRA fine-tune) and one orchestration part (the agent) — keep them in separate phases. Do NOT over-engineer: no Kubernetes, no custom agent framework (use LangGraph), no auth. The demo must run on Hugging Face Spaces free tier.

---

## 0. Project Goal (read first)

Build a **multi-agent system that DOES something autonomously** — not another RAG chatbot. An "Analyst" agent ingests a domain corpus + a SQL database, plans a task, calls tools (retrieval, web search, and a **Text-to-SQL tool powered by a model YOU fine-tuned with QLoRA**), and produces a **cited analytical report** — all wrapped in **stateful memory, human-in-the-loop checkpoints, guardrails, and a rigorous evaluation harness**.

**Why this is the point:** Hiring managers in 2026 want systems that *do* things, not chatbots that answer questions. LangGraph + multi-agent + production-grade evals is the top AI-engineer skill cluster, and most teams still lack evals — building them signals maturity. Folding in your own QLoRA fine-tune proves *depth* (you can fine-tune) AND *systems thinking* (you can deploy it inside a real workflow) in one repo. This is strictly stronger than shipping a standalone fine-tune.

### The recruiter hook
A deployed Space where you watch the agent's reasoning stream live, see tool calls (including your fine-tuned SQL model) execute, and view an **evals dashboard** with faithfulness/relevancy/context-precision scores per run.

---

## 1. Tech Stack (use exactly these)

| Layer | Tool | Why |
|---|---|---|
| Agent orchestration | **LangGraph** | 2026 standard for stateful multi-agent graphs |
| Observability / tracing | **LangSmith** (free tier) | traces every step; evals adoption signal |
| RAG eval | **RAGAS** | faithfulness, context precision/recall, answer relevancy |
| Additional eval | **DeepEval** (G-Eval) | custom LLM-as-judge criteria |
| Vector store | **Chroma** (local, free) | retrieval |
| Fine-tuning | **Unsloth + PEFT + BitsAndBytes** (QLoRA) | fine-tune a small SQL model on free Colab T4 |
| Base model for fine-tune | Llama 3.1 8B or Mistral 7B (4-bit) | fits on T4 with QLoRA |
| SQL fine-tune dataset | **Spider** (text-to-SQL) | standard, high quality |
| Guardrails | a guardrails lib (e.g., Guardrails AI / NeMo Guardrails) OR custom validators | input/output safety |
| LLM (orchestrator) | an API model (OpenAI/Gemini/Anthropic) or local | reasoning/planning |
| UI | **Gradio** | streams reasoning, deploys to HF Spaces |
| Language | Python 3.11 | — |

---

## 2. Repository Structure

```
multi-agent-analyst/
├── README.md                      # recruiter-grade, written LAST
├── requirements.txt
├── app.py                         # Gradio app, deploys to HF Spaces
├── finetune/
│   ├── prepare_spider.py          # download Spider, convert to instruction format
│   ├── train_qlora.ipynb          # Colab: QLoRA fine-tune (r=16, alpha=32, all-linear)
│   ├── evaluate_sql.py            # exec-accuracy + before/after on held-out Spider
│   └── adapter/                   # exported LoRA adapter (or HF Hub link)
├── src/
│   ├── config.py                  # models, thresholds, paths in ONE place
│   ├── agents/
│   │   ├── planner.py             # decomposes the task into steps
│   │   ├── researcher.py          # retrieval + web search
│   │   └── analyst.py             # synthesizes the cited report
│   ├── tools/
│   │   ├── retrieval.py           # Chroma vector search over the corpus
│   │   ├── web_search.py          # web search tool
│   │   └── text_to_sql.py         # calls YOUR fine-tuned SQL model + executes query
│   ├── graph/
│   │   └── workflow.py            # LangGraph: nodes, edges, state, HITL checkpoint
│   ├── memory/
│   │   └── state.py               # stateful memory across steps
│   ├── guardrails/
│   │   └── validators.py          # input/output guardrails (PII, injection, SQL safety)
│   └── eval/
│       ├── ragas_eval.py          # faithfulness, context precision/recall, relevancy
│       ├── deepeval_eval.py       # G-Eval custom criteria
│       └── eval_dataset.py        # curated golden Q/A set for the corpus
├── tests/
│   ├── test_tools.py
│   ├── test_graph.py
│   └── test_guardrails.py
└── notebooks/
    └── 01_corpus_ingest.ipynb
```

---

## 3. Build Phases (do in order; confirm acceptance criteria each time)

### Phase 1 — Fine-tune the SQL tool (your Idea A, absorbed)
- `prepare_spider.py`: download Spider, convert to instruction-tuning format (schema + question → SQL).
- `train_qlora.ipynb`: QLoRA fine-tune on free Colab T4. Defaults: 4-bit base, LoRA r=16, alpha=32, target all linear layers, LR 2e-4, gradient checkpointing.
- `evaluate_sql.py`: report **execution accuracy** (does the generated SQL run and return correct rows) on held-out Spider, **before vs after** fine-tuning. NOT just BLEU — exec accuracy is what matters for SQL.
- Export the LoRA adapter (push to HF Hub or save to `adapter/`).
- **✅ Acceptance:** measurable exec-accuracy improvement after fine-tuning, documented with the before/after numbers.

### Phase 2 — Tools (each works standalone before the agent uses them)
- `retrieval.py`: ingest a corpus into Chroma, semantic search returning chunks + sources.
- `web_search.py`: a working web search tool returning results + URLs.
- `text_to_sql.py`: loads your fine-tuned adapter, turns a natural-language question into SQL, executes it against a sample DB, returns rows. Includes a SQL-safety guard (read-only, no DROP/DELETE).
- **✅ Acceptance:** each tool callable in isolation and returns structured output with sources/provenance.

### Phase 3 — LangGraph workflow + memory
- `state.py`: a typed state object carrying the task, intermediate findings, tool outputs, and citations.
- `workflow.py`: a LangGraph with nodes for planner → researcher (loops over tools) → analyst, conditional edges (e.g., "need more info? loop back"), and a **human-in-the-loop checkpoint** before the final report.
- **✅ Acceptance:** given a question, the graph plans, calls the right tools (including text_to_sql when the question is data/SQL-shaped), and produces a cited report; the HITL checkpoint pauses for approval.

### Phase 4 — Guardrails
- `validators.py`: input guardrails (block prompt injection, PII leakage) and output guardrails (citations present, no hallucinated sources, SQL is read-only). Reject/repair on violation.
- **✅ Acceptance:** a malicious input (injection attempt) and an unsafe SQL request are both caught and handled gracefully.

### Phase 5 — Evals (the maturity signal most portfolios lack)
- `eval_dataset.py`: a curated golden set of questions with reference answers/contexts for your corpus.
- `ragas_eval.py`: compute faithfulness, context precision, context recall, answer relevancy over the golden set.
- `deepeval_eval.py`: 1–2 custom G-Eval criteria (e.g., "citation correctness", "analytical depth").
- Wire **LangSmith** tracing so every agent run is inspectable.
- **✅ Acceptance:** running the eval suite produces a scorecard; LangSmith shows full traces of agent runs; you can point to a faithfulness/precision number.

### Phase 6 — Gradio app + deploy
- `app.py`: Gradio UI that (1) takes a question, (2) **streams the agent's reasoning and tool calls live**, (3) shows the final cited report, (4) has a tab showing the latest eval scorecard.
- Deploy to **Hugging Face Spaces** (Gradio SDK; ZeroGPU if the fine-tuned model runs in-Space, or call it via an endpoint/Hub).
- **✅ Acceptance:** deployed Space where a recruiter watches reasoning stream, sees tool calls fire, gets a cited report, and views eval scores — all clickable.

### Phase 7 — Tests & README
- `tests/`: tools, graph happy-path, guardrails.
- `README.md` (write last): see Phase 8.
- **✅ Acceptance:** tests green; README complete.

---

## 4. README requirements (Phase 8 — what recruiters read)
In this order:
1. One-paragraph problem + one-line value ("Autonomous multi-agent analyst with a self-fine-tuned SQL tool, guardrails, and RAGAS evals").
2. **Live demo link** + a GIF of reasoning streaming + tool calls + eval scores.
3. **Architecture diagram** (Mermaid) of the LangGraph workflow showing nodes, tools, HITL checkpoint, and guardrails.
4. **Two proof points:** (a) the QLoRA SQL before/after exec-accuracy table, (b) the RAGAS eval scorecard.
5. One-command run + quickstart; link to the Colab fine-tune notebook.
6. Tech stack + what each agent/tool does.

---

## 5. Hard Constraints / Guardrails
- **NOT a generic RAG chatbot.** It must plan, use multiple tools, and produce an analytical artifact autonomously. If it ends up as a Q&A bot, it has failed the brief.
- **The fine-tuned SQL model must actually be used as a tool** — not a separate demo. That integration is the whole point of absorbing Idea A.
- **Evals are non-negotiable.** The RAGAS scorecard + LangSmith traces are what differentiate this from the thousands of agent demos with no evaluation.
- **No Kubernetes, no custom framework.** LangGraph + Gradio + free tiers only.
- **One config file** for models/thresholds/paths.
- If time runs short, cut DeepEval and the web_search tool LAST — but the fine-tuned SQL tool, the LangGraph workflow, guardrails, and RAGAS evals MUST ship.

## 6. Definition of Done
- [ ] QLoRA SQL fine-tune with documented before/after exec accuracy
- [ ] Multi-agent LangGraph workflow that plans + uses tools autonomously
- [ ] Fine-tuned SQL model integrated as a live tool
- [ ] Guardrails catch injection + unsafe SQL
- [ ] RAGAS eval scorecard + LangSmith traces
- [ ] HITL checkpoint before final report
- [ ] Gradio app streaming reasoning + tool calls + evals, deployed to HF Spaces (clickable)
- [ ] README with demo link, architecture diagram, fine-tune table, and eval scorecard
