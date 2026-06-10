# Phase 5 — Evaluation Harness

The maturity signal most agent demos lack: a rigorous, reproducible eval suite.

## What's here

| File | Evaluates | Judge / model |
|---|---|---|
| `eval_dataset.py` | Curated golden Q/A set (8 questions) grounded in the real SEC DB + 10-Ks | — |
| `ragas_eval.py` | **faithfulness, answer relevancy, context precision, context recall** | Groq judge + ONNX embeddings |
| `deepeval_eval.py` | **Citation Correctness**, **Analytical Depth** (G-Eval) | Groq judge |
| `tracing.py` | LangSmith tracing wiring (every agent run inspectable) | LangSmith |

No OpenAI key required — both RAGAS and DeepEval are pointed at the same free Groq
model used by the agent, and embeddings are the torch-free ONNX MiniLM.

## Run

```bash
# faithfulness / relevancy / context precision+recall  -> eval_results/ragas_scorecard.json
python -m src.eval.ragas_eval                 # full set
python -m src.eval.ragas_eval --limit 3       # quick subset

# citation correctness / analytical depth             -> eval_results/deepeval_scorecard.json
python -m src.eval.deepeval_eval --limit 3

# enable full traces (set the LANGCHAIN_* vars in .env first)
python -m src.eval.tracing
```

The Gradio app's **Evals** tab reads `eval_results/*_scorecard.json`.

## Thresholds (pass/fail bar)

Defined once in `src/config.py::EVAL_THRESHOLDS`:

| Metric | Threshold |
|---|---|
| faithfulness | 0.80 |
| answer relevancy | 0.75 |
| context precision | 0.70 |
| context recall | 0.70 |

## ✅ Acceptance

> Running the eval suite produces a scorecard; LangSmith shows full traces; you can
> point to a faithfulness/precision number.

Each runner writes a JSON scorecard with per-question detail and pass/fail vs the
thresholds. Paste the headline numbers into the root README (Phase 8 proof point).
