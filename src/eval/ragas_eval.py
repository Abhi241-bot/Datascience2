"""RAGAS evaluation — faithfulness, answer relevancy, context precision & recall.

Runs the full agent on each golden question, collects the answer + the retrieved
contexts, and scores them with RAGAS. The LLM judge is **Groq** (same orchestrator
model) and embeddings are the torch-free ONNX MiniLM — so no extra API keys.

    python -m src.eval.ragas_eval            # full golden set
    python -m src.eval.ragas_eval --limit 3  # quick subset

Writes eval_results/ragas_scorecard.json (read by the Gradio evals tab).
"""
from __future__ import annotations

import argparse
import json

from src import config
from src.agents.llm import get_llm
from src.eval.eval_dataset import GOLDEN_SET
from src.graph.workflow import run_analysis


# ── adapters: Groq judge + ONNX embeddings into RAGAS ──────────────────────────
def _ragas_llm():
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(get_llm(temperature=0))


def _ragas_embeddings():
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from src.tools.retrieval import _embedder

    class ChromaONNXEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [list(map(float, v)) for v in _embedder()(list(texts))]

        def embed_query(self, text):
            return list(map(float, _embedder()([text])[0]))

    return LangchainEmbeddingsWrapper(ChromaONNXEmbeddings())


# ── run the agent and collect (answer, contexts) per golden question ───────────
def _contexts_from_findings(findings: list[dict]) -> list[str]:
    """RAGAS 'retrieved_contexts': the text the answer was grounded in."""
    ctx = []
    for f in findings:
        if f.get("tool") in ("retrieval", "text_to_sql", "web_search"):
            ctx.append(f.get("content", ""))
    return [c for c in ctx if c] or ["(no context retrieved)"]


def build_samples(limit: int = 0) -> list[dict]:
    items = GOLDEN_SET[:limit] if limit else GOLDEN_SET
    samples = []
    for i, item in enumerate(items):
        state = run_analysis(item["question"], thread_id=f"ragas-{i}")
        samples.append(
            {
                "user_input": item["question"],
                "response": state.get("report", ""),
                "retrieved_contexts": _contexts_from_findings(state.get("findings", [])),
                "reference": item["ground_truth"],
            }
        )
        print(f"  [{i+1}/{len(items)}] ran: {item['question'][:60]}…")
    return samples


def _metrics():
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    return {
        "faithfulness": Faithfulness(),
        "answer_relevancy": ResponseRelevancy(),
        "context_precision": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
    }


# ── Groq-judge fallback (used when the `ragas` package can't be installed) ─────
# Same four metrics as RAGAS, scored 0-1 by the Groq judge directly. This keeps the
# harness runnable on environments where ragas' native deps don't build (e.g. the
# scikit-network C++ wheel on Python 3.14). The scorecard records which engine ran.
_METRIC_DEFS = {
    "faithfulness": "fraction of the factual claims in the ANSWER that are directly supported by the CONTEXT",
    "answer_relevancy": "how directly and completely the ANSWER addresses the QUESTION",
    "context_precision": "fraction of the CONTEXT passages that are actually relevant to the QUESTION",
    "context_recall": "fraction of the claims in the REFERENCE answer that are covered by the CONTEXT",
}


def _judge_all(sample: dict) -> dict:
    """Score all four metrics in ONE judge call (fewer Groq calls -> fewer 429s)."""
    from src.agents.llm import chat, parse_json

    ctx = "\n".join(f"- {c}" for c in sample["retrieved_contexts"])[:4000]
    rubric = "\n".join(f'- "{m}" (0..1): {d}' for m, d in _METRIC_DEFS.items())
    prompt = (
        "You are a strict RAG evaluation judge. Score each metric from 0.0 to 1.0.\n\n"
        f"{rubric}\n\n"
        f"QUESTION: {sample['user_input']}\n\nANSWER: {sample['response'][:2500]}\n\n"
        f"REFERENCE: {sample['reference']}\n\nCONTEXT:\n{ctx}\n\n"
        'Return ONLY JSON: {"faithfulness": <f>, "answer_relevancy": <f>, '
        '"context_precision": <f>, "context_recall": <f>}'
    )
    try:
        parsed = parse_json(chat(prompt, temperature=0))
        return {m: max(0.0, min(1.0, float(parsed.get(m, 0)))) for m in _METRIC_DEFS}
    except Exception:
        return {m: 0.0 for m in _METRIC_DEFS}


def _run_groq_fallback(samples: list[dict]) -> tuple[dict, list[dict]]:
    print("Scoring with Groq-judge fallback (ragas package unavailable) …")
    per_q, agg = [], {m: [] for m in _METRIC_DEFS}
    for s in samples:
        scores = _judge_all(s)
        row = {"user_input": s["user_input"]}
        for m in _METRIC_DEFS:
            agg[m].append(scores[m])
            row[m] = round(scores[m], 3)
        per_q.append(row)
    scores = {m: round(sum(v) / len(v), 4) for m, v in agg.items() if v}
    return scores, per_q


def run(limit: int = 0) -> dict:
    print("Running agent over the golden set to gather answers + contexts …")
    samples = build_samples(limit)

    try:
        from ragas import EvaluationDataset, evaluate

        metrics = _metrics()
        print("Scoring with RAGAS (Groq judge + ONNX embeddings) …")
        result = evaluate(
            dataset=EvaluationDataset.from_list(samples),
            metrics=list(metrics.values()),
            llm=_ragas_llm(),
            embeddings=_ragas_embeddings(),
        )
        df = result.to_pandas()
        scores = {name: round(float(df[name].mean()), 4) for name in metrics if name in df}
        per_question = df.fillna(0).to_dict(orient="records")
        engine = "ragas"
    except ImportError:
        scores, per_question = _run_groq_fallback(samples)
        engine = "groq-judge-fallback"

    scorecard = {
        "engine": engine,
        "metrics": scores,
        "thresholds": config.EVAL_THRESHOLDS,
        "passed": {k: scores.get(k, 0) >= v for k, v in config.EVAL_THRESHOLDS.items()},
        "n_questions": len(samples),
        "judge_model": config.ORCHESTRATOR_MODEL,
        "per_question": per_question,
    }

    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.EVAL_RESULTS_DIR / "ragas_scorecard.json"
    out.write_text(json.dumps(scorecard, indent=2, default=str), encoding="utf-8")

    print("\n── RAGAS SCORECARD ─────────────────────────────")
    for k, v in scores.items():
        bar = "PASS" if scorecard["passed"].get(k) else "below"
        print(f"  {k:20} {v:.3f}   ({bar} thr {config.EVAL_THRESHOLDS.get(k)})")
    print(f"  -> {out}")
    return scorecard


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    run(ap.parse_args().limit)
