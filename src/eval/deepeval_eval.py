"""DeepEval (G-Eval) — custom LLM-as-judge criteria the analyst should satisfy.

Two criteria that RAGAS does not cover directly:
  - Citation Correctness: every quantitative/qualitative claim is backed by a
    citation tag that actually appears in the retrieved context.
  - Analytical Depth: the report goes beyond restating numbers — it interprets,
    compares, and notes risks/outlook (it's an analyst artifact, not a lookup).

Judge model = Groq (no OpenAI key needed), via a DeepEvalBaseLLM wrapper.

    python -m src.eval.deepeval_eval --limit 3

Writes eval_results/deepeval_scorecard.json.
"""
from __future__ import annotations

import argparse
import json

from src import config
from src.agents.llm import get_llm
from src.eval.eval_dataset import GOLDEN_SET
from src.graph.workflow import run_analysis


# ── Groq as a DeepEval judge model ─────────────────────────────────────────────
def _groq_judge():
    from deepeval.models.base_model import DeepEvalBaseLLM

    class GroqJudge(DeepEvalBaseLLM):
        def load_model(self):
            return get_llm(temperature=0)

        def generate(self, prompt: str, schema=None):
            content = self.load_model().invoke(prompt).content
            if schema is not None:
                # G-Eval may request a structured object; coerce JSON -> schema.
                from src.agents.llm import parse_json
                try:
                    return schema(**parse_json(content))
                except Exception:
                    return content
            return content

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self):
            return f"groq:{config.ORCHESTRATOR_MODEL}"

    return GroqJudge()


def _metrics(model):
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    citation = GEval(
        name="Citation Correctness",
        criteria=(
            "Determine whether every factual or numeric claim in the actual output is "
            "supported by a citation tag (e.g. [SQL] or a [filename]) that corresponds "
            "to the provided retrieval context. Penalize claims with no citation or "
            "citations that do not match the context."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=model,
        threshold=0.7,
    )
    depth = GEval(
        name="Analytical Depth",
        criteria=(
            "Assess whether the output reads like an analyst's report: it should not "
            "merely restate figures but interpret them — compare entities, explain "
            "drivers, and note risks or outlook. Reward synthesis; penalize a flat "
            "data dump or a one-line answer."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=model,
        threshold=0.7,
    )
    return [citation, depth]


# ── Groq-judge fallback (used when the `deepeval` package isn't installed) ─────
# Same two G-Eval criteria, scored 0-1 by the Groq judge directly. Keeps the
# harness runnable everywhere; the scorecard records which engine ran.
_GEVAL_CRITERIA = {
    "Citation Correctness": (
        "Judge whether every factual/numeric claim in the ANSWER is backed by a "
        "citation tag ([SQL] or a [filename]) that matches the CONTEXT. 1 = all "
        "claims correctly cited, 0 = uncited or mismatched."),
    "Analytical Depth": (
        "Judge whether the ANSWER reads like an analyst report: it interprets and "
        "compares rather than merely restating numbers, and notes risks/outlook. "
        "1 = strong synthesis, 0 = flat data dump or one-liner."),
}


def _geval_fallback(items: list[dict]) -> tuple[dict, list[dict]]:
    from src.agents.llm import chat, parse_json

    print("Scoring with Groq-judge fallback (deepeval package unavailable) …")
    agg = {name: [] for name in _GEVAL_CRITERIA}
    per_question = []
    rubric = "\n".join(f'- "{n}" (0..1): {c}' for n, c in _GEVAL_CRITERIA.items())
    for i, item in enumerate(items):
        state = run_analysis(item["question"], thread_id=f"geval-{i}")
        report = state.get("report", "")
        ctx = "\n".join(f.get("content", "") for f in state.get("findings", []))[:4000]
        row = {"question": item["question"]}
        # both criteria in ONE judge call -> fewer Groq calls / 429s
        prompt = (f"You are a strict G-Eval judge. Score each criterion 0.0-1.0:\n{rubric}\n\n"
                  f"QUESTION: {item['question']}\n\nANSWER: {report[:2500]}\n\nCONTEXT:\n{ctx}\n\n"
                  'Return ONLY JSON: {"Citation Correctness": {"score": <f>, "reason": "<s>"}, '
                  '"Analytical Depth": {"score": <f>, "reason": "<s>"}}')
        try:
            p = parse_json(chat(prompt, temperature=0))
        except Exception:
            p = {}
        for name in _GEVAL_CRITERIA:
            entry = p.get(name, {}) if isinstance(p.get(name), dict) else {}
            try:
                sc = max(0.0, min(1.0, float(entry.get("score", 0))))
            except Exception:
                sc = 0.0
            agg[name].append(sc)
            row[name] = round(sc, 3)
            row[f"{name}__reason"] = entry.get("reason", "")
        per_question.append(row)
        print(f"  [{i+1}/{len(items)}] {item['question'][:50]}… "
              + " ".join(f"{n}={row[n]}" for n in _GEVAL_CRITERIA))
    scores = {n: round(sum(v) / len(v), 4) for n, v in agg.items() if v}
    return scores, per_question


def _run_deepeval(items: list[dict]) -> tuple[dict, list[dict]]:
    from deepeval.test_case import LLMTestCase

    judge = _groq_judge()
    metrics = _metrics(judge)
    results = {m.name: [] for m in metrics}
    per_question = []
    for i, item in enumerate(items):
        state = run_analysis(item["question"], thread_id=f"deepeval-{i}")
        contexts = [f.get("content", "") for f in state.get("findings", []) if f.get("content")]
        tc = LLMTestCase(
            input=item["question"],
            actual_output=state.get("report", ""),
            retrieval_context=contexts or ["(no context)"],
        )
        row = {"question": item["question"]}
        for m in metrics:
            m.measure(tc)
            results[m.name].append(m.score or 0.0)
            row[m.name] = round(m.score or 0.0, 3)
            row[f"{m.name}__reason"] = getattr(m, "reason", "")
        per_question.append(row)
        print(f"  [{i+1}/{len(items)}] {item['question'][:55]}… "
              + " ".join(f"{m.name}={row[m.name]}" for m in metrics))
    scores = {name: round(sum(v) / len(v), 4) for name, v in results.items() if v}
    return scores, per_question


def run(limit: int = 0) -> dict:
    items = GOLDEN_SET[:limit] if limit else GOLDEN_SET
    try:
        import deepeval  # noqa: F401

        scores, per_question = _run_deepeval(items)
        engine = "deepeval"
    except ImportError:
        scores, per_question = _geval_fallback(items)
        engine = "groq-judge-fallback"

    scorecard = {
        "engine": engine,
        "metrics": scores,
        "n_questions": len(items),
        "judge_model": config.ORCHESTRATOR_MODEL,
        "per_question": per_question,
    }
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.EVAL_RESULTS_DIR / "deepeval_scorecard.json"
    out.write_text(json.dumps(scorecard, indent=2, default=str), encoding="utf-8")

    print("\n── DEEPEVAL (G-EVAL) SCORECARD ─────────────────")
    for k, v in scores.items():
        print(f"  {k:22} {v:.3f}")
    print(f"  -> {out}")
    return scorecard


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    run(ap.parse_args().limit)
