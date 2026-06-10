"""Phase 1 — Execution-accuracy evaluation, before vs after fine-tuning.

The metric that matters for text-to-SQL is **execution accuracy**: run the
generated SQL and the gold SQL against the actual SQLite database and compare the
returned result sets. BLEU/string-match is misleading (many correct SQLs differ
textually). This script:

  1. loads a base model and (optionally) the QLoRA adapter,
  2. generates SQL for each held-out Spider dev example,
  3. executes both gold and predicted SQL against the example's sqlite db,
  4. reports execution accuracy for base vs fine-tuned, plus a delta.

Run on Colab (GPU) right after training, or anywhere with the model available:

    python finetune/evaluate_sql.py \
        --dev finetune/data/dev.jsonl \
        --spider-db-dir spider/database \
        --base unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
        --adapter finetune/adapter \
        --limit 200

Spider's per-database sqlite files live under `spider/database/<db_id>/<db_id>.sqlite`
(from the official Spider zip). Point --spider-db-dir at that `database/` folder.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from prepare_spider import SYSTEM, build_prompt  # reuse the exact training prompt


# ── SQL execution + comparison ─────────────────────────────────────────────────
def run_sql(db_file: Path, sql: str):
    """Execute SQL read-only; return (ok, rows). ok=False on any error."""
    try:
        con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode("utf-8", "ignore")
        cur = con.execute(sql)
        rows = cur.fetchall()
        con.close()
        return True, rows
    except Exception:
        return False, None


def result_match(gold_rows, pred_rows) -> bool:
    """Order-insensitive set comparison of result rows (standard exec-accuracy)."""
    if gold_rows is None or pred_rows is None:
        return False
    return sorted(map(repr, gold_rows)) == sorted(map(repr, pred_rows))


# ── Model loading + generation ─────────────────────────────────────────────────
def load_model(base: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()  # fold adapter for faster inference
    model.eval()
    return model, tok


def generate_sql(model, tok, prompt: str, max_new_tokens: int = 256) -> str:
    import torch

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_sql(text)


def clean_sql(text: str) -> str:
    """Keep only the first SQL statement; strip code fences / trailing prose."""
    text = text.replace("```sql", "").replace("```", "").strip()
    # cut at the first sign of the model continuing past the answer
    for stop in ("\n###", "\nQuestion:", "\n--"):
        if stop in text:
            text = text.split(stop)[0]
    text = text.strip()
    if ";" in text:
        text = text.split(";")[0] + ";"
    return text.strip()


# ── Evaluation loop ────────────────────────────────────────────────────────────
def evaluate(model, tok, examples, spider_db_dir: Path, limit: int) -> dict:
    correct = 0
    total = 0
    runnable = 0  # SQL that at least executes without error
    for ex in examples[:limit] if limit else examples:
        db_file = spider_db_dir / ex["db_id"] / f"{ex['db_id']}.sqlite"
        if not db_file.exists():
            continue
        total += 1
        pred_sql = generate_sql(model, tok, ex["prompt"])
        gold_ok, gold_rows = run_sql(db_file, ex["completion"])
        pred_ok, pred_rows = run_sql(db_file, pred_sql)
        if pred_ok:
            runnable += 1
        if gold_ok and result_match(gold_rows, pred_rows):
            correct += 1
    return {
        "total": total,
        "correct": correct,
        "exec_accuracy": round(correct / total, 4) if total else 0.0,
        "runnable_rate": round(runnable / total, 4) if total else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Execution-accuracy eval (before/after QLoRA).")
    ap.add_argument("--dev", default="finetune/data/dev.jsonl")
    ap.add_argument("--spider-db-dir", required=True, help="path to spider/database/")
    ap.add_argument("--base", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    ap.add_argument("--adapter", default="finetune/adapter")
    ap.add_argument("--limit", type=int, default=200, help="held-out examples to score")
    ap.add_argument("--out", default="finetune/eval_report.json")
    args = ap.parse_args()

    examples = [json.loads(l) for l in Path(args.dev).read_text(encoding="utf-8").splitlines()]
    db_dir = Path(args.spider_db_dir)

    print(f"== BEFORE fine-tuning (base: {args.base}) ==")
    base_model, tok = load_model(args.base, adapter=None)
    before = evaluate(base_model, tok, examples, db_dir, args.limit)
    print(before)
    del base_model

    print(f"\n== AFTER fine-tuning (adapter: {args.adapter}) ==")
    ft_model, tok = load_model(args.base, adapter=args.adapter)
    after = evaluate(ft_model, tok, examples, db_dir, args.limit)
    print(after)

    report = {
        "base_model": args.base,
        "adapter": args.adapter,
        "n_examples": before["total"],
        "before": before,
        "after": after,
        "exec_accuracy_delta": round(after["exec_accuracy"] - before["exec_accuracy"], 4),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n── EXECUTION-ACCURACY: BEFORE vs AFTER ───────────────────────")
    print(f"  before: {before['exec_accuracy']:.1%}   after: {after['exec_accuracy']:.1%}"
          f"   Δ: {report['exec_accuracy_delta']:+.1%}")
    print(f"  report written to {args.out}")


if __name__ == "__main__":
    main()
