"""Text-to-SQL tool — powered by the QLoRA model fine-tuned in Phase 1.

Pipeline: question + live DB schema -> SQL (fine-tuned adapter, with a Groq
fallback) -> **read-only safety guard** -> execute against the sample DB ->
structured rows + provenance.

Adapter resolution order:
  1. `SQL_ADAPTER_REPO` (HF Hub repo id) if set,
  2. local `finetune/adapter/` if it contains a real adapter,
  3. otherwise fall back to the Groq orchestrator to draft SQL so the agent works
     end-to-end *before* the Colab fine-tune lands. Provenance records which path
     was used, so the demo never silently pretends the adapter ran.

Standalone use:
    from src.tools.text_to_sql import text_to_sql
    print(text_to_sql("Which company had the highest 2023 revenue?"))
"""
from __future__ import annotations

import functools
import sqlite3
from pathlib import Path
from typing import Any

from src import config

# Mirror the EXACT training prompt (finetune/prepare_spider.py) so the fine-tuned
# model sees the distribution it was trained on.
SYSTEM = (
    "You are a precise text-to-SQL engine. Given a database schema and a question, "
    "output a single valid SQLite query that answers it. Output ONLY the SQL, no prose."
)
PROMPT_TEMPLATE = (
    "{system}\n\n### Database schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n"
)


# ── SQL safety guard (read-only) ───────────────────────────────────────────────
def is_sql_safe(sql: str) -> tuple[bool, str]:
    """Return (safe, reason). Blocks writes/DDL and multi-statement injection.

    Phase 4's guardrails reuse this. Conservative by design: a single read-only
    SELECT/WITH only.
    """
    import sqlparse

    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False, "empty SQL"

    statements = [s for s in sqlparse.split(sql) if s.strip()]
    if len(statements) > 1:
        return False, "multiple statements are not allowed"

    upper = stripped.upper()
    for kw in config.SQL_FORBIDDEN_KEYWORDS:
        # word-boundary-ish check to avoid matching column names containing the kw
        if f" {kw} " in f" {upper} " or upper.startswith(kw + " "):
            return False, f"forbidden keyword: {kw}"

    first = upper.split(None, 1)[0] if upper.split() else ""
    if first not in ("SELECT", "WITH"):
        return False, f"only SELECT/WITH queries are allowed (got {first or 'nothing'})"

    return True, "ok"


# ── DB schema introspection ────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def get_schema(db_path: str | None = None) -> str:
    """Render a CREATE-TABLE-style schema string from the live SQLite DB."""
    path = Path(db_path) if db_path else config.DB_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"DB not found at {path}. Build it: python data/db/build_db.py"
        )
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall()
    con.close()
    # normalise whitespace; the committed CREATE statements are already readable
    return "\n".join(r[0].strip() + ";" for r in rows)


# ── SQL generation: fine-tuned adapter, with Groq fallback ─────────────────────
@functools.lru_cache(maxsize=1)
def _load_finetuned():
    """Lazy-load base+adapter. Returns (model, tokenizer) or None if unavailable."""
    adapter_src = config.SQL_ADAPTER_REPO or (
        str(config.ADAPTER_DIR)
        if (config.ADAPTER_DIR / "adapter_config.json").exists()
        else ""
    )
    if not adapter_src:
        return None  # no adapter yet -> caller uses Groq fallback
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(config.SQL_BASE_MODEL)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config.SQL_BASE_MODEL, device_map="auto", torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(model, adapter_src).merge_and_unload()
        model.eval()
        return model, tok
    except Exception as e:  # missing GPU/bitsandbytes/etc. -> graceful fallback
        print(f"[text_to_sql] fine-tuned model unavailable ({e}); using Groq fallback")
        return None


def _clean_sql(text: str) -> str:
    text = text.replace("```sql", "").replace("```", "").strip()
    for stop in ("\n###", "\nQuestion:", "\n--", "\n\n"):
        if stop in text:
            text = text.split(stop)[0]
    text = text.strip()
    if ";" in text:
        text = text.split(";")[0] + ";"
    return text.strip()


def _gen_with_adapter(prompt: str) -> str:
    import torch

    model, tok = _load_finetuned()
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=config.SQL_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    return _clean_sql(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def _gen_with_groq(prompt: str) -> str:
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=config.ORCHESTRATOR_MODEL, temperature=0, api_key=config.GROQ_API_KEY)
    resp = llm.invoke(prompt)
    return _clean_sql(resp.content)


def generate_sql(question: str, schema: str | None = None) -> tuple[str, str]:
    """Return (sql, provenance). provenance ∈ {finetuned-adapter, groq-fallback}."""
    schema = schema or get_schema()
    prompt = PROMPT_TEMPLATE.format(system=SYSTEM, schema=schema, question=question)
    if _load_finetuned() is not None:
        return _gen_with_adapter(prompt), "finetuned-adapter"
    return _gen_with_groq(prompt), "groq-fallback"


# ── Execution ──────────────────────────────────────────────────────────────────
def run_sql(sql: str, db_path: str | None = None) -> tuple[list[str], list[tuple]]:
    path = Path(db_path) if db_path else config.DB_PATH
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return cols, rows
    finally:
        con.close()


# ── Public tool entrypoint ─────────────────────────────────────────────────────
def text_to_sql(question: str, db_path: str | None = None) -> dict[str, Any]:
    """NL question -> SQL -> guarded execution -> structured result with provenance."""
    schema = get_schema(db_path)
    try:
        sql, provenance = generate_sql(question, schema)
    except Exception as e:
        return {"question": question, "sql": None, "error": f"generation failed: {e}",
                "safe": False, "rows": [], "columns": [], "row_count": 0, "source": "text_to_sql"}

    safe, reason = is_sql_safe(sql)
    if not safe:
        return {"question": question, "sql": sql, "error": f"blocked by SQL guard: {reason}",
                "safe": False, "rows": [], "columns": [], "row_count": 0,
                "provenance": provenance, "source": "text_to_sql"}

    try:
        cols, rows = run_sql(sql, db_path)
    except Exception as e:
        return {"question": question, "sql": sql, "error": f"execution error: {e}",
                "safe": True, "rows": [], "columns": [], "row_count": 0,
                "provenance": provenance, "source": "text_to_sql"}

    return {
        "question": question,
        "sql": sql,
        "safe": True,
        "error": None,
        "columns": cols,
        "rows": [list(r) for r in rows],
        "row_count": len(rows),
        "provenance": provenance,   # finetuned-adapter | groq-fallback
        "source": "text_to_sql",
    }


if __name__ == "__main__":
    import json

    for q in [
        "Which company had the highest revenue in 2023?",
        "List the segment revenues for Nimbus Cloud in 2023, largest first.",
        "Delete all rows from companies.",  # should be blocked
    ]:
        print("Q:", q)
        print(json.dumps(text_to_sql(q), indent=2, default=str))
        print("-" * 60)
