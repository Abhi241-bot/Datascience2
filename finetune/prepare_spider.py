"""Phase 1 — Download Spider and convert to an instruction-tuning dataset.

Spider is the standard text-to-SQL benchmark: (database schema + natural-language
question) -> SQL query. We render each example into a single instruction prompt so
the model learns to emit *only* the SQL given a schema and a question.

Output: JSONL files with a `text` field (full prompt+completion) plus structured
fields (`prompt`, `completion`, `db_id`) used later by evaluate_sql.py for
execution-accuracy scoring.

Usage:
    python finetune/prepare_spider.py --out finetune/data
    # then upload finetune/data/{train,dev}.jsonl to Colab, or load via datasets.

Runs anywhere (CPU) — no GPU needed. The heavy training is in train_qlora.ipynb.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

# ── Prompt template ────────────────────────────────────────────────────────────
# Kept identical here, in train_qlora.ipynb, and in src/tools/text_to_sql.py so
# train-time and inference-time prompts match exactly.
SYSTEM = (
    "You are a precise text-to-SQL engine. Given a database schema and a question, "
    "output a single valid SQLite query that answers it. Output ONLY the SQL, no prose."
)

PROMPT_TEMPLATE = (
    "{system}\n\n"
    "### Database schema:\n{schema}\n\n"
    "### Question:\n{question}\n\n"
    "### SQL:\n"
)


def build_prompt(schema: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(system=SYSTEM, schema=schema.strip(), question=question.strip())


def schema_from_tables(db_id: str, tables_index: dict) -> str:
    """Render a compact CREATE-TABLE-style schema string for a Spider db_id.

    Uses Spider's tables.json metadata (column names + types + foreign keys).
    """
    meta = tables_index[db_id]
    table_names = meta["table_names_original"]
    columns = meta["column_names_original"]  # [[table_idx, col_name], ...]
    col_types = meta["column_types"]
    fks = meta.get("foreign_keys", [])

    # group columns by table
    per_table: dict[int, list[str]] = {i: [] for i in range(len(table_names))}
    for col_idx, (tbl_idx, col_name) in enumerate(columns):
        if tbl_idx == -1:  # the '*' pseudo-column
            continue
        per_table[tbl_idx].append(f"{col_name} {col_types[col_idx].upper()}")

    lines = []
    for tbl_idx, tbl in enumerate(table_names):
        cols = ", ".join(per_table[tbl_idx])
        lines.append(f"CREATE TABLE {tbl} ({cols});")

    # foreign keys as a hint
    for a, b in fks:
        ta, ca = columns[a]
        tb, cb = columns[b]
        lines.append(
            f"-- FK: {table_names[ta]}.{ca} -> {table_names[tb]}.{cb}"
        )
    return "\n".join(lines)


def convert_split(examples, tables_index: dict) -> list[dict]:
    rows = []
    for ex in examples:
        db_id = ex["db_id"]
        if db_id not in tables_index:
            continue
        schema = schema_from_tables(db_id, tables_index)
        prompt = build_prompt(schema, ex["question"])
        sql = ex["query"].strip()
        rows.append(
            {
                "db_id": db_id,
                "question": ex["question"],
                "schema": schema,
                "prompt": prompt,
                "completion": sql,
                # full sequence the model trains on (prompt + answer + EOS handled by trainer)
                "text": prompt + sql,
            }
        )
    return rows


def load_spider_via_datasets():
    """Load Spider through HuggingFace `datasets`. Falls back with a clear message."""
    from datasets import load_dataset

    # `spider` ships train + validation splits with question/query/db_id, plus a
    # `tables` config holding schema metadata.
    ds = load_dataset("spider")
    tables = load_dataset("spider", "tables")  # has table_names_original etc.
    tables_index = {row["db_id"]: row for row in tables["train"]}
    return ds["train"], ds["validation"], tables_index


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Spider for QLoRA instruction tuning.")
    ap.add_argument("--out", default="finetune/data", help="output dir for jsonl files")
    ap.add_argument("--max-train", type=int, default=0, help="cap train examples (0 = all)")
    args = ap.parse_args()

    out = Path(args.out)
    print("Loading Spider via HuggingFace datasets …")
    train_raw, dev_raw, tables_index = load_spider_via_datasets()

    train_rows = convert_split(train_raw, tables_index)
    dev_rows = convert_split(dev_raw, tables_index)
    if args.max_train:
        train_rows = train_rows[: args.max_train]

    write_jsonl(train_rows, out / "train.jsonl")
    write_jsonl(dev_rows, out / "dev.jsonl")

    # one inspectable sample for the README / sanity check
    print(f"\n✅ train={len(train_rows)}  dev={len(dev_rows)}  ->  {out}/")
    print("\n── Sample prompt ─────────────────────────────────────────────")
    print(train_rows[0]["prompt"])
    print("── Expected SQL ──────────────────────────────────────────────")
    print(train_rows[0]["completion"])


if __name__ == "__main__":
    main()
