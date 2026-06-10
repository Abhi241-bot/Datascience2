"""First-run data bootstrap (used by the Gradio app / HF Spaces).

The committed artifacts are the real SEC data in source form:
  - data/db/financials.sql  (schema + INSERTs dump)  -> build the SQLite DB
  - data/corpus/*_10k.md     (real 10-K excerpts)     -> ingest into Chroma
The built .sqlite and the Chroma index are NOT committed; this rebuilds them on
startup if missing. Idempotent and fast.
"""
from __future__ import annotations

import sqlite3

from src import config


def ensure_db() -> None:
    if config.DB_PATH.exists():
        return
    if not config.DB_SCHEMA_PATH.exists():
        print(f"[bootstrap] no DB and no SQL dump at {config.DB_SCHEMA_PATH}; "
              "run `python data/fetch_sec_data.py`")
        return
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    con.executescript(config.DB_SCHEMA_PATH.read_text(encoding="utf-8"))
    con.commit()
    con.close()
    print(f"[bootstrap] built {config.DB_PATH} from {config.DB_SCHEMA_PATH.name}")


def ensure_corpus_index() -> None:
    try:
        from src.tools.retrieval import ingest_corpus

        n = ingest_corpus()
        print(f"[bootstrap] corpus index ready ({n} chunks)")
    except Exception as e:  # don't block app startup on a warm-up failure
        print(f"[bootstrap] corpus index warm-up skipped: {e}")


def ensure_data() -> None:
    ensure_db()
    ensure_corpus_index()


if __name__ == "__main__":
    ensure_data()
