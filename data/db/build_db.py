"""Build the sample financials SQLite DB from financials.sql.

    python data/db/build_db.py

Idempotent — drops/recreates tables. The .sqlite file is git-ignored; rebuild
from the committed financials.sql anywhere.
"""
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SQL_FILE = HERE / "financials.sql"
DB_FILE = HERE / "financials.sqlite"


def build() -> Path:
    script = SQL_FILE.read_text(encoding="utf-8")
    con = sqlite3.connect(DB_FILE)
    con.executescript(script)
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM financials").fetchone()[0]
    con.close()
    print(f"[ok] built {DB_FILE}  ({n} financial rows)")
    return DB_FILE


if __name__ == "__main__":
    build()
    sys.exit(0)
