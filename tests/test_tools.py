"""Tools — SQL execution/schema and retrieval. Offline (no LLM)."""
from src.tools.retrieval import ingest_corpus, retrieve
from src.tools.text_to_sql import get_schema, is_sql_safe, run_sql, text_to_sql


# ── text_to_sql: schema + guarded execution (no model needed) ──────────────────
def test_schema_lists_real_tables():
    schema = get_schema()
    assert "companies" in schema and "financials" in schema
    assert "revenue_musd" in schema


def test_run_sql_returns_real_rows():
    cols, rows = run_sql(
        "SELECT name, revenue_musd FROM companies JOIN financials USING(cik) "
        "ORDER BY revenue_musd DESC LIMIT 1"
    )
    assert cols == ["name", "revenue_musd"]
    assert rows and rows[0][0] == "Walmart Inc."   # real SEC data


def test_text_to_sql_blocks_unsafe_generated_sql(monkeypatch):
    # Force the generator to emit an unsafe query; the guard must block execution.
    monkeypatch.setattr("src.tools.text_to_sql.generate_sql",
                        lambda q, schema=None: ("DELETE FROM companies", "test"))
    out = text_to_sql("delete everything")
    assert out["safe"] is False
    assert out["rows"] == []
    assert "blocked by SQL guard" in (out["error"] or "")


def test_text_to_sql_executes_safe_generated_sql(monkeypatch):
    monkeypatch.setattr(
        "src.tools.text_to_sql.generate_sql",
        lambda q, schema=None: ("SELECT COUNT(*) AS n FROM companies", "test"),
    )
    out = text_to_sql("how many companies")
    assert out["safe"] is True and out["error"] is None
    assert out["row_count"] == 1 and out["rows"][0][0] == 6
    assert out["provenance"] == "test"


# ── retrieval (Chroma + ONNX embedder; offline, may download model once) ───────
def test_retrieval_returns_sourced_chunks():
    ingest_corpus()
    res = retrieve("Apple competition and risk factors", top_k=3)
    assert res["chunks"]
    for c in res["chunks"]:
        assert c["source"].endswith(".md")
        assert isinstance(c["score"], float)
    # an Apple-specific query should surface the Apple filing somewhere in top-k
    sources = {c["source"] for c in retrieve("Apple iPhone segments", top_k=4)["chunks"]}
    assert any("aapl" in s for s in sources)
