"""Fetch REAL company financials + 10-K text from SEC EDGAR.

No synthetic data. Sources (all public, free, authoritative):
  - Financials: SEC XBRL company-facts API (us-gaap concepts)
      https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
  - Company metadata + filing list: SEC submissions API
      https://data.sec.gov/submissions/CIK##########.json
  - 10-K narrative (Business / Risk Factors / MD&A): the filing's primary document
      https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<doc>

Outputs:
  - data/db/financials.sqlite   (built DB; git-ignored)
  - data/db/financials.sql      (schema + INSERTs dump; committed, reproducible)
  - data/corpus/<ticker>_10k.md (real 10-K excerpts; committed)

Usage:
    python data/fetch_sec_data.py

SEC requires a descriptive User-Agent with a contact email (SEC_USER_AGENT in .env).
Politeness: <=10 req/s; we sleep between calls.
"""
from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import time
import urllib.request
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve().parent
DB_FILE = HERE / "db" / "financials.sqlite"
SQL_DUMP = HERE / "db" / "financials.sql"
CORPUS_DIR = HERE / "corpus"

UA = os.getenv("SEC_USER_AGENT", "multi-agent-analyst contact@example.com")
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}

# Real public companies (CIKs are public SEC identifiers). Sector is pulled live
# from SEC (sicDescription) — nothing here is invented.
CIKS = [
    320193,   # Apple
    789019,   # Microsoft
    1045810,  # NVIDIA
    200406,   # Johnson & Johnson
    104169,   # Walmart
    34088,    # Exxon Mobil
]

N_YEARS = 5  # keep the most recent N fiscal years per company

# us-gaap concept fallbacks (companies tag revenue differently)
CONCEPTS = {
    "revenue_musd": (["RevenueFromContractWithCustomerExcludingAssessedTax",
                       "Revenues", "RevenueFromContractWithCustomerIncludingAssessedTax",
                       "SalesRevenueNet"], False),
    "gross_profit_musd": (["GrossProfit"], False),
    "operating_income_musd": (["OperatingIncomeLoss"], False),
    "net_income_musd": (["NetIncomeLoss"], False),
    # Prefer the "excluding acquired in-process" R&D line (clean operating R&D, e.g.
    # JNJ ~$15-17B); fall back to the plain tag for filers that lack it (Apple/MSFT).
    "rnd_musd": (["ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
                   "ResearchAndDevelopmentExpense"], False),
    "assets_musd": (["Assets"], True),
    "liabilities_musd": (["Liabilities"], True),
    "cash_musd": (["CashAndCashEquivalentsAtCarryingValue"], True),
}


def _get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            import gzip
            raw = gzip.decompress(raw)
        return raw


def _get_json(url: str):
    return json.loads(_get(url).decode("utf-8"))


# ── financials ─────────────────────────────────────────────────────────────────
def annual_series(facts: dict, tags: list[str], instant: bool) -> dict[int, float]:
    """{fiscal_year(end-date year): value_in_millions} from 10-K XBRL points.

    Merges across the fallback tags (companies change tags over the years), so a
    later tag fills years the primary tag is missing. First tag in `tags` wins on
    a per-year conflict; within a tag, the most recently *filed* value wins.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    merged: dict[int, float] = {}
    for tag in tags:
        node = gaap.get(tag)
        if not node:
            continue
        per_tag: dict[int, tuple[float, str]] = {}  # year -> (val, filed)
        for unit, entries in node.get("units", {}).items():
            if unit != "USD":
                continue
            for e in entries:
                if not e.get("form", "").startswith("10-K"):
                    continue
                end = e.get("end")
                if not end:
                    continue
                if instant:
                    if e.get("start"):
                        continue  # want point-in-time
                else:
                    start = e.get("start")
                    if not start:
                        continue
                    days = (date.fromisoformat(end) - date.fromisoformat(start)).days
                    if not (350 <= days <= 380):  # full fiscal year only
                        continue
                yr = int(end[:4])
                filed = e.get("filed", "")
                if yr not in per_tag or filed > per_tag[yr][1]:
                    per_tag[yr] = (e["val"], filed)
        for yr, (val, _f) in per_tag.items():
            merged.setdefault(yr, val)  # earlier tag in the list wins per year
    return {y: round(v / 1e6, 2) for y, v in merged.items()}


def fetch_company(cik: int) -> dict:
    cik10 = f"{cik:010d}"
    sub = _get_json(f"https://data.sec.gov/submissions/CIK{cik10}.json")
    time.sleep(0.2)
    facts = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")
    time.sleep(0.2)

    tickers = sub.get("tickers") or ["?"]
    company = {
        "cik": cik,
        "name": sub.get("name", "").title() if sub.get("name", "").isupper() else sub.get("name", ""),
        "ticker": tickers[0],
        "sector": sub.get("sicDescription", "Unknown"),
    }

    series = {col: annual_series(facts, tags, instant) for col, (tags, instant) in CONCEPTS.items()}
    years = sorted({y for s in series.values() for y in s}, reverse=True)[:N_YEARS]
    financials = []
    for y in sorted(years):
        row = {"cik": cik, "fiscal_year": y}
        for col in CONCEPTS:
            row[col] = series[col].get(y)
        financials.append(row)

    company["filings"] = sub.get("filings", {}).get("recent", {})
    return {"company": company, "financials": financials}


# ── 10-K narrative text ──────────────────────────────────────────────────────
def strip_html(raw_html: str) -> str:
    txt = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    txt = re.sub(r"(?s)<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    txt = re.sub(r"[ \t\xa0]+", " ", txt)
    txt = re.sub(r"\n\s*\n\s*\n+", "\n\n", txt)
    return txt.strip()


def extract_section(text: str, start_pat: str, max_chars: int) -> str:
    """Grab the real section body, skipping the table-of-contents occurrence.

    10-Ks list every "Item N" twice: once in the TOC near the top, once as the
    actual section later. We skip matches in the first 8% of the document (the TOC
    region) and prefer the one whose following text reads like prose, not a TOC.
    """
    matches = list(re.finditer(start_pat, text, re.I))
    if not matches:
        return ""
    toc_cutoff = int(len(text) * 0.08)
    body = [m for m in matches if m.start() > toc_cutoff]
    chosen = body[0] if body else matches[-1]
    chunk = text[chosen.start(): chosen.start() + max_chars]
    return chunk.strip()


def fetch_10k_text(company: dict) -> tuple[str, str, str] | None:
    """Return (filing_url, filing_date, narrative_text) for the latest 10-K."""
    recent = company["filings"]
    forms = recent.get("form", [])
    try:
        idx = forms.index("10-K")
    except ValueError:
        return None
    accession = recent["accessionNumber"][idx].replace("-", "")
    doc = recent["primaryDocument"][idx]
    fdate = recent["filingDate"][idx]
    url = f"https://www.sec.gov/Archives/edgar/data/{company['cik']}/{accession}/{doc}"
    raw = _get(url).decode("utf-8", "ignore")
    time.sleep(0.3)
    text = strip_html(raw)

    business = extract_section(text, r"Item\s*1\.?\s*Business", 6000)
    risks = extract_section(text, r"Item\s*1A\.?\s*Risk\s*Factors", 9000)
    mdna = extract_section(text, r"Management.{0,5}s\s+Discussion\s+and\s+Analysis", 7000)
    parts = [p for p in (business, risks, mdna) if p]
    narrative = "\n\n".join(parts) if parts else text[:12000]
    return url, fdate, narrative


# ── build DB + corpus ──────────────────────────────────────────────────────────
SCHEMA = """\
DROP TABLE IF EXISTS financials;
DROP TABLE IF EXISTS companies;

CREATE TABLE companies (
    cik     INTEGER PRIMARY KEY,
    name    TEXT NOT NULL,
    ticker  TEXT NOT NULL,
    sector  TEXT NOT NULL
);

CREATE TABLE financials (
    cik                   INTEGER NOT NULL REFERENCES companies(cik),
    fiscal_year           INTEGER NOT NULL,
    revenue_musd          REAL,
    gross_profit_musd     REAL,
    operating_income_musd REAL,
    net_income_musd       REAL,
    rnd_musd              REAL,
    assets_musd           REAL,
    liabilities_musd      REAL,
    cash_musd             REAL,
    PRIMARY KEY (cik, fiscal_year)
);
"""

FIN_COLS = ["cik", "fiscal_year", "revenue_musd", "gross_profit_musd",
            "operating_income_musd", "net_income_musd", "rnd_musd",
            "assets_musd", "liabilities_musd", "cash_musd"]


def main() -> None:
    print(f"SEC User-Agent: {UA}")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    # remove any prior synthetic corpus so nothing synthetic lingers
    for old in CORPUS_DIR.glob("*.md"):
        old.unlink()

    con = sqlite3.connect(DB_FILE)
    con.executescript(SCHEMA)

    for cik in CIKS:
        data = fetch_company(cik)
        c = data["company"]
        print(f"  {c['ticker']:6} {c['name']:30} {c['sector'][:30]:30} "
              f"({len(data['financials'])} yrs)")
        con.execute("INSERT INTO companies VALUES (?,?,?,?)",
                    (c["cik"], c["name"], c["ticker"], c["sector"]))
        for row in data["financials"]:
            con.execute(
                f"INSERT INTO financials ({','.join(FIN_COLS)}) VALUES ({','.join(['?']*len(FIN_COLS))})",
                tuple(row[col] for col in FIN_COLS),
            )

        tk = fetch_10k_text(c)
        if tk:
            url, fdate, narrative = tk
            md = (f"# {c['name']} ({c['ticker']}) — 10-K excerpt\n\n"
                  f"> Source: SEC EDGAR filing, filed {fdate}. {url}\n"
                  f"> Sector (SEC SIC): {c['sector']}\n\n{narrative}\n")
            (CORPUS_DIR / f"{c['ticker'].lower()}_10k.md").write_text(md, encoding="utf-8")
            print(f"         corpus: {c['ticker'].lower()}_10k.md  ({len(narrative)} chars, filed {fdate})")

    con.commit()

    # dump a committed .sql (real data, reproducible) — .sqlite is git-ignored
    with SQL_DUMP.open("w", encoding="utf-8") as f:
        f.write("-- REAL company financials from SEC EDGAR XBRL. Built by data/fetch_sec_data.py.\n")
        for line in con.iterdump():
            f.write(line + "\n")

    n_co = con.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    n_fin = con.execute("SELECT COUNT(*) FROM financials").fetchone()[0]
    con.close()
    print(f"\n[ok] {n_co} companies, {n_fin} financial rows -> {DB_FILE}")
    print(f"[ok] SQL dump -> {SQL_DUMP}")


if __name__ == "__main__":
    main()
