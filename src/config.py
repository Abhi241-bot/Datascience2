"""Single source of truth for models, thresholds, and paths.

Per the build spec (§5): ONE config file for everything. Import from here;
never hard-code a model name, threshold, or path elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # read .env if present (see .env.example)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"          # raw domain docs (company financials)
CHROMA_DIR = DATA_DIR / "chroma"          # persisted Chroma vector store
DB_PATH = DATA_DIR / "db" / "financials.sqlite"   # sample SQL database
DB_SCHEMA_PATH = DATA_DIR / "db" / "financials.sql"
ADAPTER_DIR = ROOT / "finetune" / "adapter"        # exported QLoRA adapter
EVAL_RESULTS_DIR = ROOT / "eval_results"

CHROMA_COLLECTION = "financials_corpus"

# ── Orchestrator LLM (Groq — OpenAI-compatible, free tier) ─────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# llama-3.1-8b-instant: fast + a generous free-tier daily token budget (the 70B
# model's 100K tokens/day exhausts quickly under multi-agent + eval loads). Set
# ORCHESTRATOR_MODEL=llama-3.3-70b-versatile in .env for higher-quality reasoning.
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "llama-3.1-8b-instant")
ORCHESTRATOR_TEMPERATURE = float(os.getenv("ORCHESTRATOR_TEMPERATURE", "0.1"))

# ── Embeddings (local, free, torch-free) ───────────────────────────────────────
# "onnx-default" -> Chroma's built-in ONNX all-MiniLM-L6-v2 (onnxruntime, no torch;
# works on Py 3.11-3.14, small Spaces image). Set to a sentence-transformers model
# id (e.g. "sentence-transformers/all-mpnet-base-v2") to use that instead.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "onnx-default")

# ── Text-to-SQL fine-tuned model ───────────────────────────────────────────────
# Base model fine-tuned with QLoRA on Spider; adapter loaded on top at inference.
SQL_BASE_MODEL = os.getenv("SQL_BASE_MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
# Either a HF Hub repo id with the pushed adapter, or "" to use local ADAPTER_DIR.
SQL_ADAPTER_REPO = os.getenv("SQL_ADAPTER_REPO", "")
SQL_MAX_NEW_TOKENS = int(os.getenv("SQL_MAX_NEW_TOKENS", "256"))

# ── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# ── Web search ─────────────────────────────────────────────────────────────────
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # optional; falls back to DuckDuckGo

# ── Agent loop control ─────────────────────────────────────────────────────────
MAX_RESEARCH_LOOPS = int(os.getenv("MAX_RESEARCH_LOOPS", "3"))

# ── Guardrails ─────────────────────────────────────────────────────────────────
# SQL statements that must never run (read-only enforcement).
SQL_FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "REPLACE", "GRANT", "REVOKE", "ATTACH", "DETACH", "PRAGMA",
)
PII_ENTITIES = ("EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN", "IBAN_CODE")

# ── Eval thresholds (the scorecard pass/fail bar) ──────────────────────────────
EVAL_THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.70,
}

# ── Observability (LangSmith) ──────────────────────────────────────────────────
LANGSMITH_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multi-agent-analyst")
