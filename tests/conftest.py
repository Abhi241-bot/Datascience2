"""Shared test fixtures."""
import os
import sys
from pathlib import Path

import pytest

# repo root on path so `import src...` works when running `pytest` from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bootstrap import ensure_db  # noqa: E402

# Most tests are offline (no API). Tests that call the LLM are guarded by this.
HAS_GROQ = bool(os.getenv("GROQ_API_KEY"))
requires_groq = pytest.mark.skipif(not HAS_GROQ, reason="needs GROQ_API_KEY (live LLM)")


@pytest.fixture(scope="session", autouse=True)
def _db():
    """Ensure the SQLite DB exists (rebuilt from the committed dump if needed)."""
    ensure_db()
