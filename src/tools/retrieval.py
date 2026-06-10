"""Retrieval tool — Chroma vector search over the financials corpus.

Ingests `data/corpus/*.{md,txt}` into a persistent Chroma collection using a local
(free, no-API-key) sentence-transformers embedding model, and returns the top-k
chunks **with their source filenames** so the analyst can cite them.

Standalone use:
    from src.tools.retrieval import ingest_corpus, retrieve
    ingest_corpus()                       # one-time (idempotent)
    print(retrieve("What drove Nimbus Cloud revenue growth?"))
"""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any

from src import config


# ── chunking ───────────────────────────────────────────────────────────────────
def _chunk(text: str, size: int, overlap: int) -> list[str]:
    """Simple character-window chunker with overlap; splits on blank lines first."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            # carry overlap from the tail of the previous chunk
            tail = buf[-overlap:] if overlap and buf else ""
            buf = f"{tail}\n\n{p}".strip() if tail else p
    if buf:
        chunks.append(buf)
    return chunks


@functools.lru_cache(maxsize=1)
def _client():
    import chromadb

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


@functools.lru_cache(maxsize=1)
def _embedder():
    """Local sentence-transformers embedding function for Chroma."""
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )


def _collection():
    return _client().get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        embedding_function=_embedder(),
        metadata={"hnsw:space": "cosine"},
    )


# ── ingest ───────────────────────────────────────────────────────────────────
def ingest_corpus(corpus_dir: Path | None = None, reset: bool = False) -> int:
    """Ingest corpus docs into Chroma. Idempotent (stable ids from content hash).

    Returns the number of chunks in the collection after ingest.
    """
    corpus_dir = corpus_dir or config.CORPUS_DIR
    files = sorted(p for p in corpus_dir.glob("**/*") if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise FileNotFoundError(f"No .md/.txt docs found in {corpus_dir}")

    if reset:
        try:
            _client().delete_collection(config.CHROMA_COLLECTION)
        except Exception:
            pass

    col = _collection()
    ids, docs, metas = [], [], []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for i, ch in enumerate(_chunk(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            cid = hashlib.sha1(f"{f.name}:{i}:{ch[:64]}".encode()).hexdigest()
            ids.append(cid)
            docs.append(ch)
            metas.append({"source": f.name, "chunk": i})

    # upsert keeps ingest idempotent across re-runs
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    return col.count()


# ── search ───────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int | None = None) -> dict[str, Any]:
    """Semantic search; returns chunks with source + similarity score."""
    top_k = top_k or config.RETRIEVAL_TOP_K
    col = _collection()
    if col.count() == 0:
        ingest_corpus()  # lazy first-time ingest
        col = _collection()

    res = col.query(query_texts=[query], n_results=top_k)
    chunks = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        chunks.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "score": round(1.0 - dist, 4),  # cosine distance -> similarity
            }
        )
    return {"query": query, "chunks": chunks, "source": "retrieval"}


if __name__ == "__main__":
    import json

    n = ingest_corpus(reset=True)
    print(f"[ok] ingested {n} chunks")
    print(json.dumps(retrieve("What drove Nimbus Cloud revenue growth in 2023?"), indent=2))
