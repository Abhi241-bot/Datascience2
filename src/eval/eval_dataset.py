"""Curated golden Q/A set for evaluating the analyst over the real corpus + DB.

Each item: a question, a reference answer (ground_truth) grounded in the real SEC
data, and the kind of evidence it should use. Quantitative ground truths were
verified directly against data/db/financials.sqlite (latest fiscal year per
company); qualitative ones come from the real 10-K excerpts in data/corpus/.

Used by ragas_eval.py (faithfulness / answer relevancy / context precision+recall)
and deepeval_eval.py (citation correctness / analytical depth).
"""
from __future__ import annotations

# kind: "sql" (needs the DB), "corpus" (needs 10-K retrieval), "mixed" (both)
GOLDEN_SET: list[dict] = [
    {
        "question": "Which company had the highest revenue in its most recent fiscal year, and how much was it?",
        "ground_truth": "Walmart had the highest revenue in its most recent fiscal year (FY2026), at about $706 billion ($706,413 million).",
        "kind": "sql",
    },
    {
        "question": "Which company reported the highest net income in its latest fiscal year?",
        "ground_truth": "NVIDIA reported the highest net income in its latest fiscal year (FY2026), about $120 billion ($120,067 million), ahead of Apple's ~$112 billion.",
        "kind": "sql",
    },
    {
        "question": "Did Apple or Microsoft spend more on R&D in their most recent fiscal year, and by how much?",
        "ground_truth": "Apple spent more on R&D than Microsoft in the latest fiscal year: about $34.55 billion versus Microsoft's ~$32.49 billion, a difference of roughly $2 billion.",
        "kind": "sql",
    },
    {
        "question": "How much did Johnson & Johnson spend on research and development in its most recent fiscal year?",
        "ground_truth": "Johnson & Johnson spent about $14.7 billion ($14,665 million) on R&D in its most recent fiscal year (FY2025).",
        "kind": "sql",
    },
    {
        "question": "Rank the technology companies (Apple, Microsoft, NVIDIA) by latest-year revenue.",
        "ground_truth": "By latest-year revenue: Apple first (~$416B), then Microsoft (~$282B), then NVIDIA (~$216B).",
        "kind": "sql",
    },
    {
        "question": "What does Apple identify as key competitive pressures in its business?",
        "ground_truth": "Apple describes highly competitive markets marked by aggressive price competition, downward pressure on gross margins, short product life cycles, frequent new-product introductions, and competitors imitating its products and infringing its IP; it competes by continually introducing innovative products and services.",
        "kind": "corpus",
    },
    {
        "question": "How does Apple organize its reportable operating segments?",
        "ground_truth": "Apple manages its business primarily on a geographic basis, with reportable segments: Americas, Europe, Greater China, Japan, and Rest of Asia Pacific.",
        "kind": "corpus",
    },
    {
        "question": "Which company was the most profitable by net margin in the latest year, and what risks could affect it?",
        "ground_truth": "NVIDIA had the highest net margin in the latest year (net income ~$120B on revenue ~$216B, ~55%); risks from its filing include competition, dependence on demand for its products, and supply/operational risks.",
        "kind": "mixed",
    },
]


def questions() -> list[str]:
    return [item["question"] for item in GOLDEN_SET]


def by_kind(kind: str) -> list[dict]:
    return [item for item in GOLDEN_SET if item["kind"] == kind]


if __name__ == "__main__":
    print(f"{len(GOLDEN_SET)} golden questions "
          f"({len(by_kind('sql'))} sql, {len(by_kind('corpus'))} corpus, {len(by_kind('mixed'))} mixed)")
    for i, it in enumerate(GOLDEN_SET, 1):
        print(f"{i:2}. [{it['kind']:6}] {it['question']}")
