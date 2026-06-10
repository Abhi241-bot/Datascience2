# Phase 1 — QLoRA Fine-tune of the Text-to-SQL Tool

Fine-tunes a 4-bit base model on **Spider** so it reliably turns *(schema + question) → SQL*. The exported LoRA adapter becomes a **live agent tool** in [`src/tools/text_to_sql.py`](../src/tools/text_to_sql.py) (Phase 2) — not a standalone demo.

## Files

| File | What it does | Where it runs |
|---|---|---|
| `prepare_spider.py` | Downloads Spider, renders `(schema + question → SQL)` into instruction JSONL | anywhere (CPU) |
| `train_qlora.ipynb` | QLoRA fine-tune: 4-bit base, LoRA r=16 / α=32 / all-linear, LR 2e-4, grad checkpointing | **Colab T4** |
| `evaluate_sql.py` | **Execution accuracy** on held-out Spider dev, base vs fine-tuned | Colab/GPU |
| `adapter/` | Exported LoRA adapter (or push to HF Hub and set `SQL_ADAPTER_REPO`) | — |

## Workflow

```bash
# 1. Build the dataset (local or Colab)
python finetune/prepare_spider.py --out finetune/data

# 2. Train on Colab T4 — open train_qlora.ipynb, set T4 runtime, Run all.
#    Saves adapter/ and optionally pushes to HF Hub.

# 3. Download Spider's database/ folder, then score before vs after:
python finetune/evaluate_sql.py --spider-db-dir spider/database \
    --adapter finetune/adapter --limit 200
```

## Why execution accuracy (not BLEU)

Two textually different SQL queries can return identical results. We execute both
the gold and predicted SQL against the actual SQLite db and compare result sets
(order-insensitive). That is the metric recruiters and the brief care about.

## ✅ Acceptance criteria

> Measurable exec-accuracy improvement after fine-tuning, documented with before/after numbers.

Fill in after running `evaluate_sql.py` (writes `eval_report.json`):

| Model | Exec accuracy | Runnable rate |
|---|---|---|
| Base (`Llama-3.1-8B-Instruct`, 4-bit) | _TBD_ % | _TBD_ % |
| **+ QLoRA (Spider)** | **_TBD_ %** | **_TBD_ %** |
| **Δ** | **_TBD_ pp** | — |

> ⚠️ Training requires a GPU (free Colab T4). It was **not** run in this environment
> (no GPU); the scripts/notebook are complete and ready to run. Paste the real
> numbers here and in the root README after the Colab run.
