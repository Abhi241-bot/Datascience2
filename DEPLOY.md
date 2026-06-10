# Deploying to Hugging Face Spaces

The demo runs on the **free CPU tier**. The Text-to-SQL tool uses the Groq fallback
there (no GPU); to serve the actual QLoRA adapter, use a GPU/ZeroGPU Space (Option B).

## Prerequisites
- A free Hugging Face account → https://huggingface.co/join
- A free Groq API key → https://console.groq.com/keys

## Option A — Free CPU Space (recommended for the demo)

1. **Create the Space**: huggingface.co → *New* → *Space*. Choose **SDK: Gradio**,
   **Hardware: CPU basic (free)**. Name it e.g. `multi-agent-analyst`.

2. **Add the Spaces config** to the very top of `README.md` (HF reads this YAML
   frontmatter; it's harmless on GitHub):

   ```yaml
   ---
   title: Multi-Agent Financial Analyst
   emoji: 🔎
   colorFrom: indigo
   colorTo: blue
   sdk: gradio
   sdk_version: 5.9.1
   app_file: app.py
   python_version: "3.11"
   pinned: false
   ---
   ```

3. **Set the secret**: Space → *Settings* → *Variables and secrets* → add
   `GROQ_API_KEY` = your key. (Optional: `ORCHESTRATOR_MODEL`, `TAVILY_API_KEY`,
   `LANGCHAIN_TRACING_V2`/`LANGCHAIN_API_KEY` for LangSmith traces.)

4. **Push the repo** to the Space remote:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/multi-agent-analyst
   git push space main
   ```

   The Space installs `requirements.txt`, runs `app.py`, and on first launch
   `bootstrap.ensure_data()` rebuilds the SQLite DB from `data/db/financials.sql`
   and indexes `data/corpus/*.md` into Chroma. First load downloads the ONNX
   embedder (~80 MB).

5. Open the Space URL. Put it in the README's "Live demo" link.

## Option B — Serve the fine-tuned adapter (GPU)

1. Use a **GPU** or **ZeroGPU** Space (Settings → Hardware).
2. Add the heavy deps: append `requirements-finetune.txt` to `requirements.txt`
   (torch / transformers / peft / bitsandbytes).
3. Set the secret `SQL_ADAPTER_REPO` = your pushed adapter repo (from
   `finetune/train_qlora.ipynb`). `text_to_sql` will load the QLoRA model instead
   of the Groq fallback; the report's tool provenance shows `finetuned-adapter`.

## Notes
- `GROQ_API_KEY` is read from the environment via `src/config.py` (`.env` locally,
  Space secret in production). Never commit `.env`.
- The **Evals** tab reads the committed `eval_results/*_scorecard.json`; re-generate
  with `python -m src.eval.ragas_eval` (see `requirements-dev.txt`).
- Free Groq tiers are token-capped per day; if you see 429s, wait or switch
  `ORCHESTRATOR_MODEL`.
