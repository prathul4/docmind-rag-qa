# DocMind — RAG Document Q&A Assistant

Answers questions about a PDF using Retrieval-Augmented Generation: the
document is chunked, embedded, and indexed with FAISS; questions retrieve
the most relevant chunks, which are fed to Gemini as grounding context.

## Setup

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # then edit .env and add your Gemini API key
```

Get a free Gemini API key at https://aistudio.google.com/apikey

## Usage

```
# 1. Build the vector index from the PDF (run once, or whenever the PDF changes)
venv\Scripts\python src\ingest.py

# 2. Launch the interactive UI
venv\Scripts\python -m streamlit run src\app.py

# 3. Evaluate answer/retrieval quality against the gold Q&A set
venv\Scripts\python eval\run_ragas_eval.py --tag 500tok

# 4. Compare chunk-size configurations (300 vs 500 tokens)
venv\Scripts\python eval\benchmark_chunk_sizes.py
```

## Project layout

- `src/ingest.py` — PDF -> chunks -> embeddings -> FAISS index
- `src/rag_chain.py` — retrieval + grounded generation + logging
- `src/app.py` — Streamlit UI
- `eval/gold_qa.json` — hand-written Q&A pairs with ground-truth answers
- `eval/run_ragas_eval.py` — RAGAS scoring (faithfulness, relevancy, precision/recall)
- `eval/benchmark_chunk_sizes.py` — 300 vs 500 token comparison
