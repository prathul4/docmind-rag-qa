"""Shared configuration: loads secrets from .env and defines model/path constants."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "faiss_index"

# Gemini's embedding model
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Gemini's fast/cheap chat model for answer generation
# "lite" variants have a much more generous free-tier request quota than the
# flagship models (which cap out around 20 requests/day) -- important since
# RAGAS evaluation alone makes several LLM calls per question per metric.
CHAT_MODEL = "gemini-flash-lite-latest"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
TOP_K = 4
# Empirically calibrated for gemini-embedding-001 on this document: unrelated
# questions ("capital of France") scored ~0.31-0.33, genuinely relevant chunks
# scored 0.38-0.61. 0.35 sits just above the irrelevant band. This is NOT a
# universal constant -- it must be re-tuned per embedding model / corpus.
RELEVANCE_THRESHOLD = 0.35
