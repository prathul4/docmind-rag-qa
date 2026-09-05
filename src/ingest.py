"""
Ingestion pipeline: PDF -> text -> chunks -> embeddings -> FAISS index on disk.

This is Step 1 of RAG ("index your documents once, ahead of time"). Run this
whenever the source PDF changes. Querying (rag_chain.py) then just loads the
pre-built index instead of re-embedding everything on every question.

Usage:
    python src/ingest.py
    python src/ingest.py --chunk-size 300 --chunk-overlap 50 --tag 300tok
"""

import argparse
import time

import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config

# We don't have a Gemini tokenizer handy client-side, so we approximate token
# counts with OpenAI's cl100k_base encoding. It's not exact for Gemini, but it's
# a consistent, reproducible yardstick for comparing chunk-size configurations
# against each other -- which is what "300 vs 500 tokens" on the resume means.
_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def load_and_split(pdf_path, chunk_size: int, chunk_overlap: int):
    print(f"Loading {pdf_path} ...")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()  # one Document per PDF page, with page-number metadata
    print(f"  Loaded {len(pages)} page(s).")

    # RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    # then sentences, then words -- only falling back to a hard character cut
    # if a chunk still doesn't fit. This keeps chunks semantically coherent
    # instead of slicing mid-sentence. chunk_overlap repeats a bit of text
    # between consecutive chunks so an answer that straddles a chunk boundary
    # doesn't get orphaned.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"  Split into {len(chunks)} chunk(s) "
          f"(target size={chunk_size} tokens, overlap={chunk_overlap} tokens).")
    return chunks


def build_index(chunks, output_dir):
    if not config.GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Edit docmind/.env and add your Gemini API key."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL, google_api_key=config.GOOGLE_API_KEY
    )

    print(f"Embedding {len(chunks)} chunks and building FAISS index...")
    start = time.time()
    # normalize_L2=True rescales every vector to unit length, both when
    # indexing and when querying. Gemini's embeddings are NOT unit-normed out
    # of the box, but LangChain's default relevance-score formula assumes
    # they are (it maps Euclidean distance -> [0,1] similarity using a
    # sqrt(2)-max-distance assumption that only holds for unit vectors).
    # Without this, relevance scores come out compressed into a narrow,
    # uninformative band regardless of true relevance.
    vectorstore = FAISS.from_documents(chunks, embeddings, normalize_L2=True)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s.")

    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))
    print(f"  Saved index to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=str(config.DATA_DIR / "employee_handbook.pdf"))
    parser.add_argument("--chunk-size", type=int, default=config.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--tag", default=None, help="Suffix for the index folder name")
    args = parser.parse_args()

    tag = args.tag or f"{args.chunk_size}tok"
    output_dir = config.INDEX_DIR / tag

    chunks = load_and_split(args.pdf, args.chunk_size, args.chunk_overlap)
    build_index(chunks, output_dir)


if __name__ == "__main__":
    main()
