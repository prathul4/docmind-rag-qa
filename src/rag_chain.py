"""
Step 2 of RAG: given a question, retrieve relevant chunks from the FAISS
index and ask Gemini to answer using ONLY those chunks.

Key ideas:
- similarity_search_with_relevance_scores gives each retrieved chunk a score
  in [0, 1]. We drop chunks below RELEVANCE_THRESHOLD -- a chunk that is only
  vaguely related to the question is worse than no chunk at all, because it
  invites the model to guess.
- The system prompt explicitly forbids answering from outside knowledge and
  gives the model an explicit escape hatch ("insufficient context") instead
  of letting it hallucinate a plausible-sounding but ungrounded answer.
- Every call is timed and token-counted so we can log cost/latency per query
  (this is what "production-readiness beyond a notebook prototype" means).
"""

import json
import time
from pathlib import Path

import tiktoken
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import config

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


SYSTEM_PROMPT = """You are DocMind, a question-answering assistant that answers \
strictly from the provided context excerpts taken from a document.

Rules:
1. Only use information present in the CONTEXT below. Do not use outside knowledge.
2. If the context does not contain enough information to answer confidently, \
respond with exactly: "Insufficient context to answer this question." \
Do not guess or fill gaps with assumptions.
3. When you do answer, be concise and directly address the question.
4. Do not mention that you were given "context" or "excerpts" in your answer -- \
just answer naturally, as if you know this from the document.

CONTEXT:
{context}
"""


class DocMindRAG:
    def __init__(self, tag: str = "500tok", k: int = config.TOP_K,
                 relevance_threshold: float = config.RELEVANCE_THRESHOLD):
        if not config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set. Edit docmind/.env.")

        index_dir = config.INDEX_DIR / tag
        if not index_dir.exists():
            raise FileNotFoundError(
                f"No index at {index_dir}. Run `python src/ingest.py --tag {tag}` first."
            )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL, google_api_key=config.GOOGLE_API_KEY
        )
        self.vectorstore = FAISS.load_local(
            str(index_dir), self.embeddings, allow_dangerous_deserialization=True
        )
        self.llm = ChatGoogleGenerativeAI(
            model=config.CHAT_MODEL, google_api_key=config.GOOGLE_API_KEY, temperature=0
        )
        self.k = k
        self.relevance_threshold = relevance_threshold

    def retrieve(self, question: str):
        """Return chunks above the relevance threshold, most relevant first."""
        results = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=self.k
        )
        kept = [(doc, score) for doc, score in results if score >= self.relevance_threshold]
        return kept, results  # (filtered, all-for-debugging)

    def answer(self, question: str) -> dict:
        start = time.time()
        kept, all_results = self.retrieve(question)

        if not kept:
            latency_ms = (time.time() - start) * 1000
            result = {
                "question": question,
                "answer": "Insufficient context to answer this question.",
                "sources": [],
                "retrieved_but_below_threshold": [
                    {"page": d.metadata.get("page"), "score": round(float(s), 3),
                     "text": d.page_content[:200]}
                    for d, s in all_results
                ],
                "latency_ms": round(latency_ms, 1),
                "tokens_estimated": count_tokens(question),
            }
            self._log(result)
            return result

        context = "\n\n---\n\n".join(doc.page_content for doc, _ in kept)
        prompt = SYSTEM_PROMPT.format(context=context)

        response = self.llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ])
        latency_ms = (time.time() - start) * 1000

        prompt_tokens = count_tokens(prompt) + count_tokens(question)
        completion_tokens = count_tokens(response.content)

        result = {
            "question": question,
            "answer": response.content,
            "sources": [
                {"page": doc.metadata.get("page"), "score": round(float(score), 3),
                 "text": doc.page_content}
                for doc, score in kept
            ],
            "latency_ms": round(latency_ms, 1),
            "tokens_estimated": prompt_tokens + completion_tokens,
            "prompt_tokens_estimated": prompt_tokens,
            "completion_tokens_estimated": completion_tokens,
        }
        self._log(result)
        return result

    def _log(self, result: dict):
        log_path = config.ROOT_DIR / "query_log.jsonl"
        entry = {k: v for k, v in result.items() if k != "sources"}
        entry["num_sources"] = len(result.get("sources", []))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    rag = DocMindRAG()
    while True:
        q = input("\nAsk DocMind (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit"):
            break
        result = rag.answer(q)
        print(f"\nAnswer: {result['answer']}")
        print(f"\n[{result['latency_ms']}ms, ~{result['tokens_estimated']} tokens, "
              f"{len(result['sources'])} source(s)]")
        for s in result["sources"]:
            print(f"  - page {s['page']}, score {s['score']}: {s['text'][:100]}...")
