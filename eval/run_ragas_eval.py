"""
Automated evaluation harness (RAGAS): runs the RAG pipeline against a
hand-written gold Q&A set and scores it on four axes:

- faithfulness       : does the answer only contain claims supported by the
                        retrieved context? (catches hallucination)
- answer_relevancy   : does the answer actually address the question asked?
- context_precision  : of the chunks retrieved, how many were relevant?
- context_recall     : of the info needed to answer, how much did retrieval find?

This is what turns "the demo looked fine" into a number you can defend and
compare across configurations (e.g. chunk size 300 vs 500 -- see
benchmark_chunk_sizes.py).

Usage:
    python eval/run_ragas_eval.py --tag 500tok
    python eval/run_ragas_eval.py --tag 300tok --limit 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
from rag_chain import DocMindRAG

GOLD_QA_PATH = Path(__file__).resolve().parent / "gold_qa.json"


def build_dataset(tag: str, limit: int | None) -> EvaluationDataset:
    gold = json.loads(GOLD_QA_PATH.read_text(encoding="utf-8"))
    if limit:
        gold = gold[:limit]

    rag = DocMindRAG(tag=tag)
    samples = []
    # Gemini's free tier caps gemini-flash-lite-latest at 15 chat requests per
    # minute. Each question here costs 1 chat call (embeddings are a separate
    # model/quota). Pacing ourselves at ~10/min (one call every 6s) stays
    # safely under that instead of firing all of them immediately and relying
    # on retries to sort out the resulting pileup of 429s.
    for i, item in enumerate(gold, start=1):
        print(f"[{i}/{len(gold)}] {item['question']}")
        result = rag.answer(item["question"])
        contexts = [s["text"] for s in result["sources"]] or [
            "(no context retrieved above relevance threshold)"
        ]
        samples.append(SingleTurnSample(
            user_input=item["question"],
            retrieved_contexts=contexts,
            response=result["answer"],
            reference=item["ground_truth"],
        ))
        if i < len(gold):
            time.sleep(6)
    return EvaluationDataset(samples=samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="500tok")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="CSV path for per-question results")
    args = parser.parse_args()

    dataset = build_dataset(args.tag, args.limit)

    eval_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=config.CHAT_MODEL, google_api_key=config.GOOGLE_API_KEY,
                                temperature=0)
    )
    eval_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL,
                                      google_api_key=config.GOOGLE_API_KEY)
    )

    # The Gemini free tier caps us at 15 requests/minute. RAGAS's default
    # max_workers=16 fires that many requests almost simultaneously, which
    # blows through the quota instantly and can exhaust the retry budget
    # before the per-minute window resets. max_workers=2 keeps us under the
    # cap; higher max_retries/max_wait give it enough room to ride out
    # whatever 429s still happen.
    run_config = RunConfig(max_workers=1, max_retries=20, max_wait=90, timeout=300)

    print(f"\nRunning RAGAS evaluation on index '{args.tag}' ({len(dataset)} questions)...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config,
    )

    df = result.to_pandas()
    out_path = args.out or str(Path(__file__).resolve().parent / f"results_{args.tag}.csv")
    df.to_csv(out_path, index=False)

    print("\n=== Aggregate scores ===")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in df.columns:
            print(f"  {metric:20s}: {df[metric].mean():.3f}")
    print(f"\nPer-question results saved to {out_path}")


if __name__ == "__main__":
    main()
