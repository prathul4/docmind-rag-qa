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


def _cache_path(tag: str) -> Path:
    return Path(__file__).resolve().parent / f"dataset_cache_{tag}.json"


def build_dataset(tag: str, limit: int | None, use_cache: bool = True) -> EvaluationDataset:
    cache_path = _cache_path(tag)

    # Generating answers costs 1 Gemini call per question. RAGAS scoring
    # (the step that actually tends to hit rate limits) is a separate phase
    # that we often need to re-run with different retry settings -- caching
    # the generated answers means retrying the scoring step doesn't also
    # burn quota re-generating identical answers every time.
    if use_cache and cache_path.exists():
        print(f"Loading cached answers from {cache_path}")
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if limit:
            raw = raw[:limit]
        return EvaluationDataset(samples=[SingleTurnSample(**s) for s in raw])

    gold = json.loads(GOLD_QA_PATH.read_text(encoding="utf-8"))
    if limit:
        gold = gold[:limit]

    rag = DocMindRAG(tag=tag)
    raw_samples = []
    # Gemini's free tier caps gemini-flash-lite-latest at 15 chat requests per
    # minute. Each question here costs 1 chat call (embeddings are a separate
    # model/quota). Pacing ourselves at ~7/min (one call every 9s) stays
    # safely under that instead of firing all of them immediately and relying
    # on retries to sort out the resulting pileup of 429s.
    for i, item in enumerate(gold, start=1):
        print(f"[{i}/{len(gold)}] {item['question']}")
        result = rag.answer(item["question"])
        contexts = [s["text"] for s in result["sources"]] or [
            "(no context retrieved above relevance threshold)"
        ]
        raw_samples.append({
            "user_input": item["question"],
            "retrieved_contexts": contexts,
            "response": result["answer"],
            "reference": item["ground_truth"],
        })
        if i < len(gold):
            time.sleep(9)

    cache_path.write_text(json.dumps(raw_samples, indent=2), encoding="utf-8")
    print(f"Cached generated answers to {cache_path}")
    return EvaluationDataset(samples=[SingleTurnSample(**s) for s in raw_samples])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="500tok")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="CSV path for per-question results")
    parser.add_argument("--rebuild", action="store_true",
                         help="Ignore cached answers and regenerate (costs quota)")
    args = parser.parse_args()

    dataset = build_dataset(args.tag, args.limit, use_cache=not args.rebuild)

    eval_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=config.CHAT_MODEL, google_api_key=config.GOOGLE_API_KEY,
                                temperature=0)
    )
    eval_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL,
                                      google_api_key=config.GOOGLE_API_KEY)
    )

    # The Gemini free tier caps us at both 15 requests/minute AND ~500
    # requests/day. Each retry after a 429 is itself a new request that
    # counts against the DAILY cap too, so max_retries is a real tradeoff:
    # too high and one stuck call can cascade through the day's budget; too
    # low (we tried max_retries=4) and transient per-minute 429s don't get
    # enough attempts to survive past the next per-minute reset, silently
    # producing NaN scores instead of an error. max_retries=8 with a 45s
    # wait gives ~2 per-minute reset windows of room per call.
    run_config = RunConfig(max_workers=1, max_retries=8, max_wait=45, timeout=180)

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
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    any_missing = False
    for metric in metric_names:
        if metric in df.columns:
            n_missing = df[metric].isna().sum()
            if n_missing:
                any_missing = True
            print(f"  {metric:20s}: {df[metric].mean():.3f}"
                  + (f"   [WARNING: {n_missing}/{len(df)} scores missing/failed]" if n_missing else ""))
    if any_missing:
        print("\nSome metric calls failed even after retries (rate-limit exhaustion). "
              "The averages above are computed only over the successful rows -- "
              "treat them as unreliable and re-run before reporting them anywhere.")
    print(f"\nPer-question results saved to {out_path}")


if __name__ == "__main__":
    main()
