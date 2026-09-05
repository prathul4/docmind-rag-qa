"""
Benchmarks retrieval/answer quality across chunk-size configurations
(300 vs 500 tokens) using the same RAGAS metrics, so the comparison is
apples-to-apples on the same gold Q&A set.

Usage:
    python eval/benchmark_chunk_sizes.py
    python eval/benchmark_chunk_sizes.py --limit 10   # faster, fewer questions
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

TAGS = ["300tok", "500tok"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = {}
    for i, tag in enumerate(TAGS):
        if i > 0:
            print("\nCooling down 60s before next config so the free-tier "
                  "per-minute quota window resets...")
            time.sleep(60)
        cmd = [sys.executable, "eval/run_ragas_eval.py", "--tag", tag]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print(f"\n{'='*60}\nEvaluating {tag}\n{'='*60}")
        subprocess.run(cmd, check=True)

        csv_path = Path(__file__).resolve().parent / f"results_{tag}.csv"
        df = pd.read_csv(csv_path)
        summary[tag] = {
            m: df[m].mean() for m in
            ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            if m in df.columns
        }

    print(f"\n{'='*60}\nChunk-size comparison\n{'='*60}")
    comparison = pd.DataFrame(summary).T
    comparison.loc["delta (500tok - 300tok)"] = comparison.loc["500tok"] - comparison.loc["300tok"]
    print(comparison.round(3).to_string())

    out_path = Path(__file__).resolve().parent / "chunk_size_comparison.csv"
    comparison.round(3).to_csv(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
