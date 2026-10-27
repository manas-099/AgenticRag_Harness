"""
Reads both diagnostic result files and prints the delta + conclusion.
Run this after both bm25_diagnostic.py runs complete.
"""

import json
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"

HYBRID_PATH = RESULTS_DIR / "bm25_diagnostic_hybrid.json"
DENSE_PATH = RESULTS_DIR / "bm25_diagnostic_dense_only.json"


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    if not HYBRID_PATH.exists() or not DENSE_PATH.exists():
        print("ERROR: both result files must exist before comparing.")
        print(f"  Expected: {HYBRID_PATH}")
        print(f"  Expected: {DENSE_PATH}")
        return

    hybrid = load(HYBRID_PATH)
    dense = load(DENSE_PATH)

    recall_hybrid = hybrid["context_recall"]
    recall_dense = dense["context_recall"]
    delta = recall_hybrid - recall_dense

    print(f"\n{'='*60}")
    print(f"  BM25 DIAGNOSTIC — COMPARISON")
    print(f"{'='*60}")
    print(f"  context_recall  HYBRID (dense+BM25+rerank) : {recall_hybrid:.4f}")
    print(f"  context_recall  DENSE ONLY (dense+rerank)  : {recall_dense:.4f}")
    print(f"  delta (hybrid - dense)                     : {delta:+.4f}")
    print(f"{'='*60}")

    if delta < 0.03:
        conclusion = (
            "BM25 contributes almost nothing after rerank.\n"
            "  Options:\n"
            "    (a) Remove BM25 entirely — simplifies the pipeline\n"
            "    (b) Persist BM25 index to disk if you want to keep it\n"
            "  Either way, the current in-memory cold-start is a liability."
        )
    elif delta < 0.08:
        conclusion = (
            "BM25 has marginal contribution.\n"
            "  Weigh: is the complexity + cold-start risk worth ~{:.3f} recall gain?\n"
            "  If corpus is stable and restarts are rare, persisting may be worth it.\n"
            "  If restarts are frequent, lean toward removing BM25.".format(delta)
        )
    else:
        conclusion = (
            "BM25 meaningfully fills retrieval gaps (+{:.3f} recall).\n"
            "  Action required: fix the cold-start problem.\n"
            "  Options:\n"
            "    (a) Rebuild BM25 index from Qdrant payloads on every server startup\n"
            "    (b) Persist BM25 index to disk (pickle) and reload on startup".format(delta)
        )

    print(f"\n  CONCLUSION: {conclusion}\n")


if __name__ == "__main__":
    main()
