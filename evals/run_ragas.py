"""
Phase 3 — Ragas offline eval.
Requires: pip install ragas datasets requests

Usage:
    # From project root, with backend running:
    python evals/run_ragas.py

    # Save results to CSV:
    python evals/run_ragas.py --save
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

BASE_URL = "http://localhost:8000/api/v1"
TEST_SET_PATH = Path("evals/test_set.jsonl")
RESULTS_DIR = Path("evals/results")

THRESHOLDS = {
    "context_precision": 0.70,
    "context_recall": 0.65,
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
}


def check_server() -> None:
    try:
        r = requests.get(f"{BASE_URL.replace('/api/v1', '')}/health", timeout=5)
        r.raise_for_status()
    except Exception:
        print("ERROR: Backend not reachable at", BASE_URL)
        print("Start it with: uvicorn main:app --reload (from apps/backend)")
        sys.exit(1)


def run_query(question: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/query/agentic",
        json={"question": question},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def get_last_state() -> dict:
    resp = requests.get(f"{BASE_URL}/query/agentic/last_state", timeout=10)
    resp.raise_for_status()
    return resp.json()


def build_dataset(test_set_path: Path, skip_unanswerable: bool = True) -> Dataset:
    rows = []
    entries = [json.loads(l) for l in test_set_path.read_text().splitlines() if l.strip()]

    total = len(entries)
    print(f"Running {total} queries against {BASE_URL}...\n")

    for i, item in enumerate(entries, 1):
        # Skip unanswerable — Ragas faithfulness breaks on "not available" answers
        if skip_unanswerable and item.get("difficulty") == "unanswerable":
            print(f"  [{i:02d}/{total}] SKIP (unanswerable): {item['question'][:60]}")
            continue

        print(f"  [{i:02d}/{total}] {item['question'][:70]}")
        start = time.time()

        try:
            result = run_query(item["question"])
            elapsed = time.time() - start

            answer = result.get("answer", "")
            sources = result.get("sources_used", [])

            # sources_used is a list of source strings from the agent
            # Use them as context strings for Ragas
            contexts = sources if sources else [""]

            rows.append({
                "question": item["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth_answer"],
            })
            print(f"         → {len(contexts)} sources, {elapsed:.1f}s")

        except Exception as e:
            print(f"         ✗ Query failed: {e}")
            # Add a placeholder so we don't silently skip
            rows.append({
                "question": item["question"],
                "answer": "",
                "contexts": [""],
                "ground_truth": item["ground_truth_answer"],
            })

        # Small sleep to avoid hammering the LLM backend
        time.sleep(1)

    print(f"\nBuilt dataset: {len(rows)} rows\n")
    return Dataset.from_list(rows)


def print_results(scores: dict) -> bool:
    """Print scores, compare to thresholds, return True if all pass."""
    all_pass = True
    print("=" * 50)
    print("RAGAS RESULTS")
    print("=" * 50)
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        status = "✓ PASS" if score >= threshold else "✗ FAIL"
        if score < threshold:
            all_pass = False
        print(f"  {metric:<22} {score:.4f}  (min {threshold})  {status}")
    print("=" * 50)
    return all_pass


def save_results(results, scores: dict) -> None:
    import datetime
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()
    csv_path = RESULTS_DIR / f"ragas_{date_str}.csv"
    results.to_pandas().to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save results to CSV")
    parser.add_argument(
        "--include-unanswerable",
        action="store_true",
        help="Include unanswerable questions (not recommended for Ragas)",
    )
    args = parser.parse_args()

    check_server()

    if not TEST_SET_PATH.exists():
        print(f"ERROR: Test set not found at {TEST_SET_PATH}")
        print("Run the phase2 meta-prompt first to generate evals/test_set.jsonl")
        sys.exit(1)

    dataset = build_dataset(
        TEST_SET_PATH,
        skip_unanswerable=not args.include_unanswerable,
    )

    print("Running Ragas evaluation (this calls your LLM for scoring)...\n")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {
        "context_precision": results["context_precision"],
        "context_recall": results["context_recall"],
        "faithfulness": results["faithfulness"],
        "answer_relevancy": results["answer_relevancy"],
    }

    all_pass = print_results(scores)

    if args.save:
        save_results(results, scores)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()