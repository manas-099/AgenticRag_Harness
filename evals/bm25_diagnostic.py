"""
BM25 Contribution Diagnostic
Runs the Ragas eval twice — once with hybrid retrieval, once with dense-only —
and prints the context_recall delta to determine BM25's real contribution.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
# pyrefly: ignore[missing-import]
import requests
# pyrefly: ignore[missing-import]
from dotenv import load_dotenv

# pyrefly: ignore[missing-import]
from langchain_openai import ChatOpenAI
# pyrefly: ignore[missing-import]
from ragas.llms import LangchainLLMWrapper
# pyrefly: ignore[missing-import]
from ragas.dataset_schema import SingleTurnSample
# pyrefly: ignore[missing-import]
from ragas.metrics import ContextRecall

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "apps" / "backend"

# Load backend .env for evaluator LLM configuration
load_dotenv(BACKEND_DIR / ".env")

BASE_URL = os.getenv("EVAL_BASE_URL", "http://localhost:8000/api/v1")
TEST_SET_PATH = Path(os.getenv("TEST_SET_PATH", str(EVALS_DIR / "test_set.jsonl")))
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "default-model")


def load_test_set(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_agentic_query(question: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/query/agentic",
        json={"question": question},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def collect_samples(test_set: list[dict], label: str) -> list[dict]:
    query_workers = int(os.getenv("RAG_QUERY_WORKERS", "6"))
    print(f"Collecting responses using {query_workers} parallel query workers...")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    samples_by_idx: dict[int, dict] = {}
    completed = 0
    total = len(test_set)

    def _fetch(idx: int, item: dict):
        q = item["question"]
        gt = item.get("ground_truth_answer", item.get("ground_truth", ""))
        try:
            result = run_agentic_query(q)
            retrieved_contexts = result.get("retrieved_contexts", [])
            if not retrieved_contexts:
                retrieved_contexts = [str(s) for s in result.get("sources_used", [])]

            # Normalize contexts to clean non-empty strings
            retrieved_contexts = [
                str(c.get("text") or c.get("content") or c) if isinstance(c, dict) else str(c).strip()
                for c in retrieved_contexts
                if str(c).strip()
            ]

            return idx, {
                "user_input": q,
                "response": result.get("answer", ""),
                "retrieved_contexts": retrieved_contexts,
                "reference": gt,
            }, None
        except Exception as e:
            return idx, None, e

    with ThreadPoolExecutor(max_workers=query_workers) as executor:
        futures = {executor.submit(_fetch, i, item): i for i, item in enumerate(test_set)}
        for future in as_completed(futures):
            idx, sample, err = future.result()
            completed += 1
            item = test_set[idx]
            q_preview = item["question"][:55].encode("ascii", "replace").decode("ascii")
            if err:
                print(f"  [{label}] [{completed}/{total}] WARN: query failed ({err}) - '{q_preview}...'")
            else:
                ctx_len = len(sample["retrieved_contexts"])
                print(f"  [{label}] [{completed}/{total}] OK ({ctx_len} ctx) - '{q_preview}...'")
                samples_by_idx[idx] = sample

    return [samples_by_idx[i] for i in sorted(samples_by_idx.keys())]


async def compute_context_recall(samples: list[dict]) -> float:
    if not LLM_BASE_URL or not LLM_API_KEY:
        print("[ERROR] LLM_BASE_URL and LLM_API_KEY must be set in your .env file.")
        sys.exit(1)

    chat_model = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )
    eval_llm = LangchainLLMWrapper(chat_model)
    recall_metric = ContextRecall(llm=eval_llm)

    semaphore = asyncio.Semaphore(int(os.getenv("RAGAS_EVAL_WORKERS", "10")))
    scores: list[float] = []

    async def _eval_one(sample_dict: dict):
        ctx = sample_dict["retrieved_contexts"]
        if not ctx:
            return
        sample = SingleTurnSample(
            user_input=sample_dict["user_input"],
            response=sample_dict["response"],
            retrieved_contexts=ctx,
            reference=sample_dict["reference"],
        )
        async with semaphore:
            try:
                score = await recall_metric.single_turn_ascore(sample)
                if score is not None and score == score:
                    scores.append(float(score))
            except Exception as exc:
                safe_q = sample_dict["user_input"][:50].encode("ascii", "replace").decode("ascii")
                print(f"  [WARN] context_recall failed for '{safe_q}': {exc}")

    tasks = [_eval_one(s) for s in samples]
    await asyncio.gather(*tasks)

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid"

    print(f"\n{'='*60}")
    print(f"  BM25 DIAGNOSTIC — mode: {mode.upper()}")
    print(f"{'='*60}\n")

    if not TEST_SET_PATH.exists():
        print(f"[ERROR] Test set not found at: {TEST_SET_PATH}")
        sys.exit(1)

    test_set = load_test_set(TEST_SET_PATH)
    print(f"Loaded {len(test_set)} test cases from {TEST_SET_PATH}\n")

    samples = collect_samples(test_set, label=mode)

    print(f"\nRunning Ragas context_recall evaluation ({mode})...")
    score = asyncio.run(compute_context_recall(samples))

    print(f"\n{'='*60}")
    print(f"  context_recall [{mode.upper()}]: {score:.4f}")
    print(f"{'='*60}\n")

    # Save result
    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"bm25_diagnostic_{mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "context_recall": score, "samples": len(samples)}, f, indent=2)
    print(f"Result saved to {out_path}")


if __name__ == "__main__":
    main()
