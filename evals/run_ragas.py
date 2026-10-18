"""
Phase 3 — Ragas Evaluation Script
Evaluates AgenticRAG responses using ragas 0.4.x API.
LLM  : Configured via environment variables (OpenAI-compatible endpoint)
Embs : BAAI/bge-small-en-v1.5 via sentence-transformers (no langchain-huggingface needed)
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
# pyrefly: ignore[missing-import]
import requests
from pathlib import Path
# pyrefly: ignore[missing-import]
from dotenv import load_dotenv

# ── Load .env from apps/backend ──────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / "apps" / "backend" / ".env")

# ── Evaluator LLM endpoint config (strictly loaded from .env) ─────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY  = os.getenv("LLM_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "default-model")

if not LLM_BASE_URL or not LLM_API_KEY:
    print("[ERROR] LLM_BASE_URL and LLM_API_KEY must be set in your .env file.")
    sys.exit(1)

# ── Embedding model (same one used by the backend) ───────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# ── RAG API ───────────────────────────────────────────────────────────────────
RAG_API_BASE   = os.getenv("RAG_API_BASE", "http://localhost:8000")
QUERY_ENDPOINT = f"{RAG_API_BASE}/api/v1/query"

# ── Test-set path ─────────────────────────────────────────────────────────────
TEST_SET_PATH  = Path(__file__).parent / "test_set.jsonl"

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "faithfulness":        0.80,
    "answer_relevancy":    0.75,
    "context_precision":   0.70,
    "context_recall":      0.65,
}

# ─────────────────────────────────────────────────────────────────────────────
# Minimal sentence-transformers embeddings wrapper (no langchain-huggingface)
# ─────────────────────────────────────────────────────────────────────────────

# pyrefly: ignore[missing-import]
from langchain_core.embeddings import Embeddings
# pyrefly: ignore[missing-import]
from sentence_transformers import SentenceTransformer

class STEmbeddings(Embeddings):
    """Wraps sentence-transformers directly — avoids langchain-huggingface."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Ragas imports
# ─────────────────────────────────────────────────────────────────────────────
# pyrefly: ignore[missing-import]
from ragas import evaluate, EvaluationDataset
# pyrefly: ignore[missing-import]
from ragas.run_config import RunConfig
# pyrefly: ignore[missing-import]
from ragas.metrics import (


    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
# pyrefly: ignore[missing-import]
from ragas.llms import LangchainLLMWrapper
# pyrefly: ignore[missing-import]
from ragas.embeddings import LangchainEmbeddingsWrapper
# pyrefly: ignore[missing-import]
from langchain_openai import ChatOpenAI


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_test_set(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ERROR] Test set not found: {path}")
        sys.exit(1)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"[INFO] Loaded {len(items)} test items from {path.name}")
    return items


def query_rag(question: str) -> tuple[str, list[str]]:
    """
    POST to /api/v1/query and return (answer, [context_chunk, ...]).
    Adjust the response parsing to match your actual API response shape.
    """
    payload = {"question": question}
    try:
        resp = requests.post(QUERY_ENDPOINT, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        # ── Adapt these keys to your actual API response ──────────────────
        answer   = data.get("answer", data.get("response", ""))
        # contexts: list of strings (chunk text)
        contexts = data.get("contexts", data.get("source_chunks", []))
        if not contexts and data.get("sources_used"):
            contexts = data.get("sources_used", [])
        if isinstance(contexts, list) and contexts and isinstance(contexts[0], dict):
            # If chunks are dicts with a "text" key
            contexts = [c.get("text", c.get("content", str(c))) for c in contexts]
        # ─────────────────────────────────────────────────────────────────
        return answer, contexts

    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot reach RAG API at {QUERY_ENDPOINT}. Is the server running?")
        sys.exit(1)
    except Exception as e:
        safe_q = question[:60].encode("ascii", "replace").decode("ascii")
        print(f"[WARN] Query failed for '{safe_q}': {e}")
        return "", []


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 3 — Ragas Evaluation")
    print("=" * 60)

    # 1. Load test set
    test_items = load_test_set(TEST_SET_PATH)

    # 2. Query RAG for each item concurrently
    print("\n[INFO] Querying RAG API with parallel workers...")
    samples_by_index = {}
    completed_count = 0
    total_items = len(test_items)

    def _fetch_item(index: int, item: dict):
        q = item["question"]
        gt = item.get("ground_truth_answer", item.get("answer", ""))
        ans, ctx = query_rag(q)
        return index, q, gt, ans, ctx

    # 2 concurrent workers so the single-process Uvicorn backend does not bottleneck and timeout
    max_query_workers = int(os.getenv("RAG_QUERY_WORKERS", "2"))
    with ThreadPoolExecutor(max_workers=max_query_workers) as executor:
        futures = {executor.submit(_fetch_item, i, item): i for i, item in enumerate(test_items, 1)}
        for future in as_completed(futures):
            idx, question, ground_truth, answer, contexts = future.result()
            completed_count += 1
            safe_q = question[:60].encode("ascii", "replace").decode("ascii")
            if not answer:
                print(f"  [{completed_count}/{total_items}] [SKIP] Empty answer for '{safe_q}'")
            else:
                print(f"  [{completed_count}/{total_items}] Received response for '{safe_q}'")
                samples_by_index[idx] = {
                    "user_input":         question,
                    "response":           answer,
                    "retrieved_contexts": contexts,
                    "reference":          ground_truth,
                }

    # Maintain original order
    samples = [samples_by_index[i] for i in sorted(samples_by_index.keys())]

    if not samples:
        print("[ERROR] No samples collected. Check that the RAG server is running.")
        sys.exit(1)

    print(f"\n[INFO] Collected {len(samples)} samples for evaluation.")

    # 3. Build ragas dataset
    dataset = EvaluationDataset.from_list(samples)

    # 4. Build evaluator LLM + embeddings
    print("\n[INFO] Initialising evaluator LLM and embeddings...")
    eval_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model       = LLM_MODEL,
            base_url    = LLM_BASE_URL,
            api_key     = LLM_API_KEY,
            temperature = 0,
        )
    )
    eval_emb = LangchainEmbeddingsWrapper(STEmbeddings(EMBEDDING_MODEL_NAME))

    # 5. Define metrics
    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_emb),
        LLMContextPrecisionWithoutReference(llm=eval_llm),
        LLMContextRecall(llm=eval_llm),
    ]

    # 6. Run evaluation with concurrent workers
    eval_workers = int(os.getenv("RAGAS_EVAL_WORKERS", "10"))
    run_config = RunConfig(max_workers=eval_workers, timeout=120, max_retries=3)
    print(f"\n[INFO] Running ragas evaluation (workers={eval_workers}, this will be significantly faster)...")
    results = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)

    # 7. Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    scores = results.to_pandas().mean(numeric_only=True).to_dict()

    metric_map = {
        "faithfulness":                        "faithfulness",
        "answer_relevancy":                    "answer_relevancy",
        "llm_context_precision_without_reference": "context_precision",
        "context_recall":                      "context_recall",
    }

    all_pass = True
    for raw_key, label in metric_map.items():
        score = scores.get(raw_key, None)
        threshold = THRESHOLDS.get(label, 0.0)
        if score is None:
            print(f"  {label:<30} N/A")
            continue
        status = "✅ PASS" if score >= threshold else "❌ FAIL"
        if score < threshold:
            all_pass = False
        print(f"  {label:<30} {score:.3f}  (threshold {threshold})  {status}")

    print("=" * 60)
    print("OVERALL:", "✅ ALL PASS" if all_pass else "❌ SOME METRICS BELOW THRESHOLD")
    print("=" * 60)

    # 8. Save detailed results
    out_path = Path(__file__).parent / "ragas_results.json"
    results_df = results.to_pandas()
    results_df.to_json(out_path, orient="records", indent=2)
    print(f"\n[INFO] Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()