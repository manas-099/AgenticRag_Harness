"""
Phase 3 — Ragas Evaluation Script
Evaluates AgenticRAG responses using ragas 0.4.x collections API.
LLM  : OpenAI-compatible endpoint configured via .env  (AsyncOpenAI)
Embs : BAAI/bge-small-en-v1.5 via ragas.embeddings.HuggingFaceEmbeddings
"""

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# pyrefly: ignore[missing-import]
import requests
# pyrefly: ignore[missing-import]
from dotenv import load_dotenv

# ── Load .env from apps/backend ───────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / "apps" / "backend" / ".env")

# ── Evaluator LLM config ──────────────────────────────────────────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY  = os.getenv("LLM_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "default-model")

if not LLM_BASE_URL or not LLM_API_KEY:
    print("[ERROR] LLM_BASE_URL and LLM_API_KEY must be set in your .env file.")
    sys.exit(1)

# ── Embedding model (points to same LM Studio Nomic embedder as ingestion) ───
EMBEDDING_NOMIC_BASE_URL = os.getenv("EMBEDDING_NOMIC_BASE_URL", "http://127.0.0.1:1234")
EMBEDDING_NOMIC_MODEL    = os.getenv("EMBEDDING_NOMIC_MODEL", "text-embedding-nomic-embed-text-v1.5@q8_0")

# ── RAG API ───────────────────────────────────────────────────────────────────
RAG_API_BASE   = os.getenv("RAG_API_BASE", "http://localhost:8000")
QUERY_ENDPOINT = f"{RAG_API_BASE}/api/v1/query"

# ── Test-set path ─────────────────────────────────────────────────────────────
TEST_SET_PATH = Path(__file__).parent / "test_set.jsonl"

# ── Pass/fail thresholds ──────────────────────────────────────────────────────
THRESHOLDS = {
    "faithfulness":      0.80,
    "answer_relevancy":  0.75,
    "context_precision": 0.70,
    "context_recall":    0.65,
}

# ── Minimum contexts required for retrieval metrics ───────────────────────────
#   Samples with empty retrieved_contexts are skipped for those metrics but
#   still scored for answer_relevancy (which needs no contexts).
MIN_CONTEXTS_FOR_RETRIEVAL = 1


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
    POST to /api/v1/query → (answer, [context_chunk, ...]).

    Tries several common response shapes your RAG server might return.
    Prints the raw response keys on the first call so you can verify.
    """
    payload = {"question": question}
    try:
        resp = requests.post(QUERY_ENDPOINT, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        answer = (
            data.get("answer")
            or data.get("response")
            or data.get("result")
            or ""
        )

        # ── Try every key that commonly holds context chunks ──────────────────
        contexts: list = []
        for key in ("retrieved_contexts", "contexts", "source_chunks",
                    "sources_used", "context", "chunks", "documents"):
            val = data.get(key)
            if val:
                contexts = val
                break

        # Normalise: if list of dicts, extract text field
        if isinstance(contexts, list) and contexts and isinstance(contexts[0], dict):
            contexts = [
                c.get("text") or c.get("content") or c.get("page_content") or str(c)
                for c in contexts
            ]

        # Keep only non-empty strings
        contexts = [str(c).strip() for c in contexts if str(c).strip()]

        return answer, contexts

    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot reach RAG API at {QUERY_ENDPOINT}. Is the server running?")
        sys.exit(1)
    except Exception as e:
        safe_q = question[:60].encode("ascii", "replace").decode("ascii")
        print(f"[WARN] Query failed for '{safe_q}': {e}")
        return "", []


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
#
# Per-metric accepted fields (ragas 0.4.x collections API):
#   Faithfulness              : user_input, response, retrieved_contexts
#   AnswerRelevancy           : user_input, response
#   ContextPrecisionWithout.. : user_input, response, retrieved_contexts
#   ContextRecall             : user_input, retrieved_contexts, reference
# ─────────────────────────────────────────────────────────────────────────────
async def _score_all(
    samples:  list[dict],
    eval_llm,
    eval_emb,
) -> dict[str, float]:
    # pyrefly: ignore[missing-import]
    from ragas.dataset_schema import SingleTurnSample
    # pyrefly: ignore[missing-import]
    from ragas.metrics import (
        AnswerRelevancy,
        ContextRecall,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
    )

    faithfulness_metric = Faithfulness(llm=eval_llm)
    relevancy_metric    = AnswerRelevancy(llm=eval_llm, embeddings=eval_emb)
    precision_metric    = LLMContextPrecisionWithoutReference(llm=eval_llm)
    recall_metric       = ContextRecall(llm=eval_llm)

    eval_workers = int(os.getenv("RAGAS_EVAL_WORKERS", "10"))
    semaphore    = asyncio.Semaphore(eval_workers)

    totals: dict[str, list[float]] = {
        "faithfulness":      [],
        "answer_relevancy":  [],
        "context_precision": [],
        "context_recall":    [],
    }

    context_missing_count = 0
    scored_count = 0
    total_samples = len(samples)

    try:
        # pyrefly: ignore[missing-import]
        from tqdm.asyncio import tqdm_asyncio
        has_tqdm = True
    except ImportError:
        try:
            # pyrefly: ignore[missing-import]
            from tqdm import tqdm
            has_tqdm = False
        except ImportError:
            has_tqdm = False

    async def _score_sample(sample_dict: dict):
        nonlocal context_missing_count, scored_count
        q   = sample_dict["user_input"]
        r   = sample_dict["response"]
        ctx = sample_dict["retrieved_contexts"]   # may be []
        ref = sample_dict.get("reference", "")
        safe = q[:50].encode("ascii", "replace").decode("ascii")
        has_ctx = bool(ctx and len(ctx) >= MIN_CONTEXTS_FOR_RETRIEVAL)

        sample = SingleTurnSample(
            user_input=q,
            response=r,
            retrieved_contexts=ctx if has_ctx else [],
            reference=ref,
        )

        async with semaphore:
            async def _calc_faithfulness():
                if not has_ctx:
                    return
                try:
                    score = await faithfulness_metric.single_turn_ascore(sample)
                    if score is not None and score == score:
                        totals["faithfulness"].append(float(score))
                except Exception as exc:
                    print(f"  [WARN] faithfulness failed for '{safe}': {exc}")

            async def _calc_relevancy():
                try:
                    score = await relevancy_metric.single_turn_ascore(sample)
                    if score is not None and score == score:
                        totals["answer_relevancy"].append(float(score))
                except Exception as exc:
                    print(f"  [WARN] answer_relevancy failed for '{safe}': {exc}")

            async def _calc_precision():
                if not has_ctx:
                    return
                try:
                    score = await precision_metric.single_turn_ascore(sample)
                    if score is not None and score == score:
                        totals["context_precision"].append(float(score))
                except Exception as exc:
                    print(f"  [WARN] context_precision failed for '{safe}': {exc}")

            async def _calc_recall():
                if not has_ctx:
                    return
                try:
                    score = await recall_metric.single_turn_ascore(sample)
                    if score is not None and score == score:
                        totals["context_recall"].append(float(score))
                except Exception as exc:
                    print(f"  [WARN] context_recall failed for '{safe}': {exc}")

            if not has_ctx:
                context_missing_count += 1

            # Run all 4 metrics simultaneously for this sample
            await asyncio.gather(
                _calc_faithfulness(),
                _calc_relevancy(),
                _calc_precision(),
                _calc_recall(),
            )

            scored_count += 1
            if not has_tqdm:
                print(f"  [SCORING {scored_count}/{total_samples}] Completed evaluation for '{safe}'")

    tasks = [_score_sample(s) for s in samples]
    if has_tqdm:
        await tqdm_asyncio.gather(*tasks, desc="Scoring samples", total=total_samples)
    else:
        await asyncio.gather(*tasks)

    if context_missing_count:
        print(
            f"\n  [INFO] {context_missing_count}/{len(samples)} samples had no retrieved_contexts "
            f"— faithfulness / precision / recall skipped for those."
        )
        print(
            "  [INFO] If this is unexpected, check that your RAG API returns contexts. "
            "See the 'Diagnosing empty contexts' note at the bottom of this script."
        )

    return {
        label: (sum(vals) / len(vals) if vals else float("nan"))
        for label, vals in totals.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 3 — Ragas Evaluation")
    print("=" * 60)

    # ── 1. Load test set ──────────────────────────────────────────────────────
    test_items = load_test_set(TEST_SET_PATH)

    # ── 2. Query RAG concurrently ─────────────────────────────────────────────
    print("\n[INFO] Querying RAG API with parallel workers...")
    samples_by_index: dict[int, dict] = {}
    completed_count  = 0
    total_items      = len(test_items)
    raw_keys_logged  = False   # log response shape once for debugging

    def _fetch_item(index: int, item: dict):
        nonlocal raw_keys_logged
        q   = item["question"]
        gt  = item.get("ground_truth_answer", item.get("answer", ""))
        ans, ctx = query_rag(q)

        # Log the raw response keys the first time so you can verify
        if not raw_keys_logged:
            try:
                resp_data = requests.post(
                    QUERY_ENDPOINT, json={"question": q}, timeout=30
                ).json()
                print(f"\n  [DEBUG] RAG response keys on first call: {list(resp_data.keys())}")
                print(f"  [DEBUG] contexts value: {repr(ctx[:1]) if ctx else '(empty list)'}\n")
            except Exception:
                pass
            raw_keys_logged = True

        return index, q, gt, ans, ctx

    max_query_workers = int(os.getenv("RAG_QUERY_WORKERS", "2"))
    with ThreadPoolExecutor(max_workers=max_query_workers) as executor:
        futures = {
            executor.submit(_fetch_item, i, item): i
            for i, item in enumerate(test_items, 1)
        }
        for future in as_completed(futures):
            idx, question, ground_truth, answer, contexts = future.result()
            completed_count += 1
            safe_q = question[:60].encode("ascii", "replace").decode("ascii")
            ctx_tag = f"[{len(contexts)} ctx]" if contexts else "[NO CTX]"
            if not answer:
                print(f"  [{completed_count}/{total_items}] [SKIP] Empty answer for '{safe_q}'")
            else:
                print(f"  [{completed_count}/{total_items}] OK {ctx_tag} '{safe_q}'")
                samples_by_index[idx] = {
                    "user_input":         question,
                    "response":           answer,
                    "retrieved_contexts": contexts,
                    "reference":          ground_truth,
                }

    samples = [samples_by_index[i] for i in sorted(samples_by_index.keys())]

    if not samples:
        print("[ERROR] No samples collected. Check that the RAG server is running.")
        sys.exit(1)

    ctx_count = sum(1 for s in samples if s["retrieved_contexts"])
    print(f"\n[INFO] Collected {len(samples)} samples "
          f"({ctx_count} with contexts, {len(samples)-ctx_count} without).")

    # ── 3. Build evaluator LLM (AsyncOpenAI) and embeddings ───────────────────
    #   ragas collections API uses agenerate() internally → must use AsyncOpenAI
    print("\n[INFO] Initialising evaluator LLM and embeddings (via LM Studio Nomic)...")
    # pyrefly: ignore[missing-import]
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    # pyrefly: ignore[missing-import]
    from ragas.llms import LangchainLLMWrapper
    # pyrefly: ignore[missing-import]
    from ragas.embeddings import LangchainEmbeddingsWrapper

    chat_model = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )
    eval_llm = LangchainLLMWrapper(chat_model)

    lmstudio_base_url = (
        f"{EMBEDDING_NOMIC_BASE_URL}/v1"
        if not EMBEDDING_NOMIC_BASE_URL.endswith("/v1")
        else EMBEDDING_NOMIC_BASE_URL
    )
    eval_emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBEDDING_NOMIC_MODEL,
            base_url=lmstudio_base_url,
            api_key="lm-studio",
            check_embedding_ctx_length=False,
        )
    )

    # ── 4. Score ──────────────────────────────────────────────────────────────
    eval_workers = int(os.getenv("RAGAS_EVAL_WORKERS", "2"))
    print(f"\n[INFO] Running ragas evaluation (workers={eval_workers})...")
    scores = asyncio.run(_score_all(samples, eval_llm, eval_emb))

    # ── 5. Print results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    all_pass = True
    for label, threshold in THRESHOLDS.items():
        score = scores.get(label)
        if score is None or score != score:   # None or NaN
            print(f"  {label:<30} N/A")
            all_pass = False
            continue
        status = "✅ PASS" if score >= threshold else "❌ FAIL"
        if score < threshold:
            all_pass = False
        print(f"  {label:<30} {score:.3f}  (threshold {threshold})  {status}")

    print("=" * 60)
    print("OVERALL:", "✅ ALL PASS" if all_pass else "❌ SOME METRICS BELOW THRESHOLD")
    print("=" * 60)

    # ── 6. Save detailed results ──────────────────────────────────────────────
    out_path = Path(__file__).parent / "ragas_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scores": {k: (None if (v != v) else v) for k, v in scores.items()},
                "samples_evaluated":      len(samples),
                "samples_with_contexts":  ctx_count,
            },
            f, indent=2,
        )
    print(f"\n[INFO] Results saved to {out_path}")

    # ── 7. Diagnosing empty contexts ──────────────────────────────────────────
    if ctx_count == 0:
        print("""
[HELP] All 50 samples returned empty retrieved_contexts.
       This means your RAG /api/v1/query endpoint is not returning context chunks.

       Check:
         1. What keys does your API response actually contain?
            → The [DEBUG] line above shows them.  Add the right key to
              query_rag() in this script under the 'contexts' loop.

         2. Does your API return contexts under a nested key?
            e.g.  data["result"]["source_documents"]
            → Add a custom extraction block in query_rag() for that shape.

         3. Is context retrieval disabled in your backend?
            → Check your backend config / RAG chain to ensure retriever
              is wired and returning source documents.

       Without contexts, faithfulness / context_precision / context_recall
       cannot be computed.  Only answer_relevancy will score.
""")


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSING EMPTY CONTEXTS — quick one-liner to inspect your API:
#
#   python -c "
#   import requests, json
#   r = requests.post('http://localhost:8000/api/v1/query',
#                     json={'question': 'test'}, timeout=30)
#   print(json.dumps(r.json(), indent=2, default=str))
#   "
# ─────────────────────────────────────────────────────────────────────────────