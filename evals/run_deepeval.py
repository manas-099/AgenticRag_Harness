"""
Phase 4 — DeepEval Agent Trajectory Evaluation
Evaluates the AgenticRAG agent's reasoning steps and tool usage.

Uses /api/v1/query/agentic which returns:
  - answer        : final LLM answer
  - action_steps  : [{action, thought, iteration}, ...]  ← trajectory
  - action_history: [str, ...]

"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# pyrefly: ignore[missing-import]
import requests
# pyrefly: ignore[missing-import]
from dotenv import load_dotenv

# ── Load .env from apps/backend ───────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / "apps" / "backend" / ".env")

# ── Judge LLM config (strictly from .env) ────────────────────────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY  = os.getenv("LLM_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "default-model")

if not LLM_BASE_URL or not LLM_API_KEY:
    print("[ERROR] LLM_BASE_URL and LLM_API_KEY must be set in your .env file.")
    sys.exit(1)

# ── Point DeepEval at the configured LLM (OpenAI-compatible) ─────────────────
os.environ["OPENAI_API_KEY"]  = LLM_API_KEY
os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL

# ── RAG API ───────────────────────────────────────────────────────────────────
RAG_API_BASE     = os.getenv("RAG_API_BASE", "http://localhost:8000")
AGENTIC_ENDPOINT = f"{RAG_API_BASE}/api/v1/query/agentic"

# ── Test-set path ─────────────────────────────────────────────────────────────
TEST_SET_PATH = Path(__file__).parent / "test_set.jsonl"

# ── Workers — keep at 1 to avoid overwhelming NomicEmbedder (LM Studio) ──────
# LM Studio is single-threaded; concurrent queries cause embedding timeouts.
RAG_QUERY_WORKERS = 1

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "task_completion":  0.70,
    "tool_correctness": 0.80,
    "answer_relevancy": 0.75,
}

# ─────────────────────────────────────────────────────────────────────────────
# DeepEval imports
# ─────────────────────────────────────────────────────────────────────────────
try:
    # pyrefly: ignore[missing-import]
    from deepeval import evaluate
    # pyrefly: ignore[missing-import]
    from deepeval.test_case import LLMTestCase, ToolCall
    # pyrefly: ignore[missing-import]
    from deepeval.metrics import (
        TaskCompletionMetric,
        ToolCorrectnessMetric,
        AnswerRelevancyMetric,
    )
except ImportError:
    print("[ERROR] deepeval not installed. Run: uv pip install deepeval")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_test_set(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ERROR] Test set not found: {path}")
        print("        Generate it first — place test_set.jsonl in evals/")
        sys.exit(1)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"[INFO] Loaded {len(items)} test items.")
    return items


def query_agentic(question: str, index: int, total: int) -> dict | None:
    """POST to /api/v1/query/agentic and return the full response dict."""
    safe_q = question[:65].encode("ascii", "replace").decode("ascii")
    print(f"  [{index}/{total}] Querying: '{safe_q}'")
    t0 = time.time()
    try:
        resp = requests.post(
            AGENTIC_ENDPOINT,
            json={"question": question},   # FIX 1: was {"query": question} — wrong field name, caused HTTP 422
            timeout=120,
        )
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        iterations = data.get("iterations_used", 0)
        degraded   = data.get("degraded", False)
        steps      = data.get("action_steps", [])
        tools      = [s.get("action", "") for s in steps if s.get("action") and s.get("action") != "answer"]
        deg_str    = " ⚠️ DEGRADED (sparse-only)" if degraded else ""
        print(f"         ✅ Done in {elapsed:.1f}s | {iterations} iters | tools: {tools}{deg_str}")
        return data
    except requests.exceptions.Timeout:
        elapsed = time.time() - t0
        print(f"         ❌ TIMEOUT after {elapsed:.0f}s — NomicEmbedder overloaded? (RAG_QUERY_WORKERS={RAG_QUERY_WORKERS})")
        return None
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Cannot reach server at {AGENTIC_ENDPOINT}")
        print("        Start the server first:")
        print("          cd apps/backend && agent_env\\Scripts\\activate")
        print("          uvicorn rag_harness.api.main:app --reload")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        elapsed = time.time() - t0
        print(f"         ❌ HTTP {e.response.status_code} after {elapsed:.1f}s: {e}")
        return None
    except Exception as e:
        elapsed = time.time() - t0
        print(f"         ❌ FAILED after {elapsed:.1f}s: {e}")
        return None


def build_tool_calls(action_steps: list[dict]) -> list[ToolCall]:
    """
    Convert action_steps from the API into DeepEval ToolCall objects.
    action_steps format: [{action, thought, iteration}, ...]
    """
    tool_calls = []
    for step in action_steps:
        action = step.get("action", "")
        if action and action != "answer":
            tool_calls.append(
                ToolCall(
                    name=action,
                    reasoning=step.get("thought", ""),   # FIX 2: was description= (wrong field)
                )
            )
    return tool_calls


def build_expected_tools() -> list[ToolCall]:
    """For any retrieval question, agent must call search_documents."""
    return [ToolCall(name="search_documents")]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 4 — DeepEval Agent Trajectory Evaluation")
    print("=" * 60)
    print(f"Judge LLM : {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"Agent API : {AGENTIC_ENDPOINT}")
    print(f"Workers   : {RAG_QUERY_WORKERS} (sequential — prevents NomicEmbedder timeout)")
    print()

    # 1. Load test set
    test_items = load_test_set(TEST_SET_PATH)
    total = len(test_items)

    # 2. Query agent sequentially (workers=1 to protect LM Studio)
    print(f"[INFO] Querying agentic endpoint (workers={RAG_QUERY_WORKERS})...")
    print("-" * 60)

    test_cases: list[LLMTestCase] = []
    skipped = 0
    degraded_count = 0
    run_start = time.time()

    for i, item in enumerate(test_items, 1):
        question    = item["question"]
        ground_truth = item.get("ground_truth_answer", item.get("answer", ""))

        data = query_agentic(question, i, total)

        if not data:
            print(f"         [SKIP] No response — skipping #{i}")
            skipped += 1
            continue

        answer       = data.get("answer", "")
        action_steps = data.get("action_steps", [])
        degraded     = data.get("degraded", False)

        if not answer:
            print(f"         [SKIP] Empty answer — skipping #{i}")
            skipped += 1
            continue

        if degraded:
            degraded_count += 1

        tools_called   = build_tool_calls(action_steps)
        expected_tools = build_expected_tools()

        test_cases.append(LLMTestCase(
            input           = question,
            actual_output   = answer,
            expected_output = ground_truth,
            tools_called    = tools_called,
            expected_tools  = expected_tools,
        ))

    elapsed_query = time.time() - run_start
    print("-" * 60)
    print(f"[INFO] Query phase done in {elapsed_query:.0f}s")
    print(f"       Built  : {len(test_cases)}/{total} test cases")
    print(f"       Skipped: {skipped}/{total}")
    print(f"       Degraded (sparse-only): {degraded_count}/{len(test_cases)}")

    if not test_cases:
        print("\n[ERROR] No test cases built. Check server is running and test_set.jsonl exists.")
        sys.exit(1)

    # 3. Define metrics
    print(f"\n[INFO] Running DeepEval on {len(test_cases)} test cases...")
    print("       (This calls Medha LLM as judge — may take a few minutes)")
    print("-" * 60)

    metrics = [
        TaskCompletionMetric(
            threshold    = THRESHOLDS["task_completion"],
            model        = LLM_MODEL,
            verbose_mode = False,
        ),
        ToolCorrectnessMetric(
            threshold = THRESHOLDS["tool_correctness"],
        ),
        AnswerRelevancyMetric(
            threshold    = THRESHOLDS["answer_relevancy"],
            model        = LLM_MODEL,
            verbose_mode = False,
        ),
    ]

    # 4. Run evaluation
    eval_start = time.time()
    results = evaluate(test_cases=test_cases, metrics=metrics, print_results=False)
    elapsed_eval = time.time() - eval_start
    print(f"\n[INFO] Evaluation done in {elapsed_eval:.0f}s")

    # 5. Per-question detail log
    print("\n" + "=" * 60)
    print("PER-QUESTION RESULTS")
    print("=" * 60)

    label_map = {
        "Task Completion":  "task_completion",
        "Tool Correctness": "tool_correctness",
        "Answer Relevancy": "answer_relevancy",
    }

    metric_scores: dict[str, list[float]] = {}
    per_question_fails: list[dict] = []

    for tc in test_cases:
        q_short = tc.input[:70].encode("ascii", "replace").decode("ascii")
        scores_line = []
        q_failed = False
        for md in (tc.metrics_data or []):
            name  = md.name
            score = md.score or 0.0
            passed = md.success
            if name not in metric_scores:
                metric_scores[name] = []
            metric_scores[name].append(score)
            icon = "✅" if passed else "❌"
            scores_line.append(f"{name[:4]}={score:.2f}{icon}")
            if not passed:
                q_failed = True
        status = "FAIL" if q_failed else "pass"
        print(f"  [{status}] {q_short}")
        print(f"         {' | '.join(scores_line)}")
        if q_failed:
            per_question_fails.append({"question": tc.input, "metrics": tc.metrics_data})

    # 6. Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    all_pass = True
    for display_name, threshold_key in label_map.items():
        scores = metric_scores.get(display_name, [])
        if not scores:
            print(f"  {display_name:<25} N/A")
            continue
        avg   = sum(scores) / len(scores)
        thr   = THRESHOLDS.get(threshold_key, 0.0)
        passd = avg >= thr
        if not passd:
            all_pass = False
        status = "✅ PASS" if passd else "❌ FAIL"
        low    = min(scores)
        high   = max(scores)
        print(f"  {display_name:<25} avg={avg:.3f}  min={low:.3f}  max={high:.3f}  threshold={thr}  {status}")

    print(f"\n  Samples evaluated : {len(test_cases)}/{total}")
    print(f"  Samples skipped   : {skipped}/{total}")
    print(f"  Degraded runs     : {degraded_count}/{len(test_cases)}")
    print(f"  Failing questions : {len(per_question_fails)}")
    print("=" * 60)
    print("OVERALL:", "✅ ALL PASS" if all_pass else "❌ SOME METRICS BELOW THRESHOLD")
    print("=" * 60)

    # 7. Save per-question detail
    out_path = Path(__file__).parent / "deepeval_results.json"
    detail = []
    for tc in test_cases:
        row = {
            "question":     tc.input,
            "answer":       tc.actual_output,
            "tools_called": [t.name for t in (tc.tools_called or [])],
            "metrics":      {},
        }
        for md in (tc.metrics_data or []):
            row["metrics"][md.name] = {
                "score":  md.score,
                "passed": md.success,
                "reason": md.reason,
            }
        detail.append(row)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Per-question detail saved → {out_path}")


if __name__ == "__main__":
    main()