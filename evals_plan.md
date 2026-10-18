# AgenticRAG Harness — Evaluation Plan

## Overview

The system has four independently-failable layers. A bad answer can come from any one of them,
so each layer gets its own eval signals. Evals are split into two modes:

- **Offline eval** — runs against a fixed labeled test set, gates CI on every meaningful change
- **Online eval** — runs in production on live traffic, catches slow drift and operational failures

```
Ingestion  →  Retrieval  →  Agent Loop  →  Generation + Verification
   Layer 1         Layer 2        Layer 3              Layer 4
```

---

## Frameworks

| Framework | Role | Install |
|-----------|------|---------|
| **Ragas** | Offline retrieval + generation quality metrics | `pip install ragas` |
| **Arize Phoenix** | Online tracing of every LangGraph run + eval overlays | `pip install arize-phoenix openinference-instrumentation-langchain` |
| **DeepEval** | Agent trajectory checks in CI (tool selection, order correctness) | `pip install deepeval` |
| **Custom logger** | Structured JSON logs from `/query/agentic` for operational metrics | built-in |

Phoenix is the production backbone. Ragas + DeepEval are CI tools. Custom logs feed a dashboard.

---

## Phase 1 — Structured Logging 

**Goal:** capture all agent signals that already exist in the API response before touching any
eval framework. Zero code risk, immediate visibility.

Add one structured log line in `apps/backend/src/rag_harness/api/v1/routers/query.py`
after every agentic run:

```python
# After the agent run, before returning the response
from collections import Counter
import json, logging

eval_logger = logging.getLogger("rag_harness.eval")

action_counts = Counter(raw_actions)
eval_logger.info(json.dumps({
    "event": "agentic_query",
    "query_hash": cache_key,
    "degraded": result.get("degraded", False),
    "degrade_reason": result.get("degrade_reason"),
    "iterations_used": result.get("iterations_used", 0),
    "max_iterations": pipeline.agentic_use_case.loop_settings.MAX_ITERATIONS,
    "validation_retries": result.get("validation_retries", 0),
    "token_budget_used": result.get("token_budget_used", 0),
    "max_token_budget": pipeline.agentic_use_case.loop_settings.MAX_TOKEN_BUDGET,
    "sources_count": len(result.get("sources_used", [])),
    "from_cache": False,
    "action_distribution": dict(action_counts),
    "search_attempts": action_counts.get("search_documents", 0)
                       + action_counts.get("search_documents_reformulated", 0),
}))
```

**Metrics you get for free immediately:**

| Metric | Target | Alert if |
|--------|--------|----------|
| Degradation rate | < 5% | > 10% over 1h window |
| Avg iterations used | < 5 of 8 | P95 > 7 (hitting ceiling) |
| Verification retry rate | < 15% | > 30% |
| Token budget burn | < 70% of max | P95 > 90% |
| Cache hit rate | track baseline | sudden drop (corpus changed) |

---

## Phase 2 — Build the Test Set 

**This is the hardest step and nothing downstream works without it.**

Create `evals/test_set.jsonl` — 50 to 100 entries minimum, 200 is better.

Each entry:

```jsonc
{
  "id": "q001",
  "question": "What is the difference between standard RAG and agentic RAG?",
  "ground_truth_answer": "Standard RAG is single-pass retrieve-then-generate. Agentic RAG uses a ReAct loop where the agent iteratively retrieves, reasons, and decides what to do next based on what it has found.",
  "relevant_doc_ids": ["doc_abc123", "doc_def456"],   // which docs should be retrieved
  "relevant_chunk_ids": [],                            // optional, fill in after first run
  "difficulty": "easy",                                // easy / medium / hard / unanswerable
  "tags": ["concept", "comparison"]
}
```

**Sampling strategy:**
- 40% easy — factual lookups with a single source chunk
- 30% medium — multi-chunk synthesis required
- 20% hard — requires comparison, temporal reasoning, or reformulation
- 10% unanswerable — questions with no answer in the corpus (tests graceful degradation)

**Tooling to build the set quickly:**

```python
# scripts/generate_test_questions.py
# Run once: use your LLM to generate candidate Q/A pairs from your documents,
# then manually review and prune. 1 hour of human review beats 10 hours of
# prompt engineering a perfect generator.
```

---

## Phase 3 — Ragas Offline Eval 

**Goal:** measure retrieval and generation quality against the labeled test set.

### Setup

```python
# evals/run_ragas.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
import requests, json

BASE_URL = "http://localhost:8000/api/v1"

def run_agentic_query(question: str) -> dict:
    resp = requests.post(f"{BASE_URL}/query/agentic", json={"question": question})
    return resp.json()

def build_ragas_dataset(test_set_path: str) -> Dataset:
    rows = []
    with open(test_set_path) as f:
        for line in f:
            item = json.loads(line)
            result = run_agentic_query(item["question"])
            # Pull full chunk texts from the registry via last_state endpoint
            state = requests.get(f"{BASE_URL}/query/agentic/last_state").json()
            rows.append({
                "question": item["question"],
                "answer": result["answer"],
                "contexts": [s for s in result.get("sources_used", [])],
                "ground_truth": item["ground_truth_answer"],
            })
    return Dataset.from_list(rows)

dataset = build_ragas_dataset("evals/test_set.jsonl")
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(results)
# results.to_pandas().to_csv("evals/ragas_results.csv")
```

### Metrics and Thresholds

| Metric | Meaning | Minimum passing |
|--------|---------|-----------------|
| `context_precision` | Are top chunks actually relevant? | 0.70 |
| `context_recall` | Are all needed chunks present? | 0.65 |
| `faithfulness` | Does the answer stay within the chunks? | 0.80 |
| `answer_relevancy` | Does the answer address the question? | 0.75 |

### When to Re-run Ragas

Ragas must run (and pass thresholds) before merging any PR that changes:

- Chunking settings (`CHUNK_SIZE_CHARS`, `CHUNK_OVERLAP_CHARS`)
- Embedding model
- Retrieval settings (`TOP_K_DENSE`, `TOP_K_SPARSE`, `RRF_K`, rerank model)
- Generation prompt in `nodes.py::make_generate_node`
- Decision engine prompt in `decision.py`
- Verification prompt in `verify_answer_use_case.py`

For everything else (infra changes, API changes, logging changes), Ragas is optional.

---

## Phase 4 — Agent Trajectory Eval with DeepEval 

**Goal:** catch regressions in agent decision quality — wrong tool choice, unnecessary loops,
failure to reformulate when stuck.

```python
# evals/run_trajectory_eval.py
from deepeval import evaluate as deval_evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToolCallMetric, TaskCompletionMetric
import requests, json

BASE_URL = "http://localhost:8000/api/v1"

def build_trajectory_cases(test_set_path: str) -> list[LLMTestCase]:
    cases = []
    with open(test_set_path) as f:
        for line in f:
            item = json.loads(line)
            result = requests.post(f"{BASE_URL}/query/agentic",
                                   json={"question": item["question"]}).json()
            state = requests.get(f"{BASE_URL}/query/agentic/last_state").json()

            cases.append(LLMTestCase(
                input=item["question"],
                actual_output=result["answer"],
                expected_output=item["ground_truth_answer"],
                tools_called=result.get("action_history", []),
            ))
    return cases

cases = build_trajectory_cases("evals/test_set.jsonl")
deval_evaluate(cases, metrics=[TaskCompletionMetric(threshold=0.7)])
```

**Custom trajectory metrics to track (compute from your logs, no framework needed):**

```python
# evals/trajectory_metrics.py

def progress_rate(action_history: list[str], chunk_counts: list[int]) -> float:
    """Fraction of search steps that found new chunks."""
    search_steps = [a for a in action_history if "search" in a]
    if not search_steps:
        return 0.0
    productive = sum(1 for i, count in enumerate(chunk_counts[1:], 1)
                     if count > chunk_counts[i - 1])
    return productive / len(search_steps)

def reformulation_rate(action_history: list[str]) -> float:
    """Fraction of searches that used reformulation."""
    searches = [a for a in action_history if "search" in a]
    if not searches:
        return 0.0
    reformulated = sum(1 for a in searches if "reformulated" in a)
    return reformulated / len(searches)

def answer_without_burnout(action_history: list[str], max_iterations: int) -> bool:
    """Did the agent answer before hitting the iteration cap?"""
    return (action_history[-1] == "answer" if action_history else False)
```

---

## Phase 5 — Phoenix Production Tracing 

**Goal:** trace every LangGraph node in production, get per-node latency, spot failures in real time.

### Instrumentation

Add to `apps/backend/main.py`:

```python
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor

# Launch Phoenix collector (or point to a remote Phoenix server)
# For local dev: px.launch_app() opens UI at http://localhost:6006
# For production: set PHOENIX_COLLECTOR_ENDPOINT env var to your hosted instance
px.launch_app()
LangChainInstrumentor().instrument()
```

This automatically instruments every LangGraph `StateGraph` invocation. No code changes needed
to graph nodes — Phoenix intercepts at the LangChain/LangGraph layer.

### What you see in the Phoenix UI

- Every agent run as a trace tree: `agent_decide → tool_dispatch → generate → verify`
- Per-node latency (spot if reranking or LLM calls are your bottleneck)
- Token usage per node
- Full scratchpad at each step
- Which runs degraded and why

### Eval overlays in Phoenix

Phoenix lets you attach eval functions to any span. Add these to auto-score every production run:

```python
from phoenix.evals import llm_classify, HallucinationEvaluator, QAEvaluator

# Run nightly against the previous day's traces
hallu_eval = HallucinationEvaluator(model=your_llm)
qa_eval = QAEvaluator(model=your_llm)
# Phoenix pulls spans from its store; evals write scores back to the same span
```

---

## Phase 6 — BM25 Warm-Up Diagnostic (One-Time)

**This is the most important unknown in your system.**

BM25 resets to empty on every server restart (`BM25Retriever` is in-memory only). Sparse retrieval
contributes nothing until documents are re-ingested in the current session.

Run this diagnostic once to decide whether BM25 is worth the complexity:

```python
# evals/bm25_diagnostic.py
# 1. Run your full test set → record context_recall scores  (hybrid = dense + sparse)
# 2. Set TOP_K_SPARSE=0 in settings (effectively disables BM25)
# 3. Re-run test set → record context_recall scores         (dense only)
# 4. Compare delta

# If delta < 0.03: BM25 contributes almost nothing after rerank.
#                  Consider removing it or persisting it to disk.
# If delta > 0.08: BM25 meaningfully fills keyword-matching gaps.
#                  You need a warm-up strategy (persist BM25 index to disk,
#                  or rebuild on startup from Qdrant payloads).
```

---

## Operational Dashboard (Ongoing)

Aggregate your structured logs into a simple dashboard. Target metrics and thresholds:

| Signal | Healthy | Investigate | Alert |
|--------|---------|-------------|-------|
| Degradation rate (rolling 1h) | < 5% | 5–10% | > 10% |
| Avg iterations per query | < 4 | 4–6 | > 6.5 (ceiling pressure) |
| Verification retry rate | < 15% | 15–25% | > 25% |
| MyLLM fallback rate | < 10% | 10–30% | > 30% (primary is down) |
| Cache hit rate | track baseline | sudden -20% | sudden -50% |
| Ingestion preamble failure rate | < 2% | 2–10% | > 10% |
| P95 end-to-end latency | < 15s | 15–25s | > 30s |

---

## CI Gate (Summary)

```yaml
# .github/workflows/eval.yml  (or equivalent)
# Trigger: PR changing retrieval/, generation/, verification/, config/settings.py

steps:
  - name: Start backend
    run: uvicorn main:app &

  - name: Run Ragas offline eval
    run: python evals/run_ragas.py
    # Fails if any metric < threshold defined in Phase 3

  - name: Run trajectory eval
    run: python evals/run_trajectory_eval.py
    # Fails if TaskCompletionMetric < 0.70

  - name: Check degradation rate on test set
    run: python evals/check_degradation.py
    # Fails if degradation rate on test set > 10%
```

---

## File Structure

```
evals/
  test_set.jsonl              # labeled Q/A/relevant_docs test cases
  run_ragas.py                # Phase 3 — offline retrieval + generation eval
  run_trajectory_eval.py      # Phase 4 — agent trajectory eval (DeepEval)
  trajectory_metrics.py       # Phase 4 — custom progress rate, reformulation rate
  bm25_diagnostic.py          # Phase 6 — one-time BM25 contribution analysis
  check_degradation.py        # CI gate — degradation rate on test set
  results/                    # gitignored, output CSVs from eval runs
    ragas_YYYY-MM-DD.csv
    trajectory_YYYY-MM-DD.csv
```

---

## Priority Order

1. **Structured logging** — do this first, costs nothing, gives immediate signals
2. **Test set** — nothing else is meaningful without labeled data
3. **Ragas offline eval** — catches the most common regressions (retrieval + faithfulness)
4. **Phoenix tracing** — production visibility, essential once you have real users
5. **BM25 diagnostic** — answers a fundamental architecture question
6. **DeepEval trajectory** — useful once you have > 100 test cases
7. **Dashboard** — operationalize the log signals once patterns emerge
