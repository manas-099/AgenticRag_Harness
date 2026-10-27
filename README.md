<!-- <div align="center">

```
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝
         R A G   H A R N E S S
``` -->

<div align="center">

<!-- ![RAG Harness](img/image.png) -->
<pre>
<!-- ``` -->
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗     ██████╗  █████╗  ██████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝     ██╔══██╗██╔══██╗██╔════╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║          ██████╔╝███████║██║  ███╗
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║          ██╔══██╗██╔══██║██║   ██║
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗     ██║  ██║██║  ██║╚██████╔╝
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝

 ██╗  ██╗ █████╗ ██████╗ ███╗   ██╗███████╗███████╗███████╗
 ██║  ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝
 ███████║███████║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗
 ██╔══██║██╔══██║██╔══██╗██║╚██╗██║██╔══╝  ╚════██║╚════██║
 ██║  ██║██║  ██║██║  ██║██║ ╚████║███████╗███████║███████║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝

<!-- 
``` -->
</pre>
</div>


** agentic RAG system built around a  ReAct loop.**  
The agent decides at every step — search, reformulate, fetch, compare, or answer.  
Every generated claim is verified against its source before being returned.

<br/>

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Loop-FF6B35?style=flat-square)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_Store-DC143C?style=flat-square)
![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)

</div>

---

## ⚡ Core Design Principle

> **The LLM only makes judgment calls — what to search, when to stop, what a claim means.  
> Everything checkable is enforced in code, never trusted to model self-report.**

Citation existence → checked in code. Rank fusion → rank position, not scores. Relevance gate → code blocks generation. Loop control → graph enforces it. None of this is a prompt.

---

## 🗺️ System Architecture

### High-Level Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  📄 Document │───▶│  ✂️  Chunk   │───▶│  🗄️  Store  │
│   Ingestion  │    │  + Preamble │    │   (Qdrant)  │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
┌────────────────────────────────────────────▼──────┐
│                  🤖 Agent Decision Loop            │
│                                                    │
│   ┌──────────────────────────────────────────┐    │
│   │  Structured LLM output picks ONE action: │    │
│   │                                          │    │
│   │  search_documents                        │    │
│   │  search_documents_reformulated           │    │
│   │  get_page                                │    │
│   │  get_context                             │    │
│   │  compare_chunks                          │    │
│   │  check_document_freshness                │    │
│   │  answer ──────────────────────┐          │    │
│   └───────────────────────────────│──────────┘    │
│                                   │               │
│   Loop guards (enforced by graph):│               │
│   • Iteration cap                 │               │
│   • Token budget cap              │               │
│   • Search-attempt cap            │               │
│   • Stuck-loop detector           │               │
└───────────────────────────────────│───────────────┘
                                    ▼
                        ┌─────────────────────┐
                        │  ✍️  Generate Answer  │
                        │  (relevance-gated)   │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  ✅ Verify Claims    │
                        │  code → NLI → retry  │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┴─────────────┐
                    │              │              │
                  pass           fail           degrade
                    │              │              │
                 return         retry          best partial
                                (capped)      + caveat
```

---

## 🏛️ Hexagonal Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           api/                                  │
│          Thin FastAPI layer — HTTP ↔ use-case only              │
│          POST /v1/documents  POST /v1/query  GET /v1/health     │
├─────────────────────────────────────────────────────────────────┤
│                        application/                             │
│     Use cases & orchestration. Depends only on domain.          │
│     agent/  generation/  ingestion/  retrieval/  verification/  │
├─────────────────────────────────────────────────────────────────┤
│                          domain/                                │
│   Pure Python — no frameworks, no I/O. Models + abstract ports  │
│   Chunk  RetrievalResult  Claim  BaseLLMClient(ABC)             │
├──────────────────────┬──────────────────────────────────────────┤
│   infrastructure/    │   Concrete implementations of ports      │
│                      │                                          │
│   document_parsers/  │   PyMuPDF (fast) + Docling (OCR/tables)  │
│   embeddings/        │   SentenceTransformer                    │
│   llm/               │   Groq · MyLLM · OpenRouter factory      │
│   sparse_search/     │   BM25 retriever                         │
│   vector_stores/     │   Qdrant chunk + section collections     │
│   cache/             │   InMemoryCache (swap for RedisCache)    │
└──────────────────────┴──────────────────────────────────────────┘
```

**Swap rule:** Changing the vector store → touch only `infrastructure/vector_stores/`. Changing the LLM backend → touch only `infrastructure/llm/`. Nothing in `application/` changes either way — it depends on `domain/ports/`, not concrete classes.

---

## 🔬 Pipeline Deep Dive

### 1 · Ingestion

```
PDF Document
     │
     ▼
┌────────────────────────────┐
│  Per-page classification   │  ← not per-document — a PDF can mix clean
│  (structural heuristic)    │    text, scans, and tables across pages
└──────┬──────────────┬──────┘
       │              │
  Clean text       Scanned / mixed
       │            (text + table + image)
       ▼              ▼
  PyMuPDF          Docling
  (fast-path)    (OCR + table recognition,
                  runs ONCE per doc — not
                  per-page; avoids reloading
                  OCR models repeatedly)
       │              │
       └──────┬───────┘
              ▼
     Post-extraction quality gate
     (garbled encoding? suspiciously short?
      → reroute to Docling even if classified clean)
```

### 2 · Chunking

```
Raw text
    │
    ▼  Try in order, fall back only if needed:
    ├── 1. Markdown headers
    ├── 2. Paragraph boundaries
    ├── 3. Sentence boundaries
    ├── 4. Word boundaries
    └── 5. Hard character cut  ← last resort only
    │
    ▼
Contextual preamble generation
    │  Cheap LLM call prepends 1–2 sentences per chunk:
    │  "This chunk is from [doc], section [X], discussing [entities/topic/dates]."
    │  Fixes: "revenue increased 12%" — whose revenue? what year?
    │
    │  Concurrent via ThreadPoolExecutor (I/O-bound)
    │  Order preserved via indexed futures (not append-as-completed)
    ▼
Section title extraction → feeds metadata-filtering layer
```

### 3 · Retrieval

```
User Query
     │
     ├──────────────────────────────────┐
     │                                  │
     ▼                                  ▼
Dense search                      Sparse search
(vector similarity)               (BM25)
     │                                  │
     │   Run in parallel                │
     │   (ThreadPoolExecutor)           │
     │   Latency = max(two), not sum    │
     └──────────────┬───────────────────┘
                    │
                    ▼
         Reciprocal Rank Fusion
         (combines by rank position —
          dense similarity and BM25 scores
          live on incomparable scales)
                    │
                    ▼
         Deduplication
         (embedding-similarity based,
          BEFORE reranking — don't waste
          the expensive step on near-dupes)
                    │
                    ▼
         Cross-encoder reranking
         (scores actual query–chunk pairs;
          precision pass on the small set)
```

### 4 · Metadata Filtering (Section-Based)

```
Query
  │
  ▼
Dedicated Qdrant collection
(section titles with dense + native sparse vectors)
  │
  ▼
RRF-fused dense+sparse search over sections
  │
  ▼
Cross-encoder rerank candidates
  │
  ▼
LLM structured output picks a section
  │
  ├── Validated against known-sections list
  │   (LLM output trusted only after code check)
  │
  ├── Match found → filtered retrieval
  └── No clear match → unfiltered retrieval (safe degradation)
```

### 5 · Verification

```
Generated answer
        │
        ▼
┌───────────────────┐
│ Claim extraction  │  (structured LLM output, not regex)
│ → {claim, chunk_id}│
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────┐
│ 1. Citation validator (code)   │  CHEAPEST — runs first
│    Was chunk_id actually        │  A citation to a chunk that wasn't
│    retrieved this turn?         │  retrieved is a hallucination.
└────────┬───────────────────────┘  No model needed to catch it.
         │
         ▼
┌────────────────────────────────┐
│ 2. NLI groundedness check      │  MOST IMPORTANT
│    Is each claim actually       │  Catches: citation is real but claim
│    entailed by its source text? │  overstates or misstates the source.
└────────┬───────────────────────┘
         │
    ┌────┴─────┐
   pass       fail
    │          │
  return    retry with specific failure reason (capped)
              │
           still fails → best-grounded partial answer + caveat
                          (never a silent failure or confident bluff)
```

---

## 📁 Project Structure

```
  🏛️ domain          ──→       ⚡ application        ──→      🔧 infrastructure
  ─────────────────            ─────────────────────           ─────────────────
  Pure Python                  Orchestration &                 Concrete impls
  No frameworks, no I/O        use cases only                  of domain ports
  Models + contracts           Depends on domain               Swap freely
       │                             │                               │
       ▼                             ▼                               ▼
  models/                      agent/                          vector_stores/
  ports/                       retrieval/                      llm/
                                ingestion/                      document_parsers/
                                verification/                   embeddings/
                                generation/                     sparse_search/

                          🌐 api/  ←  thin HTTP shell only, no logic
```

<details>
<summary><strong>📂 Full file tree</strong></summary>

```
📦 agentic_rag_harness/
│
├── 📱 apps/
│   └── 🖥️  backend/
│       ├── 📂 src/
│       │   └── 📂 rag_harness/
│       │       │
│       │       ├── 🏛️  domain/                    # Pure business logic — no I/O, no frameworks
│       │       │   ├── 📐 models/
│       │       │   │   ├── 🧩 chunk.py             # Chunk + ChunkMetadata
│       │       │   │   ├── 🤖 agent_state.py       # ReAct loop state
│       │       │   │   ├── 🔍 retrieval.py         # RetrievalResult
│       │       │   │   └── ✅ verification.py      # Claim + VerificationResult
│       │       │   └── 🔌 ports/                   # Abstract contracts (ABCs)
│       │       │       ├── llm_port.py             # BaseLLMClient
│       │       │       ├── embedder_port.py
│       │       │       ├── vector_store_port.py
│       │       │       └── document_parser_port.py
│       │       │
│       │       ├── ⚡ application/                 # Use cases — depends only on domain
│       │       │   ├── 📋 rag_pipeline.py          # Pipeline orchestrator
│       │       │   ├── 🤖 agent/                   # LangGraph ReAct loop
│       │       │   │   ├── decision.py             # Structured LLM action decision
│       │       │   │   ├── graph_builder.py        # State graph wiring
│       │       │   │   ├── nodes.py                # Graph execution nodes
│       │       │   │   ├── routing.py              # Conditional edge routing
│       │       │   │   └── tools.py                # Agent tools
│       │       │   ├── ✍️  generation/              # Answer generation + prompts
│       │       │   ├── 📥 ingestion/               # Chunker · splitter · file router
│       │       │   ├── 🔍 retrieval/               # RRF · dedup · reranking · filters
│       │       │   └── ✅ verification/            # Citation validator · NLI entailment
│       │       │
│       │       ├── 🔧 infrastructure/              # Concrete port implementations
│       │       │   ├── 📄 document_parsers/        # PyMuPDF (fast) + Docling (OCR/tables)
│       │       │   ├── 🧠 embeddings/              # SentenceTransformer embedder
│       │       │   ├── 💬 llm/                     # Groq · MyLLM · OpenRouter
│       │       │   │   └── llm_factory.py          # Single factory for all backends
│       │       │   ├── 🔎 sparse_search/           # BM25 retriever
│       │       │   ├── 🗄️  vector_stores/          # Qdrant chunk + section collections
│       │       │   └── 🗃️  cache/                  # InMemoryCache (swap for RedisCache)
│       │       │
│       │       ├── 🌐 api/                         # FastAPI — routing only, zero logic
│       │       │   └── v1/
│       │       │       ├── routers/
│       │       │       │   ├── documents.py        # POST /v1/documents
│       │       │       │   ├── query.py            # POST /v1/query · /v1/query/agentic
│       │       │       │   └── health.py           # GET  /v1/health
│       │       │       └── schemas/               # Pydantic request/response models
│       │       │
│       │       └── ⚙️  config/
│       │           └── settings.py                 # Pydantic BaseSettings
│       │
│       ├── 📜 scripts/
│       │   ├── ingest_cli.py
│       │   └── query_cli.py
│       │
│       └── 🧪 tests/
│           ├── unit/                               # Fast — no external dependencies
│           ├── integration/
│           └── e2e/                                # Requires Qdrant + LLM backend
│
├── 🔬 experiments/                                 # Notebooks for iterative dev
│   ├── Agentic_rag_Harness.ipynb
│   ├── INGESTION_README.md
│   └── RETRIEVAL_README.md
│
└── 🏗️  infra/
    ├── k8s/
    └── terraform/
```

</details>

---

## 🚀 Getting Started

### Backend

```bash
cd apps/backend
uv sync                 # installs all dependencies from uv.lock
cp .env.example .env    # fill in MYLLM_AUTH_TOKEN, GROQ_API_KEY, OPENROUTER_API_KEY
```

```bash
# Ingest a document
python scripts/ingest_cli.py --file path/to/doc.pdf --doc-id doc1 --version v1

# Simple retrieve-then-generate
python scripts/query_cli.py --question "What is this document about?"

# Full ReAct agent loop
python scripts/query_cli.py --question "What is this document about?" --agentic

# API server
uvicorn rag_harness.api.main:app --reload
```

**Endpoints:** `POST /v1/documents` · `POST /v1/query` · `POST /v1/query/agentic` · `GET /v1/health`

### Frontend

```bash
cd apps/frontend
npm install
cp .env.example .env    # set VITE_API_BASE_URL to backend URL (leave empty for local dev proxy)
npm run dev
```

### Tests

```bash
cd apps/backend
pytest tests/unit                      # fast, no external dependencies
RUN_E2E_TESTS=1 pytest tests/e2e       # requires Qdrant + LLM backend
```

---

## ⚙️ Key Design Decisions

| Decision | What was done | Why |
|---|---|---|
| Per-page PDF classification | Each page classified individually | A single PDF can mix clean text, scans, and tables |
| Docling runs once per doc | Batch all flagged pages into one Docling call | Per-page reloads OCR models — measurably slow |
| RRF over score averaging | Rank position used for fusion | Dense similarity and BM25 scores are incomparable scales |
| Dedup before reranking | Embedding-similarity dedup runs first | Don't waste cross-encoder compute on near-duplicate chunks |
| Relevance gate before generation | Best chunk score checked in code | Primary hallucination guardrail — more reliable than any prompt |
| Citation check before NLI | Code checks chunk_id existence first | Cheapest check runs first; model time saved for harder judgments |
| Stuck-loop detector | Zero new chunks → force degrade | Prevents infinite loop when reformulation isn't helping |
| Version-keyed cache | Cache entries tagged with `doc_version` | Re-ingesting a doc selectively invalidates stale entries |


---

## 📊 Evaluation Results

Run on the 50-sample labeled test set (`evals/test_set.jsonl`), per the eval plan's
Phase 3 (Ragas) and Phase 4 (DeepEval trajectory) layers.

### Ragas — Retrieval & Generation Quality

| Metric | Score |
|--------|-------|
| `faithfulness` | **0.973** |
| `context_precision` | **0.909** |
| `context_recall` | **0.906** |
| `answer_relevancy` | **0.666** |

**Samples evaluated:** 50/50 · **Samples with contexts:** 50/50

### DeepEval — Agent Trajectory & Task Layer

| Metric | avg | min | max |
|--------|-----|-----|-----|
| Task Completion | 1.000 | 1.000 | 1.000 |
| Tool Correctness | 1.000 | 1.000 | 1.000 |
| Answer Relevancy | 0.867 | 0.500 | 1.000 |

**Degraded runs:** 0/10 · **Failing questions:** 3/10

### Reading the numbers

Retrieval is solid — `context_precision` and `context_recall` both at ~0.91,
confirming the right chunks are reaching generation. `faithfulness` at 0.973
confirms the verification layer is doing its job: answers stay grounded in
retrieved text.

---

### 🔬 BM25 Contribution Diagnostic (A/B Test)

#### Why This Diagnostic Was Run
In hybrid retrieval setups (Dense vector + BM25 sparse + Cross-Encoder rerank), sparse lexical search (BM25) is often kept in memory (`BM25Retriever`). Because in-memory indices reset upon server restarts, they represent a potential cold-start liability unless warmed up or persisted to disk.

To objectively evaluate whether BM25 is genuinely contributing to recall or acting as dead weight, we ran an isolated A/B diagnostic on the exact same 50-sample evaluation set (`evals/test_set.jsonl`):
- **Run A (Hybrid Baseline):** Dense (Nomic embed) + BM25 Sparse + Cross-Encoder Rerank
- **Run B (Dense-Only):** BM25 disabled (`RETRIEVAL_TOP_K_SPARSE=0`), Dense + Cross-Encoder Rerank

#### Dedicated Diagnostic Files & Locations
| File | Path | Description |
|------|------|-------------|
| **Diagnostic Runner** | [`evals/bm25_diagnostic.py`](evals/bm25_diagnostic.py) | Executes parallelized multi-worker query sampling against `/api/v1/query/agentic` and runs Ragas `context_recall`. |
| **Comparison Script** | [`evals/bm25_compare.py`](evals/bm25_compare.py) | Compares Run A and Run B JSON artifacts and computes recall delta and architectural recommendations. |
| **Hybrid Result Artifact** | [`evals/results/bm25_diagnostic_hybrid.json`](evals/results/bm25_diagnostic_hybrid.json) | Saved benchmark output for the hybrid baseline (`context_recall`: **0.9365**). |
| **Dense-Only Result Artifact** | [`evals/results/bm25_diagnostic_dense_only.json`](evals/results/bm25_diagnostic_dense_only.json) | Saved benchmark output for dense-only retrieval (`context_recall`: **0.7803**). |

#### Diagnostic Results

```
============================================================
  BM25 DIAGNOSTIC — COMPARISON
============================================================
  context_recall  HYBRID (dense+BM25+rerank) : 0.9365
  context_recall  DENSE ONLY (dense+rerank)  : 0.7803
  delta (hybrid - dense)                     : +0.1562
============================================================
```

#### Observations & Conclusion
- **Massive Recall Delta (+0.1562 / +15.6%)**: Disabling BM25 caused `context_recall` to drop sharply from **0.9365** to **0.7803**.
- **Lexical Precision Is Essential**: In highly regulated, domain-specific corpora (such as NBFC, RBI circulars, statutory definitions, and exact section codes), dense embeddings alone fail to retrieve precise exact-match terms and alphanumeric tokens. BM25 reliably rescues these critical passages before cross-encoder reranking.
- **Architectural Takeaway**: BM25 is **not dead weight**—it is a vital component of the retrieval pipeline. Consequently, resolving BM25 index persistence/cold-start is non-negotiable:
  - **Implemented Protection**: Automatic startup warm-up via FastAPI lifespan hook (`scroll_all_chunks()` from Qdrant payloads into `BM25Retriever`).
  - **Production Option**: Serialized disk cache (pickle/mmap) for instant startup on very large corpora.


---

## 🔭 Production Considerations

**Section-title extraction uses regex-based heuristics**
Currently detects markdown `#` headers — reliable on Docling-routed pages, no-op on plain-text (AnyDoc) pages. Metadata filtering degrades safely to unfiltered when no section is found. The natural next step is reading Docling's item type directly, which removes the dependency on markdown syntax entirely.

**`InMemoryCache` is intentionally single-process**
The cache interface is backend-agnostic — swapping to `RedisCache` (implementing the same interface) requires zero changes to calling code. Designed this way deliberately to keep local dev dependency-free.

**Embedded Qdrant for local development**
Set `QDRANT_URL` in `.env` to point at a real Qdrant server for multi-process or production deployments. The client code is identical either way — only the config changes.

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).