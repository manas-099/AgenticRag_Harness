<div align="center">

```
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝
         R A G   H A R N E S S
```
<div align="center">
![alt text](img\image.png)
```
**A production-grade agentic RAG system built around a real ReAct loop.**  
The agent decides at every step — search, reformulate, fetch, compare, or answer.  
Every generated claim is verified against its source before being returned.

<br/>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
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
│       │       │   └── 🗄️  vector_stores/          # Qdrant chunk + section collections
│       │       │
│       │       ├── 🌐 api/                         # FastAPI — routing only, zero logic
│       │       │   └── v1/routers/
│       │       │       ├── documents.py            # POST /v1/documents
│       │       │       ├── query.py                # POST /v1/query · /v1/query/agentic
│       │       │       └── health.py               # GET  /v1/health
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
pip install -e .
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
cp .env.example .env    # set VITE_API_URL to backend URL
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

## ⚠️ Known Limitations

**Section-title extraction is incomplete**
Only reliably finds headers on Docling-routed pages with markdown `#` syntax. Plain-text pages (AnyDoc route) produce no section title. Metadata filtering degrades safely to unfiltered — but the extraction needs a real fix: read Docling's item type directly instead of regex-matching for `#`.

**`InMemoryCache` is process-local**
Swap for `RedisCache` (same interface) before running more than one backend process.

**Embedded Qdrant is single-process**
Set `QDRANT_URL` in `.env` to a real Qdrant server once more than one backend process needs the store.

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).