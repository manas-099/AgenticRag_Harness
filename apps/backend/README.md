**Path:** `README.md`

```markdown
# Agentic RAG Harness

A production-grade, agentic Retrieval-Augmented Generation system built
around a real ReAct loop. The agent doesn't run a fixed retrieve-then-generate
sequence — at every step it decides, via structured LLM output, whether to
search, reformulate a failed search, fetch a specific page, compare two
pieces of context, check a document's freshness, or answer. Every generated
claim is checked against its cited source before being returned to the user.

## Core design principle

**The LLM only makes judgment calls — what to search, when to stop, what a
claim means. Everything checkable is enforced in code, never trusted to
model self-report.**

This shows up everywhere: citation existence is checked in code before any
model judges groundedness; retrieval fusion uses rank position (a checkable
signal) instead of trusting raw scores across incomparable scales; the
relevance gate blocks generation entirely rather than asking the model to
self-assess confidence; loop control (iteration caps, token budgets,
stuck-loop detection) is enforced by the graph, not by asking the agent to
behave.

---

## System Design

### High-level flow

```
Document → Ingestion → Chunking → Embedding → Storage (Qdrant)
                                                      │
User Query → Agent Decision Loop ─────────────────────┤
                 │                                     │
                 ├── search_documents ───────────────────┤
                 ├── search_documents_reformulated ───────┤
                 ├── get_page                             │
                 ├── compare_chunks                       │
                 ├── check_document_freshness             │
                 └── answer ──→ Generate ──→ Verify ──→ pass: return
                                                │
                                       fail: retry (feedback) or degrade
```

### 1. Ingestion

Each PDF page is classified individually — not the whole file at once —
because a single PDF can mix clean text, scanned images, and tables across
different pages.

- **Clean text pages** → fast structural extraction (PyMuPDF), no OCR.
- **Scanned or mixed pages** (text + table + image together) → routed
  whole-page to Docling (OCR + table recognition). Deliberately not split
  into regions — simpler, and Docling handles mixed content adequately as
  a unit.
- **Docling runs once per document**, not once per flagged page. Running it
  per-page reloads OCR models repeatedly — a real, measured slowness bug
  that batching fixes.
- **Post-extraction quality gate**: even a page classified as clean text
  gets its output checked for garbled encoding or suspiciously short
  results, and rerouted to Docling if the check fails. The classifier is a
  structural heuristic, not a guarantee.
- Markdown files skip classification — they have no "might secretly be a
  scan" problem.

### 2. Chunking

- **Structure-aware splitting**: tries markdown headers first, then
  paragraph → sentence → word boundaries, only falling back to a hard
  character cut as a last resort. Prevents severing a clause mid-sentence
  across two chunks.
- **Contextual preambles**: a cheap LLM call prepends 1–2 sentences situating
  each chunk within its document (entities, topic, dates) before embedding —
  fixes the "orphaned pronoun" problem where a chunk like *"revenue increased
  12%"* loses its subject and timeframe once isolated.
- **Concurrent preamble generation** via a thread pool (I/O-bound LLM calls),
  with chunk order preserved through indexed futures rather than
  append-as-completed, since order matters for windowed-context lookups.
- **Section title extraction** from document structure, feeding the
  metadata-filtering layer.

### 3. Storage

- **Qdrant**, storing each chunk's vector alongside structured metadata
  (`doc_id`, `doc_version`, `page_num`, `section_title`, `ingestion_route`).
- **Primary/fallback embedding model pair** sharing the same vector
  dimension — a model failure switches to the fallback for the rest of the
  session rather than crashing ingestion, and the shared dimension means the
  switch never corrupts the index.
- Environment-aware path selection: the same code runs unmodified locally or
  in Colab; production points at a real Qdrant server instead of embedded
  local storage.

### 4. Retrieval

- **Dense and sparse search run in parallel** (`ThreadPoolExecutor`) —
  independent of each other, so total latency is the max of the two, not
  the sum.
- **Reciprocal Rank Fusion (RRF)** combines the two ranked lists using rank
  position, not raw scores — dense similarity and BM25 scores live on
  incomparable scales.
- **Deduplication** (embedding-similarity based) runs before reranking, so
  the most expensive step isn't spent scoring near-duplicate chunks.
- **Cross-encoder reranking** as the final precision pass on the now-small
  candidate set — retrieval trades precision for speed; a cross-encoder
  scores actual query–chunk pairs directly.

### 5. Metadata filtering (section-based)

- A dedicated Qdrant collection holds unique section titles with both dense
  and native sparse vectors — no hand-rolled sparse scoring.
- A query is matched against candidate sections via RRF-fused dense+sparse
  search, reranked, then an LLM makes the final pick via structured output —
  validated against the real known-sections list before being trusted.
- **Degrades to unfiltered retrieval** if nothing clearly matches, rather
  than forcing a possibly-wrong filter on a general query.

### 6. Generation

- Fixed prompt order: system rules → context (tagged with doc/page/chunk_id
  for citation) → question — so the model reads all evidence before
  committing to an answer.
- **Relevance gate**: if the best reranked chunk score falls below
  threshold, the LLM is never called — the system returns "insufficient
  information" directly. This is the primary hallucination guardrail, more
  load-bearing than any prompt wording.

### 7. Caching

Three tiers — query (exact match, fastest), embedding, and retrieval
(semantic-adjacent) — all keyed with document version, so re-ingesting a
document can selectively invalidate stale entries instead of wiping the
whole cache.

### 8. Verification

- **Structured claim extraction** (not regex) pulls individual claims and
  their citations out of a generated answer.
- **Citation validator** (code-only, cheapest, runs first): confirms every
  cited `chunk_id` was actually retrieved. A citation to a nonexistent chunk
  is a hallucination, not a judgment call — no model needed to catch it.
- **NLI groundedness check**: a cross-encoder NLI model verifies each claim
  is actually *entailed* by its cited source text — catching the more
  dangerous case where the citation is real but the claim overstates or
  misstates what the source says.
- On failure, the specific failure reason is fed back into a regeneration
  retry (capped), after which the system returns the best-grounded partial
  answer rather than a bare refusal — repair, not refuse.

### 9. Agent (ReAct loop, LangGraph)

- The agent **genuinely chooses** its next action via structured LLM output
  — `search_documents`, `search_documents_reformulated`, `get_page`,
  `compare_chunks`, `check_document_freshness`, or `answer` — rather than
  following a hardcoded sequence.
- **Loop control**: iteration cap, token budget cap, search-attempt cap, and
  a **stuck-loop detector** (if repeated searches add zero new chunks to the
  registry, the loop force-degrades instead of spinning).
- **Graceful degradation** throughout: any budget exhaustion or repeated
  verification failure returns the best-grounded partial answer with an
  explicit caveat — never a silent failure or a confident bluff.

---

## Architecture (code organization)

Hexagonal / clean architecture, so every layer is independently testable and
swappable:

```
domain          → framework-agnostic models + abstract ports (contracts)
application     → use cases; depends ONLY on domain, never on infrastructure directly
infrastructure  → concrete implementations of ports (Qdrant, Groq, MyLLM, Docling...)
api             → thin FastAPI layer; translates HTTP <-> use-case calls, never holds logic
```

Swapping Qdrant for another vector store touches only
`infrastructure/vector_stores/`. Swapping the LLM backend touches only
`infrastructure/llm/`. Nothing in `application/` needs to change either way,
because it depends on `domain/ports/`, not concrete classes.

## Project structure

```
apps/
├── backend/
│   ├── src/rag_harness/
│   │   ├── domain/            # models, ports, exceptions
│   │   ├── application/       # ingestion, retrieval, generation, verification, agent
│   │   ├── infrastructure/    # llm, embeddings, vector_stores, sparse_search, document_parsers, cache
│   │   ├── api/                # FastAPI routers, schemas, dependency wiring
│   │   └── config/             # settings, logging
│   ├── scripts/                 # ingest_cli.py, query_cli.py
│   ├── tests/                   # unit, integration, e2e
│   ├── requirements.txt
│   └── pyproject.toml
└── frontend/                    # scaffolded, not yet built
```

## Getting started

```bash
cd apps/backend
pip install -e .
cp .env.example .env   # fill in MYLLM_AUTH_TOKEN, GROQ_API_KEY

# Ingest a document
python scripts/ingest_cli.py --file path/to/doc.pdf --doc-id doc1 --version v1

# Ask a question (simple retrieve-then-generate)
python scripts/query_cli.py --question "What is this document about?"

# Ask a question via the full ReAct agent loop
python scripts/query_cli.py --question "What is this document about?" --agentic

# Or run the API
uvicorn rag_harness.api.main:app --reload
```

API endpoints once running: `POST /v1/documents` (ingest), `POST /v1/query`,
`POST /v1/query/agentic`, `GET /v1/health`.

## Running tests

```bash
cd apps/backend
pytest tests/unit                    # fast, no external dependencies
RUN_E2E_TESTS=1 pytest tests/e2e     # requires real Qdrant + LLM backend configured
```

## Known limitations / open work

- **Section-title extraction** currently only reliably finds headers on
  Docling-routed pages whose text happens to carry markdown `#` syntax;
  plain-text (AnyDoc-routed) pages produce no section title at all. Metadata
  filtering degrades safely to "unfiltered" in this case, but the extraction
  itself needs a real fix (read Docling's item type directly instead of
  regex-matching for `#`, plus a heuristic for plain-text pages).
- Frontend is scaffolded but not implemented.
- `InMemoryCache` is process-local — swap for `RedisCache` (implementing the
  same interface) before running more than one backend process.
- Embedded/local Qdrant supports one process at a time — use a real Qdrant
  server (`QDRANT_URL` in `.env`) once more than one process needs the store.

## Status

Backend is feature-complete across all layers described above and has run
successfully end-to-end in prototyping. Currently being hardened and
migrated into this modular structure module by module.
```