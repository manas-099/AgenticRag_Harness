# Agentic RAG Harness

agentic-rag-harness/
├── apps/
│   ├── backend/
│   │   ├── src/
│   │   │   └── rag_harness/
│   │   │       ├── __init__.py
│   │   │       │
│   │   │       ├── domain/                    # PURE business logic — no framework, no I/O libs
│   │   │       │   ├── __init__.py
│   │   │       │   ├── models/                # Chunk, ChunkMetadata, Claim, AnswerResult, etc.
│   │   │       │   │   ├── chunk.py
│   │   │       │   │   ├── retrieval.py
│   │   │       │   │   ├── verification.py
│   │   │       │   │   └── agent_state.py
│   │   │       │   ├── exceptions.py           # HarnessError hierarchy
│   │   │       │   └── ports/                  # ABSTRACT interfaces — the "contracts" infra must fulfill
│   │   │       │       ├── llm_port.py         # BaseLLMClient (ABC)
│   │   │       │       ├── vector_store_port.py
│   │   │       │       ├── embedder_port.py
│   │   │       │       └── document_parser_port.py
│   │   │       │
│   │   │       ├── application/                # USE CASES — orchestration logic, depends only on domain + ports
│   │   │       │   ├── __init__.py
│   │   │       │   ├── ingestion/
│   │   │       │   │   ├── ingest_document_use_case.py
│   │   │       │   │   ├── pdf_page_strategy.py
│   │   │       │   │   ├── chunking_service.py
│   │   │       │   │   └── quality_gate.py
│   │   │       │   ├── retrieval/
│   │   │       │   │   ├── retrieve_use_case.py
│   │   │       │   │   ├── fusion_service.py    # RRF, dedup, rerank
│   │   │       │   │   └── metadata_filter_service.py
│   │   │       │   ├── generation/
│   │   │       │   │   ├── generate_answer_use_case.py
│   │   │       │   │   └── prompts.py
│   │   │       │   ├── verification/
│   │   │       │   │   ├── verify_answer_use_case.py
│   │   │       │   │   ├── citation_validator.py
│   │   │       │   │   └── groundedness_checker.py
│   │   │       │   └── agent/
│   │   │       │       ├── agent_decision_service.py
│   │   │       │       ├── tools.py
│   │   │       │       ├── graph_builder.py     # LangGraph wiring
│   │   │       │       └── run_agentic_query_use_case.py
│   │   │       │
│   │   │       ├── infrastructure/              # CONCRETE implementations of the ports — swappable
│   │   │       │   ├── __init__.py
│   │   │       │   ├── llm/
│   │   │       │   │   ├── myllm_client.py
│   │   │       │   │   ├── groq_client.py
│   │   │       │   │   └── llm_factory.py
│   │   │       │   ├── embeddings/
│   │   │       │   │   └── sentence_transformer_embedder.py
│   │   │       │   ├── vector_stores/
│   │   │       │   │   ├── qdrant_chunk_store.py
│   │   │       │   │   └── qdrant_section_store.py
│   │   │       │   ├── sparse_search/
│   │   │       │   │   └── bm25_retriever.py
│   │   │       │   ├── document_parsers/
│   │   │       │   │   ├── anydoc_parser.py
│   │   │       │   │   └── docling_parser.py
│   │   │       │   └── cache/
│   │   │       │       ├── in_memory_cache.py
│   │   │       │       └── redis_cache.py       # production swap-in, same interface
│   │   │       │
│   │   │       ├── api/                         # FastAPI — HTTP layer only, thin, no business logic
│   │   │       │   ├── __init__.py
│   │   │       │   ├── main.py                  # app factory
│   │   │       │   ├── dependencies.py          # DI wiring — builds use cases from settings
│   │   │       │   ├── v1/
│   │   │       │   │   ├── routers/
│   │   │       │   │   │   ├── documents.py     # POST /v1/documents (ingest)
│   │   │       │   │   │   ├── query.py         # POST /v1/query, /v1/query/agentic
│   │   │       │   │   │   └── health.py
│   │   │       │   │   └── schemas/             # request/response DTOs — NEVER expose domain models directly
│   │   │       │   │       ├── document_schemas.py
│   │   │       │   │       └── query_schemas.py
│   │   │       │   └── middleware/
│   │   │       │       ├── error_handler.py
│   │   │       │       └── request_logging.py
│   │   │       │
│   │   │       ├── config/
│   │   │       │   ├── settings.py              # Pydantic BaseSettings, env-layered
│   │   │       │   └── logging_config.py
│   │   │       │
│   │   │       └── shared/
│   │   │           └── utils.py
│   │   │
│   │   ├── tests/
│   │   │   ├── unit/                            # mirrors src/ structure, mocks all ports
│   │   │   │   ├── domain/
│   │   │   │   ├── application/
│   │   │   │   └── infrastructure/
│   │   │   ├── integration/                     # real Qdrant (testcontainers), real API calls
│   │   │   └── e2e/                             # full ingest -> query -> answer flow
│   │   │
│   │   ├── scripts/
│   │   │   ├── ingest_cli.py
│   │   │   └── query_cli.py
│   │   │
│   │   ├── alembic/ or migrations/              # if you add a relational DB for users/sessions later
│   │   ├── Dockerfile
│   │   ├── pyproject.toml                       # deps, ruff/black/mypy config
│   │   ├── .env.example
│   │   └── README.md
│   │
│   └── frontend/
│       ├── src/
│       │   ├── app/                             # Next.js app router (or pages/ for CRA/Vite)
│       │   │   ├── layout.tsx
│       │   │   ├── page.tsx                     # chat/query UI
│       │   │   └── documents/
│       │   │       └── page.tsx                 # upload/manage documents UI
│       │   ├── components/
│       │   │   ├── chat/
│       │   │   ├── upload/
│       │   │   └── ui/                          # design system primitives
│       │   ├── lib/
│       │   │   ├── api-client.ts                # typed client for backend API
│       │   │   └── types.ts                     # mirrors backend schemas
│       │   ├── hooks/
│       │   └── store/                           # state management (Zustand/Redux)
│       ├── public/
│       ├── package.json
│       ├── tsconfig.json
│       └── Dockerfile
│
├── packages/                                    # shared code between frontend/backend, if any
│   └── shared-types/                             # e.g. OpenAPI-generated TS types from backend schemas
│
├── infra/
│   ├── docker-compose.yml                       # local dev: backend + frontend + qdrant + redis
│   ├── docker-compose.prod.yml
│   ├── k8s/                                      # if/when you deploy to a cluster
│   │   ├── backend-deployment.yaml
│   │   ├── frontend-deployment.yaml
│   │   └── qdrant-statefulset.yaml
│   └── terraform/                                # if managing cloud infra as code
│
├── .github/
│   └── workflows/
│       ├── backend-ci.yml                        # lint, type-check, test on every PR
│       └── frontend-ci.yml
│
├── docs/
│   ├── architecture.md
│   ├── api-reference.md
│   └── adr/                                      # Architecture Decision Records — e.g. "why two-stage RRF over single-stage"
│
├── .gitignore
├── .editorconfig
└── README.md