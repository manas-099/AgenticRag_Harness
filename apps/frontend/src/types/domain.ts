/**
 * domain.ts
 * ---------------------------------------------------------------------------
 * Types mirroring the FastAPI backend's Pydantic response schemas
 * (see backend: src/rag_harness/api/v1/schemas/*). Keeping these in sync
 * with the backend is a manual contract for now — if you add/rename a field
 * in a Pydantic schema, update the matching type here.
 */

/** A single chunk of a source document, as returned in `sources_used`. */
export interface SourceChunk {
  chunk_id: string;
  doc_id?: string;
  page_num?: number;
}

/** Response body of `POST /v1/query` (standard, non-agentic RAG). */
export interface StandardQueryResult {
  answer: string;
  sources_used: string[];
  is_insufficient: boolean;
  from_cache: boolean;
}

/** Response body of `POST /v1/query/agentic`. */
export interface AgenticQueryResult {
  answer: string;
  sources_used: string[];
  degraded: boolean;
  degrade_reason: string | null;
  iterations_used: number;
  validation_retries: number;
  action_history: string[];
}

/** Response body of `GET /v1/health`. */
export interface BackendHealth {
  status: "ok" | string;
  qdrant_chunk_count: number;
  sparse_index_built: boolean;
  sparse_corpus_size: number;
  query_cache_size: number;
  retrieval_cache_size: number;
}

/** Per-role LLM chain status, one entry of `GET /v1/health/llm`. */
export interface LLMChainStatus {
  role: "agent" | "generate";
  order: string[];
  last_backend_used: string | null;
  myllm_reachable?: boolean;
}

/** Response body of `GET /v1/health/llm`. */
export interface LLMHealth {
  agent: LLMChainStatus;
  generate: LLMChainStatus;
}

/** Response body of `POST /v1/documents` (ingestion). */
export interface IngestDocumentResult {
  doc_id: string;
  chunks_created: number;
}

/** A document shown in the Knowledge panel. The backend doesn't currently
 *  expose a "list all ingested documents" endpoint — this is tracked
 *  client-side as documents are ingested through this UI. See
 *  api/documents.ts and store/knowledgeStore.ts for details. */
export interface KnowledgeDocument {
  docId: string;
  fileName: string;
  version: string;
  chunkCount: number;
  status: "ingesting" | "ready" | "error";
  errorMessage?: string;
}

/** One step in the agent's ReAct pipeline, used to drive the step tracker
 *  UI (StepTracker + StepGraph). Steps are inferred client-side from
 *  `action_history` in the agentic response — see utils/parseActionHistory.ts. */
export type AgentStepKind = "retrieve" | "evaluate" | "reretrieve" | "synthesize" | "answer";

export interface AgentStep {
  id: string;
  kind: AgentStepKind;
  label: string;
  status: "pending" | "active" | "done" | "error";
  detail?: string;
}
