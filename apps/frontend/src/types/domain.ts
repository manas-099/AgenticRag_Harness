// =============================================================================
// FILE: src/types/domain.ts
// TARGET PATH: apps/frontend/src/types/domain.ts
// REPLACES: previous domain.ts
//
// WHAT CHANGED:
//   - AgenticQueryResult.from_cache: boolean added
//   - ChatMessage.agenticMeta.from_cache added
// =============================================================================

export interface ActionStep {
  action: string;
  thought: string;
  iteration: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  agenticMeta?: {
    degraded: boolean;
    degrade_reason: string | null;
    iterations_used: number;
    validation_retries: number;
    token_budget_used: number;
    action_steps: ActionStep[];
    sources_used: string[];
    from_cache: boolean;   // ← NEW
  };
  isLoading?: boolean;
}

export interface StandardQueryResult {
  answer: string;
  sources_used: string[];
  is_insufficient: boolean;
  from_cache: boolean;
}

export interface AgenticQueryResult {
  answer: string;
  sources_used: string[];
  degraded: boolean;
  degrade_reason: string | null;
  iterations_used: number;
  validation_retries: number;
  token_budget_used: number;
  action_steps: ActionStep[];
  action_history: string[];
  from_cache: boolean;   // ← NEW
}

export interface DebugState {
  query: string;
  degraded: boolean;
  degrade_reason: string | null;
  iterations_used: number;
  validation_retries: number;
  token_budget_used: number;
  sources_count: number;
  action_steps: ActionStep[];
}

export interface BackendHealth {
  status: "ok" | string;
  qdrant_chunk_count: number;
  sparse_index_built: boolean;
  sparse_corpus_size: number;
  query_cache_size: number;
  retrieval_cache_size: number;
}

export interface LLMChainStatus {
  role: "agent" | "generate";
  order: string[];
  last_backend_used: string | null;
  myllm_reachable?: boolean;
}

export interface LLMHealth {
  agent: LLMChainStatus;
  generate: LLMChainStatus;
}

export interface IngestDocumentResult {
  doc_id: string;
  chunks_created: number;
}

export interface KnowledgeDocument {
  docId: string;
  fileName: string;
  version: string;
  chunkCount: number;
  status: "ingesting" | "ready" | "error";
  errorMessage?: string;
}

export type AgentStepKind = "retrieve" | "evaluate" | "reretrieve" | "synthesize" | "answer";

export interface AgentStep {
  id: string;
  kind: AgentStepKind;
  label: string;
  status: "pending" | "active" | "done" | "error";
  detail?: string;
}
