// =============================================================================
// FILE: src/types/domain.ts
// TARGET PATH IN PROJECT: apps/frontend/src/types/domain.ts
// REPLACES: existing domain.ts (full replacement)
//
// WHAT CHANGED:
//   - ActionStep added: mirrors backend ActionStep (thought + action + iteration)
//   - AgenticQueryResult.action_steps added (list of ActionStep)
//   - AgenticQueryResult.token_budget_used + validation_retries added
//   - ChatMessage type added: represents one turn in the persistent chat history
//   - DebugState type added: shape of GET /v1/query/agentic/last_state response
//
// WHY:
//   New Chat UI needs ChatMessage for the conversation thread.
//   Debug Panel needs DebugState + ActionStep with thought text.
//
// WHERE IT CONNECTS:
//   store/chatStore.ts (ChatMessage), store/debugStore.ts (DebugState),
//   utils/parseActionHistory.ts (ActionStep), components/debug/DebugPanel.tsx
// =============================================================================

/** One structured step from the agent's ReAct loop. */
export interface ActionStep {
  action: string;    // e.g. "search_documents", "answer"
  thought: string;   // agent's reasoning text
  iteration: number; // 1-based iteration number
}

/** A single message in the chat conversation. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  // Only present on assistant messages from agentic mode
  agenticMeta?: {
    degraded: boolean;
    degrade_reason: string | null;
    iterations_used: number;
    validation_retries: number;
    token_budget_used: number;
    action_steps: ActionStep[];
    sources_used: string[];
  };
  // While streaming / in-flight
  isLoading?: boolean;
}

/** Response body of POST /v1/query (standard, non-agentic RAG). */
export interface StandardQueryResult {
  answer: string;
  sources_used: string[];
  is_insufficient: boolean;
  from_cache: boolean;
}

/** Response body of POST /v1/query/agentic. */
export interface AgenticQueryResult {
  answer: string;
  sources_used: string[];
  degraded: boolean;
  degrade_reason: string | null;
  iterations_used: number;
  validation_retries: number;
  token_budget_used: number;
  action_steps: ActionStep[];
  action_history: string[]; // flat list kept for parseActionHistory compat
}

/** Debug snapshot from GET /v1/query/agentic/last_state */
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

/** Response body of GET /v1/health. */
export interface BackendHealth {
  status: "ok" | string;
  qdrant_chunk_count: number;
  sparse_index_built: boolean;
  sparse_corpus_size: number;
  query_cache_size: number;
  retrieval_cache_size: number;
}

/** Per-role LLM chain status. */
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

/** Response body of POST /v1/documents/upload (ingestion). */
export interface IngestDocumentResult {
  doc_id: string;
  chunks_created: number;
}

/** A document shown in the Ingest tab. Tracked client-side. */
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
