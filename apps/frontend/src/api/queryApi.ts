// =============================================================================
// FILE: src/api/queryApi.ts
// TARGET PATH IN PROJECT: apps/frontend/src/api/queryApi.ts
// REPLACES: existing queryApi.ts (full replacement)
//
// WHAT CHANGED:
//   - runAgenticQuery now returns AgenticQueryResult which includes action_steps
//   - Added fetchLastAgentState() for the Debug Panel
//
// WHY:
//   The new AgenticQueryResult type has action_steps (with thoughts).
//   DebugPanel needs fetchLastAgentState() to show the last run snapshot
//   without re-running.
//
// WHERE IT CONNECTS:
//   store/chatStore.ts calls runStandardQuery, runAgenticQuery
//   components/debug/DebugPanel.tsx can call fetchLastAgentState
// =============================================================================

import { apiRequest } from "@/api/client";
import type { AgenticQueryResult, DebugState, StandardQueryResult } from "@/types/domain";

export function runStandardQuery(question: string, signal?: AbortSignal): Promise<StandardQueryResult> {
  return apiRequest<StandardQueryResult>("/query", { method: "POST", body: { question }, signal });
}

export function runAgenticQuery(question: string, signal?: AbortSignal): Promise<AgenticQueryResult> {
  return apiRequest<AgenticQueryResult>("/query/agentic", { method: "POST", body: { question }, signal });
}

export function fetchLastAgentState(signal?: AbortSignal): Promise<DebugState> {
  return apiRequest<DebugState>("/query/agentic/last_state", { signal });
}
