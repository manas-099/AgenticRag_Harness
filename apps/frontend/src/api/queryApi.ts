/**
 * queryApi.ts — typed wrappers around POST /v1/query and /v1/query/agentic.
 */
import { apiRequest } from "@/api/client";
import type { AgenticQueryResult, StandardQueryResult } from "@/types/domain";

export function runStandardQuery(question: string, signal?: AbortSignal): Promise<StandardQueryResult> {
  return apiRequest<StandardQueryResult>("/query", { method: "POST", body: { question }, signal });
}

export function runAgenticQuery(question: string, signal?: AbortSignal): Promise<AgenticQueryResult> {
  return apiRequest<AgenticQueryResult>("/query/agentic", { method: "POST", body: { question }, signal });
}
