/**
 * healthApi.ts — typed wrappers around GET /v1/health and /v1/health/llm.
 */
import { apiRequest } from "@/api/client";
import type { BackendHealth, LLMHealth } from "@/types/domain";

export function fetchBackendHealth(signal?: AbortSignal): Promise<BackendHealth> {
  return apiRequest<BackendHealth>("/health", { signal });
}

export function fetchLLMHealth(signal?: AbortSignal): Promise<LLMHealth> {
  return apiRequest<LLMHealth>("/health/llm", { signal });
}
