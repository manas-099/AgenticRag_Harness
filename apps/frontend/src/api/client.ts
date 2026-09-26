/**
 * client.ts
 * ---------------------------------------------------------------------------
 * Single fetch wrapper used by every api/* module. Responsibilities:
 *   1. Resolve the backend base URL from VITE_API_BASE_URL (falls back to
 *      the Vite dev proxy "/api" -> see vite.config.ts).
 *   2. Attach BYOK headers when the user has switched the Config tab to
 *      "custom" mode (see store/llmConfigStore.ts) — X-LLM-Provider,
 *      X-LLM-Model, X-LLM-Api-Key, X-LLM-Base-Url. The backend does not act
 *      on these yet (see types/llmConfig.ts doc comment + README) but the
 *      contract is ready on the frontend side.
 *   3. Normalize errors into ApiError so components can render one
 *      consistent error shape instead of catching fetch/JSON quirks.
 */

import { useLLMConfigStore } from "@/store/llmConfigStore";

const BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() || "/api";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function buildLLMHeaders(): Record<string, string> {
  const { mode, custom } = useLLMConfigStore.getState();
  if (mode !== "custom" || !custom.apiKey) return {};

  const headers: Record<string, string> = {
    "X-LLM-Provider": custom.provider,
    "X-LLM-Model": custom.model,
    "X-LLM-Api-Key": custom.apiKey,
  };
  if (custom.provider === "custom" && custom.baseUrl) {
    headers["X-LLM-Base-Url"] = custom.baseUrl;
  }
  return headers;
}

export async function apiRequest<TResponse>(
  path: string,
  options: { method?: "GET" | "POST" | "DELETE"; body?: unknown; signal?: AbortSignal } = {},
): Promise<TResponse> {
  const { method = "GET", body, signal } = options;

  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    signal,
    headers: {
      "Content-Type": "application/json",
      ...buildLLMHeaders(),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  }).catch((err: unknown) => {
    throw new ApiError(
      err instanceof Error ? `Network error reaching backend: ${err.message}` : "Network error reaching backend",
      0,
    );
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiError(text || `Request failed with status ${res.status}`, res.status);
  }

  // 204 No Content, etc.
  if (res.status === 204) return undefined as TResponse;
  return (await res.json()) as TResponse;
}
