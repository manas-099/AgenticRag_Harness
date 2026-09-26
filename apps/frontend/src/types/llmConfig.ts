/**
 * llmConfig.ts
 * ---------------------------------------------------------------------------
 * The "Config" tab lets a user either:
 *   (a) DEFAULT  — use whatever the backend is already configured with via
 *       its own .env (the MyLLM -> OpenRouter -> Groq chain). No credentials
 *       leave the browser; the frontend sends nothing extra.
 *   (b) CUSTOM   — bring their own provider + model + API key, entered here
 *       and stored ONLY in localStorage (never sent anywhere except as
 *       request headers to your own backend, over HTTPS in production).
 *
 * IMPORTANT — backend contract: for CUSTOM mode to actually change which
 * model answers, the FastAPI backend must read the `X-LLM-*` headers this
 * client sends (see api/client.ts) and use them to build a one-off LLMChain
 * for that request instead of its default chain. That hook does not exist
 * in the backend yet — see README.md "Wiring up bring-your-own-key on the
 * backend" for the ~15-line addition needed. Until then, CUSTOM mode still
 * sends the headers (so you can wire it up whenever you're ready) but the
 * backend will simply ignore them and use its default chain.
 */

export type LLMProvider = "openrouter" | "groq" | "openai" | "anthropic" | "custom";

export const LLM_PROVIDERS: { id: LLMProvider; label: string; defaultModel: string }[] = [
  { id: "openrouter", label: "OpenRouter", defaultModel: "liquid/lfm-2.5-2.6b:free" },
  { id: "groq", label: "Groq", defaultModel: "openai/gpt-oss-120b" },
  { id: "openai", label: "OpenAI", defaultModel: "gpt-4o-mini" },
  { id: "anthropic", label: "Anthropic", defaultModel: "claude-3-5-haiku-latest" },
  { id: "custom", label: "Custom endpoint", defaultModel: "" },
];

export interface CustomLLMConfig {
  provider: LLMProvider;
  model: string;
  apiKey: string;
  /** Only used when provider === "custom" — a full chat-completions URL. */
  baseUrl?: string;
}

export type LLMMode = "default" | "custom";

export interface LLMConfigState {
  mode: LLMMode;
  custom: CustomLLMConfig;
}

export const DEFAULT_LLM_CONFIG: LLMConfigState = {
  mode: "default",
  custom: {
    provider: "openrouter",
    model: LLM_PROVIDERS[0].defaultModel,
    apiKey: "",
    baseUrl: "",
  },
};
