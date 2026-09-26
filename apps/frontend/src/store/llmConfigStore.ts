/**
 * llmConfigStore.ts
 * ---------------------------------------------------------------------------
 * Backing store for the Config tab. Persisted to localStorage (NOT sent
 * anywhere on its own) so a user's custom key survives a page reload.
 * Read by api/client.ts to decide whether to attach BYOK headers.
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { DEFAULT_LLM_CONFIG, type CustomLLMConfig, type LLMMode } from "@/types/llmConfig";

interface LLMConfigStore {
  mode: LLMMode;
  custom: CustomLLMConfig;
  setMode: (mode: LLMMode) => void;
  updateCustom: (patch: Partial<CustomLLMConfig>) => void;
  reset: () => void;
}

export const useLLMConfigStore = create<LLMConfigStore>()(
  persist(
    (set) => ({
      mode: DEFAULT_LLM_CONFIG.mode,
      custom: DEFAULT_LLM_CONFIG.custom,
      setMode: (mode) => set({ mode }),
      updateCustom: (patch) => set((state) => ({ custom: { ...state.custom, ...patch } })),
      reset: () => set(DEFAULT_LLM_CONFIG),
    }),
    {
      name: "rag-harness-studio.llm-config",
      // API keys live in localStorage only — never logged, never sent to
      // any endpoint except your own backend's request headers.
    },
  ),
);
