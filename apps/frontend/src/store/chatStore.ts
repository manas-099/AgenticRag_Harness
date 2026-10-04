// =============================================================================
// FILE: src/store/chatStore.ts
// TARGET PATH: apps/frontend/src/store/chatStore.ts
// REPLACES: previous chatStore.ts
//
// WHAT CHANGED:
//   - stripCitations() applied to answer text before storing in message
//   - agenticMeta.from_cache tracked from response
//   - Standard query also tracks from_cache
// =============================================================================

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { runAgenticQuery, runStandardQuery } from "@/api/queryApi";
import { buildInFlightSteps, parseActionHistory } from "@/utils/parseActionHistory";
import { stripCitations } from "@/utils/stripCitations";
import type { AgentStep, ChatMessage } from "@/types/domain";
import { useDebugStore } from "@/store/debugStore";

export type ChatMode = "standard" | "agentic";

interface ChatStore {
  messages: ChatMessage[];
  mode: ChatMode;
  isRunning: boolean;
  steps: AgentStep[];
  stepPanelOpen: boolean;

  setMode: (mode: ChatMode) => void;
  setStepPanelOpen: (open: boolean) => void;
  send: (question: string) => Promise<void>;
  clearHistory: () => void;
}

let inFlightTimer: ReturnType<typeof setInterval> | null = null;

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      messages: [],
      mode: "agentic",
      isRunning: false,
      steps: [],
      stepPanelOpen: true,

      // mode is locked to agentic — Standard tab removed
      setMode: (_mode) => {}, // no-op
      setStepPanelOpen: (open) => set({ stepPanelOpen: open }),
      clearHistory: () => set({ messages: [], steps: [] }),

      send: async (question: string) => {
        if (!question.trim() || get().isRunning) return;

        const userMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: "user",
          content: question,
          timestamp: Date.now(),
        };
        const loadingId = crypto.randomUUID();
        const loadingMsg: ChatMessage = {
          id: loadingId,
          role: "assistant",
          content: "",
          timestamp: Date.now(),
          isLoading: true,
        };

        set((s) => ({
          messages: [...s.messages, userMsg, loadingMsg],
          isRunning: true,
          steps: buildInFlightSteps(0),
        }));

        let stage = 0;
        if (inFlightTimer) clearInterval(inFlightTimer);
        inFlightTimer = setInterval(() => {
          stage = Math.min(stage + 1, 3);
          set({ steps: buildInFlightSteps(stage) });
        }, 900);

        try {
          const result = await runAgenticQuery(question);
          const assistantMsg: ChatMessage = {
            id: loadingId,
            role: "assistant",
            content: stripCitations(result.answer),
            timestamp: Date.now(),
            agenticMeta: {
              degraded: result.degraded,
              degrade_reason: result.degrade_reason,
              iterations_used: result.iterations_used,
              validation_retries: result.validation_retries,
              token_budget_used: result.token_budget_used,
              action_steps: result.action_steps ?? [],
              sources_used: result.sources_used,
              from_cache: result.from_cache,
            },
          };
          // If from cache, don't show step panel (no new steps)
          const newSteps = result.from_cache
            ? []
            : parseActionHistory(result.action_history, result.degraded);

          set((s) => ({
            messages: s.messages.map((m) => (m.id === loadingId ? assistantMsg : m)),
            steps: newSteps,
          }));
          useDebugStore.getState().setLastResult({
            query: question,
            degraded: result.degraded,
            degrade_reason: result.degrade_reason,
            iterations_used: result.iterations_used,
            validation_retries: result.validation_retries,
            token_budget_used: result.token_budget_used,
            sources_count: result.sources_used.length,
            action_steps: result.action_steps ?? [],
          });
        } catch (err) {
          const errMsg: ChatMessage = {
            id: loadingId,
            role: "assistant",
            content: err instanceof Error ? err.message : "Query failed. Check backend connection.",
            timestamp: Date.now(),
          };
          set((s) => ({
            messages: s.messages.map((m) => (m.id === loadingId ? errMsg : m)),
          }));
        } finally {
          if (inFlightTimer) clearInterval(inFlightTimer);
          inFlightTimer = null;
          set({ isRunning: false });
        }
      },
    }),
    {
      name: "rag-chat-history",
      partialize: (state) => ({ messages: state.messages, mode: state.mode }),
    }
  )
);
