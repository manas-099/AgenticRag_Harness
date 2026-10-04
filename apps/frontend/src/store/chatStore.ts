// =============================================================================
// FILE: src/store/chatStore.ts
// TARGET PATH IN PROJECT: apps/frontend/src/store/chatStore.ts
// NEW FILE (does not exist yet)
//
// WHAT IT DOES:
//   Central Zustand store for the Chat tab. Maintains:
//     - messages[]: the full conversation thread (user + assistant bubbles)
//     - mode: "standard" | "agentic" — which pipeline to use
//     - isRunning: true while waiting for a response
//     - steps: AgentStep[] for the step visualization panel (agentic only)
//     - activeStepIndex: which step is currently "active" (animated)
//
//   send(question) does:
//     1. Appends user message bubble immediately
//     2. Appends a loading assistant bubble (shows thinking animation)
//     3. Calls the backend (standard or agentic)
//     4. Replaces the loading bubble with the real answer + metadata
//     5. Updates the step visualization from action_history
//
// WHY:
//   The old queryStore had no concept of a conversation thread — each run
//   was stateless. ChatStore is the persistent multi-turn replacement.
//   It uses localStorage to persist the thread across page refreshes.
//
// WHERE IT CONNECTS:
//   components/chat/ChatWindow.tsx (renders messages[])
//   components/chat/ChatInput.tsx (calls send())
//   components/steps/StepPanel.tsx (reads steps)
//   store/debugStore.ts (reads action_steps from last agentic result)
// =============================================================================

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { runAgenticQuery, runStandardQuery } from "@/api/queryApi";
import { buildInFlightSteps, parseActionHistory } from "@/utils/parseActionHistory";
import type { AgentStep, ChatMessage } from "@/types/domain";
import { useDebugStore } from "@/store/debugStore";

export type ChatMode = "standard" | "agentic";

interface ChatStore {
  messages: ChatMessage[];
  mode: ChatMode;
  isRunning: boolean;
  steps: AgentStep[];

  setMode: (mode: ChatMode) => void;
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

      setMode: (mode) => set({ mode }),

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

        // Animate steps while waiting
        let stage = 0;
        if (inFlightTimer) clearInterval(inFlightTimer);
        if (get().mode === "agentic") {
          inFlightTimer = setInterval(() => {
            stage = Math.min(stage + 1, 3);
            set({ steps: buildInFlightSteps(stage) });
          }, 900);
        }

        try {
          if (get().mode === "standard") {
            const result = await runStandardQuery(question);
            const assistantMsg: ChatMessage = {
              id: loadingId,
              role: "assistant",
              content: result.answer,
              timestamp: Date.now(),
              agenticMeta: {
                degraded: result.is_insufficient,
                degrade_reason: result.is_insufficient ? "insufficient_context" : null,
                iterations_used: 1,
                validation_retries: 0,
                token_budget_used: 0,
                action_steps: [],
                sources_used: result.sources_used,
              },
            };
            set((s) => ({
              messages: s.messages.map((m) => (m.id === loadingId ? assistantMsg : m)),
              steps: [],
            }));
          } else {
            const result = await runAgenticQuery(question);
            const assistantMsg: ChatMessage = {
              id: loadingId,
              role: "assistant",
              content: result.answer,
              timestamp: Date.now(),
              agenticMeta: {
                degraded: result.degraded,
                degrade_reason: result.degrade_reason,
                iterations_used: result.iterations_used,
                validation_retries: result.validation_retries,
                token_budget_used: result.token_budget_used,
                action_steps: result.action_steps ?? [],
                sources_used: result.sources_used,
              },
            };
            set((s) => ({
              messages: s.messages.map((m) => (m.id === loadingId ? assistantMsg : m)),
              steps: parseActionHistory(result.action_history, result.degraded),
            }));
            // Update debug store
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
          }
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
      // Only persist messages and mode, not transient state
      partialize: (state) => ({ messages: state.messages, mode: state.mode }),
    }
  )
);
