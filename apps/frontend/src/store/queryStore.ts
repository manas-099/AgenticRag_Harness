/**
 * queryStore.ts
 * ---------------------------------------------------------------------------
 * Central state for the query workspace: which mode is active (standard /
 * agentic / compare), the in-flight animated step pipeline, the latest
 * result, and a rolling history of past runs (shown as pills in the top
 * strip, matching the mockup's "now · rollback steps for prod" / "2m ago ·
 * ..." tabs).
 *
 * The fake-but-honest "in-flight" animation: real step-by-step server-sent
 * events aren't implemented on the backend (it returns one JSON blob at the
 * end), so we simulate progressive step activation client-side while the
 * request is in flight, then reconcile with the REAL action_history once
 * the response lands (see utils/parseActionHistory.ts). This keeps the UI
 * honest — the final rendered steps always reflect what actually happened,
 * the animation is only a "the agent is working" indicator, not fabricated
 * data.
 */
import { create } from "zustand";
import { runAgenticQuery, runStandardQuery } from "@/api/queryApi";
import { buildInFlightSteps, parseActionHistory } from "@/utils/parseActionHistory";
import type { AgentStep } from "@/types/domain";

export type QueryMode = "standard" | "agentic" | "compare";

export interface HistoryEntry {
  id: string;
  question: string;
  mode: QueryMode;
  timestamp: number;
}

interface QueryStore {
  mode: QueryMode;
  question: string;
  isRunning: boolean;
  error: string | null;
  steps: AgentStep[];
  answer: string | null;
  sourcesUsed: string[];
  degraded: boolean;
  degradeReason: string | null;
  fromCache: boolean;
  history: HistoryEntry[];

  setMode: (mode: QueryMode) => void;
  setQuestion: (q: string) => void;
  run: () => Promise<void>;
}

let inFlightTimer: ReturnType<typeof setInterval> | null = null;

export const useQueryStore = create<QueryStore>()((set, get) => ({
  mode: "agentic",
  question: "",
  isRunning: false,
  error: null,
  steps: [],
  answer: null,
  sourcesUsed: [],
  degraded: false,
  degradeReason: null,
  fromCache: false,
  history: [],

  setMode: (mode) => set({ mode }),
  setQuestion: (question) => set({ question }),

  run: async () => {
    const { question, mode } = get();
    if (!question.trim() || get().isRunning) return;

    set({
      isRunning: true,
      error: null,
      answer: null,
      sourcesUsed: [],
      degraded: false,
      degradeReason: null,
      steps: buildInFlightSteps(0),
    });

    // Advance the "the agent is working" animation one stage every ~900ms
    // while we wait for the real response — purely cosmetic, see doc comment above.
    let stage = 0;
    if (inFlightTimer) clearInterval(inFlightTimer);
    inFlightTimer = setInterval(() => {
      stage = Math.min(stage + 1, 3);
      set({ steps: buildInFlightSteps(stage) });
    }, 900);

    try {
      if (mode === "standard") {
        const result = await runStandardQuery(question);
        set({
          answer: result.answer,
          sourcesUsed: result.sources_used,
          fromCache: result.from_cache,
          steps: [{ id: "retrieve-std", kind: "retrieve", label: "retrieve", status: "done" }],
        });
      } else {
        // "compare" runs both and shows agentic's step trace alongside
        // standard's answer — see components/query/ComparePanel.tsx.
        const result = await runAgenticQuery(question);
        set({
          answer: result.answer,
          sourcesUsed: result.sources_used,
          degraded: result.degraded,
          degradeReason: result.degrade_reason,
          steps: parseActionHistory(result.action_history, result.degraded),
        });
      }

      set((s) => ({
        history: [
          { id: crypto.randomUUID(), question, mode, timestamp: Date.now() },
          ...s.history,
        ].slice(0, 20),
      }));
    } catch (err) {
      set({ error: err instanceof Error ? err.message : "Query failed" });
    } finally {
      if (inFlightTimer) clearInterval(inFlightTimer);
      inFlightTimer = null;
      set({ isRunning: false });
    }
  },
}));
